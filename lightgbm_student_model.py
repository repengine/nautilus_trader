"""
LightGBMStudentDistiller
------------------------
A base student that learns to mimic a heavy teacher (e.g., TFT), then exports a single
ONNX with sigmoid + optional Platt calibration baked in.

Supports 3 loss modes for distillation:
1) "logit_mse":    MSE on teacher logits (z_T = logit(q_T))
2) "soft_ce":      Cross-entropy on teacher probs q_T
3) "hybrid":       Weighted mix of (1) and (2)

The exported sidecar meta enforces train↔serve parity:
- ordered feature_names
- feature_schema_hash
- dtypes (float32 by default)
- output_schema: {kind: "binary_proba", shape: [None, 1]}
- best_iteration, opset, trainer_version, train_date_range

NOTE: This file focuses on the student. The teacher is handled separately.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# LightGBM can be absent in some environments; delay import errors to fit().
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional at import time
    lgb = None

# Optional calibration
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover - optional
    LogisticRegression = None
    IsotonicRegression = None

# ONNX export
try:
    import onnx
    from onnx import helper, numpy_helper
    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm as convert_lgbm_booster
    from onnxmltools.convert.common._topology import Topology, Variable
    from onnxmltools.convert.common.data_types import FloatTensorType
except Exception:  # pragma: no cover - optional
    onnx = None
    helper = None
    numpy_helper = None
    convert_lgbm_booster = None
    Topology = None
    Variable = None
    FloatTensorType = None


def schema_hash(feature_names: List[str], dtypes: Optional[List[str]] = None) -> str:
    payload = "|".join(feature_names) + "||" + "|".join(dtypes or [])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class StudentMeta:
    model_id: str
    feature_names: List[str]
    feature_dtypes: List[str]
    feature_schema_hash: str
    output_schema: Dict
    best_iteration: Optional[int]
    opset: int
    trainer_version: str
    train_date_range: Optional[Tuple[str, str]] = None
    calibrator_kind: Optional[str] = None
    calibrator_params: Optional[Dict] = None
    flags: Optional[Dict] = None


class LightGBMStudentDistiller:
    """Student that distills a teacher's outputs to a LightGBM model and exports ONNX.

    Parameters
    ----------
    objective : str
        One of {"logit_mse", "soft_ce", "hybrid"}.
    kd_lambda : float
        Weight for CE part in hybrid (0..1). Only used if objective=="hybrid".
    lgb_params : dict
        Standard LightGBM params. We'll override objective via `fobj` if needed.
    early_stopping : int
        Early stopping rounds.
    opset : int
        ONNX opset for export.
    """

    def __init__(self,
                 objective: str = "logit_mse",
                 kd_lambda: float = 0.5,
                 lgb_params: Optional[Dict] = None,
                 early_stopping: int = 200,
                 opset: int = 17,
                 trainer_version: str = "0.1.0"):
        self.objective = objective
        self.kd_lambda = kd_lambda
        self.lgb_params = lgb_params or {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 32,
            "verbose": -1,
        }
        self.early_stopping = early_stopping
        self.opset = opset
        self.trainer_version = trainer_version

        self.model = None            # LightGBM Booster
        self.best_iteration = None
        self._calibrator = None      # (kind, params) OR fitted sklearn object
        self._calibrator_kind = None

    # ---------- Distillation targets ----------

    @staticmethod
    def _clip01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.clip(x, eps, 1.0 - eps)

    @staticmethod
    def teacher_logits(q: np.ndarray) -> np.ndarray:
        q = LightGBMStudentDistiller._clip01(q)
        return np.log(q / (1.0 - q))

    # ---------- Custom objectives (return grad,hess) ----------

    def _obj_hybrid(self, pred_raw: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        # pred_raw = s (student logit); labels = q_T (teacher prob)
        q = dataset.get_label()
        p = 1.0 / (1.0 + np.exp(-pred_raw))
        g_ce = (p - q)
        h_ce = p * (1.0 - p)

        z = self.teacher_logits(q)
        g_mse = (pred_raw - z)
        h_mse = np.ones_like(pred_raw)

        lam = float(self.kd_lambda)
        grad = lam * g_ce + (1.0 - lam) * g_mse
        hess = lam * h_ce + (1.0 - lam) * h_mse
        return grad.astype(np.float32), hess.astype(np.float32)

    # ---------- Fit ----------

    def fit(self,
            X_train: np.ndarray,
            q_train: np.ndarray,
            X_val: np.ndarray,
            y_val_true: Optional[np.ndarray] = None) -> "LightGBMStudentDistiller":
        if lgb is None:
            raise ImportError("lightgbm is required to train the student.")
        X_train = np.asarray(X_train, dtype=np.float32, order="C")
        X_val   = np.asarray(X_val,   dtype=np.float32, order="C")
        q_train = np.asarray(q_train, dtype=np.float32)

        train_set = lgb.Dataset(X_train, label=q_train)
        val_set   = lgb.Dataset(X_val,   label=(y_val_true if y_val_true is not None else q_train[:len(X_val)]))

        params = dict(self.lgb_params)  # copy
        # We'll always train a raw-score model (logit space); metrics can be logloss on truth.
        fobj = None
        if self.objective == "logit_mse":
            params.update({"objective": "regression", "metric": ["l2"]})
            fobj = None
        elif self.objective == "soft_ce":
            # treat teacher prob as soft label under binary logloss
            params.update({"objective": "binary", "metric": ["binary_logloss"]})
            fobj = None  # built-in CE; labels in (0,1) are accepted by recent LGBM
        elif self.objective == "hybrid":
            params.update({"objective": "regression", "metric": ["l2"]})
            fobj = self._obj_hybrid
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        self.model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            fobj=fobj,
            early_stopping_rounds=self.early_stopping,
            verbose_eval=50,
        )
        self.best_iteration = getattr(self.model, "best_iteration", None)

        # Optional post-calibration on true labels if provided
        if y_val_true is not None:
            z_val = self._predict_raw(X_val)
            p_val = 1.0 / (1.0 + np.exp(-z_val))
            if LogisticRegression is not None:
                lr = LogisticRegression(solver="lbfgs")
                lr.fit(p_val.reshape(-1,1), y_val_true.astype(int))
                self._calibrator = lr
                self._calibrator_kind = "platt"
            elif IsotonicRegression is not None:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_val, y_val_true.astype(float))
                self._calibrator = iso
                self._calibrator_kind = "isotonic"
            else:
                self._calibrator = None
                self._calibrator_kind = None
        return self

    # ---------- Predict ----------

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        X = np.asarray(X, dtype=np.float32, order="C")
        return self.model.predict(X, num_iteration=self.best_iteration)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self._predict_raw(X)
        p = 1.0 / (1.0 + np.exp(-z))
        if self._calibrator is not None:
            if self._calibrator_kind == "platt":
                p = self._calibrator.predict_proba(p.reshape(-1,1))[:,1]
            elif self._calibrator_kind == "isotonic":
                p = self._calibrator.transform(p)
        return p.astype(np.float32).reshape(-1,1)

    # ---------- Export ----------

    def export_onnx(self,
                    feature_names: List[str],
                    out_dir: str,
                    model_id: str,
                    train_date_range: Optional[Tuple[str, str]] = None,
                    flags: Optional[Dict] = None) -> Tuple[str, str]:
        """Export ONNX (trees) + bake sigmoid (+ Platt) and save sidecar metadata.

        Returns
        -------
        (onnx_path, meta_path)
        """
        if onnx is None or convert_lgbm_booster is None or FloatTensorType is None:
            raise ImportError("onnx/onnxmltools are required for export. pip install onnx onnxmltools")
        if self.model is None:
            raise RuntimeError("Train the model before export.")

        n_features = len(feature_names)

        # Convert the LightGBM Booster to an ONNX subgraph that outputs raw score z
        # Build a fake topology with an input tensor of shape [None, n_features].
        # Note: onnxmltools will produce a graph with an output named 'output' (float).
        # We'll post-append Sigmoid and optional Platt (affine + Sigmoid) nodes.
        initial_type = [("input", FloatTensorType([None, n_features]))]
        # This API is low-level; use convert function to get the model container directly.
        onnx_model = convert_lgbm_booster(self.model, initial_types=initial_type)

        # Add post-processing: Sigmoid of raw score → probability.
        # If Platt is present: p = sigmoid(a*z + b). Implement as Gemm + Sigmoid.
        # Find the single output name (assume first)
        raw_output_name = onnx_model.graph.output[0].name

        # Create intermediate node: Gemm (scale + bias) if calibrator exists
        last_output = raw_output_name
        nodes = []
        initializers = []

        if self._calibrator_kind == "platt" and self._calibrator is not None:
            # sklearn LR: predict_proba(sigmoid(w*z + b)). We want p = sigmoid(w*z + b).
            coef = float(self._calibrator.coef_.ravel()[0])
            bias = float(self._calibrator.intercept_.ravel()[0])
            w_name = "platt_W"
            b_name = "platt_b"
            W = numpy_helper.from_array(np.array([[coef]], dtype=np.float32), name=w_name)
            B = numpy_helper.from_array(np.array([bias], dtype=np.float32), name=b_name)
            initializers.extend([W, B])
            gemm_out = "gemm_out"
            nodes.append(helper.make_node("Gemm",
                                          inputs=[last_output, w_name, b_name],
                                          outputs=[gemm_out],
                                          alpha=1.0, beta=1.0, transB=1))
            last_output = gemm_out

        # Sigmoid to get probability
        prob_name = "probability"
        nodes.append(helper.make_node("Sigmoid", inputs=[last_output], outputs=[prob_name]))

        # Append our nodes
        onnx_model.graph.node.extend(nodes)
        onnx_model.graph.output[0].name = prob_name  # redefine main output as probability

        # Save ONNX
        out_dir = str(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        onnx_path = os.path.join(out_dir, "student.onnx")
        onnx.save(onnx_model, onnx_path)

        # Save metadata
        dtypes = ["float32"] * n_features
        meta = StudentMeta(
            model_id=model_id,
            feature_names=list(feature_names),
            feature_dtypes=dtypes,
            feature_schema_hash=schema_hash(feature_names, dtypes),
            output_schema={"kind": "binary_proba", "shape": [None, 1]},
            best_iteration=self.best_iteration,
            opset=self.opset,
            trainer_version=self.trainer_version,
            train_date_range=train_date_range,
            calibrator_kind=self._calibrator_kind,
            calibrator_params=None if self._calibrator_kind != "platt" else {
                "coef": float(self._calibrator.coef_.ravel()[0]),
                "intercept": float(self._calibrator.intercept_.ravel()[0]),
            },
            flags=flags or {},
        )
        meta_path = os.path.join(out_dir, "student.meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)
        return onnx_path, meta_path
