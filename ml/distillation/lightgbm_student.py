"""
LightGBM Student Distiller (teacher-student)
-------------------------------------------

Production-oriented student distillation utility which:
- Trains a LightGBM student on teacher soft labels
- Calibrates on raw scores (Platt preferred) against true labels
- Exports ONNX with Sigmoid (+ optional Platt via Mul/Add) baked in
- Emits strict sidecar metadata for train-serve parity

This module avoids heavy deps on the hot path. Calibration parameters are
stored and applied numerically (no sklearn object required during inference).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


try:  # Optional: delay lightgbm import for environments without it
    import lightgbm as lgb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional at import time
    lgb = None  # type: ignore[assignment]


def schema_hash(feature_names: list[str], dtypes: list[str] | None = None) -> str:
    payload = "|".join(feature_names) + "||" + "|".join(dtypes or [])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class StudentMeta:
    model_id: str
    feature_names: list[str]
    feature_dtypes: list[str]
    feature_schema_hash: str
    output_schema: dict[str, Any]
    best_iteration: int | None
    opset: int
    trainer_version: str
    train_date_range: tuple[str, str] | None = None
    calibrator_kind: str | None = None
    calibrator_params: dict[str, float] | None = None
    flags: dict[str, Any] | None = None


class LightGBMStudentDistiller:
    """
    Student distilled from a heavy teacher.

    Parameters
    ----------
    objective : str
        One of {"logit_mse", "soft_ce", "hybrid"}.
    kd_lambda : float
        Weight for CE part in hybrid (0..1). Only used if objective=="hybrid".
    lgb_params : dict | None
        LightGBM params. Objective will be set via `fobj` if needed.
    early_stopping : int
        Early stopping rounds.
    opset : int
        ONNX opset for export.
    trainer_version : str
        Version string captured in metadata.

    """

    def __init__(
        self,
        objective: str = "logit_mse",
        kd_lambda: float = 0.5,
        lgb_params: dict[str, Any] | None = None,
        early_stopping: int = 200,
        opset: int = 17,
        trainer_version: str = "0.1.0",
    ) -> None:
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

        self.model: Any | None = None  # lightgbm Booster
        self.best_iteration: int | None = None

        # Calibration state (no sklearn object needed for hot path)
        self._calibrator_kind: str | None = None  # "platt" | "isotonic" | None
        self._platt_coef: float | None = None
        self._platt_intercept: float | None = None
        self._iso_transform: Any | None = None  # Only used during fitting/testing

    # ---------- math utils ----------
    @staticmethod
    def _sigmoid(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))  # type: ignore[no-any-return]

    @staticmethod
    def _teacher_logits(q: npt.NDArray[np.float32], eps: float = 1e-6) -> npt.NDArray[np.float32]:
        clipped = np.clip(q, eps, 1.0 - eps, dtype=np.float32)
        return np.log(clipped / (1.0 - clipped), dtype=np.float32)  # type: ignore[no-any-return]

    # ---------- custom objective (hybrid) ----------
    def _obj_hybrid(
        self,
        pred_raw: npt.NDArray[np.float64],
        dataset: Any,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        # pred_raw = student logit; labels = teacher probs q_T
        q = dataset.get_label()  # type: ignore[no-untyped-call]
        p = 1.0 / (1.0 + np.exp(-pred_raw))
        g_ce = p - q
        h_ce = p * (1.0 - p)

        z = self._teacher_logits(q.astype(np.float32)).astype(np.float64)
        g_mse = pred_raw - z
        h_mse = np.ones_like(pred_raw)

        lam = float(self.kd_lambda)
        grad = lam * g_ce + (1.0 - lam) * g_mse
        hess = lam * h_ce + (1.0 - lam) * h_mse
        return grad.astype(np.float32), hess.astype(np.float32)

    # ---------- training ----------
    def fit(
        self,
        X_train: npt.NDArray[np.float32],
        q_train: npt.NDArray[np.float32],
        X_val: npt.NDArray[np.float32],
        y_val_true: npt.NDArray[np.float32] | None = None,
    ) -> LightGBMStudentDistiller:
        if lgb is None:  # pragma: no cover - env guard
            raise ImportError("lightgbm is required to train the student.")

        X_train = np.asarray(X_train, dtype=np.float32, order="C")
        X_val = np.asarray(X_val, dtype=np.float32, order="C")
        q_train = np.asarray(q_train, dtype=np.float32)

        train_set = lgb.Dataset(X_train, label=q_train)  # type: ignore[attr-defined]
        # If no true labels, use q_train slice to keep API shape (won't be used for metrics)
        val_labels = y_val_true if y_val_true is not None else q_train[: len(X_val)]
        val_set = lgb.Dataset(X_val, label=val_labels)  # type: ignore[attr-defined]

        params = dict(self.lgb_params)
        fobj: Any | None = None
        if self.objective == "logit_mse":
            params.update({"objective": "regression", "metric": ["l2"]})
        elif self.objective == "soft_ce":
            params.update({"objective": "binary", "metric": ["binary_logloss"]})
        elif self.objective == "hybrid":
            params.update({"objective": "regression", "metric": ["l2"]})
            fobj = self._obj_hybrid
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        self.model = lgb.train(  # type: ignore[attr-defined]
            params,
            train_set,
            valid_sets=[val_set],
            fobj=fobj,
            early_stopping_rounds=self.early_stopping,
            verbose_eval=False,
        )
        self.best_iteration = getattr(self.model, "best_iteration", None)

        # Optional post-calibration on true labels (using raw scores z)
        if y_val_true is not None:
            z_val = self._predict_raw(X_val)
            self._fit_platt_on_raw(z_val, y_val_true)

        return self

    # ---------- calibration ----------
    def _fit_platt_on_raw(
        self,
        z: npt.NDArray[np.float32],
        y_true: npt.NDArray[np.float32],
    ) -> None:
        # Fit a simple logistic regression in closed form via scipy? Keep simple: use sklearn if present; else fall back to numeric fit.
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]

            lr = LogisticRegression(solver="lbfgs")
            lr.fit(z.reshape(-1, 1), y_true.astype(int))
            # Persist parameters; drop object for hot path
            coef = float(lr.coef_.ravel()[0])
            intercept = float(lr.intercept_.ravel()[0])
            self._calibrator_kind = "platt"
            self._platt_coef = np.float32(coef).item()
            self._platt_intercept = np.float32(intercept).item()
        except Exception:  # pragma: no cover - optional dependency path
            # If sklearn not available, leave uncalibrated
            self._calibrator_kind = None
            self._platt_coef = None
            self._platt_intercept = None

    # ---------- inference ----------
    def _predict_raw(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        X_c = np.asarray(X, dtype=np.float32, order="C")
        # For LightGBM binary objective, ensure raw_score=True to get logits
        try:
            return np.asarray(
                self.model.predict(X_c, num_iteration=self.best_iteration, raw_score=True),
                dtype=np.float32,
            )
        except TypeError:
            # Some boosters ignore raw_score when objective is regression; just return directly
            return np.asarray(
                self.model.predict(X_c, num_iteration=self.best_iteration),
                dtype=np.float32,
            )

    def predict_proba(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        z = self._predict_raw(X)
        # Apply Platt if present on raw scores, else vanilla sigmoid
        if (
            self._calibrator_kind == "platt"
            and self._platt_coef is not None
            and self._platt_intercept is not None
        ):
            z = (self._platt_coef * z + self._platt_intercept).astype(np.float32)
        p = self._sigmoid(z.astype(np.float32)).astype(np.float32)
        return p.reshape(-1, 1)

    # ---------- export ----------
    def export_onnx(
        self,
        feature_names: list[str],
        out_dir: str,
        model_id: str,
        train_date_range: tuple[str, str] | None = None,
        flags: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """
        Export ONNX with Sigmoid (+ optional Platt) and sidecar metadata.

        Notes
        -----
        Uses broadcasting-friendly Mul/Add nodes for Platt instead of Gemm to
        avoid shape quirks across runtimes.

        """
        if self.model is None:
            raise RuntimeError("Train the model before export.")

        # Allow tests to monkeypatch module-level symbols; otherwise import lazily
        onnx_mod = globals().get("onnx")
        onnx_helper = globals().get("onnx_helper")
        onnx_numpy_helper = globals().get("onnx_numpy_helper")
        convert_lgbm_booster = globals().get("convert_lgbm_booster")
        FloatTensorType = globals().get("FloatTensorType")

        if any(
            x is None
            for x in (
                onnx_mod,
                onnx_helper,
                onnx_numpy_helper,
                convert_lgbm_booster,
                FloatTensorType,
            )
        ):
            try:  # type: ignore[unreachable]
                import onnx as _onnx  # type: ignore[import-not-found]
                from onnx import helper as _onnx_helper  # type: ignore[import-not-found]
                from onnx import numpy_helper as _onnx_numpy_helper  # type: ignore[import-not-found]
                from onnxmltools.convert.common.data_types import FloatTensorType as _FloatTensorType  # type: ignore[import-not-found]
                from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
                    convert_lightgbm as _convert_lgbm_booster,  # type: ignore[import-not-found]
                )

                onnx_mod = _onnx
                onnx_helper = _onnx_helper
                onnx_numpy_helper = _onnx_numpy_helper
                convert_lgbm_booster = _convert_lgbm_booster
                FloatTensorType = _FloatTensorType
            except Exception as exc:  # pragma: no cover - optional dependency path
                raise ImportError("onnx/onnxmltools are required for export.") from exc

        n_features = len(feature_names)
        initial_type = [("input", FloatTensorType([None, n_features]))]  # type: ignore[operator]
        onnx_model = convert_lgbm_booster(self.model, initial_types=initial_type)  # type: ignore[operator]

        # Original model output is raw score; append calibration then Sigmoid
        raw_output_name = onnx_model.graph.output[0].name
        last = raw_output_name
        nodes: list[Any] = []
        initializers: list[Any] = []

        if (
            self._calibrator_kind == "platt"
            and self._platt_coef is not None
            and self._platt_intercept is not None
        ):
            # y = a * raw + b
            a_name = "platt_a"
            b_name = "platt_b"
            a_init = onnx_numpy_helper.from_array(np.array(self._platt_coef, dtype=np.float32), name=a_name)  # type: ignore[operator]
            b_init = onnx_numpy_helper.from_array(np.array(self._platt_intercept, dtype=np.float32), name=b_name)  # type: ignore[operator]
            initializers.extend([a_init, b_init])
            mul_out = "platt_mul_out"
            add_out = "platt_add_out"
            nodes.extend(
                [
                    onnx_helper.make_node("Mul", inputs=[last, a_name], outputs=[mul_out]),  # type: ignore[operator]
                    onnx_helper.make_node("Add", inputs=[mul_out, b_name], outputs=[add_out]),  # type: ignore[operator]
                ],
            )
            last = add_out

        prob_name = "probability"
        nodes.append(onnx_helper.make_node("Sigmoid", inputs=[last], outputs=[prob_name]))  # type: ignore[operator]

        # Append nodes and initializers
        onnx_model.graph.node.extend(nodes)
        onnx_model.graph.initializer.extend(initializers)
        onnx_model.graph.output[0].name = prob_name

        # Persist artifacts
        os.makedirs(out_dir, exist_ok=True)
        onnx_path = os.path.join(out_dir, "student.onnx")
        onnx_mod.save(onnx_model, onnx_path)  # type: ignore[operator]

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
            calibrator_params=(
                {"coef": float(self._platt_coef), "intercept": float(self._platt_intercept)}
                if self._calibrator_kind == "platt"
                and self._platt_coef is not None
                and self._platt_intercept is not None
                else None
            ),
            flags=flags or {},
        )
        meta_path = os.path.join(out_dir, "student.meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

        return onnx_path, meta_path
