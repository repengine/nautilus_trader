"""
LightGBM student distillation implementation.

This module provides functionality for training LightGBM student models using teacher
soft labels (knowledge distillation).

"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_ONNX
from ml._imports import check_ml_dependencies
from ml._imports import lgb


class LightGBMStudentDistiller:
    """
    Distill a LightGBM student model from teacher soft labels.

    Parameters
    ----------
    objective : str
        Distillation objective ('logit_mse', 'soft_ce', 'hybrid')
    kd_lambda : float
        Weight for knowledge distillation loss
    early_stopping : int
        Early stopping rounds
    opset : int
        ONNX opset version

    """

    def __init__(
        self,
        objective: str = "logit_mse",
        kd_lambda: float = 0.5,
        early_stopping: int = 200,
        opset: int = 17,
    ) -> None:
        """
        Initialize distiller.
        """
        self.objective = objective
        self.kd_lambda = kd_lambda
        self.early_stopping = early_stopping
        self.opset = opset
        self.model = None

    def fit(
        self,
        X_train: npt.NDArray[np.float32],
        q_train: npt.NDArray[np.float32],
        X_val: npt.NDArray[np.float32],
        y_val_true: npt.NDArray[np.float32] | None = None,
    ) -> None:
        """
        Fit student model using teacher soft labels.

        Parameters
        ----------
        X_train : npt.NDArray[np.float32]
            Training features
        q_train : npt.NDArray[np.float32]
            Teacher soft labels (probabilities)
        X_val : npt.NDArray[np.float32]
            Validation features
        y_val_true : npt.NDArray[np.float32] | None
            True validation labels for hybrid loss

        """
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        # Convert soft labels to target based on objective
        if self.objective == "logit_mse":
            # Use logits as target
            y_train = np.log(q_train / (1 - q_train + 1e-10))
        else:
            # Use probabilities directly
            y_train = q_train

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(
            X_val,
            label=y_val_true if y_val_true is not None else q_train[: len(X_val)],
        )

        # Training parameters
        params = {
            "objective": "regression" if self.objective == "logit_mse" else "binary",
            "metric": "rmse" if self.objective == "logit_mse" else "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 0,
        }

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(self.early_stopping), lgb.log_evaluation(0)],
        )

    def export_onnx(
        self,
        feature_names: list[str],
        out_dir: str | Path,
        model_id: str,
        flags: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """
        Export model to ONNX format with metadata.

        Parameters
        ----------
        feature_names : list[str]
            Feature names
        out_dir : str | Path
            Output directory
        model_id : str
            Model identifier
        flags : dict[str, Any] | None
            Additional metadata flags

        Returns
        -------
        tuple[Path, Path]
            Paths to ONNX model and metadata files

        """
        if not HAS_ONNX:
            check_ml_dependencies(["onnxmltools", "onnxconverter-common"])

        if self.model is None:
            raise ValueError("Model not fitted yet")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Export to ONNX
        import onnxmltools
        from onnxconverter_common import FloatTensorType

        initial_types = [("features", FloatTensorType([None, len(feature_names)]))]
        onnx_model = onnxmltools.convert_lightgbm(
            self.model,
            initial_types=initial_types,
            target_opset=self.opset,
        )

        # Save ONNX model
        onnx_path = out_dir / f"{model_id}.onnx"
        onnxmltools.save_model(onnx_model, str(onnx_path))

        # Save metadata
        metadata = {
            "model_id": model_id,
            "feature_names": feature_names,
            "objective": self.objective,
            "kd_lambda": self.kd_lambda,
            "flags": flags or {},
        }

        meta_path = out_dir / f"{model_id}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return onnx_path, meta_path


def schema_hash(feature_names: list[str], dtypes: list[str]) -> str:
    """
    Compute hash of feature schema.

    Parameters
    ----------
    feature_names : list[str]
        Feature names
    dtypes : list[str]
        Data types

    Returns
    -------
    str
        SHA256 hash of schema

    """
    schema_str = "|".join(f"{name}:{dtype}" for name, dtype in zip(feature_names, dtypes))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
