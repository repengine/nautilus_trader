"""
LightGBM trainer for financial time series prediction (non-distilled).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import lgb
from ml._imports import pl
from ml.config.lightgbm import LightGBMTrainingConfig
from ml.training.base import BaseMLTrainer
from ml.training.export import ModelExportMixin


if TYPE_CHECKING:
    import lightgbm as lgb
    import optuna
    import polars as pl


class LightGBMTrainer(BaseMLTrainer, ModelExportMixin):
    def __init__(self, config: LightGBMTrainingConfig) -> None:
        super().__init__(config)
        self._lgb_config: LightGBMTrainingConfig = config
        self._booster: lgb.Booster | None = None
        self._categorical_features: list[int] = []
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

    def prepare_data(
        self,
        data: Any,
        target_col: str = "target",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])  # Will raise with install guidance
        # Replace asserts with explicit validation to avoid stripping under -O
        if pl is None:  # pragma: no cover - defensive
            from ml.common.metrics_manager import MetricsManager as _MM

            try:
                _MM.default().inc(
                    "ml_dependency_missing_total",
                    "Missing optional dependency in training",
                    labels={"dep": "polars", "component": "training_lightgbm"},
                    labelnames=("dep", "component"),
                )
            except Exception as log_exc:
                import logging as _logging

                _logging.getLogger(__name__).debug(
                    "Incrementing dependency-missing metric failed: %s",
                    log_exc,
                    exc_info=True,
                )
            raise RuntimeError(
                "Polars is required for LightGBMTrainer.prepare_data; install the 'polars' extra",
            )
        if not isinstance(data, pl.DataFrame):
            data = pl.DataFrame(data)
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        y = data[target_col].to_numpy()
        feature_cols = [col for col in data.columns if col != target_col]
        self._categorical_features = []
        for i, col in enumerate(feature_cols):
            if data[col].dtype in [pl.Categorical, pl.Utf8]:
                self._categorical_features.append(i)
                data = data.with_columns(pl.col(col).cast(pl.Categorical).to_physical().alias(col))
        X = data.select(feature_cols).to_numpy()
        metadata = {
            "feature_names": feature_cols,
            "categorical_features": self._categorical_features,
            "n_samples": len(data),
            "n_features": len(feature_cols),
        }
        return X, y, metadata

    def _train_model(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> dict[str, Any]:
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self._feature_names if self._feature_names else "auto",
            categorical_feature=(
                self._categorical_features if self._categorical_features else "auto"
            ),
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data,
            feature_name=self._feature_names if self._feature_names else "auto",
            categorical_feature=(
                self._categorical_features if self._categorical_features else "auto"
            ),
        )
        params = self._get_model_params()
        params.update(kwargs)
        if self._lgb_config.gpu_config and self._lgb_config.gpu_config.enabled:
            params["device"] = "gpu"
            params["gpu_platform_id"] = self._lgb_config.gpu_config.platform_id
            params["gpu_device_id"] = self._lgb_config.gpu_config.device_id
        if self._lgb_config.goss_config and self._lgb_config.goss_config.enabled:
            params["boosting_type"] = "goss"
            params["top_rate"] = self._lgb_config.goss_config.top_rate
            params["other_rate"] = self._lgb_config.goss_config.other_rate
        if self._lgb_config.dart_config and self._lgb_config.dart_config.enabled:
            params["boosting_type"] = "dart"
            params["drop_rate"] = self._lgb_config.dart_config.drop_rate
            params["max_drop"] = self._lgb_config.dart_config.max_drop
            params["skip_drop"] = self._lgb_config.dart_config.skip_drop
            params["uniform_drop"] = self._lgb_config.dart_config.uniform_drop
        if self._lgb_config.efb_config and self._lgb_config.efb_config.enabled:
            params["enable_bundle"] = True
            params["max_conflict_rate"] = self._lgb_config.efb_config.max_conflict_rate
            if self._lgb_config.efb_config.bundle_size > 0:
                params["max_bundle"] = self._lgb_config.efb_config.bundle_size
        callbacks: list[Any] = [
            lgb.early_stopping(self._lgb_config.early_stopping_rounds),
            lgb.log_evaluation(period=0),
        ]
        self._booster = lgb.train(
            params,
            train_data,
            num_boost_round=self._lgb_config.n_estimators,
            valid_sets=[val_data],
            valid_names=["eval"],
            callbacks=callbacks,
        )
        best_iteration = (
            self._booster.best_iteration if hasattr(self._booster, "best_iteration") else None
        )
        metrics = {
            "best_iteration": best_iteration,
            "feature_importance": (
                dict(
                    zip(
                        self._feature_names,
                        self._booster.feature_importance(importance_type="gain"),
                    ),
                )
                if self._feature_names
                else {}
            ),
        }
        return {"model": self._booster, "metrics": metrics}

    def predict(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float32]:
        predictions = model.predict(X, num_iteration=model.best_iteration)
        if self._lgb_config.objective in ["binary", "multiclass"]:
            if self._lgb_config.objective == "binary":
                if kwargs.get("return_labels", False):
                    threshold = kwargs.get("threshold", 0.5)
                    predictions = (predictions > threshold).astype(int)
            else:
                if kwargs.get("return_labels", False):
                    predictions = np.argmax(predictions, axis=1)
        return np.array(predictions, dtype=np.float32)

    def _create_model(self, params: dict[str, Any]) -> Any:
        return params

    def _get_model_params(self) -> dict[str, Any]:
        params = {
            "objective": self._lgb_config.objective,
            "metric": self._lgb_config.metric,
            "boosting_type": self._lgb_config.boosting_type,
            "num_leaves": self._lgb_config.num_leaves,
            "max_depth": self._lgb_config.max_depth,
            "learning_rate": self._lgb_config.learning_rate,
            "feature_fraction": self._lgb_config.feature_fraction,
            "bagging_fraction": self._lgb_config.bagging_fraction,
            "bagging_freq": self._lgb_config.bagging_freq,
            "lambda_l1": self._lgb_config.reg_alpha,
            "lambda_l2": self._lgb_config.reg_lambda,
            "min_child_samples": self._lgb_config.min_child_samples,
            "verbosity": -1,
            "seed": 42,
        }
        if self._lgb_config.scale_pos_weight is not None:
            params["scale_pos_weight"] = self._lgb_config.scale_pos_weight
        return {k: v for k, v in params.items() if v is not None}

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Suggest hyperparameters for Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.

        Returns
        -------
        dict[str, Any]
            Suggested hyperparameters.

        """
        return {
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

    def _convert_to_onnx(self, model: Any, path: Path) -> None:
        try:
            from onnxmltools.convert.common.data_types import FloatTensorType

            from ml._imports import onnxmltools
            from ml.config.names import ONNX_INPUT_NAME
            from ml.training.export import DEFAULT_ONNX_OPSET

            initial_type = [(ONNX_INPUT_NAME, FloatTensorType([None, len(self._feature_names)]))]
            if onnxmltools is None:
                raise ImportError("onnxmltools not installed")
            onnx_model = onnxmltools.convert_lightgbm(
                model,
                initial_types=initial_type,
                target_opset=DEFAULT_ONNX_OPSET,
            )
            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())
        except ImportError:
            self._log_warning("onnxmltools not installed. Install with: pip install onnxmltools")
            model.save_model(str(path.with_suffix(".txt")), num_iteration=model.best_iteration)
            self._log_info(f"Model saved in LightGBM text format: {path.with_suffix('.txt')}")

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self._is_fitted or self._booster is None:
            return None
        importance = self._booster.feature_importance(importance_type="gain")
        if self._feature_names and len(self._feature_names) == len(importance):
            return dict(zip(self._feature_names, importance))
        return None

    def plot_importance(
        self,
        importance_type: str = "gain",
        max_features: int = 20,
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        if not self._is_fitted or self._booster is None:
            raise ValueError("Model must be fitted before plotting importance")
        try:
            import matplotlib.pyplot as plt

            importance = self._booster.feature_importance(importance_type=importance_type)
            feature_names = (
                self._feature_names
                if self._feature_names
                else [f"f{i}" for i in range(len(importance))]
            )
            indices = np.argsort(importance)[-max_features:]
            sorted_importance = importance[indices]
            sorted_names = [feature_names[i] for i in indices]
            plt.figure(figsize=figsize)
            plt.barh(range(len(indices)), sorted_importance)
            plt.yticks(range(len(indices)), sorted_names)
            plt.xlabel(f"Feature Importance ({importance_type})")
            plt.title("LightGBM Feature Importance")
            plt.tight_layout()
            plt.show()
        except ImportError:
            self._log_warning("matplotlib not installed. Install with: pip install matplotlib")

    def get_model(self) -> Any:
        return self._booster if self._booster is not None else self._model

    def get_feature_names(self) -> list[str]:
        return self._feature_names

    def get_training_metadata(self) -> dict[str, Any]:
        return {
            **self._training_metrics,
            "categorical_features": self._categorical_features,
            "config": {
                "objective": self._lgb_config.objective,
                "n_estimators": self._lgb_config.n_estimators,
                "num_leaves": self._lgb_config.num_leaves,
                "learning_rate": self._lgb_config.learning_rate,
            },
        }

    def save_model(self, path: str | Path) -> None:
        if not self._is_fitted or self._booster is None:
            raise ValueError("Model must be fitted before saving")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix not in {".txt", ".lgb"}:
            save_path = save_path.with_suffix(".txt")
        self._booster.save_model(str(save_path), num_iteration=self._booster.best_iteration)
        self._log_info(f"LightGBM model saved to {save_path}")
        metadata_path = save_path.with_suffix(save_path.suffix + ".meta.json")
        best_iteration = getattr(self._booster, "best_iteration", None)
        if best_iteration is not None and not isinstance(best_iteration, int):
            best_iteration = None
        metadata = {
            "model_type": "lightgbm",
            "path": str(save_path),
            "input_shape": [None, len(self._feature_names)],
            "output_shape": [None, 1],
            "best_iteration": best_iteration,
            "training_metadata": {
                "feature_names": self._feature_names,
                "categorical_features": self._categorical_features,
                "training_metrics": self._training_metrics,
                "trainer_class": self.__class__.__name__,
                "config": {
                    "objective": self._lgb_config.objective,
                    "n_estimators": self._lgb_config.n_estimators,
                    "num_leaves": self._lgb_config.num_leaves,
                },
            },
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_model(self, path: str | Path) -> None:
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        self._booster = lgb.Booster(model_file=str(load_path))
        self._model = self._booster
        self._is_fitted = True
        metadata_path = load_path.with_suffix(load_path.suffix + ".meta.json")
        if not metadata_path.exists():
            metadata_path = load_path.with_suffix(".meta")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                if "training_metadata" in metadata:
                    training_meta = metadata["training_metadata"]
                    self._feature_names = training_meta.get("feature_names", [])
                    self._categorical_features = training_meta.get("categorical_features", [])
                    self._training_metrics = training_meta.get("training_metrics", {})
                else:
                    self._feature_names = metadata.get("feature_names", [])
                    self._categorical_features = metadata.get("categorical_features", [])
                    self._training_metrics = metadata.get("training_metrics", {})
        self._log_info(f"LightGBM model loaded from {load_path}")


UnifiedLightGBMTrainer = LightGBMTrainer
