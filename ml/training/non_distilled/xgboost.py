"""
XGBoost trainer for financial time series prediction.

This module provides XGBoost-specific training functionality, leveraging the
BaseMLTrainer for common ML operations.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_POLARS
from ml._imports import HAS_XGBOOST
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml._imports import xgb
from ml.config.xgboost import XGBoostTrainingConfig
from ml.training.base import BaseMLTrainer
from ml.training.export import DEFAULT_ONNX_OPSET
from ml.training.model_exporter import ModelExportMixin


if TYPE_CHECKING:
    import optuna
    import polars as pl
    import xgboost as xgb


class XGBoostTrainer(BaseMLTrainer, ModelExportMixin):  # type: ignore[misc]
    """
    XGBoost trainer for financial time series prediction.

    Features:
    - GPU acceleration support
    - Monotonic constraints for interpretability
    - Feature importance analysis
    - SHAP value computation (optional)
    - Cross-sectional features for portfolio models
    - Built-in support for Optuna hyperparameter optimization
    - MLflow experiment tracking
    - ONNX model export

    Parameters
    ----------
    config : XGBoostTrainingConfig
        Configuration for XGBoost training.

    """

    def __init__(self, config: XGBoostTrainingConfig) -> None:
        """
        Initialize XGBoost trainer.

        Parameters
        ----------
        config : XGBoostTrainingConfig
            Configuration for XGBoost training.

        """
        super().__init__(config)
        self._xgb_config: XGBoostTrainingConfig = config

        # XGBoost-specific attributes
        self._booster: xgb.Booster | None = None
        self._dtrain: xgb.DMatrix | None = None
        self._dval: xgb.DMatrix | None = None

        # Check dependencies
        if not HAS_XGBOOST:
            check_ml_dependencies(["xgboost"])

    def prepare_data(
        self,
        data: Any,  # pl.DataFrame when polars is available
        target_col: str = "target",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
        """
        Prepare features and target for XGBoost training.

        Parameters
        ----------
        data : Any
            The input data containing features and target (pl.DataFrame when polars available).
        target_col : str, default "target"
            The name of the target column.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]
            A tuple containing:
            - X: Feature array
            - y: Target array
            - metadata: Dictionary with feature names and other metadata

        """
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        # Ensure data is a Polars DataFrame
        if not isinstance(data, pl.DataFrame):
            data = pl.DataFrame(data)

        # Extract target
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        y = data[target_col].to_numpy()

        # Get feature columns
        feature_cols = [col for col in data.columns if col != target_col]

        # Handle missing values if configured
        if self._xgb_config.handle_missing:
            data = data.fill_nan(self._xgb_config.missing_value)

        # Extract features
        X = data.select(feature_cols).to_numpy()

        # Prepare metadata
        metadata = {
            "feature_names": feature_cols,
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
        """
        Train XGBoost model.

        Parameters
        ----------
        X_train : npt.NDArray[np.float64]
            Training features.
        y_train : npt.NDArray[np.float64]
            Training targets.
        X_val : npt.NDArray[np.float64]
            Validation features.
        y_val : npt.NDArray[np.float64]
            Validation targets.
        **kwargs : Any
            Additional training parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the trained model and training metrics.

        """
        # Create DMatrix objects
        # Note: We don't pass feature_names to DMatrix to avoid ONNX conversion issues
        # XGBoost will use default names f0, f1, f2... which work with onnxmltools
        self._dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
        )
        self._dval = xgb.DMatrix(
            X_val,
            label=y_val,
        )

        # Prepare parameters
        params = self._get_model_params()
        params.update(kwargs)

        # Set GPU if configured
        if self._xgb_config.gpu_config and self._xgb_config.gpu_config.enabled:
            params["tree_method"] = "hist"
            params["device"] = f"cuda:{self._xgb_config.gpu_config.device_id}"

        # Set monotonic constraints if provided
        if self._xgb_config.monotonic_constraints:
            params["monotone_constraints"] = self._xgb_config.monotonic_constraints

        # Prepare evaluation list
        evals = [(self._dtrain, "train"), (self._dval, "eval")]
        evals_result: dict[str, dict[str, list[float]]] = {}

        # Train model
        self._booster = xgb.train(
            params,
            self._dtrain,
            num_boost_round=self._xgb_config.n_estimators,
            evals=evals,
            early_stopping_rounds=self._xgb_config.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=False,
        )

        # Get best iteration
        best_iteration = (
            self._booster.best_iteration if hasattr(self._booster, "best_iteration") else None
        )

        # Calculate training metrics
        train_score = evals_result.get("train", {})
        val_score = evals_result.get("eval", {})

        metrics = {
            "best_iteration": best_iteration,
            "train_score": train_score,
            "val_score": val_score,
        }

        return {
            "model": self._booster,
            "metrics": metrics,
        }

    def predict(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float32]:
        """
        Make predictions using XGBoost model.

        Parameters
        ----------
        model : Any
            The trained XGBoost model.
        X : npt.NDArray[np.float64]
            Features to predict on.
        **kwargs : Any
            Additional prediction parameters.
            - return_labels: bool, default False
                If True, return predicted labels for classification.
                If False (default), return probabilities/raw values.
            - threshold: float, default 0.5
                Threshold for binary classification when return_labels=True.

        Returns
        -------
        npt.NDArray[np.float32]
            Model predictions (float32 for production/inference compatibility).
            For classification: probabilities by default, labels if return_labels=True.
            For regression: predicted values.

        """
        # Create DMatrix for prediction
        # Note: We don't pass feature_names to avoid issues - XGBoost will use default names
        dmatrix = xgb.DMatrix(X)

        # Make predictions - use best_iteration if available
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None:
            predictions = model.predict(dmatrix, iteration_range=(0, best_iteration))
        else:
            predictions = model.predict(dmatrix)

        # For classification, optionally convert probabilities to labels
        if self._xgb_config.objective in ["binary:logistic", "binary:logitraw"]:
            # By default, return probabilities for ML pipeline compatibility
            if kwargs.get("return_labels", False):
                # Only apply threshold if explicitly requested
                threshold = kwargs.get("threshold", 0.5)
                predictions = (predictions > threshold).astype(int)
            # else: keep as probabilities (default behavior)

        return np.array(predictions, dtype=np.float32)

    def _create_model(self, params: dict[str, Any]) -> Any:
        """
        Create XGBoost model instance with given parameters.

        Parameters
        ----------
        params : dict[str, Any]
            Model parameters.

        Returns
        -------
        Any
            XGBoost Booster instance.

        """
        # For XGBoost, we return params that will be used with xgb.train
        # The actual model is created during training
        return params

    def _get_model_params(self) -> dict[str, Any]:
        """
        Get XGBoost-specific default parameters.

        Returns
        -------
        dict[str, Any]
            Default XGBoost parameters.

        """
        params = {
            "objective": self._xgb_config.objective,
            "eval_metric": self._xgb_config.eval_metric,
            "max_depth": self._xgb_config.max_depth,
            "learning_rate": self._xgb_config.learning_rate,
            "subsample": self._xgb_config.subsample,
            "colsample_bytree": self._xgb_config.colsample_bytree,
            "gamma": self._xgb_config.gamma,
            "reg_alpha": self._xgb_config.reg_alpha,
            "reg_lambda": self._xgb_config.reg_lambda,
            "min_child_weight": self._xgb_config.min_child_weight,
            "seed": 42,
        }

        # Add scale_pos_weight for imbalanced data
        if self._xgb_config.scale_pos_weight is not None:
            params["scale_pos_weight"] = self._xgb_config.scale_pos_weight

        # For binary classification, ensure base_score is within valid range
        # to avoid XGBoost errors when data is highly imbalanced
        if self._xgb_config.objective in ["binary:logistic", "binary:logitraw"]:
            params["base_score"] = 0.5  # Safe default for binary classification

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        return params

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
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

    def _convert_to_onnx(self, model: Any, path: Path) -> None:
        """
        Convert XGBoost model to ONNX format.

        Parameters
        ----------
        model : Any
            Trained XGBoost model.
        path : Path
            Path to save ONNX model.

        """
        try:
            # IMPORTANT: onnxmltools expects feature names in format f0, f1, f2...
            # So we need to temporarily save the model with default feature names
            import tempfile

            from onnxmltools.convert.common.data_types import FloatTensorType

            from ml._imports import onnxmltools
            from ml.config.names import ONNX_INPUT_NAME

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                # Save model without feature names (XGBoost will use f0, f1, f2...)
                model.save_model(tmp.name)
                # Reload model without feature names
                from ml._imports import xgb as _xgb

                if _xgb is None:
                    raise ImportError("xgboost not installed")
                temp_booster = _xgb.Booster()
                temp_booster.load_model(tmp.name)

                # Define input type
                initial_type = [
                    (ONNX_INPUT_NAME, FloatTensorType([None, len(self._feature_names)])),
                ]

                # Convert model using the temp booster without custom feature names
                if onnxmltools is None:
                    raise ImportError("onnxmltools not installed")
                onnx_model = onnxmltools.convert_xgboost(
                    temp_booster,
                    initial_types=initial_type,
                    target_opset=DEFAULT_ONNX_OPSET,
                )

                # Save ONNX model
                with open(path, "wb") as f:
                    f.write(onnx_model.SerializeToString())

                # Clean up temp file
                import os

                os.unlink(tmp.name)

        except ImportError:
            self._log_warning(
                "onnxmltools not installed. Install with: pip install onnxmltools",
            )
            # Fallback to xgboost native save
            model.save_model(str(path.with_suffix(".json")))
            self._log_info(f"Model saved in XGBoost JSON format: {path.with_suffix('.json')}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Get feature importance from the trained XGBoost model.

        Returns
        -------
        dict[str, float] | None
            Feature importance scores or None if not available.

        """
        if not self._is_fitted or self._booster is None:
            return None

        # Get feature importance
        importance_dict = self._booster.get_score(importance_type="gain")

        if not importance_dict:
            return None

        # Map to feature names
        if self._feature_names:
            result = {}
            for i, fname in enumerate(self._feature_names):
                # XGBoost uses f0, f1, ... if feature names aren't set
                xgb_fname = fname if fname in importance_dict else f"f{i}"
                if xgb_fname in importance_dict:
                    val = importance_dict[xgb_fname]
                    result[fname] = float(val[0] if isinstance(val, list) else val)
                else:
                    result[fname] = 0.0
            return result

        return {k: float(v[0] if isinstance(v, list) else v) for k, v in importance_dict.items()}

    def get_shap_values(
        self,
        X: npt.NDArray[np.float64],
        interaction: bool = False,
    ) -> npt.NDArray[np.float32] | None:
        """
        Calculate SHAP values for model interpretability.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Features to explain.
        interaction : bool, default False
            Whether to calculate interaction values.

        Returns
        -------
        npt.NDArray[np.float32] | None
            SHAP values (float32) or None if not available.

        """
        if not self._is_fitted or self._booster is None:
            return None

        try:
            import shap

            # Create explainer
            explainer = shap.TreeExplainer(self._booster)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X)

            if interaction:
                shap_interaction = explainer.shap_interaction_values(X)
                return np.array(shap_interaction, dtype=np.float32)

            return np.array(shap_values, dtype=np.float32)

        except ImportError:
            self._log_warning("SHAP not installed. Install with: pip install shap")
            return None

    # ModelExportMixin implementation methods
    def get_model(self) -> Any:
        """
        Get the trained model instance.
        """
        return self._booster if self._booster is not None else self._model

    def get_feature_names(self) -> list[str]:
        """
        Get the feature names used in training.
        """
        return self._feature_names

    def get_training_metadata(self) -> dict[str, Any]:
        """
        Get training metadata.
        """
        return {
            **self._training_metrics,
            "config": {
                "objective": self._xgb_config.objective,
                "n_estimators": self._xgb_config.n_estimators,
                "max_depth": self._xgb_config.max_depth,
                "learning_rate": self._xgb_config.learning_rate,
            },
        }

    def save_model(self, path: str | Path) -> None:
        """
        Save the trained XGBoost model in native JSON format.

        Parameters
        ----------
        path : str | Path
            Path where to save the model.

        """
        if not self._is_fitted or self._booster is None:
            raise ValueError("Model must be fitted before saving")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure it's saved as JSON for production use
        if save_path.suffix not in {".json", ".xgb"}:
            save_path = save_path.with_suffix(".json")

        # Save as XGBoost native JSON format
        self._booster.save_model(str(save_path))
        self._log_info(f"XGBoost model saved to {save_path}")

        # Save metadata in standard format
        metadata_path = save_path.with_suffix(save_path.suffix + ".meta.json")

        # Include best_iteration if available and is a valid integer
        best_iteration = getattr(self._booster, "best_iteration", None)
        if best_iteration is not None and not isinstance(best_iteration, int):
            # Handle mock objects or invalid types
            best_iteration = None

        metadata = {
            "model_type": "xgboost",
            "path": str(save_path),
            "input_shape": [None, len(self._feature_names)],
            "output_shape": [None, 1],
            "best_iteration": best_iteration,  # Add best_iteration for use in inference
            "training_metadata": {
                "feature_names": self._feature_names,
                "training_metrics": self._training_metrics,
                "trainer_class": self.__class__.__name__,
                "config": {
                    "objective": self._xgb_config.objective,
                    "n_estimators": self._xgb_config.n_estimators,
                    "max_depth": self._xgb_config.max_depth,
                },
            },
        }

        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_model(self, path: str | Path) -> None:
        """
        Load a trained XGBoost model.

        Parameters
        ----------
        path : str | Path
            Path to the saved model.

        """
        if not HAS_XGBOOST:
            check_ml_dependencies(["xgboost"])

        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        # Load XGBoost model
        self._booster = xgb.Booster()
        self._booster.load_model(str(load_path))
        self._model = self._booster
        self._is_fitted = True

        # Load metadata if available (try both .meta.json and .meta for backward compat)
        metadata_path = load_path.with_suffix(load_path.suffix + ".meta.json")
        if not metadata_path.exists():
            metadata_path = load_path.with_suffix(".meta")

        if metadata_path.exists():
            import json

            with open(metadata_path) as f:
                metadata = json.load(f)
                # Handle both flat and nested metadata structures
                if "training_metadata" in metadata:
                    training_meta = metadata["training_metadata"]
                    self._feature_names = training_meta.get("feature_names", [])
                    self._training_metrics = training_meta.get("training_metrics", {})
                else:
                    self._feature_names = metadata.get("feature_names", [])
                    self._training_metrics = metadata.get("training_metrics", {})

        self._log_info(f"XGBoost model loaded from {load_path}")


# Backward compatibility aliases
UnifiedXGBoostTrainer = XGBoostTrainer
