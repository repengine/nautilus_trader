# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
XGBoost trainer for financial time series prediction.

This module provides XGBoost-specific training functionality, leveraging the
BaseMLTrainer for common ML operations.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ml._imports import HAS_POLARS
from ml._imports import HAS_XGBOOST
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml._imports import xgb
from ml.config.xgboost import XGBoostTrainingConfig
from ml.training.base import BaseMLTrainer


if TYPE_CHECKING:
    import optuna
    import polars as pl
    import xgboost as xgb


class XGBoostTrainer(BaseMLTrainer):
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
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
        tuple[np.ndarray, np.ndarray, dict[str, Any]]
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
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train XGBoost model.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training targets.
        X_val : np.ndarray
            Validation features.
        y_val : np.ndarray
            Validation targets.
        **kwargs : Any
            Additional training parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the trained model and training metrics.

        """
        # Create DMatrix objects
        self._dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            feature_names=self._feature_names if self._feature_names else None,
        )
        self._dval = xgb.DMatrix(
            X_val,
            label=y_val,
            feature_names=self._feature_names if self._feature_names else None,
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

    def predict(self, model: Any, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Make predictions using XGBoost model.

        Parameters
        ----------
        model : Any
            The trained XGBoost model.
        X : np.ndarray
            Features to predict on.
        **kwargs : Any
            Additional prediction parameters.

        Returns
        -------
        np.ndarray
            Model predictions.

        """
        # Create DMatrix for prediction
        dmatrix = xgb.DMatrix(
            X,
            feature_names=self._feature_names if self._feature_names else None,
        )

        # Make predictions
        predictions = model.predict(dmatrix)

        # For classification, apply threshold if needed
        if self._xgb_config.objective in ["binary:logistic", "binary:logitraw"]:
            threshold = kwargs.get("threshold", 0.5)
            predictions = (predictions > threshold).astype(int)

        return np.array(predictions)

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
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType

            # Define input type
            initial_type = [
                ("float_input", FloatTensorType([None, len(self._feature_names)])),
            ]

            # Convert model
            onnx_model = convert_xgboost(
                model,
                initial_types=initial_type,
                target_opset=12,
            )

            # Save ONNX model
            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())

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
        X: np.ndarray,
        interaction: bool = False,
    ) -> np.ndarray | None:
        """
        Calculate SHAP values for model interpretability.

        Parameters
        ----------
        X : np.ndarray
            Features to explain.
        interaction : bool, default False
            Whether to calculate interaction values.

        Returns
        -------
        np.ndarray | None
            SHAP values or None if not available.

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
                return np.array(shap_interaction)

            return np.array(shap_values)

        except ImportError:
            self._log_warning("SHAP not installed. Install with: pip install shap")
            return None

    def save_model(self, path: str | Path) -> None:
        """
        Save the trained XGBoost model.

        Parameters
        ----------
        path : str | Path
            Path where to save the model.

        """
        if not self._is_fitted or self._booster is None:
            raise ValueError("Model must be fitted before saving")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as XGBoost native format
        self._booster.save_model(str(save_path))
        self._log_info(f"XGBoost model saved to {save_path}")

        # Also save metadata
        metadata_path = save_path.with_suffix(".meta")
        metadata = {
            "feature_names": self._feature_names,
            "training_metrics": self._training_metrics,
            "config": {
                "objective": self._xgb_config.objective,
                "n_estimators": self._xgb_config.n_estimators,
                "max_depth": self._xgb_config.max_depth,
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

        # Load metadata if available
        metadata_path = load_path.with_suffix(".meta")
        if metadata_path.exists():
            import json

            with open(metadata_path) as f:
                metadata = json.load(f)
                self._feature_names = metadata.get("feature_names", [])
                self._training_metrics = metadata.get("training_metrics", {})

        self._log_info(f"XGBoost model loaded from {load_path}")


# Backward compatibility aliases
UnifiedXGBoostTrainer = XGBoostTrainer
