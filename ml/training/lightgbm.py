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
LightGBM trainer for financial time series prediction.

This module provides LightGBM-specific training functionality, leveraging the
BaseMLTrainer for common ML operations.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import lgb
from ml._imports import pl
from ml.config.lightgbm import LightGBMTrainingConfig
from ml.training.base import BaseMLTrainer


if TYPE_CHECKING:
    import lightgbm as lgb
    import optuna
    import polars as pl


class LightGBMTrainer(BaseMLTrainer):
    """
    LightGBM trainer for financial time series prediction.

    Features:
    - GPU acceleration support
    - Categorical feature support
    - GOSS (Gradient-based One-Side Sampling) for faster training
    - DART (Dropouts meet Multiple Additive Regression Trees)
    - EFB (Exclusive Feature Bundling) for memory efficiency
    - Built-in support for Optuna hyperparameter optimization
    - MLflow experiment tracking
    - ONNX model export

    Parameters
    ----------
    config : LightGBMTrainingConfig
        Configuration for LightGBM training.

    """

    def __init__(self, config: LightGBMTrainingConfig) -> None:
        """
        Initialize LightGBM trainer.

        Parameters
        ----------
        config : LightGBMTrainingConfig
            Configuration for LightGBM training.

        """
        super().__init__(config)
        self._lgb_config: LightGBMTrainingConfig = config

        # LightGBM-specific attributes
        self._booster: lgb.Booster | None = None
        self._categorical_features: list[int] = []

        # Check dependencies
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

    def prepare_data(
        self,
        data: Any,  # pl.DataFrame when polars is available
        target_col: str = "target",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Prepare features and target for LightGBM training.

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

        # Identify categorical features
        self._categorical_features = []
        for i, col in enumerate(feature_cols):
            if data[col].dtype in [pl.Categorical, pl.Utf8]:
                self._categorical_features.append(i)
                # Convert categorical to numeric codes
                data = data.with_columns(
                    pl.col(col).cast(pl.Categorical).to_physical().alias(col),
                )

        # Extract features
        X = data.select(feature_cols).to_numpy()

        # Prepare metadata
        metadata = {
            "feature_names": feature_cols,
            "categorical_features": self._categorical_features,
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
        Train LightGBM model.

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
        # Create LightGBM datasets
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

        # Prepare parameters
        params = self._get_model_params()
        params.update(kwargs)

        # Set GPU if configured
        if self._lgb_config.gpu_config and self._lgb_config.gpu_config.enabled:
            params["device"] = "gpu"
            params["gpu_platform_id"] = self._lgb_config.gpu_config.platform_id
            params["gpu_device_id"] = self._lgb_config.gpu_config.device_id

        # Configure GOSS if enabled
        if self._lgb_config.goss_config and self._lgb_config.goss_config.enabled:
            params["boosting_type"] = "goss"
            params["top_rate"] = self._lgb_config.goss_config.top_rate
            params["other_rate"] = self._lgb_config.goss_config.other_rate

        # Configure DART if enabled
        if self._lgb_config.dart_config and self._lgb_config.dart_config.enabled:
            params["boosting_type"] = "dart"
            params["drop_rate"] = self._lgb_config.dart_config.drop_rate
            params["max_drop"] = self._lgb_config.dart_config.max_drop
            params["skip_drop"] = self._lgb_config.dart_config.skip_drop
            params["uniform_drop"] = self._lgb_config.dart_config.uniform_drop

        # Configure EFB if enabled
        if self._lgb_config.efb_config and self._lgb_config.efb_config.enabled:
            params["enable_bundle"] = True
            params["max_conflict_rate"] = self._lgb_config.efb_config.max_conflict_rate
            if self._lgb_config.efb_config.bundle_size > 0:
                params["max_bundle"] = self._lgb_config.efb_config.bundle_size

        # Callbacks for early stopping
        callbacks = [
            lgb.early_stopping(self._lgb_config.early_stopping_rounds),
            lgb.log_evaluation(period=0),  # Disable verbose output
        ]

        # Train model
        self._booster = lgb.train(
            params,
            train_data,
            num_boost_round=self._lgb_config.n_estimators,
            valid_sets=[val_data],
            valid_names=["eval"],
            callbacks=callbacks,
        )

        # Get best iteration
        best_iteration = (
            self._booster.best_iteration if hasattr(self._booster, "best_iteration") else None
        )

        # Calculate training metrics
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

        return {
            "model": self._booster,
            "metrics": metrics,
        }

    def predict(self, model: Any, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Make predictions using LightGBM model.

        Parameters
        ----------
        model : Any
            The trained LightGBM model.
        X : np.ndarray
            Features to predict on.
        **kwargs : Any
            Additional prediction parameters.

        Returns
        -------
        np.ndarray
            Model predictions.

        """
        # Make predictions
        predictions = model.predict(X, num_iteration=model.best_iteration)

        # For classification, apply threshold if needed
        if self._lgb_config.objective in ["binary", "multiclass"]:
            if self._lgb_config.objective == "binary":
                threshold = kwargs.get("threshold", 0.5)
                predictions = (predictions > threshold).astype(int)
            else:
                # For multiclass, get the class with highest probability
                predictions = np.argmax(predictions, axis=1)

        return predictions

    def _create_model(self, params: dict[str, Any]) -> Any:
        """
        Create LightGBM model instance with given parameters.

        Parameters
        ----------
        params : dict[str, Any]
            Model parameters.

        Returns
        -------
        Any
            LightGBM parameters dict (model created during training).

        """
        # For LightGBM, we return params that will be used with lgb.train
        # The actual model is created during training
        return params

    def _get_model_params(self) -> dict[str, Any]:
        """
        Get LightGBM-specific default parameters.

        Returns
        -------
        dict[str, Any]
            Default LightGBM parameters.

        """
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

        # Add scale_pos_weight for imbalanced data
        if self._lgb_config.scale_pos_weight is not None:
            params["scale_pos_weight"] = self._lgb_config.scale_pos_weight

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
        """
        Convert LightGBM model to ONNX format.

        Parameters
        ----------
        model : Any
            Trained LightGBM model.
        path : Path
            Path to save ONNX model.

        """
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType

            # Define input type
            initial_type = [
                ("float_input", FloatTensorType([None, len(self._feature_names)])),
            ]

            # Convert model
            onnx_model = convert_lightgbm(
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
            # Fallback to LightGBM native save
            model.save_model(str(path.with_suffix(".txt")), num_iteration=model.best_iteration)
            self._log_info(f"Model saved in LightGBM text format: {path.with_suffix('.txt')}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Get feature importance from the trained LightGBM model.

        Returns
        -------
        dict[str, float] | None
            Feature importance scores or None if not available.

        """
        if not self._is_fitted or self._booster is None:
            return None

        # Get feature importance
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
        """
        Plot feature importance.

        Parameters
        ----------
        importance_type : str, default "gain"
            Type of importance: "gain" or "split".
        max_features : int, default 20
            Maximum number of features to plot.
        figsize : tuple[int, int], default (10, 6)
            Figure size.

        """
        if not self._is_fitted or self._booster is None:
            raise ValueError("Model must be fitted before plotting importance")

        try:
            import matplotlib.pyplot as plt

            # Get importance
            importance = self._booster.feature_importance(importance_type=importance_type)
            feature_names = (
                self._feature_names
                if self._feature_names
                else [f"f{i}" for i in range(len(importance))]
            )

            # Sort by importance
            indices = np.argsort(importance)[-max_features:]
            sorted_importance = importance[indices]
            sorted_names = [feature_names[i] for i in indices]

            # Plot
            plt.figure(figsize=figsize)
            plt.barh(range(len(indices)), sorted_importance)
            plt.yticks(range(len(indices)), sorted_names)
            plt.xlabel(f"Feature Importance ({importance_type})")
            plt.title("LightGBM Feature Importance")
            plt.tight_layout()
            plt.show()

        except ImportError:
            self._log_warning("matplotlib not installed. Install with: pip install matplotlib")

    def save_model(self, path: str | Path) -> None:
        """
        Save the trained LightGBM model.

        Parameters
        ----------
        path : str | Path
            Path where to save the model.

        """
        if not self._is_fitted or self._booster is None:
            raise ValueError("Model must be fitted before saving")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as LightGBM native format
        self._booster.save_model(str(save_path), num_iteration=self._booster.best_iteration)
        self._log_info(f"LightGBM model saved to {save_path}")

        # Also save metadata
        metadata_path = save_path.with_suffix(".meta")
        metadata = {
            "feature_names": self._feature_names,
            "categorical_features": self._categorical_features,
            "training_metrics": self._training_metrics,
            "config": {
                "objective": self._lgb_config.objective,
                "n_estimators": self._lgb_config.n_estimators,
                "num_leaves": self._lgb_config.num_leaves,
            },
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_model(self, path: str | Path) -> None:
        """
        Load a trained LightGBM model.

        Parameters
        ----------
        path : str | Path
            Path to the saved model.

        """
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        # Load LightGBM model
        self._booster = lgb.Booster(model_file=str(load_path))
        self._model = self._booster
        self._is_fitted = True

        # Load metadata if available
        metadata_path = load_path.with_suffix(".meta")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                self._feature_names = metadata.get("feature_names", [])
                self._categorical_features = metadata.get("categorical_features", [])
                self._training_metrics = metadata.get("training_metrics", {})

        self._log_info(f"LightGBM model loaded from {load_path}")


# Backward compatibility aliases
UnifiedLightGBMTrainer = LightGBMTrainer
