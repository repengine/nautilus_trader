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
LightGBM trainer for Nautilus Trader ML models.

This module provides a trainer class for LightGBM models that integrates with the
Nautilus Trader ML infrastructure and follows consistent patterns for training,
evaluation, and model serialization.

"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ml._imports import HAS_LIGHTGBM
from ml._imports import check_ml_dependencies
from ml._imports import lgb
from ml.config.lightgbm import LightGBMTrainingConfig
from ml.training.base import BaseMLTrainer


class LightGBMTrainer(BaseMLTrainer):
    """
    LightGBM trainer for gradient boosting models.

    This trainer provides a consistent interface for training LightGBM models
    with Nautilus Trader data, including proper validation, cross-validation,
    and model serialization.

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
            Configuration for LightGBM training parameters.

        Raises
        ------
        ImportError
            If LightGBM is not installed.

        """
        super().__init__(config)
        self._config = config

        # Validate LightGBM availability
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train LightGBM model (internal implementation).

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target values.
        X_val : np.ndarray
            Validation feature matrix for early stopping.
        y_val : np.ndarray
            Validation target values for early stopping.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            Training results containing model and metrics.

        """
        self._log_info("Starting LightGBM training")

        # Get LightGBM parameters - cast to LightGBMTrainingConfig
        from ml.config.lightgbm import LightGBMTrainingConfig

        lgb_config = self._config
        assert isinstance(lgb_config, LightGBMTrainingConfig)
        lgb_params = lgb_config.get_lgb_params()

        # Create training dataset
        train_data = lgb.Dataset(X_train, label=y_train)

        # Setup validation - always provided in this method signature
        valid_sets = []
        valid_names = []
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets.append(val_data)
        valid_names.append("validation")

        # Setup callbacks
        callbacks = []
        if lgb_config.early_stopping_rounds > 0 and valid_sets:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=lgb_config.early_stopping_rounds,
                    verbose=lgb_config.verbosity >= 0,
                ),
            )

        # Train model
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Calculate feature importance
        feature_importance = model.feature_importance(importance_type="gain")

        results = {
            "model": model,
            "feature_importance": feature_importance,
            "best_iteration": model.best_iteration,
            "num_features": model.num_feature(),
            "params": lgb_params,
        }

        self._log_info(f"Training completed. Best iteration: {model.best_iteration}")
        return results

    def predict(self, model: Any, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Make predictions with trained LightGBM model.

        Parameters
        ----------
        model : Any
            Trained LightGBM model.
        X : np.ndarray
            Feature matrix for prediction.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Model predictions.

        """
        return np.asarray(model.predict(X, num_iteration=model.best_iteration))

    def save_trained_model(self, model: Any, path: str | Path) -> None:
        """
        Save trained LightGBM model.

        Parameters
        ----------
        model : Any
            Trained LightGBM model to save.
        path : str | Path
            Output path for the model.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_model"):
            # Native LightGBM model
            model.save_model(str(path))
        else:
            # Fallback to pickle
            with open(path, "wb") as f:
                pickle.dump(model, f)

        self._log_info(f"Model saved to {path}")

    def load_trained_model(self, path: str | Path) -> Any:
        """
        Load trained LightGBM model.

        Parameters
        ----------
        path : str | Path
            Path to the saved model.

        Returns
        -------
        Any
            Loaded LightGBM model.

        """
        path = Path(path)

        try:
            # Try loading as native LightGBM model
            model = lgb.Booster(model_file=str(path))
        except Exception:
            # Fallback to pickle
            with open(path, "rb") as f:
                model = pickle.load(f)

        self._log_info(f"Model loaded from {path}")
        return model

    def prepare_data(
        self,
        data: Any,  # pl.DataFrame when polars is available
        target_col: str = "target",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Prepare features and target for training.

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
        if hasattr(data, "drop"):  # pandas/polars DataFrame
            # Extract features and target
            feature_cols = [col for col in data.columns if col != target_col]
            X = (
                data.select(feature_cols).to_numpy()
                if hasattr(data, "select")
                else data[feature_cols].values
            )
            y = (
                data.select([target_col]).to_numpy().ravel()
                if hasattr(data, "select")
                else data[target_col].values
            )

            metadata = {
                "feature_names": feature_cols,
                "n_features": len(feature_cols),
                "n_samples": len(data),
            }
        else:
            # Assume numpy arrays or similar
            X = data[:, :-1]  # All columns except last
            y = data[:, -1]  # Last column as target

            metadata = {
                "feature_names": [f"feature_{i}" for i in range(X.shape[1])],
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }

        return X, y, metadata


# Explicit exports
__all__ = [
    "LightGBMTrainer",
]
