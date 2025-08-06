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
Base trainer class for ML model training.

This module provides the foundation for building ML model trainers that work with
Nautilus Trader data and follow consistent patterns for training, evaluation, and model
serialization.

"""

from __future__ import annotations

import pickle
import time
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ml.config.base import MLFeatureConfig
from ml.config.base import MLTrainingConfig


try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None  # type: ignore[assignment]


class BaseMLTrainer(ABC):
    """
    Base class for ML model trainers.

    This class provides a consistent interface for training ML models on
    financial data, including data preparation, feature engineering,
    model training, and evaluation.

    Key features:
    - Standardized data preparation pipeline
    - Feature engineering integration
    - Performance evaluation metrics
    - Model serialization support
    - Training metrics tracking

    Parameters
    ----------
    config : MLTrainingConfig
        The configuration for model training.

    """

    def __init__(self, config: MLTrainingConfig) -> None:
        """
        Initialize the ML trainer.

        Parameters
        ----------
        config : MLTrainingConfig
            The configuration for model training.

        """
        self._config = config
        self._feature_config = config.feature_config or MLFeatureConfig()

        # Training state
        self._model: Any = None
        self._feature_names: list[str] = []
        self._training_metrics: dict[str, Any] = {}
        self._is_fitted = False

    def train(
        self,
        data: Any,  # pl.DataFrame when polars is available
        validation_data: Any | None = None,  # pl.DataFrame when polars is available
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train the ML model on the provided data.

        This method orchestrates the complete training pipeline:
        1. Data preparation and feature engineering
        2. Model training with cross-validation
        3. Model evaluation and metrics calculation
        4. Model serialization (if configured)

        Parameters
        ----------
        data : Any
            The training data containing features and target (pl.DataFrame when polars available).
        validation_data : Any, optional
            Optional validation dataset. If None, data is split automatically.
        **kwargs : Any
            Additional keyword arguments passed to the training method.

        Returns
        -------
        dict[str, Any]
            Training results including model, metrics, and metadata.

        """
        start_time = time.time()

        if not HAS_POLARS:
            raise ImportError(
                "Polars is required for training. Install with: pip install polars",
            )

        print(f"Starting training with {len(data)} samples")

        # Prepare training data
        if validation_data is None:
            train_data, val_data = self._split_data(data)
        else:
            train_data, val_data = data, validation_data

        # Prepare features and targets
        X_train, y_train, metadata = self.prepare_data(
            train_data,
            self._config.target_column,
        )
        X_val, y_val, _ = self.prepare_data(
            val_data,
            self._config.target_column,
        )

        # Store feature names
        self._feature_names = metadata.get("feature_names", [])

        print(f"Training features: {X_train.shape}, Validation features: {X_val.shape}")

        # Train the model
        training_results = self._train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            **kwargs,
        )

        self._model = training_results["model"]
        self._is_fitted = True

        # Evaluate on validation set
        val_metrics = self.evaluate(self._model, X_val, y_val)

        # Calculate trading-specific metrics if returns are available
        trading_metrics = {}
        if "returns" in val_data.columns:
            predictions = self._model.predict(X_val)
            trading_metrics = self.calculate_trading_metrics(
                val_data["returns"].to_numpy(),
                predictions,
            )

        # Combine all metrics
        all_metrics = {
            **training_results.get("metrics", {}),
            **val_metrics,
            **trading_metrics,
        }

        # Store training metadata
        training_time = time.time() - start_time
        self._training_metrics = {
            "training_time": training_time,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "feature_count": X_train.shape[1],
            "feature_names": self._feature_names,
            **all_metrics,
        }

        print(f"Training completed in {training_time:.2f}s")
        print(f"Validation metrics: {val_metrics}")

        # Save model if configured
        if self._config.save_model_path is not None:
            self.save_model(self._config.save_model_path)

        return {
            "model": self._model,
            "metrics": self._training_metrics,
            "feature_names": self._feature_names,
            "config": self._config,
        }

    @abstractmethod
    def prepare_data(
        self,
        data: Any,  # pl.DataFrame when polars is available
        target_col: str = "target",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Prepare features and target for training.

        This method should implement feature engineering, data cleaning,
        and any preprocessing required for the specific model type.

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
        ...

    @abstractmethod
    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train the specific model implementation.

        This method should implement the core training logic for the
        specific model type (e.g., XGBoost, LightGBM, scikit-learn).

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
        ...

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate model performance using standard ML metrics.

        Parameters
        ----------
        model : Any
            The trained model to evaluate.
        X : np.ndarray
            Feature array for evaluation.
        y : np.ndarray
            Target array for evaluation.

        Returns
        -------
        dict[str, float]
            Dictionary of evaluation metrics.

        """
        predictions = model.predict(X)

        # Determine if this is a classification or regression problem
        if self._is_classification_problem(y):
            return self._calculate_classification_metrics(y, predictions)
        else:
            return self._calculate_regression_metrics(y, predictions)

    def calculate_trading_metrics(
        self,
        returns: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate trading-specific performance metrics.

        Parameters
        ----------
        returns : np.ndarray
            Asset returns for the prediction period.
        predictions : np.ndarray
            Model predictions (signals).

        Returns
        -------
        dict[str, float]
            Dictionary of trading metrics.

        """
        # Convert predictions to trading signals
        if self._is_classification_problem(predictions):
            # For classification, assume binary signals (0/1 -> -1/1)
            signals = np.where(predictions > 0.5, 1, -1)
        else:
            # For regression, use sign of prediction
            signals = np.sign(predictions)

        # Calculate strategy returns
        strategy_returns = returns * signals

        # Remove any NaN or infinite values
        strategy_returns = strategy_returns[np.isfinite(strategy_returns)]
        if len(strategy_returns) == 0:
            return {}

        # Calculate metrics
        metrics = {}

        # Total return
        total_return = np.prod(1 + strategy_returns) - 1
        metrics["total_return"] = total_return

        # Sharpe ratio (annualized, assuming daily returns)
        if np.std(strategy_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
            metrics["sharpe_ratio"] = sharpe_ratio

        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        metrics["max_drawdown"] = abs(max_drawdown)

        # Win rate
        winning_trades = strategy_returns > 0
        if len(winning_trades) > 0:
            win_rate = np.mean(winning_trades)
            metrics["win_rate"] = win_rate

        # Information ratio (excess return / tracking error)
        benchmark_return = np.mean(returns)
        excess_returns = strategy_returns - benchmark_return
        if np.std(excess_returns) > 0:
            information_ratio = np.mean(excess_returns) / np.std(excess_returns)
            metrics["information_ratio"] = information_ratio

        return metrics

    def save_model(self, path: str | Path) -> None:
        """
        Save the trained model to disk.

        Parameters
        ----------
        path : str | Path
            Path where to save the model.

        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model with metadata (excluding config which can't be pickled)
        model_data = {
            "model": self._model,
            "feature_names": self._feature_names,
            "training_metrics": self._training_metrics,
        }

        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {save_path}")

    def load_model(self, path: str | Path) -> None:
        """
        Load a trained model from disk.

        Parameters
        ----------
        path : str | Path
            Path to the saved model.

        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        with open(load_path, "rb") as f:
            model_data = pickle.load(f)  # noqa: S301

        self._model = model_data["model"]
        self._feature_names = model_data.get("feature_names", [])
        self._training_metrics = model_data.get("training_metrics", {})
        self._is_fitted = True

        print(f"Model loaded from {load_path}")

    def _split_data(
        self,
        data: Any,  # pl.DataFrame when polars is available
    ) -> tuple[Any, Any]:
        """
        Split data into training and validation sets.

        Parameters
        ----------
        data : Any
            The data to split (pl.DataFrame when polars available).

        Returns
        -------
        tuple[Any, Any]
            Training and validation datasets.

        """
        n_samples = len(data)
        split_idx = int(n_samples * self._config.train_test_split)

        return data[:split_idx], data[split_idx:]

    def _is_classification_problem(self, y: np.ndarray) -> bool:
        """
        Determine if this is a classification problem based on target values.

        Parameters
        ----------
        y : np.ndarray
            Target array.

        Returns
        -------
        bool
            True if classification, False if regression.

        """
        unique_values = np.unique(y)

        # If all values are integers and there are few unique values, assume classification
        if np.all(y == y.astype(int)) and len(unique_values) <= 10:
            return True

        # Check if values are in [0, 1] range (common for binary classification)
        if np.all((y >= 0) & (y <= 1)) and len(unique_values) <= 10:
            return True

        return False

    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate classification metrics.
        """
        try:
            from sklearn.metrics import accuracy_score
            from sklearn.metrics import f1_score
            from sklearn.metrics import precision_score
            from sklearn.metrics import recall_score

            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            }
        except ImportError:
            # Fallback to simple accuracy if sklearn not available
            if len(y_true) == 0 or len(y_pred) == 0:
                return {"accuracy": 0.0}
            return {
                "accuracy": np.mean(y_true == y_pred),
            }

    def _calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate regression metrics.
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
        }
