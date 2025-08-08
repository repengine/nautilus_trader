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

import logging
import pickle
import time
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ml._imports import HAS_MLFLOW
from ml._imports import HAS_ONNX
from ml._imports import HAS_OPTUNA
from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import mlflow
from ml._imports import optuna
from ml.config.base import MLFeatureConfig
from ml.config.base import MLTrainingConfig


if TYPE_CHECKING:
    import mlflow
    import optuna


logger = logging.getLogger(__name__)


class BaseMLTrainer(ABC):
    """
    Base class for ML model trainers.

    This class provides a consistent interface for training ML models on
    financial data, including data preparation, feature engineering,
    model training, and evaluation.

    Key features:
    - Standardized data preparation pipeline
    - Feature engineering integration
    - Cross-validation support
    - Performance evaluation metrics
    - Model serialization support
    - Training metrics tracking
    - Optuna hyperparameter optimization
    - MLflow experiment tracking
    - ONNX model export

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

        # Optional components
        self._mlflow_run_id: str | None = None
        self._optuna_study: optuna.Study | None = None
        self._cv_results: list[dict[str, float]] = []

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
        2. Hyperparameter optimization (if configured)
        3. Model training with cross-validation
        4. Model evaluation and metrics calculation
        5. MLflow tracking (if configured)
        6. Model serialization (if configured)

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
            check_ml_dependencies(["polars"])

        self._log_info(f"Starting training with {len(data)} samples")

        # Start MLflow run if configured
        if self._should_use_mlflow():
            self._start_mlflow_run()

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

        self._log_info(f"Training features: {X_train.shape}, Validation features: {X_val.shape}")

        # Hyperparameter optimization if configured
        best_params = {}
        if self._should_use_optuna():
            best_params = self._optimize_hyperparameters(
                X_train,
                y_train,
                X_val,
                y_val,
                **kwargs,
            )
            kwargs.update(best_params)

        # Cross-validation if configured
        if self._should_use_cv():
            cv_results = self._cross_validate(X_train, y_train, **kwargs)
            self._cv_results = cv_results

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
            predictions = self.predict(self._model, X_val)
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

        # Add CV results if available
        if self._cv_results:
            all_metrics["cv_scores"] = self._cv_results

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

        self._log_info(f"Training completed in {training_time:.2f}s")
        self._log_info(f"Validation metrics: {val_metrics}")

        # Track with MLflow
        if self._should_use_mlflow():
            self._track_with_mlflow(self._training_metrics)

        # Save model if configured
        if self._config.save_model_path is not None:
            self.save_model(self._config.save_model_path)

        # Export to ONNX if configured
        if hasattr(self._config, "export_onnx") and self._config.export_onnx:
            onnx_path = Path(self._config.save_model_path).with_suffix(".onnx")
            self.export_to_onnx(onnx_path)

        # End MLflow run
        if self._should_use_mlflow():
            self._end_mlflow_run()

        return {
            "model": self._model,
            "metrics": self._training_metrics,
            "feature_names": self._feature_names,
            "config": self._config,
            "best_params": best_params,
        }

    def _log_info(self, message: str) -> None:
        """
        Log info message.
        """
        logger.info(message)

    def _log_warning(self, message: str) -> None:
        """
        Log warning message.
        """
        logger.warning(message)

    def _log_error(self, message: str) -> None:
        """
        Log error message.
        """
        logger.error(message)

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

    @abstractmethod
    def predict(self, model: Any, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Make predictions using the trained model.

        Parameters
        ----------
        model : Any
            The trained model.
        X : np.ndarray
            Features to predict on.
        **kwargs : Any
            Additional prediction parameters.

        Returns
        -------
        np.ndarray
            Model predictions.

        """
        ...

    @abstractmethod
    def _create_model(self, params: dict[str, Any]) -> Any:
        """
        Create a model instance with given parameters.

        Parameters
        ----------
        params : dict[str, Any]
            Model parameters.

        Returns
        -------
        Any
            Model instance.

        """
        ...

    def _get_model_params(self) -> dict[str, Any]:
        """
        Get model-specific default parameters.

        Returns
        -------
        dict[str, Any]
            Default model parameters.

        """
        if getattr(self._config, "model_params", None) is not None:
            return self._config.model_params  # type: ignore[return-value]
        raise NotImplementedError("Subclasses must implement _get_model_params")

    def _should_use_mlflow(self) -> bool:
        """
        Check if MLflow should be used.
        """
        return (
            HAS_MLFLOW
            and hasattr(self._config, "mlflow_config")
            and self._config.mlflow_config is not None
        )

    def _should_use_optuna(self) -> bool:
        """
        Check if Optuna should be used.
        """
        return (
            HAS_OPTUNA
            and hasattr(self._config, "optuna_config")
            and self._config.optuna_config is not None
        )

    def _should_use_cv(self) -> bool:
        """
        Check if cross-validation should be used.
        """
        return (
            hasattr(self._config, "cv_folds")
            and self._config.cv_folds is not None
            and self._config.cv_folds > 1
        )

    def _optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

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
            Additional parameters.

        Returns
        -------
        dict[str, Any]
            Best hyperparameters found.

        """
        if not HAS_OPTUNA:
            check_ml_dependencies(["optuna"])

        self._log_info("Starting hyperparameter optimization with Optuna")

        def objective(trial: optuna.Trial) -> float:
            # Get suggested parameters from model-specific method
            params = self._suggest_hyperparameters(trial)
            params.update(kwargs)

            # Train model with suggested params
            model = self._create_model(params)
            if hasattr(model, "fit"):
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                # For models that don't have fit method (e.g., XGBoost native)
                model = self._train_with_params(X_train, y_train, X_val, y_val, params)

            # Evaluate
            predictions = self.predict(model, X_val)
            metrics = self._calculate_objective_metric(y_val, predictions)
            return metrics

        # Create study
        study = optuna.create_study(
            direction="maximize" if self._is_classification_problem(y_train) else "minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Optimize
        n_trials = getattr(self._config.optuna_config, "n_trials", 100)
        study.optimize(objective, n_trials=n_trials)

        self._optuna_study = study
        best_params = study.best_params
        self._log_info(f"Best parameters found: {best_params}")

        return best_params

    @abstractmethod
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
        ...

    def _train_with_params(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any],
    ) -> Any:
        """
        Train model with specific parameters (for Optuna).

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
        params : dict[str, Any]
            Model parameters.

        Returns
        -------
        Any
            Trained model.

        """
        results = self._train_model(X_train, y_train, X_val, y_val, **params)
        return results["model"]

    def _calculate_objective_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Calculate metric for Optuna optimization.

        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        float
            Metric value.

        """
        if self._is_classification_problem(y_true):
            # Use accuracy for classification
            return np.mean(y_true == y_pred)
        else:
            # Use negative MSE for regression (Optuna maximizes by default)
            return -np.mean((y_true - y_pred) ** 2)

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Perform cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Features.
        y : np.ndarray
            Targets.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        list[dict[str, float]]
            Cross-validation results for each fold.

        """
        n_folds = getattr(self._config, "cv_folds", 5)
        cv_strategy = getattr(self._config, "cv_strategy", "time_series")

        self._log_info(f"Starting {n_folds}-fold {cv_strategy} cross-validation")

        if cv_strategy == "time_series":
            return self._time_series_cv(X, y, n_folds, **kwargs)
        else:
            return self._standard_cv(X, y, n_folds, **kwargs)

    def _time_series_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Time series cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Features.
        y : np.ndarray
            Targets.
        n_folds : int
            Number of folds.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        list[dict[str, float]]
            CV results.

        """
        n_samples = len(X)
        fold_size = n_samples // (n_folds + 1)
        results = []

        for i in range(n_folds):
            train_end = (i + 1) * fold_size
            val_start = train_end
            val_end = min(val_start + fold_size, n_samples)

            X_train_cv = X[:train_end]
            y_train_cv = y[:train_end]
            X_val_cv = X[val_start:val_end]
            y_val_cv = y[val_start:val_end]

            # Train model for this fold
            model = self._create_model(self._get_model_params())
            if hasattr(model, "fit"):
                model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=False)
            else:
                model = self._train_with_params(X_train_cv, y_train_cv, X_val_cv, y_val_cv, kwargs)

            # Evaluate
            predictions = self.predict(model, X_val_cv)
            fold_metrics = self.evaluate(model, X_val_cv, y_val_cv)
            results.append(fold_metrics)

        # Calculate average metrics
        avg_metrics = {}
        for key in results[0].keys():
            avg_metrics[f"cv_{key}_mean"] = np.mean([r[key] for r in results])
            avg_metrics[f"cv_{key}_std"] = np.std([r[key] for r in results])

        self._log_info(f"CV results: {avg_metrics}")
        return results

    def _standard_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Standard k-fold cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Features.
        y : np.ndarray
            Targets.
        n_folds : int
            Number of folds.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        list[dict[str, float]]
            CV results.

        """
        try:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            results = []

            for train_idx, val_idx in kf.split(X):
                X_train_cv = X[train_idx]
                y_train_cv = y[train_idx]
                X_val_cv = X[val_idx]
                y_val_cv = y[val_idx]

                # Train model for this fold
                model = self._create_model(self._get_model_params())
                if hasattr(model, "fit"):
                    model.fit(
                        X_train_cv,
                        y_train_cv,
                        eval_set=[(X_val_cv, y_val_cv)],
                        verbose=False,
                    )
                else:
                    model = self._train_with_params(
                        X_train_cv,
                        y_train_cv,
                        X_val_cv,
                        y_val_cv,
                        kwargs,
                    )

                # Evaluate
                fold_metrics = self.evaluate(model, X_val_cv, y_val_cv)
                results.append(fold_metrics)

            return results

        except ImportError:
            self._log_warning("scikit-learn not available, falling back to simple CV")
            return self._time_series_cv(X, y, n_folds, **kwargs)

    def _start_mlflow_run(self) -> None:
        """
        Start MLflow run for experiment tracking.
        """
        if not HAS_MLFLOW:
            return

        mlflow_config = self._config.mlflow_config
        if mlflow_config.tracking_uri:
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        if mlflow_config.experiment_name:
            mlflow.set_experiment(mlflow_config.experiment_name)

        run = mlflow.start_run(run_name=mlflow_config.run_name)
        self._mlflow_run_id = run.info.run_id

        # Log parameters
        mlflow.log_params(self._config_to_dict())

    def _track_with_mlflow(self, metrics: dict[str, Any]) -> None:
        """
        Track metrics with MLflow.
        """
        if not HAS_MLFLOW or self._mlflow_run_id is None:
            return

        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                for i, v in enumerate(value):
                    mlflow.log_metric(f"{key}_{i}", v)

    def _end_mlflow_run(self) -> None:
        """
        End MLflow run.
        """
        if HAS_MLFLOW and self._mlflow_run_id is not None:
            mlflow.end_run()

    def _config_to_dict(self) -> dict[str, Any]:
        """
        Convert config to dictionary for MLflow logging.
        """
        config_dict = {}
        for key, value in vars(self._config).items():
            if isinstance(value, (str, int, float, bool)):
                config_dict[key] = value
        return config_dict

    def export_to_onnx(self, path: str | Path) -> None:
        """
        Export trained model to ONNX format.

        Parameters
        ----------
        path : str | Path
            Path to save ONNX model.

        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before exporting to ONNX")

        if not HAS_ONNX:
            check_ml_dependencies(["onnx"])

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Call model-specific ONNX conversion
        self._convert_to_onnx(self._model, save_path)
        self._log_info(f"Model exported to ONNX: {save_path}")

    @abstractmethod
    def _convert_to_onnx(self, model: Any, path: Path) -> None:
        """
        Convert model to ONNX format (model-specific implementation).

        Parameters
        ----------
        model : Any
            Trained model.
        path : Path
            Path to save ONNX model.

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
        predictions = self.predict(model, X)

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

        # Save model with metadata
        model_data = {
            "model": self._model,
            "feature_names": self._feature_names,
            "training_metrics": self._training_metrics,
        }

        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

        self._log_info(f"Model saved to {save_path}")

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

        self._log_info(f"Model loaded from {load_path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Get feature importance from the trained model.

        Returns
        -------
        dict[str, float] | None
            Feature importance scores or None if not available.

        """
        if not self._is_fitted or self._model is None:
            return None

        # Try to get feature importance from the model
        importance = None
        if hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        elif hasattr(self._model, "get_score"):
            importance_dict = self._model.get_score(importance_type="gain")
            if importance_dict:
                # Convert to array format
                importance = np.zeros(len(self._feature_names))
                for fname, imp in importance_dict.items():
                    if fname in self._feature_names:
                        idx = self._feature_names.index(fname)
                        importance[idx] = imp

        if importance is not None and len(self._feature_names) == len(importance):
            return dict(zip(self._feature_names, importance))

        return None

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
