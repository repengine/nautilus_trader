"""
Base trainer class for ML model training.

This module provides the foundation for building ML model trainers that work with
Nautilus Trader data and follow consistent patterns for training, evaluation, and model
serialization.

"""

from __future__ import annotations

import logging
import time
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_MLFLOW
from ml._imports import HAS_ONNX
from ml._imports import HAS_OPTUNA
from ml._imports import HAS_POLARS
from ml._imports import HAS_SKLEARN
from ml._imports import check_ml_dependencies
from ml._imports import mlflow
from ml._imports import optuna
from ml.config.base import MLFeatureConfig
from ml.config.base import MLTrainingConfig
from ml.stores.feature_store import FeatureStore


if TYPE_CHECKING:
    import mlflow
    import optuna

    # Provide names for type annotations to satisfy static checkers without runtime import
    from ml.registry import ModelManifest

    # Type-only import to satisfy linters and type checkers without runtime import


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

        # Initialize FeatureStore if database connection provided
        self._feature_store: FeatureStore | None = None
        if hasattr(config, "db_connection") and config.db_connection:
            self._feature_store = FeatureStore(
                connection_string=config.db_connection,
                feature_config=self._feature_config,
                pipeline_spec=getattr(config, "pipeline_spec", None),
            )

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
        start_time = time.perf_counter()

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
        training_time = time.perf_counter() - start_time
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
        if (
            hasattr(self._config, "export_onnx")
            and self._config.export_onnx
            and self._config.save_model_path
        ):
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

    def prepare_data_with_feature_store(
        self,
        instrument_id: str,
        start: Any,  # datetime
        end: Any,  # datetime
        compute_if_missing: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]:
        """
        Prepare training data using FeatureStore for guaranteed parity.

        Parameters
        ----------
        instrument_id : str
            Instrument to train on.
        start : Any
            Training period start.
        end : Any
            Training period end.
        compute_if_missing : bool, default True
            Whether to compute features if not already stored.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]
            X (features), y (labels), feature_names

        """
        if self._feature_store is None:
            raise ValueError("FeatureStore not configured. Provide db_connection in config.")

        # Compute and store features if needed
        if compute_if_missing:
            rows_computed = self._feature_store.compute_and_store_historical(
                instrument_id=instrument_id,
                start=start,
                end=end,
                force_recompute=False,
            )
            if rows_computed > 0:
                self._log_info(f"Computed {rows_computed} feature rows")

        # Load features from store
        features, timestamps, feature_names = self._feature_store.get_training_data(
            instrument_id=instrument_id,
            start=start,
            end=end,
            include_bars=True,
        )

        if len(features) == 0:
            raise ValueError(f"No features found for {instrument_id} in specified period")

        # Generate labels (simplified - override in subclass for specific logic)
        labels = self._generate_labels(features, timestamps)

        self._feature_names = feature_names
        self._log_info(f"Loaded {len(features)} samples with {len(feature_names)} features")

        return features, labels, feature_names

    def _generate_labels(
        self,
        features: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        """
        Generate labels for training (override in subclass for specific logic).
        """
        # Simple example: 1 if next return > 0, else 0
        returns = np.diff(features[:, 0]) if features.shape[1] > 0 else np.array([])
        labels = (returns > 0).astype(np.float64)
        # Pad to match features length
        labels = np.append(labels, 0) if len(labels) < len(features) else labels[: len(features)]
        return labels

    @abstractmethod
    def prepare_data(
        self,
        data: Any,  # pl.DataFrame when polars is available
        target_col: str = "target",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
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
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
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
    def predict(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float32]:
        """
        Make predictions using the trained model.

        Parameters
        ----------
        model : Any
            The trained model.
        X : np.ndarray
            Features to predict on (can be float64 for compatibility).
        **kwargs : Any
            Additional prediction parameters.

        Returns
        -------
        npt.NDArray[np.float32]
            Model predictions (always float32 for inference compatibility).

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

    @abstractmethod
    def _get_model_params(self) -> dict[str, Any]:
        """
        Get model-specific default parameters.

        Returns
        -------
        dict[str, Any]
            Default model parameters.

        """
        ...

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
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
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
        n_trials = getattr(getattr(self._config, "optuna_config", None), "n_trials", 100)
        study.optimize(objective, n_trials=n_trials)

        self._optuna_study = study
        best_params = study.best_params
        self._log_info(f"Best parameters found: {best_params}")

        return dict(best_params)

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
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
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
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float32] | npt.NDArray[np.float64],
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
            return float(np.mean(y_true == y_pred))
        else:
            # Use negative MSE for regression (Optuna maximizes by default)
            return float(-np.mean((y_true - y_pred) ** 2))

    def _cross_validate(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
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
        n_folds: int = getattr(self._config, "cv_folds", 5)
        cv_strategy = getattr(self._config, "cv_strategy", "time_series")
        n_samples: int = len(X)

        # Guard against too few samples for requested folds
        if n_folds > n_samples:
            self._log_warning(
                f"cv_folds ({n_folds}) > samples ({n_samples}); reducing folds to {n_samples}",
            )
            n_folds = n_samples
        if n_folds < 2 or n_samples < 2:
            self._log_warning("Insufficient samples for cross-validation; skipping CV")
            return []

        self._log_info(f"Starting {n_folds}-fold {cv_strategy} cross-validation")

        if cv_strategy == "time_series":
            return self._time_series_cv(X, y, n_folds, **kwargs)
        else:
            return self._standard_cv(X, y, n_folds, **kwargs)

    def _time_series_cv(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
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
        n_samples: int = len(X)
        fold_size: int = int(n_samples // (n_folds + 1))
        results: list[dict[str, float]] = []

        if fold_size < 1:
            self._log_warning(
                f"Time-series CV fold_size={fold_size} too small for {n_folds} folds; skipping CV",
            )
            return results

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
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        n_folds: int,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Perform standard k-fold cross-validation.

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
            if not HAS_SKLEARN:
                raise ImportError
            from sklearn.model_selection import KFold

            # Ensure valid n_splits
            n_splits: int = int(min(max(2, n_folds), len(X)))
            if n_splits < 2:
                self._log_warning("Insufficient samples for KFold; skipping CV")
                return []
            kf: KFold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            results: list[dict[str, float]] = []

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

        mlflow_config = getattr(self._config, "mlflow_config", None)
        if mlflow_config is not None:
            if mlflow_config.tracking_uri:
                mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            if mlflow_config.experiment_name:
                mlflow.set_experiment(mlflow_config.experiment_name)

        run = mlflow.start_run(
            run_name=getattr(mlflow_config, "run_name", None) if mlflow_config else None,
        )
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
            if isinstance(value, int | float):
                mlflow.log_metric(key, value)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], int | float):
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
            if isinstance(value, str | int | float | bool):
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
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
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
        # For classification metrics, we need labels not probabilities
        if self._is_classification_problem(y):
            predictions = self.predict(model, X, return_labels=True)
            return self._calculate_classification_metrics(y, predictions)
        else:
            # For regression, we get raw values
            predictions = self.predict(model, X)
            return self._calculate_regression_metrics(y, predictions)

    def calculate_trading_metrics(
        self,
        returns: npt.NDArray[np.float64],
        predictions: npt.NDArray[np.float32] | npt.NDArray[np.float64],
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
        Save the trained model to disk in production format.

        This method should be overridden by specific trainers to save
        in their native format (XGBoost JSON, LightGBM TXT, etc).

        Parameters
        ----------
        path : str | Path
            Path where to save the model.

        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Use registry for saving models
        from ml.registry import ModelRegistry
        from ml.training.export import save_model_with_metadata

        # Determine the registry path from config or use a default
        registry_path = getattr(self._config, "registry_path", Path("./model_registry"))
        registry = ModelRegistry(registry_path)

        # Create model manifest
        manifest = self._create_model_manifest(save_path)

        # Save model artifact first
        artifact_path = save_model_with_metadata(
            model=self._model,
            path=save_path,
            training_metadata=self._training_metrics,
        )

        # Register with registry
        model_id = registry.register_model(
            model_path=artifact_path,
            manifest=manifest,
            auto_deploy=getattr(self._config, "auto_deploy", False),
        )

        self._log_info(f"Model registered with ID: {model_id} at {artifact_path}")

    def _create_model_manifest(self, save_path: Path) -> ModelManifest:
        """
        Create a model manifest from training metadata.

        Parameters
        ----------
        save_path : Path
            Path where the model will be saved

        Returns
        -------
        ModelManifest
            Complete model manifest for registry registration

        """
        import time

        from ml.registry import DataRequirements
        from ml.registry import ModelManifest
        from ml.registry import ModelRole
        from ml.registry.feature_registry import compute_schema_hash

        # Determine model role based on config
        role = getattr(self._config, "model_role", ModelRole.INFERENCE)
        if isinstance(role, str):
            role = ModelRole(role)

        # Determine data requirements based on config
        data_requirements = getattr(self._config, "data_requirements", DataRequirements.L1_ONLY)
        if isinstance(data_requirements, str):
            data_requirements = DataRequirements(data_requirements)

        # Build feature schema from feature names
        feature_schema = {}
        if self._feature_names:
            # Assume float32 for all features unless specified otherwise
            feature_dtypes = getattr(
                self._config,
                "feature_dtypes",
                ["float32"] * len(self._feature_names),
            )
            feature_schema = dict(zip(self._feature_names, feature_dtypes))

        # Compute feature schema hash
        feature_schema_hash = ""
        if feature_schema:
            pipeline_signature = getattr(self._config, "pipeline_signature", "")
            feature_schema_hash = compute_schema_hash(
                list(feature_schema.keys()),
                list(feature_schema.values()),
                pipeline_signature,
            )

        # Extract performance metrics
        performance_metrics = {}
        if self._training_metrics:
            # Filter out non-numeric values
            for key, value in self._training_metrics.items():
                if isinstance(value, (int, float)):
                    performance_metrics[key] = float(value)

        # Determine if model is serveable (ONNX format for hot path)
        serveable = save_path.suffix.lower() == ".onnx" or getattr(
            self._config,
            "export_onnx",
            False,
        )
        artifact_format = "onnx" if serveable else "native"

        return ModelManifest(
            model_id="",  # Will be generated by registry
            role=role,
            data_requirements=data_requirements,
            architecture=self.__class__.__name__.replace(
                "Trainer",
                "",
            ),  # e.g., "XGBoostTrainer" -> "XGBoost"
            feature_schema=feature_schema,
            feature_schema_hash=feature_schema_hash,
            parent_id=getattr(self._config, "parent_model_id", None),
            training_config=self._config_to_dict(),
            performance_metrics=performance_metrics,
            deployment_constraints={
                "max_inference_latency_ms": getattr(self._config, "max_inference_latency_ms", 50.0),
                "memory_limit_mb": getattr(self._config, "memory_limit_mb", 1024.0),
            },
            version=getattr(self._config, "model_version", "1.0.0"),
            created_at=time.time(),
            last_modified=time.time(),
            serveable=serveable,
            artifact_format=artifact_format,
            feature_set_id=getattr(self._config, "feature_set_id", None),
            pipeline_signature=getattr(self._config, "pipeline_signature", None),
            pipeline_version=getattr(self._config, "pipeline_version", None),
            decision_policy=getattr(self._config, "decision_policy", None),
            decision_config=getattr(self._config, "decision_config", {}),
        )

    def load_model(self, path: str | Path) -> None:
        """
        Load a trained model from disk using ProductionModelLoader.

        Parameters
        ----------
        path : str | Path
            Path to the saved model.

        """
        # Use registry for loading models by ID or path
        from ml.registry import ModelRegistry

        # Determine the registry path from config or use a default
        registry_path = getattr(self._config, "registry_path", Path("./model_registry"))
        registry = ModelRegistry(registry_path)

        # Check if path is a model ID or file path
        path_str = str(path)
        if "/" not in path_str and "." not in path_str:
            # Looks like a model ID
            model_info = registry.get_model(path_str)
            if model_info is None:
                raise ValueError(f"Model ID not found in registry: {path_str}")

            # Load model from registry
            model = registry.load_model(path_str)
            if model is None:
                raise RuntimeError(f"Failed to load model {path_str} from registry")

            self._model = model
            self._feature_names = list(model_info.manifest.feature_schema.keys())
            self._training_metrics = model_info.manifest.performance_metrics
            self._is_fitted = True

            self._log_info(f"Model loaded from registry: {path_str}")
        else:
            # Fallback to file path loading for backward compatibility
            load_path = Path(path)
            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {load_path}")

            # Use ProductionModelLoader with supported formats (no pickle)
            from ml.actors.base import ProductionModelLoader

            loader = ProductionModelLoader()
            model, metadata = loader.load_model(str(load_path))

            self._model = model
            self._feature_names = metadata.get("feature_names", [])
            self._training_metrics = metadata.get("training_metrics", {})
            self._is_fitted = True

            self._log_info(f"Model loaded from file: {load_path}")

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

    def _is_classification_problem(
        self,
        y: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> bool:
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
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Calculate classification metrics.
        """
        try:
            if not HAS_SKLEARN:
                raise ImportError
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
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float32] | npt.NDArray[np.float64],
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
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
        }
