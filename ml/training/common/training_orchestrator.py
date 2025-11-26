"""
Training orchestration component for BaseMLTrainer decomposition.

This module provides the TrainingOrchestratorComponent which encapsulates the main
training orchestration logic from BaseMLTrainer (lines 109-282), including:
- Complete training pipeline coordination
- Data preparation and splitting
- Optional hyperparameter optimization (Optuna)
- Optional cross-validation
- Model training and evaluation
- MLflow tracking
- Model persistence and ONNX export
- Logging helpers

Following Universal ML Architecture Pattern 2: Protocol-First Interface Design.

"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies


if TYPE_CHECKING:
    import numpy.typing as npt

    from ml.config.base import MLTrainingConfig


logger = logging.getLogger(__name__)


class TrainerProtocol(Protocol):
    """
    Protocol for trainer interaction with orchestrator.

    Defines the interface that any trainer must implement to work with
    the TrainingOrchestratorComponent. This follows Protocol-First design
    (Pattern 2) to enable structural typing without inheritance coupling.

    Attributes
    ----------
    _config : MLTrainingConfig
        Training configuration.
    _feature_names : list[str]
        List of feature names.
    _training_metrics : dict[str, Any]
        Dictionary to store training metrics.
    _is_fitted : bool
        Whether the model has been fitted.
    _model : Any
        The trained model object.
    _cv_results : list[dict[str, float]]
        Cross-validation results.

    """

    _config: MLTrainingConfig
    _feature_names: list[str]
    _training_metrics: dict[str, Any]
    _is_fitted: bool
    _model: Any
    _cv_results: list[dict[str, float]]

    def prepare_data(
        self,
        data: Any,
        target_col: str,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
        """Prepare features and targets from data."""
        ...

    def _train_model(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the model implementation."""
        ...

    def predict(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float32]:
        """Make predictions using the trained model."""
        ...

    def evaluate(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """Evaluate model performance."""
        ...

    def calculate_trading_metrics(
        self,
        returns: npt.NDArray[np.float64],
        predictions: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """Calculate trading-specific metrics."""
        ...

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        ...

    def export_to_onnx(self, path: Any) -> None:
        """Export model to ONNX format."""
        ...

    def _split_data(self, data: Any) -> tuple[Any, Any]:
        """Split data into train and validation sets."""
        ...

    def _should_use_optuna(self) -> bool:
        """Check if Optuna should be used."""
        ...

    def _optimize_hyperparameters(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        ...

    def _should_use_cv(self) -> bool:
        """Check if cross-validation should be used."""
        ...

    def _cross_validate(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """Perform cross-validation."""
        ...

    def _should_use_mlflow(self) -> bool:
        """Check if MLflow should be used."""
        ...

    def _start_mlflow_run(self) -> None:
        """Start MLflow run."""
        ...

    def _track_with_mlflow(self, metrics: dict[str, Any]) -> None:
        """Track metrics with MLflow."""
        ...

    def _end_mlflow_run(self) -> None:
        """End MLflow run."""
        ...


class TrainingOrchestratorComponent:
    """
    Component responsible for orchestrating the training workflow.

    This component encapsulates the main train() method logic from BaseMLTrainer
    (lines 109-270), coordinating:
    - Data preparation and splitting
    - Optional hyperparameter optimization (Optuna)
    - Optional cross-validation
    - Model training
    - Evaluation and metrics collection
    - MLflow tracking
    - Model persistence (save and ONNX export)

    The component delegates actual operations to the trainer instance through
    the TrainerProtocol interface, following Protocol-First design.

    Parameters
    ----------
    trainer : TrainerProtocol
        The trainer instance that implements the TrainerProtocol.

    Example
    -------
    >>> from ml.training.common import TrainingOrchestratorComponent
    >>> # trainer is an instance implementing TrainerProtocol
    >>> orchestrator = TrainingOrchestratorComponent(trainer)
    >>> results = orchestrator.train(data)
    >>> print(results["metrics"])

    """

    def __init__(self, trainer: TrainerProtocol) -> None:
        """
        Initialize the orchestrator with a trainer reference.

        Parameters
        ----------
        trainer : TrainerProtocol
            The trainer instance for delegation.

        """
        self._trainer = trainer

    def train(
        self,
        data: Any,
        validation_data: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Orchestrate the complete training pipeline.

        This method coordinates all steps of the training process:
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
            Training results including:
            - "model": The trained model object
            - "metrics": Dictionary of training and validation metrics
            - "feature_names": List of feature names used
            - "config": The training configuration
            - "best_params": Best hyperparameters (if Optuna was used)

        Raises
        ------
        ImportError
            If polars is not available.
        ValueError
            If data is empty or invalid.

        Example
        -------
        >>> results = orchestrator.train(train_df, validation_df)
        >>> print(f"Training time: {results['metrics']['training_time']:.2f}s")

        """
        start_time = time.perf_counter()

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        # Handle empty data
        if hasattr(data, "__len__") and len(data) == 0:
            raise ValueError("Cannot train on empty data")

        self._log_info(f"Starting training with {len(data)} samples")

        train_kwargs = dict(kwargs)
        validation_returns = train_kwargs.pop("validation_returns", None)

        # Start MLflow run if configured
        if self._trainer._should_use_mlflow():
            self._trainer._start_mlflow_run()

        # Prepare training data
        if validation_data is None:
            train_data, val_data = self._trainer._split_data(data)
        else:
            train_data, val_data = data, validation_data

        # Prepare features and targets
        X_train, y_train, metadata = self._trainer.prepare_data(
            train_data,
            self._trainer._config.target_column,
        )
        X_val, y_val, _ = self._trainer.prepare_data(
            val_data,
            self._trainer._config.target_column,
        )

        # Store feature names
        self._trainer._feature_names = metadata.get("feature_names", [])

        self._log_info(f"Training features: {X_train.shape}, Validation features: {X_val.shape}")

        # Hyperparameter optimization if configured
        best_params: dict[str, Any] = {}
        if self._trainer._should_use_optuna():
            best_params = self._trainer._optimize_hyperparameters(
                X_train,
                y_train,
                X_val,
                y_val,
                validation_returns=validation_returns,
                **train_kwargs,
            )
            train_kwargs.update(best_params)

        # Cross-validation if configured
        if self._trainer._should_use_cv():
            cv_results = self._trainer._cross_validate(X_train, y_train, **train_kwargs)
            self._trainer._cv_results = cv_results

        # Train the model
        training_results = self._trainer._train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            **train_kwargs,
        )

        self._trainer._model = training_results["model"]
        self._trainer._is_fitted = True

        # Evaluate on validation set
        val_metrics = self._trainer.evaluate(self._trainer._model, X_val, y_val)

        # Calculate trading-specific metrics if returns are available
        trading_metrics: dict[str, float] = {}
        if hasattr(val_data, "columns") and "returns" in val_data.columns:
            predictions = self._trainer.predict(self._trainer._model, X_val)
            trading_metrics = self._trainer.calculate_trading_metrics(
                val_data["returns"].to_numpy(),
                predictions,
            )

        # Combine all metrics
        all_metrics: dict[str, Any] = {
            **training_results.get("metrics", {}),
            **val_metrics,
            **trading_metrics,
        }

        # Add CV results if available
        if self._trainer._cv_results:
            all_metrics["cv_scores"] = self._trainer._cv_results

        # Store training metadata
        training_time = time.perf_counter() - start_time
        self._trainer._training_metrics = {
            "training_time": training_time,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "feature_count": X_train.shape[1] if X_train.ndim > 1 else 1,
            "feature_names": self._trainer._feature_names,
            **all_metrics,
        }

        self._log_info(f"Training completed in {training_time:.2f}s")
        self._log_info(f"Validation metrics: {val_metrics}")

        # Track with MLflow
        if self._trainer._should_use_mlflow():
            self._trainer._track_with_mlflow(self._trainer._training_metrics)

        # Save model if configured
        if self._trainer._config.save_model_path is not None:
            self._trainer.save_model(self._trainer._config.save_model_path)

        # Export to ONNX if configured
        if (
            hasattr(self._trainer._config, "export_onnx")
            and self._trainer._config.export_onnx
            and self._trainer._config.save_model_path
        ):
            onnx_path = Path(self._trainer._config.save_model_path).with_suffix(".onnx")
            self._trainer.export_to_onnx(onnx_path)

        # End MLflow run
        if self._trainer._should_use_mlflow():
            self._trainer._end_mlflow_run()

        return {
            "model": self._trainer._model,
            "metrics": self._trainer._training_metrics,
            "feature_names": self._trainer._feature_names,
            "config": self._trainer._config,
            "best_params": best_params,
        }

    def _log_info(self, message: str, *args: object, **kwargs: Any) -> None:
        """
        Log info message.

        Parameters
        ----------
        message : str
            The message to log.
        *args : object
            Positional arguments for message formatting.
        **kwargs : Any
            Keyword arguments for logger.

        """
        logger.info(message, *args, **kwargs)

    def _log_warning(self, message: str, *args: object, **kwargs: Any) -> None:
        """
        Log warning message.

        Parameters
        ----------
        message : str
            The message to log.
        *args : object
            Positional arguments for message formatting.
        **kwargs : Any
            Keyword arguments for logger.

        """
        logger.warning(message, *args, **kwargs)

    def _log_error(self, message: str, *args: object, **kwargs: Any) -> None:
        """
        Log error message.

        Parameters
        ----------
        message : str
            The message to log.
        *args : object
            Positional arguments for message formatting.
        **kwargs : Any
            Keyword arguments for logger.

        """
        logger.error(message, *args, **kwargs)


__all__ = [
    "TrainerProtocol",
    "TrainingOrchestratorComponent",
]
