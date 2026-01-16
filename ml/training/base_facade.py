"""
Facade for BaseMLTrainer decomposition.

This module provides the BaseMLTrainerFacade which wires together all 7 decomposed
components from the BaseMLTrainer class while remaining an abstract base class.
The facade delegates to components for all concrete operations while preserving
the 7 abstract methods that subclasses must implement.

Following Universal ML Architecture Pattern 2: Protocol-First Interface Design.

Components:
    TrainingOrchestratorComponent: Main training orchestration
    DataPreparationComponent: Data preparation and splitting
    CrossValidationComponent: CV strategies (time-series, purged)
    HyperparameterComponent: Optuna hyperparameter optimization
    MLflowTrackingComponent: MLflow experiment tracking (deprecated)
    EvaluationComponent: Model evaluation and trading metrics
    PersistenceComponent: Model save/load and ONNX export

"""

from __future__ import annotations

import logging
import os
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml._imports import optuna
from ml.config.base import MLFeatureConfig
from ml.config.base import MLTrainingConfig
from ml.training.common import CrossValidationComponent
from ml.training.common import DataPreparationComponent
from ml.training.common import EvaluationComponent
from ml.training.common import HyperparameterComponent
from ml.training.common import MLflowTrackingComponent
from ml.training.common import PersistenceComponent
from ml.training.common import TrainingOrchestratorComponent


if TYPE_CHECKING:
    from ml.stores.feature_store import FeatureStore


logger = logging.getLogger(__name__)


def use_legacy_trainer() -> bool:
    """
    Check if legacy trainer should be used.

    Returns
    -------
    bool
        True if ML_USE_LEGACY_TRAINER environment variable is set to "1".

    Example
    -------
    >>> os.environ["ML_USE_LEGACY_TRAINER"] = "1"
    >>> use_legacy_trainer()
    True

    """
    return os.getenv("ML_USE_LEGACY_TRAINER", "0") == "1"


class BaseMLTrainerFacade(ABC):
    """
    Facade for BaseMLTrainer that delegates to decomposed components.

    IMPORTANT: This class is ABSTRACT - it cannot be instantiated directly.
    Subclasses must implement all 7 abstract methods.

    This facade wires together all 7 decomposed components:
    - TrainingOrchestratorComponent: Main training orchestration (train method)
    - DataPreparationComponent: Data preparation and splitting
    - CrossValidationComponent: CV strategies (time-series, purged)
    - HyperparameterComponent: Optuna hyperparameter optimization
    - MLflowTrackingComponent: MLflow experiment tracking (deprecated)
    - EvaluationComponent: Model evaluation and trading metrics
    - PersistenceComponent: Model save/load and ONNX export

    The facade preserves the exact public API of the legacy BaseMLTrainer while
    delegating to components for all concrete operations. This enables:
    - Incremental migration from legacy to facade
    - Feature flag control via ML_USE_LEGACY_TRAINER environment variable
    - Better testability through component isolation
    - Clearer separation of concerns

    Parameters
    ----------
    config : MLTrainingConfig
        The configuration for model training.

    Attributes
    ----------
    _config : MLTrainingConfig
        Training configuration.
    _feature_config : MLFeatureConfig
        Feature engineering configuration.
    _model : Any
        The trained model (set after training).
    _feature_names : list[str]
        List of feature names used in training.
    _training_metrics : dict[str, Any]
        Dictionary of training metrics.
    _is_fitted : bool
        Whether the model has been fitted.
    _mlflow_run_id : str | None
        Current MLflow run ID (if MLflow enabled).
    _optuna_study : optuna.Study | None
        Optuna study (if hyperparameter optimization enabled).
    _cv_results : list[dict[str, float]]
        Cross-validation results (if CV enabled).
    _feature_store : FeatureStore | None
        Feature store for training/inference parity.

    Example
    -------
    >>> class MyTrainer(BaseMLTrainerFacade):
    ...     def prepare_data(self, data, target_col="target"):
    ...         # Implementation
    ...         ...
    ...     # ... implement other abstract methods
    ...
    >>> trainer = MyTrainer(config)
    >>> results = trainer.train(training_data)

    """

    def __init__(self, config: MLTrainingConfig) -> None:
        """
        Initialize the ML trainer facade.

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

        # Optional components state
        self._mlflow_run_id: str | None = None
        self._optuna_study: optuna.Study | None = None
        self._cv_results: list[dict[str, float]] = []

        # Initialize FeatureStore if database connection provided
        self._feature_store: FeatureStore | None = None
        if hasattr(config, "db_connection") and config.db_connection:
            from ml.stores.feature_store import FeatureStore

            self._feature_store = FeatureStore(
                connection_string=config.db_connection,
                feature_config=self._feature_config,
                pipeline_spec=getattr(config, "pipeline_spec", None),
            )

        # Initialize components (wire them to self)
        self._orchestrator = TrainingOrchestratorComponent(self)
        self._data_prep = DataPreparationComponent(self)
        self._cv_component = CrossValidationComponent(self)
        self._hyperparameter = HyperparameterComponent(self)
        self._mlflow = MLflowTrackingComponent(self)
        self._evaluation = EvaluationComponent(self)
        self._persistence = PersistenceComponent(self)

    # =========================================================================
    # Abstract Methods - MUST remain abstract, subclasses MUST implement
    # =========================================================================

    @abstractmethod
    def prepare_data(
        self,
        data: Any,
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
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]
            A tuple containing:
            - X: Feature array of shape (n_samples, n_features)
            - y: Target array of shape (n_samples,)
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
        X_train : npt.NDArray[np.float64]
            Training features of shape (n_train, n_features).
        y_train : npt.NDArray[np.float64]
            Training targets of shape (n_train,).
        X_val : npt.NDArray[np.float64]
            Validation features of shape (n_val, n_features).
        y_val : npt.NDArray[np.float64]
            Validation targets of shape (n_val,).
        **kwargs : Any
            Additional training parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "model": The trained model object
            - "metrics": Training metrics dictionary

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
        X : npt.NDArray[np.float64]
            Features to predict on of shape (n_samples, n_features).
        **kwargs : Any
            Additional prediction parameters (e.g., return_labels=True).

        Returns
        -------
        npt.NDArray[np.float32]
            Model predictions of shape (n_samples,). Always float32 for
            inference compatibility.

        """
        ...

    @abstractmethod
    def _create_model(self, params: dict[str, Any]) -> Any:
        """
        Create a model instance with given parameters.

        Parameters
        ----------
        params : dict[str, Any]
            Model parameters (hyperparameters).

        Returns
        -------
        Any
            Model instance (not yet fitted).

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

    @abstractmethod
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Suggest hyperparameters for Optuna trial.

        This method defines the hyperparameter search space for Optuna
        optimization. It should use trial.suggest_* methods to define
        the search space.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.

        Returns
        -------
        dict[str, Any]
            Suggested hyperparameters for this trial.

        Example
        -------
        >>> def _suggest_hyperparameters(self, trial):
        ...     return {
        ...         "learning_rate": trial.suggest_float("lr", 0.01, 0.1),
        ...         "max_depth": trial.suggest_int("max_depth", 3, 10),
        ...     }

        """
        ...

    @abstractmethod
    def _convert_to_onnx(self, model: Any, path: Path) -> None:
        """
        Convert model to ONNX format (model-specific implementation).

        Parameters
        ----------
        model : Any
            Trained model to convert.
        path : Path
            Path to save the ONNX model.

        """
        ...

    # =========================================================================
    # Orchestrator delegation - main train() method
    # =========================================================================

    def train(
        self,
        data: Any,
        validation_data: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train the ML model on the provided data.

        This method orchestrates the complete training pipeline by delegating
        to the TrainingOrchestratorComponent.

        Parameters
        ----------
        data : Any
            The training data containing features and target.
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

        """
        return self._orchestrator.train(data, validation_data, **kwargs)

    # =========================================================================
    # Data preparation delegation
    # =========================================================================

    def prepare_data_with_feature_store(
        self,
        instrument_id: str,
        start: Any,
        end: Any,
        compute_if_missing: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]:
        """
        Prepare training data using FeatureStore for guaranteed parity.

        Delegates to DataPreparationComponent.

        Parameters
        ----------
        instrument_id : str
            Instrument to train on.
        start : Any
            Training period start (datetime).
        end : Any
            Training period end (datetime).
        compute_if_missing : bool, default True
            Whether to compute features if not already stored.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]
            X (features), y (labels), feature_names

        """
        return self._data_prep.prepare_data_with_feature_store(
            instrument_id, start, end, compute_if_missing
        )

    def _generate_labels(
        self,
        features: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        """
        Generate labels for training.

        Delegates to DataPreparationComponent.

        Parameters
        ----------
        features : npt.NDArray[np.float64]
            Feature array.
        timestamps : npt.NDArray[np.int64]
            Timestamps for each sample.

        Returns
        -------
        npt.NDArray[np.float64]
            Labels array.

        """
        return self._data_prep._generate_labels(features, timestamps)

    def _split_data(self, data: Any) -> tuple[Any, Any]:
        """
        Split data into training and validation sets.

        Delegates to DataPreparationComponent.

        Parameters
        ----------
        data : Any
            The data to split.

        Returns
        -------
        tuple[Any, Any]
            Training and validation datasets.

        """
        return self._data_prep._split_data(data)

    # =========================================================================
    # Cross-validation delegation
    # =========================================================================

    def _should_use_cv(self) -> bool:
        """
        Check if cross-validation should be used.

        Delegates to CrossValidationComponent.

        Returns
        -------
        bool
            True if CV should be used.

        """
        return self._cv_component._should_use_cv()

    def _cross_validate(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Perform cross-validation.

        Delegates to CrossValidationComponent.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Features.
        y : npt.NDArray[np.float64]
            Targets.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        list[dict[str, float]]
            CV results for each fold.

        """
        return self._cv_component._cross_validate(X, y, **kwargs)

    def _time_series_cv(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        n_folds: int,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Time series cross-validation with expanding window.

        Delegates to CrossValidationComponent.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Features.
        y : npt.NDArray[np.float64]
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
        return self._cv_component._time_series_cv(X, y, n_folds, **kwargs)

    def _standard_cv(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        n_folds: int,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Deprecated: Use time-series aware CV.

        Delegates to CrossValidationComponent.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Features.
        y : npt.NDArray[np.float64]
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
        return self._cv_component._standard_cv(X, y, n_folds, **kwargs)

    def _purged_cv(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        n_folds: int,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        """
        Purged/embargoed walk-forward cross-validation.

        Delegates to CrossValidationComponent.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Features.
        y : npt.NDArray[np.float64]
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
        return self._cv_component._purged_cv(X, y, n_folds, **kwargs)

    # =========================================================================
    # Hyperparameter optimization delegation
    # =========================================================================

    def _should_use_optuna(self) -> bool:
        """
        Check if Optuna hyperparameter optimization should be used.

        Delegates to HyperparameterComponent.

        Returns
        -------
        bool
            True if Optuna should be used.

        """
        return self._hyperparameter._should_use_optuna()

    def _optimize_hyperparameters(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64],
        y_val: npt.NDArray[np.float64],
        *,
        validation_returns: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Delegates to HyperparameterComponent.

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
        validation_returns : npt.NDArray[np.float64] | None, optional
            Returns for Sharpe ratio calculation.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        dict[str, Any]
            Best hyperparameters found.

        """
        return self._hyperparameter._optimize_hyperparameters(
            X_train, y_train, X_val, y_val,
            validation_returns=validation_returns,
            **kwargs,
        )

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
        X_train : npt.NDArray[np.float64]
            Training features.
        y_train : npt.NDArray[np.float64]
            Training targets.
        X_val : npt.NDArray[np.float64]
            Validation features.
        y_val : npt.NDArray[np.float64]
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

    # =========================================================================
    # MLflow tracking delegation
    # =========================================================================

    def _should_use_mlflow(self) -> bool:
        """
        Check if MLflow should be used.

        Delegates to MLflowTrackingComponent.

        Returns
        -------
        bool
            True if MLflow should be used.

        """
        return self._mlflow._should_use_mlflow()

    def _start_mlflow_run(self) -> None:
        """
        Start MLflow run for experiment tracking.

        Delegates to MLflowTrackingComponent.
        """
        self._mlflow._start_mlflow_run()

    def _track_with_mlflow(self, metrics: dict[str, Any]) -> None:
        """
        Track metrics with MLflow.

        Delegates to MLflowTrackingComponent.

        Parameters
        ----------
        metrics : dict[str, Any]
            Metrics to track.

        """
        self._mlflow._track_with_mlflow(metrics)

    def _end_mlflow_run(self) -> None:
        """
        End MLflow run.

        Delegates to MLflowTrackingComponent.
        """
        self._mlflow._end_mlflow_run()

    def _config_to_dict(self) -> dict[str, Any]:
        """
        Convert config to dictionary for MLflow logging.

        Delegates to MLflowTrackingComponent.

        Returns
        -------
        dict[str, Any]
            Configuration as dictionary.

        """
        return self._mlflow._config_to_dict()

    # =========================================================================
    # Evaluation delegation
    # =========================================================================

    def evaluate(
        self,
        model: Any,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Evaluate model performance using standard ML metrics.

        Delegates to EvaluationComponent.

        Parameters
        ----------
        model : Any
            The trained model to evaluate.
        X : npt.NDArray[np.float64]
            Feature array for evaluation.
        y : npt.NDArray[np.float64]
            Target array for evaluation.

        Returns
        -------
        dict[str, float]
            Dictionary of evaluation metrics.

        """
        return self._evaluation.evaluate(model, X, y)

    def calculate_trading_metrics(
        self,
        returns: npt.NDArray[np.float64],
        predictions: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Calculate trading-specific performance metrics.

        Delegates to EvaluationComponent.

        Parameters
        ----------
        returns : npt.NDArray[np.float64]
            Asset returns for the prediction period.
        predictions : npt.NDArray[np.float32] | npt.NDArray[np.float64]
            Model predictions (signals).

        Returns
        -------
        dict[str, float]
            Dictionary of trading metrics.

        """
        return self._evaluation.calculate_trading_metrics(returns, predictions)

    def _is_classification_problem(
        self,
        y: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> bool:
        """
        Determine if this is a classification problem.

        Delegates to EvaluationComponent.

        Parameters
        ----------
        y : npt.NDArray[np.float32] | npt.NDArray[np.float64]
            Target array.

        Returns
        -------
        bool
            True if classification, False if regression.

        """
        return self._evaluation._is_classification_problem(y)

    def _is_classifier_objective(self) -> bool:
        """
        Check if the config objective indicates classification.

        Delegates to EvaluationComponent.

        Returns
        -------
        bool
            True if classifier objective.

        """
        return self._evaluation._is_classifier_objective()

    def _calculate_classification_metrics(
        self,
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Calculate classification metrics.

        Delegates to EvaluationComponent.

        Parameters
        ----------
        y_true : npt.NDArray[np.float64]
            True labels.
        y_pred : npt.NDArray[np.float32] | npt.NDArray[np.float64]
            Predicted labels.

        Returns
        -------
        dict[str, float]
            Classification metrics.

        """
        return self._evaluation._calculate_classification_metrics(y_true, y_pred)

    def _calculate_regression_metrics(
        self,
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Calculate regression metrics.

        Delegates to EvaluationComponent.

        Parameters
        ----------
        y_true : npt.NDArray[np.float64]
            True values.
        y_pred : npt.NDArray[np.float32] | npt.NDArray[np.float64]
            Predicted values.

        Returns
        -------
        dict[str, float]
            Regression metrics.

        """
        return self._evaluation._calculate_regression_metrics(y_true, y_pred)

    # =========================================================================
    # Persistence delegation
    # =========================================================================

    def save_model(self, path: str | Path) -> None:
        """
        Save the trained model to disk in production format.

        Delegates to PersistenceComponent.

        Parameters
        ----------
        path : str | Path
            Path where to save the model.

        """
        self._persistence.save_model(path)

    def load_model(self, path: str | Path) -> None:
        """
        Load a trained model from disk.

        Delegates to PersistenceComponent.

        Parameters
        ----------
        path : str | Path
            Path to the saved model or model ID.

        """
        self._persistence.load_model(path)

    def export_to_onnx(self, path: str | Path) -> None:
        """
        Export trained model to ONNX format.

        Delegates to PersistenceComponent.

        Parameters
        ----------
        path : str | Path
            Path to save ONNX model.

        """
        self._persistence.export_to_onnx(path)

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Get feature importance from the trained model.

        Delegates to PersistenceComponent.

        Returns
        -------
        dict[str, float] | None
            Feature importance scores or None if not available.

        """
        return self._persistence.get_feature_importance()

    # =========================================================================
    # Logging helpers
    # =========================================================================

    def _log_info(self, message: str, *args: object, **kwargs: Any) -> None:
        """
        Log info message.

        Parameters
        ----------
        message : str
            Message to log.
        *args : object
            Positional arguments for formatting.
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
            Message to log.
        *args : object
            Positional arguments for formatting.
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
            Message to log.
        *args : object
            Positional arguments for formatting.
        **kwargs : Any
            Keyword arguments for logger.

        """
        logger.error(message, *args, **kwargs)


BaseMLTrainer = BaseMLTrainerFacade


__all__ = [
    "BaseMLTrainer",
    "BaseMLTrainerFacade",
    "use_legacy_trainer",
]
