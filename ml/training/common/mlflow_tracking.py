"""
MLflow tracking component for BaseMLTrainer decomposition.

This module provides the MLflowTrackingComponent which encapsulates MLflow
experiment tracking logic from BaseMLTrainer (lines 483-491 and 1071-1123), including:
- MLflow enablement check (_should_use_mlflow)
- MLflow run lifecycle management (_start_mlflow_run, _end_mlflow_run)
- Metrics tracking (_track_with_mlflow)
- Configuration serialization (_config_to_dict)

Following Universal ML Architecture Pattern 2: Protocol-First Interface Design.

Note: MLflow is deprecated in favor of ModelRegistry. This component provides
backward compatibility for existing workflows.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from ml._imports import HAS_MLFLOW


if TYPE_CHECKING:
    from ml.config.base import MLTrainingConfig


logger = logging.getLogger(__name__)


class MLflowTrainerProtocol(Protocol):
    """
    Protocol for trainer interaction with MLflow tracking component.

    Defines the interface that any trainer must implement to work with
    the MLflowTrackingComponent. This follows Protocol-First design
    (Pattern 2) to enable structural typing without inheritance coupling.

    Attributes
    ----------
    _config : MLTrainingConfig
        Training configuration with optional MLflow settings.
    _mlflow_run_id : str | None
        Current MLflow run ID, set after starting a run.

    """

    _config: MLTrainingConfig
    _mlflow_run_id: str | None

    def _log_info(self, message: str, *args: object, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def _log_warning(self, message: str, *args: object, **kwargs: Any) -> None:
        """Log warning message."""
        ...


class MLflowTrackingComponent:
    """
    Component responsible for MLflow experiment tracking operations.

    This component encapsulates the MLflow tracking logic from BaseMLTrainer
    (lines 483-491 and 1071-1123), implementing:
    - MLflow enablement check based on config and availability
    - Run lifecycle management (start, end)
    - Metrics and parameters tracking
    - Configuration serialization for logging

    The component delegates configuration access and logging to the trainer instance
    through the MLflowTrainerProtocol interface, following Protocol-First design.

    Note: MLflow is deprecated in favor of ModelRegistry. This component provides
    backward compatibility for existing workflows that use MLflow.

    Parameters
    ----------
    trainer : MLflowTrainerProtocol
        The trainer instance that implements the MLflowTrainerProtocol.

    Example
    -------
    >>> from ml.training.common import MLflowTrackingComponent
    >>> # trainer is an instance implementing MLflowTrainerProtocol
    >>> mlflow_component = MLflowTrackingComponent(trainer)
    >>> if mlflow_component._should_use_mlflow():
    ...     mlflow_component._start_mlflow_run()
    ...     mlflow_component._track_with_mlflow({"accuracy": 0.95})
    ...     mlflow_component._end_mlflow_run()

    """

    def __init__(self, trainer: MLflowTrainerProtocol) -> None:
        """
        Initialize the MLflow tracking component with a trainer reference.

        Parameters
        ----------
        trainer : MLflowTrainerProtocol
            The trainer instance for delegation.

        """
        self._trainer = trainer

    def _should_use_mlflow(self) -> bool:
        """
        Check if MLflow experiment tracking should be used.

        MLflow is enabled when:
        1. HAS_MLFLOW flag is True (mlflow is installed and not deprecated)
        2. Config has mlflow_config attribute
        3. mlflow_config is not None

        Returns
        -------
        bool
            True if MLflow tracking should be performed, False otherwise.

        Example
        -------
        >>> mlflow_component = MLflowTrackingComponent(trainer)
        >>> if mlflow_component._should_use_mlflow():
        ...     mlflow_component._start_mlflow_run()

        Notes
        -----
        MLflow is deprecated in favor of ModelRegistry. This check will typically
        return False in current configurations since HAS_MLFLOW is False by default.

        """
        return (
            HAS_MLFLOW
            and hasattr(self._trainer._config, "mlflow_config")
            and self._trainer._config.mlflow_config is not None
        )

    def _start_mlflow_run(self) -> None:
        """
        Start MLflow run for experiment tracking.

        Configures MLflow with tracking URI and experiment name from config,
        then starts a new run. The run ID is stored on the trainer for later use.

        Does nothing if MLflow is not available or not configured.

        Example
        -------
        >>> mlflow_component._start_mlflow_run()
        >>> print(trainer._mlflow_run_id)  # e.g., "abc123..."

        Notes
        -----
        This method:
        1. Sets the tracking URI if configured
        2. Sets the experiment name if configured
        3. Starts a new run with optional run name
        4. Logs all config parameters to MLflow
        5. Stores the run ID on the trainer instance

        """
        if not HAS_MLFLOW:
            return

        # Import mlflow only when needed and available
        try:
            import mlflow
        except ImportError:
            self._trainer._log_warning(
                "MLflow import failed despite HAS_MLFLOW=True; skipping MLflow tracking"
            )
            return

        mlflow_config = getattr(self._trainer._config, "mlflow_config", None)
        if mlflow_config is not None:
            if hasattr(mlflow_config, "tracking_uri") and mlflow_config.tracking_uri:
                mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            if hasattr(mlflow_config, "experiment_name") and mlflow_config.experiment_name:
                mlflow.set_experiment(mlflow_config.experiment_name)

        run_name = (
            getattr(mlflow_config, "run_name", None) if mlflow_config is not None else None
        )
        run = mlflow.start_run(run_name=run_name)
        self._trainer._mlflow_run_id = run.info.run_id

        # Log parameters
        config_dict = self._config_to_dict()
        if config_dict:
            mlflow.log_params(config_dict)

        self._trainer._log_info(f"Started MLflow run: {self._trainer._mlflow_run_id}")

    def _track_with_mlflow(self, metrics: dict[str, Any]) -> None:
        """
        Track metrics with MLflow.

        Logs scalar metrics and lists of scalars to the current MLflow run.
        Handles both single values and lists of values (e.g., per-epoch metrics).

        Parameters
        ----------
        metrics : dict[str, Any]
            Dictionary of metrics to track. Supports:
            - Scalar int/float values: logged directly
            - Lists of int/float: logged as indexed metrics (key_0, key_1, ...)
            - Other types: skipped silently

        Example
        -------
        >>> mlflow_component._track_with_mlflow({
        ...     "accuracy": 0.95,
        ...     "epoch_losses": [0.5, 0.3, 0.2],
        ...     "feature_names": ["a", "b"]  # skipped - not numeric
        ... })

        Notes
        -----
        Does nothing if:
        - MLflow is not available
        - No MLflow run is active (_mlflow_run_id is None)

        """
        if not HAS_MLFLOW or self._trainer._mlflow_run_id is None:
            return

        # Import mlflow only when needed and available
        try:
            import mlflow
        except ImportError:
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
        End the current MLflow run.

        Properly closes the MLflow run and clears the run ID from the trainer.
        Does nothing if no run is active.

        Example
        -------
        >>> mlflow_component._start_mlflow_run()
        >>> # ... training and tracking ...
        >>> mlflow_component._end_mlflow_run()
        >>> print(trainer._mlflow_run_id)  # Still set, not cleared by this method

        Notes
        -----
        The run ID is NOT cleared after ending the run, maintaining
        backward compatibility with the original implementation.

        """
        if not HAS_MLFLOW or self._trainer._mlflow_run_id is None:
            return

        # Import mlflow only when needed and available
        try:
            import mlflow
        except ImportError:
            return

        mlflow.end_run()
        self._trainer._log_info(f"Ended MLflow run: {self._trainer._mlflow_run_id}")

    def _config_to_dict(self) -> dict[str, Any]:
        """
        Convert trainer config to dictionary for MLflow logging.

        Extracts all simple scalar attributes (str, int, float, bool) from
        the trainer's config for parameter logging.

        Returns
        -------
        dict[str, Any]
            Dictionary containing only loggable config values.
            Complex objects (nested configs, lists, etc.) are excluded.

        Example
        -------
        >>> config_dict = mlflow_component._config_to_dict()
        >>> print(config_dict)
        {'learning_rate': 0.01, 'max_depth': 5, 'objective': 'binary:logistic'}

        Notes
        -----
        Only extracts top-level scalar attributes. Nested configurations
        (like feature_config, optuna_config) are not flattened.

        """
        config_dict: dict[str, Any] = {}
        config = self._trainer._config

        # Handle both dataclass-style and msgspec.Struct configs
        if hasattr(config, "__dict__"):
            config_attrs = vars(config)
        else:
            # msgspec.Struct uses __struct_fields__
            config_attrs = {
                field: getattr(config, field, None)
                for field in getattr(config, "__struct_fields__", [])
            }

        for key, value in config_attrs.items():
            if isinstance(value, str | int | float | bool):
                config_dict[key] = value

        return config_dict


__all__ = [
    "MLflowTrackingComponent",
    "MLflowTrainerProtocol",
]
