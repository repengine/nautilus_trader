"""
Model lifecycle metrics collector for ML monitoring.

This module provides comprehensive tracking of model versioning, deployment, training,
and lifecycle events with Prometheus metrics.

"""

from __future__ import annotations

import time
import types
from typing import Any, Self

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.base import BaseMetricsCollector


class ModelLifecycleCollector(BaseMetricsCollector):
    """
    Collector for model lifecycle and deployment metrics.

    This collector tracks model versioning, deployment events, training metrics,
    model loading performance, and model metadata for comprehensive observability
    of the ML model lifecycle.

    Key Metrics
    -----------
    - Model version and deployment information
    - Training duration and performance
    - Model loading times and sizes
    - Model errors and deployment failures

    Parameters
    ----------
    config : MonitoringConfig
        Configuration for metrics collection.

    """

    def __init__(self, config: MonitoringConfig) -> None:
        """
        Initialize the model lifecycle collector.

        Parameters
        ----------
        config : MonitoringConfig
            Configuration for metrics collection.

        """
        super().__init__(config)

    def _initialize_metrics(self) -> None:
        """
        Initialize Prometheus metrics for model lifecycle tracking.
        """
        if not HAS_PROMETHEUS:
            return

        from ml.common.metrics_bootstrap import get_counter
        from ml.common.metrics_bootstrap import get_gauge
        from ml.common.metrics_bootstrap import get_histogram

        prefix = self._config.metrics_prefix
        buckets = self._config.get_histogram_buckets()

        # Model versioning and deployment info
        self._model_info = get_gauge(
            f"{prefix}_model_info",
            "Model deployment information",
            ["model", "version", "instrument", "deployment_time", "git_commit"],
        )
        self._register_metric("model_info", self._model_info)

        self._model_last_trained_timestamp = get_gauge(
            f"{prefix}_model_last_trained_timestamp",
            "Timestamp when model was last trained",
            ["model", "instrument"],
        )
        self._register_metric("model_last_trained_timestamp", self._model_last_trained_timestamp)

        # Training metrics
        self._model_training_duration_seconds = get_histogram(
            f"{prefix}_model_training_duration_seconds",
            "Time taken for model training phases",
            ["model", "phase"],
            buckets=buckets,
        )
        self._register_metric(
            "model_training_duration_seconds",
            self._model_training_duration_seconds,
        )

        self._model_training_samples_total = get_counter(
            f"{prefix}_model_training_samples_total",
            "Total number of training samples processed",
            ["model", "dataset"],
        )
        self._register_metric("model_training_samples_total", self._model_training_samples_total)

        # Model size and loading metrics
        self._model_size_bytes = get_gauge(
            f"{prefix}_model_size_bytes",
            "Model size in bytes",
            ["model", "format"],
        )
        self._register_metric("model_size_bytes", self._model_size_bytes)

        self._model_load_time_seconds = get_histogram(
            f"{prefix}_model_load_time_seconds",
            "Time taken to load model",
            ["model", "location"],
            buckets=buckets,
        )
        self._register_metric("model_load_time_seconds", self._model_load_time_seconds)

        # Model performance metrics
        self._model_training_score = get_gauge(
            f"{prefix}_model_training_score",
            "Training score/accuracy of the model",
            ["model", "metric_type"],
        )
        self._register_metric("model_training_score", self._model_training_score)

        self._model_validation_score = get_gauge(
            f"{prefix}_model_validation_score",
            "Validation score/accuracy of the model",
            ["model", "metric_type"],
        )
        self._register_metric("model_validation_score", self._model_validation_score)

        # Model deployment and error metrics
        self._model_deployments_total = get_counter(
            f"{prefix}_model_deployments_total",
            "Total number of model deployments",
            ["model", "status"],
        )
        self._register_metric("model_deployments_total", self._model_deployments_total)

        self._model_load_errors_total = get_counter(
            f"{prefix}_model_load_errors_total",
            "Total number of model loading errors",
            ["model", "error_type"],
        )
        self._register_metric("model_load_errors_total", self._model_load_errors_total)

    def record_model_deployment(
        self,
        model: str,
        version: str,
        instrument: str = "",
        git_commit: str = "",
        deployment_time: str | None = None,
    ) -> None:
        """
        Record a model deployment event.

        Parameters
        ----------
        model : str
            Model identifier.
        version : str
            Model version.
        instrument : str, optional
            Instrument this model is deployed for.
        git_commit : str, optional
            Git commit hash for this deployment.
        deployment_time : str, optional
            ISO timestamp of deployment. If None, uses current time.

        """
        if deployment_time is None:
            deployment_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        def _record() -> None:
            if self._model_info is not None:
                self._model_info.labels(
                    model=model,
                    version=version,
                    instrument=instrument,
                    deployment_time=deployment_time,
                    git_commit=git_commit,
                ).set(1)

            if self._model_deployments_total is not None:
                self._model_deployments_total.labels(
                    model=model,
                    status="success",
                ).inc()

        self._safe_record("model_deployment", _record)

    def record_model_training(
        self,
        model: str,
        training_duration: float,
        training_samples: int,
        training_score: float | None = None,
        validation_score: float | None = None,
        phase: str = "training",
        metric_type: str = "accuracy",
    ) -> None:
        """
        Record model training metrics.

        Parameters
        ----------
        model : str
            Model identifier.
        training_duration : float
            Training duration in seconds.
        training_samples : int
            Number of training samples processed.
        training_score : float, optional
            Training score/accuracy.
        validation_score : float, optional
            Validation score/accuracy.
        phase : str, default "training"
            Training phase (e.g., "feature_engineering", "training", "validation").
        metric_type : str, default "accuracy"
            Type of score metric (e.g., "accuracy", "f1", "precision").

        """

        def _record() -> None:
            # Training duration
            if self._model_training_duration_seconds is not None:
                self._model_training_duration_seconds.labels(
                    model=model,
                    phase=phase,
                ).observe(training_duration)

            # Training samples
            if self._model_training_samples_total is not None:
                self._model_training_samples_total.labels(
                    model=model,
                    dataset="train",
                ).inc(training_samples)

            # Training score
            if training_score is not None and self._model_training_score is not None:
                self._model_training_score.labels(
                    model=model,
                    metric_type=metric_type,
                ).set(training_score)

            # Validation score
            if validation_score is not None and self._model_validation_score is not None:
                self._model_validation_score.labels(
                    model=model,
                    metric_type=metric_type,
                ).set(validation_score)

            # Update last trained timestamp
            if self._model_last_trained_timestamp is not None:
                self._model_last_trained_timestamp.labels(
                    model=model,
                    instrument="",  # Will be set by deployment
                ).set(time.time())

        self._safe_record("model_training", _record)

    def record_model_loading(
        self,
        model: str,
        load_duration: float,
        model_size_bytes: int | None = None,
        location: str = "disk",
        format_type: str = "onnx",
        success: bool = True,
        error_type: str | None = None,
    ) -> None:
        """
        Record model loading metrics.

        Parameters
        ----------
        model : str
            Model identifier.
        load_duration : float
            Time taken to load model in seconds.
        model_size_bytes : int, optional
            Size of the model file in bytes.
        location : str, default "disk"
            Location where model was loaded from (e.g., "disk", "memory", "remote").
        format_type : str, default "onnx"
            Format of the model file (e.g., "onnx", "joblib").
        success : bool, default True
            Whether the loading was successful.
        error_type : str, optional
            Type of error if loading failed.

        """

        def _record() -> None:
            # Model loading time
            if self._model_load_time_seconds is not None:
                self._model_load_time_seconds.labels(
                    model=model,
                    location=location,
                ).observe(load_duration)

            # Model size
            if model_size_bytes is not None and self._model_size_bytes is not None:
                self._model_size_bytes.labels(
                    model=model,
                    format=format_type,
                ).set(model_size_bytes)

            # Loading errors
            if not success and error_type and self._model_load_errors_total is not None:
                self._model_load_errors_total.labels(
                    model=model,
                    error_type=error_type,
                ).inc()

            # Deployment status
            if self._model_deployments_total is not None:
                status = "success" if success else "failed"
                self._model_deployments_total.labels(
                    model=model,
                    status=status,
                ).inc()

        self._safe_record("model_loading", _record)

    def update_model_scores(
        self,
        model: str,
        training_score: float | None = None,
        validation_score: float | None = None,
        metric_type: str = "accuracy",
    ) -> None:
        """
        Update model performance scores.

        Parameters
        ----------
        model : str
            Model identifier.
        training_score : float, optional
            Updated training score.
        validation_score : float, optional
            Updated validation score.
        metric_type : str, default "accuracy"
            Type of score metric.

        """

        def _record() -> None:
            if training_score is not None and self._model_training_score is not None:
                self._model_training_score.labels(
                    model=model,
                    metric_type=metric_type,
                ).set(training_score)

            if validation_score is not None and self._model_validation_score is not None:
                self._model_validation_score.labels(
                    model=model,
                    metric_type=metric_type,
                ).set(validation_score)

        self._safe_record("model_score_update", _record)

    def time_training(self, model: str, phase: str = "training") -> ModelTrainingTimer:
        """
        Create a context manager for timing model training.

        Parameters
        ----------
        model : str
            Model identifier.
        phase : str, default "training"
            Training phase being timed.

        Returns
        -------
        ModelTrainingTimer
            Context manager for timing training operations.

        """
        return ModelTrainingTimer(self, model, phase)

    def time_loading(self, model: str, location: str = "disk") -> ModelLoadingTimer:
        """
        Create a context manager for timing model loading.

        Parameters
        ----------
        model : str
            Model identifier.
        location : str, default "disk"
            Location being loaded from.

        Returns
        -------
        ModelLoadingTimer
            Context manager for timing loading operations.

        """
        return ModelLoadingTimer(self, model, location)

    def get_model_stats(self, model: str) -> dict[str, Any]:
        """
        Get comprehensive statistics for a specific model.

        Parameters
        ----------
        model : str
            Model identifier.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing model statistics.

        """
        stats = {
            "model": model,
            "deployment_status": self.get_metric_value("model_info", {"model": model}),
            "last_trained": self.get_metric_value("model_last_trained_timestamp", {"model": model}),
            "training_score": self.get_metric_value("model_training_score", {"model": model}),
            "validation_score": self.get_metric_value("model_validation_score", {"model": model}),
        }

        return {k: v for k, v in stats.items() if v is not None}


class ModelTrainingTimer:
    """
    Context manager for timing model training operations.
    """

    def __init__(self, collector: ModelLifecycleCollector, model: str, phase: str) -> None:
        """
        Initialize the model training timer.

        Parameters
        ----------
        collector : ModelLifecycleCollector
            The collector to record metrics to.
        model : str
            Model identifier.
        phase : str
            Training phase (training, validation, testing).

        """
        self._collector = collector
        self._model = model
        self._phase = phase
        self._start_time: float = 0.0
        self._training_samples: int = 0
        self._training_score: float | None = None
        self._validation_score: float | None = None

    def __enter__(self) -> Self:
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        duration = time.perf_counter() - self._start_time

        self._collector.record_model_training(
            model=self._model,
            training_duration=duration,
            training_samples=self._training_samples,
            training_score=self._training_score,
            validation_score=self._validation_score,
            phase=self._phase,
        )

    def set_training_data(
        self,
        samples: int,
        training_score: float | None = None,
        validation_score: float | None = None,
    ) -> None:
        """
        Set training data for the timer.

        Parameters
        ----------
        samples : int
            Number of training samples.
        training_score : float, optional
            Training score achieved.
        validation_score : float, optional
            Validation score achieved.

        """
        self._training_samples = samples
        self._training_score = training_score
        self._validation_score = validation_score


class ModelLoadingTimer:
    """
    Context manager for timing model loading operations.
    """

    def __init__(self, collector: ModelLifecycleCollector, model: str, location: str) -> None:
        """
        Initialize the model loading timer.

        Parameters
        ----------
        collector : ModelLifecycleCollector
            The collector to record metrics to.
        model : str
            Model identifier.
        location : str
            Location where model is loaded from.

        """
        self._collector = collector
        self._model = model
        self._location = location
        self._start_time: float = 0.0
        self._model_size_bytes: int | None = None
        self._format_type: str = "pickle"
        self._error_type: str | None = None

    def __enter__(self) -> Self:
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        duration = time.perf_counter() - self._start_time
        success = exc_type is None

        if not success and exc_type:
            self._error_type = exc_type.__name__

        self._collector.record_model_loading(
            model=self._model,
            load_duration=duration,
            model_size_bytes=self._model_size_bytes,
            location=self._location,
            format_type=self._format_type,
            success=success,
            error_type=self._error_type,
        )

    def set_model_info(
        self,
        size_bytes: int | None = None,
        format_type: str = "pickle",
    ) -> None:
        """
        Set model information for the timer.

        Parameters
        ----------
        size_bytes : int, optional
            Size of the model in bytes.
        format_type : str, default "pickle"
            Format of the model.

        """
        self._model_size_bytes = size_bytes
        self._format_type = format_type
