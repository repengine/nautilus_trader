"""
Core metrics collector for ML system monitoring.

This module provides thread-safe metrics collection for ML components with graceful
degradation when Prometheus is not available.

"""

from __future__ import annotations

import time
import types
from typing import TYPE_CHECKING, Any, Self

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.base import BaseMetricsCollector


if TYPE_CHECKING:
    pass  # pragma: no cover


class MLMetricsCollector(BaseMetricsCollector):
    """
    Thread-safe collector for ML system metrics.

    This collector provides core metrics for monitoring ML component performance,
    with graceful degradation when Prometheus is not available.

    Parameters
    ----------
    config : MonitoringConfig
        Configuration for metrics collection.

    """

    def __init__(self, config: MonitoringConfig) -> None:
        """
        Initialize the metrics collector.

        Parameters
        ----------
        config : MonitoringConfig
            Configuration for metrics collection.

        """
        super().__init__(config)

        # Core metrics
        # Metrics are external objects; type as Any to satisfy strict typing
        self._ml_predictions_total: Any | None = None
        self._ml_prediction_latency_seconds: Any | None = None
        self._ml_model_confidence: Any | None = None
        self._ml_feature_computation_latency_seconds: Any | None = None
        self._ml_model_errors_total: Any | None = None

    def _initialize_metrics(self) -> None:
        """
        Initialize Prometheus metrics.
        """
        if not HAS_PROMETHEUS:
            return

        from ml.common.metrics_bootstrap import get_counter
        from ml.common.metrics_bootstrap import get_gauge
        from ml.common.metrics_bootstrap import get_histogram

        prefix = self._config.metrics_prefix
        buckets = self._config.get_histogram_buckets()

        # Core ML metrics
        self._ml_predictions_total = get_counter(
            f"{prefix}_predictions_total",
            "Total number of ML predictions made",
            ["model", "instrument", "prediction_class", "status"],
        )

        self._ml_prediction_latency_seconds = get_histogram(
            f"{prefix}_prediction_latency_seconds",
            "Time taken for ML model inference",
            ["model", "instrument"],
            buckets=buckets,
        )

        self._ml_model_confidence = get_gauge(
            f"{prefix}_model_confidence",
            "Current ML model confidence score",
            ["model", "instrument"],
        )

        self._ml_feature_computation_latency_seconds = get_histogram(
            f"{prefix}_feature_computation_latency_seconds",
            "Time taken for feature computation",
            ["instrument", "feature_type"],
            buckets=buckets,
        )

        self._ml_model_errors_total = get_counter(
            f"{prefix}_model_errors_total",
            "Total number of ML model errors",
            ["model", "instrument", "error_type"],
        )

        # Register metrics with base class for tracking
        self._register_metric("ml_predictions_total", self._ml_predictions_total)
        self._register_metric("ml_prediction_latency_seconds", self._ml_prediction_latency_seconds)
        self._register_metric("ml_model_confidence", self._ml_model_confidence)
        self._register_metric(
            "ml_feature_computation_latency_seconds",
            self._ml_feature_computation_latency_seconds,
        )
        self._register_metric("ml_model_errors_total", self._ml_model_errors_total)

    def record_prediction(
        self,
        model: str,
        instrument: str,
        prediction_class: str,
        latency_seconds: float,
        confidence: float,
        success: bool = True,
    ) -> None:
        """
        Record a ML prediction event.

        Parameters
        ----------
        model : str
            Model identifier.
        instrument : str
            Instrument identifier.
        prediction_class : str
            Prediction class (e.g., 'buy', 'sell', 'hold').
        latency_seconds : float
            Inference latency in seconds.
        confidence : float
            Model confidence score (0.0-1.0).
        success : bool, default True
            Whether the prediction was successful.

        """
        if not self._enabled:
            return

        with self._lock:
            status = "success" if success else "error"

            # Record prediction count
            if self._ml_predictions_total is not None:
                self._ml_predictions_total.labels(
                    model=model,
                    instrument=instrument,
                    prediction_class=prediction_class,
                    status=status,
                ).inc()

            # Record latency
            if self._ml_prediction_latency_seconds is not None:
                self._ml_prediction_latency_seconds.labels(
                    model=model,
                    instrument=instrument,
                ).observe(latency_seconds)

            # Update confidence gauge
            if self._ml_model_confidence is not None:
                self._ml_model_confidence.labels(
                    model=model,
                    instrument=instrument,
                ).set(confidence)

    def record_feature_computation(
        self,
        instrument: str,
        feature_type: str,
        latency_seconds: float,
    ) -> None:
        """
        Record feature computation metrics.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        feature_type : str
            Type of features computed (e.g., 'technical', 'microstructure').
        latency_seconds : float
            Computation latency in seconds.

        """
        if not self._enabled:
            return

        with self._lock:
            if self._ml_feature_computation_latency_seconds is not None:
                self._ml_feature_computation_latency_seconds.labels(
                    instrument=instrument,
                    feature_type=feature_type,
                ).observe(latency_seconds)

    def record_error(
        self,
        model: str,
        instrument: str,
        error_type: str,
    ) -> None:
        """
        Record ML model error.

        Parameters
        ----------
        model : str
            Model identifier.
        instrument : str
            Instrument identifier.
        error_type : str
            Type of error (e.g., 'inference', 'feature', 'timeout').

        """
        if not self._enabled:
            return

        with self._lock:
            if self._ml_model_errors_total is not None:
                self._ml_model_errors_total.labels(
                    model=model,
                    instrument=instrument,
                    error_type=error_type,
                ).inc()

    def time_prediction(self, model: str, instrument: str) -> PredictionTimer:
        """
        Create a context manager for timing predictions.

        Parameters
        ----------
        model : str
            Model identifier.
        instrument : str
            Instrument identifier.

        Returns
        -------
        PredictionTimer
            Context manager for timing predictions.

        """
        return PredictionTimer(self, model, instrument)

    def time_feature_computation(self, instrument: str, feature_type: str) -> FeatureTimer:
        """
        Create a context manager for timing feature computation.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        feature_type : str
            Feature type identifier.

        Returns
        -------
        FeatureTimer
            Context manager for timing feature computation.

        """
        return FeatureTimer(self, instrument, feature_type)

    @property
    def enabled(self) -> bool:
        """
        Check if metrics collection is enabled.

        Returns
        -------
        bool
            True if metrics collection is enabled.

        """
        return self._enabled

    @property
    def config(self) -> MonitoringConfig:
        """
        Get the monitoring configuration.

        Returns
        -------
        MonitoringConfig
            The monitoring configuration.

        """
        return self._config


class PredictionTimer:
    """
    Context manager for timing ML predictions.

    Parameters
    ----------
    collector : MLMetricsCollector
        The metrics collector instance.
    model : str
        Model identifier.
    instrument : str
        Instrument identifier.

    """

    def __init__(self, collector: MLMetricsCollector, model: str, instrument: str) -> None:
        """
        Initialize the prediction timer.

        Parameters
        ----------
        collector : MLMetricsCollector
            The metrics collector instance.
        model : str
            Model identifier.
        instrument : str
            Instrument identifier.

        """
        self._collector = collector
        self._model = model
        self._instrument = instrument
        self._start_time: float = 0.0
        self._prediction_class: str = ""
        self._confidence: float = 0.0
        self._success: bool = True

    def __enter__(self) -> Self:
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        latency = time.perf_counter() - self._start_time
        self._success = exc_type is None

        self._collector.record_prediction(
            model=self._model,
            instrument=self._instrument,
            prediction_class=self._prediction_class,
            latency_seconds=latency,
            confidence=self._confidence,
            success=self._success,
        )

    def set_prediction(self, prediction_class: str, confidence: float) -> None:
        """
        Set prediction details.

        Parameters
        ----------
        prediction_class : str
            Prediction class (e.g., 'buy', 'sell', 'hold').
        confidence : float
            Model confidence score (0.0-1.0).

        """
        self._prediction_class = prediction_class
        self._confidence = confidence


class FeatureTimer:
    """
    Context manager for timing feature computation.

    Parameters
    ----------
    collector : MLMetricsCollector
        The metrics collector instance.
    instrument : str
        Instrument identifier.
    feature_type : str
        Feature type identifier.

    """

    def __init__(self, collector: MLMetricsCollector, instrument: str, feature_type: str) -> None:
        """
        Initialize the feature timer.

        Parameters
        ----------
        collector : MLMetricsCollector
            The metrics collector instance.
        instrument : str
            Instrument identifier.
        feature_type : str
            Feature type identifier.

        """
        self._collector = collector
        self._instrument = instrument
        self._feature_type = feature_type
        self._start_time: float = 0.0

    def __enter__(self) -> Self:
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        latency = time.perf_counter() - self._start_time
        self._collector.record_feature_computation(
            instrument=self._instrument,
            feature_type=self._feature_type,
            latency_seconds=latency,
        )
