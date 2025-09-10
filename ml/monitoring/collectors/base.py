"""
Base metrics collector for ML monitoring infrastructure.

This module provides the abstract base class for all specialized metrics collectors,
ensuring consistent patterns and thread-safe operations across the ML monitoring system.

"""

from __future__ import annotations

import threading
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig


class BaseMetricsCollector(ABC):
    """
    Abstract base class for all specialized metrics collectors.

    This class provides a consistent interface and common functionality for all
    metrics collectors, including thread safety, graceful degradation without
    Prometheus, and standardized configuration management.

    Parameters
    ----------
    config : MonitoringConfig
        Configuration for metrics collection behavior.

    """

    def __init__(self, config: MonitoringConfig) -> None:
        """
        Initialize the base metrics collector.

        Parameters
        ----------
        config : MonitoringConfig
            Configuration for metrics collection.

        """
        self._config = config
        self._enabled = config.enabled and HAS_PROMETHEUS
        self._lock = threading.RLock()

        # Track metric instances for registry and cleanup
        self._metrics: dict[str, Any] = {}

        # Initialize collector-specific metrics if enabled
        if self._enabled:
            self._initialize_metrics()

    @abstractmethod
    def _initialize_metrics(self) -> None:
        """
        Initialize Prometheus metrics specific to this collector.

        This method must be implemented by each specialized collector to create the
        appropriate metrics for their domain (e.g., model lifecycle, data quality). All
        metrics should be stored in self._metrics for tracking.

        """

    @property
    def enabled(self) -> bool:
        """
        Check if metrics collection is enabled.

        Returns
        -------
        bool
            True if metrics collection is enabled and Prometheus is available.

        """
        return self._enabled

    @property
    def config(self) -> MonitoringConfig:
        """
        Get the monitoring configuration.

        Returns
        -------
        MonitoringConfig
            The monitoring configuration used by this collector.

        """
        return self._config

    @property
    def metrics(self) -> dict[str, Any]:
        """
        Get all metrics managed by this collector.

        Returns
        -------
        Dict[str, Any]
            Dictionary of metric names to metric instances.

        """
        return self._metrics.copy()

    def get_metric_value(
        self,
        metric_name: str,
        labels: dict[str, str] | None = None,
    ) -> float | None:
        """
        Get the current value of a specific metric.

        This method is primarily used for testing and debugging.

        Parameters
        ----------
        metric_name : str
            Name of the metric to query.
        labels : Dict[str, str], optional
            Labels to filter the metric value (for labeled metrics).

        Returns
        -------
        float | None
            Current metric value, or None if metric doesn't exist or is not enabled.

        """
        if not self._enabled or metric_name not in self._metrics:
            return None

        try:
            with self._lock:
                metric = self._metrics[metric_name]

                # Apply labels if provided
                if labels and hasattr(metric, "labels"):
                    metric = metric.labels(**labels)

                # Handle different metric types
                if hasattr(metric, "_value"):
                    # Gauge metric
                    return float(metric._value._value)
                elif hasattr(metric, "_get_value"):
                    # Counter metric
                    return float(metric._get_value())
                elif hasattr(metric, "get"):
                    # Some metrics might use get() method
                    return float(metric.get())
                else:
                    # For histograms and other complex metrics, return None
                    return None

        except Exception:
            # Graceful degradation - don't fail if metric access fails
            return None

    def reset_metrics(self) -> None:
        """
        Reset all metrics to their initial state.

        This method is primarily used for testing scenarios where metrics need to be
        cleared between test runs.

        """
        if not self._enabled:
            return

        try:
            with self._lock:
                for metric in self._metrics.values():
                    if hasattr(metric, "set"):
                        # Gauge metric - call set(0) directly
                        metric.set(0)
                    elif hasattr(metric, "_value") and hasattr(metric._value, "set"):
                        # Fallback for complex gauge structures
                        metric._value.set(0)
                    # Note: Counter metrics can't be reset to 0, they only increment

        except Exception:
            # Graceful degradation - don't fail if reset fails
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "Failed to reset metrics; continuing",
                exc_info=True,
            )

    def get_metric_count(self) -> int:
        """
        Get the total number of metrics managed by this collector.

        Returns
        -------
        int
            Number of metrics in this collector.

        """
        return len(self._metrics)

    def health_check(self) -> dict[str, Any]:
        """
        Perform a health check on this collector.

        Returns
        -------
        Dict[str, Any]
            Health status information including:
            - enabled: Whether collection is enabled
            - metrics_count: Number of metrics
            - prometheus_available: Whether Prometheus is available
            - config_valid: Whether configuration is valid

        """
        return {
            "enabled": self._enabled,
            "metrics_count": len(self._metrics),
            "prometheus_available": HAS_PROMETHEUS,
            "config_valid": self._validate_config(),
            "collector_type": self.__class__.__name__,
        }

    def _validate_config(self) -> bool:
        """
        Validate the collector configuration.

        Returns
        -------
        bool
            True if configuration is valid, False otherwise.

        """
        try:
            # Basic validation - ensure config has required attributes
            required_attrs = ["enabled", "metrics_prefix"]
            return all(hasattr(self._config, attr) for attr in required_attrs)
        except Exception:
            return False

    def _register_metric(self, name: str, metric: Any) -> None:
        """
        Register a metric with this collector.

        Parameters
        ----------
        name : str
            Metric name for tracking.
        metric : Any
            The Prometheus metric instance.

        """
        if self._enabled:
            with self._lock:
                self._metrics[name] = metric

    def _safe_record(self, operation_name: str, operation_func: Callable[[], None]) -> None:
        """
        Safely execute a metric recording operation with error handling.

        This method provides a consistent pattern for recording metrics while
        ensuring that failures don't impact the main application flow.

        Parameters
        ----------
        operation_name : str
            Name of the operation being recorded (for logging).
        operation_func : callable
            Function that performs the metric recording.

        """
        if not self._enabled:
            return

        try:
            with self._lock:
                operation_func()
        except Exception:
            # In production, we might want to log this error
            # For now, graceful degradation means we silently continue
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "Metric operation '%s' failed; continuing",
                operation_name,
                exc_info=True,
            )

    def __repr__(self) -> str:
        """
        Return string representation of the collector.
        """
        return f"{self.__class__.__name__}(enabled={self._enabled}, metrics_count={len(self._metrics)})"
