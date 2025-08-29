"""
Configuration classes for ML monitoring infrastructure.

This module provides type-safe configuration for monitoring components using msgspec,
following Nautilus Trader's architectural patterns.

"""

from __future__ import annotations

from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


class MonitoringConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for ML monitoring infrastructure.

    This configuration controls metrics collection, Prometheus server settings,
    and monitoring behavior for ML components.

    Parameters
    ----------
    enabled : bool, default True
        Whether monitoring is enabled. If False, no metrics are collected.
    metrics_port : PositiveInt, default 8080
        Port number for the Prometheus metrics server.
    health_check_interval : PositiveFloat, default 30.0
        Interval in seconds between health checks.
    export_interval : PositiveFloat, default 5.0
        Interval in seconds between metrics exports.
    metrics_prefix : str, default "nautilus_ml"
        Prefix for all metric names.
    enable_high_cardinality : bool, default False
        Whether to enable high cardinality metrics (may impact performance).
    max_metric_age : PositiveFloat, default 300.0
        Maximum age in seconds for metrics before they are discarded.
    histogram_buckets : list[float] | None, optional
        Custom histogram buckets for latency metrics. If None, uses default buckets.
    enable_gc_metrics : bool, default True
        Whether to enable garbage collection metrics.
    server_timeout : PositiveFloat, default 10.0
        Timeout in seconds for metrics server operations.
    max_concurrent_requests : PositiveInt, default 100
        Maximum number of concurrent requests to metrics server.

    """

    enabled: bool = True
    metrics_port: PositiveInt = 8080
    health_check_interval: PositiveFloat = 30.0
    export_interval: PositiveFloat = 5.0
    metrics_prefix: str = "nautilus_ml"
    enable_high_cardinality: bool = False
    max_metric_age: PositiveFloat = 300.0
    histogram_buckets: list[float] | None = None
    enable_gc_metrics: bool = True
    server_timeout: PositiveFloat = 10.0
    max_concurrent_requests: PositiveInt = 100

    def get_default_buckets(self) -> list[float]:
        """
        Get default histogram buckets for latency metrics.

        Returns
        -------
        list[float]
            Default buckets optimized for ML inference latency (in seconds).

        """
        return [
            0.0001,  # 0.1ms
            0.0002,  # 0.2ms
            0.0005,  # 0.5ms
            0.001,  # 1ms
            0.002,  # 2ms
            0.005,  # 5ms
            0.01,  # 10ms
            0.025,  # 25ms
            0.05,  # 50ms
            0.1,  # 100ms
            0.25,  # 250ms
            0.5,  # 500ms
            1.0,  # 1s
        ]

    def get_histogram_buckets(self) -> list[float]:
        """
        Get histogram buckets, using custom if provided or default otherwise.

        Returns
        -------
        list[float]
            Histogram buckets for metrics.

        """
        return (
            self.histogram_buckets
            if self.histogram_buckets is not None
            else self.get_default_buckets()
        )


class AlertConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for ML monitoring alerts.

    Parameters
    ----------
    enabled : bool, default False
        Whether alerting is enabled.
    latency_threshold_ms : PositiveFloat, default 10.0
        Latency threshold in milliseconds for alerts.
    error_rate_threshold : NonNegativeFloat, default 0.05
        Error rate threshold (0.0-1.0) for alerts.
    confidence_drop_threshold : NonNegativeFloat, default 0.2
        Confidence drop threshold for alerts.
    alert_cooldown_seconds : PositiveInt, default 300
        Minimum time in seconds between alerts of the same type.

    """

    enabled: bool = False
    latency_threshold_ms: PositiveFloat = 10.0
    error_rate_threshold: NonNegativeFloat = 0.05
    confidence_drop_threshold: NonNegativeFloat = 0.2
    alert_cooldown_seconds: PositiveInt = 300


class DashboardConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for the real-time monitoring dashboard.

    Parameters
    ----------
    data_dir : str
        Base directory for data inspection (e.g., tier1 parquet files).
    l1_progress_file : str
        JSON file path for L1 ingestion progress.
    feature_progress_file : str
        JSON file path for feature computation progress.
    """

    data_dir: str = "./data/tier1"
    l1_progress_file: str = "tier1_l1_progress.json"
    feature_progress_file: str = "tier1_features_progress.json"
