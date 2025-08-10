
"""
ML monitoring infrastructure for Nautilus Trader.

This package provides monitoring and observability capabilities for ML components,
including metrics collection, health monitoring, and Prometheus integration.

"""

from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collector import MLMetricsCollector
from ml.monitoring.server import MetricsServer


__version__ = "1.0.0"

__all__ = [
    "MLMetricsCollector",
    "MetricsServer",
    "MonitoringConfig",
]
