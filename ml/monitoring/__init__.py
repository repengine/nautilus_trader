"""
ML monitoring infrastructure for Nautilus Trader.

This package provides comprehensive monitoring and observability capabilities for ML
components, including metrics collection, health monitoring, drift detection, and
real-time performance tracking with Prometheus integration.

Key Features
------------
- **Metrics Collection**: Thread-safe collection of ML metrics via specialized collectors
- **Performance Monitoring**: Real-time tracking of model performance and degradation
- **Resource Monitoring**: System resource utilization and bottleneck detection
- **Data Quality Monitoring**: Data ingestion, validation, and quality metrics
- **Feature Engineering Monitoring**: Feature computation performance and drift detection
- **Model Lifecycle Tracking**: Model deployment, training, and version management
- **Drift Detection**: Statistical drift monitoring for features and predictions
- **Grafana Integration**: Dashboard factory and API client for visualization
- **Real-time Dashboard**: Terminal-based monitoring interface

Architecture Patterns
---------------------
This package follows the Universal ML Architecture Patterns:

**Pattern 5 - Centralized Metrics Bootstrap**:
All metrics use ml.common.metrics_bootstrap instead of direct prometheus_client imports.
This prevents metric registry conflicts and ensures consistent naming.

**Protocol-First Design**:
Components use typing.Protocol for duck typing support and clean contracts.

**Progressive Fallback**:
Graceful degradation when Prometheus is unavailable via HAS_PROMETHEUS checks.

**Thread Safety**:
All collectors use threading.RLock for safe concurrent access.

**Hot/Cold Path Separation**:
Hot path operations avoid blocking I/O and heavy computations.

Usage Examples
--------------

Basic ML Metrics Collection:
    >>> from ml.monitoring import MLMetricsCollector, MonitoringConfig
    >>> config = MonitoringConfig(enabled=True, metrics_port=8080)
    >>> collector = MLMetricsCollector(config)
    >>>
    >>> # Record a prediction
    >>> collector.record_prediction(
    ...     model="lgb_v1",
    ...     instrument="EUR/USD",
    ...     prediction_class="BUY",
    ...     latency_seconds=0.002,
    ...     confidence=0.85
    ... )

Centralized Registry Usage:
    >>> from ml.monitoring import MLMetricsRegistry
    >>> registry = MLMetricsRegistry(config)
    >>>
    >>> with registry:
    ...     # All collectors and server are automatically started
    ...     model_collector = registry.get_collector("model")
    ...     model_collector.record_model_deployment(
    ...         model="transformer_v2",
    ...         version="2.1.0",
    ...         instrument="BTC/USD"
    ...     )

Performance Monitoring:
    >>> from ml.monitoring.collectors import PerformanceDegradationMonitor
    >>> monitor = PerformanceDegradationMonitor(config)
    >>>
    >>> # Track model accuracy over time
    >>> monitor.record_model_performance(
    ...     model="ensemble_v1",
    ...     accuracy=0.78,
    ...     window="1h",
    ...     confidence_scores=[0.6, 0.8, 0.9, 0.7]
    ... )

Metrics Server:
    >>> from ml.monitoring import MetricsServer
    >>> server = MetricsServer(config)
    >>>
    >>> with server:
    ...     # Prometheus metrics available at http://localhost:8080/metrics
    ...     # Health check at http://localhost:8080/health
    ...     pass

Resource Monitoring:
    >>> from ml.monitoring.collectors import ResourceUtilizationCollector
    >>> resources = ResourceUtilizationCollector(config)
    >>>
    >>> # Start background monitoring
    >>> resources.start_monitoring()
    >>>
    >>> # Record specific resource usage
    >>> resources.record_model_memory_usage("bert_large", 2048000000)  # 2GB
    >>> resources.stop_monitoring()

Data Quality Monitoring:
    >>> from ml.monitoring.collectors import DataQualityCollector
    >>> data_monitor = DataQualityCollector(config)
    >>>
    >>> # Monitor data loading
    >>> with data_monitor.time_data_load("EUR/USD", "bars") as timer:
    ...     # Load data here
    ...     timer.set_load_result(rows=10000, cache_hit=True)

Feature Engineering Monitoring:
    >>> from ml.monitoring.collectors import FeatureEngineeringCollector
    >>> feature_monitor = FeatureEngineeringCollector(config)
    >>>
    >>> # Monitor feature computation
    >>> with feature_monitor.time_feature_computation("GBP/USD", "technical") as timer:
    ...     # Compute features here
    ...     timer.set_computation_result(features_computed=26, cache_hit=False)

Grafana Integration:
    >>> from ml.monitoring import GrafanaClient, GrafanaDashboardFactory
    >>>
    >>> # Create dashboards programmatically
    >>> factory = GrafanaDashboardFactory()
    >>> dashboard = factory.create_base_dashboard(
    ...     title="ML Performance Monitor",
    ...     uid="ml-perf-monitor",
    ...     tags=["ml-monitoring", "performance"]
    ... )
    >>>
    >>> # Deploy to Grafana
    >>> with GrafanaClient("http://localhost:3000", api_token="...") as client:
    ...     result = client.create_dashboard({"dashboard": dashboard})

Configuration
-------------
All monitoring components are configured via MonitoringConfig:

    >>> config = MonitoringConfig(
    ...     enabled=True,                    # Enable monitoring
    ...     metrics_port=8080,              # Prometheus server port
    ...     metrics_prefix="nautilus_ml",   # Metric name prefix
    ...     health_check_interval=30.0,     # Health check frequency
    ...     enable_high_cardinality=False,  # Performance vs detail tradeoff
    ...     histogram_buckets=[0.001, 0.005, 0.01, 0.05, 0.1],  # Custom buckets
    ...     enable_gc_metrics=True          # Python GC metrics
    ... )

Thread Safety
-------------
All monitoring components are thread-safe and designed for concurrent access:
- Collectors use threading.RLock for metric operations
- Graceful degradation when Prometheus unavailable
- Background monitoring threads with proper cleanup
- Safe for use in multi-threaded ML pipelines

Integration with Nautilus Trader
--------------------------------
This monitoring package integrates with Nautilus Trader's event system:
- Automatic timestamp handling (ts_event, ts_init in nanoseconds)
- Instrument identification via nautilus_trader.model.identifiers
- Event-driven metric updates for real-time monitoring
- Compatible with Nautilus Trader's actor lifecycle management

See Also
--------
- ml.common.metrics_bootstrap: Centralized metrics initialization
- ml.common.metrics_manager: MetricsManager for advanced metric management
- nautilus_trader.core.data: Core data types and schemas
- nautilus_trader.model.identifiers: Instrument and identifier types

"""

from __future__ import annotations

# Core monitoring components
from ml.monitoring._config import AlertConfig
from ml.monitoring._config import DashboardConfig
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collector import FeatureTimer
from ml.monitoring.collector import MLMetricsCollector
from ml.monitoring.collector import PredictionTimer

# Specialized collectors (from collectors package)
from ml.monitoring.collectors import BaseMetricsCollector
from ml.monitoring.collectors import DataQualityCollector
from ml.monitoring.collectors import FeatureEngineeringCollector
from ml.monitoring.collectors import MLMetricsRegistry
from ml.monitoring.collectors import ModelLifecycleCollector
from ml.monitoring.collectors import PerformanceDegradationMonitor
from ml.monitoring.collectors import ResourceUtilizationCollector

# Grafana integration
from ml.monitoring.dashboard_factory import GrafanaDashboardFactory
from ml.monitoring.dashboard_factory import GrafanaPanelFactory
from ml.monitoring.grafana_client import GrafanaAPIError
from ml.monitoring.grafana_client import GrafanaClient

# Real-time monitoring
from ml.monitoring.realtime_dashboard import DashboardUI
from ml.monitoring.realtime_dashboard import SystemMonitor
from ml.monitoring.server import MetricsServer


__version__ = "1.0.0"

# ruff: noqa: RUF022
__all__ = [
    "AlertConfig",
    "BaseMetricsCollector",
    "DashboardConfig",
    "DashboardUI",
    "DataQualityCollector",
    "FeatureEngineeringCollector",
    "FeatureTimer",
    "GrafanaAPIError",
    "GrafanaClient",
    "GrafanaDashboardFactory",
    "GrafanaPanelFactory",
    "MLMetricsCollector",
    "MLMetricsRegistry",
    "MetricsServer",
    "ModelLifecycleCollector",
    "MonitoringConfig",
    "PerformanceDegradationMonitor",
    "PredictionTimer",
    "ResourceUtilizationCollector",
    "SystemMonitor",
]
