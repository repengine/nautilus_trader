# ML Monitoring Module Context

## Overview
The `ml/monitoring/` module provides comprehensive observability infrastructure for ML trading systems, including Prometheus metrics collection, Grafana dashboard management, and real-time performance monitoring with graceful degradation patterns.

## Core Components

### Metrics Server Infrastructure

**server.py** - Prometheus Metrics HTTP Server
```python
class MetricsServer:
    """Lightweight HTTP server for exposing Prometheus metrics."""
```

**Key Features:**
- **📝 ADDITION:** Threaded HTTP server with concurrent request handling via ThreadingMixIn
- **✨ ENHANCEMENT:** Dual endpoints: /metrics (Prometheus scraping) and /health (service monitoring)
- **📝 ADDITION:** Thread-safe shutdown with configurable timeout (default: 10s)
- **📝 ADDITION:** Context manager support for clean resource management
- **⚠️ CORRECTION:** Port conflict detection and graceful error handling (errno 98)
- **📝 ADDITION:** Connection readiness validation with wait_for_ready()

**HTTP Endpoints:**
- `/metrics`: Prometheus-formatted metrics export
- `/health`: JSON health status for orchestration

### Metrics Collection Framework

**collector.py** - Core ML Metrics Collector
```python
class MLMetricsCollector(BaseMetricsCollector):
    """Thread-safe collector for ML system metrics."""
```

**Core Metrics Tracked:**
- **ml_predictions_total**: Prediction counts with model/instrument/class/status labels
- **ml_prediction_latency_seconds**: Inference latency histograms with model/instrument labels
- **ml_model_confidence**: Current confidence scores as gauges
- **ml_feature_computation_latency_seconds**: Feature processing latency
- **ml_model_errors_total**: Error counts by type (inference/feature/timeout)

**Advanced Features:**
- **✨ ENHANCEMENT:** Context managers for automatic timing (PredictionTimer, FeatureTimer)
- **📝 ADDITION:** Thread-safe recording with RLock protection
- **📝 ADDITION:** Configurable histogram buckets optimized for ML inference (<5ms focus)
- **📝 ADDITION:** Graceful degradation when Prometheus unavailable

### Base Collector Framework

**collectors/base.py** - Abstract Base Collector
```python
class BaseMetricsCollector(ABC):
    """Abstract base class for all specialized metrics collectors."""
```

**Infrastructure Features:**
- **📝 ADDITION:** Thread-safe operations with RLock synchronization
- **📝 ADDITION:** Health check capabilities with validation
- **📝 ADDITION:** Metric value inspection for testing and debugging
- **✨ ENHANCEMENT:** Safe recording patterns with error handling
- **📝 ADDITION:** Automatic metric registry management
- **📝 ADDITION:** Reset capabilities for test scenarios

### Dashboard Management

**dashboard_factory.py** - Grafana Dashboard Factory
```python
class GrafanaDashboardFactory:
    """Factory for creating complete Grafana dashboards."""
```

**Panel Types Supported:**
- **Stat Panels**: Single value displays with thresholds and alerts
- **Time Series Panels**: Historical data visualization with legend configuration  
- **Table Panels**: Tabular data display with transformations
- **Heatmap Panels**: Correlation and pattern visualization
- **Row Panels**: Dashboard section organization

**Key Features:**
- **📝 ADDITION:** Template variables for model/instrument filtering
- **📝 ADDITION:** Alert configuration with customizable thresholds
- **✨ ENHANCEMENT:** Consistent styling and color schemes across panels
- **📝 ADDITION:** Dashboard linking for ML monitoring navigation
- **🔄 UPDATE:** Grafana 10.2.3 compatibility with modern panel configurations

### Grafana API Integration

**grafana_client.py** - Comprehensive Grafana API Client
```python
class GrafanaClient:
    """Client for interacting with Grafana HTTP API."""
```

**API Capabilities:**
- **Dashboard Management**: CRUD operations with UID-based access
- **Folder Operations**: Organization and hierarchy management
- **Data Source Testing**: Connection validation and health checks
- **Annotation Retrieval**: Event correlation and analysis
- **Search Functions**: Tag-based dashboard discovery

**Advanced Features:**
- **✨ ENHANCEMENT:** Retry strategy with exponential backoff (3 retries max)
- **📝 ADDITION:** SSL verification control for development environments
- **📝 ADDITION:** Multiple authentication methods (API token preferred, basic auth fallback)
- **📝 ADDITION:** Comprehensive error handling with status code interpretation
- **📝 ADDITION:** Context manager support for automatic session cleanup

### Specialized Collectors

**collectors/model.py** - Model Lifecycle Monitoring
- Model loading/deployment tracking
- Version comparison and A/B testing metrics
- Model performance degradation detection
- Registry integration for model manifest tracking

**collectors/features.py** - Feature Pipeline Monitoring
- Feature computation latency by type
- Feature quality metrics and drift detection
- Pipeline health and throughput tracking
- Store integration for feature lifecycle

**collectors/data.py** - Data Quality Monitoring
- Data ingestion rates and latency
- Quality checks and validation failures
- Schema evolution tracking
- Missing data detection and alerting

**collectors/performance.py** - System Performance Monitoring
- CPU/Memory utilization by component
- GC metrics and allocation tracking
- Thread pool utilization
- Hot path performance profiling

## Architecture Patterns

### Graceful Degradation Framework
**📝 ADDITION:** All monitoring components implement progressive fallback:
```python
class MonitoringComponent:
    def __init__(self, config: MonitoringConfig):
        self._enabled = config.enabled and HAS_PROMETHEUS
        
    def record_metric(self, operation_name: str, operation_func: Callable[[], None]) -> None:
        if not self._enabled:
            return  # Silent no-op when disabled
        
        self._safe_record(operation_name, operation_func)
```

### Thread-Safe Collection Pattern
**✨ ENHANCEMENT:** Consistent thread safety across all collectors:
```python
def record_prediction(self, model: str, latency: float, confidence: float) -> None:
    with self._lock:  # RLock for nested calls
        if self._ml_predictions_total is not None:
            self._ml_predictions_total.labels(model=model).inc()
```

### Context Manager Timing
**📝 ADDITION:** Automatic timing with exception handling:
```python
with collector.time_prediction("model_v1", "SPY.XNAS") as timer:
    prediction = model.predict(features)
    timer.set_prediction("buy", confidence=0.85)
# Metrics automatically recorded on exit
```

### Centralized Bootstrap Integration
**📝 ADDITION:** All metrics use ml.common.metrics_bootstrap for registry safety:
```python
from ml.common.metrics_bootstrap import get_counter, get_histogram
counter = get_counter("ml_predictions_total", "Total predictions made")
```

## Configuration Management

### MonitoringConfig Structure
**📝 ADDITION:** Type-safe configuration with NautilusConfig inheritance:
```python
class MonitoringConfig(NautilusConfig, kw_only=True, frozen=True):
    enabled: bool = True
    metrics_port: PositiveInt = 8080
    metrics_prefix: str = "nautilus_ml"
    histogram_buckets: list[float] | None = None
    enable_high_cardinality: bool = False
    max_metric_age: PositiveFloat = 300.0
```

**Optimized Histogram Buckets** for ML inference:
- **Hot Path Focus**: 0.1ms to 5ms (primary range)
- **Cold Path Coverage**: 10ms to 1s (training/batch operations)
- **Custom Override**: Configurable via histogram_buckets parameter

### Alert Configuration
**📝 ADDITION:** ML-specific alerting thresholds:
```python
class AlertConfig(NautilusConfig, kw_only=True, frozen=True):
    latency_threshold_ms: PositiveFloat = 10.0  # Hot path SLA
    error_rate_threshold: NonNegativeFloat = 0.05  # 5% error rate
    confidence_drop_threshold: NonNegativeFloat = 0.2  # Model degradation
    alert_cooldown_seconds: PositiveInt = 300  # Prevent alert spam
```

### Dashboard Configuration
**📝 ADDITION:** Real-time monitoring dashboard settings:
```python
class DashboardConfig(NautilusConfig, kw_only=True, frozen=True):
    data_dir: str = "./data/tier1"  # Parquet inspection
    l1_progress_file: str = "tier1_l1_progress.json"  # L1 data progress
    feature_progress_file: str = "tier1_features_progress.json"  # Feature computation
```

## Integration Points

### Prometheus Integration
**✨ ENHANCEMENT:** Full Prometheus ecosystem compatibility:
- **Metrics Export**: Standard /metrics endpoint with appropriate content-type
- **Service Discovery**: Consistent labeling for automatic target discovery
- **Recording Rules**: Pre-defined rules for common ML aggregations
- **Federation**: Support for multi-cluster metric aggregation

### Grafana Dashboard Ecosystem
**📝 ADDITION:** Complete dashboard management lifecycle:
- **Templating**: Model/instrument selection variables
- **Alerting**: Integrated alert rules with notification channels
- **Annotation**: Deployment and event correlation
- **Linking**: Navigation between related ML dashboards

### Store Integration
**✨ ENHANCEMENT:** Native integration with 4-store architecture:
- **MetricStore**: Not implemented (metrics go to Prometheus)
- **ModelStore**: Performance metrics correlation
- **FeatureStore**: Feature computation tracking
- **DataStore**: Data quality and ingestion monitoring

## Performance Monitoring

### Hot Path Metrics (Inference)
**Performance Requirements:**
- **Latency Target**: <5ms P99 for predictions
- **Error Rate**: <1% for production models
- **Confidence Threshold**: >0.7 for trading signals
- **Throughput**: 1000+ predictions/second capacity

**Key Metrics:**
- `ml_prediction_latency_seconds` (histogram with <5ms focus buckets)
- `ml_predictions_total{status="success|error"}` (counter)
- `ml_model_confidence` (gauge, real-time confidence tracking)

### Cold Path Metrics (Training/Batch)
**Performance Requirements:**
- **Feature Computation**: <30s for batch processing
- **Model Training**: Progress tracking and convergence monitoring
- **Data Pipeline**: Throughput and quality validation
- **Registry Operations**: Model deployment and validation tracking

**Key Metrics:**
- `ml_feature_computation_latency_seconds` (histogram with broader buckets)
- `ml_data_quality_checks_total` (counter by check type and result)
- `ml_model_training_progress` (gauge for training progress)

## Real-Time Monitoring

### Metrics Collection Scripts
**📝 ADDITION:** Monitoring utilities and automation:
- **scripts/validate_config.py**: Configuration validation and testing
- **scripts/validate_dashboards.py**: Dashboard JSON validation
- **scripts/import_dashboards.py**: Automated dashboard deployment
- **scripts/export_dashboards.py**: Dashboard backup and version control

### Health Check Integration
**✨ ENHANCEMENT:** Multi-level health monitoring:
- **Component Health**: Individual collector status
- **System Health**: Overall ML system health aggregation
- **External Dependencies**: Prometheus/Grafana connectivity
- **Performance Health**: SLA compliance tracking

### Real-Time Dashboard Features
**📝 ADDITION:** Live monitoring capabilities:
- **Auto-refresh**: 30-second default with configurable intervals
- **Alert Integration**: Visual alert status in dashboard
- **Drill-down**: Navigation from high-level to detailed metrics
- **Time Range Controls**: Flexible time window selection

## Deployment and Operations

### Container Integration
**📝 ADDITION:** Docker-ready monitoring deployment:
- **Health Check Endpoints**: /health for container orchestration
- **Graceful Shutdown**: Signal handling and resource cleanup
- **Configuration Management**: Environment variable overrides
- **Log Integration**: Structured logging for monitoring correlation

### Service Mesh Integration
**✨ ENHANCEMENT:** Kubernetes and service mesh compatibility:
- **Service Discovery**: Prometheus annotation support
- **Network Policies**: Secure metrics endpoint access
- **Resource Limits**: Configurable memory/CPU constraints for metrics collection
- **Horizontal Scaling**: Multi-replica metrics aggregation

### Development Tools
**📝 ADDITION:** Developer-friendly monitoring:
- **Local Dashboard**: Single-node development monitoring
- **Mock Collectors**: Testing without Prometheus dependency
- **Metric Validation**: Test utilities for metric correctness
- **Performance Profiling**: Built-in timing and analysis tools

## Best Practices

### Metric Design Guidelines
- **High Cardinality Control**: Enable only when needed (`enable_high_cardinality=False`)
- **Label Consistency**: Standardized model/instrument/status labels
- **Metric Aging**: Automatic cleanup of old metrics (300s default)
- **Naming Conventions**: `nautilus_ml_` prefix with descriptive suffixes

### Dashboard Design Patterns
- **Layered Monitoring**: Overview → Component → Detail drill-down
- **Alert Integration**: Visual status indicators with alert context
- **Performance Focus**: Hot path metrics prioritized in layouts
- **Time Alignment**: Consistent time ranges across related panels

### Operational Guidelines
- **Error Handling**: Silent degradation, never fail application flow
- **Resource Management**: Bounded memory usage for metric storage
- **Security**: Authentication required for dashboard access
- **Backup**: Dashboard configuration version control and automation

This monitoring module ensures comprehensive observability of ML trading systems while maintaining performance and reliability through thoughtful architectural patterns and graceful degradation strategies.