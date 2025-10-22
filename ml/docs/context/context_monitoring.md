# ML Monitoring Infrastructure Context

**Last Updated:** 2025-10-19
**Module Size:** ~9,100 lines of code across monitoring infrastructure
**Status:** Production-ready core infrastructure with comprehensive observability capabilities

## Executive Summary

The `ml/monitoring/` module provides production-ready observability and metrics collection for Nautilus Trader's ML components, implementing a comprehensive monitoring architecture with Prometheus metrics, Grafana dashboards, and real-time performance tracking. The infrastructure follows Universal ML Architecture Pattern #5 (Centralized Metrics Bootstrap) and integrates seamlessly with the mandatory 4-store + 4-registry system.

**Critical Architecture Patterns:**
- **Centralized Metrics Bootstrap** - All metrics via `ml.common.metrics_bootstrap` to prevent registry conflicts
- **Progressive Fallback** - Graceful degradation when Prometheus/Grafana unavailable (HAS_PROMETHEUS checks)
- **Thread-Safe Operations** - RLock-protected metric operations across all collectors
- **Hot/Cold Path Separation** - Performance-critical metrics on hot path, observability data off hot path

**Key Implementation Reality:**
- ✅ **6 Specialized Collectors** - Complete implementations (data, features, model, performance, resources, registry)
- ✅ **Grafana Integration** - Full API client + 6 pre-built dashboards + dashboard factory
- ✅ **Production Deployment** - Docker Compose stack with Prometheus, Grafana, Alertmanager, Pushgateway
- ✅ **Real-time Dashboard** - Rich terminal UI for live system monitoring
- ✅ **Alert Management** - 18+ critical alerts with runbook integration
- ✅ **Health Checking** - Comprehensive service health validation

## Module Structure & File Organization

### Core Infrastructure (ml/monitoring/)

**Primary Entry Points:**
```
ml/monitoring/__init__.py                 # Package exports, usage examples (229 lines)
ml/monitoring/_config.py                  # MonitoringConfig, AlertConfig, DashboardConfig (150 lines)
ml/monitoring/collector.py                # MLMetricsCollector, PredictionTimer, FeatureTimer (421 lines)
ml/monitoring/server.py                   # HTTP metrics server with /metrics and /health (329 lines)
```

**Configuration Classes** (`_config.py:17-150`):
- `MonitoringConfig` - Main monitoring configuration with msgspec validation
  - Configurable histogram buckets optimized for ML inference (0.1ms to 1s)
  - Default metrics port: 8080, health check interval: 30s
  - High cardinality control, GC metrics toggle, server timeouts
- `AlertConfig` - Alert thresholds and cooldown management
- `DashboardConfig` - Real-time dashboard data directory configuration

**Core Collector** (`collector.py:24-421`):
- `MLMetricsCollector(BaseMetricsCollector)` - Thread-safe core metrics collection
  - Metrics: predictions_total, prediction_latency, model_confidence, feature_computation_latency, model_errors
  - Context managers: `PredictionTimer`, `FeatureTimer` for automatic latency tracking
  - Integration: Uses `MetricsManager.default()` for centralized metric creation

**Metrics Server** (`server.py:96-329`):
- `ThreadedHTTPServer` - Concurrent request handling with thread safety
- Endpoints:
  - `/metrics` - Prometheus text format metrics exposition
  - `/health` - JSON health check response
- Graceful shutdown with timeout handling
- Port conflict detection and clear error messaging

### Specialized Collectors (ml/monitoring/collectors/)

**Base Collector** (`collectors/base.py:21-293`):
```python
class BaseMetricsCollector(ABC):
    """Abstract base for all specialized collectors."""
    def __init__(self, config: MonitoringConfig):
        self._config = config
        self._enabled = config.enabled and HAS_PROMETHEUS
        self._lock = threading.RLock()  # Thread safety
        self._metrics: dict[str, Any] = {}
        if self._enabled:
            self._initialize_metrics()  # Subclass-specific
```

**Implementation Details:**
- Thread-safe metric access via `threading.RLock` (line 48)
- Graceful degradation when `HAS_PROMETHEUS=False` (line 47)
- Metric registry tracking in `_metrics` dict (line 51)
- Centralized metric value retrieval for testing/debugging (lines 107-150)

**Data Quality Collector** (`collectors/data.py:25-571`):
```python
class DataQualityCollector(BaseMetricsCollector):
    """
    Monitors data loading, quality ratios, staleness, and cache performance.

    Metrics (14 total):
    - data_load_duration_seconds (histogram)
    - data_load_errors_total (counter)
    - data_quality_ratio (gauge)
    - data_staleness_seconds (gauge)
    - data_missing_values_ratio (gauge)
    - data_cache_hit_ratio (gauge)
    - data_cache_misses_total (counter)
    - data_validation_failures_total (counter)
    - data_outliers_detected_total (counter)
    - data_schema_mismatches_total (counter)
    - data_rows_loaded_total (counter)
    - data_bytes_loaded_total (counter)
    - data_source_availability (gauge)
    - data_load_queue_depth (gauge)
    """
```

**Key Methods:**
- `record_data_load()` - Track data loading with duration, rows, cache hits
- `record_data_quality()` - Missing values, outliers, quality scores
- `time_data_load()` - Context manager for automatic timing
- Integration: Uses `MetricsManager.default()` (line 61)

**Feature Engineering Collector** (`collectors/features.py:25-618`):
```python
class FeatureEngineeringCollector(BaseMetricsCollector):
    """
    Monitors feature computation, drift detection, cache efficiency.

    Metrics (16 total):
    - feature_computation_duration_seconds (histogram)
    - feature_computation_errors_total (counter)
    - feature_drift_score (gauge)
    - feature_cache_hit_ratio (gauge)
    - feature_cache_misses_total (counter)
    - features_computed_total (counter)
    - feature_staleness_seconds (gauge)
    - feature_validation_failures_total (counter)
    - feature_nan_ratio (gauge)
    - feature_inf_ratio (gauge)
    - feature_correlation_drift (gauge)
    - feature_distribution_shift (gauge)
    - feature_engineering_queue_depth (gauge)
    - feature_batch_size (histogram)
    - feature_dependency_wait_seconds (histogram)
    - feature_cache_evictions_total (counter)
    """
```

**Drift Detection Features:**
- Statistical drift monitoring with configurable thresholds
- Distribution shift detection (KL divergence, Wasserstein distance)
- Correlation drift tracking for feature relationships
- NaN/Inf ratio monitoring for data quality

**Model Lifecycle Collector** (`collectors/model.py:25-572`):
```python
class ModelLifecycleCollector(BaseMetricsCollector):
    """
    Tracks model deployment, training, inference, and ONNX operations.

    Metrics (18 total):
    - model_deployment_timestamp (gauge)
    - model_version_info (gauge with labels)
    - model_training_duration_seconds (histogram)
    - model_training_samples (histogram)
    - model_inference_batch_size (histogram)
    - model_inference_queue_depth (gauge)
    - model_warm_start_duration_seconds (histogram)
    - model_load_errors_total (counter)
    - model_export_errors_total (counter)
    - model_onnx_conversion_duration_seconds (histogram)
    - model_onnx_inference_duration_seconds (histogram)
    - model_onnx_optimization_gain (gauge)
    - model_retraining_triggers_total (counter)
    - model_rollback_events_total (counter)
    - model_a_b_test_active (gauge)
    - model_size_bytes (gauge)
    - model_parameter_count (gauge)
    - model_flop_count (gauge)
    """
```

**ONNX Support:**
- Conversion duration tracking
- ONNX vs. native inference latency comparison
- Optimization gain measurement

**Performance Degradation Monitor** (`collectors/performance.py:25-548`):
```python
class PerformanceDegradationMonitor(BaseMetricsCollector):
    """
    Monitors accuracy degradation and distribution shifts.

    Metrics (12 total):
    - model_accuracy_rolling (gauge)
    - model_accuracy_window_std (gauge)
    - model_confidence_mean (gauge)
    - model_confidence_std (gauge)
    - prediction_distribution_shift (gauge)
    - feature_distribution_shift (gauge)
    - concept_drift_detected (counter)
    - prediction_error_rate (gauge)
    - false_positive_rate (gauge)
    - false_negative_rate (gauge)
    - model_calibration_error (gauge)
    - retraining_recommended (gauge)
    """
```

**Drift Detection Algorithms:**
- Rolling accuracy windows with standard deviation tracking
- Distribution shift via KL divergence
- Concept drift detection with threshold triggers
- Model calibration error (Expected Calibration Error)

**Resource Utilization Collector** (`collectors/resources.py:25-595`):
```python
class ResourceUtilizationCollector(BaseMetricsCollector):
    """
    Monitors system resources: CPU, memory, GPU, disk, network.

    Metrics (20 total):
    - cpu_usage_percent (gauge)
    - memory_usage_bytes (gauge)
    - memory_usage_percent (gauge)
    - memory_available_bytes (gauge)
    - gpu_utilization_percent (gauge)
    - gpu_memory_usage_bytes (gauge)
    - gpu_memory_total_bytes (gauge)
    - gpu_temperature_celsius (gauge)
    - gpu_power_usage_watts (gauge)
    - disk_usage_bytes (gauge)
    - disk_usage_percent (gauge)
    - disk_io_read_bytes_total (counter)
    - disk_io_write_bytes_total (counter)
    - network_io_sent_bytes_total (counter)
    - network_io_recv_bytes_total (counter)
    - process_cpu_percent (gauge)
    - process_memory_rss_bytes (gauge)
    - thread_count (gauge)
    - file_descriptor_count (gauge)
    - context_switches_total (counter)
    """
```

**Background Monitoring:**
- `start_monitoring()` / `stop_monitoring()` for continuous resource tracking
- Configurable collection interval (default: 5 seconds)
- Thread-safe daemon thread implementation

**Registry Health Collector** (`collectors/registry.py:25-433`):
```python
class RegistryHealthCollector(BaseMetricsCollector):
    """
    Monitors registry operations and schema validation.

    Metrics (13 total):
    - registry_operations_total (counter by registry_type, operation, status)
    - registry_schema_validation_failures_total (counter)
    - registry_health_score (gauge by registry_type)
    - registry_cache_hit_ratio (gauge)
    - registry_query_duration_seconds (histogram)
    - registry_entries_total (gauge)
    - registry_schema_versions_total (gauge)
    - registry_manifest_operations_total (counter)
    - registry_watermark_lag_seconds (gauge)
    - registry_lineage_depth (histogram)
    - registry_artifact_size_bytes (histogram)
    - registry_sync_errors_total (counter)
    - registry_lock_wait_duration_seconds (histogram)
    """
```

**Integration Points:**
- Tracks operations on FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry
- Schema hash validation tracking
- Manifest operation monitoring (read/write/validate)

**Metrics Registry** (`collectors/registry.py:300-433`):
```python
class MLMetricsRegistry:
    """
    Centralized registry for all ML metrics collectors.

    Manages lifecycle of all specialized collectors with unified configuration.
    """
    def __init__(self, config: MonitoringConfig):
        self._collectors = {
            "data": DataQualityCollector(config),
            "features": FeatureEngineeringCollector(config),
            "model": ModelLifecycleCollector(config),
            "performance": PerformanceDegradationMonitor(config),
            "resources": ResourceUtilizationCollector(config),
            "registry": RegistryHealthCollector(config),
        }
```

### Grafana Integration

**API Client** (`grafana_client.py:56-673`):
```python
class GrafanaClient:
    """
    Comprehensive HTTP API client for Grafana dashboard management.

    Features:
    - Session management with retry strategy (3 retries, exponential backoff)
    - Authentication via API token or username/password
    - SSL verification control
    - Connection pooling via requests.Session
    """
```

**Key Methods:**
- `health_check()` - Verify Grafana availability (line 231)
- `create_dashboard()`, `update_dashboard()`, `delete_dashboard()` - CRUD operations (lines 351-422)
- `search_dashboards()` - Query with filters (query, tag, starred) (lines 282-325)
- `get_folders()`, `create_folder()` - Organization management (lines 448-510)
- `get_datasources()`, `test_datasource()` - Data source validation (lines 512-548)
- `get_annotations()` - Fetch alert annotations (lines 550-606)

**Error Handling:**
- Custom `GrafanaAPIError` with status codes and response data (lines 27-54)
- Automatic retry on 429, 500, 502, 503, 504 status codes (line 123)
- Context manager support for automatic session cleanup (lines 615-630)

**Dashboard Factory** (`dashboard_factory.py:20-655`):
```python
class GrafanaPanelFactory:
    """Factory for creating standardized Grafana panel components."""

    # Panel Types:
    create_stat_panel()       # Single value with thresholds (lines 26-103)
    create_timeseries_panel() # Time-based graphs (lines 106-181)
    create_table_panel()      # Tabular data display (lines 184-250)
    create_heatmap_panel()    # Distribution heatmaps (lines 253-328)
    create_row_panel()        # Section organization (lines 331-364)
```

**Dashboard Factory** (`dashboard_factory.py:367-655`):
```python
class GrafanaDashboardFactory:
    """Factory for creating complete Grafana dashboards."""

    def create_base_dashboard(title, uid, tags, ...):
        """
        Creates complete dashboard structure with:
        - Default template variables (datasource, model, interval)
        - Annotation support
        - Dashboard linking
        - Time range controls
        """

    def create_alert_config(alert_name, condition_value, ...):
        """Alert rule configuration with severity levels."""
```

**Default Template Variables:**
- `$datasource` - Prometheus datasource selector
- `$model` - Multi-select model filter from `label_values(ml_predictions_total, model)`
- `$interval` - Time interval selector (1m, 5m, 15m, 30m, 1h)

**Pre-built Dashboards** (`ml/monitoring/grafana/dashboards/`):
1. **ml-overview.json** (15KB) - System overview with key metrics
2. **data-quality.json** (25KB) - Data ingestion and quality monitoring
3. **feature-engineering.json** (24KB) - Feature computation and drift
4. **model-lifecycle.json** (21KB) - Model deployment and training
5. **performance-degradation.json** (24KB) - Accuracy and drift tracking
6. **resource-utilization.json** (23KB) - System resource monitoring

### Real-time Monitoring

**System Monitor** (`realtime_dashboard.py:30-205`):
```python
class SystemMonitor:
    """
    Monitors system metrics and health via file-based progress tracking.

    Metrics Collected:
    - Data: L0/L1/L2 symbol counts, total size, ingestion rate
    - Features: Computed symbols, latency, cache hit rate
    - Models: Loaded models, inference time, predictions/sec, accuracy
    - System: CPU, memory, disk usage, PostgreSQL connections, Redis ops
    """
```

**Implementation Details:**
- Reads progress from JSON files (`l1_progress_file`, `feature_progress_file`)
- Uses `psutil` for system resource monitoring
- Alert checking with configurable thresholds (CPU>80%, memory>85%, etc.)
- File-based metrics for data ingestion progress tracking

**Dashboard UI** (`realtime_dashboard.py:207-400`):
```python
class DashboardUI:
    """
    Rich terminal UI for monitoring dashboard.

    Layout Structure:
    - Header: Title, last update timestamp
    - Body (2 columns):
      - Left: Data panel, Feature panel
      - Right: Model panel, System panel
    - Footer: Alerts with color-coded severity
    """
```

**Rich Terminal Features:**
- Live updating with `rich.live.Live`
- Color-coded panels (blue=data, green=features, yellow=models, red=system)
- Alert severity visualization (INFO, WARNING, CRITICAL)
- Progress indicators for ingestion and computation

### Prometheus Configuration

**Main Configuration** (`prometheus/prometheus.yml:1-128`):
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'nautilus-ml'
    environment: 'production'
```

**Recording Rules** (lines 25-55):
- `ml:inference_latency:avg5m` - Average inference latency by model/instrument
- `ml:cache_effectiveness:ratio` - Cache performance metric
- `ml:model_accuracy:trend1h` - Hourly accuracy trends
- `ml:resource_efficiency:score` - Predictions per CPU unit

**Scrape Targets** (lines 57-116):
- `ml-application` - ML metrics endpoint (port 8000, 5s interval)
- `pushgateway` - Batch job metrics (port 9091)
- `node-exporter` - System metrics (port 9100)
- `prometheus` - Self-monitoring (port 9090)
- `grafana` - Dashboard service (port 3000)
- `alertmanager` - Alert routing (port 9093)

**Metric Relabeling** (lines 119-128):
- Drop high-cardinality `go_memstats_*` metrics
- Drop debug metrics in production (`*_debug_*`)

### Alert Management

**Critical Alerts** (`prometheus/alerts/ml_critical.yml:3-118`):

**Model Performance (8 alerts):**
1. `ModelAccuracyCriticallyLow` - Accuracy < 60% for 5 minutes
2. `InferenceLatencyExceeded` - P99 > 5 seconds for 2 minutes
3. `InferenceTimeoutRateHigh` - >5% timeouts for 2 minutes

**System Resources (2 alerts):**
4. `SystemMemoryExhausted` - Memory >95% for 1 minute
5. `GPUMemoryExhausted` - GPU memory >95% for 1 minute

**Data Pipeline (2 alerts):**
6. `DataPipelineFailure` - Error rate >0.1/sec for 3 minutes
7. `DataStalenessExceeded` - Data >10 minutes old for 5 minutes

**Service Health (2 alerts):**
8. `MLServiceDown` - Service unreachable for 1 minute
9. `ModelLoadFailure` - Model loading failures detected

**Alert Metadata:**
- Labels: `severity`, `team`, `category`
- Annotations: `summary`, `description`, `runbook_url`
- Runbook links: `https://wiki.nautilus.io/runbooks/...`

**Additional Alert Files:**
- `ml_warning.yml` - P1 alerts (investigation needed, 1-4 hour response)
- `ml_info.yml` - P2 alerts (monitoring only, next business day)

### Docker Deployment

**Service Stack** (`docker-compose.yml:5-140`):

**Prometheus** (lines 6-29):
```yaml
image: prom/prometheus:v2.48.0
ports: ["9090:9090"]
volumes:
  - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
  - ./prometheus/alerts/:/etc/prometheus/alerts/:ro
  - prometheus_data:/prometheus
command:
  - '--storage.tsdb.retention.time=30d'
  - '--storage.tsdb.retention.size=10GB'
  - '--web.enable-lifecycle'
healthcheck:
  test: ["CMD", "wget", "--spider", "http://localhost:9090/-/healthy"]
  interval: 30s
```

**Grafana** (lines 31-58):
```yaml
image: grafana/grafana:10.2.3
ports: ["3000:3000"]
environment:
  - GF_SECURITY_ADMIN_USER=${GRAFANA_ADMIN_USER:-admin}
  - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-nautilus123}
  - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
  - GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/ml-overview.json
  - GF_UNIFIED_ALERTING_ENABLED=true
volumes:
  - ./grafana/provisioning:/etc/grafana/provisioning:ro
  - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:3000/api/health || exit 1"]
```

**Alertmanager** (lines 60-79):
```yaml
image: prom/alertmanager:v0.27.0
ports: ["9093:9093"]
command:
  - '--config.file=/etc/alertmanager/alertmanager.yml'
  - '--storage.path=/alertmanager'
```

**Supporting Services:**
- `pushgateway` - Batch job metrics (port 9091)
- `node-exporter` - System metrics with host mounts (`/proc`, `/sys`, `/`)
- `dcgm-exporter` - GPU metrics (optional, commented out)

**Health Checking** (`deployment/check_health.py:27-197`):
```python
def check_service_health(service_name: str, check_func: Callable[[], bool]):
    """Check health of a specific service."""
    try:
        result = check_func()
        return result, "OK" if result else "UNHEALTHY"
    except Exception as e:
        return False, f"ERROR: {e!s}"

# Service Checks:
- check_docker_compose()  # Docker Compose services running
- check_postgres()        # pg_isready via docker-compose exec
- check_redis()           # redis-cli ping
- check_ml_pipeline()     # HTTP /health endpoint (port 8080)
- check_prometheus()      # HTTP /-/healthy
- check_grafana()         # HTTP /api/health
```

### Dashboard Import/Export Scripts

**Import Script** (`scripts/import_dashboards.py:108-482`):
```python
def import_dashboard_file(
    client: GrafanaClient,
    file_path: Path,
    folder_id: int = 0,
    overwrite: bool = True,
    validate: bool = True,
) -> tuple[bool, str]:
    """
    Import single dashboard with validation.

    Validation Checks:
    - Required fields: title, panels
    - UID format (no spaces, slashes, invalid characters)
    - Panels structure (list type)
    - Duplicate panel IDs
    """
```

**Key Features:**
- Automatic folder creation for "ML Monitoring"
- Dashboard validation before import
- Overwrite protection control
- Import manifest generation
- Batch directory import

**Export Script** (`scripts/export_dashboards.py`):
- Search and export by tags
- Backup existing dashboards
- JSON formatting and sanitization

### Integration with ML Infrastructure

**Metrics Bootstrap Integration:**

All collectors use centralized metrics creation to prevent registry conflicts:

```python
# Pattern 1: MetricsManager (preferred)
from ml.common.metrics_manager import MetricsManager

mm = MetricsManager.default()
self._counter = mm.counter(
    f"{prefix}_operations_total",
    "Total operations",
    ["component", "status"],
)

# Pattern 2: Direct bootstrap (legacy)
from ml.common.metrics_bootstrap import get_counter, get_gauge, get_histogram

self._counter = get_counter(
    f"{prefix}_operations_total",
    "Total operations",
    ["component", "status"],
)
```

**Integration Points:**

1. **ml.common.metrics** (`ml/common/metrics.py`):
   - Central definition of 40+ production metrics
   - Helper functions: `record_pipeline_event()`, `update_pipeline_health()`
   - Monitoring collectors extend these with domain-specific metrics

2. **ml.common.metrics_bootstrap** (`ml/common/metrics_bootstrap.py`):
   - Idempotent metric creation functions
   - Thread-safe registry access
   - Graceful degradation when Prometheus unavailable

3. **ml.observability** (`ml/observability/`):
   - Off-hot-path structured data collection
   - Complements real-time Prometheus metrics
   - DataFrame materialization for analysis

4. **ml.deployment** (`ml/deployment/`):
   - Health check integration
   - Docker Compose orchestration
   - Prometheus/Grafana provisioning

**Actor Integration:**

Monitoring automatically integrated via `BaseMLInferenceActor`:

```python
from ml.actors.base import BaseMLInferenceActor

class MyMLActor(BaseMLInferenceActor):
    def __init__(self, config: MLActorConfig):
        super().__init__(config)
        # Automatic integration:
        # - Circuit breaker with metrics
        # - Health monitoring
        # - 4 stores + 4 registries
```

**Circuit Breaker Metrics:**
- `circuit_breaker_state` gauge (0=closed, 0.5=half_open, 1=open)
- `circuit_breaker_trips_total` counter (state transitions)
- Automatic health score updates

## Architecture Patterns & Best Practices

### Pattern 1: Centralized Metrics Bootstrap

**Requirement:** Never import `prometheus_client` directly. Always use `ml.common.metrics_bootstrap` or `ml.common.metrics_manager`.

**Rationale:**
- Prevents duplicate metric registration errors
- Safe for module reloads and testing
- Consistent naming and labeling
- Type-safe even when prometheus_client not installed

**Implementation:**
```python
# ✅ CORRECT - Using MetricsManager
from ml.common.metrics_manager import MetricsManager

mm = MetricsManager.default()
counter = mm.counter(
    "nautilus_ml_predictions_total",
    "Total predictions",
    ["model", "instrument"],
)

# ✅ CORRECT - Using bootstrap functions
from ml.common.metrics_bootstrap import get_counter

counter = get_counter(
    "nautilus_ml_predictions_total",
    "Total predictions",
    ["model", "instrument"],
)

# ❌ WRONG - Direct prometheus_client import
from prometheus_client import Counter  # NEVER DO THIS
```

**Evidence:**
- All collectors import from `ml.common.metrics_bootstrap` (grep results show 8 files)
- `__init__.py:25` explicitly documents this pattern
- `collector.py:65`, `data.py:61`, `features.py:61`, `model.py:61` all follow pattern

### Pattern 2: Thread-Safe Operations

**Implementation in BaseMetricsCollector:**
```python
def __init__(self, config: MonitoringConfig):
    self._lock = threading.RLock()  # Re-entrant lock for nested calls

def record_metric(self, ...):
    with self._lock:
        # Metric operations protected
        self._counter.labels(...).inc()
```

**RLock Benefits:**
- Re-entrant (same thread can acquire multiple times)
- Safe for nested metric operations
- Prevents race conditions in concurrent environments

**Evidence:**
- `base.py:48` initializes `threading.RLock()`
- All metric recording methods use `with self._lock:` context

### Pattern 3: Progressive Fallback

**Graceful Degradation Strategy:**

```python
from ml._imports import HAS_PROMETHEUS

class BaseMetricsCollector:
    def __init__(self, config: MonitoringConfig):
        self._enabled = config.enabled and HAS_PROMETHEUS
        if self._enabled:
            self._initialize_metrics()
        # If not enabled, metric operations become no-ops

    def record_metric(self, ...):
        if not self._enabled:
            return  # Graceful no-op
        # ... actual metric recording
```

**Fallback Chain:**
- PRIMARY: Prometheus + Grafana full stack
- FALLBACK 1: Metrics disabled but application continues
- FALLBACK 2: DummyMetrics for testing (no persistence)

**Evidence:**
- `base.py:47` checks `config.enabled and HAS_PROMETHEUS`
- `server.py:127-150` warns but doesn't crash when Prometheus unavailable
- All `record_*` methods check `self._enabled` before operations

### Pattern 4: Hot/Cold Path Separation

**Hot Path (< 5ms P99):**
- Metric increment/observation operations only
- No blocking I/O, network calls, or file operations
- Pre-allocated metric objects (created at initialization)

**Cold Path:**
- Metric initialization (one-time at startup)
- Dashboard creation/updates
- Alert rule management
- Observability data flushing

**Example:**
```python
# Hot path - called per prediction (fast)
def record_prediction(self, model, instrument, latency):
    if not self._enabled:
        return  # No-op if disabled
    with self._lock:
        self._ml_predictions_total.labels(
            model=model,
            instrument=instrument,
        ).inc()  # Just increment counter

# Cold path - initialization (slow, one-time)
def _initialize_metrics(self):
    from ml.common.metrics_manager import MetricsManager
    mm = MetricsManager.default()
    self._ml_predictions_total = mm.counter(...)  # Create metric object
```

### Pattern 5: Configuration-Driven Behavior

**MonitoringConfig Pattern:**

```python
from ml.monitoring import MonitoringConfig

config = MonitoringConfig(
    enabled=True,                    # Master enable switch
    metrics_port=8080,              # Prometheus server port
    metrics_prefix="nautilus_ml",   # Metric name prefix
    histogram_buckets=[0.001, 0.005, 0.01, 0.05, 0.1],  # Custom buckets
    enable_high_cardinality=False,  # Performance vs detail tradeoff
    health_check_interval=30.0,     # Health check frequency
    export_interval=5.0,            # Metrics export frequency
)
```

**Bucket Optimization** (`_config.py:73-103`):
- Default buckets optimized for ML inference latency
- Range: 0.1ms to 1 second
- Logarithmic distribution for better precision at low latencies

**Environment Variable Support:**
```python
# Grafana client supports env vars
GRAFANA_URL="http://localhost:3000"
GRAFANA_API_TOKEN="your_token_here"

# Health check configurable
ML_PIPELINE_HOST_PORT="8080"  # Override default port
```

## Metrics Catalog

### Core ML Metrics (collector.py)

**Predictions:**
- `nautilus_ml_predictions_total` (counter)
  - Labels: model, instrument, prediction_class, status
  - Use: Track prediction volume and success rate

**Latency:**
- `nautilus_ml_prediction_latency_seconds` (histogram)
  - Labels: model, instrument
  - Buckets: 0.1ms to 1s (optimized for ML inference)
  - Use: Monitor inference performance, P50/P95/P99

**Confidence:**
- `nautilus_ml_model_confidence` (gauge)
  - Labels: model, instrument
  - Use: Track model confidence over time

**Feature Computation:**
- `nautilus_ml_feature_computation_latency_seconds` (histogram)
  - Labels: instrument, feature_type
  - Use: Monitor feature engineering performance

**Errors:**
- `nautilus_ml_model_errors_total` (counter)
  - Labels: model, instrument, error_type
  - Use: Track inference failures, feature errors, timeouts

### Data Quality Metrics (14 metrics)

**Loading Performance:**
- `data_load_duration_seconds` (histogram) - Load latency distribution
- `data_rows_loaded_total` (counter) - Total rows ingested
- `data_bytes_loaded_total` (counter) - Total bytes ingested

**Quality Indicators:**
- `data_quality_ratio` (gauge) - Overall quality score (0-1)
- `data_missing_values_ratio` (gauge) - Percentage of missing data
- `data_outliers_detected_total` (counter) - Outlier count

**Freshness:**
- `data_staleness_seconds` (gauge) - Time since last update
- `data_source_availability` (gauge) - Source health (0=down, 1=up)

**Cache Performance:**
- `data_cache_hit_ratio` (gauge) - Cache effectiveness
- `data_cache_misses_total` (counter) - Cache miss count

**Validation:**
- `data_validation_failures_total` (counter) - Validation errors
- `data_schema_mismatches_total` (counter) - Schema violations

**System:**
- `data_load_queue_depth` (gauge) - Pending load operations
- `data_load_errors_total` (counter) - Loading failures

### Feature Engineering Metrics (16 metrics)

**Computation:**
- `feature_computation_duration_seconds` (histogram) - Computation latency
- `features_computed_total` (counter) - Total features generated
- `feature_batch_size` (histogram) - Batch size distribution

**Drift Detection:**
- `feature_drift_score` (gauge) - Statistical drift measure
- `feature_correlation_drift` (gauge) - Correlation matrix changes
- `feature_distribution_shift` (gauge) - Distribution changes (KL divergence)

**Quality:**
- `feature_nan_ratio` (gauge) - NaN percentage
- `feature_inf_ratio` (gauge) - Infinite value percentage
- `feature_staleness_seconds` (gauge) - Time since last computation

**Cache:**
- `feature_cache_hit_ratio` (gauge) - Cache effectiveness
- `feature_cache_misses_total` (counter) - Cache misses
- `feature_cache_evictions_total` (counter) - Cache evictions

**System:**
- `feature_engineering_queue_depth` (gauge) - Pending computations
- `feature_dependency_wait_seconds` (histogram) - Dependency resolution time
- `feature_validation_failures_total` (counter) - Validation errors
- `feature_computation_errors_total` (counter) - Computation failures

### Model Lifecycle Metrics (18 metrics)

**Deployment:**
- `model_deployment_timestamp` (gauge) - Deployment time (Unix timestamp)
- `model_version_info` (gauge) - Version tracking (labels: model, version)
- `model_size_bytes` (gauge) - Model file size
- `model_parameter_count` (gauge) - Number of parameters
- `model_flop_count` (gauge) - Floating point operations

**Training:**
- `model_training_duration_seconds` (histogram) - Training time
- `model_training_samples` (histogram) - Training set size
- `model_retraining_triggers_total` (counter) - Retraining events

**Inference:**
- `model_inference_batch_size` (histogram) - Batch size distribution
- `model_inference_queue_depth` (gauge) - Pending inferences
- `model_warm_start_duration_seconds` (histogram) - Model loading time

**ONNX:**
- `model_onnx_conversion_duration_seconds` (histogram) - Conversion time
- `model_onnx_inference_duration_seconds` (histogram) - ONNX inference latency
- `model_onnx_optimization_gain` (gauge) - Speedup ratio

**Errors & Rollbacks:**
- `model_load_errors_total` (counter) - Loading failures
- `model_export_errors_total` (counter) - Export failures
- `model_rollback_events_total` (counter) - Version rollbacks

**A/B Testing:**
- `model_a_b_test_active` (gauge) - A/B test status (0/1)

### Performance Degradation Metrics (12 metrics)

**Accuracy Tracking:**
- `model_accuracy_rolling` (gauge) - Rolling window accuracy
- `model_accuracy_window_std` (gauge) - Accuracy standard deviation
- `prediction_error_rate` (gauge) - Prediction error percentage

**Confidence:**
- `model_confidence_mean` (gauge) - Average confidence score
- `model_confidence_std` (gauge) - Confidence standard deviation

**Distribution Shift:**
- `prediction_distribution_shift` (gauge) - Output distribution changes
- `feature_distribution_shift` (gauge) - Input distribution changes
- `concept_drift_detected` (counter) - Drift detection events

**Classification Metrics:**
- `false_positive_rate` (gauge) - FP / (FP + TN)
- `false_negative_rate` (gauge) - FN / (FN + TP)

**Calibration:**
- `model_calibration_error` (gauge) - Expected Calibration Error (ECE)
- `retraining_recommended` (gauge) - Retraining flag (0/1)

### Resource Utilization Metrics (20 metrics)

**CPU:**
- `cpu_usage_percent` (gauge) - System CPU utilization
- `process_cpu_percent` (gauge) - Process CPU utilization

**Memory:**
- `memory_usage_bytes` (gauge) - System memory used
- `memory_usage_percent` (gauge) - System memory percentage
- `memory_available_bytes` (gauge) - Available memory
- `process_memory_rss_bytes` (gauge) - Process resident set size

**GPU:**
- `gpu_utilization_percent` (gauge) - GPU compute utilization
- `gpu_memory_usage_bytes` (gauge) - GPU memory used
- `gpu_memory_total_bytes` (gauge) - GPU memory total
- `gpu_temperature_celsius` (gauge) - GPU temperature
- `gpu_power_usage_watts` (gauge) - GPU power consumption

**Disk:**
- `disk_usage_bytes` (gauge) - Disk space used
- `disk_usage_percent` (gauge) - Disk space percentage
- `disk_io_read_bytes_total` (counter) - Cumulative disk reads
- `disk_io_write_bytes_total` (counter) - Cumulative disk writes

**Network:**
- `network_io_sent_bytes_total` (counter) - Cumulative network sent
- `network_io_recv_bytes_total` (counter) - Cumulative network received

**Process:**
- `thread_count` (gauge) - Number of threads
- `file_descriptor_count` (gauge) - Open file descriptors
- `context_switches_total` (counter) - Context switch count

### Registry Health Metrics (13 metrics)

**Operations:**
- `registry_operations_total` (counter)
  - Labels: registry_type (feature/model/strategy/data), operation (read/write/validate), status
  - Use: Track registry usage patterns

**Schema:**
- `registry_schema_validation_failures_total` (counter) - Schema validation errors
- `registry_schema_versions_total` (gauge) - Number of schema versions

**Health:**
- `registry_health_score` (gauge)
  - Labels: registry_type
  - Use: Overall registry health (0-1)

**Performance:**
- `registry_query_duration_seconds` (histogram) - Query latency
- `registry_cache_hit_ratio` (gauge) - Cache effectiveness
- `registry_lock_wait_duration_seconds` (histogram) - Lock contention time

**Manifest:**
- `registry_manifest_operations_total` (counter) - Manifest read/write/validate
- `registry_watermark_lag_seconds` (gauge) - Processing lag
- `registry_lineage_depth` (histogram) - Lineage chain length

**Storage:**
- `registry_entries_total` (gauge) - Total registry entries
- `registry_artifact_size_bytes` (histogram) - Artifact size distribution

**Errors:**
- `registry_sync_errors_total` (counter) - Synchronization failures

## Alert Configuration & Runbooks

### Alert Severity Levels

**Critical (P0) - Response Time: 0-15 minutes:**
- `ModelAccuracyCriticallyLow` - Accuracy < 60% for 5m
- `InferenceLatencyExceeded` - P99 > 5s for 2m
- `SystemMemoryExhausted` - Memory > 95% for 1m
- `GPUMemoryExhausted` - GPU memory > 95% for 1m
- `DataPipelineFailure` - Error rate > 0.1/sec for 3m
- `MLServiceDown` - Service unreachable for 1m

**Warning (P1) - Response Time: 1-4 hours:**
- `DataDriftDetected` - Feature drift > 0.3 for 10m
- `ModelAccuracyDegrading` - Accuracy 60-75% for 10m
- `CacheEfficiencyLow` - Cache hit ratio < 70% for 15m
- `HighMemoryUsage` - Memory 80-95% for 5m

**Info (P2) - Response Time: Next business day:**
- `ModelRetrainingRecommended` - Retraining flag set for 30m
- `FeatureComputationSlow` - P95 > 200ms for 20m
- `ModelSizeIncreasing` - Model > 1GB

### Alert Annotations

All alerts include:
- `summary` - One-line description with metric values
- `description` - Detailed explanation with context
- `runbook_url` - Link to remediation steps
- Labels: `severity`, `team`, `category`

**Example Alert:**
```yaml
- alert: ModelAccuracyCriticallyLow
  expr: ml_model_accuracy_rolling < 0.6
  for: 5m
  labels:
    severity: critical
    team: ml-ops
    category: model-performance
  annotations:
    summary: "Critical: Model accuracy degraded to {{ $value | humanizePercentage }}"
    description: "Model {{ $labels.model }} accuracy has fallen below critical threshold (60%) for instrument {{ $labels.instrument }}. Current accuracy: {{ $value | humanizePercentage }}. Immediate action required."
    runbook_url: "https://wiki.nautilus.io/runbooks/ml-accuracy-degraded"
```

## Operational Procedures

### Deployment

**1. Start Full Monitoring Stack:**
```bash
cd ml/monitoring
docker-compose up -d

# Verify services
docker-compose ps

# Check logs
docker-compose logs -f prometheus
docker-compose logs -f grafana
```

**2. Health Check:**
```bash
cd ml/deployment
python check_health.py

# Expected output:
# ✓ Docker Compose - [OK]
# ✓ PostgreSQL - [OK]
# ✓ Redis - [OK]
# ✓ ML Pipeline - [OK]
# ✓ Prometheus - [OK]
# ✓ Grafana - [OK]
```

**3. Access Dashboards:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / nautilus123)
- Alertmanager: http://localhost:9093
- ML Metrics: http://localhost:8080/metrics

### Dashboard Management

**Import Dashboards:**
```bash
cd ml/monitoring/scripts

# Import all dashboards
python import_dashboards.py \
  --url http://localhost:3000 \
  --token <api_token> \
  --input ../grafana/dashboards/ \
  --setup-folder

# Import specific dashboard
python import_dashboards.py \
  --url http://localhost:3000 \
  --token <api_token> \
  --file ../grafana/dashboards/ml-overview.json
```

**Export Dashboards:**
```bash
# Export by tag
python export_dashboards.py \
  --url http://localhost:3000 \
  --token <api_token> \
  --tag ml-monitoring \
  --output ./dashboard_backups/
```

**Validate Dashboards:**
```bash
python validate_dashboards.py --input ../grafana/dashboards/
```

### Real-time Monitoring

**Terminal Dashboard:**
```bash
cd ml/monitoring
python realtime_dashboard.py

# Configure data directory
python realtime_dashboard.py --data-dir /path/to/data/tier1
```

**Metrics Server (Standalone):**
```bash
python -c "
from ml.monitoring import MetricsServer, MonitoringConfig
config = MonitoringConfig(enabled=True, metrics_port=8080)
server = MetricsServer(config)
with server:
    import time
    while True:
        time.sleep(1)
"
```

### Troubleshooting

**Metrics Not Appearing:**
1. Check ML application metrics endpoint: `curl http://localhost:8080/metrics`
2. Verify Prometheus scrape targets: http://localhost:9090/targets
3. Check Prometheus logs: `docker-compose logs prometheus`
4. Verify network connectivity between containers

**Grafana Dashboard Errors:**
1. Verify datasource configuration: http://localhost:3000/datasources
2. Test PromQL queries in Prometheus UI first
3. Check template variable values
4. Review Grafana logs: `docker-compose logs grafana`

**Alerts Not Firing:**
1. Check alert rules in Prometheus: http://localhost:9090/alerts
2. Verify Alertmanager configuration: http://localhost:9093/#/status
3. Test notification channels manually
4. Review alert inhibition rules

**High Memory Usage:**
1. Check Prometheus retention settings (30d default)
2. Review metric cardinality: `http://localhost:9090/api/v1/status/tsdb`
3. Adjust scrape intervals in `prometheus.yml`
4. Enable metric relabeling to drop high-cardinality metrics

## Integration Examples

### Basic Metrics Collection

```python
from ml.monitoring import MLMetricsCollector, MonitoringConfig

# Configure monitoring
config = MonitoringConfig(
    enabled=True,
    metrics_port=8080,
    metrics_prefix="nautilus_ml",
)

# Initialize collector
collector = MLMetricsCollector(config)

# Record prediction
collector.record_prediction(
    model="lgb_v1",
    instrument="EURUSD.SIM",
    prediction_class="BUY",
    latency_seconds=0.002,
    confidence=0.85,
    success=True,
)

# Record feature computation
collector.record_feature_computation(
    instrument="EURUSD.SIM",
    feature_type="technical",
    latency_seconds=0.001,
)

# Context manager for automatic timing
with collector.time_prediction("lgb_v1", "EURUSD.SIM") as timer:
    prediction = model.predict(data)
    timer.set_prediction("BUY", confidence=0.87)
```

### Using Specialized Collectors

```python
from ml.monitoring.collectors import (
    DataQualityCollector,
    FeatureEngineeringCollector,
    ModelLifecycleCollector,
    PerformanceDegradationMonitor,
    ResourceUtilizationCollector,
)

config = MonitoringConfig(enabled=True)

# Data quality monitoring
data_monitor = DataQualityCollector(config)
with data_monitor.time_data_load("EURUSD.SIM", "bars") as timer:
    data = load_bars("EURUSD.SIM")
    timer.set_load_result(rows=10000, cache_hit=True)
data_monitor.record_data_quality(
    source="databento",
    instrument="EURUSD.SIM",
    quality_score=0.95,
    missing_ratio=0.01,
)

# Feature monitoring with drift detection
feature_monitor = FeatureEngineeringCollector(config)
feature_monitor.record_feature_drift(
    instrument="EURUSD.SIM",
    feature_set="technical",
    drift_score=0.15,  # Low drift
)

# Model lifecycle tracking
model_monitor = ModelLifecycleCollector(config)
model_monitor.record_model_deployment(
    model="transformer_v2",
    version="2.1.0",
    instrument="BTCUSD.SIM",
)

# Performance degradation tracking
perf_monitor = PerformanceDegradationMonitor(config)
perf_monitor.record_model_performance(
    model="ensemble_v1",
    accuracy=0.78,
    window="1h",
    confidence_scores=[0.6, 0.8, 0.9, 0.7],
)

# Resource monitoring (background)
resource_monitor = ResourceUtilizationCollector(config)
resource_monitor.start_monitoring()  # Starts background thread
# ... run application
resource_monitor.stop_monitoring()
```

### Centralized Registry Usage

```python
from ml.monitoring.collectors import MLMetricsRegistry

# Initialize all collectors at once
registry = MLMetricsRegistry(config)

# Access individual collectors
data_collector = registry.get_collector("data")
model_collector = registry.get_collector("model")

# Use context manager for automatic lifecycle
with registry:
    # All collectors started
    model_collector.record_model_deployment(...)
    data_collector.record_data_load(...)
    # Collectors automatically stopped on exit
```

### Grafana Client Usage

```python
from ml.monitoring import GrafanaClient, GrafanaDashboardFactory

# Initialize client
client = GrafanaClient(
    base_url="http://localhost:3000",
    api_token="your_api_token_here",
    verify_ssl=False,
)

# Health check
if not client.health_check():
    print("Grafana unavailable")
    exit(1)

# Create dashboard programmatically
factory = GrafanaDashboardFactory()
dashboard = factory.create_base_dashboard(
    title="ML Performance Monitor",
    uid="ml-perf-monitor",
    tags=["ml-monitoring", "performance"],
)

# Add panels
dashboard["panels"].append(
    factory.panel_factory.create_stat_panel(
        title="Prediction Rate",
        expr="rate(nautilus_ml_predictions_total[5m])",
        panel_id=1,
        grid_pos={"h": 4, "w": 6, "x": 0, "y": 0},
        unit="ops",
    )
)

# Deploy to Grafana
result = client.create_dashboard({"dashboard": dashboard, "overwrite": True})
print(f"Dashboard created: {result}")
```

### Metrics Server Deployment

```python
from ml.monitoring import MetricsServer, MonitoringConfig

config = MonitoringConfig(
    enabled=True,
    metrics_port=8080,
    server_timeout=10.0,
)

server = MetricsServer(config)

# Context manager (automatic shutdown)
with server:
    print(f"Metrics available at: {server.get_metrics_url()}")
    print(f"Health check at: {server.get_health_url()}")

    # Wait for ready
    if server.wait_for_ready(timeout=30.0):
        print("Server ready")

    # Keep running
    import time
    while True:
        time.sleep(1)

# Or manual lifecycle
server.start()
print(f"Server running: {server.is_running()}")
# ... do work
server.stop(timeout=10.0)
```

## Testing Patterns

### Collector Testing

```python
import pytest
from ml.monitoring import MonitoringConfig, MLMetricsCollector

@pytest.fixture
def monitoring_config():
    return MonitoringConfig(
        enabled=True,
        metrics_port=8080,
    )

@pytest.fixture
def collector(monitoring_config):
    return MLMetricsCollector(monitoring_config)

def test_record_prediction(collector):
    # Record prediction
    collector.record_prediction(
        model="test_model",
        instrument="TEST.SIM",
        prediction_class="BUY",
        latency_seconds=0.001,
        confidence=0.9,
        success=True,
    )

    # Verify metric (if Prometheus available)
    if collector.enabled:
        # Metric incremented
        assert collector.get_metric_value(
            "ml_predictions_total",
            labels={"model": "test_model", "status": "success"},
        ) == 1.0

def test_prediction_timer(collector):
    # Use context manager
    with collector.time_prediction("test_model", "TEST.SIM") as timer:
        # Simulate prediction work
        import time
        time.sleep(0.01)
        timer.set_prediction("BUY", confidence=0.85)

    # Metrics automatically recorded
```

### Server Testing

```python
def test_metrics_server():
    config = MonitoringConfig(enabled=True, metrics_port=8081)
    server = MetricsServer(config)

    try:
        server.start()
        assert server.is_running()
        assert server.wait_for_ready(timeout=5.0)

        # Test endpoints
        import requests
        response = requests.get(server.get_health_url())
        assert response.status_code == 200

        response = requests.get(server.get_metrics_url())
        assert response.status_code == 200
        assert b"# HELP" in response.content
    finally:
        server.stop()
```

### Grafana Client Testing

```python
def test_grafana_client():
    client = GrafanaClient(
        base_url="http://localhost:3000",
        api_token="test_token",
    )

    # Mock health check
    with patch.object(client, '_make_request', return_value={}):
        assert client.health_check()

    # Test dashboard creation
    dashboard_data = {"title": "Test Dashboard", "panels": []}
    with patch.object(client, '_make_request', return_value={"uid": "test-uid"}):
        result = client.create_dashboard({"dashboard": dashboard_data})
        assert result["uid"] == "test-uid"
```

## Known Limitations & Future Work

### Current Limitations

1. **Dashboard Factory Incomplete:**
   - Basic panel types implemented (stat, timeseries, table, heatmap)
   - Advanced panel types not yet supported (alertlist, logs, trace)
   - No template for complete end-to-end dashboard generation
   - **Workaround:** Use pre-built dashboards in `grafana/dashboards/`

2. **Alert Threshold Tuning:**
   - Alerts use static thresholds
   - No dynamic threshold adjustment based on historical data
   - No anomaly detection integration
   - **Planned:** ML-powered threshold tuning

3. **High Cardinality Risk:**
   - Instrument-level labels can create high cardinality
   - No automatic cardinality limiting
   - **Mitigation:** Set `enable_high_cardinality=False` in config

4. **GPU Metrics:**
   - DCGM exporter commented out by default
   - Requires NVIDIA GPU runtime
   - No AMD GPU support
   - **Workaround:** Uncomment `dcgm-exporter` in `docker-compose.yml`

5. **Distributed Tracing:**
   - No OpenTelemetry/Jaeger integration
   - Request tracing across components not available
   - **Planned:** Distributed tracing support

### Future Enhancements

**Phase 1 (Next 3 months):**
- Complete dashboard factory with all panel types
- Dynamic alert threshold tuning based on historical patterns
- Anomaly detection integration for metric monitoring
- Mobile-responsive web dashboard

**Phase 2 (Next 6 months):**
- Distributed tracing with OpenTelemetry
- Multi-environment configuration (dev/staging/prod)
- Custom metric exporters (DataDog, New Relic)
- Advanced visualization components (correlation heatmaps, dependency graphs)

**Phase 3 (Next 12 months):**
- ML-powered anomaly detection for all metrics
- Predictive alerting based on trend analysis
- Auto-remediation triggers for common issues
- Comprehensive SLO/SLI tracking and reporting

## Cross-Module References

**Related Documentation:**
- `context_data.md` - Data ingestion monitoring integration
- `context_stores.md` - Store operation metrics
- `context_registry.md` - Registry health monitoring
- `context_actors.md` - Actor monitoring via BaseMLInferenceActor
- `context_deployment.md` - Docker deployment and health checks

**Key Integration Points:**
- `ml/common/metrics_bootstrap.py` - Centralized metrics creation
- `ml/common/metrics.py` - Production metrics catalog (40+ metrics)
- `ml/observability/` - Off-hot-path structured data collection
- `ml/actors/base.py` - BaseMLInferenceActor with automatic monitoring
- `ml/deployment/` - Docker Compose stack and health checks

## Key Takeaways

1. **Production-Ready Infrastructure:**
   - Complete monitoring stack with Prometheus, Grafana, Alertmanager
   - 6 specialized collectors covering all ML domains
   - 80+ metrics across data, features, models, performance, resources
   - 18+ critical alerts with runbook integration

2. **Centralized Metrics Pattern:**
   - All metrics via `ml.common.metrics_bootstrap` or `ml.common.metrics_manager`
   - Never import `prometheus_client` directly
   - Idempotent, thread-safe metric creation
   - Graceful degradation when Prometheus unavailable

3. **Comprehensive Observability:**
   - Real-time Prometheus metrics (hot path)
   - Structured observability data collection (cold path)
   - Grafana dashboards for visualization
   - Alert management with severity levels
   - Health checking for all services

4. **Operational Maturity:**
   - Docker Compose deployment with health checks
   - Dashboard import/export automation
   - Real-time terminal monitoring
   - Comprehensive troubleshooting procedures
   - Clear alert runbooks and escalation paths

5. **Integration Excellence:**
   - Automatic integration via BaseMLInferenceActor
   - Circuit breaker metrics included
   - 4-store + 4-registry monitoring
   - Thread-safe operations throughout
   - Progressive fallback chains

**Bottom Line:** The ml/monitoring/ module provides a production-ready, comprehensive monitoring infrastructure that follows Universal ML Architecture patterns and integrates seamlessly with Nautilus Trader's ML components. The implementation is mature, well-tested, and ready for production deployment.
