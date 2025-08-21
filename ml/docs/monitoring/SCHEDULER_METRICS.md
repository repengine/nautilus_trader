# DataScheduler Prometheus Metrics

## Overview

The DataScheduler now includes comprehensive Prometheus metrics for production monitoring, providing real-time observability into data collection, feature computation, and pipeline operations.

## Metrics Categories

### 1. Data Collection Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `nautilus_ml_data_collected_total` | Counter | Total data points collected | source, instrument, data_type |
| `nautilus_ml_data_collection_errors_total` | Counter | Total collection errors | source, instrument, error_type |
| `nautilus_ml_data_collection_latency_seconds` | Histogram | Collection latency | source, instrument |
| `nautilus_ml_active_collection_tasks` | Gauge | Active collection tasks | - |

### 2. Feature Computation Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `nautilus_ml_features_computed_total` | Counter | Total features computed | instrument, feature_type |
| `nautilus_ml_feature_computation_errors_total` | Counter | Feature computation errors | instrument, error_type |
| `nautilus_ml_feature_computation_latency_seconds` | Histogram | Computation latency | instrument, stage |
| `nautilus_ml_active_feature_tasks` | Gauge | Active feature tasks | - |

### 3. Pipeline Stage Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `nautilus_ml_pipeline_stage_latency_seconds` | Histogram | Stage execution time | stage |
| `nautilus_ml_pipeline_runs_total` | Counter | Total pipeline runs | status |

### 4. Data Quality Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `nautilus_ml_data_missing_ratio` | Gauge | Ratio of missing data | instrument, data_type |
| `nautilus_ml_data_staleness_seconds` | Gauge | Age of most recent data | instrument |

### 5. API Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `nautilus_ml_api_request_total` | Counter | Total API requests | endpoint, status_code |
| `nautilus_ml_api_rate_limit_hits_total` | Counter | Rate limit hits | endpoint |

### 6. Storage Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `nautilus_ml_catalog_write_operations_total` | Counter | Catalog write operations | status |
| `nautilus_ml_catalog_write_latency_seconds` | Histogram | Write latency | - |
| `nautilus_ml_feature_store_operations_total` | Counter | Feature store operations | operation, status |
| `nautilus_ml_feature_store_latency_seconds` | Histogram | Operation latency | operation |
| `nautilus_ml_data_retention_cleanup_total` | Counter | Cleanup operations | status |

## Configuration

### Basic Setup

```python
from ml.data.scheduler import DataScheduler
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

# Initialize with metrics server
scheduler = DataScheduler(
    catalog=catalog,
    config=config,
    metrics_port=8000,  # Prometheus metrics port
    start_metrics_server=True,  # Enable metrics server
)
```

### Advanced Configuration

```python
# Custom metrics port
scheduler = DataScheduler(
    catalog=catalog,
    config=config,
    metrics_port=9090,  # Custom port
    start_metrics_server=True,
)

# Disable metrics (for testing)
scheduler = DataScheduler(
    catalog=catalog,
    config=config,
    start_metrics_server=False,  # No metrics server
)
```

## Accessing Metrics

### HTTP Endpoint

Metrics are exposed at `http://localhost:<port>/metrics` in Prometheus format:

```bash
# View metrics
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

### Programmatic Access

```python
from ml._imports import generate_latest

# Get metrics as bytes
metrics_data = generate_latest()

# Parse metrics
metrics_text = metrics_data.decode('utf-8')
for line in metrics_text.split('\n'):
    if line.startswith('nautilus_ml_'):
        print(line)
```

## Integration with Prometheus

### Prometheus Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'nautilus_ml_scheduler'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s
```

### Grafana Dashboard

Example queries for Grafana:

```promql
# Data collection rate
rate(nautilus_ml_data_collected_total[5m])

# Error rate
rate(nautilus_ml_data_collection_errors_total[5m])

# P95 latency
histogram_quantile(0.95,
  rate(nautilus_ml_pipeline_stage_latency_seconds_bucket[5m]))

# Active tasks
nautilus_ml_active_collection_tasks
```

## Alert Rules

Example Prometheus alert rules:

```yaml
groups:
  - name: scheduler_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(nautilus_ml_data_collection_errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate in data collection"

      - alert: SlowPipeline
        expr: histogram_quantile(0.99, nautilus_ml_pipeline_stage_latency_seconds_bucket) > 120
        for: 10m
        annotations:
          summary: "Pipeline execution taking too long"

      - alert: StaleData
        expr: nautilus_ml_data_staleness_seconds > 86400
        for: 1h
        annotations:
          summary: "Data is more than 24 hours old"
```

## Performance Considerations

### Metric Cardinality

The implementation carefully manages label cardinality:

- Instrument labels are bounded by configured symbols
- Error types use a fixed set of categories
- No unbounded labels (e.g., timestamps, IDs)

### Overhead

Metrics collection adds minimal overhead:

- Counter increments: < 1μs
- Histogram observations: < 5μs
- Gauge updates: < 1μs
- HTTP server: Separate thread, no blocking

### Best Practices

1. **Use appropriate histogram buckets**: The default buckets are optimized for the expected latency ranges
2. **Monitor cardinality**: Keep total metric cardinality < 10,000
3. **Set appropriate scrape intervals**: 5-15 seconds recommended
4. **Use metric aggregation**: Aggregate metrics in Prometheus, not the application

## Troubleshooting

### Metrics Server Won't Start

```python
# Check if port is in use
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('127.0.0.1', 8000))
if result == 0:
    print("Port 8000 is already in use")
```

### Missing Metrics

```python
# Verify Prometheus client is installed
from ml._imports import HAS_PROMETHEUS
if not HAS_PROMETHEUS:
    print("Install prometheus-client: pip install prometheus-client")
```

### High Memory Usage

If metrics cause high memory usage:

1. Check metric cardinality
2. Reduce histogram buckets
3. Increase Prometheus scrape interval

## Example Usage

See `ml/examples/scheduler_with_metrics.py` for a complete example demonstrating:

- Scheduler initialization with metrics
- Accessing the metrics endpoint
- Monitoring pipeline operations
- Programmatic metric access

## Testing

Run the test suite:

```bash
pytest ml/tests/integration/test_scheduler_metrics.py -v
```

This validates:

- Metric server initialization
- Metric recording during operations
- Error handling and edge cases
- Prometheus format compliance
