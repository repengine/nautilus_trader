# Extended Prometheus Metrics Architecture for ML System Observability

## Executive Summary
This document outlines the comprehensive extension of the ML monitoring infrastructure to provide deep observability for ML systems in Nautilus Trader. The design maintains backward compatibility while adding specialized collectors for different aspects of ML operations.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MetricsRegistry (Central)                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │MLMetrics     │  │ModelLifecycle│  │DataQuality   │     │
│  │Collector     │  │Collector     │  │Collector     │     │
│  │(existing)    │  │(new)         │  │(new)         │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │Feature       │  │Performance   │  │Resource      │     │
│  │Engineering   │  │Degradation   │  │Utilization   │     │
│  │Collector     │  │Monitor       │  │Collector     │     │
│  │(new)         │  │(new)         │  │(new)         │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  MetricsServer    │
                    │  /metrics endpoint │
                    └───────────────────┘
```

## 1. Metric Categories and Schemas

### 1.1 Model Lifecycle Metrics

```python
# Model versioning and deployment tracking
nautilus_ml_model_info{
    model="xgboost_v1",
    version="1.2.3",
    instrument="EURUSD",
    deployment_time="2025-08-06T10:00:00Z",
    git_commit="abc123"
} 1  # Gauge with labels

nautilus_ml_model_last_trained_timestamp{
    model="xgboost_v1",
    instrument="EURUSD"
} 1707307200  # Unix timestamp

nautilus_ml_model_training_duration_seconds{
    model="xgboost_v1",
    phase="feature_engineering|training|validation"
} 234.5  # Histogram

nautilus_ml_model_size_bytes{
    model="xgboost_v1",
    format="pickle|onnx|joblib"
} 10485760  # Gauge

nautilus_ml_model_load_time_seconds{
    model="xgboost_v1",
    location="memory|disk|remote"
} 0.234  # Histogram
```

### 1.2 Data Quality Metrics

```python
# Data quality and integrity monitoring
nautilus_ml_data_missing_values_ratio{
    instrument="EURUSD",
    data_type="bars|quotes|trades",
    column="close|volume|bid"
} 0.002  # Gauge (0.0-1.0)

nautilus_ml_data_outliers_detected_total{
    instrument="EURUSD",
    detection_method="zscore|iqr|isolation_forest"
} 42  # Counter

nautilus_ml_feature_null_ratio{
    instrument="EURUSD",
    feature="sma_20|rsi_14|volume_ratio"
} 0.0  # Gauge per feature

nautilus_ml_data_staleness_seconds{
    instrument="EURUSD",
    data_type="bars|quotes|trades"
} 3.5  # Gauge (time since last update)

nautilus_ml_data_validation_failures_total{
    instrument="EURUSD",
    validation_type="schema|range|consistency"
} 0  # Counter
```

### 1.3 Feature Engineering Metrics

```python
# Feature computation and quality metrics
nautilus_ml_feature_computation_errors_total{
    instrument="EURUSD",
    feature_type="technical|microstructure|statistical",
    error_type="calculation|timeout|invalid_input"
} 0  # Counter

nautilus_ml_feature_cache_hit_ratio{
    instrument="EURUSD",
    cache_level="memory|disk"
} 0.85  # Gauge (0.0-1.0)

nautilus_ml_feature_drift_score{
    instrument="EURUSD",
    feature="sma_20|rsi_14",
    reference_window="training|last_24h"
} 0.12  # Gauge (KL divergence or similar)

nautilus_ml_feature_importance_score{
    model="xgboost_v1",
    feature="sma_20|rsi_14|volume_ratio"
} 0.234  # Gauge (0.0-1.0)

nautilus_ml_feature_computation_latency_seconds{
    feature_set="technical|all",
    computation_mode="batch|streaming"
} # Histogram with buckets
```

### 1.4 Performance Degradation Metrics

```python
# Model performance monitoring and degradation detection
nautilus_ml_model_accuracy_rolling{
    model="xgboost_v1",
    window="1h|24h|7d",
    metric_type="accuracy|precision|recall|f1"
} 0.82  # Gauge

nautilus_ml_prediction_distribution_shift{
    model="xgboost_v1",
    shift_metric="psi|kl_divergence|wasserstein"
} 0.05  # Gauge

nautilus_ml_inference_timeout_ratio{
    model="xgboost_v1",
    threshold_ms="5|10|50"
} 0.001  # Gauge (0.0-1.0)

nautilus_ml_model_retraining_required{
    model="xgboost_v1",
    reason="drift|performance|schedule"
} 0  # Gauge (0 or 1, binary alert)

nautilus_ml_prediction_confidence_percentiles{
    model="xgboost_v1",
    percentile="p50|p75|p90|p95|p99"
} 0.73  # Gauge
```

### 1.5 Resource Utilization Metrics

```python
# ML-specific resource monitoring
nautilus_ml_gpu_utilization_percent{
    device="cuda:0",
    metric="compute|memory"
} 45.2  # Gauge

nautilus_ml_model_memory_usage_bytes{
    model="xgboost_v1",
    memory_type="resident|virtual|gpu"
} 536870912  # Gauge

nautilus_ml_feature_store_size_bytes{
    storage_type="memory|disk|redis"
} 10737418240  # Gauge

nautilus_ml_inference_batch_size{
    model="xgboost_v1"
} 32  # Gauge

nautilus_ml_training_data_rows_processed_total{
    dataset="train|validation|test"
} 1000000  # Counter
```

## 2. Integration Points

### 2.1 MLDataLoader Integration

```python
class MLDataLoader:
    def __init__(self, catalog, cache_size=1000, enable_cache=True,
                 metrics_collector=None):
        # ... existing code ...
        self._metrics = metrics_collector or DataQualityCollector()

    def load_bars(self, instrument, start=None, end=None):
        with self._metrics.time_data_load(instrument, "bars") as timer:
            # Load data
            df = self._load_bars_internal(instrument, start, end)

            # Record data quality metrics
            timer.set_rows(len(df))
            timer.set_cache_hit(self._check_cache_hit(instrument))

            # Check data quality
            missing_ratio = df.isnull().sum() / len(df)
            self._metrics.record_data_quality(
                instrument=instrument,
                missing_ratio=missing_ratio,
                outliers=self._detect_outliers(df)
            )

            return df
```

### 2.2 FeatureEngineer Integration

```python
class FeatureEngineer:
    def __init__(self, config, metrics_collector=None):
        # ... existing code ...
        self._metrics = metrics_collector or FeatureEngineeringCollector()
        self._feature_cache = {}

    def compute_features(self, bars):
        with self._metrics.time_feature_computation("technical") as timer:
            # Check cache
            cache_key = self._get_cache_key(bars)
            if cache_key in self._feature_cache:
                self._metrics.record_cache_hit("memory")
                return self._feature_cache[cache_key]

            # Compute features
            features = self._compute_technical_features(bars)

            # Track feature importance if available
            if hasattr(self, 'feature_importances_'):
                for feature, importance in self.feature_importances_.items():
                    self._metrics.record_feature_importance(
                        feature=feature,
                        importance=importance
                    )

            # Cache and return
            self._feature_cache[cache_key] = features
            timer.set_features_computed(len(features.columns))

            return features
```

### 2.3 Future Component Integration (XGBoostTrainer)

```python
class XGBoostTrainer:
    def __init__(self, config, metrics_collector=None):
        self._metrics = metrics_collector or ModelLifecycleCollector()

    def train(self, X, y):
        with self._metrics.time_training(self.model_name) as timer:
            # Training phases
            timer.mark_phase("feature_engineering")
            X_processed = self._preprocess_features(X)

            timer.mark_phase("training")
            self.model = xgb.XGBClassifier(**self.config)
            self.model.fit(X_processed, y)

            timer.mark_phase("validation")
            score = self._validate_model(X_processed, y)

            # Record model metadata
            self._metrics.record_model_info(
                model=self.model_name,
                version=self.version,
                size_bytes=self._get_model_size(),
                training_rows=len(X),
                score=score
            )

            return self.model
```

## 3. Collector Architecture

### 3.1 Base Collector Pattern

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseMetricsCollector(ABC):
    """Base class for all specialized collectors."""

    def __init__(self, config: MonitoringConfig):
        self._config = config
        self._enabled = config.enabled and HAS_PROMETHEUS
        self._lock = threading.RLock()

        if self._enabled:
            self._initialize_metrics()

    @abstractmethod
    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        pass

    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._enabled
```

### 3.2 Specialized Collectors

```python
class ModelLifecycleCollector(BaseMetricsCollector):
    """Tracks model versioning, deployment, and lifecycle metrics."""

    def _initialize_metrics(self):
        self._model_info = Gauge(
            f"{self._config.metrics_prefix}_model_info",
            "Model deployment information",
            ["model", "version", "instrument", "deployment_time", "git_commit"]
        )
        # ... more metrics ...

    def record_model_deployment(self, model: str, version: str, **kwargs):
        """Record model deployment event."""
        # Implementation

class DataQualityCollector(BaseMetricsCollector):
    """Monitors data quality and integrity."""

    def _initialize_metrics(self):
        self._missing_values_ratio = Gauge(
            f"{self._config.metrics_prefix}_data_missing_values_ratio",
            "Ratio of missing values in data",
            ["instrument", "data_type", "column"]
        )
        # ... more metrics ...

    def record_data_quality(self, instrument: str, **metrics):
        """Record data quality metrics."""
        # Implementation

class FeatureEngineeringCollector(BaseMetricsCollector):
    """Tracks feature computation and quality."""

    def _initialize_metrics(self):
        self._feature_drift = Gauge(
            f"{self._config.metrics_prefix}_feature_drift_score",
            "Feature drift score",
            ["instrument", "feature", "reference_window"]
        )
        # ... more metrics ...

    def record_feature_drift(self, feature: str, drift_score: float):
        """Record feature drift metrics."""
        # Implementation
```

### 3.3 Composite Registry

```python
class MLMetricsRegistry:
    """Central registry for all ML metrics collectors."""

    def __init__(self, config: MonitoringConfig):
        self.config = config

        # Initialize all collectors
        self.ml_metrics = MLMetricsCollector(config)  # Existing
        self.model_lifecycle = ModelLifecycleCollector(config)
        self.data_quality = DataQualityCollector(config)
        self.feature_engineering = FeatureEngineeringCollector(config)
        self.performance = PerformanceDegradationMonitor(config)
        self.resources = ResourceUtilizationCollector(config)

        # Single metrics server for all collectors
        self.server = MetricsServer(config)

    def start(self):
        """Start metrics collection and server."""
        self.server.start()

    def stop(self):
        """Stop metrics collection and server."""
        self.server.stop()

    def get_collector(self, collector_type: str) -> BaseMetricsCollector:
        """Get specific collector by type."""
        collectors = {
            "ml": self.ml_metrics,
            "model": self.model_lifecycle,
            "data": self.data_quality,
            "features": self.feature_engineering,
            "performance": self.performance,
            "resources": self.resources
        }
        return collectors.get(collector_type)
```

## 4. Implementation Strategy

### Phase 1: Foundation (Week 1)

1. Create BaseMetricsCollector abstract class
2. Implement MLMetricsRegistry for centralized management
3. Add configuration extensions to MonitoringConfig
4. Create comprehensive unit tests

### Phase 2: Core Collectors (Week 2)

1. Implement ModelLifecycleCollector
2. Implement DataQualityCollector
3. Integrate with MLDataLoader
4. Add integration tests

### Phase 3: Advanced Collectors (Week 3)

1. Implement FeatureEngineeringCollector
2. Implement PerformanceDegradationMonitor
3. Integrate with FeatureEngineer
4. Add feature drift detection

### Phase 4: Resource & Optimization (Week 4)

1. Implement ResourceUtilizationCollector
2. Add GPU monitoring (if available)
3. Optimize metric cardinality
4. Performance testing

## 5. Testing Strategy

### Unit Tests

```python
def test_model_lifecycle_collector():
    """Test model lifecycle metrics collection."""
    config = MonitoringConfig(enabled=True)
    collector = ModelLifecycleCollector(config)

    # Record deployment
    collector.record_model_deployment(
        model="xgboost_v1",
        version="1.2.3",
        instrument="EURUSD"
    )

    # Verify metrics
    assert collector.get_metric_value("model_info") == 1
```

### Integration Tests

```python
def test_ml_data_loader_with_metrics():
    """Test MLDataLoader with metrics collection."""
    config = MonitoringConfig(enabled=True)
    registry = MLMetricsRegistry(config)

    loader = MLDataLoader(
        catalog,
        metrics_collector=registry.get_collector("data")
    )

    # Load data and verify metrics
    df = loader.load_bars("EURUSD")

    # Check metrics were recorded
    metrics = registry.data_quality.get_metrics()
    assert "data_missing_values_ratio" in metrics
```

### Performance Tests

```python
def test_metrics_overhead():
    """Ensure metrics collection overhead is acceptable."""
    # Without metrics
    start = time.perf_counter()
    for _ in range(1000):
        compute_features(bars)
    baseline = time.perf_counter() - start

    # With metrics
    start = time.perf_counter()
    for _ in range(1000):
        compute_features_with_metrics(bars)
    with_metrics = time.perf_counter() - start

    # Overhead should be < 5%
    overhead = (with_metrics - baseline) / baseline
    assert overhead < 0.05
```

## 6. Migration from OLD System

### Mapping OLD Metrics to New Architecture

- `model_prediction_counter` → `MLMetricsCollector.record_prediction()`
- `portfolio_value` → Future: PortfolioMetricsCollector
- `feature_parity_drift` → `FeatureEngineeringCollector.record_feature_drift()`
- `system_health` → `MetricsServer./health` endpoint

### Migration Steps

1. Identify actively used metrics from OLD system
2. Map to new collector architecture
3. Provide compatibility layer if needed
4. Gradual migration with feature flags

## 7. Performance Considerations

### Metric Cardinality Management

```python
class CardinalityLimiter:
    """Limit metric label cardinality to prevent explosion."""

    MAX_LABELS = {
        "instrument": 100,
        "model": 50,
        "feature": 200,
    }

    def check_cardinality(self, label_type: str, value: str) -> bool:
        """Check if adding this label would exceed limits."""
        # Implementation
```

### Aggregation Strategy

- Use histograms for latency (pre-aggregated percentiles)
- Use summaries sparingly (expensive)
- Aggregate at collection time, not query time
- Use recording rules in Prometheus for complex queries

### Resource Limits

```python
# In MonitoringConfig
max_metrics_per_collector: int = 1000
max_label_cardinality: Dict[str, int] = {
    "instrument": 100,
    "model": 50,
}
metric_ttl_seconds: float = 3600  # Auto-expire old metrics
```

## 8. Grafana Dashboard Templates

### Model Performance Dashboard

```json
{
  "dashboard": {
    "title": "ML Model Performance",
    "panels": [
      {
        "title": "Model Accuracy (24h rolling)",
        "targets": [
          {
            "expr": "nautilus_ml_model_accuracy_rolling{window='24h'}"
          }
        ]
      },
      {
        "title": "Prediction Latency P95",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, nautilus_ml_prediction_latency_seconds)"
          }
        ]
      }
    ]
  }
}
```

### Data Quality Dashboard

```json
{
  "dashboard": {
    "title": "ML Data Quality",
    "panels": [
      {
        "title": "Missing Data Ratio",
        "targets": [
          {
            "expr": "nautilus_ml_data_missing_values_ratio"
          }
        ]
      },
      {
        "title": "Feature Drift Score",
        "targets": [
          {
            "expr": "nautilus_ml_feature_drift_score"
          }
        ]
      }
    ]
  }
}
```

## 9. Alert Rules

```yaml
groups:
  - name: ml_alerts
    rules:
      - alert: ModelPerformanceDegraded
        expr: nautilus_ml_model_accuracy_rolling{window="1h"} < 0.7
        for: 5m
        annotations:
          summary: "Model {{ $labels.model }} accuracy below threshold"

      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, nautilus_ml_prediction_latency_seconds) > 0.01
        for: 5m
        annotations:
          summary: "P95 latency above 10ms for {{ $labels.model }}"

      - alert: FeatureDriftDetected
        expr: nautilus_ml_feature_drift_score > 0.3
        for: 10m
        annotations:
          summary: "Feature drift detected for {{ $labels.feature }}"
```

## 10. Example Usage

```python
from ml.monitoring import MLMetricsRegistry
from ml.data.loader import MLDataLoader
from ml.features.engineering import FeatureEngineer

# Initialize monitoring
config = MonitoringConfig(
    enabled=True,
    metrics_port=8080,
    enable_high_cardinality=False
)
metrics = MLMetricsRegistry(config)
metrics.start()

# Use with components
loader = MLDataLoader(
    catalog,
    metrics_collector=metrics.get_collector("data")
)

engineer = FeatureEngineer(
    config,
    metrics_collector=metrics.get_collector("features")
)

# Load data with automatic metrics
df = loader.load_bars("EURUSD", start="2024-01-01", end="2024-12-31")

# Compute features with automatic metrics
features = engineer.compute_features(df)

# Access metrics endpoint
# curl http://localhost:8080/metrics
```

## Conclusion

This extended metrics architecture provides:

1. **Comprehensive Observability**: Deep insights into all ML components
2. **Production-Ready**: Graceful degradation, thread-safety, performance limits
3. **Easy Integration**: Minimal code changes to existing components
4. **Backward Compatible**: Existing MLMetricsCollector continues to work
5. **Scalable**: Modular design allows adding new collectors easily
6. **Performance Conscious**: Cardinality limits, efficient aggregation
7. **Testing Coverage**: Unit, integration, and performance tests

The architecture follows Nautilus Trader patterns and maintains the high quality standards of the existing codebase.
