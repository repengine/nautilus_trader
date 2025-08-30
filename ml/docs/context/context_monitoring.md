# ML Monitoring Infrastructure Context

## Executive Summary

The ML monitoring infrastructure for Nautilus Trader provides comprehensive observability for machine learning components through a Prometheus/Grafana/AlertManager stack. This production-ready system offers real-time metrics collection, visualization, and alerting across data quality, feature engineering, model lifecycle, performance degradation, and resource utilization.

Operational notes:
- Include DB preflight status and engine pool telemetry in dashboards where useful. Preflight utility: `ml/stores/db_preflight.py`. Pool status: `EngineManager.get_pool_status()`.

### Key Features

- **Production-Ready**: Docker-based deployment with health checks and automatic restarts
- **Comprehensive Coverage**: Specialized metrics collectors covering all ML pipeline stages
- **Real-time Alerting**: Multi-tier alert system with Slack, email, and PagerDuty integration
- **Performance Optimized**: Sub-5ms latency overhead with graceful degradation
- **Extensible Architecture**: Modular collector design supporting future ML components
- **Centralized Metrics**: All metrics must be acquired via `ml/common/metrics_bootstrap.py` (get_counter/get_histogram/get_gauge) or imported from `ml/common/metrics.py` to avoid duplication

### Current Status

- ✅ Core infrastructure implemented and tested
- ✅ Basic metrics collection (MLMetricsCollector) operational
- ✅ Centralized metrics in `ml/common/metrics.py` (30+ metrics)
- ✅ Docker deployment stack configured
- ✅ Grafana dashboards and alert rules defined
- ✅ DataScheduler metrics integrated
- ✅ Pipeline health monitoring with SQL views (moved to migrations)
- ✅ Health check script (`check_pipeline_health.py`)
- ✅ Real-time dashboard (`realtime_dashboard.py`)
- ✅ Extended metrics collectors implemented (data, features, model, performance, registry, resources)
- ✅ MetricsServer with /metrics and /health endpoints
- 🔄 Full integration with all ML components in progress

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     ML Application                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Data Loaders │  │ Feature Eng. │  │ ML Actors    │     │
│  │ + Metrics    │  │ + Metrics    │  │ + Metrics    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────┬───────────────────────────────────────┘
                      │ /metrics endpoint (port 8000)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Monitoring Stack                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Prometheus  │  │   Grafana    │  │ AlertManager │     │
│  │  (port 9090) │  │ (port 3000)  │  │ (port 9093)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │ Pushgateway  │  │Node Exporter │                       │
│  │ (port 9091)  │  │ (port 9100)  │                       │
│  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │ Alert Channels   │
            │ - Slack          │
            │ - Email          │
            │ - PagerDuty      │
            └──────────────────┘
```

### Technology Stack

- **Metrics Collection**: Prometheus Client (Python)
- **Time Series Database**: Prometheus 2.48.0
- **Visualization**: Grafana 10.2.3
- **Alerting**: AlertManager 0.27.0
- **Container Orchestration**: Docker Compose
- **Reverse Proxy**: Optional (nginx/traefik for production)

## Metrics Catalog
### Bootstrap Usage and Naming Standards

- Always acquire metrics via `ml.common.metrics_bootstrap`:
  - `get_counter(name, description, labelnames)`
  - `get_histogram(name, description, labelnames, buckets=...)`
  - `get_gauge(name, description, labelnames)`
- Do not instantiate Prometheus collectors directly in modules. This prevents duplicate registration and ensures idempotency across tests and processes.
- Naming conventions: prefix all ML metrics with `nautilus_ml_` and use snake_case. Labels should be stable and minimal (e.g., `model`, `instrument`, `stage`, `status`).
- Central, widely used metrics live in `ml/common/metrics.py`; domain‑specific metrics use the bootstrap in their owning module.

### Core Metrics (MLMetricsCollector)

| Metric Name | Type | Description | Labels | Alert Threshold |
|-------------|------|-------------|--------|-----------------|
| `nautilus_ml_predictions_total` | Counter | Total ML predictions made | model, instrument, prediction_class, status | N/A |
| `nautilus_ml_prediction_latency_seconds` | Histogram | Model inference latency | model, instrument | P99 > 5s |
| `nautilus_ml_model_confidence` | Gauge | Current model confidence | model, instrument | < 0.6 |
| `nautilus_ml_feature_computation_latency_seconds` | Histogram | Feature computation time | instrument, feature_type | P95 > 500ms |
| `nautilus_ml_model_errors_total` | Counter | Model error count | model, instrument, error_type | Rate > 5/min |

### Centralized Metrics (ml/common/metrics.py)

#### Data Pipeline Metrics
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `nautilus_ml_data_events_total` | Counter | Total data events processed | dataset_type, component, stage, source, status |
| `nautilus_ml_watermark_lag_seconds` | Gauge | Lag since last processing | dataset, instrument, source |
| `nautilus_ml_stage_coverage_pct` | Gauge | Coverage between pipeline stages | dataset, from_stage, to_stage |
| `nautilus_ml_contract_violations_total` | Counter | Contract validation violations | dataset, rule |

#### Data Collection Metrics
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `nautilus_ml_data_collection_duration` | Histogram | Duration of collection operations | source, schema |
| `nautilus_ml_data_collection_errors` | Counter | Total collection errors | source, instrument, error_type |
| `nautilus_ml_catalog_write_operations` | Counter | Catalog write operations | status |

#### Feature Store Metrics
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `nautilus_ml_feature_store_operations` | Counter | Total operations | operation, status |
| `nautilus_ml_feature_computation_duration_seconds` | Histogram | Feature computation time | feature_set, mode |
| `nautilus_ml_feature_drift_score` | Gauge | Feature drift score (0-1) | feature_set, feature_name |

#### Model Store Metrics
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `nautilus_ml_model_store_operations` | Counter | Total operations | operation, status |
| `nautilus_ml_model_inference_duration_seconds` | Histogram | Model inference time | model_id, version |
| `nautilus_ml_model_accuracy` | Gauge | Model accuracy score | model_id, version |
| `nautilus_ml_model_confidence` | Gauge | Average confidence score | model_id, version |

#### Strategy Store Metrics
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `nautilus_ml_strategy_store_operations` | Counter | Total operations | operation, status |
| `nautilus_ml_strategy_signal_generation_duration_seconds` | Histogram | Signal generation time | strategy_id |
| `nautilus_ml_strategy_pnl` | Gauge | Strategy P&L | strategy_id, timeframe |

#### Validation Metrics
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `nautilus_ml_validation_violations` | Counter | Validation violations | dataset_id, rule_type, severity |
| `nautilus_ml_validation_duration_seconds` | Histogram | Validation time | dataset_id |
| `nautilus_ml_schema_mismatches` | Counter | Schema validation failures | dataset, mismatch_type |
| `nautilus_ml_write_rejections` | Counter | Rejected writes | dataset_id, reason |
| `nautilus_ml_data_quality_score` | Histogram | Data quality distribution | dataset_id |

#### System Health Metrics
| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `nautilus_ml_pipeline_health` | Gauge | Overall health score (0-1) | component |
| `nautilus_ml_system_ready` | Gauge | System readiness (0/1) | component |

### Pipeline Metrics (DataScheduler)

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `nautilus_ml_pipeline_stage_latency_seconds` | Histogram | Stage execution latency | stage |
| `nautilus_ml_pipeline_runs_total` | Counter | Total pipeline runs | status |

### Extended Collectors (Implemented)

#### DataQualityCollector (ml/monitoring/collectors/data.py)
- Data loading performance and cache efficiency
- Missing values and data quality ratios
- Outlier detection and validation failures
- Data staleness and freshness monitoring

#### FeatureEngineeringCollector (ml/monitoring/collectors/features.py)
- Feature computation performance
- Feature drift detection and scoring
- Cache hit ratios and efficiency
- Feature importance tracking

#### ModelLifecycleCollector (ml/monitoring/collectors/model.py)
- Model deployment metadata and versioning
- Training duration and performance tracking
- Model size and loading time monitoring
- Model errors and deployment failures

#### PerformanceDegradationCollector (ml/monitoring/collectors/performance.py)
- Rolling accuracy and performance tracking
- Prediction distribution shift detection
- Inference timeout monitoring
- Retraining requirement indicators

#### ResourceUtilizationCollector (ml/monitoring/collectors/resources.py)
- Memory and CPU utilization tracking
- GPU utilization monitoring (when available)
- Disk I/O and storage metrics
- Feature store and model registry size

#### RegistryHealthCollector (ml/monitoring/collectors/registry.py)
- Registry operation metrics
- Schema validation tracking
- Manifest registration events
- Version management metrics

## Dashboard Specifications

### 1. ML Overview Dashboard (`ml-overview.json`)
**Purpose**: High-level system health and key performance indicators

**Panels**:

- Model Accuracy Trend (Time Series)
- Inference Latency P99 (Stat Panel)
- Prediction Rate (Gauge)
- System Health Score (Stat Panel)
- Recent Alerts (Table)
- Resource Utilization Summary (Row of Gauges)

**Refresh**: 10 seconds
**Variables**: `$model`, `$instrument`, `$interval`

### 2. Pipeline Health Dashboard (`pipeline-health.json`)
**Purpose**: Monitor data pipeline operations and health

**Panels**:

- Pipeline Run Status (Time Series) - Success/failure over time
- Data Collection Latency (Histogram)
- Feature Computation Rate (Gauge)
- FeatureStore Write Performance (Stat Panel)
- Data Staleness by Instrument (Table)
- Databento API Usage (Counter)
- Last Successful Run (Stat Panel) - Alert: >25 hours
- Error Rate Trend (Time Series)

**Refresh**: 30 seconds
**Variables**: `$instrument`, `$stage`

### 3. Data Quality Dashboard (`data-quality.json`)
**Purpose**: Monitor data pipeline health and quality metrics

**Panels**:

- Missing Values Ratio (Gauge) - Alert: >5%
- Outliers Detected Rate (Time Series)
- Data Staleness (Stat Panel) - Alert: >300s
- Validation Failures (Counter) - Alert: >10/min
- Data Load Latency Distribution (Histogram)
- Data Source Status (Table)

**Refresh**: 30 seconds

### 4. Feature Engineering Dashboard (`feature-engineering.json`)
**Purpose**: Track feature computation performance and quality

**Panels**:

- Feature Computation Latency (Histogram)
- Feature Drift Heatmap (Heatmap)
- Feature Cache Hit Ratio (Gauge) - Target: >80%
- Feature Importance Rankings (Bar Chart)
- Computation Error Rate (Time Series)
- Feature Null Ratio Table (Table)

**Refresh**: 30 seconds

### 5. Model Lifecycle Dashboard (`model-lifecycle.json`)
**Purpose**: Monitor model deployment, training, and versioning

**Panels**:

- Active Model Information (Stat Panels)
- Training Duration Trends (Time Series)
- Model Size Tracking (Gauge)
- Load Time Performance (Histogram)
- Deployment History (Table)
- Model Registry Status (Table)

**Refresh**: 60 seconds

### 6. Performance Degradation Dashboard (`performance-degradation.json`)
**Purpose**: Detect and visualize model performance issues

**Panels**:

- Rolling Accuracy (Time Series) - Alert: <70%
- Prediction Distribution Shift (Heatmap)
- Inference Timeout Ratio (Gauge) - Alert: >1%
- Confidence Score Distribution (Histogram)
- Retraining Indicators (Binary Panel)
- Performance Alert History (Table)

**Refresh**: 15 seconds

### 7. Resource Utilization Dashboard (`resource-utilization.json`)
**Purpose**: Monitor system resource consumption

**Panels**:

- CPU Usage Timeline (Time Series) - Alert: >80%
- Memory Usage (Gauge) - Alert: >90%
- GPU Utilization (Time Series) - Alert: >95% (if available)
- Disk I/O Rates (Time Series)
- Inference Batch Size (Histogram)
- Feature Store Growth (Time Series)

**Refresh**: 15 seconds

### 8. Real-time Terminal Dashboard (`realtime_dashboard.py`)
**Purpose**: Live monitoring from the command line using Rich library

**Features**:

- System metrics monitoring with live updates
- Data ingestion rate tracking
- Feature computation performance
- Model inference metrics
- Alert notifications
- Resource utilization display

**Usage**:
```bash
python ml/monitoring/realtime_dashboard.py
```

The terminal dashboard provides:
- Live data ingestion metrics (L0/L1/L2 symbols, size, rates)
- Feature computation statistics
- Model performance indicators
- System health monitoring
- Alert summary panel
- Auto-refresh with configurable intervals

## Alert Configuration

### Alert Severity Levels

#### Critical (P0) - Immediate Action Required

- **ModelAccuracyCriticallyLow**: Accuracy < 60% for 5 minutes
- **InferenceLatencyExceeded**: P99 latency > 5 seconds for 2 minutes
- **SystemMemoryExhausted**: Memory usage > 95% for 1 minute
- **DataPipelineFailure**: Error rate > 0.1/sec for 3 minutes
- **MLServiceDown**: Service unreachable for 1 minute

**Response Time**: Immediate (0-15 minutes)
**Escalation**: Slack #critical-alerts + PagerDuty + Email

#### Warning (P1) - Investigation Needed

- **DataDriftDetected**: Feature drift score > 0.3 for 10 minutes
- **ModelAccuracyDegrading**: Accuracy 60-75% for 10 minutes
- **CacheEfficiencyLow**: Cache hit ratio < 70% for 15 minutes
- **HighMemoryUsage**: Memory 80-95% for 5 minutes
- **ValidationFailuresIncreasing**: Validation failures > 1/sec for 10 minutes

**Response Time**: 1-4 hours
**Escalation**: Slack team channels + Email

#### Info (P2) - Monitoring Only

- **ModelRetrainingRecommended**: Retraining flag set for 30 minutes
- **FeatureComputationSlow**: P95 latency > 200ms for 20 minutes
- **ModelTrainingDurationHigh**: Training > 1 hour
- **ModelSizeIncreasing**: Model size > 1GB

**Response Time**: Next business day
**Escalation**: Email digest only

### Alert Routing Rules

```yaml
route:
  group_by: ['alertname', 'severity', 'team']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'ml-team'

  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 0s
      repeat_interval: 5m

    - match:
        team: ml-ops
      receiver: 'ml-team'
      group_by: ['alertname', 'model']

    - match:
        team: data-engineering
      receiver: 'data-team'
      group_by: ['alertname', 'source']
```

### Notification Channels

#### Slack Integration

- **#critical-alerts**: P0 alerts with immediate escalation
- **#ml-alerts**: P1/P2 alerts with context and runbook links
- **#data-alerts**: Data quality and pipeline alerts
- **#infra-alerts**: Resource and infrastructure alerts

#### Email Configuration

- **<oncall@nautilus.io>**: Critical alerts with escalation
- **<ml-team@nautilus.io>**: Standard ML operational alerts
- **<data-team@nautilus.io>**: Data quality and pipeline alerts
- **<ml-digest@nautilus.io>**: Daily/weekly summary reports

#### PagerDuty Integration

- **Service Key**: Critical alerts only (P0)
- **Escalation Policy**: 5 min → primary → 10 min → secondary → 15 min → manager

## Integration Patterns

### MLDataLoader Integration

```python
from ml.monitoring.collector import MLMetricsCollector
from ml.monitoring._config import MonitoringConfig
from ml.data.loader import MLDataLoader

# Initialize metrics
config = MonitoringConfig(enabled=True, metrics_port=8000)
metrics = MLMetricsCollector(config)

class MonitoredMLDataLoader(MLDataLoader):
    def __init__(self, catalog, metrics_collector=None):
        super().__init__(catalog)
        self.metrics = metrics_collector

    def load_bars(self, instrument, start=None, end=None):
        with self.metrics.time_feature_computation(instrument, "data_load"):
            # Load data with automatic timing
            return super().load_bars(instrument, start, end)
```

### FeatureEngineer Integration

```python
from ml.features.engineering import FeatureEngineer

class MonitoredFeatureEngineer(FeatureEngineer):
    def __init__(self, config, metrics_collector=None):
        super().__init__(config)
        self.metrics = metrics_collector

    def compute_features(self, data):
        with self.metrics.time_feature_computation("EURUSD", "technical"):
            features = super().compute_features(data)

            # Record quality metrics (would use DataQualityCollector)
            if self.metrics:
                null_ratio = features.isnull().sum().sum() / features.size
                # metrics.record_feature_quality("technical", null_ratio)

            return features
```

### BaseMLInferenceActor Integration

```python
from ml.actors.base import BaseMLInferenceActor

class ModelActor(BaseMLInferenceActor):
    def __init__(self, config):
        super().__init__(config)
        # Metrics collector can be initialized if monitoring is enabled
        if config.monitoring.enabled:
            self.metrics = MLMetricsCollector(config.monitoring)

    def on_data(self, data):
        if hasattr(self, 'metrics'):
            with self.metrics.time_prediction("xgboost_v1", "EURUSD") as timer:
                prediction = self.model.predict(data)
                confidence = self.model.predict_proba(data).max()

                timer.set_prediction(
                    prediction_class="buy" if prediction > 0 else "sell",
                    confidence=confidence
                )

                return prediction
        else:
            # Monitoring disabled, just run prediction
            return self.model.predict(data)
```

## Performance Baselines

### System Performance Targets

- **Metric Collection Overhead**: < 5ms per operation
- **Memory Footprint**: < 100MB for collector components
- **CPU Overhead**: < 2% additional CPU usage
- **Network Bandwidth**: < 1MB/min metrics export

### Dashboard Performance

- **Load Time**: < 2 seconds for standard dashboards
- **Query Response**: < 500ms for 95% of queries
- **Refresh Rate**: 10s for critical, 30s for standard panels
- **Concurrent Users**: Support 20+ simultaneous users

### Storage Requirements

- **Metrics Retention**: 30 days at full resolution
- **Storage Growth**: ~10GB per month with default scrape intervals
- **Backup Size**: ~100MB dashboard configurations
- **Log Retention**: 7 days for application logs

## Operational Procedures

### Daily Operations

#### Health Check Routine

1. Verify all containers running: `docker-compose ps`
2. Check Prometheus targets: <http://localhost:9090/targets>
3. Verify Grafana accessibility: <http://localhost:3000>
4. Review critical alerts: <http://localhost:9093>

#### Performance Monitoring

1. Check query performance in Grafana
2. Review slow queries in Prometheus
3. Monitor resource utilization trends
4. Validate metric collection rates

### Weekly Maintenance

#### Dashboard Review

1. Review dashboard usage analytics
2. Update template variables if needed
3. Optimize slow-performing queries
4. Archive unused dashboards

#### Alert Tuning

1. Review alert firing frequency
2. Adjust thresholds based on historical data
3. Update notification channels as needed
4. Test alert escalation paths

### Monthly Tasks

#### Capacity Planning

1. Analyze storage growth trends
2. Review metric cardinality
3. Plan for scaling if needed
4. Update retention policies

#### Security Audit

1. Review access permissions
2. Rotate API keys and passwords
3. Update SSL certificates
4. Audit metric exposure

### Disaster Recovery

#### Backup Procedures

```bash
# Backup Grafana dashboards
docker exec nautilus_grafana grafana-cli admin export-dashboard

# Backup Prometheus configuration
tar -czf prometheus-config-$(date +%Y%m%d).tar.gz prometheus/

# Backup metrics data (if needed)
docker exec nautilus_prometheus promtool tsdb snapshot /prometheus
```

#### Recovery Procedures

```bash
# Restore from backup
docker-compose down
# Restore configuration files
docker-compose up -d

# Import dashboards
python scripts/import_dashboards.py --all
```

### Troubleshooting Guide

#### Common Issues

**Metrics Not Appearing**

1. Check ML application metrics endpoint: `curl http://localhost:8000/metrics`
2. Verify Prometheus scrape config
3. Check network connectivity between containers
4. Review Prometheus logs: `docker-compose logs prometheus`

**Grafana Dashboard Errors**

1. Verify datasource configuration
2. Check PromQL query syntax
3. Review template variable values
4. Validate time range selection

**Alerts Not Firing**

1. Check alert rule syntax in Prometheus
2. Verify AlertManager configuration
3. Test notification channels manually
4. Review alert inhibition rules

**Performance Issues**

1. Check query complexity and duration
2. Review metric cardinality
3. Analyze dashboard panel count
4. Monitor resource utilization

## Current Implementation Status

### ✅ Completed Components

- **Core Infrastructure**: Docker compose stack with health checks
- **Basic Metrics Collection**: MLMetricsCollector with 5 core metrics
- **Centralized Metrics**: 30+ metrics defined in `ml/common/metrics.py`
- **Extended Collectors**: 6 specialized collectors implemented
  - DataQualityCollector
  - FeatureEngineeringCollector
  - ModelLifecycleCollector
  - PerformanceDegradationCollector
  - ResourceUtilizationCollector
  - RegistryHealthCollector
- **Metrics Server**: HTTP server with /metrics and /health endpoints
- **Configuration System**: Type-safe configuration with MonitoringConfig
- **Dashboard Factory**: Programmatic dashboard generation
- **Grafana Client**: API client for dashboard management
- **Alert Rules**: Critical, warning, and info alert definitions
- **Pipeline Health Monitoring**: SQL views (migrated to ml/stores/migrations/005_views.sql)
- **Health Check Script**: `check_pipeline_health.py` for automated checks
- **Real-time Dashboard**: `realtime_dashboard.py` for live monitoring
- **Documentation**: Comprehensive setup and operation guides

### 🔄 In Development

- **Full Integration**: Complete integration with all ML components
- **Advanced Dashboard Panels**: Fine-tuning dashboard layouts
- **Integration Testing**: End-to-end validation with ML components
- **Performance Optimization**: Query optimization and caching
- **Security Hardening**: Authentication and TLS configuration

### 📋 Planned Enhancements

- **Distributed Tracing**: Integration with Jaeger/Zipkin
- **Log Aggregation**: ELK stack integration
- **Mobile Dashboard**: Responsive design for mobile access
- **AI-Powered Anomaly Detection**: ML-based alert threshold tuning
- **Multi-Environment Support**: Dev/staging/prod configurations

## Critical Monitoring Points

### Data Pipeline Health

- **Data Freshness**: Monitor staleness across all data sources
- **Quality Metrics**: Track missing values, outliers, and validation failures
- **Load Performance**: Monitor data loading latency and throughput
- **Cache Effectiveness**: Ensure high cache hit ratios for performance
- **Pipeline Execution**: Monitor daily runs, backfills, and real-time streaming
- **FeatureStore Integration**: Track feature computation and persistence
- **API Health**: Monitor external API usage and data download metrics

### Model Performance

- **Accuracy Tracking**: Continuous monitoring of model accuracy
- **Latency Monitoring**: Ensure inference times meet SLA requirements
- **Distribution Shift**: Detect changes in input/output distributions
- **Resource Utilization**: Monitor memory and CPU usage by models

### System Infrastructure

- **Container Health**: Monitor all service containers
- **Resource Limits**: Track memory, CPU, and disk usage
- **Network Connectivity**: Ensure communication between components
- **Storage Growth**: Monitor disk usage and plan for capacity

### Alert Response

- **Mean Time to Detection**: Target < 5 minutes for critical issues
- **False Positive Rate**: Keep < 5% to maintain team confidence
- **Escalation Effectiveness**: Ensure proper routing to responsible teams
- **Resolution Tracking**: Monitor mean time to resolution

## Integration with Nautilus Components

### Core Data Model Compliance

- All metrics include `instrument_id`, `ts_event`, and `ts_init`
- Timestamps expressed in nanoseconds since epoch
- Uses Nautilus domain types from `nautilus_trader.model.identifiers`
- Compatible with existing catalog and data infrastructure

### Centralized Metrics Architecture

The monitoring system uses a centralized metrics approach to avoid duplication and registration conflicts:

- **ml/common/metrics.py**: Central definition of all Prometheus metrics
- **Helper functions**: `record_pipeline_event()` and `update_pipeline_health()` for consistent labeling
- **Import pattern**: All components import metrics from this central location
- **No duplication**: Metrics are defined once and reused across all collectors

### SQL Health Views

Pipeline health monitoring is implemented through SQL views (located in ml/stores/migrations/005_views.sql):

- **ml.pipeline_health**: Overall pipeline health with daily aggregates
- **ml.data_collection_stats**: Data collection metrics per instrument
- **ml.feature_computation_stats**: Feature computation performance
- **ml.data_freshness**: Monitor data staleness per instrument
- **ml.error_summary**: Summary of errors across components
- **ml.model_performance_summary**: Model inference performance
- **ml.strategy_signal_summary**: Strategy signal generation metrics
- **ml.pipeline_processing_summary**: Processing statistics by stage
- **ml.data_quality_metrics**: Comprehensive quality tracking

These views provide:
- Nanosecond timestamp support with helper functions
- Health scoring algorithms
- Performance percentiles (P95, P99)
- Quality metrics and error tracking
- Real-time freshness monitoring

### ML Infrastructure Integration

- **FeatureStore**: Metrics on feature persistence and retrieval
- **ModelStore**: Tracking prediction storage and model performance
- **StrategyStore**: Monitoring strategy execution and trading decisions
- **Registry System**: Integration with feature, model, and strategy registries

### Actor System Integration

- **BaseMLInferenceActor**: Built-in metrics collection for custom actors
- **MLSignalActor**: Signal generation metrics and performance tracking
- **MLTradingStrategy**: Trading strategy execution and performance metrics
- **Event Processing**: Integration with Nautilus event system

This monitoring infrastructure provides the foundation for production-ready ML system observability in Nautilus Trader, ensuring reliable operation and early detection of issues across the entire ML pipeline.

## Key Implementation Files

### Core Components
- **ml/monitoring/__init__.py**: Package exports (MLMetricsCollector, MetricsServer, MonitoringConfig)
- **ml/monitoring/collector.py**: Main MLMetricsCollector implementation
- **ml/monitoring/_config.py**: MonitoringConfig dataclass
- **ml/monitoring/server.py**: HTTP metrics server for Prometheus scraping
- **ml/common/metrics.py**: Centralized Prometheus metrics definitions

### Specialized Collectors
- **ml/monitoring/collectors/base.py**: BaseMetricsCollector abstract class
- **ml/monitoring/collectors/data.py**: DataQualityCollector
- **ml/monitoring/collectors/features.py**: FeatureEngineeringCollector
- **ml/monitoring/collectors/model.py**: ModelLifecycleCollector
- **ml/monitoring/collectors/performance.py**: PerformanceDegradationCollector
- **ml/monitoring/collectors/resources.py**: ResourceUtilizationCollector
- **ml/monitoring/collectors/registry.py**: RegistryHealthCollector

### Dashboards and Visualization
- **ml/monitoring/realtime_dashboard.py**: Terminal-based real-time dashboard
- **ml/monitoring/dashboard_factory.py**: Programmatic dashboard generation
- **ml/monitoring/grafana_client.py**: Grafana API client
- **ml/monitoring/grafana/dashboards/**: JSON dashboard definitions

### Health Monitoring
- **ml/scripts/check_pipeline_health.py**: Pipeline health check script
- **ml/stores/migrations/005_views.sql**: SQL health monitoring views
- **ml/schema/pipeline_health.sql**: Reference SQL (migrated to migrations)

### Configuration
- **ml/monitoring/docker-compose.yml**: Docker stack configuration
- **ml/monitoring/prometheus/prometheus.yml**: Prometheus configuration
- **ml/monitoring/alertmanager/alertmanager.yml**: Alert routing configuration
- **ml/monitoring/prometheus/alerts/**: Alert rule definitions

## Cross-Module References

- **Data Pipeline**: See `context_data.md` for data ingestion and collection
- **Feature Engineering**: See `context_features.md` for feature computation
- **Stores**: See `context_stores.md` for persistence layer
- **Training**: See `context_training.md` for model training pipelines
- **Registry**: See `context_registry.md` for lifecycle management
- **Strategies**: See `context_strategies.md` for trading strategy framework
- **Deployment**: See `context_deployment.md` for containerization
- **Actors**: See `context_actors.md` for inference actors
- **Models**: See `context_models.md` for model implementations
### Health Summary

- `MLIntegrationManager.aggregate_health()` aggregates per-component health into domain and system summaries using the universal protocol.
- CLI utility: `python -m ml.cli.health [--db-connection <url>] [--strict]` prints a JSON summary suitable for dashboards or checks.
