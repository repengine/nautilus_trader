"""
Centralized Prometheus metrics for the ML system.

This module defines all metrics once via the metrics bootstrap to avoid duplicate
registration and direct prometheus_client imports. Callers should import metrics from
here rather than instantiating their own collectors.

"""

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram


# ============================================================================
# DATA PIPELINE METRICS
# ============================================================================

# Event tracking across all pipeline stages
data_events_total = get_counter(
    "nautilus_ml_data_events_total",
    "Total data events processed by stage",
    ["dataset_type", "component", "stage", "source", "status"],
)

# Watermark lag tracking
watermark_lag_seconds = get_gauge(
    "nautilus_ml_watermark_lag_seconds",
    "Lag in seconds since last successful processing",
    ["dataset", "instrument", "source"],
)

# Stage coverage percentage
stage_coverage_pct = get_gauge(
    "nautilus_ml_stage_coverage_pct",
    "Coverage percentage between pipeline stages",
    ["dataset", "from_stage", "to_stage"],
)

# Contract violations
contract_violations_total = get_counter(
    "nautilus_ml_contract_violations_total",
    "Total contract validation violations",
    ["dataset", "rule"],
)

# ============================================================================
# DATA COLLECTION METRICS
# ============================================================================

data_collection_duration = get_histogram(
    "nautilus_ml_data_collection_duration_seconds",
    "Duration of data collection operations",
    ["source", "schema"],
)

data_collection_errors_total = get_counter(
    "nautilus_ml_data_collection_errors",
    "Total data collection errors",
    ["source", "instrument", "error_type"],
)

catalog_write_operations_total = get_counter(
    "nautilus_ml_catalog_write_operations",
    "Total catalog write operations",
    ["status"],
)

# ============================================================================
# FEATURE STORE METRICS
# ============================================================================

feature_store_operations_total = get_counter(
    "nautilus_ml_feature_store_operations",
    "Total feature store operations",
    ["operation", "status"],
)

feature_computation_duration = get_histogram(
    "nautilus_ml_feature_computation_duration_seconds",
    "Duration of feature computation",
    ["feature_set", "mode"],  # mode: batch or realtime
)

feature_drift_score = get_gauge(
    "nautilus_ml_feature_drift_score",
    "Feature drift score (0-1)",
    ["feature_set", "feature_name"],
)

# ============================================================================
# MODEL STORE METRICS
# ============================================================================

model_store_operations_total = get_counter(
    "nautilus_ml_model_store_operations",
    "Total model store operations",
    ["operation", "status"],
)

model_inference_duration = get_histogram(
    "nautilus_ml_model_inference_duration_seconds",
    "Duration of model inference",
    ["model_id", "version"],
)

model_accuracy = get_gauge(
    "nautilus_ml_model_accuracy",
    "Model accuracy score",
    ["model_id", "version"],
)

model_confidence = get_gauge(
    "nautilus_ml_model_confidence",
    "Average model confidence score",
    ["model_id", "version"],
)

# Backwards-compatibility aliases expected by some tests
# Timer aliases (histograms)
MODEL_INFERENCE_TIMER = model_inference_duration
FEATURE_CALCULATION_TIMER = feature_computation_duration


# Prediction counter (compat alias for tests) – avoid duplicate registration
class _ProxyMetric:
    def labels(self, **kwargs: object) -> "_ProxyMetric":
        return self

    def inc(self, _amount: float = 1.0, **kwargs: object) -> None:
        return None

    def observe(self, _amount: float = 0.0, **kwargs: object) -> None:
        return None


PREDICTION_COUNTER = _ProxyMetric()

# ============================================================================
# STRATEGY STORE METRICS
# ============================================================================

strategy_store_operations_total = get_counter(
    "nautilus_ml_strategy_store_operations",
    "Total strategy store operations",
    ["operation", "status"],
)

strategy_signal_generation_duration = get_histogram(
    "nautilus_ml_strategy_signal_generation_duration_seconds",
    "Duration of signal generation",
    ["strategy_id"],
)

strategy_pnl = get_gauge(
    "nautilus_ml_strategy_pnl",
    "Strategy P&L",
    ["strategy_id", "timeframe"],
)

# ============================================================================
# VALIDATION METRICS
# ============================================================================

validation_violations_counter = get_counter(
    "nautilus_ml_validation_violations",
    "Data validation violations by type and severity",
    ["dataset_id", "rule_type", "severity"],
)

validation_duration_histogram = get_histogram(
    "nautilus_ml_validation_duration_seconds",
    "Time spent on data validation",
    ["dataset_id"],
)

schema_mismatch_counter = get_counter(
    "nautilus_ml_schema_mismatches",
    "Schema validation failures",
    ["dataset", "mismatch_type"],
)

write_rejection_counter = get_counter(
    "nautilus_ml_write_rejections",
    "Writes rejected due to validation failures",
    ["dataset_id", "reason"],
)

quality_score_histogram = get_histogram(
    "nautilus_ml_data_quality_score",
    "Distribution of data quality scores",
    ["dataset_id"],
    buckets=(0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

# ============================================================================
# SYSTEM HEALTH METRICS
# ============================================================================

pipeline_health = get_gauge(
    "nautilus_ml_pipeline_health",
    "Overall pipeline health score (0-1)",
    ["component"],
)

system_ready = get_gauge(
    "nautilus_ml_system_ready",
    "System readiness status (0=not ready, 1=ready)",
    ["component"],
)

# ============================================================================
# BACKPRESSURE & CIRCUIT BREAKER METRICS
# ============================================================================

# Backpressure drops (e.g., throttled, queue_full)
backpressure_drops_total = get_counter(
    "nautilus_ml_backpressure_drops_total",
    "Total events dropped due to backpressure",
    ["component", "reason"],
)

# Optional queue depth gauge for actor-side bridge
backpressure_queue_depth = get_gauge(
    "nautilus_ml_backpressure_queue_depth",
    "Current depth of actor-side domain event queue",
    ["component"],
)

# Circuit breaker state (0=closed, 0.5=half_open, 1=open)
circuit_breaker_state = get_gauge(
    "nautilus_ml_circuit_breaker_state",
    "Circuit breaker state (0=closed, 0.5=half_open, 1=open)",
    ["component"],
)

# Circuit breaker transitions counter
circuit_breaker_trips_total = get_counter(
    "nautilus_ml_circuit_breaker_trips_total",
    "Total circuit breaker transitions",
    ["component", "to_state"],
)

# ============================================================================
# CONSUMER / AGGREGATOR METRICS
# ============================================================================

aggregator_buffer_size = get_gauge(
    "nautilus_ml_aggregator_buffer_size",
    "Current buffered envelope count per instrument",
    ["instrument"],
)

aggregator_duplicates_total = get_counter(
    "nautilus_ml_aggregator_duplicates_total",
    "Total duplicate envelopes dropped by id",
    [],
)

aggregator_flushed_total = get_counter(
    "nautilus_ml_aggregator_flushed_total",
    "Total envelopes flushed after watermark gating",
    ["instrument"],
)

aggregator_watermark_lag_seconds = get_gauge(
    "nautilus_ml_aggregator_watermark_lag_seconds",
    "Watermark lag (watermark_ns - last_flushed_ts) per instrument",
    ["instrument"],
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def record_pipeline_event(
    dataset_type: str,
    component: str,
    stage: str,
    source: str = "unknown",
    status: str = "success",
    count: int = 1,
) -> None:
    """
    Record a pipeline event with consistent labeling.

    Parameters
    ----------
    dataset_type : str
        Dataset type (bars, features, predictions, signals)
    component : str
        Component identifier (e.g., feature_set_id, model_id, strategy_id or schema)
    stage : str
        Pipeline stage (CATALOG_WRITTEN, FEATURE_COMPUTED, etc.)
    source : str
        Data source (live, historical, backfill)
    status : str
        Event status (success, failure)
    count : int
        Number of events to record

    """
    data_events_total.labels(
        dataset_type=dataset_type,
        component=component,
        stage=stage,
        source=source,
        status=status,
    ).inc(count)


def update_pipeline_health(component: str, score: float) -> None:
    """
    Update pipeline health score.

    Parameters
    ----------
    component : str
        Component name (data, features, models, strategies)
    score : float
        Health score between 0 and 1

    """
    pipeline_health.labels(component=component).set(score)


__all__ = [
    "FEATURE_CALCULATION_TIMER",
    "MODEL_INFERENCE_TIMER",
    "PREDICTION_COUNTER",
    "aggregator_buffer_size",
    "aggregator_duplicates_total",
    "aggregator_flushed_total",
    "aggregator_watermark_lag_seconds",
    "catalog_write_operations_total",
    "contract_violations_total",
    "data_collection_duration",
    "data_collection_errors_total",
    "data_events_total",
    "feature_computation_duration",
    "feature_drift_score",
    "feature_store_operations_total",
    "model_accuracy",
    "model_confidence",
    "model_inference_duration",
    "model_store_operations_total",
    "pipeline_health",
    "quality_score_histogram",
    "record_pipeline_event",
    "schema_mismatch_counter",
    "stage_coverage_pct",
    "strategy_pnl",
    "strategy_signal_generation_duration",
    "strategy_store_operations_total",
    "system_ready",
    "update_pipeline_health",
    "validation_duration_histogram",
    "validation_violations_counter",
    "watermark_lag_seconds",
    "write_rejection_counter",
]
