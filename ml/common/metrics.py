"""
Centralized Prometheus metrics for the ML system.

This module defines all metrics once to avoid duplication and registration conflicts.
All components should import metrics from here rather than defining their own.
"""

from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram


# ============================================================================
# DATA PIPELINE METRICS
# ============================================================================

# Event tracking across all pipeline stages
data_events_total = Counter(
    "nautilus_ml_data_events_total",
    "Total data events processed by stage",
    ["dataset_type", "component", "stage", "source", "status"],
)

# Watermark lag tracking
watermark_lag_seconds = Gauge(
    "nautilus_ml_watermark_lag_seconds",
    "Lag in seconds since last successful processing",
    ["dataset", "instrument", "source"],
)

# Stage coverage percentage
stage_coverage_pct = Gauge(
    "nautilus_ml_stage_coverage_pct",
    "Coverage percentage between pipeline stages",
    ["dataset", "from_stage", "to_stage"],
)

# Contract violations
contract_violations_total = Counter(
    "nautilus_ml_contract_violations_total",
    "Total contract validation violations",
    ["dataset", "rule"],
)

# ============================================================================
# DATA COLLECTION METRICS
# ============================================================================

data_collection_duration = Histogram(
    "nautilus_ml_data_collection_duration_seconds",
    "Duration of data collection operations",
    ["source", "schema"],
)

data_collection_errors_total = Counter(
    "nautilus_ml_data_collection_errors",
    "Total data collection errors",
    ["source", "instrument", "error_type"],
)

catalog_write_operations_total = Counter(
    "nautilus_ml_catalog_write_operations",
    "Total catalog write operations",
    ["status"],
)

# ============================================================================
# FEATURE STORE METRICS
# ============================================================================

feature_store_operations_total = Counter(
    "nautilus_ml_feature_store_operations",
    "Total feature store operations",
    ["operation", "status"],
)

feature_computation_duration = Histogram(
    "nautilus_ml_feature_computation_duration_seconds",
    "Duration of feature computation",
    ["feature_set", "mode"],  # mode: batch or realtime
)

feature_drift_score = Gauge(
    "nautilus_ml_feature_drift_score",
    "Feature drift score (0-1)",
    ["feature_set", "feature_name"],
)

# ============================================================================
# MODEL STORE METRICS
# ============================================================================

model_store_operations_total = Counter(
    "nautilus_ml_model_store_operations",
    "Total model store operations",
    ["operation", "status"],
)

model_inference_duration = Histogram(
    "nautilus_ml_model_inference_duration_seconds",
    "Duration of model inference",
    ["model_id", "version"],
)

model_accuracy = Gauge(
    "nautilus_ml_model_accuracy",
    "Model accuracy score",
    ["model_id", "version"],
)

model_confidence = Gauge(
    "nautilus_ml_model_confidence",
    "Average model confidence score",
    ["model_id", "version"],
)

# ============================================================================
# STRATEGY STORE METRICS
# ============================================================================

strategy_store_operations_total = Counter(
    "nautilus_ml_strategy_store_operations",
    "Total strategy store operations",
    ["operation", "status"],
)

strategy_signal_generation_duration = Histogram(
    "nautilus_ml_strategy_signal_generation_duration_seconds",
    "Duration of signal generation",
    ["strategy_id"],
)

strategy_pnl = Gauge(
    "nautilus_ml_strategy_pnl",
    "Strategy P&L",
    ["strategy_id", "timeframe"],
)

# ============================================================================
# VALIDATION METRICS
# ============================================================================

validation_violations_counter = Counter(
    "nautilus_ml_validation_violations",
    "Data validation violations by type and severity",
    ["dataset_id", "rule_type", "severity"],
)

validation_duration_histogram = Histogram(
    "nautilus_ml_validation_duration_seconds",
    "Time spent on data validation",
    ["dataset_id"],
)

schema_mismatch_counter = Counter(
    "nautilus_ml_schema_mismatches",
    "Schema validation failures",
    ["dataset", "mismatch_type"],
)

write_rejection_counter = Counter(
    "nautilus_ml_write_rejections",
    "Writes rejected due to validation failures",
    ["dataset_id", "reason"],
)

quality_score_histogram = Histogram(
    "nautilus_ml_data_quality_score",
    "Distribution of data quality scores",
    ["dataset_id"],
    buckets=(0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

# ============================================================================
# SYSTEM HEALTH METRICS
# ============================================================================

pipeline_health = Gauge(
    "nautilus_ml_pipeline_health",
    "Overall pipeline health score (0-1)",
    ["component"],
)

system_ready = Gauge(
    "nautilus_ml_system_ready",
    "System readiness status (0=not ready, 1=ready)",
    ["component"],
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


__all__ = [  # noqa: RUF022
    # Core Prometheus types
    "Counter",
    "Gauge",
    "Histogram",
    # Event metrics
    "data_events_total",
    "watermark_lag_seconds",
    "stage_coverage_pct",
    "contract_violations_total",
    # Collection metrics
    "data_collection_duration",
    "data_collection_errors_total",
    "catalog_write_operations_total",
    # Feature metrics
    "feature_store_operations_total",
    "feature_computation_duration",
    "feature_drift_score",
    # Model metrics
    "model_store_operations_total",
    "model_inference_duration",
    "model_accuracy",
    "model_confidence",
    # Strategy metrics
    "strategy_store_operations_total",
    "strategy_signal_generation_duration",
    "strategy_pnl",
    # Validation metrics
    "validation_violations_counter",
    "validation_duration_histogram",
    "schema_mismatch_counter",
    "write_rejection_counter",
    "quality_score_histogram",
    # Health metrics
    "pipeline_health",
    "system_ready",
    # Helper functions
    "record_pipeline_event",
    "update_pipeline_health",
]
