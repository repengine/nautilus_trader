"""
Canonical names and labels used across ML components (metrics, registry keys).

Defines a single source of truth for Prometheus metric names, label keys, and other
shared identifiers to avoid string drift.

"""

from __future__ import annotations


# Metric label keys
LABEL_ACTOR_ID = "actor_id"
LABEL_MODEL_NAME = "model_name"
LABEL_STRATEGY = "strategy"
LABEL_FEATURE_SET_ID = "feature_set_id"
LABEL_REGIME = "regime"
LABEL_SIGNAL_TYPE = "signal_type"
LABEL_ACTION = "action"
LABEL_REASON = "reason"

# Base actor metrics
METRIC_PREDICTIONS_TOTAL = "nautilus_ml_predictions_total"
METRIC_PREDICTION_LATENCY_SECONDS = "nautilus_ml_prediction_latency_seconds"
METRIC_SIGNAL_CONFIDENCE = "nautilus_ml_signal_confidence"

# Signal actor metrics
METRIC_PREDICTION_DISTRIBUTION = "nautilus_ml_prediction_distribution"
METRIC_CONFIDENCE_DISTRIBUTION = "nautilus_ml_confidence_distribution"
METRIC_SIGNAL_GENERATION_SECONDS = "nautilus_ml_signal_generation_seconds"
METRIC_FEATURE_TIME_BY_SET_SECONDS = "nautilus_ml_feature_time_by_feature_set_seconds"
METRIC_SIGNALS_GENERATED_TOTAL = "nautilus_ml_signals_generated_total"
METRIC_ADAPTIVE_THRESHOLD = "nautilus_ml_adaptive_threshold"
METRIC_MARKET_REGIME_TOTAL = "nautilus_ml_market_regime_total"

# Static-audit remediation metrics (PR-01 scaffolding only)
METRIC_CAUSALITY_MONOTONIC_VIOLATIONS_TOTAL = (
    "nautilus_ml_causality_monotonic_violations_total"
)
METRIC_INFERENCE_DEADLINE_TIMEOUTS_TOTAL = "nautilus_ml_inference_deadline_timeouts_total"
METRIC_DRIFT_POLICY_ACTIONS_TOTAL = "nautilus_ml_drift_policy_actions_total"
METRIC_ML_FAILURE_ACTIONS_TOTAL = "nautilus_ml_failure_actions_total"
METRIC_REGISTRY_COMPATIBILITY_MIGRATION_BYPASS_TOTAL = (
    "nautilus_ml_registry_compatibility_migration_bypass_total"
)
METRIC_REGISTRY_UNSIGNED_ARTIFACT_OVERRIDE_TOTAL = (
    "nautilus_ml_registry_unsigned_artifact_override_total"
)

# Default histogram buckets for signal actor
SIGNAL_LATENCY_BUCKETS = [0.0001, 0.0005, 0.001, 0.002, 0.005]
FEATURE_TIME_BUCKETS = [0.00005, 0.0001, 0.0005, 0.001, 0.002]

__all__ = [
    "FEATURE_TIME_BUCKETS",
    "LABEL_ACTION",
    "LABEL_ACTOR_ID",
    "LABEL_FEATURE_SET_ID",
    "LABEL_INSTRUMENT",
    "LABEL_MODEL_NAME",
    "LABEL_ORDER_SIDE",
    "LABEL_REASON",
    "LABEL_REGIME",
    "LABEL_SIGNAL_SOURCE",
    "LABEL_SIGNAL_TYPE",
    "LABEL_STRATEGY",
    "LABEL_STRATEGY_ID",
    "METRIC_ADAPTIVE_THRESHOLD",
    "METRIC_CAUSALITY_MONOTONIC_VIOLATIONS_TOTAL",
    "METRIC_CONFIDENCE_DISTRIBUTION",
    "METRIC_DRIFT_POLICY_ACTIONS_TOTAL",
    "METRIC_FEATURE_TIME_BY_SET_SECONDS",
    "METRIC_INFERENCE_DEADLINE_TIMEOUTS_TOTAL",
    "METRIC_MARKET_REGIME_TOTAL",
    "METRIC_ML_FAILURE_ACTIONS_TOTAL",
    "METRIC_POSITION_COUNT",
    "METRIC_PREDICTION_DISTRIBUTION",
    "METRIC_PREDICTION_LATENCY_SECONDS",
    "METRIC_REGISTRY_COMPATIBILITY_MIGRATION_BYPASS_TOTAL",
    "METRIC_REGISTRY_UNSIGNED_ARTIFACT_OVERRIDE_TOTAL",
    "METRIC_SIGNALS_GENERATED_TOTAL",
    "METRIC_SIGNALS_RECEIVED_TOTAL",
    "METRIC_SIGNAL_CONFIDENCE",
    "METRIC_SIGNAL_GENERATION_SECONDS",
    "METRIC_SIGNAL_TO_TRADE_LATENCY_SECONDS",
    "METRIC_TRADES_EXECUTED_TOTAL",
    "MODE_BATCH",
    "MODE_ONLINE",
    "ONNX_INPUT_NAME",
    "SIGNAL_LATENCY_BUCKETS",
]

# ONNX input tensor name
ONNX_INPUT_NAME = "input"

# Strategy metrics
METRIC_SIGNALS_RECEIVED_TOTAL = "nautilus_ml_signals_received_total"
METRIC_TRADES_EXECUTED_TOTAL = "nautilus_ml_trades_executed_total"
METRIC_SIGNAL_TO_TRADE_LATENCY_SECONDS = "nautilus_ml_signal_to_trade_latency_seconds"
METRIC_POSITION_COUNT = "nautilus_ml_position_count"
METRIC_STRATEGY_DECISIONS_PERSISTED_TOTAL = "nautilus_ml_strategy_decisions_persisted_total"
METRIC_STRATEGY_STORE_WRITE_LATENCY_SECONDS = "nautilus_ml_strategy_store_write_latency_seconds"
METRIC_STRATEGY_STORE_BATCH_SIZE = "nautilus_ml_strategy_store_batch_size"

# Strategy label keys
LABEL_STRATEGY_ID = "strategy_id"
LABEL_SIGNAL_SOURCE = "signal_source"
LABEL_ORDER_SIDE = "order_side"
LABEL_INSTRUMENT = "instrument"

# Modes
MODE_BATCH = "batch"
MODE_ONLINE = "online"
