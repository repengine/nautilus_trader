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

# Default histogram buckets for signal actor
SIGNAL_LATENCY_BUCKETS = [0.0001, 0.0005, 0.001, 0.002, 0.005]
FEATURE_TIME_BUCKETS = [0.00005, 0.0001, 0.0005, 0.001, 0.002]

__all__ = [
    # labels
    "LABEL_ACTOR_ID",
    "LABEL_MODEL_NAME",
    "LABEL_STRATEGY",
    "LABEL_FEATURE_SET_ID",
    "LABEL_REGIME",
    "LABEL_SIGNAL_TYPE",
    # actor metrics
    "METRIC_PREDICTIONS_TOTAL",
    "METRIC_PREDICTION_LATENCY_SECONDS",
    "METRIC_SIGNAL_CONFIDENCE",
    # signal metrics
    "METRIC_PREDICTION_DISTRIBUTION",
    "METRIC_CONFIDENCE_DISTRIBUTION",
    "METRIC_SIGNAL_GENERATION_SECONDS",
    "METRIC_FEATURE_TIME_BY_SET_SECONDS",
    "METRIC_SIGNALS_GENERATED_TOTAL",
    "METRIC_ADAPTIVE_THRESHOLD",
    "METRIC_MARKET_REGIME_TOTAL",
    # buckets
    "SIGNAL_LATENCY_BUCKETS",
    "FEATURE_TIME_BUCKETS",
    # onnx input name
    "ONNX_INPUT_NAME",
    # strategy metrics
    "METRIC_SIGNALS_RECEIVED_TOTAL",
    "METRIC_TRADES_EXECUTED_TOTAL",
    "METRIC_SIGNAL_TO_TRADE_LATENCY_SECONDS",
    "METRIC_POSITION_COUNT",
    # strategy labels
    "LABEL_STRATEGY_ID",
    "LABEL_SIGNAL_SOURCE",
    "LABEL_ORDER_SIDE",
    "LABEL_INSTRUMENT",
    # modes
    "MODE_BATCH",
    "MODE_ONLINE",
]

# ONNX input tensor name
ONNX_INPUT_NAME = "input"

# Strategy metrics
METRIC_SIGNALS_RECEIVED_TOTAL = "nautilus_ml_signals_received_total"
METRIC_TRADES_EXECUTED_TOTAL = "nautilus_ml_trades_executed_total"
METRIC_SIGNAL_TO_TRADE_LATENCY_SECONDS = "nautilus_ml_signal_to_trade_latency_seconds"
METRIC_POSITION_COUNT = "nautilus_ml_position_count"

# Strategy label keys
LABEL_STRATEGY_ID = "strategy_id"
LABEL_SIGNAL_SOURCE = "signal_source"
LABEL_ORDER_SIDE = "order_side"
LABEL_INSTRUMENT = "instrument"

# Modes
MODE_BATCH = "batch"
MODE_ONLINE = "online"
