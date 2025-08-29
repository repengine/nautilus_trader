# Performance Targets and Metrics

This document defines the baseline performance targets and metric names used by the
ML actors and strategies.

Targets (guidance, verify in your environment):
- Feature computation (p99): < 0.5 ms
- Model inference (p99): < 2.0 ms
- End-to-end signal generation (p99): < 5.0 ms

Key metrics (Prometheus):
- `nautilus_ml_prediction_latency_seconds` (actor_id, model_name)
- `nautilus_ml_feature_time_by_set_seconds` (actor_id, feature_set_id)
- `nautilus_ml_signal_generation_seconds` (actor_id, strategy)

Implementation notes:
- The actor records feature time and inference time separately via `PerformanceMonitor`.
- All metrics are created idempotently via `ml.common.metrics_bootstrap` to avoid double registration.

