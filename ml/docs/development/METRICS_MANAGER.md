# MetricsManager HOWTO

This guide standardizes how to acquire and use metrics in Nautilus Trader ML without importing `prometheus_client` directly from domain code.

Overview
- `ml/common/metrics_manager.py` provides a small, typed facade over `ml/common/metrics_bootstrap.py`.
- It ensures idempotent collector creation and hides Prometheus client details.
- Use it in non‑hot code paths and service initialization. Keep emit operations (`inc/observe/set`) minimal in hot loops.

When to use what
- Use `MetricsManager` when you want a consistent, terse pattern to create and store metric objects:
  - Good for module/class initialization and service code (actors, strategies, monitoring, pipelines).
- Use `metrics_bootstrap` directly only when you need explicit factory control in low‑level utilities under `ml/common/*`.
- Never import `prometheus_client` directly outside `ml/common/*`.

Patterns
- Create metrics once at import/init; then only call emit operations in loops.
- Preserve existing metric names and label sets to avoid breaking dashboards.
- Keep labels centralized via `ml.config.names` where applicable.

Examples
- Module/class init (preferred):

```python
from ml.common.metrics_manager import MetricsManager
from ml.config.names import METRIC_PREDICTIONS_TOTAL, LABEL_ACTOR_ID, LABEL_MODEL_NAME

_MM = MetricsManager.default()
PREDICTIONS = _MM.counter(METRIC_PREDICTIONS_TOTAL, "Total predictions", [LABEL_ACTOR_ID, LABEL_MODEL_NAME])
LATENCY = _MM.histogram("ml_prediction_latency_seconds", "Inference latency", [LABEL_ACTOR_ID, LABEL_MODEL_NAME])
```

- Emitting inside a loop (hot path):

```python
# Avoid allocations: reuse metric objects; keep labels low-cardinality
PREDICTIONS.labels(actor_id=actor_id, model_name=model_id).inc()
LATENCY.labels(actor_id=actor_id, model_name=model_id).observe(duration)
```

- Convenience helpers (non‑hot path only):

```python
# For ad‑hoc increments outside hot loops
mm = MetricsManager.default()
mm.inc("nautilus_ml_build_runner_runs_total", "Build tasks executed", labels={"status": "success"})
```

Hot‑Path Guidance
- No DataFrame creation, file I/O, or network calls.
- No new allocations in tight loops; prebind labels if needed and reuse collector children.
- Wrap bus publishes in try/except; defer persistence off the hot path.

Validation
- Types: `uv run --active --no-sync mypy ml --strict`
- Lint: `make ruff`
- Metrics: `make validate-metrics`
- Events: `make validate-events`

Notes
- The wrapper delegates to `metrics_bootstrap`, which caches collectors by name+labels and gracefully handles Prometheus availability.
- Tests may import `prometheus_client` directly; production/domain code should not.
