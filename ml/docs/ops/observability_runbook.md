# Observability Runbook (Ops)

This runbook summarizes operational procedures for the ML Observability pipeline.

## Sink selection

- `file` sink: Writes JSONL/CSV under `ML_OBS_BASE_PATH` (default `./observability`). Best for local dev or batch ingestion.
- `db` sink: Writes to SQL via `ML_OBS_DB_URL` (SQLite, Postgres). Suitable for dashboards and downstream SQL queries.

## Bootstrapping

- Auto-start via env: set `ML_OBS_*` vars and ensure app calls `auto_start_if_configured(mgr)`. Container entrypoints already do this.
- Manual start: use `ml.cli.observability start` for ad-hoc sessions.

### Async Worker (advanced)

For high-throughput scenarios, use the async worker to enqueue observability rows off the hot path and persist in the background:

```python
from pathlib import Path
from ml.observability.service import ObservabilityService
from ml.observability.async_worker import ObservabilityAsyncWorker

svc = ObservabilityService()
worker = ObservabilityAsyncWorker(
    service=svc,
    sink="file",                 # or "db"
    base_path=Path("./observability"),
    db_connection_string=None,
    flush_interval_seconds=5.0,
    queue_maxsize=4096,
)

# Start background task
worker.start()

# Enqueue small items cheaply (hot path safe)
worker.enqueue_latency(
    correlation_id="c1",
    instrument_id="EURUSD.SIM",
    pipeline_stage="FEATURE_COMPUTED",
    ts_stage_start=1,
    ts_stage_end=2,
)

# On shutdown
import asyncio
asyncio.run(worker.stop(drain=True))
```

Metrics: `nautilus_ml_observability_enqueued_total`, `nautilus_ml_observability_queue_depth`, and `nautilus_ml_backpressure_drops_total{component="obs_async_worker"}`. Dashboard panels exist under the “Observability” row.

## Health and alerts

- Prometheus: ensure metric scraping of ML processes (see `ml/common/metrics_bootstrap.py`).
- Alert rules: sample rules are provided in `ml/deployment/alerts.yml`.
  - Latency: `MLModelInferenceLatencyHighP99` triggers when P99 inference latency >200ms for 5m.
  - Health: `MLPipelineHealthLow` triggers when `nautilus_ml_pipeline_health` < 0.8 for 5m.
  - To enable, add to Prometheus config:
    - `rule_files:` section including `/etc/prometheus/alerts.yml` and mount `ml/deployment/alerts.yml` into the container.
- Grafana: dashboards for latency watermarks, component health, and correlation summaries.
  - Seed dashboard JSON: `ml/deployment/grafana/ml_pipeline_health.json`
  - Quick import via Makefile (requires a local Grafana at <http://localhost:3000> and an API token):
    - `make -C ml/deployment grafana-import GRAFANA_API_TOKEN=<<your_token>>`
  - Manual import: Grafana UI → Dashboards → Import → Upload `ml_pipeline_health.json`

## Common errors

- Misconfigured DB URL: background flusher logs errors but remains off hot path; correct `ML_OBS_DB_URL`.
- Permission issues writing to path: set `ML_OBS_BASE_PATH` to a writable directory; check container volume mounts.
- Excessive file growth: use JSONL rotation/compaction (planned) or switch to DB sink.

## Validation

- Contract tests validate Pandera schemas for in-memory and persisted tables.
- Quick checks: run `uv run -m ml.cli.observability flush-jsonl --seed-sample` and inspect outputs.
- Persisted JSONL contract: `ml/tests/contracts/test_observability_persisted_schemas.py` validates JSONL against Pandera models.

## Troubleshooting

- Enable more logging via `LOG_LEVEL=DEBUG`.
- For background flusher issues, run a single-shot flush with `interval=0` using the CLI to isolate persistence problems.
