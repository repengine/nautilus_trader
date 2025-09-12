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
from ml.config.events import Stage

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
    pipeline_stage=Stage.FEATURE_COMPUTED.value,
    ts_stage_start=1,
    ts_stage_end=2,
)

# On shutdown
import asyncio
asyncio.run(worker.stop(drain=True))
```

Metrics: `nautilus_ml_observability_enqueued_total`, `nautilus_ml_observability_queue_depth`, and `nautilus_ml_backpressure_drops_total{component="obs_async_worker"}`. Dashboard panels exist under the “Observability” row.

#### Alerts and thresholds (recommended)

- Backpressure drops (warning): any increase over 5m
  - PromQL: `increase(nautilus_ml_backpressure_drops_total[5m]) > 0`
- Backpressure drops (critical): sustained rate > 0.5 drops/sec for 10m
  - PromQL: `rate(nautilus_ml_backpressure_drops_total[5m]) > 0.5`
- Queue depth high (warning): > 75% capacity for 10m
  - PromQL: `nautilus_ml_observability_queue_depth{component="obs_async_worker"} > 3072`
- Async flush latency high (warning): P99 > 500ms for 10m
  - PromQL:
    `histogram_quantile(0.99, sum(rate(nautilus_ml_observability_async_flush_duration_seconds_bucket[5m])) by (le)) > 0.5`

Tuning guidance:

- If queue depth is consistently > 75% but drops are zero, consider increasing `queue_maxsize` and/or flush frequency.
- If drops occur with low flush latency, ingestion rate is exceeding capacity—reduce producer rate or add sampling.
- If flush latency is high, optimize sink (e.g., switch to async DB, increase batch size) and verify storage IOPS.

## Health and alerts

- Prometheus: ensure metric scraping of ML processes (see `ml/common/metrics_bootstrap.py`).
- Alert rules: sample rules are provided in `ml/deployment/alerts.yml`.
  - Latency: `MLModelInferenceLatencyHighP99` triggers when P99 inference latency >200ms for 5m.
  - Health: `MLPipelineHealthLow` triggers when `nautilus_ml_pipeline_health` < 0.8 for 5m.
  - Backpressure: `MLObsAsyncBackpressureDrops` (warning) and `MLObsAsyncBackpressureSustained` (critical).
  - Queue depth: `MLObsAsyncQueueDepthHigh` (warning).
  - Flush latency: `MLObsAsyncFlushLatencyHighP99` (warning).
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
- Async quick check: `uv run -m ml.cli.observability start --async --duration 5 --seed-sample` then `uv run -m ml.cli.observability status`.

## Ingestion Runbook

Symptoms and checks

- Watermark lag high (e.g., > 5 minutes):
  - Check ingestion rate panel; if near zero, source may be stalled.
  - Verify provider health/rate limits; consult provider status page.
  - Ensure scheduler is active; check logs for backoff/retry messages.
- Ingestion errors increasing:
  - Inspect error types (rate_limit, connection, parse_error); adjust backoff.
  - Verify credentials/API key validity and network reachability.
  - For persistent parse errors, validate schema and recent provider changes.
- Aggregator buffer growing:
  - Confirm downstream consumers are running; inspect aggregator watermark lag.
  - Advance watermark if conditions permit or reduce batch sizes.

Metrics and alerts

- Rate: `sum by (dataset_type)(rate(nautilus_ml_data_events_total{stage="INGESTED"}[5m]))`
- Lag: `max(nautilus_ml_watermark_lag_seconds)` and `nautilus_ml_aggregator_watermark_lag_seconds`
- Errors: `sum by (error_type)(rate(nautilus_ml_data_collection_errors_total[5m]))`
- Alerts: `MLIngestErrorsHigh`, `MLIngestWatermarkLagHigh`, `MLIngestRateDrop`, `MLAggregatorBufferHigh`, `MLAggregatorDuplicatesHigh`, `MLAggregatorWatermarkLagHigh`

Playbook actions

- Reduce ingest concurrency; increase backoff on rate limits.
- Enable/verify async worker; increase queue capacity while monitoring backpressure.
- Use fixtures to reproduce: `ml.data.fixtures.make_tbbo_fixture` yields deterministic inputs to test ingestion/aggregation locally.
- Persisted JSONL contract: `ml/tests/contracts/test_observability_persisted_schemas.py` validates JSONL against Pandera models.

## Troubleshooting

- Enable more logging via `LOG_LEVEL=DEBUG`.
- For background flusher issues, run a single-shot flush with `interval=0` using the CLI to isolate persistence problems.
