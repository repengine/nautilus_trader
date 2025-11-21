# ML Dashboard Runbook (Cold Path)

## Overview

This runbook describes how to operate the ML dashboard control plane. The
service is cold-path only and exposes cached health, observability, and control
endpoints. All workflows rely on the existing infra components described in
`ml/docs/architecture/dashboard_control_plane.md`.

## Quick Start

- Launch with `python -m ml.dashboard.serve` or via the Docker Compose service
  defined in `ml/deployment/docker-compose.yml`.
- Ensure the following dependencies are reachable before exposing the UI:
  - Grafana (`GRAFANA_URL`) for provisioning and optional embeds.
  - Prometheus (`PROMETHEUS_URL`) for snapshot metrics.
  - Postgres (`ML_DB_CONNECTION`) when store summaries are enabled.
- When running under Docker Compose, keep the `streaming_persistence_worker`
  container healthy—this service materialises Redis streaming events into the
  shared state file (`/app/ml_out/streaming_training_state.json`) mounted at
  `ML_DASHBOARD_STREAMING_STATE_PATH`.
- Export `ML_DASHBOARD_USE_COMPOSE=0` when running outside Docker Compose. The
  service controllers fall back to `NoopServiceController` in this mode.

### Dashboard UI Modes

The dashboard supports three UI modes optimized for different use cases:

1. **Standard UI** (`http://localhost:8010/`)
   - Basic monitoring interface with service status and pipeline controls
   - Lightweight view for operational health checks
   - Minimal resource usage

2. **Enhanced UI** (`http://localhost:8010/?ui=enhanced`)
   - ML Pipeline Orchestrator interface with tabbed navigation
   - Advanced controls for data ingestion, dataset building, and model training
   - Comprehensive model and feature management

3. **Advanced Trading UI** (`http://localhost:8010/?ui=advanced`)
   - Professional trading command center with real-time monitoring
   - Live data ingestion visualization (bars/quotes/L2 metrics)
   - Model P&L tracking and portfolio analytics
   - Experiment tracking for hyperparameter optimization
   - Dark theme inspired by professional trading platforms
   - Chart.js integration for interactive data visualization

## Authentication Tokens

- Preferred configuration: set `ML_DASHBOARD_TOKENS` to a JSON array of
  `{ "value": str, "expires": iso8601 }` objects. Tokens are validated using
  `hmac.compare_digest` and expiry timestamps are honoured.
- Legacy fallback: `ML_DASHBOARD_TOKEN` (+ optional
  `ML_DASHBOARD_TOKEN_EXPIRES`) is still supported. Mixing both formats merges
  the token list.
- Rotate tokens by updating the environment variable and restarting the
  service. All validation outcomes are tracked via the
  `ml_dashboard_auth_validations_total{result="*"}` counter; monitor spikes in
  `invalid` to detect misuse.

## Observability Endpoints

| Endpoint | Purpose | Notes |
| --- | --- | --- |
| `/api/health/system` | Aggregated component health | Uses cached registry results when available. |
| `/api/observability/status` | Grafana provisioning status | Includes cached dashboard URL and embed metadata. |
| `/api/observability/summary` | Prometheus snapshot (request rate, latency P95, error counts) | Returns `{ "ok": false }` when Prometheus is unavailable; metrics recorded in `ml_dashboard_requests_total`. |
| `/api/observability/stores` | Store health summaries | Requires `ML_DB_CONNECTION` and enabled store health feature. |

### Store Health Triage

- The summary is sourced via `summarize_all_stores` (see
  `ml/dashboard/store_health.py`) and cached with TTL to avoid repeated DB
  hits.
- `healthy=False` with `fallback_active=True` indicates the store client or the
  database engine was unavailable. Inspect `_STORE_FALLBACK_TOTAL` metrics for
  detailed reasons.
- When `latest_event_ns` is null, investigate upstream pipelines; freshness is
  calculated per-store and for the top N datasets (`ML_DASHBOARD_STORE_TOP_DATASETS`).
- Disable the feature temporarily by exporting `ML_DASHBOARD_STORE_SUMMARY=0`.

### Grafana Provisioning

- Trigger manually via `POST /api/observability/grafana/provision`. Supply an
  optional `{"title": "custom name"}` payload to override the dashboard name.
- Provisioning is idempotent; cached success responses include `cached: true`.
- Metrics:
  - `ml_dashboard_requests_total{route="/api/observability/grafana/provision"}`
    for request outcomes.
  - `ml_dashboard_latency_seconds{route="/api/observability/grafana/provision"}`
    to monitor latency.
- For automated provisioning on boot set
  `ML_DASHBOARD_GRAFANA_PROVISION_ON_START=1`. Failures are captured in
  `_grafana_status` and surfaced via `/api/observability/status`.

### Prometheus Snapshots

- Configure `PROMETHEUS_URL` to point at the scrape API. Queries are executed
  via `PrometheusQueryHelper` with retry/backoff; timeouts are controlled by
  `ML_DASHBOARD_PROM_TIMEOUT`.
- Dashboard cards display the last snapshot value and fall back to a placeholder
  when the helper is disabled or returns `{ "ok": false }`.

### Pipeline Coverage Telemetry

- The ML pipeline `/health` endpoint now returns a `coverage` block with fields
  such as `last_run`, `last_success`, `buckets_total`,
  `buckets_restore_catalog`, `buckets_reingest_source`, `buckets_healthy`, and
  `last_error`. Surface these values alongside the existing pipeline status
  cards so ops can see whether catalog restoration succeeded.
- Prometheus metrics:
  - `nautilus_ml_coverage_buckets_total{status}` — classified bucket counts
    (`healthy`, `restore_from_catalog`, `reingest_from_source`).
  - `nautilus_ml_coverage_restore_failures_total{stage}` — failed stages
    (`classification`, `catalog`, `targeted_update`).
  - `nautilus_ml_coverage_latency_seconds` — end-to-end restoration latency.
- Environment toggles:
  - `COVERAGE_RESTORE_ENABLED=1` runs the flow automatically before ingestion.
  - `COVERAGE_MAX_BUCKETS_PER_RUN` (default `500`) caps how many buckets are
    processed per pass, preventing long recovery loops; skipped counts are
    logged under `coverage_manager.bucket_cap_applied`.
- Failure modes surfaced via `/health["errors"]`:
  - `SchemaHealthCheckError: ... ml_data_events ...` (or `ml_data_watermarks`) means migrations skipped the instrumentation tables. Run `poetry run python -m ml.stores.migrations_runner apply --db-url …` before restarting so event emission works.
  - `Invalid MARKET_DATASET_INPUTS` indicates the env var references descriptors not listed in `ml/config/market_feed_descriptors.json` or schema overrides that the dataset does not support (e.g., `mbp-10` on `EQUS.MINI`). Fix the env var and redeploy; the pipeline will not start until the combination is valid.
- Manual remediation: `poetry run python -m ml.cli.coverage_restore --json`
  boots the same workflow using local environment variables, emitting a summary
  identical to `/health["coverage"]`.

### Streaming Orchestrator Admin

- The streaming training orchestrator persists lifecycle state to
  `ml_out/streaming_orchestrator_state.json` (override with
  `ML_STREAM_ORCH_STATE_PATH`). Use the admin helpers on
  `InMemoryStreamingOrchestrator` to manage backlog and retries without blowing
  away historical context.
- `clear_backlog(dataset_id: str | None, include_active: bool = False)` removes
  persisted plans. Pass `include_active=True` before recycling workers or when a
  dataset should be relaunched from scratch.
- `resume_plan(plan_id: str)` resets retry counters, saturation flags, and
  heartbeat timestamps so the plan can proceed on the next worker heartbeat.
- `saturated_plan_ids()` surfaces plans currently throttled by
  `saturation_heartbeat_limit`. Inspect these first when the dashboard backlog
  gauge is flat.
- `expired_plans()` returns the underlying `DatasetPlanEvent` objects whose
  heartbeats aged out. Feed the results back into `enqueue_training` (or a
  manual worker run) after confirming the data path is healthy.
- **Persistence worker backlog replay**
  - Restarts now seed Redis `XREAD` with `0-0` when no cursor is stored, so the worker replays any backlog accumulated while it was offline. The cursor persists in `stream_cursor` inside `ml_out/streaming_training_state.json`.
  - To replay from scratch, stop the worker, delete the `stream_cursor` field (or remove the state file), and restart the worker; the next poll will enumerate historical events before tailing live traffic.
  - To skip directly to the head, update `stream_cursor` to the latest stream ID (check via `redis-cli XINFO STREAM ${ML_BUS_REDIS_STREAM:-ml-events}`) before restarting.
  - Monitor `ml_streaming_persistence_poll_attempts_total{outcome="failure"}` and the worker logs for cursor persistence warnings; failures leave the worker tailing the old position.

Example (run inside the `streaming_orchestrator` container or a local shell with
the repo sourced):

```python
from pathlib import Path

from ml.config.streaming_pipeline import DatasetServiceConfig, TrainingOrchestratorConfig
from ml.training.event_driven.dataset_service import StreamingDatasetPlanner
from ml.training.event_driven.orchestrator import InMemoryStreamingOrchestrator

config = TrainingOrchestratorConfig(
    command_topic="",
    result_topic="",
    heartbeat_topic="",
    enable_state_persistence=True,
)
planner = StreamingDatasetPlanner(DatasetServiceConfig(parquet_root="/data/streaming"))
orchestrator = InMemoryStreamingOrchestrator(
    config,
    planner,
    state_path=Path("/app/ml_out/streaming_orchestrator_state.json"),
)
print(orchestrator.saturated_plan_ids())
orchestrator.clear_backlog(include_active=True)
```

- The dashboard `get_streaming_training_state()` endpoint now returns dataset-level
  summaries (`dataset_details`) with outstanding plan counts, recent plan/result
  timestamps, and active worker lists. Pair these with the Prometheus gauges
  (`ml_tft_streaming_training_backlog`, `ml_tft_streaming_workers_active`) when
  building backlog widgets or saturation alerts.
- Multi-worker experiment procedures and results are tracked in
  `ml/docs/ops/streaming_scaling_experiments.md`; update this log as new worker
  counts are validated.
- Alert thresholds (derived from multi-worker experiments):
  - Warning when `summary.total_outstanding >= 4`, critical when `>= 8`.
  - Warning when `summary.total_workers < expected_workers` (default 1), critical when zero.
  - Dataset row highlights turn amber when outstanding plans > 0; red styling is applied when backlog >= 8 via the dashboard JS logic.

### Streaming Checkpoint Monitoring

- Prometheus counters expose checkpoint activity:
  - `ml_streaming_checkpoints_total{outcome,trigger}` tracks save attempts (success vs. failure, labelled by `interval_seconds`, `interval_steps`, or `manual`).
  - `ml_streaming_checkpoint_resumes_total{outcome}` records discovery, successful resume, and missing checkpoint events.
  - `ml_streaming_checkpoint_evictions_total{outcome}` increments when signal-driven saves complete (or fail) during Azure eviction notices.
- Grafana panel `Streaming Checkpoints` (in `ml/deployment/grafana/ml_pipeline_health.json`) visualises:
  - `sum by (outcome,trigger)(rate(ml_streaming_checkpoints_total[$__rate_interval]))`
  - `sum by (outcome)(rate(ml_streaming_checkpoint_evictions_total[$__rate_interval]))`
  Import or sync the dashboard to expose these panels alongside backlog charts.
- Worker logs emit structured events `checkpoint_saved`, `checkpoint_resume_detected`, and `checkpoint_resume_applied`; surface these via the dashboard log stream or `kubectl logs` to confirm that spot interruptions triggered a flush.
- The streaming runner injects checkpoint telemetry into result manifests (`checkpoint.resumed`, `checkpoint.resume_global_step`, `checkpoint.latest_checkpoint_path`). Dashboards consuming the state snapshot will display these fields under each dataset's telemetry block once a plan lands.
- Checkpoint metadata files live under the configured `StreamingWorkerConfig.checkpoint_dir` (`{plan_id}_latest.json` and associated `.ckpt` artefacts). When debugging resumes, inspect these files to confirm the recorded epoch/step before relaunching the worker.

### Azure Spot VM Lifecycle Checklist

1. **Bootstrap the VM**
   - Mount the durable checkpoint store (Blob via `blobfuse2` or Azure Files SMB) at the path referenced by `ML_STREAMING_CHECKPOINT_DIR`.
   - Export the runner CLI flags (`--checkpoint-dir`, `--checkpoint-interval-seconds`, `--checkpoint-interval-steps`) or matching env vars before starting the streaming runner service.
2. **Launch the Runner**
   - Start `ml/cli/streaming_training_runner.py` inside the spot VM session; confirm the log line `checkpoint_resume_detected` when resuming an interrupted plan.
   - Ensure `save_checkpoint_now` requests propagate to the worker by watching for `checkpoint_request_completed` after manual invocations or signal handling.
3. **Monitor During Execution**
   - Track the metrics above (especially `ml_streaming_checkpoint_evictions_total`) alongside backlog gauges to verify that eviction notices trigger final checkpoints.
   - Use the dashboard `dataset_details[*].telemetry.checkpoint` payload to confirm whether the latest plan resumed (`resumed=true`) and which checkpoint file was consumed.
4. **Handle Evictions**
   - When Azure scheduled events flag an impending eviction, the runner's scheduled-event watcher emits `azure_eviction_notice_received` and calls `save_checkpoint_now` (`checkpoint_saved` with `trigger=manual:signal`). Avoid manual termination until the save completes.
5. **Resume After Rehydration**
   - On the replacement VM, remount the storage and relaunch the runner; the worker auto-loads `{plan_id}_latest.ckpt` and emits `checkpoint_resume_applied` upon restart.
6. **Teardown & Cleanup**
   - Before deallocating the VM, confirm that the latest manifest contains `checkpoint.latest_checkpoint_path`. Optional: prune archived `.ckpt` files older than required retention if the store quota is constrained.

## UI Operations

- Access the HTML dashboard root (`/`) for a lightweight overview. Cached
  sections update on load; click “Refresh” on the Events card to trigger a new
  poll.
- Grafana embeds are shown only when `ML_DASHBOARD_GRAFANA_EMBED=1` **and** at
  least one panel ID is provided via `ML_DASHBOARD_GRAFANA_PANELS`.
- Store health tables include freshness timestamps (ISO) and age (seconds) for
  quick triage.

## Validations and Diagnostics

- Static checks: `uv run --active --no-sync mypy ml/dashboard --strict` and
  `ruff check ml/dashboard ml/tests/unit/dashboard`.
- Targeted tests: `uv run --active --no-sync pytest ml/tests/unit/dashboard/`
  and integration coverage via `uv run --active --no-sync pytest \
  ml/tests/integration/dashboard/`.
- Metrics validators: `make validate-metrics`; advisory suite `make
  validate-nautilus-patterns` before releases.
- Inspect Prometheus metrics locally with `curl -s localhost:8010/metrics | rg
  ml_dashboard_`.
- Runtime metrics snapshot: import `DashboardService` and call
  `get_metrics_snapshot()` to inspect cache hit ratios, store summary latency
  P95, Grafana provisioning success rate, and registry latency P95. Use
  `evaluate_success_criteria()` to receive a pass/fail summary against the
  roadmap thresholds.
- Strategy heartbeat fallback: the dry-run strategy container now advertises a
  temporary heartbeat (`ML_STRATEGY_HEARTBEAT_DURATION_SECONDS`, default 120)
  before marking `/health` unhealthy and shutting down. Use
  `ML_STRATEGY_HEARTBEAT_INTERVAL_SECONDS` to tune the sleep cadence or set
  `ML_STRATEGY_HEARTBEAT_ENABLED=0` to disable the behavior entirely.
- One-shot bootstrap: run `python -m ml.cli.dashboard_welcome` to start the
  docker-compose stack (`ml/deployment/docker-compose.yml`) and display a
  consolidated health summary / quick links for the dashboard UI, Grafana, and
  Prometheus. Pass `--status-only` to skip `docker compose up` and inspect an
  already running deployment.

## Troubleshooting

- **Token failures**: Check `ml_dashboard_auth_validations_total` breakdown and
  ensure `ML_DASHBOARD_TOKENS` is correctly formatted JSON. Fall back to a
  single token by setting `ML_DASHBOARD_TOKENS` empty and using
  `ML_DASHBOARD_TOKEN`.
- **Grafana provisioning errors**: Fetch `/api/observability/status` and review
  the `error` field. Verify the API token and folder UID; rerun provisioning
  with `force=true` using the service method if needed.
- **Store summary errors**: Inspect
  `ml_dashboard_store_summary_failures_total{reason}` and confirm the Postgres
  connection string. Temporarily disable the feature while databases are
  offline.
- **Event polling lag**: Validate Redis connectivity and review
  `ml_dashboard_events_failure_total`. Cached results may persist up to
  `ML_DASHBOARD_EVENTS_CACHE_TTL` seconds.
