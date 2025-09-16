# Orchestration Runbook

This runbook describes how to run the cold‑path ML pipeline on a schedule, how to promote artifacts, and the key environment variables.

## Scheduler

- Entrypoint: `python -m ml.cli.pipeline_scheduler`
- Schedule options:
  - `--schedule-time HH:MMZ` (UTC daily time, e.g., `02:30Z`)
  - `--interval-min N` (minutes, e.g., `1440`)
- Config: `--config <path>` to a JSON/TOML orchestrator config (see `ml/config/pipeline_scheduler_example.toml`).
- Dry-run and force:
  - `--dry-run` logs actions only
  - `--force` ignores existing outputs and runs anyway
- Locking:
  - `ORCH_LOCK_PATH` (defaults `<out_dir>/.orch.lock` or `/tmp/ml_orch.lock`)
  - `ORCH_LOCK_TTL_HOURS` (default `12`) to clear stale locks

Make examples:

```bash
# Dry run with example config, 24h interval
make ml-pipeline-scheduler-example

# Daily at 02:30Z (env-driven)
ORCH_SCHEDULE_TIME=02:30Z ORCH_CONFIG=ml/config/pipeline_scheduler_example.toml \
  make ml-pipeline-scheduler
```

### CI Smoke

Use a single-shot smoke that forces dummy integration and minimal coverage/writer settings:

```bash
make ml-scheduler-smoke ORCH_CONFIG=ml/config/pipeline_scheduler_example.toml DRY_RUN=1
```

This validates config parsing and orchestrator argument flow without requiring a database or catalog.

## Orchestrator (Cold Path)

The scheduler invokes the orchestrator: `python -m ml.cli.pipeline_orchestrator`.

Key flags:

- Ingestion/backfill (optional): `--ingest`, `--dataset_id`, `--schema`, `--instruments`, `--lookback_days`
- Coverage/Writer: `--coverage_mode catalog|sql`, `--catalog_path`, `--db`, `--write_mode parquet|datastore`
- Dataset: `--data_dir`, `--symbols`, `--out_dir`, `--include_macro`, `--include_micro`, `--include_l2`, `--horizon_minutes`, `--threshold`, `--lookback_periods`
- HPO: `--hpo`, `--hpo_epochs`, `--hpo_batch_size`, `--hpo_tail_rows`, `--hpo_limit_groups`
- Teacher: `--train`, `--teacher_model_id`, `--feature_registry_dir`, `--feature_set_id`, `--max_epochs`
- Promotions/Refresh:
  - Model: `--auto_register_model`, `--gates_json`, `--auto_promote`, `--deploy_target`
  - Features: `--auto_register_features`, `--feature_metrics_json`, `--refresh_features`

## Environment Variables

- `CATALOG_PATH` for ParquetDataCatalog
- `DATABENTO_API_KEY` for optional Databento ingestion
- DB URL: `NAUTILUS_DB` or use orchestrator `--db` flag
- Scheduler env:
  - `ORCH_SCHEDULE_TIME`, `ORCH_INTERVAL_MIN`, `ORCH_CONFIG`, `ORCH_DRY_RUN`, `ORCH_FORCE`
  - `ORCH_LOCK_PATH`, `ORCH_LOCK_TTL_HOURS`

## Observability & Validation

- Metrics (scheduler):
  - `nautilus_ml_orch_runs_total{status}`
  - `nautilus_ml_orch_phase_latency_seconds{phase}`
- Events: Use DataRegistry `emit_event` with `Stage/Source/EventStatus` for pipeline phases.
- Validators:
  - `make validate-metrics`
  - `make validate-events`
  - `make validate-nautilus-patterns` (advisory)

## Promotion Gates

- Example gates JSON: `ml/config/promotion_gates_example.json`
- CLI flags wired via orchestrator or config (`promotions` section in TOML/JSON):
  - `auto_register_model`, `gates_json`, `auto_promote`, `deploy_target`
  - `auto_register_features`, `feature_metrics_json`, `refresh_features`

## Resilience

- The scheduler clears stale locks older than `ORCH_LOCK_TTL_HOURS`.
- DRY_RUN allows validating scheduling and argument flow without running model training.
- Orchestrator run failures emit FAILED events; use metric counters/histograms to track trends.

## Production Hardening

### systemd Service (example)

Create `/etc/systemd/system/ml-pipeline-scheduler.service`:

```
[Unit]
Description=Nautilus ML Pipeline Scheduler
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=ORCH_CONFIG=/opt/nautilus/ml/config/pipeline_scheduler.toml
Environment=ORCH_SCHEDULE_TIME=02:30Z
Environment=CATALOG_PATH=/data/catalog
Environment=NAUTILUS_DB=postgresql://postgres:postgres@db:5432/nautilus
Environment=ML_ALLOW_DUMMY=0
Restart=always
RestartSec=5
ExecStart=/usr/bin/python -m ml.cli.pipeline_scheduler
WorkingDirectory=/opt/nautilus
User=nautilus
Group=nautilus
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Then run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ml-pipeline-scheduler
sudo systemctl status ml-pipeline-scheduler
```

### Docker Compose (example)

```yaml
version: '3.8'
services:
  scheduler:
    image: python:3.12-slim
    working_dir: /app
    volumes:
      - ./:/app:ro
      - /data/catalog:/data/catalog:ro
    environment:
      ORCH_CONFIG: /app/ml/config/pipeline_scheduler_example.toml
      ORCH_SCHEDULE_TIME: "02:30Z"
      CATALOG_PATH: /data/catalog
      NAUTILUS_DB: postgresql://postgres:postgres@db:5432/nautilus
      ML_ALLOW_DUMMY: "0"
    command: ["python", "-m", "ml.cli.pipeline_scheduler"]
    restart: unless-stopped
    depends_on:
      - db
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: nautilus
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata: {}
```

This starts the scheduler alongside a PostgreSQL container. Provide `CATALOG_PATH` via a bind mount when using `parquet` writers.
