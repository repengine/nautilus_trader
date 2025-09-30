# Orchestration Runbook

This runbook describes how to run the cold‑path ML pipeline on a schedule, how to promote artifacts, and the key environment variables.

## Scheduler

- Entrypoint: `python -m ml.cli.pipeline_scheduler`
- Schedule options:
  - `--schedule-time HH:MMZ` (UTC daily time, e.g., `02:30Z`)
  - `--interval-min N` (minutes, e.g., `1440`)
- Config: `--config <path>` to a JSON/TOML orchestrator config (see `ml/config/pipeline_scheduler_example.toml`). Include an optional `[integration]` section to mirror the runtime flags (`enabled`, `db_connection`, `auto_start_postgres`, `auto_migrate`, `ensure_healthy`, `strict_protocol_validation`, `run_validators`).
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
- Coverage/Writer: `--coverage_mode catalog|sql`, `--catalog_path`, `--db`
    - The DataStore/Registry stack is mandatory. The orchestrator always writes
      via `DataStoreMarketDataWriter`. `--write_mode parquet` enables an
      additional mirror into the Parquet catalog; `--write_mode datastore`
      keeps writes confined to the store.
  - Macro refresh: `--skip_macro_refresh`, `--macro_freshness_hours`, `--macro_series_ids`, `--macro_fred_path`
  - Vintage policy: `--vintage_policy real_time|final`, `--vintage_as_of <ISO8601>` to control ALFRED revisions
  - Instrument resolution: `--instrument_ids`
  - Dataset validation: `--validation_min_rows`, `--validation_min_positive_rate`, `--validation_max_positive_rate`, `--validation_min_feature_coverage`
  - L2 extras: `--skip_l2_ingest`, `--l2_days`, `--l2_progress_file`, `--l2_symbols`, `--l2_tier`
- Auto-fill coverage (optional): `--auto_fill_universe` enables pre-build backfills (bars/TBBO/trades) and depth ingestion; pair with overrides such as `--auto_fill_l2_days`, `--auto_fill_skip_l2`, `--auto_fill_l2_progress_file`, and `--auto_fill_allow_dataset_l2_ingest`
- HPO: `--hpo`, `--hpo_epochs`, `--hpo_batch_size`, `--hpo_tail_rows`, `--hpo_limit_groups`
- Teacher: `--train`, `--teacher_model_id`, `--feature_registry_dir`, `--feature_set_id`, `--max_epochs`
- Promotions/Refresh:
  - Model: `--auto_register_model`, `--gates_json`, `--auto_promote`, `--deploy_target`
  - Features: `--auto_register_features`, `--feature_metrics_json`, `--refresh_features`
- Runtime attachment (optional): `--attach-runtime` wires `MLIntegrationManager` after the cold path finishes. Pair it with:
  - `--runtime-db-connection` (override DB URL used for runtime wiring)
  - `--runtime-auto-start-db`, `--runtime-auto-migrate`, `--runtime-no-ensure-healthy`
  - `--runtime-strict-protocol-validation` (enforce protocol checks) and `--runtime-skip-validators` (skip metrics/event validators)

### Databento Discovery (dynamic dataset selection)

The orchestrator now auto-discovers Databento datasets per symbol and schema when no explicit `market_inputs` are provided. Discovery queries Databento metadata at runtime, evaluates coverage windows and cost estimates, and picks the cheapest viable dataset for each symbol.

- Override behaviour with environment variables (default allowlist is `XNAS.ITCH`):
  - `DATABENTO_DISCOVERY_DATASETS` – optional ordered allowlist of dataset ids to consider.
  - `DATABENTO_DISCOVERY_DENYLIST` – datasets to exclude from discovery.
  - `DATABENTO_DISCOVERY_MAX_COST_USD` – reject any candidate whose estimated cost exceeds this limit.
  - `DATABENTO_DISCOVERY_MAX_CANDIDATES` – maximum number of datasets to probe per discovery run.
- Discovery reuses the existing coverage policy (`DATABENTO_MAX_DAYS`, `DATABENTO_ALLOWED_SCHEMAS`, etc.) to clamp request windows safely.
- When a static `market_inputs` section exists in the config, discovery is skipped and the explicit bindings are honoured.
- Symbology resolution runs ahead of every cost probe and ingestion call. The orchestrator and `DatabentoIngestionService` share a resolver that invokes `Historical.symbology.resolve` to fetch the canonical instrument identifier, then normalises the symbol root (e.g., `INTC.XNAS` → instrument `4182`, symbol `INTC`). Logs surface the `input_symbol`, resolved symbol, and instrument id to aid audits.
- `discover_symbol_dataset` is exposed on the ingestion service to power auto-fill gap checks; it returns the resolved symbol, dataset id, instrument id, storage kind, and the policy-clamped coverage window. Use it when you need to inspect the binding without triggering a download.
- Cost guards now operate on the resolved symbol. If the original venue-qualified identifier fails a cost estimate, the resolver retries dataset-specific forms before falling back. Violations log `ingestion.cost_violation` with the chosen variant.

When auto-fill is enabled the orchestrator derives coverage windows from the subscription policy (7y bars, 1y L1, ~30d L2/L3), invokes `IngestionOrchestrator.backfill_gaps` for bars/TBBO/trades, and then calls `populate_l2_efficient` before the dataset build begins. By default the depth stage is considered satisfied once auto-fill runs; use `--auto_fill_allow_dataset_l2_ingest` if you still want the dataset phase to run its own L2 ingestion.

Auto-fill requests query Databento metadata before every download and clamp the ingestion window to the provider `available_end`. This keeps backfills inside the zero-cost guardrails and eliminates 422 responses. Instrument lists should use venue-qualified IDs (for example `SPY.XNYS`) so parquet writers can resolve the correct bar template from the dataset manifest.

Every dataset build now writes `dataset_metadata.json` alongside the parquet/CSV artifacts detailing dataset id, `ts_event_start/end`, overall/train/validation/test windows, and the declared vintage policy/cutoff. Promotion gates (or downstream tooling) should inspect this file to guarantee models only train on the intended window with the expected revision policy.

EQUS minute bars persisted by the orchestrator include explicit provenance columns:
`source_dataset`, `aggregation_mode`, and `scaling_factor`. Downstream readers (DataStore,
SQL readers, TFT builder) now expose these fields and registry events record aggregated
values under `source_datasets`, `aggregation_modes`, and `scaling_factors`, enabling
monitors to assert which fallback mode produced each window.

When `--attach-runtime` is enabled, the orchestrator hydrates the four stores/registries and, by default, runs the metrics/events validators so the runtime is safe for actors. Use `--runtime-skip-validators` during dry-runs if you only need wiring without the scans.

## Environment Variables

- `CATALOG_PATH` for ParquetDataCatalog
- `DATABENTO_API_KEY` for optional Databento ingestion
- DB URL: `NAUTILUS_DB` or use orchestrator `--db` flag
- Canonicalization toggles:
  - `ML_EQUS_ENABLE_TRADE_REAGG` (default `1`) enables trade-level re-aggregation for missing EQUS windows.
  - `ML_EQUS_ENABLE_VOLUME_SCALING` (default `1`) enables scaled ITCH fallback when trades are unavailable; tune with `ML_EQUS_SCALING_REFERENCE_DAYS`, `ML_EQUS_SCALING_MIN_RATIO`, `ML_EQUS_SCALING_MAX_RATIO`.
- TFT builder guardrail: `ML_TFT_ALLOW_PARQUET_FALLBACK=1` opt-in only; disabled by default so SQL read failures raise instead of silently falling back to parquet.
- Scheduler env:
  - `ORCH_SCHEDULE_TIME`, `ORCH_INTERVAL_MIN`, `ORCH_CONFIG`, `ORCH_DRY_RUN`, `ORCH_FORCE`
  - `ORCH_LOCK_PATH`, `ORCH_LOCK_TTL_HOURS`

## Observability & Validation

- Metrics (scheduler):
  - `nautilus_ml_orch_runs_total{status}`
  - `nautilus_ml_orch_phase_latency_seconds{phase}`
- Metrics (canonicalization):
  - `ml_canonicalization_volume_residual{mode,residual_type,dataset}` — absolute/relative volume residuals for EQUS fallback modes.
- Events: Use DataRegistry `emit_event` with `Stage/Source/EventStatus` for pipeline phases.
- Validators:
  - `make validate-metrics`
  - `make validate-events`
  - `make validate-nautilus-patterns` (advisory)
- Parity harness: `make parity-report` regenerates `ml/tests/validation_reports/equs_itch_parity_summary.json` by executing the built-in Tier-1 suite (multiple symbols/windows across 2023–2025). Set `DATABENTO_API_KEY` in the environment before running.
- Manual verification/backfill tool: `uv run --active --no-sync python -m ml.scripts.verify_eq_itch_parity --help` — include `--suite` to run the default matrix or `--suite-config <path>` to provide a custom JSON scenario list.

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
- Pre‑ingestion (Unified Ingestion) via config:
  - Prefer the config file to enable unified ingestion before dataset build:

```toml
# ml/config/pipeline_scheduler_example.toml (excerpt)

[dataset]
data_dir = "/data/catalog" # ParquetDataCatalog path
symbols = "SPY.XNAS,QQQ.XNAS"
out_dir = "ml_out"
include_macro = true

[pre_ingestion]
symbols = ["SPY.XNAS","QQQ.XNAS"]
retention_days = 90
[pre_ingestion.databento]
dataset = "EQUS.MINI"
schema = "ohlcv-1m"

[pre_ingestion_options]
use_orchestrator = true
dual_write = true
start_metrics_server = false
```

The orchestrator reads this config and automatically runs `DataScheduler` in orchestrator
mode before dataset build, ensuring both SQL coverage and ParquetDataCatalog are populated.

### EQUS.MINI Canonicalization & ITCH Fallback

- `DatabentoIngestionService` canonicalises all `EQUS.MINI` minute bars with `ml.data.ingest.canonicalization.canonicalize_equities_minute_bars` before any store writes. The helper trims to 08:00–16:00 ET, ensures `ts_event/ts_init` are nanosecond integers, rounds prices to four decimals, and normalises volumes to `int64`.
- When a requested window predates the provider's `EQUS.MINI` coverage, the ingestion service automatically replays the same request against `XNAS.ITCH`, applies the canonicalisation step, and persists the result under `dataset_id='EQUS.MINI'`. Lineage is captured in the registry seed (`ml/stores/migrations/004_data_registry.sql`), so downstream manifests continue to reference a single canonical dataset while acknowledging the ITCH parent.
- Fallback activations increment `ml_fallback_activations_total{component="databento_ingestion_service",level="itch_to_equs"}` and emit an `ingestion.canonicalize.applied` log with row counts and trimming stats. Operators should alarm on sustained fallback activations during live ingest.
- To spot-check canonicalisation and lineage locally:

  ```bash
  DATABENTO_API_KEY=... uv run --active --no-sync python -m ml.cli.pipeline_orchestrator \
    --stage ingest --ingest --dataset_id EQUS.MINI --schema ohlcv-1m \
    --symbols INTC --lookback_days 2 --write_mode sql --coverage_mode catalog
  ```

  The run should surface canonicalisation logs, increment the fallback counter only for pre-coverage windows, and complete without manifest constraint violations.
