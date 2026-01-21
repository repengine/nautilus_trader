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
- Teacher (core): `--train`, `--teacher_model_id`, `--feature_registry_dir`, `--feature_set_id`, `--max_epochs`
  - Resource tuning: `--batch_size`, `--dataloader_workers`, `--accelerator`, `--devices`, `--precision`
  - Model sizing: `--max_encoder_length`, `--max_prediction_length`, `--hidden_size`, `--lstm_layers`, `--attention_head_size`, `--dropout`
  - Optimizer/loss: `--learning_rate`, `--loss`, `--pos_weight`, `--seed`
  - Data caps/splits: `--limit_groups`, `--tail_rows`, `--val_days`, `--embargo_hours`, `--purge_gap`, `--cv_splits`, `--test_fraction`
  - Column overrides: `--target_col`, `--time_index_col`, `--timestamp_col`, `--group_id_col`, `--static_categoricals`, `--static_reals`, `--known_future_reals`
  - Artifacts/registration: `--save_interpretability`, `--export_torchscript`, `--export_safetensors`, `--pretrained_state_path`, `--register_teacher`
  - Decision policy: `--decision_policy`, `--decision_config`
  - Parquet preference: `--prefer_parquet/--no-prefer_parquet` (default prefers `dataset_with_vintage_age.parquet` or `dataset.parquet` when present)
- Promotions/Refresh:
  - Model: `--auto_register_model`, `--gates_json`, `--auto_promote`, `--deploy_target`
  - Features: `--auto_register_features`, `--feature_metrics_json`, `--refresh_features`
- Runtime attachment (optional): `--attach-runtime` wires `MLIntegrationManager` after the cold path finishes. Pair it with:
  - `--runtime-db-connection` (override DB URL used for runtime wiring)
  - `--runtime-auto-start-db`, `--runtime-auto-migrate`, `--runtime-no-ensure-healthy`
  - `--runtime-strict-protocol-validation` (enforce protocol checks) and `--runtime-skip-validators` (skip metrics/event validators)

### Feature dataset coverage config

The scheduler config controls market ingestion, but feature datasets (earnings,
calendar, macro releases) need an explicit manifest so the coverage manager can
inspect their SQL tables and parquet mirrors. Tier‑1 environments should set:

```bash
export COVERAGE_DATASETS_FILE=ml/config/coverage_datasets_tier1.toml
```

Each `[[datasets]]` entry declares the dataset id, schema, entity field (e.g.,
`ticker`), resolved universe (`@tier1_full` is supported), plus optional `[sql]`
and `[parquet]` sections. The loader (`ml/config/dataset_coverage.py`) normalises
the payloads into `CoverageDatasetEntry` objects and the pipeline entrypoint
appends them to the coverage manager before schema audits/classification. When a
feature bucket lands in `RESTORE_FROM_CATALOG`, the entrypoint now spins up
`FeatureCoverageRestorer` (`ml/data/coverage/feature_restorer.py`) and replays
the parquet partition through the standard `DataStore.write_earnings_*`
interfaces so SQL + registry state stay in lockstep. Operators will see a pair of
log lines: `coverage.feature_restore.pending` and, after replay completes,
`coverage.feature_restore.completed` with dataset/instrument/row counts (partial
recoveries emit `coverage.feature_restore.partial`). Every activation increments
`ml_fallback_activations_total{component="feature_coverage_restorer",
level="<dataset_id>"}` for monitoring, so keep the manifest aligned with the
active environment to maintain accurate detection and automated replays.

**Tier‑1 coverage-gate quickstart**

- Environment (Tier‑1): `COVERAGE_DATASETS_FILE=ml/config/coverage_datasets_tier1.toml COVERAGE_RESTORE_ENABLED=1 CATALOG_CLEAN_MODE=archive CATALOG_BACKUP_DIR=ml_out/catalog_archives MARKET_BACKFILL_DYNAMIC_LOOKBACK=1`
- Command: `poetry run python -m ml.cli.pipeline_orchestrator --config ml/config/pipeline_scheduler_example.toml`
- Expected evidence to capture per run:
  - Orchestrator log path (e.g., `ml_out/tier1_orchestrator_run*.log`) and run_id from log prefix.
  - Coverage metrics: `coverage.feature_restore.pending/completed/partial`, `ml_fallback_activations_total{component="feature_coverage_restorer",level=…}`, bucket counts (`pipeline_status["coverage"]`).
  - Dataset metadata and catalog archive location under `ml_out/catalog_archives/**`.
  - Validation: `make validate-metrics`, `make validate-events`, `poetry run coverage report --include "ml/*"` on the same working tree.
- Manifest enforcement: the entrypoint now rejects manifests containing unsupported dataset IDs; ensure Tier‑1 manifests only list `ml.earnings_actuals`, `ml.earnings_estimates`, `ml.macro_release_calendar`, `ml.macro_observations`, `ml.events_calendar`, `ml.microstructure_minute`, `ml.l2_minute`.

### Coverage restoration guardrails

- Coverage restoration now gates ingestion. When `COVERAGE_RESTORE_ENABLED=1` the
  pipeline aborts if the coverage manager cannot classify buckets, lacks a DB connection,
  or fails targeted ingestion. Opt out only with `COVERAGE_RESTORE_ALLOW_FAILURE=1`
  (not recommended for Tier‑1).
- Catalog coverage uses the same identifier resolver as the rehydrator. Bars default to
  `{instrument_id}-1-MINUTE-LAST-EXTERNAL` (override via `CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE`);
  TBBO/Trades/MBP default to raw `instrument_id`, and per-schema overrides can be supplied
  via `CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE_MAP`. This alignment keeps parquet interval scans
  fast and prevents redundant Databento backfills when the catalog already contains data.

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

EQUS minute bars persisted by the orchestrator now store the raw Databento payload and
tag each row with `source_dataset`. Downstream consumers (DataStore, SQL readers, TFT
builder) surface the same tag so you can distinguish native EQUS rows from any fallback
dataset (for example, `XNAS.ITCH`) without additional normalization.

When `--attach-runtime` is enabled, the orchestrator hydrates the four stores/registries and, by default, runs the metrics/events validators so the runtime is safe for actors. Use `--runtime-skip-validators` during dry-runs if you only need wiring without the scans.

## Environment Variables

- `CATALOG_PATH` for ParquetDataCatalog
- `CATALOG_CLEAN_MODE=archive` (plus optional `CATALOG_BACKUP_DIR`) to archive the current catalog into a timestamped folder before each run.
- `DATABENTO_API_KEY` for optional Databento ingestion
- DB URL: `NAUTILUS_DB` or use orchestrator `--db` flag
- TFT builder guardrail: `ML_TFT_ALLOW_PARQUET_FALLBACK=1` opt-in only; disabled by default so SQL read failures raise instead of silently falling back to parquet.
- Catalog rehydration: set `CATALOG_REHYDRATE_ENABLED=1` (plus supporting `CATALOG_REHYDRATE_*` knobs) to replay the Parquet catalog into Postgres before orchestrator ingestion.
- Docker defaults now bias to fast starts: the override file sets `CATALOG_REHYDRATE_LOOKBACK_DAYS=5`, `CATALOG_REHYDRATE_STALE_ONLY=1`, and `CATALOG_REHYDRATE_EXHAUSTIVE=0`. Override these in your shell if you intentionally want a full replay (for example after destructive DB restores).
- Dynamic Databento lookback:
  - `MARKET_BACKFILL_DYNAMIC_LOOKBACK=1` enables per-instrument gap sizing so the scheduler only requests the missing windows instead of a fixed week of data every pass.
  - `MARKET_BACKFILL_MIN_DAYS` (default `1`) clamps the lower bound, and `MARKET_BACKFILL_MAX_DAYS` optionally caps extremely stale symbols.
- Scheduler env:
  - `ORCH_SCHEDULE_TIME`, `ORCH_INTERVAL_MIN`, `ORCH_CONFIG`, `ORCH_DRY_RUN`, `ORCH_FORCE`
  - `ORCH_LOCK_PATH`, `ORCH_LOCK_TTL_HOURS`
- Port mapping reminders:
  - Primary ML stack: host `5433` → container `ml-postgres-1:5432`. Docker services (e.g., `ml_pipeline`) should export `DB_CONNECTION=postgresql://postgres:postgres@ml-postgres-1:5432/nautilus`, while host tooling uses `postgresql://postgres:postgres@localhost:5433/nautilus`.
  - Test stack: host `5434` → container `ml-test-postgres-test-1:5432`. Never point Tier‑1 orchestration or coverage restores at this DSN unless you are running the isolated test compose file.

## Catalog Hygiene

- Archive stale Parquet partitions before Tier-1 runs with:

  ```bash
  poetry run python -m ml.cli.catalog_hygiene \
    --catalog-path data/catalog \
    --backup-dir ml_out/catalog_archives
  ```

- Pipeline configs may also specify `[ingestion].catalog_clean_mode = "archive"` and `catalog_backup_dir = "ml_out/catalog_archives"` (or set `CATALOG_CLEAN_MODE=archive` / `CATALOG_BACKUP_DIR=...`) so the orchestrator and integration runtime scrub the catalog automatically.
- `CATALOG_REHYDRATE_STALE_ONLY=1` (default) samples SQL staleness before replaying the catalog. If every instrument has rows newer than `CATALOG_REHYDRATE_STALENESS_HOURS` (default `6`) the rehydration pass is skipped entirely, which keeps restarts from scanning 100+ parquet trees when the database is already in sync. Set the flag to `0` to force a full replay after a destructive restore.
- Coverage guard: set `COVERAGE_RESTORE_ENABLED=1` to fail the pipeline when coverage classification/restore cannot complete. Use `COVERAGE_RESTORE_ALLOW_FAILURE=1` only for non-critical drills; production Tier‑1 runs should keep it disabled so coverage remains authoritative.

## Observability & Validation

- Metrics (scheduler):
  - `nautilus_ml_orch_runs_total{status}`
  - `nautilus_ml_orch_phase_latency_seconds{phase}`
- Symbology/alias telemetry:
  - `nautilus_ml_symbology_alias_hits_total{dataset}` and `nautilus_ml_discovery_symbology_rejections_total{dataset}` surface alias usage and unresolved symbols during dataset discovery/ingestion.
  - `nautilus_ml_symbology_retry_total{dataset,status}` increments whenever the Databento resolver retries a transient 5xx response; correlate spikes here with Tier‑1 failures such as the DIS `502 <empty message>` outage.
- Coverage manifest telemetry:
  - `nautilus_ml_coverage_manifest_events_total{event="loaded|missing|invalid"}` increments when the orchestrator loads `COVERAGE_DATASETS_FILE`; missing or invalid manifests now attach `feature_manifest_*` markers to pipeline status/errors before classification starts.
- Tier-1 run recipe (cold path, coverage gate on):
  - Env: `COVERAGE_DATASETS_FILE=ml/config/coverage_datasets_tier1.toml COVERAGE_RESTORE_ENABLED=1 CATALOG_CLEAN_MODE=archive CATALOG_BACKUP_DIR=ml_out/catalog_archives`
  - Command: `poetry run python -m ml.cli.pipeline_orchestrator --config ml/config/pipeline_scheduler_example.toml`
  - Expected I/O: catalog hygiene archive under `ml_out/catalog_archives/**`, orchestrator log under `ml_out/tier1_orchestrator_run*.log`, SQL writes for Tier-1 bars/TBBO/trades, parquet fan-out may log overlap skips until catalog is compacted, coverage logs (`coverage.feature_restore.*`) if feature buckets require replay.
  - Validation: check `ml_data_events` for fresh `INGESTED/backfill` entries, `coverage.feature_restore.completed` metrics/logs, `dataset_metadata.json` alongside dataset artifacts, and `make validate-metrics validate-events` if validators are enabled.
- Partition hygiene:
  - `create_monthly_partitions` now copies rows out of `<table>_default` (skipping generated columns such as `spread`/`mid_price`) before attaching the new monthly partition, and treats duplicate/overlap SQLSTATEs as idempotent. This keeps bootstrap runs from failing on historical data left in the default partition.
  - When a dev database accretes orphan rows, run `SELECT create_monthly_partitions('market_data', '2023-01-01'::DATE, 60);` (or call `ml.common.db_utils.ensure_partition_tables_ready(...)`) to reattach the expected partitions without manual deletes.
  - `MLIntegrationManager` now calls `ensure_partition_helpers` during bootstrap so the refreshed `attach_partition_with_data` / `create_monthly_partitions` bodies deploy automatically even when migrations are already applied, ensuring Tier‑1 reruns inherit the default-partition rehousing logic.
- Runtime DB verification: when Docker services run overnight, confirm they are talking to the primary datastore by querying `ml_data_events` via `psql -h localhost -p 5433 -U postgres -d nautilus "SELECT dataset_id, instrument_id, stage, source, status, to_timestamp(ts_event/1e9) FROM ml_data_events ORDER BY created_at DESC LIMIT 10"`. Seeing fresh `EQUS.MINI` `INGESTED/backfill` events for Tier‑1 symbols (BAC, AMZN, CAT, etc.) ensures the pipeline did not write into the 5434 test instance.
- Events: Use DataRegistry `emit_event` with `Stage/Source/EventStatus` for pipeline phases.
- Validators:
  - `make validate-metrics`
  - `make validate-events`
  - `make validate-nautilus-patterns` (advisory)
- Parity harness: `make parity-report` regenerates `ml/tests/validation_reports/equs_itch_parity_summary.json` by executing the built-in Tier-1 suite (multiple symbols/windows across 2023–2025). Set `DATABENTO_API_KEY` in the environment before running.
- Manual verification/backfill tool: generate test ingestions directly with `ml.cli.pipeline_orchestrator --stage ingest` and inspect the resulting `source_dataset` tags to confirm coverage.

## Tier-1 orchestration evidence (2025-11-17)

- **Command:** `COVERAGE_DATASETS_FILE=ml/config/coverage_datasets_tier1.toml COVERAGE_RESTORE_ENABLED=1 poetry run python -m ml.cli.pipeline_orchestrator --config ml/config/pipeline_scheduler_example.toml` (cold-path run ID `orch_625eb3bc2266`; full console log stored at `ml_out/tier1_orchestrator_run2.log`).
- **Scope:** Tier-1 scheduler config with `VintagePolicy.REAL_TIME`, ingestion lookback 30 d, dual-write enabled (`write_mode = "sql+parquet"`), coverage manifest injected via `COVERAGE_DATASETS_FILE`.
- **Results:** Eleven symbols (AAPL, ABBV, ABT, ACN, ADBE, AMAT, AMD, AMZN, AVGO, BA, BAC) emitted `ingestion.symbol.completed` events and replayed alternating 95 k/200 k row windows into Postgres via `DataStoreMarketDataWriter`. Every parquet mirror attempt failed fast with `nautilus_trader/persistence/catalog`'s disjoint-interval assertion, so the `FanoutMarketDataWriter` currently degrades to SQL-only writes until catalog hygiene removes overlapping files.
- **Failure:** After processing BAC, the run aborted at the first symbol lacking Databento coverage (`SymbologyResolutionError: Symbol BRK not found in dataset EQUS.MINI`). The orchestration flow therefore requires (a) catalog compaction so parquet fan-out succeeds and (b) a symbology override/alias for BRK before the Tier-1 run can complete end-to-end.
- **Follow-ups:** keep `COVERAGE_RESTORE_ENABLED=1` so `coverage.feature_restore` hooks stay armed once adapters land, and document the log path/run ID alongside any fixes so reviewers can trace the parquet + symbology remediation.
- **Update (2025-11-18, run ID `orch_6dda9d5d543a`, log `ml_out/tier1_orchestrator_run3c.log`):** Running with catalog hygiene enabled plus the refreshed partition helpers allowed all Tier-1 symbols to stream into Postgres again while parquet fan-out logged `Parquet catalog write skipped due to overlapping interval` (acceptable degradation). The run now progresses past BRK (alias rewrite to BRK.B) and eventually fails when `update_watermark` attempts to record progress for HOOD because `EQUS.MINI` is missing from `ml_dataset_registry`, triggering a foreign-key violation. Register the dataset (via `ml.registry.bootstrap_datasets` or migration) before the next rerun so coverage + watermark updates can complete.
- **Update (2025-11-18, run ID `orch_57f422b18f31`, log `ml_out/tier1_orchestrator_run4.log`):** After pointing `NAUTILUS_DB`/registry env vars to the primary DB (`localhost:5433/nautilus`) and confirming `EQUS.MINI` is present in both registries, Tier‑1 ingestion again streamed every symbol into Postgres with parquet fan-out logging overlap skips. The pipeline now fails when Databento’s symbology resolver returns `502 <empty message>` for `DIS` on `EQUS.MINI`, which surfaces as `SymbologyResolutionError` and aborts the ingestion stage. Next actions: add resilience/retry logic (or bake DIS aliasing) around Databento 5xx responses, rerun once the API is stable, and capture the log for the successful end-to-end run.

## Promotion Gates

- Example gates JSON: `ml/config/promotion_gates_example.json`
- CLI flags wired via orchestrator or config (`promotions` section in TOML/JSON):
  - `auto_register_model`, `gates_json`, `auto_promote`, `deploy_target`
  - `auto_register_features`, `feature_metrics_json`, `refresh_features`

## Resilience

- The scheduler clears stale locks older than `ORCH_LOCK_TTL_HOURS`.
- DRY_RUN allows validating scheduling and argument flow without running model training.
- Orchestrator run failures emit FAILED events; use metric counters/histograms to track trends.

## Cache Hydration Helpers

Use the thin CLI `ml.cli.hydrate_feature_caches` to backfill or verify the per-minute cache layers before orchestrator runs. Example:

```bash
poetry run python -m ml.cli.hydrate_feature_caches \
  --symbols @tier1_full \
  --start-date 2025-08-04 \
  --end-date 2025-08-15 \
  --max-workers 4 \
  --micro \
  --l2 \
  --raw-dir data/tier1 \
  --micro-cache-dir data/features/micro_minute \
  --l2-cache-dir data/features/l2_minute
```

Flags such as `--force-micro` / `--force-l2` refresh existing partitions, while `--no-micro` or `--no-l2` scope the run to a single cache family. The CLI emits partition-level statistics (requested/written/skipped/empty/failures) so operators can confirm hydration parity before triggering the dataset build or orchestrator validation stages.

A Postgres DSN is now required; the CLI falls back to `DB_CONNECTION`, `NAUTILUS_DB`, or `DATABASE_URL` when `--dsn` is omitted. Every SQL write runs through the feature raw-writer so hydrated partitions immediately land in both the parquet cache (`data/features/{micro_minute,l2_minute}`) and the SQL mirrors (`ml.microstructure_minute`, `ml.l2_minute`). This keeps coverage automation green without waiting for a secondary ingest step.

`ensure_macro_ready` and `MLIntegrationManager.ingest_events` share the same dual-write plumbing. Their DataStore instances include the feature raw-writer, so `ml.events_calendar`, `ml.macro_release_calendar`, and `ml.macro_observations` receive SQL updates at the same time their parquet artifacts (`data/features/events/events.parquet`, `data/features/macro/**`) refresh.

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

## Data Hydration Playbook

### ALFRED Vintage Refresh

Keep `data/features/macro/fred/vintages/**` synchronized with ALFRED by running:

```bash
source .env
yes y | python scripts/download_alfred_vintages.py
```

The loader auto-loads the `.env` so the explicit `source` is defensive when running
outside Poetry. The 2025‑11‑16 refresh brought in the new commodity/metals series
(`PALLFNFINDEXM`, `PCOPPUSDM`, `NASDAQQGLDI`) after removing the legacy CRB/GOLD/SP500
entries from the macro universe. Those feeds have first-class ALFRED support, so
rerunning the command now writes full release calendars plus SQL dual writes via
`ensure_macro_ready`. After the refresh completes, re-run the dataset/audit jobs with
`VintagePolicy.REAL_TIME` and `fred_vintage_dir=data/features/macro/fred/vintages` to verify
macro feature hydration.

### Event / Calendar Refresh

Regenerate `data/features/events/events.parquet` whenever macro/event sources change:

```bash
PYTHONPATH=. python - <<'PY'
from datetime import UTC, datetime
from pathlib import Path
from ml.preprocessing.event_ingestion import EventIngestionConfig, EventIngestionUtility
series = tuple(sorted(p.name for p in Path('data/features/macro/fred/vintages').iterdir() if p.is_dir()))
cfg = EventIngestionConfig(
    start=datetime(2023, 1, 1, tzinfo=UTC),
    end=datetime(2025, 12, 31, tzinfo=UTC),
    out_dir=Path('data/features/events'),
    alfred_vintage_dir=Path('data/features/macro/fred/vintages'),
    economic_series=series,
    calendar_code='XNYS',
    include_options_expiry=True,
)
EventIngestionUtility(cfg).ingest()
PY
```

The 2025‑11‑08 refresh produced ~326k events covering 2023‑2025 with
`fed_meeting`, `economic_release`, `holiday`, `options_expiry`, and quarterly
`earnings` stubs. The dataset builder now has calendar/context features available
via `include_calendar=True`/`include_context_features=True` in
`tmp/feature_audit_build.py`.

### Earnings Ingestion

Hydrate the `ml.earnings_actuals` / `ml.earnings_estimates` tables and parquet
mirrors via:

```bash
ML_FILE_STORE_PATH=data/earnings_file_store \
SEC_USER_AGENT_NAME='your name' \
SEC_USER_AGENT_EMAIL='you@example.com' \
SEC_USER_AGENT_PHONE='555-555-5555' \
poetry run python -m ml.cli.ingest_earnings \
  --dsn postgresql://postgres:postgres@localhost:5432/nautilus \
  --parquet-root data/features/earnings_raw \
  --universe-mode tier1_full \
  --quarters 4
```

This dual-writes into Postgres plus `data/features/earnings_raw/earnings_{actuals,estimates}/`
using `EarningsParquetRawWriter`. Ensure the `update_watermark(...)` SQL function
exists (simple `INSERT ... ON CONFLICT` upsert into `ml_data_watermarks`) so the
DataStore can emit events without noisy errors, and keep `NAUTILUS_DB` in `.env`
pointing at the live DSN before running the audit harness.

You can also set `SEC_IDENTITY` directly if you prefer a single User-Agent string.
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata: {}

```

This starts the scheduler alongside a PostgreSQL container. Provide `CATALOG_PATH` via a bind mount when using `parquet` writers.
- Unified ingestion/backfill via config:
  - `ml/config/pipeline_scheduler_example.toml` now ships with a Tier‑1 example that
    handles ingestion, auto-fill backfilling, and dataset construction for the full TFT universe:

```toml
# ml/config/pipeline_scheduler_example.toml (excerpt)
stage = "full"

[dataset]
data_dir = "data/tier1"
out_dir = "ml_out/tier1_full_dataset"
symbols = "@tier1_full"        # expands to 95 Tier-1 instruments
include_macro = true
include_micro = true
include_l2 = true
include_events = true
include_calendar = true
include_calendar_lags = true
include_earnings = true
register_features = true
emit_dataset_events = true

[ingestion]
enabled = true
dataset_id = "EQUS.MINI"
schema = "bars"
instruments = "@tier1_full"
write_mode = "sql+parquet"
coverage_mode = "catalog"

[auto_fill]
enabled = true
dataset_id = "EQUS.MINI"
include_bars = true
include_tbbo = true
include_trades = true
include_l2 = true
l2_dataset_id = "DBEQ.BASIC"
l2_schema = "mbp-10"
instrument_ids = "@tier1_full"
```

Use `@tier1_full`, `@tier1_core`, etc. to expand symbol universes inline. The orchestrator
will ingest each schema, run catalog-based backfills via `auto_fill`, and then build the
full TFT dataset with the requested feature toggles.

### EQUS.MINI Provenance & ITCH Fallback

- `DatabentoIngestionService` now persists EQUS.MINI minute bars as received from Databento and tags each row with `source_dataset`. No trimming or scaling is performed off the hot path so cold/hot parity stays intact.
- When a requested window predates EQUS coverage the service replays the request against `XNAS.ITCH`, tags the rows with `source_dataset="XNAS.ITCH"`, and writes them unchanged. Registry lineage reflects the fallback dataset so downstream jobs can make regime-aware decisions.
- Fallback activations increment `ml_fallback_activations_total{component="databento_ingestion_service",level="itch_raw"}`. Operators should alarm on sustained fallback activations during live ingest.
- To spot-check provenance locally:

  ```bash
  DATABENTO_API_KEY=... uv run --active --no-sync python -m ml.cli.pipeline_orchestrator \
    --stage ingest --ingest --dataset_id EQUS.MINI --schema ohlcv-1m \
    --symbols INTC --lookback_days 2 --write_mode sql --coverage_mode catalog
  ```

  Inspect the resulting parquet/SQL rows and confirm `source_dataset` values match expectations (`EQUS.MINI` for native data, `XNAS.ITCH` for fallback windows).
