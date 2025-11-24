# ML Database Gap Detection & Restoration Plan

## Background

Recent outages showed that the ML pipeline (``ml_pipeline``) can restart while the
underlying PostgreSQL schema is stale or missing partitions.  We have now added:

- automatic migrations + schema health checks at pipeline startup
- a ``schema_audit`` CLI for pre-flight validation

However, the recovery workflow is still semi-manual: operators must reason about
which instruments or days are missing, whether the parquet catalog has the data,
and how to re-run Databento ingestion without duplicating work.  The goal is an
automated, repeatable flow that:

1. verifies the schema is healthy
2. identifies SQL coverage gaps per dataset/instrument/day
3. restores any missing rows from the parquet backup
4. triggers Databento ingestion for the remaining windows only

## Goals

- **Automated restoration loop:** every pipeline start (and optionally on a
  cron schedule) should run a “coverage manager” that inspects SQL vs. catalog
  coverage and orchestrates restoration.
- **Minimal Databento load:** if dual-write kept the parquet catalog current,
  only the most recent gaps should hit Databento; older holes should be filled
  from the catalog.
- **Deterministic reporting:** every run should produce metrics and structured
  logs that describe how many instruments/days were restored from catalog vs.
  re-ingested from Databento.
- **Operator escape hatch:** expose a CLI (`python -m ml.stores.coverage_manager …`)
  so ops can run the same workflow manually against specific datasets.
  For environment-driven recovery, wire a thin CLI wrapper (`python -m ml.cli.coverage_restore`)
  that bootstraps the pipeline config and executes a single restoration pass.

Non-goals:

- Rewriting `ParquetCatalogRehydrator` into a general ETL tool.
- Building UI dashboards; text logs + Prometheus metrics are sufficient for now.

## Current State

Component | Status | Gaps
--------- | ------ | ----
`ml.stores.migrations_runner` | Runs automatically at pipeline startup | Assumes DB already matches canonical partitioned schema
`ml.stores.schema_audit` | Manual CLI to validate schema | No tie-in to pipeline; only yes/no
`ParquetCatalogRehydrator` | Restores a lookback window from parquet → SQL | No per-bucket inspection; will retry every symbol/day even if SQL already has coverage
`DataScheduler` | Handles Databento ingestion/backfill | Backfill window is coarse (lookback days); doesn’t know which buckets are already satisfied

## Proposed Architecture

### 1. Coverage Manager Service

Create `ml.data.coverage.manager` with two primary responsibilities:

1. **Gap detection:** For each configured dataset/instrument, query SQL coverage
   (e.g., `SELECT MIN(ts_event), MAX(ts_event)` per day/instrument) and compare
   with parquet catalog metadata (`catalog.get_partition_bounds` or scanned
   manifests).  Output a normalized set of day buckets:

   ```json
   {
     "instrument_id": "AAPL.XNAS",
     "dataset": "EQUS.MINI_TBBO",
     "schema": "tbbo",
     "bucket": "2024-07-15",
     "status": "missing_sql_has_parquet" | "missing_both"
   }
   ```

2. **Restoration orchestration:** Split the bucket list into:
   - `catalog_restorable`: buckets where parquet has full data
   - `databento_required`: buckets missing from both SQL and parquet

   Feed the first list into `ParquetCatalogRehydrator` (with tighter targeting)
   and the second list into a new `targeted_backfill` entry point on
   `DataScheduler`.

### 2. Enhancements to Existing Components

- **ParquetCatalogRehydrator**
  - Accept a `buckets` argument that explicitly lists day + instrument combos so
    we avoid rehydrating already-satisfied data.
  - Emit metrics such as
    `catalog_rehydrate.buckets_considered_total{source="coverage_manager"}`.

- **DataScheduler**
  - Expose `run_targeted_update(buckets: Sequence[BucketSpec])` which ingests
    only the residual Databento buckets.
  - Ensure metric/telemetry parity with the existing `run_daily_update`.

- **Pipeline Runner**
  - After migrations succeed but before catalog rehydration, instantiate the
    coverage manager:

    ```
    coverage_manager = CoverageManager.from_env(config)
    coverage_manager.restore_all()
    ```

  - If coverage restoration fails, log with `exc_info=True` and mark the health
    endpoint unhealthy; the pipeline should not proceed to ingestion until
    coverage is reconciled or explicitly skipped (guarded by env flag).

### 3. Workflow Summary

1. Startup:
   - run migrations
   - `CoverageManager.restore_all()`:
     - runs schema audit (fail fast if schema invalid)
     - computes gaps
     - restores catalog buckets
     - launches targeted Databento ingestion for residual gaps
   - logs summary + metrics
2. Normal ingestion loop:
   - `run_daily_update` continues as today (dual-write keeps SQL + parquet in sync).
3. Manual CLI (`python -m ml.data.coverage.manager restore --db-url … --catalog …`)
   for remediation outside of pipeline container.

## Implementation Plan

Phase | Deliverables
----- | -----------

1. **Schema audit integration** | ✅ `CoverageManager` now runs `SchemaAuditor` before classification; pipeline bootstraps migrations + schema audit at startup.
2. **Coverage metadata helpers** | ✅ SQL/catalog coverage providers handle bucket queries; classification emits per-bucket specs (`BucketSpec` + `BucketClassification`).
3. **Coverage Manager orchestration** | ✅ `CoverageManager` + pipeline wiring classify gaps, restore catalog buckets, run targeted scheduler updates, and update `/health` coverage summaries. Prometheus metrics (`nautilus_ml_coverage_*`) record counts, failures, and latency.
4. **Scheduler + rehydrator extensions** | ✅ `DataScheduler.run_targeted_update` and `ParquetCatalogRehydrator.rehydrate_missing_data(..., buckets=…)` accept targeted bucket sets; unit tests cover both.
5. **Pipeline wiring** | ✅ Entry-point builds dataset configs from `SchedulerConfig`, toggles via `COVERAGE_RESTORE_ENABLED`, publishes health metadata, and exposes a standalone CLI (`python -m ml.cli.coverage_restore`) for out-of-band runs.
6. **Operational runbook** | ✅ `ML_DEPLOYMENT_README.md`, `ml/deployment/README.md`, this plan, and `ml/docs/ops/dashboard_runbook.md` document the workflow, metrics, env vars, and operator tooling (including the coverage CLI and health payload fields).

## Metrics & Observability

Metric | Purpose
------ | -------
`nautilus_ml_coverage_buckets_total{status=…,source=…}` | Count of buckets classified as restorable/missing
`nautilus_ml_coverage_restore_failures_total{stage=catalog|databento}` | Restoration errors with labels for dataset/symbol
`nautilus_ml_coverage_latency_seconds` | Duration of complete restoration pass

Logs should include structured entries (`coverage_manager.detected`, `coverage_manager.catalog_restore`, `coverage_manager.databento_restore`) with `instrument_id`, `bucket`, and counts to simplify debugging.

## Risks & Mitigations

- **Large backlog**: Restoring an entire year of data could take many hours.
  - Mitigation (implemented): `COVERAGE_MAX_BUCKETS_PER_RUN` (default 500) caps how many buckets the pipeline restores per pass. When residual gaps remain, the pipeline logs `coverage_manager.bucket_cap_applied`, surfaces an error entry in `/health`, and leaves the remaining buckets for the next run.
- **Parquet/catalog mismatch**: If the catalog is also missing data, the manager
  falls back to Databento via `run_targeted_update`. We now emit `coverage_manager.catalog_restore_failed` logs and increment `nautilus_ml_coverage_restore_failures_total{stage="classification|catalog|targeted_update"}` so ops can alert on repeated failures.
- **Dual-write assumptions**: Need to verify that dual-write truly keeps parquet
  current; otherwise we may still need periodic catalog validation jobs. Telemetry (`nautilus_ml_coverage_buckets_total{status=…}`) highlights when RESTORE_FROM_CATALOG vs. REINGEST_FROM_SOURCE skews heavily, signaling dual-write drift.
- **Instrumentation schema drift**: The pipeline emits coverage + ingestion events into `ml_data_events`, but recent refreshes skipped `ml/stores/migrations_runner`, leaving the table absent. Remediation: pin migrations as a hard precondition for `PipelineRunner.run()` and surface a health error when instrumentation tables are missing so operators can’t ignore the issue.
- **Databento dataset/schema mismatches**: Daily updates currently request unsupported datasets (`DBEQ.MINI`) and schemas (`mbp-10` on `EQUS.MINI`), causing ingest failures before data collection starts. Add config validators (and CLI guards) that reject unsupported combinations up front, and update `MARKET_DATASET_INPUTS` defaults/tests to cover only valid pairs.
- **Missing Databento dependency in runtime containers**: `ml_signal_actor` and `ml_strategy` images lack the `databento` package, so any refresh relying on Databento APIs crashes immediately. Bake the dependency into the Dockerfiles or gate Databento usage behind feature flags until those images are rebuilt.

## Runtime Findings & Remediation (Nov 2024)

- **Instrumentation tables missing** – The pipeline now enforces migrations by checking for `ml_data_events` and `ml_data_watermarks` immediately after `MigrationRunner` executes. Missing tables raise a `SchemaHealthCheckError` so scheduler activity never starts without telemetry.
- **Invalid Databento configs** – `_parse_market_dataset_inputs` validates every descriptor/dataset/schema pair against `ml/config/market_feed_descriptors.json`. Unsupported descriptors (e.g., `DBEQ.MINI`) or overrides (`mbp-10` on `EQUS.MINI`) raise `ValueError` early and surface in `/health["errors"]`/CLI output so ops fix the env vars before rerunning the pipeline.
- **Runtime containers lacked `databento`** – `ml/deployment/Dockerfile.actor` and `Dockerfile.strategy` now install the `databento` wheel. Rebuild the images whenever these changes are pulled so Databento imports succeed in production.
- **Migrations blocked by inline semicolons** – Several SQL migrations (`007_schema_hardening.sql`, `011_brin_indexes.sql`, `012_predictions_alias.sql`) contained `-- comment; trailing text` patterns. Our splitter treats the fragment after `;` as executable SQL, so re-running migrations (or invoking the coverage CLI) crashed with `SyntaxError: ts_event included` / `uses IF NOT EXISTS`. Comments have been rewritten without inline semicolons so `poetry run python -m ml.stores.migrations_runner apply ...` is safe to run repeatedly.
- **Local coverage CLI needs host paths** – Running `ml.cli.coverage_restore` outside Docker attempted to create `/app/data/catalog`, which fails on a workstation. Set `CATALOG_PATH=$PWD/data/catalog DB_CONNECTION=postgresql://postgres:postgres@localhost:5433/nautilus` before invoking the CLI so it uses the host catalog path and forwarded Postgres port.
- **Checksum drift when rerunning migrations locally** – Developers now gate checksum enforcement behind `ML_ALLOW_MIGRATION_DRIFT=1`. When that env var is set, `MigrationRunner` logs the mismatch, updates `ml_schema_migrations`, and continues instead of aborting. Use it only when the diff is comment-only (like the semicolon fix); prod containers should run without the flag so real schema edits still fail fast. For the running pipeline container, exec into the service and rerun bootstrap with the flag so it refreshes its cached metadata without a rebuild:

  ```bash
  docker compose -f ml/deployment/docker-compose.yml exec ml_pipeline sh -c '
      ML_ALLOW_MIGRATION_DRIFT=1 PYTHONPATH=/app python - <<\"PY\"
  from ml.deployment.entrypoint_pipeline import PipelineRunner
  runner = PipelineRunner()
  cfg = runner._create_config()
  runner._bootstrap_database(cfg)
  PY'
  ```

- **Databento collection failing on timezone math** – The scheduler computed freshness via `datetime.now() - target_date`, mixing naive and UTC-aware datetimes and aborting every catalog write. `_collect_symbol_data` now calls `datetime.now(tz=UTC)`, and a regression test exercises the code path with an aware `target_date`.
- **SQL ingest stalled when orchestrator flags were omitted** – Running `docker compose -f ml/deployment/docker-compose.yml up ml_pipeline` skips the default override, so `USE_ORCHESTRATOR`/`DUAL_WRITE` stayed `false` and the scheduler reverted to the legacy catalog-only path (`Writing … records to catalog` logs without any SQL writes). The base compose file now defaults both env vars to `1` (still overrideable) so containers always execute the orchestrator + dual-write path, and `DataScheduler.run_targeted_update` routes coverage-driven buckets through the orchestrator whenever it is enabled so SQL + catalog stay in sync.
- **Weekend ingestion guardrails** – The orchestrator previously treated “look back 1 day” literally, so Saturday/Sunday pipeline runs tried to fetch nonexistent weekend buckets and failed with Databento 422 `data_start_after_available_end` errors before SQL ingestion could begin. `DataScheduler` now expands the orchestrator lookback to cover the prior *trading* day (using `TradingDayCalculator`) so weekend runs target Friday data. Unit tests cover weekday/Sunday/Monday scenarios to prevent regressions.
- **Automatic catalog lookback** – Operators no longer need to guess `MARKET_BACKFILL_LOOKBACK_DAYS`. `DataScheduler` inspects the Parquet catalog (via `CatalogCoverageProvider`) and expands the orchestrator lookback to cover the earliest bucket on disk—currently ~960 days (Mar 2023). Coverage restoration and daily runs now walk the entire available history without manual config changes, and the expansion is logged as `scheduler.orchestrator.lookback_expanded`.

## Open Questions

1. Do we need to coordinate with the partition manager when creating new
   partitions during restoration, or will migrations already provision enough?
   - Current approach relies on migrations ensuring partitions exist; still open whether we need dynamic partition creation for historical replays beyond the default window.
2. Should we persist coverage snapshots (e.g., in Redis or a table) so we can
   surface them in dashboards, or is on-demand calculation sufficient?
   - `/health` + metrics provide run-level visibility, but persistent history would make SLO tracking easier; TBD.
3. How should we handle schema evolution for parquet data (e.g., new columns)?
   Do we block restoration if schemas diverge, or attempt column alignment?
   - Schema audit currently fails fast; deciding whether to auto-align columns remains future work.
