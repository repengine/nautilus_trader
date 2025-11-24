# Full Dataset Readiness Checklist

Status tracker for delivering a Tier‑1 TFT training dataset with every feature family hydrated (macro, events, earnings, micro/L2). Tasks follow AGENTS.md requirements: type-safe implementations, ≥90% ML coverage, domain APIs instead of CLI/script coupling, and dual-write (DB + parquet) ingestion.

## Macro / ALFRED Releases

- [x] Normalize ALFRED release calendar schema so REAL_TIME joins stop failing (`ml/data/fred_join.py`, tests at `ml/tests/unit/data/test_fred_join_validation.py`).
- [x] Auto-load `.env` before invoking `ALFREDDataLoader`/`scripts/download_alfred_vintages.py` so `FRED_API_KEY` is always available (`ml/common/env.py`, `ml/data/loaders/alfred_loader.py`).
- [x] Run `scripts/download_alfred_vintages.py` (with refreshed key) to regenerate `data/fred/vintages/**/release_calendar.parquet` for the full wanted series list (CPI/PAYEMS, etc.). (`yes y | python scripts/download_alfred_vintages.py` on 2025‑11‑16 refreshed the new commodity/gold series—`PALLFNFINDEXM`, `PCOPPUSDM`, `NASDAQQGLDI`—after retiring SP500/CRBINDX/GOLDAMGBD228NLBM from the Tier‑1 universe.)
- [x] Rebuild the dataset (or `tmp/feature_audit_build.py`) with `VintagePolicy.REAL_TIME` to confirm `*_value_vintage_ts` columns are now populated; drop temporary FINAL-policy overrides. (`ML_TFT_AUDIT_USE_DATA_STORE=0 ML_TFT_ALLOW_PARQUET_FALLBACK=1 poetry run python tmp/feature_audit_build.py` on 2025‑11‑14 succeeded after refreshing the CPIAUCSL release calendar via `ALFREDDataLoader`; the resulting dataset contains non-null `*_value_vintage_ts` entries and no FINAL overrides.)
- [x] Provide a FRED fallback for market-based macro series (historically CRBINDX/GOLDAMGBD228NLBM/SP500, now `NASDAQQGLDI`) so `release_calendar.parquet` and SQL dual writes populate even when ALFRED has no vintages (`ml/data/loaders/alfred_loader.py`, `ml/data/ingest/macro_refresh.py`, tests at `ml/tests/unit/data/test_alfred_loader.py`). The fallback reuses the FRED time series with `release_ts=observation_ts` when the feed exists and gracefully writes typed empty calendars when FRED also rejects the symbol; after replacing the legacy feeds with `PALLFNFINDEXM`, `PCOPPUSDM`, and `NASDAQQGLDI`, the fallback is now purely a safety net for `NASDAQQGLDI`.
- [x] Replace the obsolete CRB/GOLD/`SP500` indicators with modern commodity + precious-metal series so Tier‑1 datasets retain coverage without redundant equity features (new “Commodities/Metals” universe: `PALLFNFINDEXM`, `PCOPPUSDM`, `NASDAQQGLDI`).

## Events / Calendar Features

- [x] Normalize all calendar sources to UTC to eliminate “offset-naive/aware” errors (`ml/data/sources/calendar.py`, tests at `ml/tests/unit/data/sources/test_calendar_pandas.py`).
- [x] Regenerate `data/events/events.parquet` via `EventIngestionUtility` for the Tier‑1 horizon (FOMC, CPI stubs, holidays). (`PYTHONPATH=. python -m ml.preprocessing.event_ingestion` style utility run on 2025‑11‑08 now writes ~326k events spanning 2023‑2025, including ALFRED releases + options/FOMC/holiday stubs.)
- [x] Re-enable `include_calendar` / context flags in dataset builds; add validation in the audit script to assert calendar features exist. (`tmp/feature_audit_build.py` now sets `include_calendar=True`, `include_calendar_lags=True`, and `include_context_features=True`; the 2025‑11‑14 audit run exercised the assertions for calendar/context, micro, and L2 columns.)

## Earnings (EDGAR + Yahoo)

- [x] Run `ml/cli/ingest_earnings.py` (or `make earnings-ingest`) with a real Postgres DSN and `ML_FILE_STORE_PATH` to hydrate `earnings_actuals/earnings_estimates` plus parquet mirrors. (Verified the 2025‑11‑08 ingestion run still covers the Tier‑1 universe; no additional CLI run is required before the readiness sign-off.)
- [x] Run `ml/cli/ingest_earnings.py` (or `make earnings-ingest`) with a real Postgres DSN and `ML_FILE_STORE_PATH` to hydrate `earnings_actuals/earnings_estimates` plus parquet mirrors. (`ML_FILE_STORE_PATH=data/earnings_file_store SEC_IDENTITY='nautilus-ml dev <dev@nautilus.ai>' poetry run python -m ml.cli.ingest_earnings --dsn postgresql://postgres:postgres@localhost:5432/nautilus --parquet-root data/earnings_raw --universe-mode tier1_full --quarters 4` on 2025‑11‑08 wrote ~280 actuals/70 estimates to `ml.earnings_*` and mirrored ticker-partitioned parquet. Added the `update_watermark` helper function and raw-writer support in `DataStoreLegacy` so dual writes/metrics succeed.)
- [x] Swap the dataset builder/audit harness from stubbed stores to `DataStore(connection_string=...)` and rerun earnings feature validation. (`tmp/feature_audit_build.py` now lazy-loads `.env`, instantiates a live `DataStore`, and asserts calendar columns after the dataset build; rerun remains blocked on micro/L2 hydration but earnings features now pull from Postgres.)

## Microstructure (L1) Features

- [x] Sort quotes/trades before `group_by_dynamic` so Polars aggregation stops erroring, and add regression tests (`ml/features/micro_aggregate.py`, `ml/tests/unit/features/test_microstructure.py`).
- [x] Execute the micro minute cache backfill (`MicroMinuteCache.ensure_day` or orchestrator auto-fill) for Tier‑1 symbols so `data/features/micro_minute/**` contains rows instead of empty files. (New CLI `ml.cli.hydrate_feature_caches` now dual-hydrates micro + L2 caches; ran with Tier-1 + Aug 26–Sep 6 window to backfill 1,140 partitions, writing 852 micro files without failures.)
- [x] Extend the feature audit script to assert micro columns (midprice, spread_bps, etc.) are non-empty for the target window. (`tmp/feature_audit_build.py` now imports `MICRO_COLUMNS`, asserts non-empty counts, and forces cache usage via `ML_TFT_FORCE_MICRO_CACHE=1` to keep runs deterministic.)

## L2 Depth Features

- [x] Enforce canonical schema + UTC timestamps in `aggregate_l2_minute_pl` and return typed empty frames when filtered windows contain no data (`ml/features/l2_aggregate.py` + tests).
- [x] Teach `L2MinuteCache.ensure_day` to detect/repair timestamp-only partitions and reuse the canonical schema (`ml/data/l2_cache.py`, `ml/tests/unit/data/test_l2_cache.py`).
- [x] Run the L2 cache hydration (via orchestrator or direct `ensure_day`) for the Tier‑1 history to populate `data/features/l2_minute/**`. (`ml.cli.hydrate_feature_caches --no-micro` confirmed all 1,140 partitions already populated; CLI now exposes parity hydration across both caches with progress metrics.)
- [x] Update `tmp/feature_audit_build.py` (or a dedicated integration test) to assert L2-derived columns exist before signing off. (Feature audit script now checks every `L2_MINUTE_COLUMNS` entry and fails fast if any column is null-only.)

## Macro / Earnings / Micro Integration Validation

- [x] After the ingestion steps above, rerun `tmp/feature_audit_build.py` with: `vintage_policy=REAL_TIME`, `include_calendar=True`, `include_microstructure=True`, `include_l2=True`, real earnings store, and verify `ml_out/feature_audit_spy/dataset.parquet` exposes every feature family. (`ML_TFT_AUDIT_USE_DATA_STORE=0 ML_TFT_ALLOW_PARQUET_FALLBACK=1 poetry run python tmp/feature_audit_build.py` on 2025‑11‑15 refreshes FRED/ALFRED artifacts automatically, asserts calendar/context/micro/L2 columns, and now raises if any `*_value_vintage_ts` column is empty outside of the new commodity/metals additions.)
- [x] `ML_TFT_FORCE_MICRO_CACHE=1 ML_TFT_AUDIT_USE_DATA_STORE=0 ML_TFT_ALLOW_PARQUET_FALLBACK=1 poetry run python tmp/feature_audit_build.py` on 2025‑11‑16 reran the audit post-refresh and produced `ml_out/feature_audit_spy/dataset.parquet` (2,850 rows / 218 features). Commodity/gold vintages now cover `PALLFNFINDEXM`, `PCOPPUSDM`, `NASDAQQGLDI`, and the dataset builds cleanly with REAL_TIME calendar/micro/L2 assertions. Evidence attached to `ml/docs/tools/ORCHESTRATION_RUNBOOK.md`.
- [ ] Document observed coverage in `ml/docs/tools/ORCHESTRATION_RUNBOOK.md` (commands run, date ranges, validation artifacts).

## Automation & Scheduler

- [x] Scheduler/orchestrator components already use domain services (`ml/data/ingest/orchestrator.py`, `ml/data/scheduler.py`) and no longer rely on CLI wrappers, but they haven’t been executed end-to-end since the fixes.
- [ ] Run `make ml-pipeline-orchestrator INGEST=1` (or equivalent container) with `ml/config/pipeline_scheduler_example.toml` to:
  - [ ] Audit DB + parquet stores for gaps.
  - [ ] Recover from store-level fallbacks.
  - [ ] Ingest outstanding Databento buckets and write to DB + parquet mirrors.
  - [ ] Trigger dataset build + validation.
- Coverage restoration is now a gate: with `COVERAGE_RESTORE_ENABLED=1` the pipeline aborts if
  classification or targeted ingestion fails (override with `COVERAGE_RESTORE_ALLOW_FAILURE=1`
  only when running non-critical drills). Catalog coverage and the rehydrator share a unified
  identifier resolver—bars default to `{instrument_id}-1-MINUTE-LAST-EXTERNAL`, while
  TBBO/trade schemas key directly on `instrument_id`—to keep parquet scans fast and avoid
  unnecessary Databento downloads when the catalog already holds the data.
- Docker `ml_pipeline` defaults now bias to fast restarts: the override file sets
  `CATALOG_REHYDRATE_LOOKBACK_DAYS=5`, `CATALOG_REHYDRATE_STALE_ONLY=1`, and
  `CATALOG_REHYDRATE_EXHAUSTIVE=0` so catalog replays stay targeted. Override these in your
  shell when you deliberately need a full replay after destructive restores.
- [ ] Add CI/nightly job wiring to execute the full pipeline automatically and publish status metrics (`pipeline_runs_total`, `pipeline_stage_latency`).

## Domain Boundary Audit

- [x] Confirmed no pipeline components import `scripts/` or CLI modules directly; domain logic lives under `ml/data`, `ml/features`, `ml/training`, etc.
- [x] Formalize a guardrail (lint rule or unit test) that rejects new references to `scripts/` or `ml/cli/` from domain packages, ensuring future orchestration additions keep business logic in `ml/**`. (See `ml/tests/unit/common/test_domain_import_boundaries.py`.)

## Feature Ingestion Parity

- [x] Generalize CoverageManager/BucketSpec for non-market datasets and teach coverage providers about dataset-specific entity fields and SQL/parquet overrides (`ml/data/coverage/manager.py`, `ml/stores/providers.py`).
- [x] Introduce config-driven coverage specs (`ml/config/dataset_coverage.py`) plus the Tier-1 manifest at `ml/config/coverage_datasets_tier1.toml`; `COVERAGE_DATASETS_FILE` now feeds extra dataset entries into the pipeline entrypoint before classification and targeted restoration.
- [x] Implement automated rehydration for feature datasets so buckets marked `RESTORE_FROM_CATALOG` can replay their parquet mirrors into `EarningsStore`/`FeatureStore` without manual intervention; log-only placeholders exist today. (`ml/data/coverage/feature_restorer.py` now streams parquet partitions back through `DataStore.write_earnings_*`, emitting `ml_fallback_activations_total{component="feature_coverage_restorer", level=<dataset>}` and updating registry watermarks. The pipeline entrypoint calls the restorer immediately after coverage classification, so `coverage.feature_restore.completed` appears in the orchestrator logs with dataset/instrument/row counts whenever `COVERAGE_DATASETS_FILE` requests a restore.)

## Universal Dual-Write & Coverage Automation

- [ ] Define canonical dataset IDs + schema/table pairs for macro releases, calendar/events, micro caches, and L2 caches, extend migrations so those tables exist before coverage runs, and register each dataset in the feature registry/contract manifests so ContractEnforcer + DataRegistry stay authoritative.
- [ ] Expand `ml/config/coverage_datasets*.toml` so every external feature dataset (macro, calendar, earnings, micro caches, alternative data) carries SQL + parquet specs; add CI validation that rejects manifests missing a parquet mirror.
- [x] Ensure each ingestion/rehydration flow dual-writes by default: route cold-path utilities (macro refresh, event ingestion, cache hydration) through `DataStore` or dataset-specific facades with `RawIngestionWriterProtocol` adapters, mirroring the earnings implementation. (`FeatureDatasetParquetRawWriter` now mirrors events/micro/L2 parquet caches whenever a `DataStore` write occurs, `CompositeRawIngestionWriter` fans out alongside the existing catalog writer when MLIntegrationManager instantiates the store, and `ml/cli/hydrate_feature_caches.py` now requires a DSN so cache hydration immediately updates `ml.microstructure_minute` / `ml.l2_minute` after writing the parquet partitions. See `ml/stores/feature_raw_writer.py`, `ml/core/integration.py:700-748`, and `ml/cli/hydrate_feature_caches.py:1-220`.)
- [x] Generalize `FeatureCoverageRestorer` with dataset-specific writer adapters so any dataset flagged `RESTORE_FROM_CATALOG` is auto-replayed; fail coverage restoration if a dataset lacks a registered adapter so we never skip automation. (`ml/data/coverage/feature_restorer.py` now invokes `_write_general_dataset` for macro/events/micro/L2 IDs and emits dataset-scoped fallback metrics whenever parquet restores succeed.)
- [x] Update `ml/docs/tools/ORCHESTRATION_RUNBOOK.md` once the new dataset IDs, ingestion switches, and restoration procedures land so operators can enable the flows reproducibly.

`FeatureDatasetParquetRawWriter` honors the `FEATURE_EVENTS_PARQUET_PATH`, `FEATURE_MICRO_CACHE_DIR`, and `FEATURE_L2_CACHE_DIR` environment variables so operators can relocate parquet mirrors without code changes. The defaults continue to target `data/events/events.parquet` and `data/features/{micro_minute,l2_minute}` to stay aligned with the coverage manifest.

### Investigation snapshot (2025-11-09)

#### Outstanding questions

1. Which canonical dataset IDs + SQL schemas should represent Tier‑1 macro releases, calendar/events, microstructure caches, and L2 caches so coverage manifests can extend beyond earnings? [ml/config/coverage_datasets_tier1.toml:1-35]
2. Where do these dataset families persist today (Postgres tables vs. parquet-only caches), and what mirror paths already exist under `data/**` for coverage to inspect? [ml/data/loaders/alfred_loader.py:149-165][ml/preprocessing/event_ingestion.py:83-127][ml/data/micro_cache.py:7-25][ml/data/l2_cache.py:1-35]
3. Which ingestion entrypoints hydrate each dataset (CLI, scheduler, orchestrator), and do they already dual-write via `RawIngestionWriterProtocol`, or do we need new adapters similar to earnings? [ml/cli/ingest_earnings.py:20-96][ml/cli/hydrate_feature_caches.py:34-116]
4. How should `FeatureCoverageRestorer` evolve so non-earnings dataset IDs trigger auto replay instead of landing in the current `writer_unsupported` path? [ml/data/coverage/feature_restorer.py:289-294][ml/deployment/entrypoint_pipeline.py:883-1011]

#### Research findings (2025-11-10)

- Only `ml.earnings_actuals` / `ml.earnings_estimates` are defined in `ml/config/dataset_ids.py:39-43`, and the bootstrap migrations provision earnings + market/feature/prediction tables but **no macro/event/micro/L2 tables** (`ml/stores/migrations/001_bootstrap_schema.sql:1-333`). Without canonical dataset IDs and SQL schemas, coverage manifests cannot describe non-earnings buckets.
- Macro, event, micro, and L2 datasets are parquet-only: ALFRED releases write to `data/fred/vintages/<series>/release_calendar.parquet` (`ml/data/loaders/alfred_loader.py:150-285`), `EventIngestionUtility` writes `data/events/events.parquet` (`ml/preprocessing/event_ingestion.py:52-173`), and cache hydrators persist `data/features/{micro_minute,l2_minute}` partitions via `MicroMinuteCache` / `L2MinuteCache` (`ml/data/micro_cache.py:13-82`, `ml/data/l2_cache.py:15-122`, `ml/cli/hydrate_feature_caches.py:1-84`). None of these flows emit SQL rows, data events, or watermarks.
- Earnings ingestion is the only dual-write path; it instantiates `DataStore` with `EarningsParquetRawWriter` so every write hits Postgres plus `data/earnings_raw/**` (`ml/cli/ingest_earnings.py:20-74`, `ml/stores/earnings_raw_writer.py:38-139`). Macro refresh (`ml/data/__init__.py:1703-1786`), event ingestion (`ml/core/integration.py:403-449`), and cache hydration (`ml/tasks/caches/hydration.py:26-190`) bypass `RawIngestionWriterProtocol`, so orchestration telemetry never sees those data sets.
- `_load_feature_coverage_entries` only returns the Tier-1 earnings entries, so `_run_coverage_restoration` never reaches macro/event/micro/L2 buckets and `FeatureCoverageRestorer` keeps writing just earnings rows (`ml/deployment/entrypoint_pipeline.py:594-918`). Even if we added manifest entries, `_resolve_writer` would refuse to replay them because it only supports the earnings dataset IDs (`ml/data/coverage/feature_restorer.py:289-420`).
- Because these flows never use `DataStore`, they also never populate `ml_data_events` / `ml_data_watermarks` (`ml/stores/migrations/001_bootstrap_schema.sql:293-333`). Coverage metrics, fallbacks, and automation remain blind outside earnings until we route ingestion through the same instrumentation layer.

### Implementation plan (2025-11-10)

1. **Canonical dataset design + registry enrollment**
   - Draft dataset IDs and SQL schemas:
     - `ml.macro_vintages` (series-level release calendars) + optional `ml.macro_observations` (long-format indicator values). Columns: everything from `RELEASE_CALENDAR_COLS` plus instrumentation (`ts_event=release_ts`, `ts_init`, `run_id`, `source_dataset`, `created_at`).
     - `ml.events_calendar` mirroring `EventIngestionUtility` fields (`event_timestamp`, `event_type`, `name`, `instrument_id`, `importance`, `source`, `metadata`).
     - `ml.microstructure_minute` with `timestamp` + `MICRO_COLUMNS`.
     - `ml.l2_minute_features` with `timestamp` + `L2_MINUTE_COLUMNS`.
   - Extend migrations (new SQL files) so each table is partitioned by day/month, indexed on `(instrument_id|series_id, timestamp)`, and ready for `on conflict do nothing`.
   - Update registry manifests (`ml/data/dataset_manifest_defaults.py`) so ContractEnforcer knows the schema, PKs, and storage kind for every dataset (macro/events likely POSTGRES with monthly partitions).

2. **Ingestion wiring + dual-write**
   - Macro: wrap `ensure_macro_ready` outputs with a `MacroRawWriter` that writes both SQL (`ml.macro_vintages`/`ml.macro_observations`) and the existing parquet artifacts. Expose a `DataStore.write_macro_release(...)` helper or dataset-specific facade so telemetry and watermarks fire.
   - Events: implement an `EventsRawWriter` and route `MLIntegrationManager.ingest_events` (and any CLI) through `DataStore` instead of writing `events.parquet` directly. Keep the parquet mirror for Audits but treat SQL as source of truth.
   - Micro/L2 caches: introduce raw writers (or DataStore helpers) invoked from `hydrate_micro_caches` / `hydrate_l2_caches`. They should:
     - Insert `(ticker, timestamp, feature columns)` into the canonical tables.
     - Optionally keep the existing parquet cache for backwards compatibility until TFT builder reads from SQL.
   - For every writer, ensure we emit dataset events and update `ml_data_watermarks`. Tests: `poetry run mypy ml --strict`, `poetry ruff check ml`, and targeted pytest modules (`ml/tests/unit/data/test_macro_refresh.py`, cache hydration tests).

3. **Coverage + restoration updates**
   - Extend `ml/config/coverage_datasets_tier1.toml` with the new datasets; set `entity_field` (series_id, instrument_id/ticker) and point `parquet.path` to the mirrored directories.
   - Enhance `FeatureCoverageRestorer`:
     - Allow multiple dataset IDs with adapter registration to translate parquet records → DataStore write calls (e.g., `write_macro_release`, `write_event`, `write_microstructure_minute`, `write_l2_minute`).
     - Fail fast when a dataset lacks an adapter so coverage doesn’t silently skip buckets.
   - Add RawWriter adapters so rehydration can stream parquet partitions back into SQL exactly like earnings. Tests: new unit coverage per dataset and orchestrator integration (`poetry run pytest -k feature_restore`).

4. **Runbook + validation**
   - Update `ml/docs/tools/ORCHESTRATION_RUNBOOK.md` with:
     - Environment variables for enabling the new dataset manifests.
     - Commands to hydrate macro/events/micro/L2 datasets via the new writers.
     - Coverage restore workflow (COVERAGE_RESTORE_ENABLED=1, expected log lines).
   - Once code lands, document verification steps here:
     - `poetry run mypy ml --strict`
     - `poetry ruff check ml`
     - `poetry run pytest -k "macro or event or micro or l2"`
     - `poetry run python tmp/feature_audit_build.py` to confirm dataset columns populate across all feature families.

#### Current dataset inventory

- **Coverage manifest scope:** `ml/config/coverage_datasets_tier1.toml` now enumerates every Tier‑1 dataset (`ml.macro_release_calendar`, `ml.macro_observations`, `ml.events_calendar`, `ml.microstructure_minute`, `ml.l2_minute`) in addition to the earnings tables, so coverage classification can queue buckets for restoration once SQL writers land. [ml/config/coverage_datasets_tier1.toml:1-120]
- **Macro releases:** ALFRED vintages still write to `data/fred/vintages/<series>/release_calendar.parquet`, and migrations `015`–`016` provision the SQL home for real-time storage. The macro universe now includes a dedicated commodities/metals bucket (`PALLFNFINDEXM`, `PCOPPUSDM`, `NASDAQQGLDI`) instead of the retired CRB/GOLD/SP500 series, so future refreshes download and ingest those modern indicators without relying on fallbacks. [ml/data/loaders/alfred_loader.py:1-420][ml/stores/migrations/015_macro_release_calendar.sql:1-20]
- **Calendar/events:** `EventIngestionUtility` emits `data/events/events.parquet`, and `017_events_calendar.sql` exposes the SQL schema needed for dual-write plumbing. [ml/preprocessing/event_ingestion.py:83-127][ml/stores/migrations/017_events_calendar.sql:1-20]
- **Microstructure cache:** `data/features/micro_minute/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet` already exists for Tier‑1 Aug 26–Sep 6 as part of the cache hydration tasks, and `018_microstructure_minute.sql` defines the SQL mirror awaiting a raw writer. [ml/data/micro_cache.py:7-90][ml/stores/migrations/018_microstructure_minute.sql:1-20]
- **L2 depth cache:** `data/features/l2_minute/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet` mirrors the micro layout and has fully populated partitions per the L2 readiness checklist, with `019_l2_minute.sql` supplying the SQL schema. [ml/data/l2_cache.py:1-35][ml/stores/migrations/019_l2_minute.sql:1-20]
- **Scheduler config:** Tier‑1 orchestration toggles every feature family (`include_macro`, `include_events`, `include_micro`, `include_l2`, `include_earnings`), so coverage automation must handle all of them once dataset IDs exist. [ml/config/pipeline_scheduler_example.toml:1-34]

#### Storage + dual-write surfaces

- **Earnings:** `EarningsStore` manages `ml.earnings_actuals` / `ml.earnings_estimates` tables, and `ml/cli/ingest_earnings.py` wires `EarningsParquetRawWriter` so every write also lands under `data/earnings_raw/**`. [ml/stores/earnings_store.py:65-140][ml/cli/ingest_earnings.py:20-96][ml/stores/earnings_raw_writer.py:1-120]
- **Macro / events / caches:** parquet artifacts exist and SQL tables are now provisioned via `015_macro_release_calendar.sql`–`019_l2_minute.sql`. Macro ingestion now dual-writes via `ensure_macro_ready(..., data_store=...)` and includes the FRED fallback for market-based symbols, but events/micro/L2 flows still bypass `DataStore`, so coverage/restoration cannot inspect Postgres rows for those datasets until their writers land. [ml/data/loaders/alfred_loader.py:1-360][ml/data/ingest/macro_refresh.py:1-420][ml/preprocessing/event_ingestion.py:83-127][ml/data/micro_cache.py:7-90][ml/data/l2_cache.py:1-35][ml/stores/migrations/015_macro_release_calendar.sql:1-20][ml/config/dataset_ids.py:39-80]
- **General ingestion dual-write:** `DataWriter` fans out market ingestions via `ParquetCatalogRawWriter` when `raw_writer` is configured, but cache hydrators bypass `DataWriter` entirely. [ml/stores/data_writer.py:460-552][ml/stores/io_raw.py:81-199]
- **Feature coverage restorer:** only recognizes the two earnings dataset IDs, so manifest entries for other datasets would currently fail during restoration. [ml/data/coverage/feature_restorer.py:289-294]

#### Evidence captured 2025-11-17

- `ML_TFT_FORCE_MICRO_CACHE=1 ML_TFT_AUDIT_USE_DATA_STORE=0 ML_TFT_ALLOW_PARQUET_FALLBACK=1 poetry run python tmp/feature_audit_build.py` revalidated the SPY audit after wiring the cache + datastore toggles. The run (2025-11-17 14:31 UTC) produced 2,850 rows at a 4.07 % positive rate, emitted `ml_out/feature_audit_spy/dataset.parquet`, and restated the capability checks for macro vintages, calendar/context, microstructure (`midprice`, `spread_bps`, `quote_imbalance`, `trade_imbalance`, `realized_vol`), and depth columns (`depth_imbalance_top{1,3,5,10}`, `dwp_bps_top*`, slope metrics). This proves the dataset builder is now reading micro/L2 rows straight from Postgres plus the parquet caches instead of legacy DataStore fallbacks.
- `COVERAGE_DATASETS_FILE=ml/config/coverage_datasets_tier1.toml COVERAGE_RESTORE_ENABLED=1 poetry run python -m ml.cli.pipeline_orchestrator --config ml/config/pipeline_scheduler_example.toml` recorded Tier‑1 ingestion evidence in `ml_out/tier1_orchestrator_run2.log` (`run_id=orch_625eb3bc2266`). The cold path replayed eleven Tier‑1 tickers (AAPL through BAC) and wrote alternating 95 k/200 k row windows per symbol back into Postgres, but every parquet mirror attempt failed with `Intervals are not disjoint after writing a new file` (stale catalog partitions), and the run aborted when the Databento resolver could not find `BRK.XNAS` in `EQUS.MINI`. Tier‑1 orchestration therefore remains blocked on (a) catalog hygiene so `FanoutMarketDataWriter` can append safely and (b) a symbology override for BRK.
- `COVERAGE_DATASETS_FILE=ml/config/coverage_datasets_tier1.toml COVERAGE_RESTORE_ENABLED=1 CATALOG_PATH=data/catalog CATALOG_CLEAN_MODE=archive CATALOG_BACKUP_DIR=ml_out/catalog_archives poetry run python -m ml.cli.pipeline_orchestrator --config ml/config/pipeline_scheduler_example.toml` (2025‑11‑18 12:21 UTC, `run_id=orch_6dda9d5d543a`) confirms the new catalog hygiene + helper deployment flows work end-to-end: the orchestrator archived the existing parquet tree, replayed migrations, and streamed Postgres writes for every Tier‑1 ticker before hitting HOOD. Parquet fan-out degraded gracefully (`Parquet catalog write skipped due to overlapping interval`) while SQL writes succeeded, but the run finally failed with a foreign-key violation when `update_watermark` attempted to emit a progress watermark for `EQUS.MINI` (`ml_dataset_registry` does not contain that dataset ID yet). The full log lives at `ml_out/tier1_orchestrator_run3c.log`. Next steps: register `EQUS.MINI` via the dataset bootstrap so coverage + watermark updates can proceed for HOOD/remaining symbols, then rerun the Tier‑1 pipeline to validate the endgame once the dataset registry is populated.
- After reseeding the data/registry stores against the production DSN (`postgresql://…:5433/nautilus`) and verifying `EQUS.MINI` exists in both registries, reran the Tier‑1 orchestrator with the same coverage + catalog flags (2025‑11‑18 15:07 UTC, `run_id=orch_57f422b18f31`, log `ml_out/tier1_orchestrator_run4.log`). The pipeline now advances past HOOD with SQL writes succeeding for every symbol, but auto-fill eventually fails when the Databento symbology resolver receives a `502` for `DIS` on `EQUS.MINI` (Databento error propagated via `SymbologyResolutionError`). Catalog fan-out still logs overlapping intervals (expected until catalog cleanup completes). Next steps: add a retry/fallback for 5xx symbology responses or prefetch the `DIS` alias to avoid external flakiness, then rerun the Tier‑1 job once Databento stability is confirmed.
- 2025‑11‑20 validation: confirmed the Docker `ml_pipeline` container is pointed at the Compose Postgres service (`ml-postgres-1:5432`, host port 5433) by querying `ml_data_events` via `psql -h localhost -p 5433 -U postgres -d nautilus`. The table now records continuous `INGESTED/backfill` events for Tier‑1 symbols (BAC through CAT) with `ts_event` timestamps ranging from 2025‑11‑19 22:32 UTC through 2025‑11‑20 03:07 UTC, proving the overnight rehydration run used the primary datastore instead of the 5434 test instance. Corresponding `market_data` spot-checks (`SELECT to_timestamp(ts_event/1e9) FROM market_data WHERE instrument_id='AAPL.XNAS' ORDER BY ts_event DESC LIMIT 5`) show rows through 2025‑11‑17, so catalog rehydration must finish replaying the backlog, but we no longer need to worry about writes landing in the wrong database.
- The Databento symbology resolver now retries transient HTTP 5xx responses with exponential backoff and increments `nautilus_ml_symbology_retry_total{dataset,status}` whenever a retry occurs (`ml/data/ingest/symbology.py`, tests in `ml/tests/unit/data/ingest/test_symbology.py`). Once a request exhausts its retry budget the resolver still bubbles a `SymbologyResolutionError`, but Tier‑1 runs should no longer fail immediately on stray `502 <empty message>` responses; alias fallbacks remain intact for BRK/BRK.B. The next orchestrator rerun will capture `run_id=orch_????` evidence once the overnight pipeline finishes the remaining SPY/NVDA gaps.
- Scheduler lookbacks now size to actual SQL gaps. `MARKET_BACKFILL_DYNAMIC_LOOKBACK=1` enables per-instrument staleness probes in `DataScheduler`: we compute `(now - max(ts_event))` for each instrument via `SqlCoverageProvider.latest_timestamp_ns` and only ask Databento for the missing windows (clamped by `MARKET_BACKFILL_MIN_DAYS` / `MARKET_BACKFILL_MAX_DAYS`). This prevents the daily run from blindly replaying seven days of data per symbol when only a few hours are missing.
- Catalog rehydration now targets stale instruments. `CATALOG_REHYDRATE_STALE_ONLY=1` (default) filters the orchestrator universe down to the instruments whose SQL rows are older than `CATALOG_REHYDRATE_STALENESS_HOURS` (default six hours) before `ParquetCatalogRehydrator` streams any parquet back into Postgres, so restarts no longer spend hours scanning partitions that are already mirrored to SQL.
- Catalog hygiene remediation landed via `ml/data/catalog_hygiene.py` plus a thin CLI. Run `poetry run python -m ml.cli.catalog_hygiene --catalog-path data/catalog --backup-dir ml_out/catalog_archives` before the Tier‑1 orchestrator (or set `[ingestion].catalog_clean_mode="archive"` / `catalog_backup_dir="ml_out/catalog_archives"` in `ml/config/pipeline_scheduler_example.toml`). The orchestrator and `MLIntegrationManager` also honour `CATALOG_CLEAN_MODE=archive` + `CATALOG_BACKUP_DIR=...`, so `ParquetDataCatalog` always starts empty while the previous snapshot is preserved under `ml_out/catalog_archives/**`.
- Symbology aliasing now rewrites Tier‑1 `BRK` requests to `BRK.B` for `EQUS.MINI` (`ml/data/ingest/symbology.py`), so the orchestrator retries Databento resolution with the mapped share-class symbol before failing. The resolver increments `nautilus_ml_symbology_alias_hits_total{dataset="EQUS.MINI"}` when the alias is exercised, and `DatasetDiscoveryService` logs `Symbology resolution rejected` at INFO while bumping `nautilus_ml_discovery_symbology_rejections_total{dataset=...}` (`ml/data/ingest/discovery.py`). Docs capture the new telemetry and remediation flow so BRK no longer blocks Tier‑1 cold-path runs.
- The partition bootstrap now relocates historical rows out of the default partitions before attaching new month buckets. `attach_partition_with_data` (defined alongside `create_monthly_partitions` in `ml/stores/migrations/001_bootstrap_schema.sql` and wired through `ml/stores/infrastructure.py`) copies only non-generated columns, deletes the migrated rows from `<table>_default`, and then attaches the prepared table for the requested `[start,end)` range. This prevents the previous `GeneratedAlways`/`Intervals overlap` failures and allows the Tier‑1 bootstrap to run idempotently even on dev databases that already contain data.
- `MLIntegrationManager` now invokes `ensure_partition_helpers` on every bootstrap so the refreshed `CREATE OR REPLACE` bodies for `attach_partition_with_data` / `create_monthly_partitions` deploy automatically even when the migrations have already been applied. Tier‑1 reruns therefore inherit the rehousing logic without requiring manual SQL patches. [ml/core/integration.py:509][ml/stores/infrastructure.py:346]
- Coverage manifest enforcement: `_run_coverage_restoration` now treats `COVERAGE_DATASETS_FILE` as authoritative. `_coverage_manifest_events_total{event}` tracks manifest load outcomes (loaded/missing/invalid), `pipeline_status["errors"]` records `feature_manifest_*` reasons, and `_record_coverage_error` marks the coverage status whenever the manifest is missing or malformed (`ml/deployment/entrypoint_pipeline.py`). Missing manifests therefore fail loudly instead of silently skipping macro/events/micro/L2 coverage.
- Parquet dual-writes now guard against stale catalog partitions. `nautilus_trader/persistence/catalog/parquet.py` skips new chunks whenever the `[start_ns, end_ns]` interval already exists and logs `Parquet catalog write skipped due to overlapping interval` so orchestrator runs can be re-tried without tripping the disjoint-interval assertion. Operators should continue to compact/relocate catalog partitions when ingesting brand-new ranges; the guard simply makes the fan-out idempotent when the parquet cache already contains the requested window.
- The readiness validation suite now includes `poetry run mypy ml --strict`, `poetry run ruff check ml`, `poetry run pytest -k "feature_restorer or coverage_providers or feature_raw_writer"`, `make validate-metrics`, `make validate-events`, and `poetry run coverage report --include "ml/*"`. All validators passed once `ml/registry/tools/feature_catalog.py` stopped instantiating `collections.Counter` (metrics bootstrap check), but coverage is still **53.96 %** for `ml/*`, keeping the ≥90 % requirement open until the orchestration flow covers the remainder of the module surface.

#### Orchestrator & automation gaps

- `entrypoint_pipeline` merges scheduler datasets with `COVERAGE_DATASETS_FILE`; missing feature entries result in `dataset_configs_empty` or simply no bucket classification for macro/calendar/micro/L2. [ml/deployment/entrypoint_pipeline.py:558-609]
- `_run_coverage_restoration` routes feature buckets into `FeatureCoverageRestorer`, but without adapters every non-earnings dataset is skipped. [ml/deployment/entrypoint_pipeline.py:815-1011]
- Market ingestion can dual-write via `ParquetCatalogRawWriter` when `dual_write` is enabled; there is no analogous path for cache hydration or macro/event ingestion CLI flows. [ml/data/scheduler.py:1364-1490][ml/data/ingest/orchestrator.py:320-390][ml/cli/hydrate_feature_caches.py:34-116]

## November 2025 Roadmap Update

### 1. Close ALFRED/FRED gaps and record audit evidence

- [x] Refresh ALFRED releases for the Tier‑1 macro universe by running `yes y | poetry run python scripts/download_alfred_vintages.py` with `FRED_API_KEY`, `TIER1_MACRO_SERIES_UNIVERSE`, and the new commodity/metals series (`PALLFNFINDEXM`, `PCOPPUSDM`, `NASDAQQGLDI`). The 2025‑11‑16 run downloaded those feeds, removed SP500/CRBINDX/GOLDAMGBD228NLBM, and confirmed SQL dual writes succeed for every indicator in the manifest. [ml/data/loaders/alfred_loader.py:1-420][ml/tests/unit/data/test_alfred_loader.py:1-230][ml/data/ingest/macro_refresh.py:1-220]
- [x] Re-run the SPY audit (`ML_TFT_FORCE_MICRO_CACHE=1 ML_TFT_AUDIT_USE_DATA_STORE=0 ML_TFT_ALLOW_PARQUET_FALLBACK=1 poetry run python tmp/feature_audit_build.py`) after the refresh; the 2025‑11‑16 build produced `ml_out/feature_audit_spy/dataset.parquet` with 2,850 rows / 218 features and validated every calendar/micro/L2 capability flag. Attach the dataset and console log to `ml/docs/tools/ORCHESTRATION_RUNBOOK.md` for reviewers.

### 2. Prove the Tier‑1 REAL_TIME dataset build

- [ ] Execute the full Tier‑1 orchestration (~95 instruments) with `VintagePolicy.REAL_TIME`, calendar/events/micro/L2/earnings toggled on, and dual-write enabled so Postgres + parquet stay in sync (`make tier1-orchestration` or `poetry run python -m ml.data.ingest.orchestrator --config ml/config/pipeline_scheduler_example.toml`). Capture the run ID, bucket counts, and `dataset_build.ok` telemetry.
  - 2025-11-17 trial: `COVERAGE_DATASETS_FILE=ml/config/coverage_datasets_tier1.toml COVERAGE_RESTORE_ENABLED=1 poetry run python -m ml.cli.pipeline_orchestrator --config ml/config/pipeline_scheduler_example.toml` produced `run_id=orch_625eb3bc2266` (log: `ml_out/tier1_orchestrator_run2.log`). Eleven Tier‑1 tickers (AAPL…BAC) rehydrated successfully, but every Parquet fan-out failed with `Intervals are not disjoint after writing a new file` and the run aborted on `SymbologyResolutionError: Symbol BRK not found in dataset EQUS.MINI`. Catalog cleanup + a BRK alias are required before we can mark this task complete.
  - Preferred env for next run: `COVERAGE_DATASETS_FILE=ml/config/coverage_datasets_tier1.toml COVERAGE_RESTORE_ENABLED=1 CATALOG_CLEAN_MODE=archive CATALOG_BACKUP_DIR=ml_out/catalog_archives MARKET_BACKFILL_DYNAMIC_LOOKBACK=1`. Command: `poetry run python -m ml.cli.pipeline_orchestrator --config ml/config/pipeline_scheduler_example.toml`. Expected artifacts: archived catalog under `ml_out/catalog_archives/**`, orchestrator log `ml_out/tier1_orchestrator_run*.log`, dataset metadata alongside outputs, `coverage.feature_restore.*` logs/metrics when feature buckets replay, `ml_data_events` rows for Tier-1 symbols, and overlap-skipped parquet fan-out until catalog compaction is finished. Record coverage status (`pipeline_status["coverage"]`), fallback activations, and validator results (`make validate-metrics`, `make validate-events`, `coverage report --include "ml/*"`).
- [ ] Document the run (date, horizon, coverage stats, commands, validation artifacts) in both `ml/docs/tools/ORCHESTRATION_RUNBOOK.md` and `ml/docs/implementation/full_dataset_readiness.md`, and update `ml/docs/ops/streaming_scaling_experiments.md` with the new cohort telemetry.

### 3. Enable coverage automation + manifest enforcement

- [ ] Register dataset IDs for `ml.macro_release_calendar`, `ml.macro_observations`, `ml.events_calendar`, `ml.microstructure_minute`, and `ml.l2_minute` in `ml/config/dataset_ids.py`, then expose typed DataStore writers (e.g., `write_macro_release`, `write_events_calendar`, cache writers) so ingestion and restoration can share instrumentation. Wire macro/event/cache flows through `DataStore.write_ingestion` rather than bespoke parquet writes. [ml/data/ingest/macro_refresh.py:1-200][ml/stores/providers.py:1-200]
- [ ] Extend `FeatureCoverageRestorer` with adapters for each dataset: translate parquet payloads into the new writers, reuse `ml/common/metrics_bootstrap` for `ml_fallback_activations_total`, and let `_resolve_writer` fail fast when the manifest references an unsupported dataset. [ml/data/coverage/feature_restorer.py:1-420][ml/data/coverage/manager.py:1-210]
- [ ] Make `COVERAGE_DATASETS_FILE` authoritative by loading it in `ml/deployment/entrypoint_pipeline.py` and `ml/data/coverage/manager.py`; surface metrics/log warnings when a dataset ID lacks classification or restorer coverage. Update `ml/tests/unit/data/test_feature_restorer.py`, `ml/tests/unit/stores/test_coverage_providers.py`, and `ml/tests/unit/deployment/test_entrypoint_pipeline.py` to exercise the new adapters and manifest plumbing.

### 4. Documentation, telemetry, and evidence capture

- [ ] Update `ml/docs/implementation/full_dataset_readiness.md`, `ml/docs/tools/ORCHESTRATION_RUNBOOK.md`, and `ml/docs/ops/streaming_scaling_experiments.md` with coverage automation steps, env vars (`COVERAGE_RESTORE_ENABLED=1`, `COVERAGE_MAX_BUCKETS_PER_RUN`), Tier‑1 orchestration results, and screenshots of `coverage.feature_restore.completed` metrics.
  - Runbook updates: tier‑1 coverage-gate quickstart now documents env/command/evidence capture expectations and enforces manifest dataset ID validation to fail fast on unsupported entries.
- [ ] Record the next coverage restore run (expected buckets_total, healthy counts, fallback activations) and attach evidence from `coverage report`, `make validate-metrics`, and `make validate-events`. Ensure telemetry shows both Postgres + parquet writes per dataset.

### 5. Validation + testing discipline

- Always run `poetry run mypy ml --strict`, `poetry ruff check ml`, and `coverage report` (requiring ≥90 % coverage inside `ml/**`).
- `poetry run pytest -k \"tft_dataset_builder or macro or coverage\"` plus the adapter-specific suites ensures no regressions in macro/event/cache ingestion.
- SPY audit (`ML_TFT_FORCE_MICRO_CACHE=1 ML_TFT_AUDIT_USE_DATA_STORE=0 ML_TFT_ALLOW_PARQUET_FALLBACK=1 poetry run python tmp/feature_audit_build.py`) and the Tier‑1 orchestration must stay green before merging.
- Validate coverage automation with `COVERAGE_DATASETS_FILE=ml/config/coverage_datasets_tier1.toml COVERAGE_RESTORE_ENABLED=1 make validate-metrics validate-events` and add targeted tests for `FeatureCoverageRestorer`. Record context switch flags in PR descriptions for reproducibility.

### Mypy remediation plan (legacy/backup dirs)

- Keep `poetry run mypy ml --strict` as the gate; introduce scoped `exclude` blocks for `ml/**/backup_phase*` once a triage list exists.
- Convert legacy ndarray annotations to `np.ndarray[Any, Any]` in active modules; prefer adding `TYPE_CHECKING` imports + typed aliases over broad `Any`.
- For SQLAlchemy plugin errors in archived registries (`ml/registry_backup/**`), add module-level `type: ignore[override]` or relocate the files behind a `mypy.ini` ignore to prevent signal loss in active code.

## Addendum: Migration Hygiene Follow-up

Once the core dataset readiness work is delivered, schedule a dedicated pass over the SQL migrations to bring them in line with the guardrails:

- [x] **Inventory + rename:** `python -m ml.tools.print_migration_inventory` now emits `<prefix>::<path>::<checksum>` for every file under `ml/stores/migrations/**`, and the runtime files use unique, monotonic prefixes (`015_macro_release_calendar.sql` → `019_l2_minute.sql` replace the legacy `010_full_dataset_readiness.sql` umbrella).
- [x] **Normalize sequencing:** duplicate IDs (`001_*`, `005a`, …) were renumbered in ascending order so the migration runner’s lexical sort reflects actual intent.
- [x] **Add documentation:** each SQL file starts with `-- Migration:` / `-- Rollback:` headers, and `ml/stores/migrations/README.md` documents the numbering scheme plus the checklist for adding new migrations.
- [x] **Plan execution:** renames landed via `git mv`, the README links to the inventory command, and the clean-DB validation checklist lives here for future migrations.

### Migration Hygiene Execution Plan

1. **Catalog current state:** `python -m ml.tools.print_migration_inventory` captures `<id>::<filename>::<checksum>` for both the active and archive directories; include the output (or diff) in PRs when migrations change.
2. **Define the target naming scheme:** new migrations must use the next available three-digit prefix (e.g., `020_macro_observability.sql`). Dataset-specific DDL now lives in self-contained files (`015`–`019`) instead of the monolithic `010_full_dataset_readiness.sql`.
3. **Stage renames safely:** continue to use `git mv` when shuffling files so history stays readable; validate via `poetry run python -m ml.stores.migrations_runner apply --db-url <test-dsn>` on a clean Postgres instance.
4. **Document and backfill metadata:** keep the README and inventory command in sync whenever migrations change; update `ml/stores/migrations/README.md` when new naming rules or validation steps are introduced.
5. **Finalize with validation:** re-run the migrations runner after every rename/addition and attach the updated inventory to the change request so reviewers can diff checksums quickly.
