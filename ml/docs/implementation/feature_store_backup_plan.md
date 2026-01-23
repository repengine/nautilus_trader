# Feature Store Backup and API-Less Rehydration Plan

## Goal
Ensure every dataset in the universe can be rehydrated without external API calls,
using durable parquet mirrors and SQL restoration paths. API calls should be used
only to backfill new or missing history.

## Scope

- FeatureStore computed features (`ml_feature_values`)
- External feature families (macro, events, earnings, micro/L2)
- Coverage classification and restore loops

## Current State (Code-Based)

- Market data: Parquet catalog mirror + rehydration already exists.
- Macro/events/earnings/micro/L2: parquet mirrors exist, are registered in coverage
  manifests, and restore via `FeatureCoverageRestorer`; coverage detection uses
  `PartitionedParquetCoverageProvider` (scan-based today).
- FeatureStore computed features: mirror writer + backfill utility exist and are
  wired through FeatureStore; historical compute now mirrors to parquet and falls
  back to `public.market_data` when `public.bar` is absent. `ml.feature_values`
  is in coverage manifests and supported by `FeatureCoverageRestorer`. Remaining
  gaps are validation/hygiene and coverage-detection performance tuning.
- Coverage restore can now skip legacy `market_data` via the schema-audit allowlist
  so feature rehydration can run without migrating that table.

## Plan Phases

### Phase 1: FeatureStore Parquet Mirrors (Computed Features)

1. [x] Provide a parquet mirror writer for FeatureStore writes.
   - Mirror `ml_feature_values` writes to a partitioned parquet layout.
   - Partition by `instrument_id` and day/month; include `ts_event`/`ts_init`.
2. [x] Wire the mirror into the FeatureStore write path (cold path only).
3. [x] Add config for mirror path (env + config file).
4. [x] Add SQL -> parquet backfill utility for initial mirror seeding.
5. Implementation targets (in place):
   - `ml/stores/common/feature_writer.py` (mirror hook on batch writes)
   - `ml/stores/feature_store_facade.py` (config plumbing)
   - `ml/stores/feature_raw_writer.py` (FeatureValuesParquetMirrorWriter)
   - `ml/cli/feature_store_mirror_backfill.py` (backfill entry point)

### Phase 2: FeatureStore Rehydration Path

1. [x] Restore parquet mirrors into `ml_feature_values` via FeatureStore/DataStore.
2. [x] Register the dataset in coverage manifests (`ml/config/coverage_datasets_*.toml`).
3. [x] Ensure restoration emits metrics and uses structured logs with `exc_info=True`.
4. Implementation targets (in place):
   - `ml/data/coverage/feature_restorer.py` (FeatureStore mirrors supported)
   - `ml/deployment/entrypoint_pipeline.py` (coverage restoration hook)
   - `ml/config/coverage_datasets_tier1.toml` (includes `ml_feature_values`)

### Phase 3: Coverage Detection for Parquet Mirrors

1. [x] Implement `PartitionedParquetCoverageProvider.read_bucket_coverage`.
   - Current implementation scans parquet files with pandas.
   - File-backed datasets now prefer pyarrow dataset scans when available.
   - TODO: use parquet statistics or precomputed metadata to avoid full scans.
2. [ ] Validate classification across market + feature datasets.
3. [x] Add tests for parquet coverage detection.
4. Implementation targets:
   - `ml/stores/providers.py` (parquet coverage reading)
   - `ml/tests/unit/stores/test_coverage_providers.py` (coverage tests)

### Phase 4: Mirror Hygiene + Validation

1. [ ] Add periodic mirror validation (schema checks + row counts).
2. [x] Add restore dry-run mode to confirm API-less rehydration readiness.
3. Implementation targets:
   - `ml/cli/coverage_restore.py` (dry-run mode, if needed)
   - `ml/docs/implementation/full_dataset_readiness.md` (update operational guidance)

## Config and Layout Details

- Mirror base path (env): `ML_FEATURE_PARQUET_MIRROR_DIR`
- Mirror enable flag (env): `ML_FEATURE_PARQUET_MIRROR_ENABLE`
- Optional schema-audit allowlist (env): `COVERAGE_SCHEMA_AUDIT_TABLES`
  (comma-separated table names; use to skip legacy `market_data` during feature-only restores).
- Mirror layout (default):
  - `data/features/store/feature_values/{instrument_id}/year=YYYY/month=MM/day=DD.parquet`
- Required columns in mirror:
  - `feature_set_id`, `instrument_id`, `ts_event`, `ts_init`, `values`
  - plus any metadata columns required by FeatureStore schema

## Restore Flow Wiring

- Coverage classification identifies missing buckets:
  - SQL provider reads `ml_feature_values` and parquet provider reads mirror paths.
- Restoration:
  - Read parquet partitions for the bucket.
  - Write via FeatureStore/DataStore to ensure validation + registry updates.

## DRY Reuse Notes (Market Data Parity)

- Reuse the same coverage classification pipeline (`ml/data/coverage/manager.py`)
  and provider plumbing (`ml/stores/providers.py`) that market data uses.
- Reuse `day_partition_path` and `ParquetCoverageSpec` to keep parquet layout and
  bucket math consistent across market and feature datasets.
- Feature restoration already mirrors the market-data flow: `ParquetCatalogRehydrator`
  handles catalog buckets, and `FeatureCoverageRestorer` handles feature mirrors.

## Acceptance Criteria

- Cold start from empty Postgres restores:
  - Market data from Parquet catalog
  - Macro/events/earnings/micro/L2 from parquet mirrors
  - FeatureStore computed features from parquet mirrors
- `COVERAGE_RESTORE_ENABLED=1` completes without API calls for historical ranges.
- Coverage metrics report zero `restore_from_catalog` buckets after run.

## Test Plan

- Unit:
  - Mirror writer writes correct partition paths.
  - Coverage provider returns expected buckets from parquet stats.
  - Feature restorer replays feature parquet into SQL.
- Integration:
  - Spin Postgres down, restore from mirrors, build dataset without API calls.

## Execution Checklist

- [x] Define mirror schema + partition layout and document in this plan.
- [x] Implement FeatureValues parquet mirror writer + backfill utility.
- [x] Wire mirror writer into FeatureStore write path (cold path only).
- [x] Add `ML_FEATURE_PARQUET_MIRROR_ENABLE`/`ML_FEATURE_PARQUET_MIRROR_DIR` config and defaults.
- [x] Extend coverage manifest with `ml_feature_values`.
- [x] FeatureCoverageRestorer supports `ml_feature_values`.
- [x] Implement parquet coverage provider bucket detection (scan-based).
- [x] Add pyarrow dataset scan for file-backed parquet coverage.
- [ ] Optimize parquet coverage detection (parquet stats or metadata).
- [x] Add write-path mirror tests for feature values.
- [ ] Add coverage classification tests for feature values.
- [ ] Run targeted tests and validate coverage restoration locally.
- [ ] Mirror hygiene: periodic validation + dry-run restore mode (Phase 4).

## Immediate Next Tasks (Implementation Order)

### A) Data quality + point-in-time integrity
- [x] Fix `earnings_estimates.ts_event` semantics when Yahoo returns future timestamps
  (clamp to ingestion time to avoid forward-dated `ts_event`).
- [x] Treat economic events as global (store `__GLOBAL__`), while retaining
  per-instrument IDs for corporate events.
- [x] Remove legacy `events_calendar` rows with missing `instrument_id`.
- [x] Align micro/L2 cache partitions with dataset joins (write to full instrument IDs,
  read base fallbacks, and promote base partitions to full when encountered).

### B) Restore loop validation
- [x] Decide defaults for `COVERAGE_RESTORE_LOOKBACK_DAYS` (365) and
  `COVERAGE_MAX_BUCKETS_PER_RUN` (2000) for feature restores.
- [ ] Re-run coverage restore and verify SQL counts for macro/earnings/events/micro/L2
  and `ml_feature_values`.
- [x] Run a one-time micro/L2 cache promotion pass so base-only partitions are copied
  into full instrument ID directories before coverage restore.
- [x] Reingest micro/L2 cache partitions into SQL with full instrument IDs.
- [ ] Validate feature reingest/restore in Docker with `FEATURE_REINGEST_ENABLED=1`
  and confirm parquet mirrors land under `data/features/**`.

### C) Pipeline runtime prerequisites
- [x] Add unscoped coverage/restore mode for events using a global entity bucket.
- [x] Wire a `FeatureEngineer` into `ml_pipeline` so scheduled feature computation can run.
- [x] Allow feature computation to use `market_data` when the `bar` table is absent.
- [x] Add feature reingest handlers for `REINGEST_FROM_SOURCE` buckets
  (macro/events/earnings/micro/L2) to complete the auto-fill loop without parquet.
- [x] Install `fredapi` in the pipeline image so FRED/ALFRED reingest runs in Docker.
- [x] Provide `FRED_API_KEY` and SEC identity environment values so macro/earnings
  reingest can run without warnings (via `--env-file .env.local`).
- [x] Schedule or trigger `FeatureEngineer` runs so `ml_feature_values` populates
  for real instruments.

## Risks / Open Questions

- Mirror storage growth and compaction strategy.
- Consistency guarantees between SQL writes and parquet mirrors.
- Performance impact of mirroring on feature write throughput.
- Scan-based parquet coverage can be slow for large mirrors; decide on stats-based or
  metadata-driven coverage tracking.
- Events parquet now uses `__GLOBAL__` for economic events; corporate events still
  require per-instrument IDs to be useful in joins.
- EDGAR fetches require a SEC User-Agent identity via `SEC_IDENTITY` or
  `SEC_USER_AGENT_*` environment variables.
- Computed feature mirrors may only include test/sim instruments until a live feature
  computation run is wired into the pipeline.
- Earnings estimates now clamp future `ts_event` values at ingest time; historical
  rows may still require cleanup.
- Micro/L2 cache paths now prefer full instrument IDs; raw data that is ticker-only may
  still produce duplicate cache content across venues unless deduplicated upstream.

## Assumptions to Validate (Before Implementation)

- Confirm `ml_feature_values` schema and required columns for parity restores.
- Decide whether mirror writes are best-effort or must be transactional with SQL.
- Define idempotent restore keys (upsert) to prevent duplicate rows.
- Verify `DataRegistry` and validation contracts exist for `ml_feature_values`.
- Confirm pandas/polars availability in the pipeline image for mirror writes/restores.

## Deliverables

- Parquet mirror writer for FeatureStore.
- FeatureStore restore tool integrated with coverage restoration.
- Completed parquet coverage provider implementation.
- Coverage manifest updated for all feature datasets.
