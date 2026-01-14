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
- Macro/events/earnings/micro/L2: parquet mirrors exist and restore via
  `FeatureCoverageRestorer`, but coverage detection from parquet is incomplete.
- FeatureStore computed features: no parquet mirror or restore path; fallback is
  `postgresql -> cached -> dummy` (no file-backed mirror).

## Plan Phases

### Phase 1: FeatureStore Parquet Mirrors (Computed Features)

1. Add a parquet mirror writer for FeatureStore writes.
   - Mirror `ml_feature_values` writes to a partitioned parquet layout.
   - Partition by `instrument_id` and day/month; include `ts_event`/`ts_init`.
2. Wire the mirror into the FeatureStore write path (cold path only).
3. Add config for mirror path (env + config file).
4. Implementation targets:
   - `ml/stores/common/feature_writer.py` (mirror hook on batch writes)
   - `ml/stores/feature_store_facade.py` (config plumbing)
   - `ml/stores/feature_raw_writer.py` (extend or add a dedicated mirror writer)

### Phase 2: FeatureStore Rehydration Path

1. Implement a restorer that reads feature parquet mirrors and writes into
   `ml_feature_values` via FeatureStore/DataStore.
2. Register the dataset in coverage manifests (`ml/config/coverage_datasets_*.toml`).
3. Ensure restoration emits metrics and uses structured logs with `exc_info=True`.
4. Implementation targets:
   - `ml/data/coverage/feature_restorer.py` (extend to cover FeatureStore mirrors)
   - `ml/deployment/entrypoint_pipeline.py` (coverage restoration hook)
   - `ml/config/coverage_datasets_tier1.toml` (add `ml_feature_values` entry)

### Phase 3: Coverage Detection for Parquet Mirrors

1. Implement `PartitionedParquetCoverageProvider.read_bucket_coverage`.
   - Use parquet statistics where possible; avoid full scans.
2. Validate classification across market + feature datasets.
3. Add tests for parquet coverage detection and restore eligibility.
4. Implementation targets:
   - `ml/stores/providers.py` (implement parquet coverage reading)
   - `ml/tests/unit/stores/test_coverage_providers.py` (coverage tests)

### Phase 4: Mirror Hygiene + Validation

1. Add periodic mirror validation (schema checks + row counts).
2. Add restore dry-run mode to confirm API-less rehydration readiness.
3. Implementation targets:
   - `ml/cli/coverage_restore.py` (dry-run mode, if needed)
   - `ml/docs/implementation/full_dataset_readiness.md` (update operational guidance)

## Config and Layout Details

- Mirror base path (env): `ML_FEATURE_PARQUET_MIRROR_DIR`
- Mirror layout (default):
  - `data/features/store/feature_values/{instrument_id}/year=YYYY/month=MM/day=DD/*.parquet`
- Required columns in mirror:
  - `instrument_id`, `ts_event`, `ts_init`, `feature_set_id`, `feature_name`, `value`
  - plus any metadata columns required by FeatureStore schema

## Restore Flow Wiring

- Coverage classification identifies missing buckets:
  - SQL provider compares `ml_feature_values` to parquet mirrors.
- Restoration:
  - Read parquet partitions for the bucket.
  - Write via FeatureStore/DataStore to ensure validation + registry updates.

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

- [ ] Define mirror schema + partition layout and document in this plan.
- [ ] Implement FeatureStore parquet mirror writer (computed features).
- [ ] Wire mirror writer into FeatureStore write path (cold path only).
- [ ] Add `ML_FEATURE_PARQUET_MIRROR_DIR` config and defaults.
- [ ] Extend coverage manifest with `ml_feature_values`.
- [ ] Add FeatureStore restore path using parquet mirrors.
- [ ] Implement parquet coverage provider bucket detection.
- [ ] Add unit tests for mirror writer + coverage provider + restorer.
- [ ] Run targeted tests and validate coverage restoration locally.

## Risks / Open Questions

- Mirror storage growth and compaction strategy.
- Consistency guarantees between SQL writes and parquet mirrors.
- Performance impact of mirroring on feature write throughput.

## Assumptions to Validate (Before Implementation)

- Confirm `ml_feature_values` schema and required columns for parity restores.
- Decide whether mirror writes are best-effort or must be transactional with SQL.
- Define idempotent restore keys (upsert) to prevent duplicate rows.
- Verify `DataRegistry` and validation contracts exist for `ml_feature_values`.
- Ensure parquet coverage provider implementation is required for restores.
- Confirm pandas/polars availability in the pipeline image for restores.

## Deliverables

- Parquet mirror writer for FeatureStore.
- FeatureStore restore tool integrated with coverage restoration.
- Completed parquet coverage provider implementation.
- Coverage manifest updated for all feature datasets.
