# EDGAR + Earnings Feature Validation Checklist

## Goal
Make EDGAR actuals ingestion produce usable data and verify earnings feature generation end-to-end.

## Scope
- EDGAR actuals ingestion (XBRL parsing, dates, fiscal metadata).
- Yahoo estimates ingestion (existing path).
- Earnings feature generation (surprise, growth, momentum, calendar).

## Preconditions
- [x] `SEC_USER_AGENT_*` env vars set in `/home/nate/projects/nautilus_trader/.env.local`
      and `/home/nate/projects/nautilus_trader/ml/deployment/.env`.
- [x] `edgartools` installed in the active poetry env.
- [x] Postgres up and reachable (e.g., `make ml-up-core`).

## Plan (Checklist)

### Phase 1: Parsing Fixes (EDGAR)
- [x] Normalize date parsing in EDGAR fetcher:
      - Accept `date`, `datetime`, and ISO strings for `period_of_report`, `filing_date`.
      - Use `report_date` as a fallback.
- [x] Infer missing fiscal year/quarter from `period_end` when EDGAR metadata is absent.
- [x] Ensure facts extraction from `FactsView` covers key tags and selects the latest period.
- [x] Add logging for parse failures to surface the root cause (no silent drops).

### Phase 2: Unit Tests (EDGAR Parsing)
- [x] Add a unit test for date parsing (date/datetime/string) in EDGAR fetcher.
- [x] Add a unit test that simulates `FactsView` output as a DataFrame and verifies extracted values.
- [x] Update/extend existing `ml/tests/unit/data/earnings/test_edgar_fetcher.py`.

### Phase 3: Integration Tests (Earnings Pipeline)
- [x] Run earnings integration tests:
      - `poetry run pytest ml/tests/integration/earnings/test_earnings_end_to_end.py`
      - `poetry run pytest ml/tests/integration/earnings/test_earnings_store_db.py`
- [x] Run unit tests for EDGAR fetcher:
      - `poetry run pytest ml/tests/unit/data/earnings/test_edgar_fetcher.py`

### Phase 4: Runtime Validation (Local Smoke)
- [x] EDGAR smoke test:
      - `poetry run python -m ml.cli.ingest_earnings --edgar-smoke-test --edgar-smoke-cik 0000320193`
- [x] Run limited ingestion (EDGAR + Yahoo) for one symbol:
      - `poetry run python -m ml.cli.ingest_earnings --symbol AAPL --quarters 4`
- [x] Verify counts in Postgres:
      - `SELECT COUNT(*) FROM ml.earnings_actuals WHERE ticker='AAPL';`
      - `SELECT COUNT(*) FROM ml.earnings_estimates WHERE ticker='AAPL';`

### Phase 5: Feature Generation Verification
- [x] Compute surprise/growth/momentum using EarningsStore and confirm non-zero results
      when sufficient history is available.
- [x] Verify `period_end` join between actuals and estimates is non-empty.

## Acceptance Criteria
- EDGAR actuals ingestion writes >1 rows for a liquid ticker (AAPL or similar).
- At least one ticker has matching `period_end` between actuals and estimates.
- Earnings surprise features are non-empty for those matches.
- Growth/momentum features non-zero once >= 4 quarters of actuals are present.

## Toward Market-Data-Style Mirroring (All Features)

### Dual-Write + Restore Path
- [x] Confirm every feature family dual-writes to Postgres + parquet mirrors:
      - Macro/events/earnings/micro/L2 (raw writers)
      - Computed features (`ml_feature_values` mirror)
- [x] Ensure `COVERAGE_RESTORE_ENABLED=1` in container runtime to restore from parquet.
- [x] Verify `ml/config/coverage_datasets_tier1.toml` includes all feature datasets.

### Backfill Once, Then Incremental
- [x] Run each external ingest once to populate parquet mirrors (macro/earnings/events/micro/L2).
- [x] Add “FeatureStore mirror backfill” utility (SQL → parquet) for `ml_feature_values`.
- [x] Execute “FeatureStore mirror backfill” to seed the mirror.
- [x] Schedule incremental ingests using data registry watermarks (avoid full re-pulls).

### API Limits + Safety
- [x] Use configured rate limits (`edgar_rate_limit`, `yahoo_rate_limit`, FRED/ALFRED staleness).
- [x] Backfill in time slices (e.g., 1–2 years/run) to avoid throttling.
- [x] Keep retries/backoff enabled before falling back.

### Size Estimation (Match Market Data Horizon)
- [x] Measure current Postgres sizes:
      - `SELECT pg_total_relation_size('public.ml_feature_values');`
      - Repeat for feature tables as needed.
- [x] Measure parquet mirror sizes:
      - `du -sh data/features/**`
      - `du -sh data/market/**`
- [x] Estimate growth: `row_count × avg_row_bytes` for each feature family.

## Rollback / Safety
- If EDGAR parsing causes instability, revert to prior release and keep Yahoo estimates only.
- Avoid introducing hot-path changes; keep parsing strictly cold-path.
