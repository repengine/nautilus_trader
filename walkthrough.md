# Walkthrough - Optimizing Rehydration Performance

## Goal
Optimize the `ml_pipeline` rehydration process to address performance bottlenecks and ensure data completeness, while adhering to strict code quality and safety standards.

## Changes

### 1. `SqlMarketDataWriter` Optimization

- Implemented `_write_postgres_copy` using PostgreSQL's `COPY` command for high-performance bulk data loading.
- **Safety Improvements:**
  - Added explicit check for `psycopg2` driver before using `COPY`.
  - Implemented robust DataFrame preparation (provenance, `NaN` handling) matching `_row` logic.
  - Used `psycopg2.sql` to safely quote table names and columns, preventing SQL injection.
  - Ensured `source_dataset` is correctly populated.

### 2. `ParquetCatalogRehydrator` Optimization

- Rewrote `_load_bucket_frame` to directly read Parquet files using `pd.read_parquet`.
- **Abstraction Improvements:**
  - Replaced hardcoded path construction with `self._catalog._make_path` to respect catalog abstraction.
  - Used `fsspec` compatible path handling (forward slashes) instead of `os.path.join` to support S3 and other filesystems.
  - Fixed `TypeError` by using correct argument name `data_cls`.

### 3. `DataScheduler` Reversion

- **Reverted** the dual-write logic in `DataScheduler` as it was unsafe and misplaced. Dual-write should be handled by the `IngestionOrchestrator`.

### 4. Integration Test Fixes

- Mocked `check_ml_dependencies` in `test_pipeline_rehydration.py`.
- Verified rehydration logic with the updated `ParquetCatalogRehydrator` and `SqlMarketDataWriter` (fallback path verified in SQLite tests).

## Verification Results

### Automated Tests
Ran `ml/tests/integration/deployment/test_pipeline_rehydration.py`:

```bash
./.venv/bin/pytest ml/tests/integration/deployment/test_pipeline_rehydration.py -vv
```

**Result:** Passed.

### Performance Impact

- **SQL Write:** `COPY` command is significantly faster than `INSERT` for large datasets (Postgres only).
- **Catalog Read:** Direct `pd.read_parquet` avoids overhead of instantiating thousands of Nautilus objects.

## Next Steps

- Monitor rehydration performance in production.
- Implement dual-write in `IngestionOrchestrator` if required.
