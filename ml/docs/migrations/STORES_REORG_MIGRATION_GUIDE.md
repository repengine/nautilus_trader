# Stores Reorganization Migration Guide

This guide helps external consumers update imports after the `ml.stores` reorganization.

All internal references have been migrated. If you consume `ml.stores` directly, update imports per the mapping below. No runtime shims remain in this repo.

## TL;DR Import Mapping

- `ml.stores.coverage_sql` → `ml.stores.providers`
  - `SqlCoverageProvider`, `SqlMarketDataWriter`
- `ml.stores.coverage_catalog` → `ml.stores.providers`
  - `CatalogCoverageProvider`
- `ml.stores.raw_io` → `ml.stores.io_raw`
  - `RawReaderProtocol`, `RawIngestionWriterProtocol`
- `ml.stores.raw_io_parquet` → `ml.stores.io_raw`
  - `ParquetCatalogRawReader`, `ParquetCatalogRawWriter`
- `ml.stores.market_data_writer` → `ml.stores.writers`
  - `DataStoreMarketDataWriter`, `ParquetCatalogMarketDataWriter`
- `ml.stores.live_data_recorder` → `ml.stores.writers`
  - `LiveDataRecorder`, `LiveDataInterceptor`
- `ml.stores.partition_manager` → `ml.stores.infrastructure`
  - `PartitionManager`, `run_partition_maintenance`
- `ml.stores.db_preflight` → `ml.stores.infrastructure`
  - `check_db_prereqs`
- Private mixins and helpers:
  - `ml.stores._buffered_store`, `_engine_mixin`, `_health_mixin`, `_init_mixin`, `_read_helpers`, `_registry_mixin`, `_upsert_mixin`, `_batch_utils` → `ml.stores.mixins`

## Quick Search & Replace Hints

Use ripgrep to audit and sed (or an IDE) to update:

```bash
rg -n "ml\.stores\.(coverage_sql|coverage_catalog|raw_io_parquet|raw_io|market_data_writer|live_data_recorder|partition_manager|db_preflight|_.*mixin|_batch_utils)" -S
```

## Suggested Compatibility Options (if you can’t update immediately)

If you maintain an app depending on the old paths and cannot bump imports quickly, you can:

- Add a small compatibility module in your app startup:

```python
import sys
from importlib import import_module

_ALIASES = {
    "ml.stores.coverage_sql": "ml.stores.providers",
    "ml.stores.coverage_catalog": "ml.stores.providers",
    "ml.stores.raw_io": "ml.stores.io_raw",
    "ml.stores.raw_io_parquet": "ml.stores.io_raw",
    "ml.stores.market_data_writer": "ml.stores.writers",
    "ml.stores.live_data_recorder": "ml.stores.writers",
    "ml.stores.partition_manager": "ml.stores.infrastructure",
    "ml.stores.db_preflight": "ml.stores.infrastructure",
}

for old, new in _ALIASES.items():
    try:
        sys.modules[old] = import_module(new)
    except Exception:
        pass
```

- Or pin to a pre-reorg version until you can migrate.

## Rationale

- Reduce file sprawl, keep hot-path APIs visible, and avoid import cycles.
- Consolidated modules: `mixins.py`, `io_raw.py`, `writers.py`, `providers.py`, `infrastructure.py`.

## Questions

If you need help migrating a specific codebase, share a short snippet of your imports and we can provide a patch.
