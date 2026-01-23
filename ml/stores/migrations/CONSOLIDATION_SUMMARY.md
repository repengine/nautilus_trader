# Migration Consolidation Summary

**Date:** 2026-01-22
**Status:** ✅ Active
**Impact:** Schema-only (bootstrap consolidation + legacy archive)

## Current Layout

- `ml/stores/migrations_bootstrap/001_bootstrap.sql`
  - Canonical bootstrap for greenfield databases.
- `ml/stores/migrations/`
  - Incremental migrations that build on the bootstrap (including
    `021_market_data_class_tables.sql` for per-data-class market tables and
    `022_update_dataset_type_constraint.sql` for registry dataset-type checks).
- `ml/stores/migrations_legacy/`
  - Archived incremental history for existing databases.

Use `ML_MIGRATIONS_PROFILE=auto|bootstrap|legacy|incremental` to select a
migration profile. The default `auto` profile applies the bootstrap only when
`ml_schema_migrations` is empty, then applies incremental migrations.

## What's Included in Bootstrap

- Core ML stores (feature/model/strategy tables + stats/lineage)
- Market data class tables (bar/quote/trade/mbp1/tbbo) plus a `market_data`
  compatibility view, metadata, statistics, and risk/position tracking
- Data registry tables, event/watermark helpers, and monitoring views
- Feature family tables (macro calendar, macro observations, event calendar,
  microstructure minute, L2 minute)
- Pipeline observability views and helper functions

Test-only seeds and deprecated partition triggers are intentionally excluded.

## How to Apply

### Fresh database

```bash
psql -U postgres -d nautilus_ml < ml/registry/migrations/001_initial_schema.sql
psql -U postgres -d nautilus_ml < ml/registry/migrations/002_add_cold_path_fields.sql
psql -U postgres -d nautilus_ml < ml/registry/migrations/003_add_artifact_digest.sql
psql -U postgres -d nautilus_ml < ml/stores/migrations_bootstrap/001_bootstrap.sql
```

### Migration runner (recommended)

```bash
python -m ml.stores.migrations_runner apply
```

### Legacy replay (existing DBs)

```bash
ML_MIGRATIONS_PROFILE=legacy python -m ml.stores.migrations_runner apply
```

## Notes

- Do not delete or reorder applied migrations.
- Add new schema changes as incremental migrations under `ml/stores/migrations/`.
- Keep test-only seeds in dedicated migrations or fixtures.
