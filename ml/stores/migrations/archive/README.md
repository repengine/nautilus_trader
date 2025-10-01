# Archived Migrations

**Archived:** 2025-10-01
**Reason:** Consolidated into `001_bootstrap_schema.sql` during greenfield cleanup

## Context

These 18 migration files evolved incrementally during development but never ran against production data. They have been consolidated into a single bootstrap schema for clarity and maintainability.

## Original Evolution

### Core Schema (001-004)
- `001_stores_schema.sql` - Initial ML tables (features, predictions, signals)
- `002_auto_partitioning.sql` - Automatic partition triggers (later disabled)
- `003_market_data.sql` - Market data tables with OHLCV support
- `004_data_registry.sql` - Data registry and event tracking

### Schema Hardening (005 series)
- `005_schema_hardening.sql` - Unique constraints and type standardization
- `005_views.sql` - Analytical views for monitoring
- `005a_feature_values_dedupe.sql` - Diagnostic script (not a true migration)

### Fixes & Optimizations (006-009)
- `006_disable_partition_triggers.sql` - Disabled triggers from 002 due to race conditions
- `007_add_event_metadata.sql` - Added JSONB metadata column to events
- `007_brin_indexes.sql` - BRIN indexes for time-range queries
- `008_predictions_alias.sql` - Prediction table aliasing
- `008_test_seed_features_dataset.sql` - Test data seeding
- `009_update_parents_predictions.sql` - Lineage updates

### Canonicalization (010)
- `010_backfill_market_data_provenance.sql` - Backfill for data that didn't exist

### Emergency Fixes
- `999_fix_partitions_immediate.sql` - Duplicate partition fix (similar to 006)

## Issues Resolved

1. **Duplicate version numbers**: 005 (×3), 007 (×2), 008 (×2)
2. **Conflicting logic**: 002 creates triggers, 006 deletes them
3. **Diagnostic scripts**: 005a was a query script, not a migration
4. **Emergency patches**: 999 duplicated work from 006
5. **Dead code**: 010 backfilled data that never existed

## Consolidation Approach

The new `001_bootstrap_schema.sql` includes:
- All table definitions from 001, 003, 004
- Indexes from 007_brin_indexes
- Helper functions from 002, 004, 007
- Views from 005_views
- **Pre-created partitions** (2023-2027) - no triggers needed
- **Removed**: partition triggers, canonicalization columns, emergency fixes

## Restoration

If you need to reference the original migrations:
```bash
# View original file
cat ml/stores/migrations/archive/003_market_data.sql

# Extract specific table definition
sed -n '/CREATE TABLE market_data/,/;/p' archive/003_market_data.sql
```

## Migration to Bootstrap

To apply the consolidated schema to a fresh database:
```bash
psql -U postgres -d nautilus_ml < ml/registry/migrations/001_initial_schema.sql
psql -U postgres -d nautilus_ml < ml/registry/migrations/002_add_cold_path_fields.sql
psql -U postgres -d nautilus_ml < ml/registry/migrations/003_add_artifact_digest.sql
psql -U postgres -d nautilus_ml < ml/stores/migrations/001_bootstrap_schema.sql
```

No data migration is required - this was a pre-deployment consolidation.
