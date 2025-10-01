# Migration Consolidation Summary

**Date:** 2025-10-01
**Status:** ✅ Complete
**Impact:** Schema-only (no production data existed)

## What Changed

**Before:** 18 fragmented migration files (2,237 LOC)
**After:** 1 bootstrap schema file (450 LOC) + 3 registry migrations

### Files Consolidated

All store migrations have been merged into `001_bootstrap_schema.sql`:

**Archived files** (moved to `archive/`):
- `001_stores_schema.sql` → Merged
- `002_auto_partitioning.sql` → Merged (triggers removed, pre-created partitions used)
- `003_market_data.sql` → Merged
- `004_data_registry.sql` → Merged
- `005_schema_hardening.sql` → Merged
- `005_views.sql` → Merged
- `005a_feature_values_dedupe.sql` → Removed (was diagnostic, not migration)
- `006_disable_partition_triggers.sql` → Obsolete (no triggers in bootstrap)
- `007_add_event_metadata.sql` → Merged
- `007_brin_indexes.sql` → Merged
- `008_predictions_alias.sql` → Merged
- `008_test_seed_features_dataset.sql` → Removed (test-only)
- `009_update_parents_predictions.sql` → Merged
- `010_backfill_market_data_provenance.sql` → Removed (no data to backfill)
- `999_fix_partitions_immediate.sql` → Obsolete (no triggers to fix)

**Registry migrations** (kept separate, already clean):
- `ml/registry/migrations/001_initial_schema.sql`
- `ml/registry/migrations/002_add_cold_path_fields.sql`
- `ml/registry/migrations/003_add_artifact_digest.sql`

## Code Changes

### Migration Runners Updated

1. **ml/tasks/db.py** - `_BASE_MIGRATIONS` reduced from 9 to 4 files
2. **ml/core/integration.py** - Fallback migration list updated
3. **ml/tests/fixtures/database_fixtures.py** - Test schema files updated

### Documentation References

**⚠️ Many docs reference old migration filenames** - These are now historical references. The canonical schema is in `001_bootstrap_schema.sql`.

Key docs mentioning old migrations:
- `ml/docs/context/context_stores.md` - Lists all 18 migrations
- `ml/docs/context/context_migrations.md` - Detailed migration guide
- `ml/docs/context/context_deployment.md` - Deployment instructions
- `ml/docs/context/context_data.md` - Data layer references
- `.github/workflows/nightly.yml` - CI pipeline
- Various other context docs

**Action items for doc updates:**
- Replace migration lists with: "Apply `001_bootstrap_schema.sql`"
- Update table references from "`003_market_data.sql`" to "`001_bootstrap_schema.sql`"
- Note that partition triggers were removed (race conditions)

## How to Apply Bootstrap Schema

### Fresh Database

```bash
# Apply registry migrations
psql -U postgres -d nautilus_ml < ml/registry/migrations/001_initial_schema.sql
psql -U postgres -d nautilus_ml < ml/registry/migrations/002_add_cold_path_fields.sql
psql -U postgres -d nautilus_ml < ml/registry/migrations/003_add_artifact_digest.sql

# Apply consolidated stores schema
psql -U postgres -d nautilus_ml < ml/stores/migrations/001_bootstrap_schema.sql
```

### Via Migration Runner

```bash
# Uses ml/tasks/db.py migration list
python -m ml.cli.apply_migrations --db postgresql://...
```

### Via Integration Manager

```python
from ml.core.integration import MLIntegrationManager

manager = MLIntegrationManager(auto_migrate=True)
manager.setup()  # Automatically applies migrations
```

## What's in Bootstrap Schema

### Tables Created

**Feature Store:**
- `ml_feature_values` (partitioned)
- `ml_feature_computation_stats`
- `ml_feature_lineage`

**Model Store:**
- `ml_model_predictions` (partitioned)

**Strategy Store:**
- `ml_strategy_signals` (partitioned)

**Market Data:**
- `market_data` (partitioned, with `source_dataset` for provenance)
- `market_data_metadata`
- `market_data_statistics`
- `ml_positions`
- `ml_risk_limits`

**Data Registry:**
- `ml_dataset_registry`
- `ml_data_events`
- `ml_data_watermarks`
- `ml_data_lineage`

### Indexes

- BRIN indexes on all partitioned tables (`ts_event`)
- Standard B-tree indexes for lookups
- Partial indexes for filtered queries

### Helper Functions

- `create_monthly_partitions()` - Manual partition creation
- `emit_data_event()` - Event emission with watermark updates
- `emit_data_event_ext()` - Extended version with metadata
- `update_watermark()` - Watermark management
- `check_risk_limits()` - Risk validation

### Partitions

Pre-created monthly partitions for 2023-2027 (60 months) for:
- `ml_feature_values`
- `ml_model_predictions`
- `ml_strategy_signals`
- `market_data`

**Note:** No automatic partition triggers (avoided race conditions from old migrations)

## Issues Resolved

1. **Duplicate version numbers** - 005 (×3), 007 (×2), 008 (×2)
2. **Conflicting logic** - 002 created triggers, 006 deleted them
3. **Emergency fixes** - 999 duplicated 006's work
4. **Diagnostic scripts** - 005a was queries, not a migration
5. **Dead code** - 010 backfilled data that didn't exist
6. **Labyrinthine history** - 18 files for greenfield deployment

## Testing

### Unit Tests

Tests updated to use consolidated schema:
- `ml/tests/fixtures/database_fixtures.py` - Now loads `001_bootstrap_schema.sql` only
- All integration tests pass with new schema

### Validation

```bash
# 1. Fresh database smoke test
dropdb nautilus_test && createdb nautilus_test
psql -d nautilus_test < ml/registry/migrations/001_initial_schema.sql
psql -d nautilus_test < ml/stores/migrations/001_bootstrap_schema.sql

# 2. Verify tables exist
psql -d nautilus_test -c "\dt ml_*"
psql -d nautilus_test -c "\dt market_*"

# 3. Verify partitions
psql -d nautilus_test -c "SELECT tablename FROM pg_tables WHERE tablename LIKE '%_2023_%' ORDER BY tablename LIMIT 10;"

# 4. Run test suite
pytest ml/tests/unit/ -k migration
```

## Rollback (if needed)

Bootstrap consolidation is **irreversible** for existing databases. However, since no production data existed, this is safe.

If you need to reference old migration logic:
```bash
# View archived migration
cat ml/stores/migrations/archive/003_market_data.sql

# Extract specific table
sed -n '/CREATE TABLE market_data/,/;/p' ml/stores/migrations/archive/003_market_data.sql
```

## Future Migrations

When adding new schema changes:

1. Create `ml/stores/migrations/002_descriptive_name.sql`
2. Add to `_BASE_MIGRATIONS` in `ml/tasks/db.py`
3. Ensure idempotency (`CREATE TABLE IF NOT EXISTS`, etc.)
4. Test on fresh database before committing

**Do NOT:**
- Modify `001_bootstrap_schema.sql` (it's the baseline)
- Create multiple files with the same version number
- Include diagnostic queries in migrations
- Add "emergency fix" files (fix the root cause)

## Documentation Updates Needed

Search and replace in docs:
- `001_stores_schema.sql` → `001_bootstrap_schema.sql`
- `003_market_data.sql` → `001_bootstrap_schema.sql` (market_data table)
- `004_data_registry.sql` → `001_bootstrap_schema.sql` (registry tables)
- Lists of 001-010 migrations → "Apply `001_bootstrap_schema.sql`"

Note: This is purely documentation cleanup - code already uses new schema.

## Success Criteria

- ✅ Bootstrap schema creates all required tables
- ✅ Partitions pre-created for 2023-2027
- ✅ Helper functions work correctly
- ✅ Test suite passes
- ✅ No partition trigger race conditions
- ✅ Migration runners updated
- ✅ Archived migrations preserved for reference
- ✅ Reduced from 2,237 to ~450 LOC
- ✅ Eliminated 7 duplicate version numbers
- ✅ Removed 2 emergency hotfixes
- ✅ No breaking changes for greenfield deployment

## Contact

For questions about the consolidation, see:
- Archive README: `ml/stores/migrations/archive/README.md`
- Original migrations: `ml/stores/migrations/archive/*.sql`
- This summary: `ml/stores/migrations/CONSOLIDATION_SUMMARY.md`
