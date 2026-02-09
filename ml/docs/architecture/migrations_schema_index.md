# ML Stores Schema Index

*Repository: `/home/nate/projects/nautilus_trader`*

*Last updated: 2025‑11‑13*

This document inventories every SQL migration under `ml/stores/migrations/` (including the archived history) so we can reason about ownership, consolidate functionality, and plan fixture/migration cleanup without spelunking through dated filenames.

## How to use this document

- **Reference**: when touching a store/registry table, consult this index to find the canonical definition, the original migration, and the owning domain.
- **Consolidation planning**: guides the rewrite of `001_bootstrap_schema.sql` (or future modular SQL files) so each domain’s requirements remain intact.
- **Fixture hygiene**: highlights which tables exist purely because the bootstrap pre-creates partitions, informing truncation fixtures and test scoping.

## Summary

- Current runtime migrations: `ml/stores/migrations_runner.py` and `MLIntegrationManager` only apply `ml/stores/migrations/001_bootstrap_schema.sql` plus the three registry migrations. All other SQL files are legacy artifacts.
- The bootstrap file contains objects for five domains: **Feature Store**, **Model Store**, **Strategy Store**, **Market Data**, **Data Registry**. Helper functions (partition creation, watermark emitters) are shared across domains.
- Partition tables are pre-created for 60 months (2023–2027) for `ml_feature_values`, `ml_model_predictions`, `ml_strategy_signals`, and `market_data`, which directly impacts fixture cleanup times.

## Schema Index (by original migration)

| Migration | Status | Objects defined | Notes / Owners |
| --- | --- | --- | --- |
| `001_bootstrap_schema.sql` | Active | Consolidated schema: all store tables, registry tables, helper functions, partition pre-creation. | Current runtime migration plan (applied by tasks + integration manager). |
| `002_stores_schema.sql` | Archived → merged | Tables: `ml_feature_values`, `ml_model_predictions`, `ml_strategy_signals`; indexes; constraints; initial partitions. | Core ML stores (Feature/Model/Strategy). Owners: Data/Feature/Model/Strategy store teams. |
| `003_auto_partitioning.sql` | Archived | Functions `create_monthly_partitions`, triggers for auto-creating partitions on insert. | Triggers were later disabled (race conditions). Bootstrap now pre-creates partitions. |
| `004_market_data.sql` | Archived → merged | Tables: `market_data`, `market_data_metadata`, `market_data_statistics`, `ml_positions`, `ml_risk_limits`; indexes. | Market data ingestion / provenance. Owners: Data ingestion team. |
| `005_data_registry.sql` | Archived → merged | Tables: `ml_dataset_registry`, `ml_data_events`, `ml_data_watermarks`, `ml_data_lineage`; helper functions: `emit_data_event`, `update_watermark`. | Data Registry / telemetry. Owners: Registry/Observability team. |
| `007_schema_hardening.sql` | Optional SQL still present | Adds `NOT NULL` constraints, `CHECK`s, default values to store tables. | Hardening pass. Needs review to merge into bootstrap or kept separate. |
| `008_views.sql` | Optional SQL still present | Materialized/standard views for reporting. | Consumers/Analytics. |
| `006_feature_values_dedupe.sql` | Archived (removed) | Diagnostic query only. | No action. |
| `009_disable_partition_triggers.sql` | Optional SQL still present | Drops the triggers from `003_auto_partitioning`. | Should be obsolete once bootstrap owns partition creation. |
| `010_add_event_metadata.sql` | Archived → merged | Extends event tables with metadata columns, adds indexes. | Registry team. |
| `011_brin_indexes.sql` | Optional SQL still present | Adds BRIN indexes to partitioned tables. | Performance/hot-path owners. |
| `012_predictions_alias.sql` | Optional SQL still present | Creates backward-compatible views/aliases for predictions tables. | Strategy/model store consumers. |
| `013_test_seed_features_dataset.sql` | Archived (removed) | Seed data for tests only. | No action. |
| `014_update_parents_predictions.sql` | Optional SQL still present | Updates parent table metadata for predictions. | Strategy/model store owners. |
| `015_macro_release_calendar.sql` | Active | SQL store for ALFRED release calendars; replaces portion of the legacy full-readiness migration. | Macro ingestion team. |
| `016_macro_observations.sql` | Active | Long-format macro observation store. | Macro ingestion team. |
| `017_events_calendar.sql` | Active | Structured events calendar table for context features. | Events/Calendar ingestion. |
| `018_microstructure_minute.sql` | Active | Microstructure (L1) minute aggregate store. | Feature ingestion (micro). |
| `019_l2_minute.sql` | Active | L2 depth minute aggregate store. | Feature ingestion (depth). |

> *`015`–`019` replace the removed `010_full_dataset_readiness.sql`, ensuring each dataset family has a single-role migration.

## Domain View

### Feature Store

- Tables: `ml_feature_values`, `ml_feature_computation_stats`, `ml_feature_lineage` (all now defined in bootstrap).
- Indexes: B-tree on `(feature_set_id, instrument_id)`, BRIN on `ts_event` (from `011_brin_indexes.sql`).
- Partition strategy: monthly partitions pre-created (bootstrap function call). Sources: 001 + 007.

### Model Store

- Tables: `ml_model_predictions`, supporting indexes (B-tree + BRIN) and alias views (`012_predictions_alias.sql`).
- Functions: none; relies on shared partition helper. Sources: 001 + 007 + 008 + 009.

### Strategy Store

- Tables: `ml_strategy_signals`, performance metrics tables. Indexes + BRIN. Sources: 001 + 007 + optional alias/perf SQL.
- Aggregation helpers (if any) are in application code; migrations only set schema/indices.

### Market Data

- Tables: `market_data`, `market_data_metadata`, `market_data_statistics`, `ml_positions`, `ml_risk_limits`. Sources: 003 + hardening.
- Partition plan: same helper function, pre-created 60 months.

### Data Registry

- Tables: `ml_dataset_registry`, `ml_data_events`, `ml_data_watermarks`, `ml_data_lineage`. Functions: `emit_data_event`, `emit_data_event_ext`, `update_watermark`. Sources: 004 + 007.

## Database Snapshot (2025‑11‑13)

Actual state of the shared PostgreSQL instance (`postgresql://postgres:postgres@localhost:5434/nautilus_test`):

- Table counts (names starting with `ml_`):
  - `public` schema: **233** tables
  - `ml_registry` schema: **134** tables
- Partition counts (from `pg_inherits`):
  - `ml_feature_values`, `ml_model_predictions`, `ml_strategy_signals`: **61 partitions each** (plus matching entries for their primary keys/BRIN indexes)
  - `market_data`: **121 partitions** (suggesting 10+ years of pre-created monthly partitions)
  - `ml_data_events`: **37 partitions**
  - Identical partition/index objects exist in both `public` and `ml_registry` schemas, effectively doubling the number of relations.
- Sizes: parent tables currently report `0 bytes` (partitions hold data), but the sheer partition count is what drives teardown latency.
- Implication: `clean_postgres_db_module` has to truncate **300+** relations every time it runs, each requiring `ACCESS EXCLUSIVE` locks; with `statement_timeout` fixed at 2000 ms this regularly times out, as seen in strategy store integration tests.
- Fixtures like `patch_engine_manager`, `real_engine_manager`, and store bundles already call `EngineManager.dispose_all()` at setup/teardown, so the lock contention stems from the volume of partitions, not leaked connections. We still plan to scope truncations per domain and raise/parameterize the timeout.

This snapshot confirms that the bootstrap migration’s “pre-create 60+ months of partitions per table (and in multiple schemas)” choice is the direct cause of the current truncation overhead.

## Open Questions / To-dos

- Optional SQL (005–009) is still present but never run by our migration runners. Decide whether to:
  1. Merge their contents into the bootstrap file (with clear section headers),
  2. Keep them as separate files but reintroduce a structured migration list, or
  3. Remove them officially after confirming all functionality lives in bootstrap.
- CI / docs / tests still reference the old numbered files. Need to update:
  - `.github/workflows/nightly.yml` (runs `005_data_registry.sql`).
  - `ml/tests/unit/config/test_config.py` (expects every legacy filename).
  - `ml/docs/context/*.md`, `ROADMAP.md`, monitoring docs, etc.
- Fixture performance: consider reducing pre-created partitions (e.g., 12 months for tests) or adding schema namespaces to limit truncation scope.

## Next Steps (proposed)

1. **Finalize Schema Index** – expand the table above with per-column notes if needed.
2. **Bootstrap Refactor** – reorganize `001_bootstrap_schema.sql` into labeled sections (Feature/Model/Strategy/Market/Registry) or modular includes.
3. **Doc & CI Updates** – point all instructions/scripts to the bootstrap + schema index, retire legacy filenames in the narrative.
4. **Fixture Optimization** – use the domain map to scope truncation fixtures (e.g., strategy suites truncate only strategy tables) and/or reduce partition counts for tests.

## Detailed Bootstrap Refactor Plan

### Goals

1. Make `001_bootstrap_schema.sql` readable/maintainable by splitting it into clearly labeled sections (or sub-files) per domain.
2. Reduce the number of pre-created partitions (especially for tests) or gate the partition creation behind parameters to avoid populating 60+ months for every environment.
3. Ensure optional SQL (005–009) that is still relevant gets merged or explicitly re-run as part of the plan.

### Proposed Structure

```
001_bootstrap_schema.sql
├── Header (search_path, SETs, helper functions)
├── Section A: Feature Store
│   ├── Tables + constraints
│   ├── Indexes (B-tree + BRIN)
│   └── Partition creation
├── Section B: Model Store
│   ...
├── Section C: Strategy Store
├── Section D: Market Data
├── Section E: Data Registry
└── Section F: Optional Views/Aliases (from 008_views, 012_predictions_alias)
```

### Optional Includes
Instead of a single monolithic file, we can restructure as:

```
stores/
  migrations/
    bootstrap/
      00_header.sql
      10_feature_store.sql
      20_model_store.sql
      30_strategy_store.sql
      40_market_data.sql
      50_data_registry.sql
      60_views_aliases.sql
    001_bootstrap_schema.sql -- uses \i bootstrap/*.sql
```

This keeps the runtime entrypoint unchanged while making each domain file manageable.

### Partition Strategy Adjustments

- Add a flag/argument (environment variable or SQL plpgsql parameter) that controls how many months get pre-created. For tests we only need e.g., 3 months forward/backward; production can still pre-seed multiple years.
- Alternatively, move partition seeding into a separate SQL script that test fixtures can skip entirely.

### Optional SQL Integration

- `007_schema_hardening.sql`: merge constraint changes into the per-domain sections.
- `011_brin_indexes.sql`: move BRIN creation statements next to their tables (ensuring `IF NOT EXISTS` so reruns are safe).
- `012_predictions_alias.sql`: relocate alias/view creation to the model store section, gated by `CREATE OR REPLACE VIEW`.
- `014_update_parents_predictions.sql`: if still relevant, convert into explicit `ALTER TABLE`/`UPDATE` statements within the model store section.
- `009_disable_partition_triggers.sql` becomes obsolete once 002 is fully removed; can be dropped.

### Test Fixture Alignment

- After restructuring, update `ml/tests/fixtures/database_fixtures.py` to use a parameterized bootstrap (e.g., `TEST_DB_PARTITION_MONTHS=6`) so test DB setup doesn’t explode with hundreds of partitions.
- Rework `clean_postgres_db_module` to truncate only the tables touched by the module (leveraging the new section names/namespaces).

### CI & Script Updates

- Replace `.github/workflows/nightly.yml` call to `005_data_registry.sql` with the bootstrap file (or the new modular path).
- Update `ml/tests/unit/config/test_config.py` to assert against the new structure (e.g., checking the include list rather than legacy filenames).
- Refresh docs (`context_*`, `ROADMAP`, monitoring READMEs) to describe the new structure and point readers at this schema index.

### Milestones

1. Draft modular SQL files + regenerate `001_bootstrap_schema.sql` via `\i` includes.
2. Update fixtures + migration runners to handle the new layout and reduced partitions.
3. Clean up legacy references (tests, docs, CI).
4. Remove archived files once Git history + docs capture their contents.

---

# Part 2: Schema Consolidation V2 Plan

**Date**: 2025-11-14
**Methodology**: Agent Task Framework + Critical Safeguards
**Execution**: 5-phase TDD validation per task
**Goal**: Fix orphaned schemas, reduce partition explosion, improve test performance

## Executive Summary

The V2 plan addresses critical issues discovered through comprehensive analysis:

1. **Orphaned Schemas**: 8 tables defined but never created (earnings, macro, cross-asset, instrument metadata)
2. **Partition Explosion**: 60 months × 4 tables = 240 partitions → 300+ total relations → test timeouts
3. **Schema Duplication**: Objects in both `public` and `ml_registry` schemas (367 total tables)
4. **Test Performance**: Truncation timeouts, slow test suite, flaky CI

**Impact**: Production features broken (earnings ingestion), tests unreliable, development velocity impacted.

## V2 Recommended Approach

### Key Principles

1. **Activate, Don't Merge**: Add orphaned schemas as new migrations, don't rewrite existing ones
2. **Environment-Aware Partitions**: Single SQL file, runtime configuration (18 months production, 6 months tests)
3. **5-Phase TDD Validation**: Every task follows Test Design → Implementation → Static → Integration → System
4. **No Schema Changes During Refactor**: Only ADDS tables, never ALTERs existing ones (CRITICAL_SAFEGUARD)
5. **Rollback Safety**: Feature flags where applicable, reversible migrations, comprehensive testing

### Differences from Original Proposal

| Original Plan | V2 Plan | Rationale |
|---------------|---------|-----------|
| Merge orphaned schemas into bootstrap | Create separate migrations | Easier to track, test, and rollback |
| Modular includes (`bootstrap/*.sql`) | Single file with clear sections | Less complexity, proven pattern |
| Manual partition reduction | Environment-aware dynamic sizing | Flexible for production vs tests |
| Focus on file organization | Focus on execution and validation | Solves actual problems first |

## Critical Issues Identified

### Issue 1: Orphaned Schemas Cause Runtime Failures (HIGH PRIORITY)

**Evidence**:

- `ml/schema/earnings.sql` defines 3 tables NEVER executed
- `ml/stores/migrations/015_macro_release_calendar.sql` → `019_l2_minute.sql` define dataset tables NEVER in `_BASE_MIGRATIONS`
- 12+ files reference `ml.earnings_actuals` table that doesn't exist
- Code will raise `NoSuchTableError` when earnings features are used

**Impact**: Earnings ingestion broken, macro refresh broken, event features unavailable

**Solution**: Phase 1.1 and 1.2 (activate schemas)

### Issue 2: Partition Explosion Causes Test Timeouts (HIGH PRIORITY)

**Evidence**:

- Bootstrap creates 60 months of partitions (2023-2027)
- 4 tables × 60 months = 240 partitions
- Plus indexes, plus schema duplication = **300+ relations**
- `clean_postgres_db_module` timeout: 2000ms < 3000ms actual = regular failures

**Impact**: Test flakiness, CI failures, slow development feedback

**Solution**: Phase 1.3 (partition parameterization) + Phase 1.4 (timeout increase)

### Issue 3: Schema Duplication Wastes Resources (MEDIUM PRIORITY)

**Evidence**:

- 233 tables in `public` schema
- 134 tables in `ml_registry` schema
- Many duplicates across schemas
- Total: **367 tables** (should be <150)

**Impact**: Confusion about schema ownership, slower queries, harder maintenance

**Solution**: Phase 2.1 (consolidate namespaces)

## Task Breakdown

### Phase 0: Foundation (3 tasks, ~2 hours)

**Purpose**: Establish baseline metrics and validate assumptions

| Task | Estimated Time | Dependencies | Output |
|------|---------------|--------------|--------|
| 0.1 Audit Schema Duplication | 45 min | None | Schema ownership matrix |
| 0.2 Baseline Test Performance | 30 min | None | Performance baseline report |
| 0.3 Validate Orphaned References | 45 min | None | Impact assessment + reproduction tests |

**Deliverables**:

- Complete inventory of schema objects
- Decision matrix for consolidation
- Baseline performance metrics
- Evidence of orphaned table failures

### Phase 1: Immediate Fixes (4 tasks, ~5 hours)

**Purpose**: Stop runtime failures and eliminate test timeouts

| Task | Estimated Time | Dependencies | Output |
|------|---------------|--------------|--------|
| 1.1 Activate Earnings Schema | 1.5 hours | 0.3 | Working earnings ingestion |
| 1.2 Activate Macro Events Schema | 1.5 hours | 1.1 | Working macro refresh |
| 1.3 Implement Partition Parameterization | 2 hours | 0.2 | Configurable partitions (24 for tests) |
| 1.4 Increase Truncation Timeout | 30 min | 1.3 | Zero timeout failures |

**Success Metrics**:

- Zero `NoSuchTableError` for earnings/macro tables
- Partition count: 240 → 24 (tests) = **90% reduction**
- Truncation timeout failures: 5 → 0 = **100% elimination**
- Test suite reliability: Flaky → Stable

### Phase 2: Schema Cleanup (3 tasks, ~7.5 hours)

**Purpose**: Long-term maintainability and test performance

| Task | Estimated Time | Dependencies | Output |
|------|---------------|--------------|--------|
| 2.1 Consolidate Schema Namespace | 3 hours | 0.1 | <150 total tables |
| 2.2 Modularize Bootstrap | 2 hours | 2.1 | Readable, documented bootstrap |
| 2.3 Domain-Scoped Fixtures | 2.5 hours | 2.2 | 75% faster test cleanup |

**Success Metrics**:

- Table count: 367 → <150 = **59% reduction**
- Duplicate tables: Many → 0 = **100% elimination**
- Cleanup time for domain tests: 2000ms → 50ms = **97% improvement**
- Test suite duration: -30% overall

## Agent Task Execution Instructions

### For Future Claude Window

Each task follows the **5-Phase TDD Workflow** from AGENT_TASK_FRAMEWORK.md:

```
Phase 1: Test Design Agent
  ↓ (designs tests BEFORE implementation)
Phase 2: Implementation Agent
  ↓ (writes code to satisfy tests)
Phase 3: Static Validation Agent
  ├─ PASS → Phase 4
  └─ FAIL → Back to Phase 2
       ↓
Phase 4: Integration Validation Agent
  ├─ PASS → Phase 5 (if required) or APPROVED
  └─ FAIL → Back to Phase 2
       ↓
Phase 5: System Validation Agent (optional)
  ├─ PASS → APPROVED (commit)
  └─ FAIL → Back to Phase 2
```

### To Execute This Plan

**Single task**:

```
Execute schema consolidation task Phase 1.1
```

**Full phase**:

```
Execute all Phase 1 schema consolidation tasks sequentially
```

**Full plan**:

```
Execute complete schema consolidation plan (Phases 0-2)
```

### Task Definitions Location

All task definitions are in:

```
/home/nate/projects/nautilus_trader/tasks/schema_consolidation/
  phase_0_1_audit_schema_duplication.md
  phase_0_2_baseline_test_performance.md
  phase_0_3_validate_orphaned_refs.md
  phase_1_1_activate_earnings_schema.md
  phase_1_2_activate_macro_events_schema.md
  phase_1_3_implement_partition_parameterization.md
  phase_1_4_increase_truncation_timeout.md
  phase_2_1_consolidate_schema_namespace.md
  phase_2_2_modularize_bootstrap.md
  phase_2_3_domain_scoped_fixtures.md
```

### Reports Location

Agent outputs will be written to:

```
/home/nate/projects/nautilus_trader/reports/schema_consolidation/
  tests/                    # Phase 1: Test design reports
  implementations/          # Phase 2: Implementation reports
  validations/             # Phase 3-5: Validation reports
```

## Critical Safeguards Applied

### From CRITICAL_SAFEGUARDS.md

**Category 5: No Schema Changes During Refactor**

- ✅ **Applied**: Only ADDS tables (002_earnings, 003_macro_events)
- ✅ **Applied**: Never ALTERs existing tables
- ✅ **Validated**: Phase 3 checks for ALTER TABLE statements (auto-REJECT)

**Category 3: Test Execution Verification**

- ✅ **Applied**: Phase 4 MUST verify "X passed" not "X collected"
- ✅ **Applied**: Reproduction tests for orphaned tables
- ✅ **Validated**: Earnings/macro tests must RUN after activation

**Category 2: No Stubs/TODOs**

- ✅ **Applied**: All SQL migrations must be complete
- ✅ **Validated**: Phase 3 greps for NotImplementedError/TODO (auto-REJECT)

**Category 9: Rollback Strategy**

- ✅ **Applied**: Each migration has down migration documented
- ✅ **Applied**: Git commits only after Phase 4 PASS (or Phase 5 if required)
- ✅ **Validated**: Can revert any task independently

**Category 10: Report Standardization**

- ✅ **Applied**: All agents follow standardized report templates
- ✅ **Applied**: Each phase generates required sections
- ✅ **Validated**: Next agent validates previous agent's report format

## Schema Consolidation Strategy

### Active Migrations (After Phase 1)

```python
# ml/stores/migrations_runner.py
_BASE_MIGRATIONS: Final[tuple[str, ...]] = (
    "ml/registry/migrations/001_initial_schema.sql",
    "ml/registry/migrations/002_add_cold_path_fields.sql",
    "ml/registry/migrations/003_add_artifact_digest.sql",
    "ml/stores/migrations/001_bootstrap_schema.sql",
    "ml/stores/migrations/002_earnings_schema.sql",      # Phase 1.1
    "ml/stores/migrations/003_macro_events_schema.sql",  # Phase 1.2
)

# After Phase 2.1:
_BASE_MIGRATIONS: Final[tuple[str, ...]] = (
    # ... same as above ...
    "ml/stores/migrations/004_consolidate_schema_namespace.sql",  # Phase 2.1
)
```

### Partition Strategy

**Before Phase 1.3**:

- 60 months hardcoded
- 240 partitions pre-created
- 300+ total relations

**After Phase 1.3**:

```sql
-- Environment-aware in 001_bootstrap_schema.sql
DO $$
DECLARE
    partition_months INTEGER;
BEGIN
    partition_months := COALESCE(
        NULLIF(current_setting('ml.partition_months', true), '')::INTEGER,
        18  -- Default: production
    );
    -- Tests set: ml.partition_months = 6
    PERFORM create_monthly_partitions('ml_feature_values', ..., partition_months);
END $$;
```

**Result**:

- Production: 18 months = 72 partitions
- Tests: 6 months = 24 partitions
- 90% reduction for tests

### Schema Ownership (After Phase 2.1)

| Schema | Purpose | Tables |
|--------|---------|--------|
| `public` | Store data (time-series values) | ml_feature_values, ml_model_predictions, ml_strategy_signals, market_data |
| `ml_registry` | Registry metadata | models, features, strategies, registry_audit_log |
| `ml` | Domain-specific features | earnings_*, macro_*, events_*, cross_asset_* |

**Rule**: No duplicates across schemas

## Success Criteria

### Phase 0 Complete When

- [ ] Schema duplication matrix created (all objects inventoried)
- [ ] Baseline performance metrics captured (partition counts, truncation times)
- [ ] Orphaned table failures reproduced (evidence for Phase 1)

### Phase 1 Complete When

- [ ] Earnings tables created and tested (`ml.earnings_actuals`, `ml.earnings_estimates`, `ml.earnings_calendar`)
- [ ] Macro tables created and tested (5 tables in `ml` schema)
- [ ] Partition count reduced to 24 for tests (from 240)
- [ ] Truncation timeout set to 10000ms (from 2000ms)
- [ ] Zero timeout failures in test suite
- [ ] All orphaned table reproduction tests now PASS (not fail)

### Phase 2 Complete When

- [ ] Total table count <150 (from 367)
- [ ] Zero duplicate tables across schemas
- [ ] Bootstrap file reorganized with clear sections
- [ ] Domain-scoped fixtures implemented
- [ ] Test cleanup time <500ms on average (from 2000ms)
- [ ] Test suite 30% faster overall

### Overall Success

- [ ] All entries in `ml/config/dataset_ids.py` have corresponding tables
- [ ] Earnings ingestion CLI works: `python -m ml.cli.ingest_earnings`
- [ ] Macro refresh CLI works: `python -m ml.cli.hydrate_feature_caches`
- [ ] Full test suite passes with zero flakes
- [ ] CI/CD duration reduced by 20-30%
- [ ] Developer feedback loop significantly faster

## Risk Mitigation

### High Risk Items

**1. Schema Migrations in Production**

- Mitigation: Each migration is idempotent (`CREATE TABLE IF NOT EXISTS`)
- Mitigation: Down migrations documented for rollback
- Mitigation: Applied during low-traffic windows

**2. Test Suite Breakage**

- Mitigation: Backward compatible (unmarked tests get full cleanup)
- Mitigation: Extensive Phase 4 validation before approval
- Mitigation: Can rollback any phase independently

**3. Partition Count Too Low**

- Mitigation: Runtime partition creation via `PartitionManager`
- Mitigation: 18 months production default is conservative
- Mitigation: Can adjust via environment variable

### Medium Risk Items

**4. Schema Namespace Conflicts**

- Mitigation: Phase 0.1 audit identifies all conflicts
- Mitigation: Phase 2.1 uses decision matrix
- Mitigation: Phase 5 system validation required

**5. Performance Regression**

- Mitigation: Phase 0.2 baseline established
- Mitigation: Phase 2.3 measures before/after
- Mitigation: Can increase partition count if needed

## Estimated Timeline

| Phase | Duration | Parallelizable | Total |
|-------|----------|----------------|-------|
| Phase 0 | 2 hours | Yes (3 tasks) | 2 hours |
| Phase 1 | 5 hours | Partially (1.1+1.2 parallel after 1.3) | 5 hours |
| Phase 2 | 7.5 hours | No (sequential dependencies) | 7.5 hours |
| **Total** | **14.5 hours** | - | **~2 dev days** |

**With 5-phase validation overhead**: ~3 dev days total

## Next Actions

1. **Review this plan** with team/stakeholders
2. **Verify task definitions** in `tasks/schema_consolidation/`
3. **Execute Phase 0** to validate assumptions
4. **Based on Phase 0 results**, confirm or adjust Phase 1-2 approach
5. **Execute sequentially**, committing after each approved phase

## References

- **Task Definitions**: `/home/nate/projects/nautilus_trader/tasks/schema_consolidation/`
- **Agent Framework**: `/home/nate/projects/nautilus_trader/AGENT_TASK_FRAMEWORK.md`
- **Critical Safeguards**: `/home/nate/projects/nautilus_trader/CRITICAL_SAFEGUARDS.md`
- **Coding Standards**: `/home/nate/projects/nautilus_trader/CLAUDE.md`

---

**Document Version**: 2.0
**Original Plan**: Lines 1-163 (preserved as Part 1)
**V2 Addendum**: Lines 165+ (added 2025-11-14)
**Status**: Ready for agent execution
**Last Updated**: 2025-11-14
