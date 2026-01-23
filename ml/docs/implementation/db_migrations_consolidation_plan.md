# ML Store Migrations Consolidation Plan

## Background

The current `ml/stores/migrations/` folder mixes a legacy incremental series
with a consolidated bootstrap migration. Both are applied on a fresh database,
which causes duplicate numbering and a disjointed setup flow. The migration
runner and Docker init scripts execute all `*.sql` files in order, so a clean
DB sees both tracks.

Definition of Done:
- The problem statement and root cause are captured in this plan.
- The consolidation strategy is summarized in a single, shared location.

## Goals

- Provide a single canonical bootstrap migration for greenfield databases.
- Keep legacy incremental migrations available for existing databases.
- Reduce migration count to a small, intentional set going forward.
- Support per-data-class market tables with a compatibility path.
- Preserve schema safety and compatibility while modernizing layout.

Definition of Done:
- Goals are approved and mapped to concrete tasks.
- Success criteria are measurable (clean DB bootstraps without duplicate apply).

## Guardrails

- Do not delete or reorder migrations that may already be applied.
- No destructive changes to existing databases; new bootstrap applies only to
  empty DBs or opt-in profiles.
- Use the project migration runner and SQL splitter; avoid inline semicolons in
  comments that can break parsing.
- Maintain 4-store and 4-registry alignment; keep required fields in persisted
  records (`instrument_id`, `ts_event`, `ts_init`).
- Keep hot paths clean of file I/O or heavy allocations; store writers must
  remain non-blocking on the event bus.
- Tunables live in config; no hard-coded schema behavior in code paths.

Definition of Done:
- Guardrails are documented and referenced in task acceptance criteria.
- The rollout plan includes checks that enforce the guardrails.

## Scope and Inventory (Current State)

- Core ML stores: feature values, predictions, signals, and related helpers
  appear in both the bootstrap and older incremental schema files.
- Market data: a monolithic `public.market_data` table exists in legacy
  migrations and the bootstrap.
- Data registry and events: dataset registry, events, watermarks, lineage,
  and event metadata functions appear in multiple migrations.
- Views and helpers: duplicate view and helper definitions exist across
  separate migration files.
- Partition helpers: auto-partitioning helpers exist, then later disabled via
  follow-on migrations.
- Feature families: macro and microstructure tables exist as standalone
  migrations.
- Seeds and hardening: test seeds, schema hardening, and BRIN indexes are
  scattered across incremental files.

Definition of Done:
- Every existing migration is mapped to a canonical schema section.
- Duplicate or deprecated definitions are clearly tagged for exclusion from
  the new bootstrap.

## Tasks

### Phase 1: Canonical inventory and consolidation map

- [ ] Audit all `ml/stores/migrations/*.sql` files and map each object
      (tables, functions, views, indexes, constraints) to a canonical section.
- [ ] Identify duplicates, conflicts, and deprecations (e.g., partition
      triggers that are later disabled).
- [ ] Decide which items are excluded from the bootstrap (test seeds,
      legacy helpers, or superseded objects).
- [ ] Produce a consolidated schema outline that becomes the blueprint for the
      new bootstrap file.

Definition of Done:
- A single inventory document exists with object-to-source mapping.
- Exclusions and replacements are explicit and reviewed.

### Phase 2: New bootstrap + migration layout

- [x] Create `ml/stores/migrations_bootstrap/001_bootstrap.sql` with organized
      sections for helpers, core stores, registry/events, market data, feature
      families, views, and indexes.
- [x] Create `ml/stores/migrations_legacy/` to archive incremental migrations
      used by existing databases.
- [x] Move or copy legacy migrations without changing their contents or order.
- [x] Ensure dataset registry type expansions and event metadata helpers are
      included in the bootstrap.

Definition of Done:
- A clean DB created with the bootstrap mirrors the intended schema.
- Legacy migrations remain intact and accessible for existing DBs.

### Phase 3: Runner and Docker init selection

- [x] Update the migration runner to select bootstrap for empty DBs or allow
      `ML_MIGRATIONS_PROFILE=bootstrap|legacy`.
- [x] Update Docker init to mount only the bootstrap SQL for fresh DBs so
      incremental history is not double-applied.
- [x] Preserve compatibility with existing DBs by keeping legacy migrations
      available to the runner.

Definition of Done:
- Fresh DBs apply only the bootstrap path.
- Existing DBs can continue on legacy migrations without schema drift.

### Phase 4: Market data table modernization

- [x] Add a single new migration that creates per-data-class tables
      (`bar`, `quote_tick`, `trade_tick`, `mbp1`, `tbbo`).
- [x] Add a compatibility view for `public.market_data` if legacy readers
      must remain supported.
- [x] Update writers, coverage, and catalog rehydration to target the correct
      per-class table.
- [x] Update dataset registry typing or mapping so the data class routing is
      explicit and testable.

Definition of Done:
- Writes and reads route to the correct per-class table.
- Legacy readers are not broken (view or compatibility path exists).

### Phase 5: Validation and testing

- [x] Run `poetry run mypy ml --strict` and address any typing regressions.
- [x] Run `poetry ruff check ml` and keep it clean.
- [x] Run `make validate-fixtures`.
- [ ] Run `make validate-metrics` if metrics changes are introduced.
- [x] Run focused pytest suites for migrations runner and store writers.
- [ ] Run `make validate-events` if event schema changes are introduced.

Definition of Done:
- All required commands pass or have documented, approved exceptions.
- Schema audit on a clean DB passes with the new bootstrap.

### Phase 6: Documentation and rollout

- [x] Document the new migration layout and selection rules.
- [x] Note the legacy path policy and any migration drift handling guidance.
- [ ] Communicate the change to operators and contributors.

Definition of Done:
- Documentation is updated and cross-referenced in relevant READMEs.
- Operators know which profile to use for new vs existing databases.

## Open Questions

- [ ] Do we need additional log/metadata tables beyond registry and events?
- [ ] Should per-data-class market tables replace the monolithic table entirely
      or keep it as a long-term compatibility view?
- [ ] Are any migrations in `ml/stores/migrations` still required for test
      environments only (seed data) and should be isolated to test profiles?

Definition of Done:
- Each open question has an owner and a recorded decision.
