# DRY Deduplication Plan (Phased Checklist)

## Objective
Eliminate `ml/tasks/**` as a business-logic owner.

End state:
- Reusable logic lives in `ml/<domain>/**`, `ml/<domain>/common/**`, or `ml/common/**` (per `AGENTS.md` boundaries).
- `ml/cli/**` is the only CLI argument-parsing boundary.
- `ml/tasks/**` becomes temporary compatibility shims, then is removed.

## Scope
In scope:
- Entire `ml/tasks/**` tree.
- Canonical destination modules under `ml/data/**`, `ml/orchestration/**`, `ml/training/**`, `ml/registry/**`, `ml/stores/**`, `ml/observability/**`, `ml/core/common/**`, and `ml/common/**`.
- Scheduler-adjacent dedupe work previously tracked in this plan.

Out of scope:
- Feature/model algorithm changes.
- Hot-path behavioral changes beyond strict equivalence.

## Non-Negotiable Architecture Constraints
- Domain modules must not import `ml/tasks/**`.
- Domain/service modules must not own `argparse`, `input()`, or `sys.exit()` control flow.
- Shared helpers must move to `ml/common/**` when cross-domain; `ml/<domain>/common/**` when intra-domain.
- One implementation per capability (no parallel task/domain copies).

## Current Status
Completed from prior plan:
- [x] Shared market input parser extraction to `ml/common/cli_parsers.py`.
- [x] Coverage persistence/backend resolution dedupe in `ml/tasks/monitoring/coverage.py`.
- [x] Full `ml/tasks/**` migration ledger created: `ml/docs/development/DRY_DEDUP_TASK_MIGRATION_LEDGER.md`.
- [x] Non-CLI runtime imports of `ml.tasks/**` removed from orchestration/core runtime paths.
- [x] Retired root compatibility namespace module `ml/tasks/__init__.py`.
- [x] Retired monitoring and observability task namespace shims (`ml/tasks/monitoring/__init__.py`, `ml/tasks/observability/__init__.py`).
- [x] Retired package namespace shims for caches/datasets/ingest/pipelines/training/dev (`ml/tasks/{caches,datasets,ingest,pipelines,training,dev}/__init__.py`) after migrating CLI consumers to canonical owners.
- [x] Retired leaf module shims `ml/tasks/dev/sanity_check.py` and `ml/tasks/pipelines/{runner,scheduler}.py` after import migration completion.
- [x] Retired `ml/tasks/db.py` shim after migrating CLI/tests to canonical store migration owners.
- [x] Retired additional low-consumer leaf shims `ml/tasks/{caches/hydration,registry,training/quick,training/hpo_tft}.py` after test-consumer migration.
- [x] Retired dataset and ingest leaf shims `ml/tasks/datasets/{production,report,splits,tft,tft_cli}.py` and `ml/tasks/ingest/{alternative,backfill,l2,recent,supplementary,yahoo}.py` after migrating fixture/test consumers to canonical owners.
- [x] Retired final monitoring/observability leaf shims `ml/tasks/monitoring/{coverage,health}.py` and `ml/tasks/observability/{backfill,flush}.py` after migrating compatibility tests to canonical imports.
- [x] Removed stale `ml/tasks/**` references from docs/context artifacts and refreshed canonical ownership callouts.
- [x] Tightened Python stale-import guardrails to quarantine `ml.tasks` references to migration-history tests only.

Remaining:
- [ ] Finish full-suite quality gate validation for final closure.

## Canonical Ownership Targets
Capability | Canonical owner (target) | Transitional source
--- | --- | ---
CLI parsers/env coercion | `ml/common/cli_parsers.py` and `ml/common/**` | `ml/tasks/*`
Dataset build/report/splits orchestration | `ml/data/**` and `ml/training/common/**` | `ml/tasks/datasets/*`
Ingestion orchestration helpers | `ml/data/ingest/**` | `ml/tasks/ingest/*`
Coverage planning/classification/backfill orchestration | `ml/data/coverage/**` plus monitoring domain modules | `ml/tasks/monitoring/coverage.py`
Pipeline health aggregation/reporting | `ml/core/common/health_monitoring.py` plus monitoring domain modules | `ml/tasks/monitoring/health.py`
Cache hydration orchestration | `ml/data/**` (rehydration/cache components) | `ml/tasks/caches/hydration.py`
Pipeline schedule/run orchestration | `ml/orchestration/**` | `ml/tasks/pipelines/*`
Registry operations | `ml/registry/**` | `ml/tasks/registry.py`
Migration planning/sql splitting | `ml/stores/**` and `ml/core/common/**` | `ml/tasks/db.py`
Training quick/HPO workflows | `ml/training/**` | `ml/tasks/training/*`
Observability flush/backfill orchestration | `ml/observability/**` and `ml/core/common/observability.py` | `ml/tasks/observability/*`

## Phase 0: Freeze Boundaries and Baseline
Checklist:
- [x] Record module-by-module ownership map for every file under `ml/tasks/**` (see `ml/docs/development/DRY_DEDUP_TASK_MIGRATION_LEDGER.md`).
- [x] Add temporary guardrail: no new feature logic lands under `ml/tasks/**`.
- [x] Add temporary guardrail: non-CLI runtime code cannot add new imports from `ml/tasks/**`.
- [ ] Freeze baseline behavior with focused tests for affected CLI and orchestration flows.

Guardrail implementation notes:
- `ml/tests/unit/common/test_tasks_migration_guardrails.py` enforces ledger coverage, shim-only status for `done` task modules, and frozen hashes for non-done task modules.
- `ml/tests/unit/common/test_tasks_migration_guardrails.py` also enforces a deterministic baseline list of non-CLI runtime modules that still import `ml.tasks` (fail-on-drift).

Verification:
- [ ] `poetry run pytest -q ml/tests/unit/cli`
- [ ] `poetry run pytest -q ml/tests/unit/data -k "scheduler or coverage"`
- [ ] `poetry run pytest -q ml/tests/unit/orchestration`
- [ ] `poetry run mypy ml --strict`
- [ ] `poetry run ruff check ml`

## Phase 1: Move Non-CLI Task Libraries First (Low Risk)
Targets (no `argparse`/`main` ownership):
- `ml/tasks/datasets/{__init__.py,tft.py,splits.py,report.py,production.py}`
- `ml/tasks/ingest/{alternative.py,l2.py,recent.py,supplementary.py,yahoo.py}`
- `ml/tasks/training/quick.py`
- `ml/tasks/registry.py`
- `ml/tasks/db.py`
- `ml/tasks/caches/hydration.py`

Checklist:
- [ ] Relocate each module to canonical domain path with equivalent typed API.
- [ ] Keep `ml/tasks/*` modules as thin re-export compatibility shims only.
- [ ] Add deprecation note/warning path for shim imports (non-breaking).
- [ ] Update non-CLI imports to new canonical modules first.
- [ ] Add/adjust tests at canonical module paths.
- [x] Migrate `ml/tasks/datasets/tft.py` to canonical owners (`ml/data/build.py`, `ml/data/__init__.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/datasets/splits.py` to canonical owner (`ml/training/common/cross_validation.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/datasets/report.py` to canonical owner (`ml/data/validation.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/datasets/production.py` to canonical owner (`ml/data/collectors/production_collector.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/datasets/__init__.py` to shim-only re-exports from canonical owners (`ml/data/__init__.py`).
- [x] Remove non-CLI runtime imports of `ml.tasks.datasets*` in `ml/experiments/{chronos_training_experiment.py,tft_training_experiment.py}`, `ml/pipelines/build_runner.py`, and `ml/orchestration/pipeline_orchestrator_cli.py`.
- [x] Migrate `ml/tasks/training/quick.py` to canonical training owner (`ml/training/teacher/quick.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/registry.py` to canonical registry owner (`ml/registry/feature_operations.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/caches/hydration.py` to canonical rehydration owner (`ml/data/rehydration/cache_hydration.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/caches/__init__.py` to canonical rehydration package exports (`ml/data/rehydration/__init__.py`).
- [x] Migrate `ml/tasks/db.py` to canonical owners (`ml/stores/migrations_runner.py`, `ml/core/common/database_lifecycle.py`, `ml/stores/common/sql_splitter.py`) and convert task module to shim.
- [x] Remove non-CLI runtime imports of `ml.tasks.db` in `ml/stores/migrations_runner.py` and `ml/core/common/database_lifecycle.py`.
- [x] Start migration of `ml/tasks/ingest/__init__.py` by moving non-backfill exports to canonical ingest package (`ml/data/ingest/__init__.py`) and routing task package exports through canonical owners.
- [x] Complete `ml/tasks/ingest/__init__.py` shim-only migration for `ingest_backfill_main` after `ml/tasks/ingest/backfill.py` canonicalization.
- [x] Migrate `ml/tasks/training/__init__.py` to shim-only re-exports from canonical training package (`ml/training/__init__.py`).
- [x] Migrate `ml/tasks/monitoring/__init__.py` to shim-only re-exports from canonical monitoring package (`ml/monitoring/__init__.py`).
- [x] Migrate `ml/tasks/observability/__init__.py` to shim-only re-exports from canonical observability package (`ml/observability/__init__.py`).
- [x] Migrate `ml/tasks/pipelines/__init__.py` to shim-only re-exports from canonical orchestration package (`ml/orchestration/__init__.py`).
- [x] Migrate `ml/tasks/pipelines/scheduler.py` to canonical scheduler owner (`ml/orchestration/scheduler.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/pipelines/runner.py` to canonical runner owner (`ml/orchestration/pipeline_runner.py`) and convert task module to shim.

Verification:
- [ ] `poetry run pytest -q ml/tests/unit -k "datasets or ingest or training or registry or migrations or hydration"`
- [ ] `poetry run mypy ml --strict`
- [ ] `poetry run ruff check ml`

## Phase 2: Split CLI Control Flow from Task Modules (Medium/High Risk)
Targets (currently mixed service + CLI logic):
- `ml/tasks/datasets/tft_cli.py`
- `ml/tasks/ingest/backfill.py`
- `ml/tasks/monitoring/coverage.py`
- `ml/tasks/monitoring/health.py`
- `ml/tasks/observability/{flush.py,backfill.py}`
- `ml/tasks/training/hpo_tft.py`
- `ml/tasks/dev/sanity_check.py`

Checklist:
- [ ] Extract pure service/orchestration logic into domain modules (`ml/data/**`, `ml/monitoring/**`, `ml/observability/**`, `ml/training/**`, `ml/common/**`).
- [ ] Keep CLI parsing and process exit behavior in `ml/cli/**` only.
- [ ] Remove `input()` and `sys.exit()` calls from domain/service functions; return typed results/errors.
- [ ] Preserve CLI flags, output contracts, and behavior.
- [ ] Leave `ml/tasks/*` as shim layer (delegation only).
- [x] Migrate `ml/tasks/datasets/tft_cli.py` to canonical owners (`ml/cli/build_tft_dataset.py`, data-domain dataset build helpers) and convert task module to shim.
- [x] Migrate `ml/tasks/ingest/backfill.py` to canonical owners (`ml/data/ingest/orchestrator.py`, `ml/orchestration/ingestion_coordinator.py`, `ml/cli/ingest_backfill.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/dev/sanity_check.py` to canonical owner (`ml/tools/sanity_check.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/observability/backfill.py` to canonical owners (`ml/observability/backfill.py`, `ml/cli/observability_backfill.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/observability/flush.py` to canonical owners (`ml/core/common/observability.py`, `ml/observability/scheduler.py`, `ml/cli/observability.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/monitoring/health.py` to canonical owners (`ml/core/common/health_monitoring.py`, `ml/monitoring/health.py`, `ml/cli/{health,check_pipeline_health}.py`) and convert task module to shim.
- [x] Migrate `ml/tasks/monitoring/coverage.py` to canonical owners (`ml/cli/coverage.py`, `ml/data/coverage/manager.py`, ingestion/orchestration coverage backfill owners) and convert task module to shim.
- [x] Migrate `ml/tasks/training/hpo_tft.py` to canonical owners (`ml/training/teacher/hpo_tft.py`, `ml/cli/hpo_tft.py`) and convert task module to shim.

Verification:
- [x] `poetry run pytest -q ml/tests/unit/cli`
- [x] `poetry run pytest -q ml/tests/unit -k "coverage or health or observability or hpo or ingest_backfill or tft_cli"`
- [x] `poetry run mypy ml --strict`
- [x] `poetry run ruff check ml`

## Phase 3: Scheduler and Parallel Manager Unification (Carried Forward)
Targets:
- `ml/data/scheduler.py`
- `ml/data/common/{data_collection.py,orchestrator_collection.py,scheduler_init.py,daily_update_orchestrator.py,data_cleanup.py,scheduler_feature_job.py}`
- `ml/data/{initialization_manager.py,registry_integrator.py,data_retention_manager.py,trading_day_calculator.py}`

Checklist:
- [ ] Decide canonical scheduler subflow ownership and remove parallel implementations.
- [ ] Delegate `DataScheduler` to canonical components only.
- [ ] Retire redundant manager modules or convert to temporary shims.
- [ ] Preserve metrics/event semantics and fallback logging (`exc_info=True`).

Verification:
- [ ] `poetry run pytest -q ml/tests/unit/data/common`
- [ ] `poetry run pytest -q ml/tests/unit/data -k scheduler`
- [ ] `poetry run pytest -q ml/tests/unit -k "orchestrator or registry"`
- [ ] `poetry run mypy ml --strict`
- [ ] `poetry run ruff check ml`

## Phase 4: Remove Runtime Dependency on `ml/tasks/**`
Checklist:
- [x] Update runtime imports in `ml/core/**`, `ml/orchestration/**`, `ml/deployment/**`, `ml/training/**`, `ml/stores/**` to canonical domain modules.
- [x] Enforce import boundary: only `ml/cli/**` (and temporary compatibility tests) may import `ml/tasks/**`.
- [x] Validate no direct runtime/business import remains.

Verification:
- [x] `rg -n "from ml\\.tasks|import ml\\.tasks|ml\\.tasks\\." ml --glob '*.py' -g '!ml/cli/**' -g '!ml/tests/**'` returns no results.
- [x] `poetry run mypy ml --strict`
- [x] `poetry run ruff check ml`
- [x] `make validate-fixtures`

## Phase 5: Remove `ml/tasks/**` Compatibility Layer
Checklist:
- [x] Migrate remaining package-level imports from `ml/tasks/{caches,datasets,ingest,pipelines,training,dev}` to canonical owners (`ml/data/**`, `ml/data/rehydration/**`, `ml/data/ingest/**`, `ml/orchestration/**`, `ml/training/**`, `ml/tools/**`).
- [x] Retire package namespace shims `ml/tasks/{caches,datasets,ingest,pipelines,training,dev}/__init__.py` after import migration.
- [x] Retire low-risk leaf module shims `ml/tasks/dev/sanity_check.py` and `ml/tasks/pipelines/{runner,scheduler}.py` after migration completion.
- [x] Migrate `ml/cli/apply_migrations.py` and fixture wiring from `ml.tasks.db` imports to canonical `ml/stores/**` owners.
- [x] Retire `ml/tasks/db.py` compatibility shim after migration of remaining consumers.
- [x] Retire additional low-consumer leaf shims `ml/tasks/{caches/hydration,registry,training/quick,training/hpo_tft}.py` after migration of remaining test consumers.
- [x] Retire dataset and ingest leaf module shims `ml/tasks/datasets/{production,report,splits,tft,tft_cli}.py` and `ml/tasks/ingest/{alternative,backfill,l2,recent,supplementary,yahoo}.py` after compatibility test/fixture migration.
- [x] Retire final monitoring/observability shim modules `ml/tasks/monitoring/{coverage,health}.py` and `ml/tasks/observability/{backfill,flush}.py` after migration completion.
- [x] Update docs/examples/tests to canonical imports only.
- [x] Remove stale references in `ml/docs/development/ANTI_PATTERNS.md` and architecture docs.

Verification:
- [ ] `rg -n "from ml\\.tasks|import ml\\.tasks|ml\\.tasks\\." ml --glob '*.py'` returns no results.
- [ ] `poetry run pytest -q ml/tests`
- [ ] `poetry run mypy ml --strict`
- [ ] `poetry run ruff check ml`
- [ ] `make validate-metrics`
- [ ] `make validate-events`
- [ ] `poetry run coverage report`

## Suggested Execution Order
- [ ] Phase 0 PR: ownership map + guardrails.
- [ ] Phase 1 PR(s): non-CLI library relocations.
- [ ] Phase 2 PR(s): CLI/service splits for heavy task modules.
- [ ] Phase 3 PR: scheduler + manager canonicalization.
- [x] Phase 4 PR: import-boundary cleanup.
- [ ] Phase 5 PR: remove `ml/tasks/**`.

## Ranked Next Modules (Post-Session Snapshot)
Low-to-high risk ordering for the next continuation:

1. Remove stale `ml/tasks/**` references from docs/context artifacts and runbooks.
Rationale: completed.
Expected blast radius: completed.

2. Tighten final stale-import guardrails and boundary scans to disallow any Python `ml.tasks` references outside migration-history tests.
Rationale: completed.
Expected blast radius: completed.

3. Run full Phase 5 closure gates (`pytest -q ml/tests`, coverage report) and prune residual migration-history-only references if no longer needed.
Rationale: medium risk because whole-suite validation may expose latent coupling unrelated to task shim retirement.
Expected blast radius: full `ml/tests` suite, coverage baseline, and migration-history tests.

4. Resume Phase 3 scheduler/parallel-manager unification after Phase 5 closure gates are green.
Rationale: high risk due orchestration/data scheduler coupling and operational behavior sensitivity.
Expected blast radius: `ml/data/**`, `ml/orchestration/**`, scheduler tests, and integration pipelines.

## Risk Controls
- [ ] One capability cluster per PR; avoid large mixed moves.
- [ ] Add before/after tests in same PR as each relocation.
- [ ] Preserve CLI contracts unless explicitly versioned.
- [ ] Keep temporary compatibility shims only as long as required.

## Completion Definition
- [x] `ml/tasks/**` no longer owns business logic.
- [x] Every capability has a single canonical owner in domain/common modules.
- [x] No runtime imports of `ml/tasks/**` remain.
- [ ] All quality gates pass and docs are updated.
