# Legacy Removal Plan (Component-Facade Only)

## Objective
Remove legacy implementations and feature flags while preserving real functional behavior. Implementation-specific differences (and tests that encode them) are expected to change; what must remain is the observable, contract-defined behavior. Any functionality lost by removing legacy paths must be replaced by explicit, tested behavior in the canonical implementation. The component-facade path becomes the single production path with explicit contracts, performance guardrails, and observability.

## Guiding Principles
- Use the objective above as the decision filter: prioritize real behavior and explicit contracts over legacy-implementation equivalence.
- Keep changes minimal, typed, and safe (mypy/ruff clean).
- Maintain hot-path rules (no I/O, no DataFrame creation, no allocations in tight loops).
- Preserve config-driven behavior and explicit event/bus schemas.
- Parity tests prove behavior; shadow/dual-run proves real data safety.

## Current Focus (Legacy Removal Closure)
- [x] Confirm no legacy switches, flags, or selectors remain in code (facade-only path).
- [x] Update docs/tests to remove legacy-flag semantics (remaining references are historical notes).
- [x] Confirm no orphaned legacy modules remain on disk.

## Current Baseline (as of this plan)
- ML-only coverage: 59.76% (`poetry run coverage report --include='ml/*'`).
- Performance tests skipped under coverage by design.

## Recent Progress (latest session)
Note: repo scan shows several dual-implementation pairs still present; validate historical bullets below against current branch before acting.
- Removed `use_legacy_trainer` helpers from `ml/training/base_facade.py`; decide whether `ml/training/base.py` should become a facade shim or remain the canonical trainer module.
- Updated training imports/exports (`ml/training/__init__.py`, trainer modules, CLI validation, models/examples).
- Converted trainer parity suite to facade-only contract tests and removed legacy flag tests.
- Updated dashboard blueprints to use `get_integration_manager()` and added registry deployment helpers in the facade path.
- Docs refreshed to point to facade paths (training, dashboard, analyzer fixtures).
- Validation: `poetry run mypy ml --strict`, `poetry run ruff check ml`, targeted pytest across training + dashboard (warnings noted below).
- Investigated pytest warnings: converted `ml/tests/fixtures/stores.py` to `TYPE_CHECKING` import for `TestDatabase`; warning for `ml.tests.fixtures.database_fixtures` still persists.
- Latest warning repro: `poetry run pytest ml/tests/unit/training/test_base_trainer_facade.py -q` (assert-rewrite + external dep deprecations).
- ModelRegistry remains canonical in `ml/registry/model_registry_facade.py`; added `ml/registry/model_registry.py` shim for non-facade imports.
- Updated imports/tests/docs to use `ml.registry.model_registry_facade` and removed env-var toggles from ModelRegistry E2E tests.
- Consolidated scheduler: `ml/data/scheduler.py` is the canonical component-based implementation, `ml/data/scheduler_facade.py` is a shim, `ml/data/scheduler_legacy.py` removed, and scheduler tests now patch the canonical module.
- Added a shared scheduler metrics module and moved component imports to `ml/data/scheduler_metrics.py`.
- Updated scheduler integration/unit tests to use canonical component APIs (Databento loader patching, targeted update helpers).
- Validation: `poetry run ruff check ml`, `poetry run mypy ml --strict --no-incremental`, targeted pytest for scheduler suites (unit/facade/e2e/integration/analyzer).
- Consolidated TFT dataset builder: `ml/data/tft_dataset_builder.py` is canonical, `ml/data/tft_dataset_builder_facade.py` is a shim, `ml/data/tft_dataset_builder_legacy.py` removed, and unit/facade/integration tests now patch the canonical module.
- Removed legacy integration manager switch and monolith; `ml/core/integration.py` now re-exports the facade, helpers moved into `ml/core/integration_facade.py`, and parity tests were converted to facade invariants.
- Added facade-only integration fallbacks (dummy mode, health/protocol checks) and aligned core tests for the new facade-only path.
- Validation: `poetry run ruff check ml`, `poetry run mypy ml --strict`, targeted pytest for integration manager + typing suites (see warnings below).
- Consolidated dashboard app/service: `ml/dashboard/app.py` and `ml/dashboard/service.py` are canonical, facade modules are shims, and tests now target canonical imports.
- Consolidated data registry: `ml/registry/data_registry.py` is canonical, `ml/registry/data_registry_facade.py` is a shim, and tests/imports now target the canonical registry.
- Consolidated pipeline orchestrator: `ml/orchestration/pipeline_orchestrator.py` is canonical with component delegation; `ml/orchestration/pipeline_orchestrator_facade.py` is a shim, and legacy feature flag helpers are no-ops with tests updated to alias parity.
- Consolidated signal actor: `ml/actors/signal.py` now re-exports the facade implementation (`ml/actors/signal_facade_impl.py`), `ml/actors/signal_facade.py` is a shim, and legacy flag/tests were removed.
- Consolidated strategy base: `ml/strategies/base.py` is a shim to `ml/strategies/base_facade.py`, and legacy flag/tests were removed.
- Converted `ml/actors/base_legacy.py` to a shim re-exporting `ml/actors/base.py`.
- Added parity smoke-check fields to `ml/config/actors.MLSignalActorConfig` for canonical config ownership.
- Removed `_legacy_orchestrator` delegation in `ml/orchestration/ingestion_coordinator.py` and deleted the field from `ml/orchestration/pipeline_orchestrator.py`.
- Dropped `as_legacy_cython=True` toggles in data collection modules and deleted `ml/data/scheduler.py.bak`.
- Removed legacy flag shims (`ml/config/feature_flags.py`, scheduler/TFT builder/dashboard/trainer helpers) and cleaned related tests/docs.
- Removed `_use_legacy_strategy_base` shim from `ml/strategies/base_facade.py` and deleted its tests.
- Added `ml/registry/model_registry.py` shim to support non-facade imports.
- Unified feature engineering types and entrypoints: `ml/features/config.py` + `ml/features/indicators.py` are canonical, and `ml/features/facade.py` now imports them directly (legacy module removed).
- Deleted `ml/actors/base_legacy.py`, removed the redundant parity test, and removed `ml/docs/implementation/streaming_pipeline_legacy_persistence_plan.md`.

### Warnings Observed (to investigate)
- PytestCollectionWarning: helper classes with `Test*` names and `__init__` (fix by marking `__test__ = False` or renaming).
- PytestAssertRewriteWarning: fixture plugins imported before rewrite (resolved by deferring `database_fixtures` import via plugin manager + lazy fixture exports).
- Deprecation warnings from `edgar` and Databento coverage policy (external deps).
- Mypy warning: unused section(s) in `pyproject.toml` (config hygiene; not a functional failure).
- Latest warning repro: `poetry run pytest ml/tests/unit/core/test_integration_manager_types.py::TestMLIntegrationManagerTypeAnnotations::test_data_store_has_concrete_type_with_optional_none` (edgar deprecation).

## Next Steps (handoff order)
1. **Remove remaining legacy runtime paths** (done)
   - [x] Feature engineer legacy selection in `ml/features/facade.py` + `ml/features/__init__.py`.
   - [x] DataStore `_legacy_impl` hooks in `ml/stores/data_store_facade.py`.
   - [x] Orchestrator `_legacy_orchestrator` delegation in `ml/orchestration/ingestion_coordinator.py` and field in `ml/orchestration/pipeline_orchestrator.py`.
   - [x] `as_legacy_cython=True` toggles in data collection modules.
   - [x] Remove backup artifact `ml/data/scheduler.py.bak`.
2. **Prune unused legacy flags and tests** (done)
   - [x] Remove no-op flags from `ml/config/feature_flags.py` and update tests/docs.
3. **Facade-only module naming cleanup** (done)
   - [x] Add `ml/registry/model_registry.py` shim for non-facade imports.
   - [x] Make `ml/training/base.py` a shim to `ml/training/base_facade.py` and update imports.
   - [x] Convert `ml/features/engineering.py` into a shim for `ml/features/facade.py` and update imports to use shared config/indicator modules.
4. **Pytest warning cleanup (optional but recommended)**
   - [x] Remove remaining runtime imports of `ml.tests.fixtures.database_fixtures`
     (see `ml/docs/test_restoration_plan.md`) to eliminate the assert-rewrite warning.
5. **Stabilize and gate**
   - [ ] Re-run mypy/ruff/pytest/coverage gates to reach ML ≥ 90%.

## Scope
### In Scope
- Remove `*_legacy.py` modules and any `USE_LEGACY` branches.
- Collapse runtime switches so a single component-facade implementation is used.
- Update imports, registries, and factory bindings to new path.
- Replace legacy-only tests with component-facade equivalents.
- Update documentation and fixtures to reflect the single path.
- Bring ML coverage to >= 90% without excluding core functionality.

### Out of Scope (for this effort)
- New ML features or model changes not needed for parity.
- Major redesign of external interfaces or APIs.

## Inventory (Detected Legacy Surface)

### Feature Flags and Selectors
| Flag | Legacy implementation | Facade implementation | Selector/notes |
| --- | --- | --- | --- |
| `ML_USE_LEGACY_FEATURE_ENGINEER` | Removed (legacy module deleted) | `ml/features/facade.py` | Removed; facade-only path (legacy selector deleted). |
| `ML_USE_LEGACY_DATA_STORE` | Removed (no legacy DataStore module) | `ml/stores/data_store_facade.py` | Removed; `_legacy_impl` hooks deleted. |
| `ML_USE_LEGACY_TRAINER` | Removed (no legacy trainer module) | `ml/training/base_facade.py` | Removed; `use_legacy_trainer()` helper deleted. |
| `ML_USE_LEGACY_SCHEDULER` | Removed (legacy scheduler deleted) | `ml/data/scheduler.py` | Removed; `use_legacy_scheduler()` helper deleted. |
| `ML_USE_LEGACY_DASHBOARD_APP` | Removed (legacy app deleted) | `ml/dashboard/app.py` | Removed; `use_legacy_dashboard_app()` helper deleted. |
| `ML_USE_LEGACY_TFT_BUILDER` | Removed (legacy builder deleted) | `ml/data/tft_dataset_builder.py` | Removed; `use_legacy_builder()` helper deleted. |
| `ML_USE_LEGACY_ML_SIGNAL_ACTOR` | Removed (legacy actor deleted) | `ml/actors/signal_facade_impl.py` | Removed; helper deleted. |
| `ML_USE_LEGACY_STRATEGY_BASE` | Removed (legacy base deleted) | `ml/strategies/base_facade.py` | Removed; `_use_legacy_strategy_base()` deleted. |
| `ML_USE_LEGACY_PIPELINE_ORCHESTRATOR` | Removed (legacy orchestrator deleted) | `ml/orchestration/pipeline_orchestrator.py` | Removed; facade-only path enforced. |
| `ML_USE_LEGACY_INTEGRATION_MANAGER` | Removed (legacy integration manager deleted) | `ml/core/integration_facade.py` | Selector removed. |
| `ML_USE_LEGACY_DATA_REGISTRY` | Removed (legacy registry deleted) | `ml/registry/data_registry.py` | Flag removed; factory always returns canonical registry. |
| `ML_USE_LEGACY_MODEL_REGISTRY` | Removed (legacy registry deleted) | `ml/registry/model_registry_facade.py` | Flag removed; `ml/registry/model_registry.py` shim added. |

### Facade/Non-Facade Pair Inventory (Needs Consolidation)
**Compatibility entrypoints (current behavior)**
Importer counts are unique importing modules (excludes `__init__.py`).
- `ml/core/integration.py` re-exports `ml/core/integration_facade.py` (72 / 3).
- `ml/stores/data_store.py` re-exports `ml/stores/data_store_facade.py` (11 / 42).
- `ml/stores/feature_store.py` re-exports `ml/stores/feature_store_facade.py` (19 / 43).
- `ml/actors/signal.py` is a compatibility layer delegating to `ml/actors/signal_facade_impl.py` (59 / 2).
- `ml/actors/signal_facade.py` re-exports `ml/actors/signal_facade_impl.py` (1 / 2).
- `ml/data/scheduler_facade.py` wraps `ml/data/scheduler.py` (1 / 20).
- `ml/data/tft_dataset_builder_facade.py` re-exports `ml/data/tft_dataset_builder.py` (3 / 14).
- `ml/dashboard/app_facade.py` re-exports `ml/dashboard/app.py` (2 / 5).
- `ml/dashboard/service_facade.py` re-exports `ml/dashboard/service.py` (1 / 19).
- `ml/orchestration/pipeline_orchestrator.py` re-exports `ml/orchestration/pipeline_orchestrator_facade.py` (16 / 5).
- `ml/registry/model_registry.py` re-exports `ml/registry/model_registry_facade.py` (3 / 5).
- `ml/strategies/base.py` re-exports `ml/strategies/base_facade.py` (5 / 3).

**Parallel facades still imported directly**
- `ml/registry/data_registry_facade.py` (2 importers) alongside canonical `ml/registry/data_registry.py` (44).
- `ml/training/base_facade.py` (3 importers) alongside canonical `ml/training/base.py` (5).
- `ml/actors/signal_facade_impl.py` is imported directly by 2 modules; intended as implementation detail.

**Facade-only module (no non-facade counterpart)**
- None (feature engineering now has a non-facade shim).

### Legacy Modules Not On Any Flagged Path
None (legacy shims removed).

### Legacy-Named Files Still Present (from scan)
None (legacy-named artifacts removed).

### Flag Name Aliases Still Present
None in code/tests; only historical references remain in documentation/report files.

### Remaining Removal Targets
- Root/common duplicate modules and duplicate tests (see "Duplicate Module Audit" below).
- [x] Pytest warning cleanup (assert-rewrite/import-order).
- [ ] Stabilize lint/test/coverage gates for the legacy removal scope (deferred).

### Duplicate Module Audit (Root vs common)
Explicit root/common pairs to audit, migrate, and delete or shim (exclude `__init__.py`).

**Stores**
- [x] `ml/stores/data_writer.py` vs `ml/stores/common/data_writer.py` (canonical: common component; root removed)
- [x] `ml/stores/data_reader.py` vs `ml/stores/common/data_reader.py` (canonical: common component; root removed)
- [x] `ml/stores/schema_validator.py` vs `ml/stores/common/schema_validator.py` (canonical: common component; root removed)
- [x] `ml/stores/contract_enforcer.py` vs `ml/stores/common/contract_enforcer.py` (canonical: common component; root removed)
- [x] `ml/stores/protocols.py` vs `ml/stores/common/protocols.py` (audited; distinct layers, keep both)

**Registry**
- [x] `ml/registry/manifest_manager.py` vs `ml/registry/common/manifest_manager.py` (canonical: common component; root removed)
- [x] `ml/registry/model_persistence.py` vs `ml/registry/common/model_persistence.py` (canonical: common component; root removed)
- [x] `ml/registry/watermark_manager.py` vs `ml/registry/common/watermark_manager.py` (canonical: common component; root removed)

**Orchestration**
- [x] `ml/orchestration/protocols.py` vs `ml/orchestration/common/protocols.py` (audited; root shim retained for public import path)

**Tests (duplicate suites)**
- [x] `ml/tests/unit/stores/test_data_writer.py` vs `ml/tests/unit/stores/common/test_data_writer.py` (root test removed)
- [x] `ml/tests/unit/stores/test_data_reader.py` vs `ml/tests/unit/stores/common/test_data_reader.py` (root test removed)
- [x] `ml/tests/unit/stores/test_schema_validator.py` vs `ml/tests/unit/stores/common/test_schema_validator.py` (root test removed)
- [x] `ml/tests/unit/stores/test_contract_enforcer.py` vs `ml/tests/unit/stores/common/test_contract_enforcer.py` (root test removed)
- [x] `ml/tests/unit/orchestration/test_config_resolver.py` vs `ml/tests/unit/orchestration/common/test_config_resolver.py` (common test removed)
- [x] `ml/tests/unit/registry/test_deployment_manager.py` vs `ml/tests/unit/registry/common/test_deployment_manager.py` (audited; integration vs component coverage, keep both)
- [x] `ml/tests/unit/registry/test_model_persistence.py` vs `ml/tests/unit/registry/common/test_model_persistence.py` (root test removed)

### Inventory Sweep Results (Code + Docs)
- **Flags present in code**: none (legacy selectors removed).
- **Legacy-named files still present**: none.
- **Non-env legacy toggles**: removed (`_legacy_orchestrator`, `_legacy_builder`, `as_legacy_cython`).
- **Legacy tests**: replaced with facade-only contract checks or removed.
- **Deprecation scope**: only historical doc/report references remain.

## Decision Register: Legacy Env Vars

### Global Decisions (apply to every flag)
- Canonical naming scheme (keep current names vs rename to consistent pattern).
- Alias policy: add compatibility aliases vs update tests/docs to match one name.
- Default behavior until removal (legacy default vs facade default where applicable).
- Legacy flags remain (some unused/no-op); decide removal schedule and truthy parsing while they exist.
- Deprecation schedule and warning/telemetry when legacy flags are used.
- Post-removal behavior: delete flags entirely vs keep as no-op with warnings.

### Variable-by-Variable Decisions
| Variable | Current behavior | Decisions required |
| --- | --- | --- |
| `ML_USE_LEGACY_FEATURE_ENGINEER` | Active selector in `ml/features/__init__.py` and `FeatureEngineerFacade`. | Remove legacy engineer and flag; update parity tests. |
| `ML_USE_LEGACY_DATA_STORE` | Flag defined but unused; no selector. | Remove flag + tests/docs. |
| `ML_USE_LEGACY_ML_PIPELINE_ORCHESTRATOR` | No-op flag; `use_legacy_ml_pipeline_orchestrator()` returns False. | Remove flag + tests/docs; decide on `_legacy_orchestrator` delegation removal. |
| `ML_USE_LEGACY_ORCHESTRATOR` | Alias env var parsed but no-op. | Remove alias + tests/docs. |
| `ML_USE_LEGACY_ML_SIGNAL_ACTOR` | No-op flag; `use_legacy_ml_signal_actor()` returns False. | Remove flag + tests/docs once external dependencies are updated. |
| `ML_USE_LEGACY_STRATEGY_BASE` | No-op flag; `use_legacy_strategy_base()` returns False. | Remove flag + tests/docs once external dependencies are updated. |
| `ML_USE_LEGACY_TRAINER` | Flag helper reads env but no selection path. | Remove flag + tests/docs. |
| `ML_USE_LEGACY_SCHEDULER` | Config flag unused; scheduler helper always False. | Remove flag + tests/docs. |
| `ML_USE_LEGACY_DASHBOARD_APP` | Config flag unused; app helper always False. | Remove flag + tests/docs. |
| `ML_USE_LEGACY_TFT_BUILDER` | Config flag default True but unused; builder helper always False. | Remove flag + tests/docs; confirm no external references. |
| `ML_USE_LEGACY_DATA_REGISTRY` | Removed from config. | None (legacy name only). |
| `ML_USE_LEGACY_MODEL_REGISTRY` | Removed from config. | None (legacy name only). |

### Alias Env Vars Still Parsed (no-op)
- `ML_USE_LEGACY_ORCHESTRATOR` (alias for pipeline orchestrator; no-op).
- `ML_USE_LEGACY_PIPELINE_ORCHESTRATOR` (alias; no-op).
- `ML_USE_LEGACY_ML_PIPELINE_ORCHESTRATOR` (alias; no-op).

### Deprecated Flag Names (not in config)
- `ML_USE_LEGACY_DATA_REGISTRY`
- `ML_USE_LEGACY_MODEL_REGISTRY`
- `ML_USE_LEGACY_DASHBOARD_SERVICE` (still referenced in tests/docs).
- `ML_USE_LEGACY_TFT_DATASET_BUILDER` (still referenced in tests).
- `ML_USE_LEGACY_DATA_SCHEDULER` (still referenced in tests).

## Phase 1 Task List (Inventory + Consistency)
- [x] Confirm legacy switches removed or shimmed (facade-only).
- [x] Remove legacy flag semantics; update tests/docs to match.
- [x] Inventory facade/non-facade duplicates and define canonical module names.
- [x] Audit duplicate/unused modules created during component extraction (e.g., root vs `common/` modules), remove or shim non-canonical copies, and update tests/docs.
- [x] Remove or archive orphaned legacy modules not reachable from any selector (none found).
- [x] Produce a dependency graph of legacy entrypoints → facade components.
- [x] Record parity surface per domain (actors, stores, schedulers, registries, training, dashboard).

## Next Execution Checklist (Immediate)
1. **Facade consolidation execution (single canonical modules)**
   - [x] Confirm canonical module names (prefer non-facade import paths).
   - [x] Collapse dual-implementation pairs into single implementations (see consolidation plan below).
   - [x] Convert non-canonical modules to thin shims and remove feature-flag branches.
   - [x] Confirm non-facade shim coverage (`model_registry.py` exists; decide whether to add a shim for `base_facade.py` or keep direct imports).
   - [x] Update internal imports/docs to canonical paths where shims are in place.
2. **Inventory sweep + mapping refresh**
   - [x] `rg "USE_LEGACY" ml` + `rg "_legacy" ml` to confirm remaining switches and modules (only docs/tests remain).
   - [x] Update the inventory table and dependency graph to match current call sites.
   - [x] Cross-check dataset/schema references against `ml/schema.py` for identifier template parity.
   - [x] Enumerate duplicate modules from component extraction (root vs `common/` or renamed modules), confirm canonical imports, then remove or convert unused modules to shims with tests/docs updated (only `stores/protocols.py` + `orchestration/protocols.py` remain; both intentionally kept).
3. **Legacy flag cleanup verification**
   - [x] Remove runtime legacy toggles and update docs/tests (remaining references only in plan/docs).
   - [x] Update `ML_DEPLOYMENT_README.md` with facade-only behavior if needed (no changes required).
4. **Parity contract definition**
   - [x] Define parity contracts per domain (actors/stores/scheduler/registry/training/dashboard).
   - [x] Specify schema/topic invariants and event semantics (Stage/Source/EventStatus, message topic builder).
5. **Parity test suite build-out**
   - [ ] Deferred: convert legacy-vs-facade parity tests into contract/invariant tests for the canonical path.
   - [ ] Deferred: add property/contract/metamorphic tests using shared fixtures (`ml/tests/fixtures/**`) only when they validate real behavior.
   - [ ] Deferred: target contract boundaries (schemas, event payloads, bus topics, identifier templates).
   - [ ] Deferred: document expected tolerances + invariants in test docstrings.
6. **Shadow/dual-run validation**
   - [x] Implement mismatch metrics (`ml_parity_mismatch_total`, max diff stats) and parity helpers.
   - [ ] Deferred: store diffs under `ml/tests/validation_reports/` and record thresholds (run shadow/dual-run).
7. **Removal order (low-risk → high-risk)**
 - [x] Dashboard app/service switches
 - [x] Trainer switch
 - [x] Model registry switch
 - [x] Scheduler switch
 - [x] Integration manager switch
 - [x] Pipeline orchestrator switch
 - [x] Signal actor switch
8. **Stabilize and gate**
   - [x] Re-run `mypy`, `ruff`, focused `pytest -k <area>`, and coverage (current ML coverage: 9.82%).
   - [ ] Defer coverage gates (ML ≥ 90%, general ≥ 80%) until plan completion.

## Coverage + Shadow Validation Plan (Draft)
- Coverage sweep (post-removal): run `pytest -q ml` with coverage, capture top 15 lowest-coverage modules, and add contract/property tests targeting stores, training components, and event payloads.
- Prioritize high-impact zero-coverage areas: `ml/actors/**`, `ml/training/datasets/**`, `ml/tasks/**`, and `ml/observability/**` with deterministic fixtures after removal scope is final.
- Add component-level tests for training common modules (`hyperparameter`, `persistence`, `evaluation`) and store shims to raise baseline coverage once the removal plan completes.
- Shadow/dual-run: implement mismatch counters (`ml_parity_mismatch_total`, max diff stats) in integration manager or pipeline orchestrator, store diffs under `ml/tests/validation_reports/`, and document tolerances per feature family.
- Gate: require CI to run `poetry run coverage report --include='ml/*'` + dual-run parity test marker before removal completion.

## Parity Test Strategy (Post-Legacy)
- Legacy-vs-facade parity tests are transitional; once legacy paths are removed, retire or convert them.
- Keep parity testing only where it validates real behavior (feature parity, schema/topic invariants, event semantics).
- Avoid tests that assert internal delegation details; prefer contracts against observable outputs and published schemas.
- Shadow/dual-run parity checks are for runtime safety, not for preserving implementation-specific equivalence.

## Parity Contracts (Current Definition)
### Global invariants
- Use `Stage`, `Source`, and `EventStatus` enums for events; never raw strings.
- Build topics with `ml.common.message_topics.build_topic_for_stage` and `MessageBusConfig.from_env()`.
- Identifier templates must include `{instrument_id}` and use `ml/schema.py` defaults; overrides must pass `validate_identifier_template`/`validate_schema_identifier_templates`.
- Required record fields: `instrument_id`, `ts_event`, `ts_init` (nanoseconds) on persisted records.
- Hot-path rule: no DataFrames, file I/O, network calls, or training.

### Actors (inference + signal)
- Inherit `BaseMLInferenceActor` and call `super().__init__(config)` first.
- Use integration manager-provided stores/registries; do not instantiate stores directly.
- Signal emission remains non-blocking; wrap bus publish in `try/except` and log with `exc_info=True`.
- Feature parity checks and confidence metrics use `ml.common.metrics_bootstrap`.

### Stores (data/feature/model/strategy)
- Preserve 4-store responsibilities and 4-registry alignment; no cross-store writes.
- Schema/contract validation uses registry dataclasses and enforcement helpers; watermark progression must be monotonic.
- Progressive fallback chain (PRIMARY → CACHED → FILE → DUMMY) with `ml_fallback_activations_total` metrics.

### Scheduler / ingestion
- Dataset identifiers and templates resolve through `ml/schema.py` (schema + dataset-type overrides validated).
- Orchestrator/collector integration must preserve dataset types, stages, and topic semantics.

### Registry (data/model/strategy/feature)
- DataRegistry maintains manifests, lineage, and watermarks; event emissions use `Stage`/`Source`/`EventStatus`.
- ModelRegistry maintains deployment/version compatibility and uses ONNX artifacts (no pickles).

### Training
- Training uses `ml/_imports.py` for optional deps and exports ONNX artifacts.
- FeatureStore access stays consistent with training/inference parity requirements.

### Dashboard / services
- Dashboard services obtain stores via `get_integration_manager()` and keep API response schemas stable.

## Facade Consolidation Plan (Single Canonical Modules)
Canonical module names should drop the `_facade` suffix; the component-based implementation remains
the source of truth. Convert the non-canonical module to a thin re-export, then remove the shim
once all imports/tests/docs are migrated.

| Canonical module (target) | Current implementation to keep | Action notes |
| --- | --- | --- |
| `ml/actors/signal.py` | `ml/actors/signal_facade_impl.py` | Completed: `signal.py` re-exports facade impl; `signal_facade.py` remains a shim. |
| `ml/dashboard/app.py` | `ml/dashboard/app_facade.py` | Completed: `app.py` canonical; `app_facade.py` shim in place. |
| `ml/dashboard/service.py` | `ml/dashboard/service_facade.py` | Completed: `service.py` canonical; `service_facade.py` shim in place. |
| `ml/data/scheduler.py` | `ml/data/scheduler_facade.py` | Completed: canonical scheduler merged; facade shim in place; `ml/data/scheduler_legacy.py` removed. |
| `ml/data/tft_dataset_builder.py` | `ml/data/tft_dataset_builder_facade.py` | Completed: canonical builder merged; facade shim in place; `ml/data/tft_dataset_builder_legacy.py` removed. |
| `ml/orchestration/pipeline_orchestrator.py` | `ml/orchestration/pipeline_orchestrator_facade.py` | Completed: `pipeline_orchestrator.py` canonical; facade shim in place. |
| `ml/registry/data_registry.py` | `ml/registry/data_registry_facade.py` | Completed: `data_registry.py` canonical; facade shim in place. |
| `ml/strategies/base.py` | `ml/strategies/base_facade.py` | Completed: `base.py` shim to `base_facade.py`; legacy flag removed. |
| `ml/training/base.py` | `ml/training/base_facade.py` | Completed: legacy base removed; facade-only path remains. |
| `ml/registry/model_registry.py` (new) | `ml/registry/model_registry_facade.py` | Add shim or rename module to drop suffix. |

## Workstreams
### 1) Inventory and Mapping
- Identify every legacy module, flag, and call site.
- Map each legacy capability to the component-facade equivalent.
- Produce a dependency graph for removal sequencing.

Exit Criteria:
- No unknown or un-mapped legacy entrypoints remain.

### 2) Parity Definition and Testing
- Define parity contracts (inputs/outputs, tolerances, ordering, event semantics).
- Add property/metamorphic tests for invariants (ordering, idempotency, bounds).
- Add contract tests at boundaries (schemas, event payloads, bus topics).

Exit Criteria:
- Parity suite passes with explicit tolerances and deterministic fixtures.

### 3) Dual-Run Validation (Shadow Mode)
- Run legacy and component-facade on identical inputs (record/replay).
- Emit mismatch metrics (`ml_parity_mismatch_total`, max diff stats).
- Store diffs under `ml/tests/validation_reports/` and document thresholds.

Exit Criteria:
- Zero or explicitly accepted mismatches in shadow runs for key flows.

### 4) Legacy Removal and Refactor
- Delete legacy modules and remove toggles.
- Consolidate factories/mappings and update imports.
- Remove legacy-only configuration and docs.

Exit Criteria:
- Build, lint, and targeted tests pass with legacy removed.

### 5) Performance and Hot-Path Guardrails
- Re-run microbenchmarks if hot paths were touched.
- Verify zero-allocation guardrails where required.

Exit Criteria:
- P99 hot-path constraints remain within limits.

### 6) Coverage and Regression Validation
- Fix remaining legacy-removal test failures.
- Run `pytest ml` without coverage for fast feedback, then with coverage.
- Achieve ML coverage >= 90%.

Exit Criteria:
- Full test suite green and coverage gates met.

## Milestones
1. Inventory complete + parity contract doc committed.
2. Shadow-mode parity suite implemented and stable.
3. Legacy removal spike merged into a feature branch (all tests green).
4. Coverage gate met and hot-path benchmarks verified.
5. Legacy fully removed; docs and fixtures updated.

## Risks and Mitigations
- Hidden behavior in legacy paths: mitigate with shadow runs and property tests.
- Hot-path regressions: isolate and test perf suites; pre-allocate buffers.
- Incomplete schema/topic compatibility: enforce contract tests.
- Coverage drag from deprecated modules: remove modules rather than exclude.

## Deliverables
- Removal patch set with clear commit boundaries.
- Parity/contract test suite + validation reports.
- Updated docs: `REMEDIATION_PLAN.md`, `ML_DEPLOYMENT_README.md`, fixture guides.

## Open Questions
- Confirm whether any legacy features have no component-facade equivalent.
- Decide on shadow-run scope (which flows and datasets are mandatory).
- Define tolerances for feature parity (absolute/relative, per feature family).
