# ML Coverage Uplift Plan

## Scope
- Included domains: actors, stores, data, registry, training, features, observability, orchestration, strategies, evaluation, monitoring, deployment, cli, tasks, pipelines, config, core, dashboard, models, preprocessing, `_imports.py`.
- Excluded domains: tools, scripts, experiments.

## How To Use This Plan
- Read the latest checkpoint entry at the bottom and pick up the next unchecked item.
- Use the batch template to define a small, focused test batch.
- Update the checkpoint after each batch with what changed and what to do next.

## Guardrails (Must Follow)
- Use shared fixtures via pytest injection only; do not import fixtures directly.
- Run `make validate-fixtures`, `poetry run mypy ml --strict`, and `poetry ruff check ml` for every batch.
- Prefer property, contract/schema, metamorphic, and pairwise tests over brittle example tests.
- No hot-path IO or heavy allocations; publish/metrics off hot paths.
- Use `mock_onnx_runtime`, `patch_dataset_bars`, `fresh_store_bundle`, and `isolated_prometheus_registry` where relevant.

## Coverage Commands (xdist-safe)
- `poetry run coverage erase`
- `poetry run pytest -n auto ml --cov=ml --cov-report=term-missing`
- `poetry run coverage report --include "ml/*"`
- `poetry run coverage report --sort=cover --include "ml/*"`

## Batch Template
- [ ] Batch ID: 
- [ ] Date: 
- [ ] Domain: 
- [ ] Target modules: 
- [ ] Test types planned: 
- [ ] Fixtures needed: 
- [ ] Commands run: 
- [ ] Result summary: 
- [ ] Next item to pick up: 

## Phase 1: Core Runtime Hot Paths
Domains: `ml/actors`, `ml/stores`, `ml/data`, `ml/registry`, `ml/orchestration`, `ml/strategies`, `ml/training/event_driven`.
Focus: invariants and contracts (store ordering, watermark monotonicity, actor routing, non-blocking bus publish), minimal integration tests for critical paths.
Primary test types: property tests, contract/schema tests, targeted unit tests for error/fallback paths.

Phase 1 checklist
- [ ] Phase 1 coverage snapshot recorded.
- [ ] Stores workstream completed.
- [ ] Actors workstream completed.
- [ ] Data workstream completed.
- [ ] Registry workstream completed.
- [ ] Orchestration workstream completed.
- [ ] Strategies workstream completed.
- [ ] Training event-driven workstream completed.

Phase 1.1 Stores workstream
- [x] Add unit coverage for `feature_versioning`, `feature_table_manager`, `schedule_partitions`, `_strict_conformance_check`.
- [x] Add unit coverage for `feature_persistence` and `feature_retrieval`.
- [x] Add unit coverage for `feature_dataset_store` and `infrastructure` preflight helpers.
- [x] Add unit coverage for `providers.py` and `data_frame_converters.py` edge cases.
- [x] Add unit coverage for `strategy_store.py` low-coverage paths.
- [x] Add unit coverage for `feature_dataset_store` L2 path and conflict updates (if still low).
- [x] Re-run store-only coverage and record delta.

Phase 1.2 Actors workstream
- [x] Identify lowest-coverage actor modules (sorted report, `--include "ml/actors/*"`).
- [ ] Add property/contract tests for hot-path invariants.
- [x] Add targeted unit tests for fallback/error paths.
- [x] Re-run actor-only coverage and record delta.
- [ ] TODO: Return to lift actors coverage to >=90% once other Phase 1 domains progress.

Phase 1.2 task checklist
- [x] Add unit tests for `MultiInstrumentSignalActor` batch/universe/infer paths.
- [x] Add unit tests for `StoreOperationsComponent` fallback, health, stop, and flush paths.
- [x] Add unit tests for `EnhancedMLInferenceActor` buffer + prediction stubs.
- [x] Add tests for `RegistryComponent` fallback + cache/metadata paths.
- [x] Add targeted tests for `BaseMLInferenceActor` hot-path guards and persistence fallbacks.
- [x] Add tests for `ml/actors/adapters.py` and `ml/actors/common/chronos_inference.py` edge cases.
- [x] Add tests for `ml/actors/signal_facade_impl.py` `_try_generate_signal` no-strategy path + min-signal-separation early return.
- [x] Add tests for `ml/actors/signal_facade_impl.py` publish disabled + missing decision_metadata error handling.
- [x] Add tests for `ml/actors/signal_facade_impl.py` `_publish_signal` no-bridge path + publish failure handling.
- [x] Add tests for `ml/actors/signal_facade_impl.py` `_generate_prediction_protected` FORCE_SIGNAL_MODE behavior.

Phase 1.3 Data workstream
- [x] Identify lowest-coverage data modules (sorted report, `--include "ml/data/*"`).
- [x] Add dataset ingestion/validation tests using `patch_dataset_bars`.
- [x] Add contract tests for schema boundaries.
- [x] Re-run data-only coverage and record delta.

Phase 1.4 Registry workstream
- [ ] Identify lowest-coverage registry modules (sorted report, `--include "ml/registry/*"`).
- [ ] Add tests for registry events, contract enforcement, fallback behaviors.
- [ ] Re-run registry-only coverage and record delta.

Phase 1.5 Orchestration workstream
- [ ] Identify lowest-coverage orchestration modules (sorted report, `--include "ml/orchestration/*"`).
- [ ] Add tests for orchestrator wiring and scheduler guards.
- [ ] Re-run orchestration-only coverage and record delta.

Phase 1.6 Strategies workstream
- [ ] Identify lowest-coverage strategy modules (sorted report, `--include "ml/strategies/*"`).
- [ ] Add tests for signal ordering, bounds, and invariants.
- [ ] Re-run strategies-only coverage and record delta.

Phase 1.7 Training event-driven workstream
- [ ] Identify lowest-coverage event-driven training modules.
- [ ] Add contract tests for manifest/event emissions.
- [ ] Re-run event-driven training coverage and record delta.

## Phase 2: Supporting Runtime Services
Domains: `ml/features`, `ml/observability`, `ml/monitoring`, `ml/pipelines`, `ml/tasks`.
Focus: feature transform invariants, observability DTO contracts, pipeline wiring, task scheduling guards.
Primary test types: metamorphic tests for transforms, contract tests for DTOs and schemas, pairwise config tests.

Phase 2 checklist
- [ ] Phase 2 coverage snapshot recorded.
- [ ] Features workstream completed.
- [ ] Observability workstream completed.
- [ ] Monitoring workstream completed.
- [ ] Pipelines workstream completed.
- [ ] Tasks workstream completed.

## Phase 3: Interfaces and Entry Points
Domains: `ml/cli`, `ml/deployment`, `ml/dashboard`, `ml/core`, `ml/_imports.py`, `ml/config`, `ml/preprocessing`.
Focus: thin CLI adapters, config validation, dependency availability checks, and dashboard/service wiring with isolated envs.
Primary test types: unit tests with patched dependencies, config validation tests, smoke-level contract checks.

Phase 3 checklist
- [ ] Phase 3 coverage snapshot recorded.
- [ ] CLI workstream completed.
- [ ] Deployment workstream completed.
- [ ] Dashboard workstream completed.
- [ ] Core workstream completed.
- [ ] `_imports.py` workstream completed.
- [ ] Config workstream completed.
- [ ] Preprocessing workstream completed.

## Phase 4: Model/Training Algorithms (Non-Event-Driven)
Domains: `ml/training` (non-event-driven modules), `ml/models`, `ml/preprocessing` (algorithmic helpers).
Focus: model invariants, bounds, and stability; ensure config-driven defaults and dependency gates.
Primary test types: metamorphic tests, pairwise config tests, minimal integration only where required.

Phase 4 checklist
- [ ] Phase 4 coverage snapshot recorded.
- [ ] Non-event-driven training workstream completed.
- [ ] Models workstream completed.
- [ ] Preprocessing algorithm workstream completed.

## Phase 5: High-Coverage Maintenance
Domains: `ml/evaluation` and any domain already above target.
Focus: regression protection for invariants and contracts, avoid coverage regressions.

Phase 5 checklist
- [ ] Phase 5 coverage snapshot recorded.
- [ ] Regression guard tests added where coverage is already high.

## Success Criteria
- ML modules coverage >= 90%.
- Tests follow fixture and strategy guidance in `ml/tests/fixtures/FIXTURE_GUIDE.md` and `ml/tests/docs/TESTING_STRATEGY.md`.
- No new hot-path regressions or fixture violations.

## Checkpoint Log
- [ ] 2026-02-04: Phase 1.1 Stores workstream started.
- [ ] 2026-02-04: Added tests for `feature_versioning`, `feature_table_manager`, `schedule_partitions`, `_strict_conformance_check`.
- [ ] 2026-02-04: Added tests for `feature_persistence`, `feature_retrieval`.
- [ ] 2026-02-04: Added tests for `feature_dataset_store`, `infrastructure` preflight helpers.
- [ ] 2026-02-04: Added edge-case tests for `providers.py`, `data_frame_converters.py`, `strategy_store.py`.
- [ ] 2026-02-04: Added `feature_dataset_store` L2/conflict tests; store-only coverage now 64.04% (lowest: `feature_dataset_mirror.py` 0%, `feature_store_mirror_backfill.py` 0%, `data_processor.py` 15.79%, `io_raw.py` 18.64%, `services/strategy_services.py` 34.28%).
- [ ] 2026-02-04: Actors coverage run (excluded flaky `test_signal_actor_state_machine`); actors total 67.39%.
- [ ] 2026-02-04: Added actor unit tests for `multi_signal`, `store_operations`, `enhanced` (multi_signal 55.56%, store_operations 72.93%, enhanced 84.00%).
- [ ] 2026-02-04: Added targeted tests for `RegistryComponent` cache/fallback metadata + `BaseMLInferenceActor` hot path/persistence; actors total now 69.02% (base 44.35%, registry 65.77%).
- [ ] 2026-02-04: Added tests for `adapters.py` + `common/chronos_inference.py`; actors total now 69.88% (adapters 86.36%, chronos_inference 97.96%).
- [ ] 2026-02-04: Added `signal_facade_impl` hot-path tests (signal separation, publish disabled, missing decision metadata, FORCE_SIGNAL_MODE, actor-bus publish failure); actors total now 70.36% (`signal_facade_impl.py` 67.24%).
- [ ] 2026-02-04: Added `multi_signal` cold-path tests (universe management, feature-dim alignment, registry metadata universe loading); actors total now 71.86% (`multi_signal.py` 79.31%).
- [ ] 2026-02-04: Expanded `signal_facade_impl` coverage (feature metrics, history overrides, hot reload, ONNX prep/output guards, persist/record helpers); actors total now 75.07% (`signal_facade_impl.py` 84.61%).
- [ ] 2026-02-04: Remaining lowest actor modules: `base.py` 44.66%, `common/registry.py` 65.77%, `common/store_operations.py` 72.93%.
- [ ] Next up: Phase 1.2 Actors workstream, target `common/registry.py` gaps, then `base.py` helpers.
- [ ] 2026-02-04: Added registry cache/error/fallback coverage + HealthMonitor unit tests; actors total now 75.89% (`common/registry.py` 77.52%, `base.py` 44.86%).
- [ ] 2026-02-04: Remaining lowest actor modules: `base.py` 44.86%, `common/store_operations.py` 72.93%, `ml_domain_events.py` 73.77%, `common/model.py` 74.21%.
- [ ] 2026-02-05: Added model loader + actor bus error-path tests; actors total now 77.77% (`base.py` 51.98%, `ml_domain_events.py` 93.44%).
- [ ] 2026-02-05: Remaining lowest actor modules: `base.py` 51.98%, `common/store_operations.py` 72.93%, `common/model.py` 74.21%, `common/registry.py` 77.52%.
- [ ] 2026-02-05: Added signal actor factory routing tests + actor-factory dispatch; actors total now 77.79% (`signal.py` 92.05%).
- [ ] 2026-02-05: Remaining lowest actor modules: `base.py` 51.98%, `common/store_operations.py` 72.93%, `common/model.py` 74.21%, `common/registry.py` 77.52%.
- [ ] 2026-02-05: Added manifest-aware signal actor factory routing tests; actors total now 77.73% (`signal.py` 85.38%).
- [ ] 2026-02-05: Remaining lowest actor modules: `base.py` 51.98%, `common/store_operations.py` 72.93%, `common/model.py` 74.21%, `common/registry.py` 77.52%.
- [ ] 2026-02-05: Added base actor cold-path helper tests (init stores/registry + model id + hot reload check); actors total now 79.45% (`base.py` 59.82%).
- [ ] 2026-02-05: Remaining lowest actor modules: `base.py` 59.82%, `common/store_operations.py` 72.93%, `common/model.py` 74.21%, `common/registry.py` 77.52%.
- [ ] 2026-02-05: Added base actor registry/fallback/warm-up/reload coverage; actors total now 80.98% (`base.py` 66.84%).
- [ ] 2026-02-05: Remaining lowest actor modules: `base.py` 66.84%, `common/store_operations.py` 72.93%, `common/model.py` 74.21%, `common/registry.py` 77.52%.
- [ ] 2026-02-05: Added base actor cold-path helpers (decision metadata, scheduling, health status, persistence); actors total now 81.45% (`base.py` 68.87%).
- [ ] 2026-02-05: Remaining lowest actor modules: `base.py` 68.87%, `common/store_operations.py` 72.93%, `common/model.py` 74.21%, `common/registry.py` 77.52%.
- [ ] 2026-02-05: Data coverage run after ingestion/validation + schema contract tests; data total now 19.69% (prev 14.41%). Lowest data modules: `autogluon_adapter.py` 0.00%, `catalog_hygiene.py` 0.00%, `collection_coordinator.py` 0.00%, `ingest/api.py` 0.00%, `ingest/dbn_archive.py` 0.00%, `ingest/l2_efficient.py` 0.00%.
- [ ] 2026-02-05: Added ingest API + catalog hygiene unit tests; data total now 15.01% on this run (`catalog_hygiene.py` 95.65%, `ingest/api.py` 96.55%).
- [ ] 2026-02-05: Remaining lowest data modules: `autogluon_adapter.py` 0.00%, `collection_coordinator.py` 0.00%, `ingest/dbn_archive.py` 0.00%, `ingest/l2_efficient.py` 0.00%, `ingest/macro_refresh.py` 0.00%, `ingest/nautilus_adapters.py` 0.00%, `ingest/state.py` 0.00%.
- [ ] Next up: Phase 1.3 Data workstream, target `ingest/state.py`, `ingest/nautilus_adapters.py`, then `ingest/dbn_archive.py`.
