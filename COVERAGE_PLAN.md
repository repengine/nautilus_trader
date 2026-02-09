# ML Coverage Uplift Plan

## Scope
- Included domains: actors, stores, data, registry, training, features, observability, orchestration, strategies, evaluation, monitoring, deployment, cli, tasks, pipelines, config, core, dashboard, models, preprocessing, `_imports.py`.
- Excluded domains: tools, scripts, experiments.

## Operating Model
- Track progress inside each phase/workstream.
- Each workstream must start with a **module inventory** (coverage-first queue) before adding tests.
- Mark modules complete as they are lifted, and keep next-up modules in that same section.
- Do not use a global checkpoint log; execution history stays with the relevant phase.

## Guardrails (Must Follow)
- Use shared fixtures via pytest injection only; do not import fixtures directly.
- Run `make validate-fixtures`, `poetry run mypy ml --strict`, and `poetry run ruff check ml` for every batch.
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
- [ ] Next module queue:

## Phase 1: Core Runtime Hot Paths
Domains: `ml/actors`, `ml/stores`, `ml/data`, `ml/registry`, `ml/orchestration`, `ml/strategies`, `ml/training/event_driven`.
Focus: invariants and contracts (store ordering, watermark monotonicity, actor routing, non-blocking bus publish), minimal integration tests for critical paths.
Primary test types: property tests, contract/schema tests, targeted unit tests for error/fallback paths.

Phase 1 checklist
- [ ] Phase 1 coverage snapshot recorded.
- [ ] Stores workstream complete to target.
- [ ] Actors workstream complete to target.
- [ ] Data workstream complete to target.
- [ ] Registry workstream complete to target.
- [ ] Orchestration workstream complete to target.
- [ ] Strategies workstream complete to target.
- [ ] Training event-driven workstream complete to target.

### Phase 1.1 Stores workstream
Status: In progress (round 1 complete, additional low modules remain).

Module inventory and queue
- [x] `ml/stores/feature_versioning.py`
- [x] `ml/stores/feature_table_manager.py`
- [x] `ml/stores/schedule_partitions.py`
- [x] `ml/stores/_strict_conformance_check.py`
- [x] `ml/stores/feature_persistence.py`
- [x] `ml/stores/feature_retrieval.py`
- [x] `ml/stores/feature_dataset_store.py`
- [x] `ml/stores/infrastructure/*` preflight helpers
- [x] `ml/stores/providers.py`
- [x] `ml/stores/data_frame_converters.py`
- [x] `ml/stores/strategy_store.py`
- [ ] `ml/stores/feature_dataset_mirror.py` (0.00% in 2026-02-04 snapshot)
- [ ] `ml/stores/feature_store_mirror_backfill.py` (0.00% in 2026-02-04 snapshot)
- [ ] `ml/stores/data_processor.py` (15.79% in 2026-02-04 snapshot)
- [ ] `ml/stores/io_raw.py` (18.64% in 2026-02-04 snapshot)
- [ ] `ml/stores/services/strategy_services.py` (34.28% in 2026-02-04 snapshot)

Execution history
- 2026-02-04: Added store unit coverage for versioning, table manager, scheduling, conformance, persistence, retrieval, dataset store, infra preflight, providers, dataframe converters, and strategy store.
- 2026-02-04: Store-only coverage snapshot: 64.04%; queue updated with next low modules listed above.

### Phase 1.2 Actors workstream
Status: In progress (broad progress, still below target).

Module inventory and queue
- [x] `ml/actors/multi_signal.py`
- [x] `ml/actors/common/store_operations.py`
- [x] `ml/actors/enhanced.py`
- [x] `ml/actors/common/registry.py`
- [x] `ml/actors/base.py` (multiple cold-path/helper batches)
- [x] `ml/actors/adapters.py`
- [x] `ml/actors/common/chronos_inference.py`
- [x] `ml/actors/signal_facade_impl.py`
- [x] `ml/actors/signal.py` factory routing paths
- [ ] `ml/actors/base.py` remaining low paths (68.87% in latest actor snapshot)
- [ ] `ml/actors/common/store_operations.py` remaining low paths (72.93%)
- [ ] `ml/actors/common/model.py` remaining low paths (74.21%)
- [ ] `ml/actors/common/registry.py` remaining low paths (77.52%)

Execution history
- 2026-02-04 to 2026-02-05: Added targeted actor coverage for fallback/error/hot-path guards across main actor components.
- 2026-02-05: Actor workstream snapshot: 81.45% total; queue narrowed to the four modules above.

### Phase 1.3 Data workstream
Status: In progress (D8 and D9 complete).

Module inventory and queue (coverage-first)
- [x] `ml/data/catalog_hygiene.py` (95.65% in 2026-02-05 run)
- [x] `ml/data/ingest/api.py` (96.55% in 2026-02-05 run)
- [x] `ml/data/ingest/state.py` (100.00% in 2026-02-05 run)
- [x] `ml/data/ingest/nautilus_adapters.py` (96.30% in 2026-02-05 run)
- [x] `ml/data/ingest/dbn_archive.py` (98.43% in 2026-02-06 targeted run)
- [x] `ml/data/ingest/l2_efficient.py` (82.59% -> 93.36% in 2026-02-06 D9 run)
- [x] `ml/data/autogluon_adapter.py` (94.88% in 2026-02-06 targeted run)
- [x] `ml/data/collection_coordinator.py` (91.90% in 2026-02-06 targeted run)
- [x] `ml/data/ingest/macro_refresh.py` (98.76% in 2026-02-06 D5 round-2 targeted run)
- [x] `ml/data/ingest/yfinance_adapter.py` (93.98% in 2026-02-06 D5 round-2 targeted run)
- [x] `ml/data/loaders/alternative.py` (99.01% in 2026-02-06 D6 batch-1 loader run)
- [x] `ml/data/loaders/ohlcv_recent.py` (91.45% in 2026-02-06 D6 batch-1 loader run)
- [x] `ml/data/loaders/supplementary.py` (95.07% in 2026-02-06 D6 loader inventory run)
- [x] `ml/data/loaders/fred_loader.py` (93.15% in 2026-02-06 D6 round-2 loader run)
- [x] `ml/data/loaders/fama_french_loader.py` (95.65% in 2026-02-06 D6 round-2 loader run)
- [x] `ml/data/loaders/alfred_loader.py` (92.21% in 2026-02-06 D6 round-2 loader run)
- [x] `ml/data/ingest/common.py` (28.57% -> 100.00% in 2026-02-06 D7 run)
- [x] `ml/data/ingest/databento_adapter.py` (9.94% -> 95.03% in 2026-02-06 D7 run)
- [x] `ml/data/ingest/discovery.py` (72.33% -> 94.00% in 2026-02-06 D7 run)
- [x] `ml/data/ingest/resume.py` (78.45% -> 93.10% in 2026-02-06 D7 run)
- [x] `ml/data/ingest/service.py` (75.19% -> 92.69% in 2026-02-06 D7 run)
- [x] `ml/data/collector.py` (15.46% -> 92.37% in 2026-02-06 D8 completion run)
- [x] `ml/data/scheduler.py` (42.25% -> 90.47% in 2026-02-06 D8 completion run)
- [x] `ml/data/ingest/orchestrator.py` (72.75% -> 91.61% in 2026-02-06 D8 completion run)

Workstream checklist
- [x] Identify lowest-coverage data modules from sorted report (`--include "ml/data/*"`).
- [x] Add ingestion/validation tests using fixture-safe patterns.
- [x] Add contract-oriented boundary tests for targeted modules.
- [x] Lift `dbn_archive` and `l2_efficient` coverage with branch/error-path tests.
- [x] Lift `autogluon_adapter` and `collection_coordinator` coverage with targeted unit tests.
- [x] Complete D5 final lift for `macro_refresh` and `yfinance_adapter` to >=90%.
- [x] Create D6 loader module inventory with per-module baseline percentages.
- [x] Execute D6 batch-1 targeted tests for `alternative` and `ohlcv_recent`.
- [x] Complete remaining queued loader modules (`fred_loader`, `fama_french_loader`, `alfred_loader`) in D6 round-2.
- [x] Re-run full data-domain coverage snapshot and refresh queue percentages.
- [x] Complete D7 ingestion-core lift (`common`, `databento_adapter`, `discovery`, `resume`, `service`) to >=90%.
- [x] Start D8 baseline and first lift batch for `collector`/`scheduler`/`ingest.orchestrator`.
- [x] Complete D8 and bring all three modules to >=90%.
- [x] Complete D9 relift for `l2_efficient` to >=90%.

Execution history
- 2026-02-05: Data baseline after initial ingestion/validation + schema work: 19.69% (prev 14.41%).
- 2026-02-05: Added `ingest/api` + `catalog_hygiene` tests.
- 2026-02-05: Added `ingest/state` + `ingest/nautilus_adapters` tests; data snapshot reached 20.57%.
- 2026-02-06: Added focused `dbn_archive` + `l2_efficient` tests; targeted module run now shows `dbn_archive` 98.43% and `l2_efficient` 82.46%.
- 2026-02-06: Completed Batch D4 targeted tests for `autogluon_adapter` + `collection_coordinator`; targeted module run now shows `autogluon_adapter` 94.88% and `collection_coordinator` 91.90%.
- 2026-02-06: Started Batch D5 targeted tests for `macro_refresh` + `yfinance_adapter`; targeted module run now shows `macro_refresh` 85.45% and `yfinance_adapter` 89.16%.
- 2026-02-06: Completed D5 round-2 targeted tests; `macro_refresh` now 98.76% and `yfinance_adapter` now 93.98%.
- 2026-02-06: Established D6 loader inventory baseline: `fred_loader` 26.58%, `ohlcv_recent` 49.81%, `alternative` 53.96%, `fama_french_loader` 76.09%, `supplementary` 78.48%, `alfred_loader` 78.79%.
- 2026-02-06: Completed D6 batch-1 (`alternative` + `ohlcv_recent`) and reran loader shard; now `alternative` 99.01%, `ohlcv_recent` 91.45%, `supplementary` 95.07%.
- 2026-02-06: Completed D6 round-2 targeted tests for `fred_loader`, `fama_french_loader`, and `alfred_loader`; loader shard now shows `fred_loader` 93.15%, `fama_french_loader` 95.65%, `alfred_loader` 92.21%.
- 2026-02-06: Full `ml/tests/unit/data` snapshot (`--include "ml/data/*"`) now 67.15% overall for the data domain; loader suite is >=90% across all modules, and next queue reprioritized to ingestion/scheduler/collection low modules.
- 2026-02-06: D7 ingestion-core baseline (targeted shard) captured: `common` 28.57%, `databento_adapter` 9.94%, `discovery` 72.33%, `resume` 78.45%, `service` 75.19%.
- 2026-02-06: D7 completed with targeted branch-path tests: `common` 100.00%, `databento_adapter` 95.03%, `discovery` 94.00%, `resume` 93.10%, `service` 92.69%.
- 2026-02-06: D8 baseline captured: `collector` 15.46%, `scheduler` 42.25%, `ingest/orchestrator` 72.75%.
- 2026-02-06: D8 batch-1 completed for `collector` (new targeted tests); updated run now `collector` 87.42%, `scheduler` 42.25%, `ingest/orchestrator` 72.75%.
- 2026-02-06: D8 completed with expanded deterministic unit shards; targeted run now `collector` 92.37%, `scheduler` 90.47%, `ingest/orchestrator` 91.61%.
- 2026-02-06: D9 completed for `l2_efficient`; dedicated loader shard now `l2_efficient` 93.36% (from 82.59% baseline).

Next module queue
- [x] Complete Phase 1.5 orchestration batch O1 lift for `pipeline_runner`, `parquet_live_replay_harness`, and `signature`.
- [x] Complete Phase 1.5 orchestration batch O2 lift for `vintage` plus relifts for `pipeline_runner` and `parquet_live_replay_harness`.
- [x] Complete Phase 1.5 orchestration batch O3 lift for `registry_synchronizer`, `stage2_engine`, and `dataset_builder`.
- [x] Continue Phase 1.5 queue with O5 relifts (`ingestion_coordinator`, `registry_synchronizer`, `stage2_engine`) while preserving `config_loader` and `pipeline_orchestrator_facade_helpers`.
- [ ] Complete remaining O5 tail module to >=90 (`dataset_builder`).
- [x] Keep deferred Phase 1.4 tail queue tracked for relift (`common/data_persistence`, `base`, `data_registry_facade`, `common/ab_testing`) unless touched by Phase 1.5 changes.

### Phase 1.4 Registry workstream
Status: Paused (R1-R9 focused lift batches complete; tail queue deferred while Phase 1.5 starts).

Module inventory and queue (coverage-first; snapshot from `ml/tests/unit/registry`)
- [x] `ml/registry/_typing_utils.py` (0.00% -> 100.00% in 2026-02-06 R1 run)
- [x] `ml/registry/artifacts.py` (0.00% -> 100.00% in 2026-02-06 R2 run)
- [x] `ml/registry/contract_manager.py` (0.00% -> 94.74% in 2026-02-06 R1 run)
- [x] `ml/registry/event_manager.py` (0.00% -> 97.06% in 2026-02-06 R1 run)
- [x] `ml/registry/lineage_manager.py` (0.00% -> 94.01% in 2026-02-06 R2 run)
- [x] `ml/registry/mixins.py` (0.00% -> 96.61% in 2026-02-06 R1 run)
- [x] `ml/registry/model_registry.py` (0.00% -> 100.00% in 2026-02-06 R2 run)
- [x] `ml/registry/ab_testing_manager.py` (0.00% -> 94.83% in 2026-02-06 R3 run)
- [x] `ml/registry/canary_deployment_mgr.py` (0.00% -> 91.56% in 2026-02-06 R3 run)
- [x] `ml/registry/model_deployment_mgr.py` (0.00% -> 95.58% in 2026-02-06 R3 run)
- [x] `ml/registry/utils.py` (23.88% -> 98.51% in 2026-02-06 R4 run)
- [x] `ml/registry/statistics.py` (35.90% -> 100.00% in 2026-02-06 R4 run)
- [x] `ml/registry/data_registry.py` (42.94% -> 93.33% in 2026-02-07 R7 run)
- [x] `ml/registry/common/watermark_manager.py` (46.36% -> 94.04% in 2026-02-06 R4 run)
- [x] `ml/registry/common/manifest_defaults.py` (47.37% -> 100.00% in 2026-02-06 R4 run)
- [x] `ml/registry/common/lineage_tracker.py` (57.40% -> 96.45% in 2026-02-07 R5 run)
- [x] `ml/registry/strategy_registry.py` (59.22% -> 92.96% in 2026-02-07 R7 run)
- [x] `ml/registry/feature_registry.py` (65.40% -> 95.71% in 2026-02-07 R7 run)
- [x] `ml/registry/summaries.py` (65.52% -> 100.00% in 2026-02-07 R6 run)
- [x] `ml/registry/feature_operations.py` (70.97% -> 100.00% in 2026-02-07 R6 run)
- [x] `ml/registry/model_registry_facade.py` (71.71% -> 91.54% in 2026-02-07 R7 run)
- [x] `ml/registry/common/event_emission.py` (73.13% -> 98.51% in 2026-02-07 R8 run)
- [x] `ml/registry/protocols.py` (73.47% -> 100.00% in 2026-02-07 R8 run)
- [x] `ml/registry/dataclasses.py` (79.67% -> 98.89% in 2026-02-07 R8 run)
- [x] `ml/registry/persistence.py` (80.84% -> 98.60% in 2026-02-07 R9 run)
- [x] `ml/registry/tools/feature_catalog.py` (80.90% -> 98.99% in 2026-02-07 R9 run)
- [x] `ml/registry/abstract_registry.py` (81.25% -> 100.00% in 2026-02-07 R9 run)

Workstream checklist
- [x] Identify lowest-coverage registry modules (`--include "ml/registry/*"`).
- [x] Add deterministic tests for registry events, contract enforcement, and utility fallback behavior.
- [x] Re-run registry-only coverage and record deltas.
- [x] Execute next zero-coverage batch from queue.
- [x] Execute next high-ROI low-module queue batch and start `data_registry` helper-path lift.
- [x] Execute dedicated PostgreSQL-branch lift for `data_registry` and complete `common/lineage_tracker` high-ROI queue item.
- [x] Execute R6 multi-module queue lift (`strategy_registry`, `feature_registry`, `summaries`, `feature_operations`, `model_registry_facade`) and refresh full-suite aggregate.
- [x] Execute R7 final sub-90 lift batch for `data_registry`, `strategy_registry`, `model_registry_facade`, and `feature_registry`; all four now >=90%.
- [x] Execute R8 support-module lift for `common/event_emission`, `protocols`, and `dataclasses`; refresh aggregate and queue.
- [x] Execute R9 support-module lift for `persistence`, `tools/feature_catalog`, and `abstract_registry`; refresh aggregate and queue.
- [x] Record phase-transition decision and defer remaining registry tail queue while starting Phase 1.5.

Execution history
- 2026-02-06: Captured Phase 1.4 baseline with `poetry run coverage run -m pytest -q ml/tests/unit/registry` + `poetry run coverage report --sort=cover --include 'ml/registry/*'`; registry total measured 58.00%.
- 2026-02-06: Added `ml/tests/unit/registry/test_registry_support_modules.py` covering `_typing_utils`, `mixins`, `contract_manager`, and `event_manager` with deterministic unit-only paths (including mocked Postgres fallback branches for event emission).
- 2026-02-06: Re-ran the same registry shard and coverage reports; registry total lifted to 63.23%, with module deltas `_typing_utils` 0.00% -> 100.00%, `contract_manager` 0.00% -> 94.74%, `event_manager` 0.00% -> 97.06%, and `mixins` 0.00% -> 96.61%.
- 2026-02-06: Captured R2 module baseline for `artifacts`, `lineage_manager`, and `model_registry` via `poetry run coverage report --include 'ml/registry/artifacts.py,ml/registry/lineage_manager.py,ml/registry/model_registry.py'`; all three measured 0.00% before edits.
- 2026-02-06: Added `ml/tests/unit/registry/test_registry_support_modules_r2.py` with deterministic mock-only tests for artifact update flows (success/error/fallback), lineage JSON/Postgres branches, and model-registry shim aliases.
- 2026-02-06: Re-ran `poetry run coverage run -m pytest -q ml/tests/unit/registry` and refreshed reports; registry total lifted 63.23% -> 66.45%, with module deltas `artifacts` 0.00% -> 100.00%, `lineage_manager` 0.00% -> 94.01%, and `model_registry` 0.00% -> 100.00%.
- 2026-02-06: Captured R3 baseline for `ab_testing_manager`, `canary_deployment_mgr`, and `model_deployment_mgr`; each measured 0.00%, and full registry aggregate (`--include 'ml/registry/*'`) measured 66.45%.
- 2026-02-06: Extended `ml/tests/unit/registry/test_registry_support_modules_r2.py` with deterministic unit coverage for legacy A/B testing, canary rollout, and deployment lifecycle managers (including success/failure/no-op branches and callback wiring).
- 2026-02-06: Re-ran `poetry run coverage run -m pytest -q ml/tests/unit/registry` and refreshed reports; registry aggregate lifted 66.45% -> 74.84%, with module deltas `ab_testing_manager` 0.00% -> 94.83%, `canary_deployment_mgr` 0.00% -> 91.56%, and `model_deployment_mgr` 0.00% -> 95.58%.
- 2026-02-06: Captured R4 baseline from a fresh full registry-suite run (`poetry run coverage run -m pytest -q ml/tests/unit/registry` + reports); registry aggregate measured 74.84%, with target baselines `utils` 23.88%, `statistics` 35.90%, `manifest_defaults` 47.37%, `watermark_manager` 46.36%, and `data_registry` 42.94%.
- 2026-02-06: Extended `ml/tests/unit/registry/test_registry_support_modules.py` with deterministic coverage for registry utils/statistics/default primary-key resolution, watermark-manager PostgreSQL branches, and focused data-registry helper/legacy-report paths.
- 2026-02-06: Re-ran the same full registry coverage workflow; registry aggregate lifted 74.84% -> 78.25%, with module deltas `utils` 23.88% -> 98.51%, `statistics` 35.90% -> 100.00%, `manifest_defaults` 47.37% -> 100.00%, `watermark_manager` 46.36% -> 94.04%, and `data_registry` 42.94% -> 48.05%.
- 2026-02-07: Captured R5 baseline from a fresh full registry-suite run; registry aggregate measured 78.25%, with queue baselines `data_registry` 48.05%, `lineage_tracker` 57.40%, `strategy_registry` 59.22%, `feature_registry` 65.40%, and `summaries` 65.52%.
- 2026-02-07: Extended `ml/tests/unit/registry/test_data_registry_unit.py` with deterministic PostgreSQL-branch tests for `register_dataset`, `update_manifest`, `list/get_manifest`, `get_contract`, event emission fallbacks, watermark paths, and lineage paths (session-missing/error/success branches).
- 2026-02-07: Extended `ml/tests/unit/registry/common/test_lineage_tracker.py` with deterministic PostgreSQL and exception-path coverage (link/iter branches, invalid JSON decode branches, signature fallback/error branches).
- 2026-02-07: Re-ran full registry coverage workflow; registry aggregate lifted 78.25% -> 83.55%, with module deltas `data_registry` 48.05% -> 80.09% and `common/lineage_tracker` 57.40% -> 96.45%.
- 2026-02-07: Captured R6 baseline from a fresh full registry-suite run; registry aggregate measured 83.55%, with queue baselines `strategy_registry` 59.22%, `feature_registry` 65.40%, `summaries` 65.52%, `feature_operations` 70.97%, `model_registry_facade` 71.71%, and residual `data_registry` 80.09%.
- 2026-02-07: Extended `ml/tests/unit/registry/test_strategy_registry.py`, `ml/tests/unit/registry/test_feature_registry.py`, and `ml/tests/unit/registry/test_model_registry_facade.py` with deterministic branch coverage for POSTGRES/session fallback paths, helper/property branches, quality-gate edges, cache/access helpers, and auto-deploy routing logic.
- 2026-02-07: Added focused module tests `ml/tests/unit/registry/test_summaries.py` and `ml/tests/unit/registry/test_feature_operations.py`; re-ran full registry coverage workflow and lifted aggregate 83.55% -> 87.33%, with module deltas `strategy_registry` 59.22% -> 80.10%, `feature_registry` 65.40% -> 87.63%, `summaries` 65.52% -> 100.00%, `feature_operations` 70.97% -> 100.00%, and `model_registry_facade` 71.71% -> 81.46% (`data_registry` unchanged at 80.09%).
- 2026-02-07: Captured R7 baseline from a fresh full registry-suite run; registry aggregate measured 87.33%, with baselines `data_registry` 80.09%, `strategy_registry` 80.10%, `model_registry_facade` 81.46%, and `feature_registry` 87.63%.
- 2026-02-07: Extended `ml/tests/unit/registry/test_data_registry_unit.py`, `ml/tests/unit/registry/test_strategy_registry.py`, `ml/tests/unit/registry/test_model_registry_facade.py`, and `ml/tests/unit/registry/test_feature_registry.py` with deterministic branch-path coverage for session-missing guards, JSON/Postgres helper fallbacks, lineage/watermark iter filters, delegate wrappers, quality/validation helper edges, and manifest/lineage accessor paths.
- 2026-02-07: Re-ran full registry coverage workflow and lifted aggregate 87.33% -> 91.31%, with module deltas `data_registry` 80.09% -> 93.33%, `strategy_registry` 80.10% -> 92.96%, `model_registry_facade` 81.46% -> 91.54%, and `feature_registry` 87.63% -> 95.71%.
- 2026-02-07: Captured R8 baseline from a fresh full registry-suite run; registry aggregate measured 91.31%, with baselines `common/event_emission` 73.13%, `protocols` 73.47%, and `dataclasses` 79.67%.
- 2026-02-07: Extended `ml/tests/unit/registry/common/test_event_emission.py`, `ml/tests/unit/registry/test_registry_support_modules.py`, and `ml/tests/unit/registry/common/test_deployment_manager.py` with deterministic branch-path tests for PostgreSQL event-emission fallbacks, protocol stub callability, dataclass validation guards, and direct canary/rollout branch outcomes.
- 2026-02-07: Re-ran full registry coverage workflow and lifted aggregate 91.31% -> 92.79%, with module deltas `common/event_emission` 73.13% -> 98.51%, `protocols` 73.47% -> 100.00%, and `dataclasses` 79.67% -> 98.89%; queue updated to the next sorted low modules (`persistence`, `tools/feature_catalog`, `abstract_registry`, `common/data_persistence`, `base`, `data_registry_facade`, `common/ab_testing`).
- 2026-02-07: Captured R9 baseline from a fresh full registry-suite run; registry aggregate measured 92.79%, with baselines `persistence` 80.84%, `tools/feature_catalog` 80.90%, and `abstract_registry` 81.25%.
- 2026-02-07: Extended `ml/tests/unit/registry/test_registry_support_modules.py` and `ml/tests/unit/registry/test_feature_catalog.py` with deterministic branch-path tests for persistence config/init/session/audit flows, abstract-registry JSON/audit/health helpers, and feature-catalog classification/render/serialization paths.
- 2026-02-07: Re-ran full registry coverage workflow and lifted aggregate 92.79% -> 93.99%, with module deltas `persistence` 80.84% -> 98.60%, `tools/feature_catalog` 80.90% -> 98.99%, and `abstract_registry` 81.25% -> 100.00%; queue updated to the remaining sub-90 modules (`common/data_persistence`, `base`, `data_registry_facade`, `common/ab_testing`).
- 2026-02-07: Phase transition decision recorded: begin Phase 1.5 orchestration workstream next, with remaining Phase 1.4 tail modules deferred and tracked as follow-up queue.

### Phase 1.5 Orchestration workstream
Status: In progress (O5 lift completed on 2026-02-07; aggregate now 82.04%).

Module inventory and queue (coverage-first; snapshot from `ml/tests/unit/orchestration`)
- [x] Baseline captured for orchestration domain (`--include "ml/orchestration/*"`): 60.73% aggregate
- [x] `ml/orchestration/signature.py` (27.91% -> 100.00% in 2026-02-07 O1)
- [x] `ml/orchestration/pipeline_runner.py` (20.96% -> 89.34% in 2026-02-07 O1 -> 99.63% in 2026-02-07 O2)
- [x] `ml/orchestration/parquet_live_replay_harness.py` (27.17% -> 89.76% in 2026-02-07 O1 -> 93.54% in 2026-02-07 O2)
- [x] `ml/orchestration/vintage.py` (32.65% -> 100.00% in 2026-02-07 O2)
- [x] `ml/orchestration/registry_synchronizer.py` (41.56% -> 83.38% in 2026-02-07 O3 -> 96.47% in 2026-02-07 O5)
- [x] `ml/orchestration/stage2_engine.py` (43.51% -> 77.10% in 2026-02-07 O3 -> 96.95% in 2026-02-07 O5)
- [x] `ml/orchestration/dataset_builder.py` (46.71% -> 83.47% in 2026-02-07 O3)
- [x] `ml/orchestration/ingestion_coordinator.py` (50.75% -> 70.60% in 2026-02-07 O4 -> 91.74% in 2026-02-07 O5)
- [x] `ml/orchestration/config_loader.py` (60.86% -> 90.35% in 2026-02-07 O4)
- [x] `ml/orchestration/pipeline_orchestrator_facade_helpers.py` (62.39% -> 93.88% in 2026-02-07 O4)
- [ ] Remaining relift tail to >=90 (`dataset_builder` 83.47%)

Workstream checklist
- [x] Identify lowest-coverage orchestration modules (`--include "ml/orchestration/*"`).
- [x] Execute O1 targeted deterministic tests for first lift batch (`pipeline_runner`, `parquet_live_replay_harness`, `signature`).
- [x] Execute O2 targeted deterministic tests for next batch (`vintage`) and >=90 relifts (`pipeline_runner`, `parquet_live_replay_harness`).
- [x] Execute O3 targeted deterministic tests for `registry_synchronizer`, `stage2_engine`, and `dataset_builder`.
- [x] Execute O4 targeted deterministic tests for `ingestion_coordinator`, `config_loader`, and `pipeline_orchestrator_facade_helpers`.
- [x] Execute O5 deterministic relift tests for `ingestion_coordinator`, `registry_synchronizer`, and `stage2_engine`.
- [ ] Add tests for orchestrator wiring and scheduler guards.
- [x] Re-run orchestration-only coverage and record deltas.

Execution history
- 2026-02-07: Captured Phase 1.5 O1 baseline with `poetry run coverage run -m pytest -q ml/tests/unit/orchestration` + `poetry run coverage report --sort=cover --include 'ml/orchestration/*'`; orchestration aggregate measured 60.73%.
- 2026-02-07: Queue initialized from sorted baseline; first target trio set to `pipeline_runner`, `parquet_live_replay_harness`, and `signature`.
- 2026-02-07: Captured fresh O1 baseline for this session using the same full-suite command pair; baselines confirmed at `pipeline_runner` 20.96%, `parquet_live_replay_harness` 27.17%, `signature` 27.91%, orchestration aggregate 60.73%.
- 2026-02-07: Added deterministic unit coverage in `ml/tests/unit/orchestration/test_pipeline_runner.py`, `ml/tests/unit/orchestration/test_parquet_live_replay_harness_config.py`, and new shard `ml/tests/unit/orchestration/test_signature.py`.
- 2026-02-07: Re-ran full orchestration coverage workflow and lifted aggregate 60.73% -> 66.46% (+5.73), with module deltas `pipeline_runner` 20.96% -> 89.34% (+68.38), `parquet_live_replay_harness` 27.17% -> 89.76% (+62.59), and `signature` 27.91% -> 100.00% (+72.09).
- 2026-02-07: Captured fresh O2 baseline with the same full-suite command pair; baselines confirmed at `vintage` 32.65%, `pipeline_runner` 89.34%, `parquet_live_replay_harness` 89.76%, orchestration aggregate 66.46%.
- 2026-02-07: Added deterministic unit coverage in `ml/tests/unit/orchestration/test_vintage.py` and extended `ml/tests/unit/orchestration/test_pipeline_runner.py` plus `ml/tests/unit/orchestration/test_parquet_live_replay_harness_config.py`.
- 2026-02-07: Re-ran full orchestration coverage workflow and lifted aggregate 66.46% -> 67.36% (+0.90), with module deltas `vintage` 32.65% -> 100.00% (+67.35), `pipeline_runner` 89.34% -> 99.63% (+10.29), and `parquet_live_replay_harness` 89.76% -> 93.54% (+3.78).
- 2026-02-07: Captured fresh O3 baseline with `poetry run coverage erase`, `poetry run coverage run -m pytest -q ml/tests/unit/orchestration`, and report commands; baselines confirmed at `registry_synchronizer` 41.56%, `stage2_engine` 43.51%, `dataset_builder` 46.71%, orchestration aggregate 67.36%.
- 2026-02-07: Added deterministic O3 unit coverage in `ml/tests/unit/orchestration/common/test_registry_synchronizer.py`, `ml/tests/unit/orchestration/common/test_dataset_builder.py`, and new shard `ml/tests/unit/orchestration/test_stage2_engine.py`.
- 2026-02-07: Re-ran full orchestration coverage workflow and lifted aggregate 67.36% -> 72.41% (+5.05), with module deltas `registry_synchronizer` 41.56% -> 83.38% (+41.82), `stage2_engine` 43.51% -> 77.10% (+33.59), and `dataset_builder` 46.71% -> 83.47% (+36.76).
- 2026-02-07: Updated Phase 1.5 queue to remaining lowest modules (`ingestion_coordinator`, `config_loader`, `pipeline_orchestrator_facade_helpers`) plus O3 relift tail to >=90; Phase 1.4 registry tail remains deferred unless touched.
- 2026-02-07: Captured fresh O4 baseline with `poetry run coverage erase`, `poetry run coverage run -m pytest -q ml/tests/unit/orchestration`, and report commands; baselines confirmed at `ingestion_coordinator` 50.75%, `config_loader` 60.86%, `pipeline_orchestrator_facade_helpers` 62.39%, orchestration aggregate 72.41%.
- 2026-02-07: Added deterministic O4 unit coverage in `ml/tests/unit/orchestration/common/test_ingestion_coordinator.py`, `ml/tests/unit/orchestration/test_config_loader.py`, and new shard `ml/tests/unit/orchestration/test_pipeline_orchestrator_facade_helpers.py`.
- 2026-02-07: Re-ran full orchestration coverage workflow and lifted aggregate 72.41% -> 78.88% (+6.47), with module deltas `ingestion_coordinator` 50.75% -> 70.60% (+19.85), `config_loader` 60.86% -> 90.35% (+29.49), and `pipeline_orchestrator_facade_helpers` 62.39% -> 93.88% (+31.49).
- 2026-02-07: Captured fresh O5 baseline with `poetry run coverage erase`, `poetry run coverage run -m pytest -q ml/tests/unit/orchestration`, and report commands; baselines confirmed at `ingestion_coordinator` 70.60%, `registry_synchronizer` 83.38%, `stage2_engine` 77.10%, `dataset_builder` 83.47%, `config_loader` 90.35%, `pipeline_orchestrator_facade_helpers` 93.88%, orchestration aggregate 78.88%.
- 2026-02-07: Extended deterministic O5 unit coverage in `ml/tests/unit/orchestration/common/test_ingestion_coordinator.py`, `ml/tests/unit/orchestration/common/test_registry_synchronizer.py`, and `ml/tests/unit/orchestration/test_stage2_engine.py`.
- 2026-02-07: Re-ran full orchestration coverage workflow and lifted aggregate 78.88% -> 82.04% (+3.16), with module deltas `ingestion_coordinator` 70.60% -> 91.74% (+21.14), `registry_synchronizer` 83.38% -> 96.47% (+13.09), `stage2_engine` 77.10% -> 96.95% (+19.85), `dataset_builder` 83.47% -> 83.47% (+0.00), `config_loader` 90.35% -> 90.35% (+0.00), and `pipeline_orchestrator_facade_helpers` 93.88% -> 93.88% (+0.00).
- 2026-02-07: Updated Phase 1.5 queue to remaining sub-90 orchestration target `dataset_builder` plus pending wiring/scheduler guard tests.
- 2026-02-07: Phase 1.4 registry tail queue remains deferred unless touched by future Phase 1.5 work (`common/data_persistence`, `base`, `data_registry_facade`, `common/ab_testing`).
- 2026-02-07: Phase transition decision recorded for next session: proceed to Phase 1.6 strategies workstream now, and return later for deferred Phase 1.5 tail (`dataset_builder` plus orchestration wiring/scheduler guards).

### Phase 1.6 Strategies workstream
Status: Paused (S1-S5 completed; residual tail queue deferred while Phase 1.7 starts).

Module inventory and queue
- [x] Capture fresh strategies baseline (`poetry run coverage erase`, `poetry run coverage run -m pytest -q ml/tests/unit/strategies`, `poetry run coverage report --sort=cover --include 'ml/strategies/*'`): 67.34% aggregate.
- [x] Build module-first sorted queue from the baseline: `analytics` 26.22%, `base_facade` 46.39%, `ml_strategy` 59.93%, `portfolio` 63.51%, `returns_updater` 65.56%, `sizing` 66.47%, `risk` 67.09%.
- [x] Execute S1 deterministic test lifts in existing shards plus metric-stability relifts:
  `test_ml_trading_strategy_branches_unit.py`,
  `test_risk_action_decisions.py`,
  `test_portfolio_and_exposure_invariants.py`,
  `test_sizing_and_risk_invariants.py`,
  `test_decision_publisher_service.py`,
  `test_risk_exposure_notional.py`,
  `common/test_positions_provider.py`.
- [x] Re-run strategies-only coverage and record deltas.
- [x] Execute S2 deterministic branch lifts in existing shards:
  `ml/tests/unit/strategies/test_base_ml_strategy_facade.py`,
  `ml/tests/unit/strategies/common/test_returns_updater.py`,
  `ml/tests/unit/strategies/common/test_position_management_component.py`.
- [x] Re-run full strategies coverage and record S2 module deltas and aggregate delta.
- [x] Execute S3 deterministic branch lifts in existing shards:
  `ml/tests/unit/strategies/test_base_ml_strategy_facade.py`,
  `ml/tests/unit/strategies/test_ml_trading_strategy_branches_unit.py`,
  `ml/tests/unit/strategies/test_ml_trading_strategy_exit_policy.py`,
  `ml/tests/unit/strategies/test_portfolio_and_exposure_invariants.py`,
  `ml/tests/unit/strategies/common/test_model_exit_policy.py`,
  `ml/tests/unit/strategies/test_sizing_and_risk_invariants.py`,
  `ml/tests/unit/strategies/test_risk_correlation_provider.py`.
- [x] Re-run full strategies coverage and record S3 module deltas and aggregate delta.
- [x] Capture fresh S4 baseline with exact required commands:
  `poetry run coverage erase`;
  `poetry run coverage run -m pytest -q ml/tests/unit/strategies`;
  `poetry run coverage report --sort=cover --include 'ml/strategies/*'`;
  `poetry run coverage report --include 'ml/strategies/base_facade.py,ml/strategies/ml_strategy.py,ml/strategies/portfolio.py,ml/strategies/common/order_submission.py,ml/strategies/common/position_management.py,ml/strategies/risk.py,ml/strategies/common/returns_updater.py,ml/strategies/common/model_exit_policy.py'`.
- [x] Execute S4 deterministic branch lifts in existing shards:
  `ml/tests/unit/strategies/test_base_ml_strategy_facade.py`,
  `ml/tests/unit/strategies/test_ml_trading_strategy_branches_unit.py`,
  `ml/tests/unit/strategies/test_portfolio_and_exposure_invariants.py`,
  `ml/tests/unit/strategies/test_risk_correlation_provider.py`,
  `ml/tests/unit/strategies/common/test_model_exit_policy.py`,
  `ml/tests/unit/strategies/common/test_order_submission_component.py`,
  `ml/tests/unit/strategies/common/test_position_management_component.py`,
  `ml/tests/unit/strategies/common/test_returns_updater.py`.
- [x] Re-run full strategies coverage and record S4 module deltas and aggregate delta.
- [x] Capture fresh S5 baseline with exact required commands:
  `poetry run coverage erase`;
  `poetry run coverage run -m pytest -q ml/tests/unit/strategies`;
  `poetry run coverage report --sort=cover --include 'ml/strategies/*'`;
  `poetry run coverage report --include 'ml/strategies/base_facade.py,ml/strategies/ml_strategy.py,ml/strategies/common/intent_positions.py,ml/strategies/risk.py,ml/strategies/common/position_management.py,ml/strategies/common/order_submission.py,ml/strategies/common/returns_updater.py,ml/strategies/portfolio.py,ml/strategies/common/model_exit_policy.py'`.
- [x] Execute S5 deterministic branch lifts in existing shards:
  `ml/tests/unit/strategies/test_base_ml_strategy_facade.py`,
  `ml/tests/unit/strategies/test_ml_trading_strategy_branches_unit.py`,
  `ml/tests/unit/strategies/common/test_intent_position_tracker.py`.
- [x] Re-run full strategies coverage and record S5 module deltas and aggregate delta.
- [ ] S5 next-up low modules after S5: `base_facade` (61.36%), `risk` (78.78%), `common/position_management` (80.09%), `common/order_submission` (80.76%), `common/returns_updater` (83.78%), `ml_strategy` (84.53%), `portfolio` (88.39%), `common/model_exit_policy` (91.54%), `common/intent_positions` (98.57%).
- [ ] Deferred Phase 1.5 tail remains queued for later revisit: `dataset_builder` (83.47%) + orchestration wiring/scheduler guards (explicitly still deferred in S5).
- [ ] Deferred Phase 1.4 registry tail remains deferred unless touched: `common/data_persistence`, `base`, `data_registry_facade`, `common/ab_testing` (explicitly unchanged in S5).

Execution history
- 2026-02-07: Captured fresh S1 baseline using the required full strategies command pattern; strategies aggregate measured 67.34%, with sorted low modules led by `analytics` (26.22%) and `base_facade` (46.39%).
- 2026-02-07: Full baseline suite completed with 6 pre-existing failing assertions in metric-count tests; stabilized these tests via deterministic counter-delta assertions in `ml/tests/unit/strategies/test_risk_exposure_notional.py` and `ml/tests/unit/strategies/common/test_positions_provider.py` so full-suite runs are order-independent.
- 2026-02-07: Extended existing deterministic shards for first S1 lift batch:
  `ml/tests/unit/strategies/test_ml_trading_strategy_branches_unit.py`,
  `ml/tests/unit/strategies/test_risk_action_decisions.py`,
  `ml/tests/unit/strategies/test_portfolio_and_exposure_invariants.py`,
  `ml/tests/unit/strategies/test_sizing_and_risk_invariants.py`,
  and `ml/tests/unit/strategies/test_decision_publisher_service.py`.
- 2026-02-07: Re-ran full strategies coverage workflow and lifted aggregate 67.34% -> 69.48% (+2.14), with key S1 module deltas:
  `ml_strategy` 59.93% -> 61.51% (+1.58),
  `portfolio` 63.51% -> 73.22% (+9.71),
  `risk` 67.09% -> 75.48% (+8.39),
  `sizing` 66.47% -> 91.18% (+24.71),
  `services/decision_publisher` 95.35% -> 100.00% (+4.65).
- 2026-02-07: Full strategies unit suite now passes (`387 passed`), and queue updated to prioritize structural low-coverage tail modules (`analytics`, `base_facade`) in S2 while explicitly keeping deferred Phase 1.5 and Phase 1.4 tails unchanged.
- 2026-02-07: Captured fresh S2 baseline with the required full strategies command pattern; aggregate measured 69.48%, with focused targets `analytics` 26.22%, `base_facade` 46.39%, `returns_updater` 65.56%, `common/position_management` 69.59%, and `protocols` 71.11%.
- 2026-02-07: Extended deterministic existing shards (`test_base_ml_strategy_facade.py`, `common/test_returns_updater.py`, `common/test_position_management_component.py`) with branch-focused tests for analytics lifecycle/reporting paths, base-facade helper/delegation paths, protocol runtime stubs, returns-updater fallback/metric-failure paths, and position-management fallback/risk/provider paths.
- 2026-02-07: Re-ran full strategies coverage workflow and lifted aggregate 69.48% -> 74.90% (+5.42), with S2 module deltas `analytics` 26.22% -> 90.64% (+64.42), `base_facade` 46.39% -> 55.60% (+9.21), `returns_updater` 65.56% -> 78.89% (+13.33), `common/position_management` 69.59% -> 77.09% (+7.50), and `protocols` 71.11% -> 100.00% (+28.89).
- 2026-02-07: Full strategies unit suite remains green after S2 (`413 passed`); deferred Phase 1.5 tail (`dataset_builder` + wiring/scheduler guards) and deferred Phase 1.4 registry tail remain explicitly unchanged.
- 2026-02-07: Captured fresh S3 baseline with exact required commands:
  `poetry run coverage erase`;
  `poetry run coverage run -m pytest -q ml/tests/unit/strategies`;
  `poetry run coverage report --sort=cover --include 'ml/strategies/*'`;
  `poetry run coverage report --include 'ml/strategies/base_facade.py,ml/strategies/ml_strategy.py,ml/strategies/portfolio.py,ml/strategies/common/model_exit_policy.py,ml/strategies/risk.py'`.
  Baseline measured `74.90%` aggregate, with focused targets `base_facade` 55.60%, `ml_strategy` 61.51%, `portfolio` 73.22%, `common/model_exit_policy` 75.38%, and `risk` 75.48%.
- 2026-02-07: Extended existing deterministic shards with S3 branch-path tests for base-facade helper methods/store accessors/intent guards, ml-strategy horizon/exit helpers and side mapping, portfolio limits+correlation+rebalance gates, model-exit policy side/flip/min-hold edge cases, and risk exposure/correlation/provider fallback branches.
- 2026-02-07: Re-ran full strategies coverage workflow and lifted aggregate `74.90%` -> `75.70%` (`+0.80`), with S3 per-module deltas:
  `base_facade` 55.60% -> 56.56% (`+0.96`),
  `ml_strategy` 61.51% -> 63.44% (`+1.93`),
  `portfolio` 73.22% -> 75.36% (`+2.14`),
  `common/model_exit_policy` 75.38% -> 81.54% (`+6.16`),
  `risk` 75.48% -> 77.89% (`+2.41`).
- 2026-02-07: Full strategies unit suite remains green after S3 (`435 passed`); deferred Phase 1.5 tail (`dataset_builder` + orchestration wiring/scheduler guards) remains deferred, and deferred Phase 1.4 registry tail (`common/data_persistence`, `base`, `data_registry_facade`, `common/ab_testing`) remains untouched and deferred.
- 2026-02-08: Captured fresh S4 baseline with exact required commands:
  `poetry run coverage erase`;
  `poetry run coverage run -m pytest -q ml/tests/unit/strategies`;
  `poetry run coverage report --sort=cover --include 'ml/strategies/*'`;
  `poetry run coverage report --include 'ml/strategies/base_facade.py,ml/strategies/ml_strategy.py,ml/strategies/portfolio.py,ml/strategies/common/order_submission.py,ml/strategies/common/position_management.py,ml/strategies/risk.py,ml/strategies/common/returns_updater.py,ml/strategies/common/model_exit_policy.py'`.
  Baseline measured `75.70%` aggregate, with focused targets `base_facade` 56.56%, `ml_strategy` 63.44%, `portfolio` 75.36%, `common/order_submission` 76.64%, `common/position_management` 77.09%, `risk` 77.89%, `common/returns_updater` 78.89%, and `common/model_exit_policy` 81.54%.
- 2026-02-08: Extended deterministic existing shards with S4 branch-path tests for store/registry accessor and strategy-id sync error branches, ml-strategy returns updater and entry/reversal guard branches, portfolio correlation snapshot/matrix and metrics helpers, risk correlation provider error/invalid paths, model-exit side/timestamp metadata branches, returns-updater bar/annualization/max-age helper branches, order-submission helper/read-path branches, and position-management protocol/config branches.
- 2026-02-08: Re-ran full strategies coverage workflow and lifted aggregate `75.70%` -> `78.27%` (`+2.57`), with S4 per-module deltas:
  `base_facade` 56.56% -> 58.04% (`+1.48`),
  `ml_strategy` 63.44% -> 67.31% (`+3.87`),
  `portfolio` 75.36% -> 88.39% (`+13.03`),
  `common/order_submission` 76.64% -> 80.76% (`+4.12`),
  `common/position_management` 77.09% -> 80.09% (`+3.00`),
  `risk` 77.89% -> 78.78% (`+0.89`),
  `common/returns_updater` 78.89% -> 83.78% (`+4.89`),
  `common/model_exit_policy` 81.54% -> 91.54% (`+10.00`).
- 2026-02-08: Full strategies unit suite remains green after S4 (`455 passed`); required validations passed (`make validate-fixtures`, `poetry run mypy ml --strict`, `poetry run ruff check ml`). Deferred Phase 1.5 tail (`dataset_builder` + orchestration wiring/scheduler guards) remains explicitly deferred, and deferred Phase 1.4 registry tail (`common/data_persistence`, `base`, `data_registry_facade`, `common/ab_testing`) remains unchanged and deferred.
- 2026-02-08: Captured fresh S5 baseline with exact required commands:
  `poetry run coverage erase`;
  `poetry run coverage run -m pytest -q ml/tests/unit/strategies`;
  `poetry run coverage report --sort=cover --include 'ml/strategies/*'`;
  `poetry run coverage report --include 'ml/strategies/base_facade.py,ml/strategies/ml_strategy.py,ml/strategies/common/intent_positions.py,ml/strategies/risk.py,ml/strategies/common/position_management.py,ml/strategies/common/order_submission.py,ml/strategies/common/returns_updater.py,ml/strategies/portfolio.py,ml/strategies/common/model_exit_policy.py'`.
  Baseline measured `78.27%` aggregate, with focused targets `base_facade` 58.04%, `ml_strategy` 67.31%, `common/intent_positions` 75.00%, `risk` 78.78%, `common/position_management` 80.09%, `common/order_submission` 80.76%, `common/returns_updater` 83.78%, `portfolio` 88.39%, and `common/model_exit_policy` 91.54%.
- 2026-02-08: Extended deterministic existing shards with S5 branch-path tests for `SimpleMLStrategyFacade._process_ml_signal` hold/entry/reverse paths, ml-strategy hold-no-position plus entry/reversal/submission and multi-model helper branches, and intent-position guard/reduce-only/side-flip paths.
- 2026-02-08: Re-ran full strategies coverage workflow and lifted aggregate `78.27%` -> `80.63%` (`+2.36`), with S5 per-module deltas:
  `base_facade` 58.04% -> 61.36% (`+3.32`),
  `ml_strategy` 67.31% -> 84.53% (`+17.22`),
  `common/intent_positions` 75.00% -> 98.57% (`+23.57`),
  `risk` 78.78% -> 78.78% (`+0.00`),
  `common/position_management` 80.09% -> 80.09% (`+0.00`),
  `common/order_submission` 80.76% -> 80.76% (`+0.00`),
  `common/returns_updater` 83.78% -> 83.78% (`+0.00`),
  `portfolio` 88.39% -> 88.39% (`+0.00`),
  `common/model_exit_policy` 91.54% -> 91.54% (`+0.00`).
- 2026-02-08: Full strategies unit suite remains green after S5 (`471 passed`); required validations passed (`make validate-fixtures`, `poetry run mypy ml --strict`, `poetry run ruff check ml`). Deferred Phase 1.5 tail (`dataset_builder` + orchestration wiring/scheduler guards) remains explicitly deferred, and deferred Phase 1.4 registry tail (`common/data_persistence`, `base`, `data_registry_facade`, `common/ab_testing`) remains unchanged and deferred.
- 2026-02-08: Phase transition decision recorded for next session: proceed to Phase 1.7 training event-driven workstream now, and return later for deferred Phase 1.6 tail queue (`base_facade`, `risk`, `common/position_management`, `common/order_submission`, `common/returns_updater`, `ml_strategy`, `portfolio`).

### Phase 1.7 Training event-driven workstream
Status: Paused (T1-T2 completed on 2026-02-08; residual tail deferred while Phase 2 starts).

Module inventory and queue
- [x] Capture fresh Phase 1.7 baseline with required commands:
  `git status --porcelain`;
  `poetry run coverage erase`;
  `poetry run coverage run -m pytest -q ml/tests/unit/training/event_driven`;
  `poetry run coverage report --sort=cover --include 'ml/training/event_driven/*,ml/training/event_driven/guardrails/*'`.
- [x] Build focused queue report from baseline:
  `poetry run coverage report --include 'ml/training/event_driven/guardrails/validation_bundle.py,ml/training/event_driven/azure_events.py,ml/training/event_driven/teacher_export.py,ml/training/event_driven/sweep.py,ml/training/event_driven/worker.py,ml/training/event_driven/wave_planner.py,ml/training/event_driven/global_run.py,ml/training/event_driven/orchestrator.py'`.
- [x] Execute deterministic module-first lifts in existing shards:
  `ml/tests/unit/training/event_driven/test_validation_bundle.py`,
  `ml/tests/unit/training/event_driven/test_azure_events.py`,
  `ml/tests/unit/training/event_driven/test_teacher_export_validation.py`,
  `ml/tests/unit/training/event_driven/test_sweep.py`.
- [x] Re-run full event-driven coverage workflow and record exact per-module and aggregate deltas.
- [x] Execute deterministic module-first lifts in existing shards:
  `ml/tests/unit/training/event_driven/test_wave_planner.py`,
  `ml/tests/unit/training/event_driven/test_global_run.py`,
  `ml/tests/unit/training/event_driven/test_dataset_service.py`,
  `ml/tests/unit/training/event_driven/test_bus.py`.
- [x] Re-run full event-driven coverage workflow and record exact per-module and aggregate deltas.
- [ ] Next-up module queue after T2 (sorted by current coverage):
  `worker` (74.11%),
  `guardrails/join_checks` (80.00%),
  `orchestrator` (86.21%),
  `guardrails/validation_bundle` (89.47%),
  `azure_events` (89.76%),
  `economic_metrics` (92.12%),
  `guardrails/dataset` (97.63%),
  `global_run` (97.70%),
  `payloads` (98.09%),
  `wave_planner` (100.00%).
- [ ] Deferred Phase 1.6 strategies tail remains queued for later revisit:
  `ml/strategies/base_facade.py`,
  `ml/strategies/risk.py`,
  `ml/strategies/common/position_management.py`,
  `ml/strategies/common/order_submission.py`,
  `ml/strategies/common/returns_updater.py`,
  `ml/strategies/ml_strategy.py`,
  `ml/strategies/portfolio.py`.
- [ ] Deferred Phase 1.5 orchestration tail remains unchanged:
  `ml/orchestration/dataset_builder.py` (83.47%) plus orchestration wiring/scheduler guard tests.
- [ ] Deferred Phase 1.4 registry tail remains unchanged unless touched:
  `ml/registry/common/data_persistence.py`,
  `ml/registry/base.py`,
  `ml/registry/data_registry_facade.py`,
  `ml/registry/common/ab_testing.py`.
- [ ] Phase transition note: proceed to Phase 2 now; return later to complete this Phase 1.7 residual queue (worker-first).

Execution history
- 2026-02-08: Captured dirty baseline snapshot with `git status --porcelain`; concurrent overlap detected in `ml/tests/unit/training/event_driven/test_worker.py`, `ml/training/event_driven/worker.py`, and `COVERAGE_PLAN.md`, so this batch avoided worker edits and targeted non-overlapping event-driven shards.
- 2026-02-08: Captured fresh Phase 1.7 baseline using the required command sequence (`coverage erase`, full event-driven unit shard run, sorted include report). Baseline aggregate measured `75.20%`.
- 2026-02-08: Captured focused baseline queue report with:
  `validation_bundle` `48.58%`,
  `azure_events` `53.61%`,
  `teacher_export` `65.38%`,
  `sweep` `73.50%`,
  `worker` `74.11%`,
  `wave_planner` `77.04%`,
  `global_run` `77.70%`,
  `orchestrator` `86.21%`.
- 2026-02-08: Extended deterministic existing shards only (`test_validation_bundle.py`, `test_azure_events.py`, `test_teacher_export_validation.py`, `test_sweep.py`) with branch/fallback/error-path coverage.
- 2026-02-08: Re-ran full event-driven coverage workflow with the same command pattern (plus focused include report) and lifted aggregate `75.20%` -> `81.56%` (`+6.36`), with targeted module deltas:
  `guardrails/validation_bundle` `48.58%` -> `89.47%` (`+40.89`),
  `azure_events` `53.61%` -> `89.76%` (`+36.15`),
  `teacher_export` `65.38%` -> `96.15%` (`+30.77`),
  `sweep` `73.50%` -> `98.00%` (`+24.50`).
- 2026-02-08: Required validations after edits:
  `make validate-fixtures` passed;
  `poetry run mypy ml --strict` passed;
  `poetry run ruff check ml` reports one pre-existing unrelated failure in `ml/core/integration_facade.py` (unused import `EngineManager`), with Phase 1.7-local lint issues resolved.
- 2026-02-08: Deferred tails explicitly unchanged this session:
  Phase 1.6 strategies tail deferred,
  Phase 1.5 `dataset_builder` + wiring/scheduler guards deferred,
  Phase 1.4 registry tail deferred unless touched.
- 2026-02-08: Captured fresh T2 baseline and queue report from the existing T1 state using:
  `git status --porcelain`;
  `poetry run coverage erase`;
  `poetry run coverage run -m pytest -q ml/tests/unit/training/event_driven`;
  `poetry run coverage report --sort=cover --include 'ml/training/event_driven/*,ml/training/event_driven/guardrails/*'`;
  `poetry run coverage report --include 'ml/training/event_driven/worker.py,ml/training/event_driven/economic_metrics.py,ml/training/event_driven/wave_planner.py,ml/training/event_driven/guardrails/dataset.py,ml/training/event_driven/global_run.py,ml/training/event_driven/payloads.py,ml/training/event_driven/orchestrator.py,ml/training/event_driven/guardrails/validation_bundle.py,ml/training/event_driven/azure_events.py'`.
  Baseline aggregate measured `81.56%`.
- 2026-02-08: Extended deterministic existing shards only (`test_wave_planner.py`, `test_global_run.py`, `test_dataset_service.py`, `test_bus.py`) with branch/fallback/error-path coverage, then re-ran full event-driven coverage twice (second pass focused on `guardrails/dataset` residual branches).
- 2026-02-08: Re-ran full event-driven coverage workflow and lifted aggregate `81.56%` -> `86.32%` (`+4.76`), with targeted module deltas:
  `economic_metrics` `75.76%` -> `92.12%` (`+16.36`),
  `wave_planner` `77.04%` -> `100.00%` (`+22.96`),
  `guardrails/dataset` `77.51%` -> `97.63%` (`+20.12`),
  `global_run` `77.70%` -> `97.70%` (`+20.00`),
  `payloads` `80.25%` -> `98.09%` (`+17.84`),
  `worker` `74.11%` -> `74.11%` (`+0.00`),
  `orchestrator` `86.21%` -> `86.21%` (`+0.00`),
  `guardrails/validation_bundle` `89.47%` -> `89.47%` (`+0.00`),
  `azure_events` `89.76%` -> `89.76%` (`+0.00`).
- 2026-02-08: Required validations after T2 edits:
  `make validate-fixtures` passed;
  `poetry run mypy ml --strict` passed;
  `poetry run ruff check ml` still reports the same pre-existing unrelated failure in `ml/core/integration_facade.py` (unused import `EngineManager`).
- 2026-02-08: Deferred tails remain explicitly unchanged after T2:
  Phase 1.6 strategies tail deferred,
  Phase 1.5 `dataset_builder` + orchestration wiring/scheduler guards deferred,
  Phase 1.4 registry tail deferred unless touched.
- 2026-02-08: Phase transition decision recorded for next session: start Phase 2 supporting runtime services baseline/inventory now, then return to Phase 1.7 residual tail queue (`worker`, `guardrails/join_checks`, `orchestrator`, `guardrails/validation_bundle`, `azure_events`) after Phase 2 progress.

## Phase 2: Supporting Runtime Services
Domains: `ml/features`, `ml/observability`, `ml/monitoring`, `ml/pipelines`, `ml/tasks`.
Focus: feature transform invariants, observability DTO contracts, pipeline wiring, task scheduling guards.
Primary test types: metamorphic tests for transforms, contract tests for DTOs and schemas, pairwise config tests.
Status: In progress (P2-B1, P2-B2, and P2-B3 completed on 2026-02-08; queue reprioritized to remaining observability/features modules).

Phase 2 checklist
- [x] Phase 2 coverage snapshot recorded.
- [ ] Features workstream completed.
- [ ] Observability workstream completed.
- [ ] Monitoring workstream completed.
- [ ] Pipelines workstream completed.
- [ ] Tasks workstream completed.

### Phase 2.1 Observability kickoff (P2-B1)
Status: In progress (first module-first lift complete; queue reprioritized from fresh sorted baseline).

Module inventory and queue (coverage-first from fresh Phase 2 sorted report)
- [x] Fresh Phase 2 aggregate baseline captured: `55.67%`.
- [x] `ml/observability/migrations.py` (`25.26%` -> `98.95%` on 2026-02-08 P2-B1).
- [x] `ml/observability/tracing.py` (`67.20%` -> `91.94%` on 2026-02-08 P2-B1).
- [x] `ml/observability/bootstrap.py` (`71.43%` -> `100.00%` on 2026-02-08 P2-B1).
- [ ] Next-up Phase 2 low-module queue (unchanged files deferred until targeted): `ml/monitoring/health.py` (`24.49%`), `ml/monitoring/collector.py` (`24.56%`), `ml/pipelines/build_runner.py` (`50.91%`), `ml/observability/ml_async_persistence.py` (`72.15%`), `ml/observability/scheduler.py` (`81.46%`), `ml/features/pipeline_stream.py` (`83.64%`).

Execution history
- 2026-02-08: Captured required dirty-worktree snapshot with `git status --porcelain`; discovered Phase 2 test directories with `for d in ml/tests/unit/features ml/tests/unit/observability ml/tests/unit/monitoring ml/tests/unit/pipelines ml/tests/unit/tasks; do [ -d "$d" ] && echo "$d"; done`.
- 2026-02-08: Captured fresh Phase 2 baseline and sorted queue with
  `poetry run coverage erase`;
  `poetry run coverage run -m pytest -q ml/tests/unit/features ml/tests/unit/observability ml/tests/unit/monitoring ml/tests/unit/pipelines ml/tests/unit/tasks || true`;
  `poetry run coverage report --sort=cover --include 'ml/features/*,ml/observability/*,ml/monitoring/*,ml/pipelines/*,ml/tasks/*'`.
  Baseline aggregate measured `55.67%` (run contains pre-existing feature-suite failures in `test_feature_engineering_hypothesis.py` and `test_feature_validation.py` due `KeyError '__pyx_vtable__'` import path).
- 2026-02-08: Captured focused queue baseline for selected low modules with `poetry run coverage report --include 'ml/observability/migrations.py,ml/observability/tracing.py,ml/observability/bootstrap.py'`; focused baseline measured `53.90%` (`migrations` `25.26%`, `tracing` `67.20%`, `bootstrap` `71.43%`).
- 2026-02-08: Extended deterministic existing shards only (`ml/tests/unit/observability/test_migrations.py`, `ml/tests/unit/observability/test_tracing_unit.py`, `ml/tests/unit/observability/test_bootstrap_env.py`) and validated with `poetry run pytest -q ml/tests/unit/observability/test_migrations.py ml/tests/unit/observability/test_tracing_unit.py ml/tests/unit/observability/test_bootstrap_env.py` (56 passed).
- 2026-02-08: Re-ran full Phase 2 coverage workflow with the same command trio and refreshed focused include report; Phase 2 aggregate lifted `55.67%` -> `56.63%` (`+0.96`), with targeted module deltas:
  `ml/observability/migrations.py` `25.26%` -> `98.95%` (`+73.69`),
  `ml/observability/tracing.py` `67.20%` -> `91.94%` (`+24.74`),
  `ml/observability/bootstrap.py` `71.43%` -> `100.00%` (`+28.57`).
- 2026-02-08: Required validations after P2-B1 edits:
  `make validate-fixtures` passed;
  `poetry run mypy ml --strict` passed;
  `poetry run ruff check ml` passed.
  Confirmation update: the previously known unrelated `ruff` F401 in `ml/core/integration_facade.py` is no longer present.
- 2026-02-08: Deferred-note status after P2-B1 remains explicit and unchanged:
  Phase 1.7 residual queue deferred for later revisit (`worker`, `guardrails/join_checks`, `orchestrator`, `guardrails/validation_bundle`, `azure_events`);
  Phase 1.6 strategies tail deferred;
  Phase 1.5 `dataset_builder` + orchestration wiring/scheduler guards deferred;
  Phase 1.4 registry tail deferred unless touched.

### Phase 2.2 Monitoring low-module lift (P2-B2)
Status: In progress (two lowest modules lifted in this batch; next queue updated).

Module inventory and queue (coverage-first from fresh Phase 2 sorted report)
- [x] Fresh Phase 2 aggregate baseline captured for this batch: `56.63%`.
- [x] `ml/monitoring/health.py` (`24.49%` -> `93.88%` on 2026-02-08 P2-B2).
- [x] `ml/monitoring/collector.py` (`24.56%` -> `95.61%` on 2026-02-08 P2-B2).
- [ ] Next-up Phase 2 low-module queue: `ml/pipelines/build_runner.py` (`50.91%`), `ml/observability/ml_async_persistence.py` (`72.15%`), `ml/observability/scheduler.py` (`81.46%`), `ml/features/pipeline_stream.py` (`83.64%`).

Execution history
- 2026-02-08: Captured required dirty-worktree snapshot with `git status --porcelain`; discovered Phase 2 test directories with `for d in ml/tests/unit/features ml/tests/unit/observability ml/tests/unit/monitoring ml/tests/unit/pipelines ml/tests/unit/tasks; do [ -d "$d" ] && echo "$d"; done`.
- 2026-02-08: Captured fresh Phase 2 baseline and sorted queue with
  `poetry run coverage erase`;
  `poetry run coverage run -m pytest -q ml/tests/unit/features ml/tests/unit/observability ml/tests/unit/monitoring ml/tests/unit/pipelines ml/tests/unit/tasks || true`;
  `poetry run coverage report --sort=cover --include 'ml/features/*,ml/observability/*,ml/monitoring/*,ml/pipelines/*,ml/tasks/*'`.
  Batch baseline aggregate measured `56.63%` (run contains the same pre-existing feature-suite failures in `test_feature_engineering_hypothesis.py` and `test_feature_validation.py` due `KeyError '__pyx_vtable__'` import path).
- 2026-02-08: Captured focused queue baseline with
  `poetry run coverage report --include 'ml/monitoring/health.py,ml/monitoring/collector.py,ml/pipelines/build_runner.py,ml/observability/ml_async_persistence.py,ml/observability/scheduler.py,ml/features/pipeline_stream.py'`;
  focused baseline measured `55.02%` (`health` `24.49%`, `collector` `24.56%`, `build_runner` `50.91%`, `ml_async_persistence` `72.15%`, `scheduler` `81.46%`, `pipeline_stream` `83.64%`).
- 2026-02-08: Added deterministic monitoring tests in existing domain shards with new files
  `ml/tests/unit/monitoring/test_health_additional.py` and `ml/tests/unit/monitoring/test_collector.py`;
  validated focused shard with `poetry run pytest -q ml/tests/unit/monitoring` (49 passed).
- 2026-02-08: Re-ran full Phase 2 coverage workflow with the same command trio and refreshed focused include report; Phase 2 aggregate lifted `56.63%` -> `59.49%` (`+2.86`), with targeted module deltas:
  `ml/monitoring/health.py` `24.49%` -> `93.88%` (`+69.39`),
  `ml/monitoring/collector.py` `24.56%` -> `95.61%` (`+71.05`).
- 2026-02-08: Required validations after P2-B2 edits:
  `make validate-fixtures` passed;
  `poetry run mypy ml --strict` passed;
  `poetry run ruff check ml` passed.
  Confirmation update: `poetry run ruff check ml/core/integration_facade.py` also passes; the previously known unrelated issue remains resolved.
- 2026-02-08: Deferred-note status after P2-B2 remains explicit and unchanged:
  Phase 1.7 residual queue deferred for later revisit (`worker`, `guardrails/join_checks`, `orchestrator`, `guardrails/validation_bundle`, `azure_events`) with explicit return after Phase 2 queue progress;
  Phase 1.6 strategies tail deferred;
  Phase 1.5 `dataset_builder` + orchestration wiring/scheduler guards deferred;
  Phase 1.4 registry tail deferred unless touched.

### Phase 2.3 Pipeline runner lift (P2-B3)
Status: In progress (next queue head completed; queue reprioritized from fresh baseline).

Module inventory and queue (coverage-first from fresh Phase 2 sorted report)
- [x] Fresh Phase 2 aggregate baseline captured for this batch: `59.49%`.
- [x] `ml/pipelines/build_runner.py` (`50.91%` -> `95.15%` on 2026-02-08 P2-B3).
- [ ] Next-up Phase 2 low-module queue: `ml/observability/ml_async_persistence.py` (`72.15%`), `ml/observability/scheduler.py` (`81.46%`), `ml/features/pipeline_stream.py` (`83.64%`).

Execution history
- 2026-02-08: Captured required dirty-worktree snapshot with `git status --porcelain`; discovered Phase 2 test directories with `for d in ml/tests/unit/features ml/tests/unit/observability ml/tests/unit/monitoring ml/tests/unit/pipelines ml/tests/unit/tasks; do [ -d "$d" ] && echo "$d"; done`.
- 2026-02-08: Captured fresh Phase 2 baseline and sorted queue with
  `poetry run coverage erase`;
  `poetry run coverage run -m pytest -q ml/tests/unit/features ml/tests/unit/observability ml/tests/unit/monitoring ml/tests/unit/pipelines ml/tests/unit/tasks || true`;
  `poetry run coverage report --sort=cover --include 'ml/features/*,ml/observability/*,ml/monitoring/*,ml/pipelines/*,ml/tasks/*'`.
  Batch baseline aggregate measured `59.49%` (run contains the same pre-existing feature-suite failures in `test_feature_engineering_hypothesis.py` and `test_feature_validation.py` due `KeyError '__pyx_vtable__'` import path).
- 2026-02-08: Captured focused queue baseline with
  `poetry run coverage report --include 'ml/pipelines/build_runner.py,ml/observability/ml_async_persistence.py,ml/observability/scheduler.py,ml/features/pipeline_stream.py'`;
  focused baseline measured `69.75%` (`build_runner` `50.91%`, `ml_async_persistence` `72.15%`, `scheduler` `81.46%`, `pipeline_stream` `83.64%`).
- 2026-02-08: Extended deterministic tests in the existing pipeline shard only (`ml/tests/unit/pipelines/test_build_runner.py`) and validated with `poetry run pytest -q ml/tests/unit/pipelines/test_build_runner.py` (11 passed).
- 2026-02-08: Re-ran full Phase 2 coverage workflow with the same command trio and refreshed focused include report; Phase 2 aggregate lifted `59.49%` -> `60.67%` (`+1.18`), with targeted module delta:
  `ml/pipelines/build_runner.py` `50.91%` -> `95.15%` (`+44.24`).
- 2026-02-08: Required validations after P2-B3 edits:
  `make validate-fixtures` passed;
  `poetry run mypy ml --strict` passed;
  `poetry run ruff check ml` passed;
  `poetry run ruff check ml/core/integration_facade.py` passed.
  Confirmation update: the previously known unrelated `ruff` F401 in `ml/core/integration_facade.py` remains resolved.
- 2026-02-08: Deferred-note status after P2-B3 remains explicit and unchanged:
  Phase 1.7 residual queue deferred for later revisit (`worker`, `guardrails/join_checks`, `orchestrator`, `guardrails/validation_bundle`, `azure_events`) with explicit return after continued Phase 2 progress;
  Phase 1.6 strategies tail deferred;
  Phase 1.5 `dataset_builder` + orchestration wiring/scheduler guards deferred;
  Phase 1.4 registry tail deferred unless touched.

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
