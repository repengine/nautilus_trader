# ML Remediation Plan

Working plan for stabilizing the Nautilus Trader ML layer end-to-end (stores → events/bus → features → orchestration → observability) with strict typing, repeatable tests, and hot‑path budgets.

## Guardrails & References
- Follow typing/lint/testing mandates (`poetry run mypy ml --strict`, `poetry ruff check ml`, focused pytest shards) [AGENTS.md].
- Prefer property/contract/metamorphic/pairwise tests; avoid config identity and enum isinstance/identity checks; add `@pytest.mark.serial` for DB tests; keep mocks test-scoped [ml/tests/docs/TESTING_STRATEGY.md], [ml/tests/docs/TEST_ANTI_PATTERNS.md].
- Use shared fixtures via pytest plug-in; lean on `fresh_store_bundle`, `cloned_test_database`, and isolation guidance [ml/tests/fixtures/FIXTURE_GUIDE.md], [db-fixture-optimizations.md].
- Schema registry, identifier templates, and dataset type defaults live in `ml/schema.py`; subscriptable type shims in `ml/ml_types.py` are the safe place for typing adjustments [ml/schema.py], [ml/ml_types.py].

## Scope & Milestones
### Milestone B (confirmed)
Primary goal: stabilize deployment‑critical correctness first (DB/migrations, stores/registry integration, bus topics + correlation, feature parity invariants, orchestrator/deployment wiring, observability/tracing). Performance budgets are tracked but treated as a follow-on lane once correctness is green.
Exit gate for Milestone B: `mypy` + `ruff` green and targeted deployment-critical pytest shards green (Postgres/migrations, orchestrator/deployment, feature parity/invariants, tracing/dashboard), even if perf shards remain red.

## Definition of Done (lanes)
- Types: `poetry run mypy ml --strict`
- Lint: `poetry ruff check ml`
- Unit/contracts: `poetry run pytest -q ml/tests/unit ml/tests/contracts`
- Integration/E2E: `poetry run pytest -q ml/tests/integration ml/tests/e2e`
- Performance/hot path: `poetry run pytest -q ml/tests/performance`
- Validators (when touching events/metrics): `make validate-events`, `make validate-metrics`
- Coverage: ML modules ≥90% (`coverage report`)

Checkbox convention: only mark `[x]` when the relevant lane(s) above are green for the latest recorded baseline.

## Test Environment (current reality)
- Most DB-using tests run against `DATABASE_URL` loaded by pytest from `ml/tests/.env` (see `ml/tests/conftest.py`), currently `postgresql://postgres:postgres@localhost:5434/nautilus_test`.
- Helper defaults:
  - `TEST_DB_PORT` defaults to `5434` (`ml/tests/fixtures/database_fixtures.py`, `ml/tests/utils/db.py`)
  - `TEST_DB_PASSWORD` defaults to `postgres` (`ml/tests/utils/db.py`)
  - Template DB name defaults to `nautilus_template` (`TEST_DB_TEMPLATE_NAME`)
- Best-practice lane split (per `ml/tests/docs/TESTING_STRATEGY.md`):
  - Unit/contracts/property: **DB-free** (mock/patch EngineManager; do not attempt real DNS/DB connects).
  - Integration/E2E/perf: real Postgres via `cloned_test_database` / `fresh_store_bundle`.
  - DB-free suite selection: running only `ml/tests/unit`, `ml/tests/contracts`, `ml/tests/property`, `ml/tests/metamorphic`, or `ml/tests/combinatorial` skips DB initialization and skips `@pytest.mark.database` tests by default; set `ML_FORCE_DB_INIT=1` to force DB init.
  - Auto-marking: tests requesting DB-backed fixtures (e.g., `test_database`, `cloned_test_database`, `fresh_store_bundle`) are auto-marked `@pytest.mark.database` + `@pytest.mark.serial` in `ml/tests/conftest.py`.

### Postgres Connectivity (recommendation)
- Treat `DATABASE_URL` as the single source of truth for DB credentials in tests; avoid hard-coded DSNs in unit/contract suites.
- If local Postgres is not `postgres/postgres` on `:5434`, prefer overriding via env (`DATABASE_URL` or `TEST_DB_PASSWORD`) rather than editing test code.
- Ensure helpers like `build_postgres_url(...)` do not silently drift from `DATABASE_URL` defaults (e.g., `database="nautilus"` vs `nautilus_test`) in DB-touching tests.

## Bus Topic Scheme (confirmed)
- Decision: `domain_op` is the canonical/default production scheme (matches `MessageBusConfig` default and groups by domain/operation via `ml.common.message_topics.build_topic_for_stage`).
- Decision: `stage_first` remains supported via env (`ML_BUS_SCHEME=stage_first`, `ML_BUS_TOPIC_PREFIX=...`) for compatibility and stage-specific consumers.
- Note: `ML_BUS_TOPIC_PREFIX` applies only to `stage_first` topics; `domain_op` uses canonical `ml.{domain}.{operation}.{instrument}`.
- Testing rule: contract/unit tests must validate topic behavior without touching the DB by using shared fixtures (e.g., `patch_engine_manager`), not fake DSNs like `postgresql://ignored`.

## Recent Targeted Verification (2025-12-19)
- Lint/types: `poetry ruff check ml`, `poetry run mypy ml --strict` (green).
- Unit/contracts (DB-free selection): `poetry run pytest -q ml/tests/unit ml/tests/contracts` (green; DB init skipped; `@pytest.mark.database` tests skipped).
- E2E: `poetry run pytest -q ml/tests/e2e/test_feature_store_e2e.py::test_10_health_check` (green).

## Recent Targeted Verification (post-skip remediation)
- Lint/types: `poetry ruff check ml`, `poetry run mypy ml --strict` (green).
- Targeted suites: `poetry run pytest -q ml/tests/integration/features/test_feature_calculator_facade_integration.py ml/tests/unit/actors/test_facade_parity.py::test_parity_async_persistence ml/tests/unit/actors/common/test_features.py ml/tests/unit/features/common/test_feature_calculator.py::TestComputeFeatures::test_compute_features_legacy_compatibility ml/tests/performance/test_feature_calculator_microbench.py::TestFeatureCalculatorPerformance::test_compute_features_legacy_microbench ml/tests/unit/training/teacher/test_tft_teacher_streaming.py::test_collect_streaming_logits_aligns_with_model_device ml/tests/fixtures/test_pollution_detection.py::test_engine_manager_pool_statistics_show_growth` (green).

## Latest Full-Suite Baseline (2025-12-17)
- Summary: `52 failed, 6811 passed, 19 skipped, 122 deselected, 1 xfailed, 1 xpassed, 3 errors` in `2716s (~45m)`
- Slowest tests (top 10):
```
111.74s call     tests/property/test_model_store_predictions_advanced.py::TestModelStorePredictionInvariants::test_version_consistency_invariant
83.84s call     tests/property/test_model_store_predictions_advanced.py::TestModelStorePredictionInvariants::test_confidence_bounds_invariant
63.15s call     tests/property/test_model_store_predictions_advanced.py::TestModelStorePerformanceInvariants::test_no_data_loss_during_flush_invariant
47.99s call     tests/property/test_model_store_predictions_advanced.py::test_model_store_stateful
22.83s call     tests/property/test_model_store_predictions_advanced.py::TestModelStorePredictionInvariants::test_batch_atomicity_invariant
19.13s call     tests/property/test_model_store_predictions_advanced.py::TestModelStorePredictionInvariants::test_temporal_consistency_invariant
15.19s call     tests/property/test_feature_calculator_properties.py::TestFeatureCalculatorProperties::test_batch_online_parity_property
9.65s call     tests/unit/data/providers/test_factory.py::TestTransformProviderAdapter::test_adapter_handles_arbitrary_data_sizes
9.37s call     tests/integration/test_transform_provider_integration.py::TestTransformProviderIntegration::test_provider_scalability
8.27s call     tests/integration/deployment/test_deployment_integration.py::TestDeploymentIntegration::test_pipeline_with_stores_initialization
```

### Baseline blockers by cluster (status updated 2025-12-19)
- Postgres/migrations + env isolation:
  - [x] `ml/tests/integration/test_postgres_integration.py::test_migrations_applied`
  - [x] `ml/tests/integration/registry/test_data_registry_postgres_backend_smoke.py`
  - [x] `ml/tests/e2e/test_feature_store_e2e.py::test_10_health_check` (FeatureStore ctor should instantiate and report unhealthy on invalid DSN)
  - [x] `ml/tests/contracts/test_store_env_topic_config_contracts.py`
- Feature parity + invariants:
  - [x] `ml/tests/integration/test_end_to_end_pipeline.py::TestEndToEndPipeline::test_online_feature_parity`
  - [x] Volume-ratio boundedness invariants: `ml/tests/property/test_feature_calculator_properties.py`
  - [x] Trade-flow metamorphic expectations: `ml/tests/metamorphic/test_feature_transforms.py`
  - [x] Facade parity: `poetry run pytest -q ml/tests/facades -k feature_engineer`
- Orchestrator/facades:
  - [x] Alias token mismatch: `ml/tests/integration/orchestration/test_ml_pipeline_orchestrator_facade.py::test_default_alias_uses_facade`
  - [x] Read-only attribute in smoke: `ml/tests/e2e/test_orchestrator_smoke.py`
  - [x] TFT builder init contract: `ml/tests/facades/test_tft_builder_parity.py`
  - [x] Rehydration exits: `ml/tests/integration/deployment/test_pipeline_rehydration.py`
- Observability/dashboard:
  - [x] Trace context injection + cold-path decorators: `ml/tests/integration/test_observability_tracing.py`
  - [x] Dashboard service export: `ml/tests/integration/dashboard/test_streaming_state_endpoint.py`
- Performance/hot path:
  - [x] Perf suites: `poetry run pytest -q ml/tests/performance`
- [x] Logging API compatibility: `ml/tests/unit_tests/actors/test_multi_signal_actor.py`
- [x] Typing contract: `ml/tests/unit/core/test_create_integrated_actor_generic.py`

## Work Plan (prioritized)
### Next Window Start Here
- Remaining focus:
  - [x] Make unit/contract suites DB-free by default (no real DNS/DB connects).
  - [x] Fix `ml/tests/e2e/test_feature_store_e2e.py::test_10_health_check` (FeatureStore should not connect during `__init__` for invalid DSNs).

### P0 — Postgres/migrations + test env isolation
- [x] Make unit/contract suites DB-free by default (no real DNS/DB connects). Replace fake DSNs like `postgresql://ignored` with `patch_engine_manager`/`mock_engine_manager` so store constructors cannot trigger engine creation.
- [x] Ensure cloned DB fixtures always include migrations and required tables (`ml_feature_values`, `ml_data_events`) and match deployment expectations (`ml.stores.migrations_runner` + instrumentation tables per `ML_DEPLOYMENT_README.md`).
- [x] Normalize DB URL usage in tests to reduce drift (`DATABASE_URL` from `ml/tests/.env` vs helpers like `build_postgres_url(database="nautilus")`); keep overrides via env (`DATABASE_URL`, `TEST_DB_PORT`, `TEST_DB_PASSWORD`) working; do not change `ml/tests/.env` unless absolutely required.
- [x] Re-run: `poetry run pytest -q ml/tests/integration/test_postgres_integration.py -k migrations_applied` and `poetry run pytest -q ml/tests/integration/registry/test_data_registry_postgres_backend_smoke.py`.

### P0 — Message bus/topic standardization (env scheme/prefix)
- [x] Ensure every publisher builds topics via `ml.common.message_topics.build_topic_for_stage` and honors `MessageBusConfig.from_env()`.
- [x] Ensure publish paths are best-effort (`try/except`), off hot paths, and include correlation IDs.
- [x] Re-run: `poetry run pytest -q ml/tests/contracts/test_store_env_topic_config_contracts.py` and `make validate-events`.

### P0 — Feature parity + invariants + hot-path guardrails
- [x] Restore facade-visible `FeatureEngineer.n_features` and any parity metadata.
- [x] Fix volume-ratio boundedness and trade-flow metamorphic expectations (batch/online parity).
- [x] Restore facade parity across modes/config variations and feature flags.
- [x] Re-run: `poetry run pytest -q ml/tests/facades -k feature_engineer` and `poetry run pytest -q ml/tests/property/test_feature_calculator_properties.py`.

### P0 — Orchestrator/facade + dashboard parity
- [x] Normalize orchestrator alias token (`component-based` vs `component_based`) and facade input validation.
- [x] Fix orchestrator smoke read-only attribute and TFT builder init signature parity.
- [x] Re-run: `poetry run pytest -q ml/tests/e2e/test_orchestrator_smoke.py` and `poetry run pytest -q ml/tests/integration/orchestration/test_ml_pipeline_orchestrator_facade.py`.

### P0 — Observability/tracing
- [x] Ensure trace context is injected into bus/payload metadata and decorators degrade gracefully when otel is unavailable.
- [x] Restore dashboard `ObservabilityService` export used by streaming state endpoint.
- [x] Re-run: `poetry run pytest -q ml/tests/integration/test_observability_tracing.py` and `poetry run pytest -q ml/tests/integration/dashboard/test_streaming_state_endpoint.py`.

### P0 — Performance budgets (P99 + zero allocations)
- [x] Fix perf harness exceptions (`test_database` NameError; streaming report `summary` key).
- [x] Re-establish P99 budgets and near-zero allocation invariants for hot paths.
- [x] Re-run: `poetry run pytest -q ml/tests/performance -k 'hot_path or zero_allocation or parity_buffer_guardrails or ml_hot_path_benchmarks'`.

### P0 — Skipped test remediation (fixtures + compatibility)
- [x] Rework async persistence parity to use `fresh_store_bundle` + `FeatureStore.read_range` (no manual schema mutation).
- [x] Enable DataFrame inputs for the `FeatureCalculator.compute_features` shim; unskip DataFrame compatibility tests/microbench.
- [x] Unskip FeatureCalculator facade integration workflows; adjust assertions for timestamp column and numeric-only scaling checks.
- [x] Convert placeholder FeaturesComponent "integration" tests to DB-free checks (mock store/registry).
- [x] Make CUDA alignment test CPU-capable when CUDA unavailable; use `cloned_test_database` for pool stats.

### P1 — Logging + typing contracts
- [x] Update exception logging calls to satisfy both runtime logger API and traceback retention (Nautilus `Logger.exception(..., exc)` or stdlib `exc_info=True`).
- [x] Restore `create_integrated_actor` public signature to accept `type[ActorT]` (keep mypy strict internally).

## Completed (green in the 2025-12-17 full-suite baseline)
- [x] Pandera `Series[...]` typing wave
- [x] Scheduler lookback/targeted/ts extraction shards
- [x] Streaming teacher/worker path (CPU-only default; CUDA alignment test remains opt-in)
