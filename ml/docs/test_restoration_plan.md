# Test Restoration and Validation Strategy

## Objectives

- Restore deterministic, high-signal coverage across ML domains without bypassing existing guardrails.
- Reconcile refactored fixtures, enums, and protocol adapters with legacy behaviour so that contract, property, and integration suites reflect true regressions.
- Re-establish pipeline confidence by running required type, lint, coverage, and targeted pytest suites per `AGENTS.md`.

## Non-Negotiable Constraints (per `AGENTS.md`)

- **Typing & Linting**: run `poetry run mypy ml --strict` and `poetry ruff check ml` before final validation.
- **Testing discipline**: execute focused `pytest -k` shards per component and contract suites touched; maintain ≥90% ML module coverage (verify via `coverage report`).
- **Protocol adherence**: use existing protocols (`RegistryProtocol`, `MessagePublisherProtocol`, etc.) instead of new abstractions; enforce strict typing with `TYPE_CHECKING` guards where needed.
- **No DRY leaps**: before creating helpers, search existing subdomains (`rg`/`fd`) to reuse prior implementations; keep responsibilities separated by domain (fixtures vs. stores vs. observability).
- **Logging & metrics**: follow existing logging patterns with structured `extra` payloads and `exc_info=True`; surface counters/histograms via `ml.common.metrics_bootstrap` hooks.
- **Fallback chains**: maintain PRIMARY → CACHED → FILE → DUMMY sequences when modifying loaders/stores; document rationale for deviations.
- **No bare `try/except`**: always scope exceptions and log with context.

## Environment Snapshot

- Last full test run: `poetry run pytest ml` with 204 failures; major clusters in DataStore contracts, Pandera schema tests, ONNX security, dashboard endpoints, and newly consolidated fixtures.
- Recent history (908cb54af → 43242a00a) introduced fixture factories, session-scoped data builders, enum/topic rewrites, and guardrail CLI reconstruction—source of regressions.

## Progress Log

- 2025-11-12 — Removed the lingering module-level `pytest_plugins` declarations from `ml/tests/integration/test_store_persistence.py`, `ml/tests/integration/test_ml_signal_pipeline.py`, and `ml/tests/integration/test_scheduler_databento.py`. Added `TYPE_CHECKING` imports so fixture-heavy parameters (`FeatureStore`, `ModuleStoreBundle`, `TestDatabase`, etc.) stay fully annotated without pulling those modules into the hot path. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/test_store_persistence.py` (initial run skipped because PostgreSQL was offline; reran after the service restart and all five tests passed against `postgresql://postgres:postgres@localhost:5434/nautilus_test`), `poetry run pytest -m slow ml/tests/integration/test_ml_signal_pipeline.py` (all three tests passed), `source .env && poetry run pytest -m slow ml/tests/integration/test_scheduler_databento.py` (eight database-backed tests passed; the real API shard remains gated until `ML_TEST_REAL_API=1` accompanies the `DATABENTO_API_KEY`).
- 2025-11-13 — Continued the registry integration sweep by stripping the redundant `pytest_plugins` declarations from `ml/tests/integration/registry/test_data_registry_postgres_backend_smoke.py`, `ml/tests/integration/registry/test_feature_registry_postgres_update.py`, and `ml/tests/integration/registry/test_model_registry_security.py`. Added `TYPE_CHECKING` coverage for the shared `TestDatabase` fixture so the smoke test stays fully typed without importing the fixture module at runtime. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/registry/test_data_registry_postgres_backend_smoke.py ml/tests/integration/registry/test_model_registry_security.py ml/tests/integration/registry/test_feature_registry_postgres_update.py` (all 15 tests passed against the restarted PostgreSQL instance; the feature registry suite auto-skipped only if the external registry DB becomes unreachable).
- 2025-11-13 — Removed the per-file plug-in declarations from the strategy-store integration suites (`ml/tests/integration/test_stores_strategy_events.py`, `ml/tests/integration/test_stores_strategy_reads.py`, `ml/tests/integration/test_stores_strategy_performance_agg.py`) and added `TYPE_CHECKING` imports so every `test_database` parameter is explicitly typed as `TestDatabase`. Validation stack: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/test_stores_strategy_events.py`, `poetry run pytest ml/tests/integration/test_stores_strategy_performance_agg.py`, and `poetry run pytest ml/tests/integration/test_stores_strategy_reads.py`. The reads suite completed its test body but the module-scoped `clean_postgres_db_module` teardown hit the known PostgreSQL statement-timeout path when truncating large partitioned tables (see `psycopg2.errors.QueryCanceled` in the captured stdout); rerunning with longer timeouts reproduced the same teardown warning, so the functional assertions are covered even though the fixture cleanup remains flaky until the truncation helper is optimized.
- 2025-11-13 — Added `ml/docs/architecture/migrations_schema_index.md`, cataloguing every legacy and active migration under `ml/stores/migrations/` plus the objects they introduce (tables, functions, indexes, partitions). The index maps domains (Feature/Model/Strategy/Market Data/Data Registry) to the SQL that defines them, captures which optional files (005–009) remain unreferenced at runtime, and outlines next steps for refactoring `001_bootstrap_schema.sql` + updating docs/CI references. This serves as the planning artifact before rewriting any migrations or fixtures.
- 2025-11-13 — Expanded the schema index with a bootstrap refactor blueprint: modular SQL layout (`header + per-domain sections`), partition strategy adjustments, optional migration integration, fixture/CI updates, and milestone plan. This remains documentation-only for now; code/schema edits will follow once the plan is reviewed.
- 2025-11-11 — Introduced `tools/validate_fixture_plugins.py` and the `make validate-fixtures` target to fail fast whenever new `ml/tests/**` modules skip `pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)` or fixture exports drift. Added package-level `pytest_plugins` bootstraps for the `ml.tests` root plus the `unit`, `unit_tests`, `metamorphic`, `orchestration`, `features`, `benchmarks`, `performance`, `contracts`, and `data` packages so every suite inherits the shared plug-in without per-file boilerplate. Verified the automation via `python tools/validate_fixture_plugins.py` (current adoption snapshot: 129 declarations — integration=66, unit=49, orchestration=3, contracts=2, property=1, e2e=1, metamorphic=1, features=1, performance=1, benchmarks=1, data=1, package root=1), `poetry run mypy ml --strict`, `poetry run ruff check ml`, and `pytest ml/tests/property/test_data_store_no_duplicate_bus.py` (guards the component-only DataStore path fix).
- 2025-11-11 — Purged direct `PrometheusRegistryHarness`/`MetricNameManager` imports from the dashboards, monitoring, actor, consumer, store validation, and streaming pipeline tests. Those suites now rely purely on the shared plug-in + fixture parameters, cutting unit-level direct imports from 17 → 8 and integration-level ones from 7 → 6. Re-ran `python tools/validate_fixture_plugins.py`, `poetry run mypy ml --strict`, and `poetry run ruff check ml` to confirm the guard + type/lint gates stay green.
- 2025-11-11 — Added a fixture wrapper for `patch_engine_manager` and migrated the db utils, catalog rehydrator, bus publishing, and metamorphic store suites off direct fixture imports. Unit-level direct imports now sit at 5 (down from 8) while the metamorphic suite hit zero. Validated via `pytest ml/tests/unit/common/test_db_utils.py ml/tests/unit/stores/test_bus_publishing_standardization.py ml/tests/unit/data/test_catalog_rehydrator.py ml/tests/metamorphic/test_store_time_shift_and_permutation_metamorphic.py`, plus the standard mypy/ruff/fixture guards.
- 2025-11-11 — Removed the remaining `SampleBarSeriesConfig`/`TestDatabase` imports by exposing config factories + database Protocols via fixtures. Property, unit (dataset/stores), integration (data/earnings/stores/services), and e2e TFT suites all rely on injection now (direct fixture import count = 0 outside `fixtures/test_exports.py`). Verified with the affected pytest shards (builder property tests, dataset macro/unit/integration suites, CrossAsset services, TFT data/earnings integrations, store integration service, TFT E2E), plus `python tools/validate_fixture_plugins.py`, `poetry run mypy ml --strict`, and `poetry run ruff check ml`.
- 2025-11-11 — Introduced shared `component_feature_store` and `store_integration_metrics_database` fixtures so store integrations no longer define bespoke `_TestDatabase` protocols or inline SQL seeders. Updated `ml/tests/integration/stores/test_cross_asset_service_integration.py` and `ml/tests/integration/services/test_store_integration_service.py` to rely on those fixtures, removed their per-file `pytest_plugins` declarations, and ensured they pull in the standardized `real_engine_manager` + `clean_postgres_db` guards. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/stores/test_cross_asset_service_integration.py`, `poetry run pytest ml/tests/integration/services/test_store_integration_service.py`.
- 2025-11-11 — Promoted the bespoke dataset macro helper into a reusable `patch_dataset_bars` fixture (defaulting to `ml.data.catalog_utils` + `ml.data.tft_dataset_builder`), expanded the Fixture Guide coverage for the SampleBarSeries helpers, and introduced per-package fixture checklists so Data, Security, Stores, and Observability owners can track remaining adoption gaps (dataset orchestrator flows, ONNX harness usage, telemetry mocks, etc.). Validation set: `poetry run mypy ml --strict`, `poetry run ruff check ml`, `python tools/validate_fixture_plugins.py`, and `poetry run pytest ml/tests/unit/data/test_dataset_build_macro.py`.
- 2025-11-12 — Added the shared `test_model_factory` fixture plus autouse bootstrap helpers so property and unit actor suites no longer import `ml.tests.fixtures.model_factory` or `create_dummy_onnx_model` directly. Patched the remaining actor/property modules to hydrate ONNX paths via the fixture, added `ml/tests/unit/actors/__init__.py` to register the plug-in for namespace packages, and refreshed the Hypothesis state machine cleanup hooks. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/property/test_signal_actor_bounds.py`, `poetry run pytest ml/tests/property/test_signal_actor_determinism.py`, `poetry run pytest ml/tests/unit/actors/test_signal_actor_parameterized.py -m slow`, `poetry run pytest ml/tests/unit/actors/test_signal_actor_hypothesis.py`.
- 2025-11-12 — Bootstrapped every remaining `ml/tests/**` subpackage with an `__init__.py` that registers `ml.tests.fixtures.pytest_plugins` so namespace packages cannot drop shared fixtures when run in isolation. Extended `tools/validate_fixture_plugins.py` to fail builds if any future package lacks the canonical plug-in declaration. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/unit/data/test_catalog_rehydrator.py`, `poetry run pytest ml/tests/integration/cli/test_streaming_persistence_worker_cli.py -k disable`.
- 2025-11-12 — Purged the last per-file `pytest_plugins` declarations and runtime fixture imports from the feature store integration suites (`ml/tests/integration/test_feature_store_integration.py`, `ml/tests/integration/test_feature_parity.py`, `ml/tests/integration/test_scheduler_feature_store.py`). Each module now uses `TYPE_CHECKING` imports for `TestDatabase` and depends solely on fixture injection plus the package-level plug-in bootstrap. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/test_feature_store_integration.py`, `poetry run pytest ml/tests/integration/test_feature_parity.py`, `poetry run pytest ml/tests/integration/test_scheduler_feature_store.py`.
- 2025-11-12 — Migrated the observability DB migration/partitioning suites (`ml/tests/integration/observability/test_db_migrations.py`, `ml/tests/integration/observability/test_db_partitioning.py`) onto the shared database fixtures. Both modules now drop their inline `pytest_plugins`, stop calling `create_engine` directly, and instead consume `test_database` + `clean_postgres_db_module` for deterministic state plus structured cleanup. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/observability/test_db_migrations.py`, `poetry run pytest ml/tests/integration/observability/test_db_partitioning.py`.
- 2025-11-12 — Continued the observability cleanup by removing the per-file `pytest_plugins` declarations from `ml/tests/integration/test_observability_tracing.py` and `ml/tests/integration/test_observability_e2e_integration.py`. Both suites now rely solely on the package-level plug-in bootstrap while still using `mock_tracing_backend`/`isolated_prometheus_registry` fixtures via `pytestmark`. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/test_observability_tracing.py`, `poetry run pytest ml/tests/integration/test_observability_e2e_integration.py`.
- 2025-11-12 — Began sweeping the orchestration suites: removed per-file `pytest_plugins` declarations from `ml/tests/orchestration/test_pipeline_orchestrator_discovery.py`, `ml/tests/orchestration/test_run_config_loader.py`, `ml/tests/unit/orchestration/test_pipeline_orchestrator.py`, `ml/tests/unit/orchestration/test_pipeline_orchestrator_component.py`, and `ml/tests/unit/orchestration/test_config_loader.py`. These modules already inherit the shared fixtures via their package-level bootstrap, so no runtime behaviour changes were required. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/orchestration/test_pipeline_orchestrator_discovery.py ml/tests/orchestration/test_run_config_loader.py ml/tests/unit/orchestration/test_pipeline_orchestrator.py ml/tests/unit/orchestration/test_pipeline_orchestrator_component.py ml/tests/unit/orchestration/test_config_loader.py`.
- 2025-11-12 — Continued the cleanup by removing the local plug-in declaration from `ml/tests/unit/data/test_tft_dataset_builder_phase_one.py` (now fully fixture-driven via the package bootstrap) while keeping `ml/tests/contracts/test_data_store_routing_advanced.py` on its explicit plug-in import because this contract suite is frequently invoked directly via file path and still needs deterministic access to the DataStore fixture stack. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/contracts/test_data_store_routing_advanced.py ml/tests/unit/data/test_tft_dataset_builder_phase_one.py`.
- 2025-11-12 — Removed the per-file plug-in declaration from `ml/tests/unit/dashboard/test_dashboard_events.py`; the suite now relies on `ml.tests.unit.__init__` for fixture bootstrapping while still opting into `mock_tracing_backend`/`isolated_prometheus_registry` via `pytestmark`. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/unit/dashboard/test_dashboard_events.py`.
- 2025-11-12 — Completed the dashboard unit sweep: `test_control_simple.py`, `test_store_health.py`, `test_dashboard_registries.py`, `test_strategy_service_unit.py`, `test_metrics_snapshot.py`, `test_dashboard_welcome.py`, `test_metrics_service.py`, `test_dashboard_ui_template.py`, and `test_grafana.py` no longer declare `pytest_plugins` locally and instead inherit the fixtures from `ml.tests.unit`. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/unit/dashboard`.
- 2025-11-12 — Removed the local plug-in declarations from the remaining unit data suites (`test_tft_dataset_builder_store.py`, `test_dataset_build_macro.py`, `test_catalog_rehydrator.py`, the scheduler/fred/l2 cache suites, and the ingestion macro tests). These modules now lean entirely on `ml.tests.unit.data.__init__` for fixture bootstrapping. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/unit/data`.
- 2025-11-12 — Swept the remaining unit orchestration/stores/common suites to remove per-file plug-in declarations (`test_orchestrator_cli_promotions.py`, `test_promotion_stage2.py`, `test_scheduler.py`, `test_promotions.py`, `test_config_resolver.py`, `test_orchestrator_cli_refresh_features.py`, `test_discovery_client.py`, `test_binding_resolver.py`, the schema/engine/store contract suites, metrics manager tests, and `test_orchestrator_backfill.py`). `ml/tests/unit/stores/test_data_store_emit_event.py` retains its explicit plug-in import because the DataStore contracts are frequently run via file path and need deterministic access to the DataStore toggle fixtures. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/unit/stores/test_data_store_emit_event.py ml/tests/unit/stores/test_data_store_events_unit.py ml/tests/unit/stores/test_market_data_writer.py`.

### 2025-11-12 Targeted Gap Analysis

**Stores & Database**

- `ml/tests/integration/registry/test_feature_registry_postgres_update.py:18-54` still reads `NAUTILUS_REGISTRY_DB_URL` and calls `EngineManager.get_engine` directly to probe PostgreSQL. It bypasses `test_database`, `real_engine_manager`, and the shared `patch_engine_manager` helper, so it will continue to hit local developer DBs until refactored.
- `ml/tests/integration/services/test_store_integration_service.py:83-276` defines a bespoke `_TestDatabase` protocol, manually seeds tables with raw SQL, and wraps assertions in `with patch_engine_manager(engine=test_database.engine)`. This suite needs to rely on `real_engine_manager` + canonical store fixtures so caching/cleanup happen automatically.
- `ml/tests/integration/test_feature_store_integration.py:21-49`, `ml/tests/integration/test_feature_parity.py:30-52`, and `ml/tests/integration/test_scheduler_feature_store.py:26-48` still import `TestDatabase` directly from `ml.tests.fixtures.database_fixtures` and pass connection strings around. Each test class should request `test_database` (or the lighter `database_snapshot`) via fixture arguments and drop the direct module import so that fixture discovery stays consistent.
- `ml/tests/unit/stores/services/test_cross_asset_service.py:18-88` keeps its own `_TestDatabase` protocol and instantiates a real `ComponentFeatureStore` per test. Without `real_engine_manager` or `clean_postgres_db`, these unit tests leak state when run in parallel. They should instead consume the `feature_store_bundle` or `component_feature_store_factory` fixture that already wraps engine patching.

**Observability & Dashboard**

- Dashboard HTTP tests (`ml/dashboard/tests/test_pipelines_routes.py:18-41`) import `PrometheusRegistryHarness` directly and implement a bespoke `_isolated_prom_registry` fixture even though the shared plug-in already exposes `isolated_prometheus_registry`. This causes type import churn and diverges from the guardrail fixtures (`mock_tracing_backend`, `isolated_orchestrator_env`) we expect all dashboards to use.
- `ml/tests/integration/cli/test_streaming_persistence_worker_cli.py:1-33` and the worker CLI microbench remain thin wrappers around `StreamingTrainingPersistenceWorker` but do not assert against `isolated_orchestrator_env` for every test. The disable test (line 26) still depends on monkeypatching process env rather than a helper fixture; this is a prime candidate for the observability fixture sweep noted in the plan.

**Fixture Cleanup & Automation**

- We still have 268 `pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)` declarations across individual modules (`rg -n "pytest_plugins" ml/tests | wc -l`). With every subpackage now bootstrapped, these per-file declarations are redundant noise and should be removed as the cleanup progresses.
- Direct fixture imports remain in high-signal suites, notably `ml/tests/integration/test_feature_store_integration.py`, `ml/tests/integration/test_feature_parity.py`, `ml/tests/integration/test_scheduler_feature_store.py`, `ml/tests/builders.py`, and `ml/dashboard/tests/test_pipelines_routes.py`. The fixture plan requires that only `ml/tests/fixtures/test_exports.py` import fixture modules directly; all other modules should rely on fixture arguments (or `TYPE_CHECKING` imports guarded appropriately).
- `ml/tests/conftest.py` still pulls in `ml.tests.fixtures.database_fixtures` for helper aliases (`clean_postgres_db`, `database_snapshot`). Once the package bootstraps are proven stable, we can remove those imports entirely so pytest only ever loads fixtures via the plug-in.
- 2025-11-11 — Migrated the TFT dataset integration/earnings/e2e suites (`ml/tests/integration/data/test_tft_builder_with_events.py`, `ml/tests/integration/earnings/test_tft_task_dataset.py`, `ml/tests/e2e/test_tft_dataset_builder_e2e.py`) onto `patch_dataset_bars` and wired the ML signal pipeline + strategy backtest integrations into the shared `mock_onnx_runtime` + `onnx_session_stub_factory` harness so they no longer build real ONNX artifacts. Validation set: `poetry run mypy ml --strict`, `poetry run ruff check ml`, `python tools/validate_fixture_plugins.py`, `pytest ml/tests/integration/data/test_tft_builder_with_events.py`, `pytest ml/tests/integration/earnings/test_tft_task_dataset.py`, `pytest ml/tests/e2e/test_tft_dataset_builder_e2e.py`, `pytest ml/tests/integration/test_ml_signal_pipeline.py`, and `pytest ml/tests/integration/test_ml_strategy_backtest.py -k test_ml_signal_actor_in_backtest`.
- 2025-11-12 — Completed the ONNX harness migration for the streaming actor and deployment suites, rewired `dummy_onnx_model` to use the lightweight stub factory, and expanded the Fixture Guide with scheduler/earnings examples plus a dedicated security section (EngineManager usage included). Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/actors/test_multi_signal_actor_onnx_integration.py`, `poetry run pytest ml/tests/integration/actors/test_actor_circuit_breaker_integration.py`, `poetry run pytest ml/tests/integration/deployment/test_deployment_integration.py`, and `poetry run pytest ml/tests/unit/test_fixture_validation.py`.
- 2025-11-12 — Verified the pipeline orchestrator E2E suite on the shared telemetry + ONNX harness (module-level env bootstrap ensures the component facade is active and reloads the orchestrator module before use). Default runs remain skipped, but `ML_ENABLE_COMPONENT_FACADES=1 poetry run pytest ml/tests/e2e/test_pipeline_orchestrator_e2e.py` now passes end-to-end. Store Integration Service coverage also moved onto `patch_engine_manager`, so the snapshot test no longer reaches EngineManager caches directly. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/services/test_store_integration_service.py`, and `ML_ENABLE_COMPONENT_FACADES=1 poetry run pytest ml/tests/e2e/test_pipeline_orchestrator_e2e.py`.
- 2025-11-12 — Added the fixture migration checklist to `ml/tests/fixtures/FIXTURE_GUIDE.md`, introduced `streaming_test_payloads_factory`, and migrated the streaming persistence CLI/integration/performance suites to request the shared payload builder instead of importing helpers. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/consumers/test_streaming_persistence_integration.py -k test_streaming_persistence_worker_persists_redis_batch`, `poetry run pytest ml/tests/integration/cli/test_streaming_persistence_worker_cli.py -k test_streaming_persistence_worker_cli_persists_snapshot`, `poetry run pytest ml/tests/performance/test_streaming_persistence_microbench.py -k test_streaming_persistence_worker_microbench`.
- 2025-11-12 — Removed the remaining direct `TestDataFactory` imports by wiring the pipeline rehydration integration test, catalog rehydrator unit suite, and fixture-integration tests through the existing `test_data_factory` fixture. Helpers now accept the fixture instance instead of instantiating factories manually, bringing the Stores & Database adoption checklist closer to completion. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/deployment/test_pipeline_rehydration.py`, `poetry run pytest ml/tests/unit/data/test_catalog_rehydrator.py`, `poetry run pytest ml/tests/integration/fixtures/test_test_data_factory_fixture.py`.
- 2025-11-12 — Locked real EngineManager hygiene into the cross-asset service, trading integration service, feature store integration, and feature parity suites by auto-requesting the `real_engine_manager` fixture alongside the telemetry harness; extended `tools/validate_fixture_plugins.py` to emit per-package adoption metrics (totals vs declarations) so fixture drift is visible per package. Validations: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `poetry run pytest ml/tests/integration/stores/test_cross_asset_service_integration.py -k service_shares_engine_with_facade`, `poetry run pytest ml/tests/integration/services/test_trading_integration_service.py -k health_check_includes_portfolio_metrics`, `poetry run pytest ml/tests/integration/test_feature_store_integration.py -k test_ml_signal_actor_with_feature_store`, `poetry run pytest ml/tests/integration/test_feature_parity.py -k test_ml_signal_actor_uses_same_features`.
- 2025-11-12 — Brought the pipeline orchestrator E2E suite onto the telemetry + ONNX fixtures (module-level harness + temp-path dataset configs) and patched the store integration service integration test to use the shared `patch_engine_manager` harness around the real Postgres engine. Validation set: `poetry run ruff check ml`, `poetry run mypy ml --strict`, `python tools/validate_fixture_plugins.py`, `ML_ENABLE_COMPONENT_FACADES=1 poetry run pytest ml/tests/e2e/test_pipeline_orchestrator_e2e.py` (fails today because `_CompatibleLegacyOrchestrator` lacks `compute_window_start_iso`/`get_health_status`; default env keeps the suite skipped until the orchestrator facade lands), and `poetry run pytest ml/tests/integration/services/test_store_integration_service.py`.
- 2025-11-10 — Retired the `ml.tests.conftest` compatibility shim by migrating the entire property suite and the remaining E2E modules onto `ml.tests.fixtures.pytest_plugins`, refreshed `ml/tests/fixtures/test_exports.py` to cover `mock_stores`, `model_factory`, and `universes`, and documented the new adoption snapshot (total plug-in declarations=119; integration=66, unit=48, property=1, orchestration=2, contracts=1, e2e=1; direct fixture imports remain 28). Backfilled plug-in declarations for the DB/store guard suites (`ml/tests/unit/common/test_db_utils.py`, `ml/tests/unit/stores/test_engine_manager_integration.py`, `ml/tests/unit/stores/test_schema_audit.py`, `ml/tests/unit/stores/services/test_cross_asset_service.py`) so they no longer rely on the removed shim. Validated with `poetry run mypy ml --strict`, `poetry ruff check ml`, `pytest ml/tests/property`, `pytest ml/tests/fixtures/test_exports.py`, `pytest ml/tests/e2e/test_data_scheduler_e2e.py`, `pytest ml/tests/e2e/test_datastore_e2e.py`, `pytest ml/tests/e2e/test_feature_store_e2e.py`, `pytest ml/tests/e2e/test_model_registry_e2e.py`, `pytest ml/tests/e2e/test_pipeline_orchestrator_e2e.py`, `pytest ml/tests/e2e/test_tft_dataset_builder_e2e.py`, `pytest ml/tests/unit/common/test_db_utils.py`, and `pytest ml/tests/integration/test_observability_tracing.py`.
- 2025-11-09 — Hardened deployment health CLI logging so failure prints/logs always carry `[✗]` + `ERROR:` context and added regression tests for log emission; validated with `pytest --cache-clear ml/tests/unit/deployment/test_check_health.py`, `poetry run ruff check ml`, and `poetry run mypy ml --strict`. Remaining risk: dataset/task telemetry fixture rollout still pending (next workstream).
- 2025-11-09 — Added the shared `tier1_symbol_loader_stub` fixture, migrated the L2/alternative task + macro refresh suites to the telemetry/orchestrator harness, and reran `pytest ml/tests/unit/tasks/test_l2_task.py ml/tests/unit/tasks/test_alternative_task.py ml/tests/unit/tasks/test_pipeline_runner.py ml/tests/unit/data/ingest/test_macro_refresh.py ml/tests/unit/data/test_dataset_build_macro.py` alongside `poetry run ruff check ml` and `poetry run mypy ml --strict`. Dataset/task shards now share deterministic universe stubs; remaining gap is propagating the same fixture into the higher-level dataset orchestration/e2e suites.
- 2025-11-09 — Completed telemetry fixture adoption for the metrics manager + schema validator suites by wiring them through `ml.tests.fixtures.pytest_plugins` with `isolated_prometheus_registry`, `mock_tracing_backend`, and `isolated_orchestrator_env`. Validated via `pytest ml/tests/unit/common/test_metrics_manager_cg.py ml/tests/unit/common/test_metrics_manager_histogram.py ml/tests/unit/common/test_metrics_manager_facade.py ml/tests/unit/stores/test_schema_validator.py`, plus `poetry run ruff check ml` and `poetry run mypy ml --strict`. Remaining telemetry gap: extend the same harness to the dataset orchestration/e2e flows before reopening guardrail/perf sweeps.
- 2025-11-09 — Propagated the telemetry/orchestrator fixtures into the standalone orchestration suites (`ml/tests/orchestration/test_run_config_loader.py`, `ml/tests/orchestration/test_pipeline_orchestrator_discovery.py`) via `ml.tests.fixtures.pytest_plugins`. Re-ran those shards with `pytest ml/tests/orchestration/test_run_config_loader.py ml/tests/orchestration/test_pipeline_orchestrator_discovery.py` plus `poetry run ruff check ml` and `poetry run mypy ml --strict`. Dataset orchestration tests now share the same isolated Prometheus/tracing harness as the unit/integration layers; next step is to finish wiring any remaining dataset E2E flows that still rely on bespoke stubs before widening the guardrail/perf runs.
- 2025-11-09 — Extended the telemetry harness to the earnings/dataset integration suites and dashboard integration tests by registering `ml.tests.fixtures.pytest_plugins` and auto-requesting `isolated_prometheus_registry`, `mock_tracing_backend`, and `isolated_orchestrator_env`. Verified with `pytest ml/tests/integration/earnings/test_tft_task_dataset.py ml/tests/integration/earnings/test_data_quality.py ml/tests/integration/earnings/test_earnings_end_to_end.py ml/tests/integration/earnings/test_earnings_store_db.py ml/tests/integration/test_dashboard_ml_integration.py`, plus `poetry run ruff check ml` and `poetry run mypy ml --strict`. Remaining dataset gap: the broader store/registry integration suites still need the shared plug-in before we rerun guardrail/perf clusters.
- 2025-11-09 — Migrated the store + registry integration suites (`ml/tests/integration/stores/test_cross_asset_service_integration.py`, `ml/tests/integration/stores/test_data_store_facade.py`, `ml/tests/integration/registry/test_data_registry_postgres_backend_smoke.py`, `ml/tests/integration/registry/test_feature_registry_postgres_update.py`, `ml/tests/integration/registry/test_model_registry_security.py`) onto `ml.tests.fixtures.pytest_plugins` with the telemetry/orchestrator fixtures. Validated each shard individually (to avoid segfaults) alongside `poetry run ruff check ml` and `poetry run mypy ml --strict`. Next action: continue rolling the plug-in across the remaining service/strategy integration suites before triggering the guardrail/perf rotations.
- 2025-11-09 — Finished wiring the shared plug-in + telemetry fixtures into all remaining integration suites (`ml/tests/integration/**`, covering stores, strategy, services, pipelines, deployment, CLI, dashboard, consumers) so every module now declares `pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)` and auto-requests `isolated_prometheus_registry`, `mock_tracing_backend`, and `isolated_orchestrator_env`. Verified the highest-risk slice with `poetry run pytest ml/tests/integration/test_stores_*` (first invocation hit the 120s command timeout; reran immediately and all 34 tests passed) and re-ran `poetry run ruff check ml` plus `poetry run mypy ml --strict`. Remaining work: march through the registry/pipeline/observability/CLI shards individually and then rerun the guardrail/perf clusters once the fixtures have soaked.
- 2025-11-09 — Verified the remaining integration shards under the shared fixtures (`poetry run pytest ml/tests/integration/registry`, `poetry run pytest ml/tests/integration/pipeline`, `poetry run pytest ml/tests/integration/test_stage2_backtest_runner.py`, `poetry run pytest ml/tests/integration/observability`, `poetry run pytest "ml/tests/integration/test_observability_e2e_integration.py ml/tests/integration/test_observability_tracing.py"`, `poetry run pytest ml/tests/integration/cli/test_streaming_persistence_worker_cli.py`, `poetry run pytest ml/tests/integration/consumers/test_streaming_persistence_integration.py`, `poetry run pytest ml/tests/integration/dashboard/test_dashboard_integration.py`, `poetry run pytest ml/tests/integration/dashboard/test_streaming_state_endpoint.py`). The tracing shard surfaced a regression where `ml.observability.tracing.get_trace_context()` ignored `ML_TRACING_ENABLED=false`; added an early `is_tracing_enabled()` guard and re-ran the shard to green. Followed up with `poetry run ruff check ml` and `poetry run mypy ml --strict` to keep guardrails enforced.
- 2025-11-09 — Re-ran the guardrail/perf clusters now that telemetry fixtures are universal: `poetry run pytest ml/tests/contracts/stores/test_store_event_contracts.py`, `poetry run pytest ml/tests/property/test_topic_scheme_parity_properties.py ml/tests/property/test_topic_scheme_parity_pairwise.py ml/tests/property/test_cascade_topic_scheme_parity.py ml/tests/property/test_message_topics_property.py`, and `poetry run pytest ml/tests/performance/test_streaming_persistence_microbench.py`. The topic parity properties exposed a stale assumption about legacy stage aliases, so the cascade property now canonicalizes stages via `to_stage_enum` before subscribing. All guardrail suites pass again; no remaining blockers for broader perf sweeps.
- 2025-11-09 — Completed the fixture adoption audit. `rg -n "ml.tests.conftest"` now only matches this plan and `ml/tests/fixtures/FIXTURE_GUIDE.md`, confirming no test modules import the legacy namespace. A Python scan counted 114 files under `ml/tests` that declare `pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)` (integration=66, unit=44, orchestration=2, contracts=1, e2e=1), and a second scan found 28 direct `from ml.tests.fixtures import ...` imports (unit=17, integration=7, property=2, metamorphic=1). Remaining e2e suites such as `ml/tests/e2e/test_pipeline_orchestrator_e2e.py:68-75` and the property package still rely on the compatibility re-export in `ml/tests/conftest.py:323-327`, so retiring that shim requires migrating those modules onto the plug-in.
- 2025-11-05 — Restored stage/source alias handling across event utilities and topic builders; validated with `poetry run pytest ml/tests -k "data_store_ingestion_other_json or store_event_contracts"` plus strict mypy/ruff.
- 2025-11-05 — Elevated DataStore alias handling and fixture hygiene; next focus is consolidating pytest fixtures under `ml/tests/fixtures` (with `ml/tests/fixtures/__init__.py` as the canonical index) and trimming `conftest.py` to a thin re-export to eliminate scope drift.
- 2025-11-05 — Post-consolidation validation: stage/topic alias contracts and DataStore routing suites pass (`pytest -k "data_store_ingestion_other_json or store_event_contracts"`, `pytest -k data_store_routing_advanced`); property/contract shards remain green under the new fixture modules.
- 2025-11-05 — Added `patch_engine_manager()` helper to `ml/tests/unit/stores/test_bus_publishing_standardization.py` so EngineManager mocking hits the shared `ml.common.db_utils` path (component + legacy stores). Confirmed the shard runs locally and appended notes here.
- 2025-11-06 — Introduced `ml.tests.fixtures.pandera` shim, migrated schema/contract suites to it, expanded `test_exports.py` coverage, and revalidated targeted Pandera contracts (observability, domain bookkeeping, event bus, watermark, databento) locally.
- 2025-11-06 — Replaced DataStore toggle context with patch-based fixture and centralized EngineManager mocking/cleanup; `pytest ml/tests/contracts/test_data_store_routing_advanced.py`, `pytest ml/tests/unit/stores/test_engine_manager_integration.py`, `pytest ml/tests/unit/stores/test_bus_publishing_standardization.py`, and `pytest ml/tests/metamorphic/test_store_time_shift_and_permutation_metamorphic.py` now pass deterministically.
- 2025-11-06 — Added observability tracing + dataset fixtures (`mock_tracing_backend`, `patch_bars_to_dataframe`) and migrated tracing/TFT builder suites to the shared helpers; verified with `pytest ml/tests/integration/test_observability_tracing.py`, `pytest ml/tests/unit/data/test_tft_builder_integration.py`, and `pytest ml/tests/integration/data/test_tft_builder_with_events.py`.
- 2025-11-06 — Introduced deterministic Prometheus registry harness (`patch_prometheus_registry`, `isolated_prometheus_registry`) and migrated streaming telemetry integration coverage to it; `pytest ml/tests/integration/training/event_driven/test_plan_to_result.py` now exercises metrics without leaking collectors.
- 2025-11-06 — Adopted dataset fixtures in earnings/TFT property suites and replaced remaining `ml.tests.conftest` imports with `ml.tests.fixtures` exports; updated fixture guide to document the new metrics helpers.
- 2025-11-06 — Dashboard telemetry suites now opt into the Prometheus registry harness (`pytest ml/tests/unit/dashboard/test_metrics_service.py`, `pytest ml/tests/unit/dashboard/test_dashboard_registries.py`, `pytest ml/dashboard/tests/test_pipelines_routes.py`).
- 2025-11-06 — Extended the Prometheus isolation harness to actor and consumer metric suites (`pytest ml/tests/unit/actors/test_signal_actor_parameterized.py`, `pytest ml/tests/unit/consumers/test_aggregator_metrics.py`).
- 2025-11-06 — Store validation Prometheus tests now rely on the shared harness (`pytest ml/tests/unit/stores/test_data_store_validation.py -k TestPrometheusMetrics`).
- 2025-11-06 — Grafana helper and metrics snapshot suites now use the registry harness (`pytest ml/tests/unit/dashboard/test_grafana.py`, `pytest ml/tests/unit/dashboard/test_metrics_snapshot.py`).
- 2025-11-07 — Migrated the monitoring collector suite to the registry harness and revalidated tracing coverage (`pytest ml/tests/unit/monitoring/collectors/test_resource_collector.py`, `pytest ml/tests/integration/test_observability_tracing.py`).
- 2025-11-07 — Confirmed remaining telemetry shards exercise the harness without collector leaks (`pytest ml/tests/unit/consumers/test_aggregator_metrics.py`, `pytest ml/tests/unit/stores/test_data_store_validation.py::TestPrometheusMetrics`, `pytest ml/tests/unit/dashboard/test_grafana.py`, `pytest ml/tests/unit/dashboard/test_metrics_snapshot.py`, `pytest ml/dashboard/tests/test_pipelines_routes.py`).
- 2025-11-07 — Added ONNX runtime harness fixtures (`mock_onnx_runtime`, `patch_onnx_runtime`) and updated security/registry suites to consume them (`pytest ml/tests/unit/common/test_security.py`, `pytest ml/tests/unit/registry/test_model_persistence.py -k "load_model"`).
- 2025-11-07 — Extended the ONNX harness to registry integration suites and validated auto-deploy coverage (`pytest ml/tests/integration/registry/test_model_registry_security.py`, `pytest ml/tests/unit/registry/test_model_registry.py::test_model_registry_register_load_deploy_lineage`).
- 2025-11-08 — Re-ran the telemetry/tracing cluster against the shared Prometheus and OTEL harnesses (`pytest ml/tests/integration/test_observability_tracing.py`, `pytest ml/dashboard/tests/test_pipelines_routes.py`, `pytest ml/tests/unit/dashboard/test_grafana.py`, `pytest ml/tests/unit/dashboard/test_metrics_snapshot.py`, `pytest ml/tests/unit/monitoring/collectors/test_resource_collector.py`, `pytest ml/tests/unit/consumers/test_aggregator_metrics.py`, `pytest ml/tests/unit/stores/test_data_store_validation.py::TestPrometheusMetrics`) to confirm deterministic state reset before tackling the remaining backlog.
- 2025-11-07 — Normalized the remaining Stage/Source emit paths (aggregator, file-backed store, data writer, legacy store watermark) and forced the dashboard + streaming loader suites to opt into `isolated_prometheus_registry`/`mock_tracing_backend`; revalidated with `pytest ml/tests/contracts/test_dataset_event_contracts.py`, `pytest ml/tests/contracts/test_data_store_routing_advanced.py`, `pytest ml/tests/unit/stores/test_data_store_emit_event.py`, `pytest ml/tests/integration/test_dashboard_ml_integration.py`, `pytest ml/tests/unit/dashboard/test_metrics_service.py`, and `pytest ml/tests/unit/training/teacher/test_streaming_loader.py`.
- 2025-11-07 — Audited the tree for `ml.tests.conftest` imports (none remain), reiterated the guidance in `ml/tests/fixtures/FIXTURE_GUIDE.md`, and reran `pytest ml/tests/fixtures/test_exports.py` to prove the fixture index stays authoritative.
- 2025-11-09 — Hardened the scheduler/ingestion + deployment shards to auto-request the telemetry/tracing fixtures and patch EngineManager before touching SQLite. Validated with `pytest ml/tests/unit/data/test_catalog_rehydrator.py`, `pytest ml/tests/unit/data/test_scheduler_lookback.py`, `pytest ml/tests/unit/data/test_scheduler_ts_extraction.py`, `pytest ml/tests/unit/data/test_scheduler_targeted.py`, `pytest ml/tests/unit/data/test_coverage_manager.py`, `pytest ml/tests/unit/data/test_fred_join_validation.py`, `pytest ml/tests/unit/deployment/test_entrypoint_pipeline.py`, and `pytest ml/tests/unit/tasks/test_pipeline_scheduler.py`; all green, remaining risk is dataset orchestrator e2e coverage still pending.
- 2025-11-09 — Wired the ingestion backfill and L2 cache unit suites to the shared pytest plug-in + telemetry fixtures so they no longer rely on `ml.tests.conftest` side effects. Reran `pytest ml/tests/unit/ingest/test_orchestrator_backfill.py ml/tests/unit/data/test_l2_cache.py` to confirm registry/tracing isolation; next step is expanding the same harness to dataset E2E builders.
- 2025-11-09 — Extended the TFT/dataset builder unit + integration suites to import via `ml.tests.fixtures.pytest_plugins` and auto-request the telemetry harness (`isolated_prometheus_registry`, `mock_tracing_backend`). Validated with `pytest ml/tests/unit/data/test_dataset_build_macro.py ml/tests/unit/data/test_dataset_pipeline_signature.py ml/tests/unit/data/test_dataset_validation.py ml/tests/unit/data/test_tft_dataset_builder_phase_one.py ml/tests/unit/data/test_tft_dataset_builder_store.py ml/tests/integration/data/test_tft_builder_with_events.py`; remaining gap is propagating the same fixtures to the dataset E2E orchestrator shards.
- 2025-11-09 — Propagated the fixture plug-in + telemetry/orchestrator harness (`isolated_prometheus_registry`, `mock_tracing_backend`, `isolated_orchestrator_env`) across the orchestration unit suite so CLI/promotions/scheduler tests no longer rely on `ml.tests.conftest`. Validated with `pytest ml/tests/unit/orchestration` (103 passed, 1 skipped). Next milestone is to extend the plug-in into the dataset E2E orchestrator flows under `ml/tests/integration/orchestration`.
- 2025-11-09 — Mirrored the telemetry/orchestrator fixtures in the orchestration integration facade suite so the component orchestrator tests inherit the same isolation guarantees (skipped when `ML_ENABLE_COMPONENT_FACADES=0`). Validated via `pytest ml/tests/integration/orchestration/test_ml_pipeline_orchestrator_facade.py` alongside `ruff`/`mypy`. Remaining gap is bringing the broader dataset E2E orchestration shards under the plug-in umbrella once the component gates are flipped on.
- 2025-11-09 — Extended the telemetry/orchestrator fixture harness into the scheduler + streaming-training integrations so Databento/FeatureStore schedulers and the plan→result pipeline all rely on `isolated_prometheus_registry`, `mock_tracing_backend`, and `isolated_orchestrator_env`. Validated with `pytest ml/tests/integration/test_scheduler_databento.py ml/tests/integration/test_scheduler_feature_store.py ml/tests/integration/training/event_driven/test_plan_to_result.py`; the FeatureStore shard now passes end-to-end once the dedicated ML Postgres service (5434) is up. Next up is wiring the remaining dataset/pipeline integration suites once their external deps are available.
- 2025-11-09 — Brought the end-to-end pipeline + ML signal integration suites onto the shared fixture plug-in so they default to telemetry/orchestrator isolation. Verified with `pytest ml/tests/integration/test_end_to_end_pipeline.py -k test_pipeline_smoke_test` (smoke path passes against the new harness) and attempted `pytest ml/tests/integration/test_ml_signal_pipeline.py -k test_ml_signals_flow_through_message_bus` (module fully deselects until Nautilus core deps are available). These suites no longer rely on `ml.tests.conftest`, and future work can focus on enabling the remaining tests once native dependencies are installed.
- 2025-11-07 — Converted `ml/dashboard/tests/test_pipelines_routes.py` to consume the shared telemetry fixtures (`isolated_prometheus_registry`, `mock_tracing_backend`) via the pytest plug-in export, removing bespoke Prometheus patching and revalidating with `pytest ml/dashboard/tests/test_pipelines_routes.py`.
- 2025-11-07 — Finished migrating DB utils to the centralized `patch_engine_manager` helper (with call capture/side-effect support) so tests no longer leak cached engines; revalidated with `pytest ml/tests/unit/common/test_db_utils.py` plus full `ruff`/`mypy`.
- 2025-11-07 — Dashboard cache/metrics suites (`pytest ml/tests/unit/dashboard/test_dashboard_events.py`, `pytest ml/tests/unit/dashboard/test_dashboard_registries.py`) now auto-load the fixture plug-in, request `mock_tracing_backend`, and isolate Prometheus collectors instead of manually resetting globals.
- 2025-11-07 — Metrics service + snapshot unit suites now use the shared telemetry harness (fixture plug-in + `mock_tracing_backend` + `isolated_prometheus_registry`), keeping counter state deterministic; validated via `pytest ml/tests/unit/dashboard/test_metrics_service.py ml/tests/unit/dashboard/test_metrics_snapshot.py`.
- 2025-11-07 — Remaining dashboard suites (store health, Grafana, control panel, strategy service, dashboard welcome/ui) now auto-load the fixture plug-in and request `mock_tracing_backend`/`isolated_prometheus_registry`; validated with `pytest ml/tests/unit/dashboard/test_store_health.py ml/tests/unit/dashboard/test_grafana.py ml/tests/unit/dashboard/test_control_simple.py ml/tests/unit/dashboard/test_strategy_service_unit.py ml/tests/unit/dashboard/test_dashboard_welcome.py ml/tests/unit/dashboard/test_dashboard_ui_template.py`.
- 2025-11-07 — Verified the ONNX/security harness across unit + integration suites (`pytest ml/tests/unit/common/test_security.py ml/tests/unit/registry/test_model_persistence.py ml/tests/integration/registry/test_model_registry_security.py ml/tests/integration/actors/test_multi_signal_actor_onnx_integration.py`); all rely on `mock_onnx_runtime` and pass with the shared fixture plug-in.
- 2025-11-08 — Migrated dataset macro/unit builds to the deterministic dataset fixtures (`SampleBarSeriesConfig` + `patch_bars_to_dataframe`), replacing bespoke builder stubs, and revalidated with `pytest -k "dataset_build_macro or tft_task"`.
- 2025-11-08 — Expanded fixture export coverage (common, dummy_model, mock_services, monitoring_collectors, streaming_events) and enforced the index via `pytest ml/tests/fixtures/test_exports.py`.
- 2025-11-08 — Orchestration scheduler suites (`pytest ml/tests/unit/orchestration/test_scheduler.py`, `pytest ml/tests/unit/tasks/test_pipeline_scheduler.py`) now rely on the `isolated_orchestrator_env` fixture + monkeypatch-based env control so ORCH_* state never leaks across shards.
- 2025-11-08 — Converted `ml.tests.fixtures` into a lazy exporter, refreshed the fixture guide with the orchestrator/env helpers, and re-ran `pytest ml/tests/fixtures/test_exports.py` to confirm deterministic ordering across the auto-discovered modules.
- 2025-11-08 — Hardened tracing unit coverage to rely on `mock_tracing_backend`, exercising the enabled path (`pytest ml/tests/unit/observability/test_tracing_unit.py`) so OTEL behaviour is validated without real collectors.
- 2025-11-08 — Hardened scheduler/ingestion coverage: `ml/tests/unit/tasks/test_pipeline_scheduler.py` now consumes `isolated_orchestrator_env`, ONNX integration tests use `mock_onnx_runtime`, and validation passed with `poetry run pytest -k "dataset_build_macro or tft_task"`, `poetry run pytest ml/tests/unit/tasks/test_pipeline_scheduler.py`, `poetry run pytest ml/tests/integration/actors/test_multi_signal_actor_onnx_integration.py`, `poetry run pytest ml/tests/fixtures/test_exports.py`, plus `poetry run ruff check ml` and `poetry run mypy ml --strict`.
- 2025-11-08 — Added advisory locks plus exception-safe partition DDL for `create_monthly_partitions` and revalidated cross-asset service + DB utility shards (`pytest ml/tests/unit/common/test_db_utils.py`, `pytest ml/tests/unit/stores/services/test_cross_asset_service.py`).
- 2025-11-08 — Patched schema auditor OID handling and aligned DataStore routing contracts with the new telemetry fixtures (`pytest ml/tests/unit/stores/test_schema_audit.py`, `pytest ml/tests/contracts/test_data_store_routing_advanced.py`).
- 2025-11-08 — Verified deployment health CLI output after logging changes (`pytest --cache-clear ml/tests/unit/deployment/test_check_health.py`).
- 2025-11-08 — Rewired the TFT dataset E2E suite to use `SampleBarSeriesConfig` via `patch_bars_to_dataframe` (with monkeypatch teardown) and re-ran `pytest --cache-clear ml/tests/e2e/test_tft_dataset_builder_e2e.py`.
- 2025-11-09 — Finished the Stage/Source normalization sweep across event emitters, store event contracts, and topic parity properties (`pytest ml/tests/unit/common/test_event_emitter.py`, `pytest ml/tests/integration/test_stores_strategy_events.py::test_strategy_store_emits_signal_events`, `pytest ml/tests/contracts/stores/test_store_event_contracts.py`, `pytest ml/tests/property/test_topic_scheme_parity_properties.py`, `pytest ml/tests/test_enum_comparison_patterns.py`).
- 2025-11-09 — Re-ran the guardrail suites that previously failed under Stage/Source drift to confirm they now pass (`pytest ml/tests/contracts/test_base_actor_initialization.py`, `pytest ml/tests/contracts/test_store_env_topic_config_contracts.py`, `pytest ml/tests/property/test_topic_scheme_parity_properties.py`).

## Current Status (2025-11-06 Morning)

Last full run (`pytest ml --maxfail=5` on 2025-11-05) surfaced **172 failing tests** across stores, registry, tracing, dashboard, dataset builders, ONNX security, and pipeline orchestration. Since then the Pandera contract cluster has been repaired—the observability, domain bookkeeping, event bus, watermark, and databento schema suites now pass locally after the new fixture shim. Remaining failures map to the other clusters below.

### Top Failure Clusters

| Cluster | Representative tests | Notes |
| --- | --- | --- |
| **Fixture gaps / Pandera Series** | `ml/tests/contracts/test_observability_pipeline_schemas.py`, `ml/tests/contracts/test_domain_bookkeeping_schemas.py` | ✅ Resolved (2025-11-06): migrated to `ml.tests.fixtures.pandera`, targeted contract shards now green. |
| **Store routing & toggles** | `ml/tests/contracts/test_data_store_routing_advanced.py::*`, `ml/tests/unit/stores/test_data_store_routing.py` | ✅ Resolved (2025-11-06): patch-based toggle preserves enum identity and restores module flags between tests. |
| **EngineManager identity & DB utils** | `ml/tests/unit/stores/test_engine_manager_integration.py`, `ml/tests/unit/common/test_db_utils.py` | New patch helper bypasses EngineManager identity checks; DB utils tests now compare against our shared MagicMock and fail. Need scoped patch strategy. |
| **Model registry / persistence** | `ml/tests/unit/registry/test_model_persistence.py`, `ml/tests/unit/registry/test_model_registry.py`, `ml/tests/integration/registry/test_model_registry_security.py` | ✅ Resolved (2025-11-07): shared ONNX runtime harness keeps tests off real sessions and restores deterministic caching behaviour. |
| **Deployment CLI** | `ml/tests/unit/deployment/test_check_health.py` | ✅ Verified (2025-11-07): CLI shard green with shared fixtures; keep monitoring for config drift. |
| **Tracing & telemetry** | `ml/tests/integration/test_observability_tracing.py::*`, dashboard telemetry tests | New fixtures bypass OpenTelemetry mocks; context injection is no-op. |
| **Dashboard API** | `ml/dashboard/tests/test_pipelines_routes.py::*`, `ml/tests/unit/dashboard/test_metrics_service.py::*` | Service mocks not injected → HTTP 503/404 responses and zeroed metrics. |
| **Dataset builder & ingestion** | `ml/tests/unit/data/test_dataset_build_macro.py::*`, `ml/tests/unit/tasks/test_pipeline_scheduler.py::*` | ✅ 2025-11-08 — Scheduler suite now uses `isolated_orchestrator_env` + ONNX harness, and `pytest -k "dataset_build_macro or tft_task"` / `pytest ml/tests/unit/tasks/test_pipeline_scheduler.py` both pass. |
| **Security / ONNX** | `ml/tests/unit/common/test_security.py::*` | With central patching removed, tests load real ONNX artifacts and fail integrity checks. |

### Status Snapshot

| Cluster | Representative tests | Status |
| --- | --- | --- |
| Fixture gaps / Pandera Series | `ml/tests/contracts/test_observability_pipeline_schemas.py::TestObservabilityPipelineIntegrationContracts::test_end_to_end_observability_contract`<br>`ml/tests/contracts/test_domain_bookkeeping_schemas.py::TestEventMessageContracts::test_event_message_schema_validation` | ✅ (2025-11-06) |
| Store routing & toggles | `ml/tests/contracts/test_data_store_routing_advanced.py::TestDataStoreRouting::test_features_routing_contract[component]` | ✅ (2025-11-06) |
| EngineManager identity | `ml/tests/unit/stores/test_engine_manager_integration.py::TestEngineManagerIntegration::test_store_engines_are_identical_for_same_url` | ⚠️ Monitoring – centralized fixture in place; ensure downstream shards adopt it |
| Model registry / persistence | `ml/tests/unit/registry/test_model_persistence.py::test_load_model_with_caching` | ✅ (2025-11-07) |
| Deployment health CLI | `ml/tests/unit/deployment/test_check_health.py::TestMainFunction::test_main_all_healthy` | ✅ (2025-11-07) |
| Tracing & telemetry | `ml/tests/integration/test_observability_tracing.py::TestTracingWithOpenTelemetry::test_trace_context_with_mocked_otel` | ❌ |
| Dataset builder & ingestion | `ml/tests/unit/data/test_dataset_build_macro.py::test_build_tft_dataset_invokes_macro_refresh` | ✅ 2025-11-08 — Dataset macros + pipeline scheduler now run on shared fixtures; targeted shards stay green. |

## Fixture Migration Audit

Latest verification (2025-11-10 15:42 UTC):

- `rg -n "ml.tests.conftest"` now only matches this plan and `ml/tests/fixtures/FIXTURE_GUIDE.md`; the compatibility re-export block has been deleted from `ml/tests/conftest.py`, so canonical fixtures must be imported via `ml.tests.fixtures`.
- Plug-in adoption snapshot (see command in *Investigation Artifacts*): 119 files under `ml/tests` declare `pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)` (integration=66, unit=48, property=1, orchestration=2, contracts=1, e2e=1). Additional adopters reside outside this tree (e.g., `ml/dashboard/tests/test_pipelines_routes.py:24-37`), so telemetry isolation is consistent across integration + dashboard suites.
- Direct `from ml.tests.fixtures import ...` imports remain in 28 modules (unit=17, integration=7, property=2, metamorphic=1). These are exclusively type-focused helpers now that the property package is wired through its own `__init__`.
- End-to-end packages inherit the plug-in via `ml/tests/e2e/__init__.py`, so every module now runs against the shared fixtures without relying on `conftest`.
- The export guard has been expanded to cover `mock_stores`, `model_factory`, and `universes`; the audit script reports zero missing modules.
- Stage/Source comparisons remain inconsistent: `rg "STAGE\." -n ml/tests` and `rg "Source\." -n ml/tests` highlight dashboards, store contracts, and enum comparison suites still hardcoding legacy string formats.
- The Pandera shim (`Series` alias) now lives in `ml.tests.fixtures.pandera`; remaining work is migrating legacy tests off direct `pandera.typing` imports.
- Store toggles: `_component_data_store_context` has been refactored to patch module flags without reloads; tests consume deterministic component/legacy switches via fixtures.
- Engine patching: Centralized `patch_engine_manager`/`mock_engine_manager` fixtures provide MagicMock-backed engines with cache disposal; update bespoke monkeypatches to leverage them.
- Monitoring/telemetry fixtures (`mock_prometheus_when_unavailable`, metric gauges) are available, but many tests are not consuming them under the new import path. We need to confirm `ml/tests/fixtures/monitoring_collectors.py` is included in `__all__` and referenced by failing telemetry tests.
- Dataset builders: `patch_bars_to_dataframe` now supplies deterministic bars and scheduler suites rely on `isolated_orchestrator_env`; continue migrating remaining ingestion/macros tests away from bespoke stubs.

### Outstanding Fixture Tasks

1. ✅ **Create Pandera shim module** in `ml/tests/fixtures/` (2025-11-06) and migrate schema tests; focus now shifts to telemetry and dataset fixtures.
2. ✅ **Replace `_component_data_store_context`** with patch-based toggles that preserve Enum identity and return typed store classes. (2025-11-06)
3. ✅ **Centralize EngineManager patching strategy** with reusable fixtures (`patch_engine_manager`, `mock_engine_manager`, `real_engine_manager`) to control caching during tests. (2025-11-06)
4. ✅ **Migrate property + remaining e2e suites** onto `pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)` so we can drop the compatibility re-export in `ml/tests/conftest.py` (2025-11-10).
5. ✅ **Extend `ml/tests/fixtures/test_exports.py` coverage** to include `mock_stores`, `model_factory`, and `universes`, matching the audit output (2025-11-10).
6. ✅ **Document + enforce plug-in adoption metrics** (2025-11-11) via `tools/validate_fixture_plugins.py` + `make validate-fixtures`, which now blocks CI when plug-in counts drift or fixture exports fall out of sync.
7. **Author migration checklist** so individual test files move to the new fixtures in an ordered sequence (DB → stores → telemetry → dataset builders) without reintroducing reloads.
8. **Eliminate direct fixture imports** (`from ml.tests.fixtures import ...`) from the remaining unit/property/integration/metamorphic suites so the plug-in is the only entry point (counts below).
9. **Extract bespoke fixtures** (local Prometheus harnesses, SampleBarSeries configs, TestDatabase shims, etc.) into `ml/tests/fixtures/**` so every test simply requests the canonical fixture.

**Compatibility note:** the legacy re-export in `ml/tests/conftest.py` has been removed. Any suite that needs fixtures must import them from `ml.tests.fixtures` (or rely on the pytest plug-in) instead of depending on implicit globals.

### Fixture Adoption Snapshot (2025-11-11)

| Package | Direct fixture imports | Representative files | Action |
| --- | ---:| --- | --- |
| `unit` | 0 | — | ✅ Completed (uses fixture injection for SampleBarSeries/TestDatabase). |
| `integration` | 0 | — | ✅ Completed (dataset + store suites now rely on fixture parameters). |
| `property` | 0 | — | ✅ Completed (property suites now use fixture injection only). |
| `metamorphic` | 0 | — | ✅ Completed (now relies solely on fixture parameters). |
| `fixtures` | 1 | `fixtures/test_exports.py` | Expected – this module enforces the canonical export list. |

Counts come from `python tools/validate_fixture_plugins.py`, which now runs as part of `make validate-fixtures`.

### Fixture Completion Criteria

- Every `ml/tests/**` package sets `pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)` (already satisfied) and test modules do **not** redeclare it.
- No test module imports fixtures directly from `ml.tests.fixtures` (the lone exception is `ml/tests/fixtures/test_exports.py`).
- Reusable helpers/fixtures live under `ml/tests/fixtures/**`, export names via `__all__`, and remain covered by `ml/tests/fixtures/test_exports.py`.
- Tests request fixtures via function/class parameters or `pytest.mark.usefixtures`, never by import.
- `make validate-fixtures` (plus `poetry run mypy ml --strict`, `poetry ruff check ml`, targeted pytest shards, coverage gates, and `poetry run pytest ml` as needed) runs before merging.

## Per-Package Fixture Checklists

### Data & Orchestrator

- [x] Promote the dataset macro helper into the shared `patch_dataset_bars` fixture; verify `pytest ml/tests/unit/data/test_dataset_build_macro.py` on every edit.
- [x] Update dataset orchestrator/unit suites (`ml/tests/integration/data/test_tft_builder_with_events.py`, `ml/tests/integration/earnings/test_tft_task_dataset.py`) to consume `patch_dataset_bars` instead of bespoke monkeypatch helpers.
- [x] Ensure ingestion/E2E flows that still instantiate `SampleBarSeriesConfig` directly (`ml/tests/e2e/test_tft_dataset_builder_e2e.py`, scheduler CLI shards) switch to the fixture factory so configs remain centralized.
- [x] Extend the fixture guide with concrete orchestration examples (pipeline scheduler, earnings dataset) now that the suites have migrated (see the new sections added on 2025-11-12).

### Security & ONNX

- [x] Keep `mock_onnx_runtime`/`patch_onnx_runtime` fixtures wired into security + registry unit/integration suites (already validated via `pytest -k secure_onnx_load`).
- [x] Wire the ML signal pipeline and strategy backtest integration suites to `mock_onnx_runtime` + `onnx_session_stub_factory` so they no longer depend on real ONNX exports.
- [x] Migrate streaming actor integrations (`ml/tests/integration/actors/test_multi_signal_actor_onnx_integration.py`, `ml/tests/integration/actors/test_actor_circuit_breaker_integration.py`) onto the ONNX harness instead of manual `ml._imports` monkeypatches.
- [x] Ensure dataset/strategy E2E suites that load ONNX artifacts request the fixtures so coverage does not hit real runtimes (`ml/tests/e2e/test_pipeline_orchestrator_e2e.py` now wires the harness, though the suite stays skipped until the component flag is enabled).
- [x] Add harness usage examples + fallback metrics expectations to `ml/tests/fixtures/FIXTURE_GUIDE.md` once adoption is complete.

### Stores & Database

- [x] Adopt `patch_engine_manager`/`mock_engine_manager` across db utils, catalog rehydrator, bus publishing, and metamorphic store suites.
- [x] Convert the remaining store integration suites (`ml/tests/integration/stores/test_cross_asset_service_integration.py`, `ml/tests/integration/services/test_store_integration_service.py`, feature parity/integration scheduler suites) to rely exclusively on the shared fixtures (no per-file `pytest_plugins`, no direct `TestDatabase` imports). Observability DB migrations/partitioning suites still need the same treatment.
- [x] Migrate observability DB migration/partitioning suites to the shared database fixtures (`test_db_migrations.py`, `test_db_partitioning.py`) so they stop instantiating bespoke engines.
- [ ] Backfill `real_engine_manager` smoke coverage for the store migration path to guarantee caches are flushed between parametrized tests.
- [x] Document EngineManager fixture expectations (recording calls, side effects) in the Fixture Guide’s database section for easy onboarding (added example snippet on 2025-11-12).

### Observability & Dashboard

- [x] Document `isolated_prometheus_registry`/`metric_name_manager` usage and require telemetry suites to opt in (dashboard + monitoring unit tests already aligned).
- [ ] Audit remaining streaming/actor/CLI suites for manual Prometheus or OTEL shims and migrate them to `mock_tracing_backend` + `isolated_prometheus_registry` (streaming persistence CLI/integration/performance suites updated 2025-11-12; still need to sweep actor/observability DB suites).
- [ ] Add structured logging/metrics fixture coverage for new dataset orchestrator workflows once their telemetry hooks are fixture-driven.
- [ ] Tie the per-package adoption status to `tools/validate_fixture_plugins.py` output so deviations fail fast (script update pending).

### Test Author Workflow

1. Read `ml/tests/fixtures/__init__.py` (or `ml/tests/fixtures/FIXTURE_GUIDE.md`) to discover existing fixtures.
2. Declare the fixture name in your test signature or attach it via `pytest.mark.usefixtures`. Avoid direct imports.
3. If the fixture you need does not exist, add it under `ml/tests/fixtures/`, update the module `__all__`, ensure `ml/tests/fixtures/test_exports.py` passes, and capture the change in this plan.
4. Run `make validate-fixtures && poetry run mypy ml --strict && poetry ruff check ml`, plus the affected pytest shards and coverage report, before requesting review. Full-suite validation remains `poetry run pytest ml` whenever fixtures touch hot paths or shared infrastructure.

### DataStore Consumer Audit (2025-11-11)

- `rg -l "DataStore(" ml/tests/property ml/tests/integration` currently returns three files. `ml/tests/property/test_data_store_no_duplicate_bus.py` already forces the component path through the `component_data_store_factory` autouse fixture that wraps every instantiation.
- `ml/tests/integration/test_scheduler_databento.py` is the only integration module that directly overwrites a scheduler’s `_data_store`. It wires the shared plug-in, runs against the `test_database` PostgreSQL fixture, and patches `_data_store = DataStore(connection_string=test_database.connection_string)`, so it always exercises the component implementation while still hitting a real database.
- `ml/tests/integration/stores/test_data_store_facade.py` toggles `ML_USE_COMPONENT_DATA_STORE` explicitly around every instantiation so the same module can assert both legacy and component parity. Because the shared plug-in is now registered at the package level, the suite automatically gets the `component_data_store_factory` context helpers any time we need to pin an implementation.

## Failure Hotspot Details

- **EngineManager identity**
  - `ml/tests/unit/stores/test_engine_manager_integration.py::test_store_engines_are_identical_for_same_url`
  - `ml/tests/unit/common/test_db_utils.py::*`
  Centralized fixtures now dispose caches after patched runs; remaining work is migrating bespoke monkeypatches in DB utils and registry suites to use `patch_engine_manager` or `real_engine_manager`.

- **Series shim (resolved 2025-11-06)**
  - Targeted contract suites (observability, domain bookkeeping, event bus, watermarks, databento) now pass with `ml.tests.fixtures.pandera`.
  Continue migrating remaining tests off legacy `conftest` imports to keep behaviour consistent.

- **Stage/Source mismatches**
  - Example: `ml/tests/contracts/stores/test_store_event_contracts.py::test_strategy_store_registry_event_contracts` expecting Source.HISTORICAL but receiving live/backfill. Confirm event emitter uses `to_source_enum` mapping and ensure tests import new helper.

- **ONNX caching**
  - Shared `mock_onnx_runtime` harness now patches `ml._imports`, `ml.common.security`, and registry layers so unit/integration shards exercise mocks instead of real ORT sessions.

- **Deployment health CLI**
  - Tests expect `main()` to exit 0/1 deterministically. After refactor, prints changed (no `[✗]` on success) and unpatched `check_service_health` usage. Need to verify our CLI patch ensures each mock path returns the expected tuple.

- **Tracing & telemetry**
  - `ml/tests/integration/test_observability_tracing.py::*` assert `start_as_current_span` usage and `trace_context` injection. Fixtures currently bypass OTEL stub—port the old `mock_otel` helper into new fixtures.

- **Dashboard endpoints**
  - `ml/dashboard/tests/test_pipelines_routes.py` now hits real services returning 503/404. Reconnect with `mock_services.py` fixtures for pipeline orchestrator and dataset builder.

## Investigation Artifacts

```
# Confirm no direct ml.tests.conftest imports remain
rg -n "ml\.tests\.conftest"

# Count plug-in adoption (per top-level package under ml/tests)
python - <<'PY'
import pathlib, re, collections
root = pathlib.Path("ml/tests")
pattern = re.compile(r'pytest_plugins\s*=\s*\("ml\.tests\.fixtures\.pytest_plugins",\)')
counts = collections.Counter()
total = 0
for path in root.rglob("*.py"):
    text = path.read_text()
    if pattern.search(text):
        rel = path.relative_to(root)
        counts[rel.parts[0]] += 1
        total += 1
print("total", total)
for key in sorted(counts):
    print(key, counts[key])
PY

# Count direct `from ml.tests.fixtures import ...` usage
python - <<'PY'
import pathlib, re, collections
root = pathlib.Path("ml/tests")
pattern = re.compile(r'from\s+ml\.tests\.fixtures\s+import\s+')
counts = collections.Counter()
total = 0
for path in root.rglob("*.py"):
    text = path.read_text()
    if pattern.search(text):
        rel = path.relative_to(root)
        counts[rel.parts[0]] += 1
        total += 1
print("total", total)
for key in sorted(counts):
    print(key, counts[key])
PY

# Identify fixture submodules missing from test_exports coverage
python - <<'PY'
from pathlib import Path
from pkgutil import iter_modules
import ml.tests.fixtures as fixtures
fixture_dir = Path(fixtures.__file__).resolve().parent
all_modules = {
    m.name
    for m in iter_modules([str(fixture_dir)])
    if m.name not in {"__init__", "pytest_plugins"}
    and not m.name.startswith(("__", "test_"))
}
covered = {
    "common",
    "database_fixtures",
    "datasets",
    "dummy_model",
    "integration",
    "mock_services",
    "monitoring_collectors",
    "observability",
    "pandera",
    "runtime",
    "security",
    "stores",
    "streaming_events",
}
print("all modules:", sorted(all_modules))
print("covered   :", sorted(covered))
print("missing   :", sorted(all_modules - covered))
PY

# Pandera Series usage (requires shim)
rg "Series" ml/tests -n

# Stage alias still used in dashboard tests
rg "STAGE\." -n ml/tests
```

These commands surface the remaining mismatches (plug-in gaps, export drift, enum aliases) that would otherwise reintroduce fixture debt.

## State Pollution Vectors

- **EngineManager cache** – Use the new `mock_engine_manager` / `real_engine_manager` fixtures to avoid stale mocks; audit suites with manual monkeypatching (`ml/tests/unit/common/test_db_utils.py`) and migrate them accordingly.
- **Store toggle globals** – Resolved via patch-based context; ensure downstream fixtures import from `ml.tests.fixtures` rather than legacy helpers to keep flags in sync.
- **Pandera + dataset builders** – Series aliases now flow through `ml.tests.fixtures.pandera`; remaining follow-up is supplying deterministic dataset macros to replace legacy `conftest` helpers.
- **Telemetry mocks** – `mock_prometheus_when_unavailable` is optional. Only a handful of suites (`ml/tests/contracts/test_fixture_contracts.py`, property tests) request it; others now hit real collectors and leave metrics behind. Similarly, tracing suites expect OTEL stubs that no longer exist.
- **Environment mutations** – Tests toggle env vars like `ML_USE_COMPONENT_FEATURE_STORE`, `ORCH_CONFIG`, `ML_REGISTRY_PATH` without guaranteed teardown. Need context managers or fixtures that capture & restore environment.
- **Dataset factories** – Builders in `ml/tests/fixtures/integration.py` provide deterministic data, but many failing suites (dataset macro, ingestion flows) still instantiate real loaders and hit empty datasets.

Mitigation plan: introduce scope-aware fixtures that enforce cleanup (EngineManager, telemetry, env vars), restore Pandera shims and dataset helpers, and eliminate module reload toggles.

## Action Plan (Pre-Implementation)

1. **Document fixture status & migration strategy** (this section).
2. **Design fixture modules** to cover:
   - OTEL/telemetry mocks with deterministic cleanup.
   - Dataset builder stubs supplying non-empty datasets.
   - Environment restoration helpers for config-driven toggles.
3. **Prioritize failing shard recovery**:
   1. `ml/tests/unit/common/test_db_utils.py` (migrate to shared EngineManager fixtures)
   2. `ml/tests/unit/registry/test_model_persistence.py` + integration security suite
   3. `ml/tests/unit/deployment/test_check_health.py`
   4. Pandera schema suites (`ml/tests/contracts/**/*schema*.py`)
   5. Observability tracing + dashboard routes
4. **Only after fixtures stabilized**, finish migrating the property + E2E packages onto the shared plug-in so the legacy `conftest` re-export can be deleted.

## Updated Validation Checklist

- Static:
  - `poetry run ruff check ml`
  - `poetry run mypy ml --strict`
- Targeted shards to keep green during remediation:
  - `poetry run pytest ml/tests/unit/stores/test_bus_publishing_standardization.py`
  - `poetry run pytest ml/tests/unit/common/test_db_utils.py`
  - `poetry run pytest ml/tests/unit/registry/test_model_persistence.py::test_load_model_with_caching`
  - `poetry run pytest ml/tests/unit/deployment/test_check_health.py`
  - `poetry run pytest ml/tests/unit/registry/test_model_registry.py::test_model_registry_register_load_deploy_lineage`
- Fixture-focused suites (run after each migration batch):
  - `poetry run pytest ml/tests/contracts/test_data_store_routing_advanced.py`
  - `poetry run pytest ml/tests/contracts/test_dataset_event_contracts.py`
  - `poetry run pytest ml/tests/contracts/test_observability_pipeline_schemas.py`
- Regression sweep once clusters are green:
  - `poetry run pytest ml --maxfail=5`

## Risks & Mitigations

- **Enum identity resets**: Avoid module reload toggles; prefer environment flags + dependency injection.
- **Unmocked external services**: Restore dataset/telemetry mocks before re-running integration suites to prevent hitting real services or empty datasets.
- **ORT session creation**: Centralize ONNX imports through `_imports.py` and provide fixtures that short-circuit to MagicMock sessions.
- **Test drift**: Update documentation and fixture guide with every migration to keep future contributors aligned.

## Next Steps

1. Finalize remaining fixture modules (telemetry/OTEL mocks, dataset helpers, env restore).
2. Automate the plug-in/export guard (script or lint) so new suites under `ml/tests/**` cannot skip `ml.tests.fixtures.pytest_plugins` or fall out of sync with `ml/tests/fixtures/test_exports.py`.
3. Keep registry auto-deploy, deployment health CLI, and model persistence shards in the regression rotation; ONNX harness adoption (2025-11-07) restored them to green, so treat them as guardrails for future changes.
4. Re-run targeted shards and update this plan with new checkpoints.

## Component Map & Investigation Strategy
| Concern | Primary Modules | Existing Utilities to Reuse |
| --- | --- | --- |
| Stage/topic normalization | `ml/config/events.py`, `ml/common/message_topics.py`, `ml/stores/data_store.py`, `ml/stores/data_writer.py` | `ml.common.events_util.to_stage_enum`, `build_topic_for_stage` |
| DataStore fixture routing | `ml/tests/conftest.py`, `ml/tests/fixtures/mock_stores.py`, contract suites | New patch-based toggle (`component_data_store_factory`), `datastore_module` context |
| Pandera schema typing | `ml/tests/contracts/**`, `ml/tests/property/**` | Historic `globals()["Series"]` shim in observability contracts, utility patterns in `00625ff84` |
| CLI validation | `ml/scripts/validate_wave.py`, `ml/training/event_driven/guardrails/validation_bundle.py` | Original `run_command` wrapper (`ml.tests.fixtures.subprocess` references) |
| Engine manager & DB utils | `ml/common/db_utils.py`, `ml/core/db_engine.py`, tests under `ml/tests/unit/common/test_db_utils.py` | `EngineManager.get_engine` cache semantics, sanitized logging helpers |
| ONNX security | `ml/common/security.py`, registry loaders, ONNX tests | `_imports` guard patterns, prior mocks in `ml/tests/unit/common/test_security.py` |
| Observability fixtures | `ml/tests/fixtures/monitoring_collectors.py`, property tests | Opt-in fixture patterns from earlier commits (pre-session scope) |
| Dashboard & API mocks | `ml/dashboard/tests/**`, metrics services | `TestMetricsService` mocks, `metrics_bootstrap` utilities |

### Execution Backlog

| Task | Details | Status |
| --- | --- | --- |
| Create Pandera shim module | Add `ml/tests/fixtures/pandera.py` exporting `Series` alias, register via `__all__`, update schema tests | ✅ 2025-11-06 |
| Replace `_component_data_store_context` | Swap module reload helper for env/patch toggles with cleanup | ✅ 2025-11-06 |
| Engineer EngineManager patch fixture | Provide fixture to choose real vs mocked engine and dispose cache after each use | ✅ 2025-11-06 |
| Security / ONNX harness | Centralize ONNX Runtime mocking via `mock_onnx_runtime` / `patch_onnx_runtime` | ✅ 2025-11-07 |
| Telemetry & OTEL fixtures | Reintroduce Prometheus and tracing mocks with deterministic cleanup | ✅ 2025-11-08 — Verified shared tracing + Prometheus harness adoption across telemetry, dashboard, collector, consumer, and store shards (`pytest ml/tests/integration/test_observability_tracing.py`, `pytest ml/dashboard/tests/test_pipelines_routes.py`, `pytest ml/tests/unit/dashboard/test_grafana.py`, `pytest ml/tests/unit/dashboard/test_metrics_snapshot.py`, `pytest ml/tests/unit/monitoring/collectors/test_resource_collector.py`, `pytest ml/tests/unit/consumers/test_aggregator_metrics.py`, `pytest ml/tests/unit/stores/test_data_store_validation.py::TestPrometheusMetrics`) |
| Dataset builder helpers | Provide reusable dataset fixtures returning non-empty data | ✅ 2025-11-06 |
| Expand `test_exports.py` coverage | Ensure all fixture modules define `__all__` and are validated | ✅ 2025-11-08 — Added common/dummy_model/mock_services/monitoring_collectors/streaming_events and guarded with `pytest ml/tests/fixtures/test_exports.py` |

#### Dependencies

| Blocked Item | Depends On |
| --- | --- |
| Restore `secure_onnx_load` shard | ✅ Completed 2025-11-07 with shared ONNX runtime harness; security and registry suites now green |
| Fix routing contract suite | ✅ Toggle refactor complete; pending dataset builder helpers |
| Dashboard telemetry suites | ✅ Dashboard metrics tests now use the Prometheus registry harness; monitor remaining suites before widening coverage |

## Workstreams & Sequencing

1. **Establish Safe Baseline**
   - Create recovery branch from current head.
   - Extract reference behaviour by diffing against last stable commit (`f925feba2`) for enums, fixtures, and CLI helpers.
   - Document notable divergences in change log (append to this plan as progress notes).

2. **Repair Shared Infrastructure (High-Fan-Out Fixes)**
   1. **Stage/Source compatibility**
      - Audit `Stage` enum changes; reintroduce alias handling via `_missing_` or mapping in `to_stage_enum`.
      - Ensure `map_stage_to_topic_segments` covers `MODEL_INFERRED` and other legacy names.
      - Tests: `pytest -k "data_store_ingestion_other_json or store_event_contracts"`.
   2. **DataStore fixture toggles**
      - Restore deterministic injection of component/legacy implementations without reloading modules (preserve enum identity).
      - Verify contract suite instantiates concrete classes (no `Any` fallbacks).
      - Tests: `pytest -k data_store_routing_advanced`.
   3. **Pandera Series exports**
      - Re-apply `globals()["Series"]` shim and ensure imports guarded by `HAS_PANDERA`.
      - Tests: `pytest -k "Series is not defined"`.
4. **Prometheus/Hypothesis fixtures**
      - ✅ Added `patch_prometheus_registry`/`isolated_prometheus_registry` for deterministic collector cleanup.
      - ✅ Dashboard metrics suites now rely on the harness; review remaining Hypothesis health-check suppressions separately.
      - Tests: `pytest ml/tests/property/test_fixture_properties.py`, `pytest ml/tests/unit/dashboard/test_metrics_service.py`.
5. **CLI `validate_wave` wrapper**
      - Re-export `run_command` to satisfy test patches; ensure script remains thin adapter per CLI rules.
      - Tests: `pytest ml/tests/unit/scripts/test_validate_wave.py`.

3. **Subsystem-Specific Remediation**
   - **Engine/Database mocks**: adjust fixture factories to respect monkeypatched `EngineManager`; ensure caches cleared between tests.
     - Tests: `pytest -k test_db_utils`.
   - **Security / ONNX**: allow tests to patch `_imports.ort` before session creation; avoid instantiating real sessions when mocked.
     - Tests: `pytest -k secure_onnx_load`.
   - **Observability tracing/dashboards**: verify guardrail changes didn’t bypass metrics bootstrapping; re-align mocks with protocols using the new Prometheus registry harness.
     - Tests: `pytest -k "dashboard or tracing"`.
   - **Dataset builders & macro tasks**: keep dataset macros and the pipeline scheduler on deterministic fixtures (`patch_bars_to_dataframe`, `isolated_orchestrator_env`) while extending the pattern to remaining ingestion flows.
     - Tests: `pytest -k "dataset_build_macro or tft_task"`, `pytest ml/tests/unit/tasks/test_pipeline_scheduler.py`.

4. **Validation & Regression Safety**
   - Run static checks (`mypy`, `ruff`).
   - Execute focused pytest shards for each repaired cluster.
   - Finish with `poetry run pytest ml -m "not slow"` (or targeted selection) once failure count is near-zero.
   - Collect coverage via `coverage run -m pytest <targets>` + `coverage report`.
   - Update `ml/docs/ops` telemetry if schema or manifest behaviour changes.

5. **Documentation & Hand-off**
   - Update this plan with completed steps, residual risks, and follow-up items.
   - If new fixtures or helpers introduced, document in appropriate `ml/docs/development` appendix.

## Tooling & Verification Commands

```bash
# Static analysis
poetry run ruff check ml
poetry run mypy ml --strict

# Focused pytest shards (examples)
poetry run pytest -k "data_store_routing_advanced"
poetry run pytest ml/tests/unit/scripts/test_validate_wave.py
poetry run pytest -k secure_onnx_load
poetry run pytest ml/tests/property/test_fixture_properties.py

# Coverage
coverage run -m pytest ml/tests/unit
coverage report
```

## Risk Register

- **Enum/topic mismap**: impacts store events, dashboards, registry watermarks. Mitigate by adding exhaustive alias handling and regression tests.
- **Fixture scope drift**: session-level caches can leak state across Hypothesis tests; convert to function scope or provide deterministic reset hooks.
- **Real service calls**: ensure engine/HTTP/Docker interactions remain mocked to avoid CI flakiness.
- **Performance regressions**: re-run microbenchmarks (`pytest -q ml/tests/performance -k microbench --benchmark-only`) after hot-path changes.

## Next Checkpoints

1. ✅ **Stage/topic alias verification** — `pytest -k "data_store_ingestion_other_json or store_event_contracts"` passes on the consolidated fixture stack.
2. ✅ **DataStore toggle stability** — `pytest -k data_store_routing_advanced` green for both component and legacy paths.
3. ✅ **Address Pandera/Prometheus fixture regressions** — property/schema shards (`pytest -k "pandera or schema"`, `pytest ml/tests/property/test_fixture_properties.py`) pass with deterministic fixtures and documented Hypothesis suppression.
4. ✅ **Repair CLI + security harnesses** — `pytest ml/tests/unit/scripts/test_validate_wave.py` and `pytest -k secure_onnx_load` pass after re-exporting `run_command` and re-synchronising ONNX guards.
5. ✅ **Consolidate pytest fixtures under `ml/tests/fixtures`** — database, store, and runtime fixtures segmented; docs updated; export lint guard added.
6. **Iterate on remaining domain-specific failures**, updating this plan after each milestone.

Progress updates should capture:

- Commit/branch references for each fix set.
- Tests executed with timestamps and results (pass/fail).
- Any deviations from guardrails with documented approvals.
