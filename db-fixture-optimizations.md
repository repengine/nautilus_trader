## DB Fixture Optimization Plan

### Current situation
- Postgres test DB runs on port 5434 with per-test locking/cleanup.
- Many integration tests spend 20–23s in setup/teardown due to repeated schema bootstrap/migrations and EngineManager disposal.
- Fixtures live in `ml/tests/fixtures/database_fixtures.py`; `FIXTURE_GUIDE.md` documents usage and discourages pollution.

### Goals
- Cut per-test setup time without introducing state pollution.
- Keep isolation guarantees: every test gets its own writable schema/db.
- Keep writers aligned with `FIXTURE_GUIDE.md` patterns and `AGENTS.md`.

### Approach
1) **Session-scoped template DB (read-only)**
   - On first use, create/migrate a “template” schema/database on the existing Postgres instance (5434) using the existing bootstrap/migration helpers.
   - Mark the template immutable: no test writes directly to it.
   - Cache the template metadata so it is created once per test session.

2) **Per-test clones**
   - For each DB-using test, create an isolated schema/db by cloning the template (e.g., `CREATE SCHEMA test_<uuid> AUTHORIZATION ...; SET search_path TO test_<uuid>, public;` or `CREATE DATABASE ... TEMPLATE template_db` if available).
   - Point EngineManager/connection strings at the cloned schema/db for the duration of the test.
   - On teardown, drop the schema/db and dispose EngineManager caches.
   - Keep the existing file lock (`acquire_db_lock`) to serialize schema creation when needed.

3) **Fixture surface**
   - Add fixtures to `ml/tests/fixtures/database_fixtures.py`:
     - `template_database` (session, read-only)
     - `cloned_test_database` (function-scoped; yields connection string/engine scoped to a fresh schema/db, cleans up afterward)
   - Wire `patch_engine_manager` so users can opt-in seamlessly; provide a helper to return an EngineManager-bound clone without manual wiring.

4) **Documentation**
   - Update `ml/tests/fixtures/FIXTURE_GUIDE.md`:
     - Add a “Performance without pollution” section.
     - State that the 5434 Postgres instance hosts the session template; tests must never mutate it.
     - Instruct to use `cloned_test_database`/`fresh_store_bundle` rather than touching the template.
     - For artifacts, reiterate the pattern: session-scoped creation is read-only; copy into `tmp_path` before mutation.
   - Consider a short ADR noting the template/clone strategy and isolation rules.

### Risk controls
- Template marked read-only; clones are isolated per test and dropped on teardown.
- Locks remain in place to avoid concurrent template rebuild.
- EngineManager disposed after each cloned test to prevent connection leakage.

### Next steps
1) ✅ Implement `template_database` and `cloned_test_database` fixtures with schema/db cloning and cleanup (template auto-created if missing on 5434).
2) ✅ Update `FIXTURE_GUIDE.md` with the new fixtures and immutability/isolation rules.
3) Patch the slowest integration suites to consume `cloned_test_database` (or `fresh_store_bundle` wired to it) and measure runtime impact. (Done: `ml/tests/integration/test_stores_strategy_reads.py`; pending: remaining list below.)
4) Validate: `poetry run mypy ml --strict`, `poetry ruff check ml`, targeted pytest for affected suites.***

### Targeted suites/files to migrate to `cloned_test_database`
- Integration/performance suites (progress):
  - ✅ `ml/tests/integration/test_stores_strategy_reads.py`
  - ✅ `ml/tests/integration/test_stores_model_reads.py`
  - ✅ `ml/tests/integration/registry/test_data_registry_postgres_backend_smoke.py`
  - ✅ `ml/tests/integration/test_stores_upsert_dedup_update.py`
  - ✅ `ml/tests/integration/test_feature_store_integration.py`
  - ✅ `ml/tests/integration/test_stores_strategy_performance_agg.py`
  - ✅ `ml/tests/integration/test_stores_strategy_basic.py`
  - ✅ `ml/tests/integration/test_stores_model_basic.py`
  - ✅ `ml/tests/integration/test_stores_partition_manager.py`
  - ✅ `ml/tests/integration/test_stores_strategy_events.py`
  - ✅ `ml/tests/integration/test_strategy_store_publishing_modes_small.py`
  - ✅ `ml/tests/integration/registry/test_feature_registry_postgres_update.py`
  - ✅ `ml/tests/integration/test_postgres_integration.py`
  - ✅ `ml/tests/integration/test_scheduler_feature_store.py`
  - ✅ `ml/tests/integration/test_scheduler_databento.py`
  - ✅ `ml/tests/integration/test_stores_strategy_flush_timer.py`
  - ✅ `ml/tests/integration/stores/test_cross_asset_service_integration.py`
  - ✅ `ml/tests/integration/test_feature_parity.py`
  - ✅ `ml/tests/integration/observability/test_db_partitioning.py`
  - ✅ `ml/tests/integration/observability/test_db_migrations.py`
  - ✅ `ml/tests/integration/test_end_to_end_pipeline.py`
  - ✅ `ml/tests/integration/orchestration/test_facade_integration.py`
  - ✅ `ml/tests/integration/test_registry_store_l2_integration.py`
  - ✅ `ml/tests/integration/earnings/test_earnings_store_db.py`
  - ✅ `ml/tests/integration/deployment/test_deployment_integration.py`
  - ✅ `ml/tests/integration/test_stores_integration.py`
  - ✅ `ml/tests/performance/test_ml_hot_path_benchmarks.py`
  - ✅ `ml/tests/performance/test_feature_calculator_microbench.py`
  - DB-free / non-Postgres (no clone migration needed):
    - `ml/tests/performance/test_parity_buffer_guardrails.py`
    - `ml/tests/performance/test_zero_allocation.py`
    - `ml/tests/integration/registry/test_model_registry_security.py`
    - `ml/tests/integration/data/test_tft_builder_integration.py`
    - `ml/tests/integration/pipeline/test_tft_pipeline_sidecar.py`
    - `ml/tests/integration/pipeline/test_tft_train_distill_pipeline.py`
    - `ml/tests/integration/earnings/test_tft_task_dataset.py` (uses stub DataStore)
  - SQLite-only (no Postgres clone): `ml/tests/integration/deployment/test_pipeline_rehydration.py`
- Property/contract suites marked with DB fixtures (progress):
  - ✅ `ml/tests/property/test_cross_asset_service_properties.py`
  - ✅ `ml/tests/contracts/test_base_actor_initialization.py`
- DB-free contract suites (no clone migration needed):
  - `ml/tests/contracts/test_store_env_topic_config_contracts.py` (uses patch_engine_manager)
  - `ml/tests/contracts/test_data_store_routing_advanced.py` (SQLite in-memory)
- Unit suites pointing at real Postgres (prefer clones or mocks) (progress):
  - ✅ `ml/tests/unit/stores/test_feature_store_facade.py`
  - ✅ `ml/tests/unit/stores/services/test_cross_asset_service.py`
  - ✅ `ml/tests/unit/stores/test_engine_manager_integration.py`
  - ✅ `ml/tests/unit/stores/test_instrument_metadata_store.py`
  - DB-free / SQLite-only (no clone needed):
    - ✅ `ml/tests/unit/core/test_create_data_store_signature.py`
    - ✅ `ml/tests/unit/core/common/test_store_initialization_component.py`
    - ✅ `ml/tests/unit/actors/common/test_model.py`
    - ✅ `ml/tests/unit/data/common/test_scheduler_init.py`
    - ✅ `ml/tests/unit/registry/common/test_model_persistence.py`
    - ✅ `ml/tests/unit/config/test_config.py` (SQLite default; Postgres only with `ML_FORCE_DB_INIT=1`)
    - ✅ `ml/tests/unit/registry/test_bootstrap_datasets_earnings.py`
    - ✅ `ml/tests/unit/stores/test_data_store_validation.py` (SQLite in-memory)
- Helpers calling `build_postgres_url` / `postgres_connection` should preferentially use `cloned_test_database` for isolation where writes occur.***
