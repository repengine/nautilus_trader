# Xdist Parallel Test Failures Remediation Plan

## Scope and goals

- Make `pytest -n` runs stable without changing test intent.
- Preserve backward compatibility and facade behavior expected by the tests.
- Keep changes minimal, typed, and safe per AGENTS guardrails.

## Context summary

- Parallel run (`pytest ml -n 12`) is fast but exposes many errors and failures that are not present without xdist.
- The failures cluster into a few root causes: database contention in session fixtures, missing legacy shims, missing API endpoints, and behavior invariants not enforced in hot-path code.

## Current status

- Latest run: `poetry run pytest ml -n 12` → 6635 passed, 91 skipped, 1 xfailed, 1 xpassed (0 failures).

## Failure clusters and evidence

1) DB template contention in xdist
   - `template_database` drops and recreates the shared template DB every session, causing `ObjectInUse` and `database does not exist` errors across workers.
   - Evidence: `ml/tests/fixtures/database_fixtures.py:171-205` and fixture guide guidance in `ml/tests/fixtures/FIXTURE_GUIDE.md:147-155`.
   - Locking only occurs per test, not during session fixture setup: `ml/tests/conftest.py:257-285`.

2) Legacy shim gaps (registry and dashboard)
   - `DataRegistry` no longer exposes `_events`, `_contracts`, `_lineage`, or `_save_registry`, but tests and bootstrap still access them.
   - Evidence: `ml/registry/data_registry.py:71-118`, `ml/registry/common/data_persistence.py:103-108`,
     `ml/registry/bootstrap_datasets.py:671-676`, `ml/tests/unit/registry/test_data_registry_unit.py:69-112`.
   - `DashboardService` lacks `_get_pipeline_service`, `_get_*_registry`, `list_models_with_performance`,
     and module-level shims (`requests`, `_to_url`, `provision_dashboard`, `publisher_from_config`,
     `ObservabilityService`), which tests patch.
   - Evidence: `ml/dashboard/service.py:210-330` and tests in `ml/tests/unit/dashboard/test_dashboard_api.py:20-220`,
     `ml/tests/facades/test_dashboard_parity.py:90-116`, `ml/tests/integration/dashboard/test_streaming_state_endpoint.py:40-120`.

3) Actor compatibility gaps and invariants
   - `MLSignalActorFacade` does not expose `_create_strategy`, but tests call it.
   - Evidence: `ml/actors/signal_facade_impl.py` (missing `_create_strategy`) and `ml/tests/unit/actors/test_signal_adapter_fallback.py:24-36`.
   - `_predict` returns raw values without clamping or NaN/Inf sanitization, which violates property tests
     that require prediction in [-1, 1] and confidence in [0, 1].
   - Evidence: `ml/actors/signal_facade_impl.py:1186-1256` and `ml/tests/property/test_signal_actor_bounds.py:1-80`.

4) Missing dashboard endpoints
   - `/api/market/tickers`, `/api/registry/models/performance`, `/api/training/streaming/state` are not registered.
   - Evidence: `ml/dashboard/app.py:60-200` and `ml/tests/unit/dashboard/test_dashboard_api.py:570-720`,
     `ml/dashboard/tests/test_streaming_monitor.py:260-300`.

5) Builder aliasing and timestamp columns
   - `TFTDatasetBuilderFacade` is a direct alias, so patching the builder class in tests does not affect the facade.
   - FeatureStore path renames `timestamp -> ts_event` without restoring `timestamp`, leading to missing column failures.
   - Evidence: `ml/data/tft_dataset_builder_facade.py` and wrapper at `ml/data/tft_dataset_builder.py`.

6) Orchestrator runtime attachment mismatch
   - StageController default runtime attachment updates its own fields; orchestrator fields remain unset.
   - Orchestrator does not inject `_attach_runtime` into StageController.
   - Evidence: `ml/orchestration/pipeline_orchestrator.py:541-555` and `ml/orchestration/common/stage_controller.py:797-857`.

7) Logger compatibility (`exc_info`)
   - Some loggers do not accept `exc_info`; tests fail with `error() got unexpected keyword argument 'exc_info'`.
   - `_SafeLogger` exists but is not consistently used.
   - Evidence: `ml/strategies/common/decision_persistence.py:115-156`, failing tests in `ml/tests/unit/strategies/test_order_executor_integration.py`.

## Constraints and guardrails

- All changes must keep strict typing (`poetry run mypy ml --strict`) and lint clean (`poetry ruff check ml`).
- Hot-path rules apply: no new allocations or I/O in event handlers and inference paths.
- Use `ml._imports` for optional ML dependencies.
- No hard-coded tunables; configuration must live under `ml/config`.

## Remediation plan

### Phase 0 - Repro harness (no code changes)

- Record baseline failures and counts for comparison.
- Suggested command:
  - `poetry run pytest ml -n 12`

### Phase 1 - Fix template DB concurrency (P0)

- Add a cross-worker lock or worker-scoped template DB to `template_database`.
  - Option A: lock around template creation; avoid drop if DB exists.
  - Option B: per-worker template name using `PYTEST_XDIST_WORKER` and `TEST_DB_TEMPLATE_NAME`.
- Ensure clones use the correct template name per worker.
- Acceptance criteria:
  - No `database ... is being accessed by other users` or `database does not exist` errors.
  - `cloned_test_database` continues to isolate per-test data.
- Status:
  - Implemented worker-scoped template DB naming via `PYTEST_XDIST_WORKER` in `ml/tests/fixtures/database_fixtures.py`.

### Phase 1b - Fixture hygiene for parallel safety (P0)

- Align with `ml/tests/fixtures/FIXTURE_GUIDE.md`:
  - Avoid writing to `template_database`; ensure all writable tests use `cloned_test_database` or `fresh_store_bundle`.
  - Prefer `fresh_store_bundle` over `store_bundle` for new/updated tests to avoid shared state.
  - Use `patch_engine_manager`/`mock_engine_manager` in tests that touch EngineManager caches.
- Align with `ml/tests/docs/TESTING_STRATEGY.md`:
  - Use `HYPOTHESIS_PROFILE=ci`, `ML_DISABLE_METRICS_SERVER=1`, and `TEST_DB_SKIP_TRUNCATE=1` for parallel runs.
  - Prefer `--dist=loadscope` for xdist to reduce cross-worker contention for class/module-scoped fixtures.
- Acceptance criteria:
  - No test writes to the template DB in parallel runs.
  - DB and metrics-related tests are isolated via fixtures rather than shared globals.

### Phase 2 - Restore legacy shims (P0)

- DataRegistry
  - Expose read-only `_events`, `_contracts`, `_lineage`, and `_save_registry` delegating to persistence.
  - Keep JSON-only behavior where expected; ensure no DB-side mutations.
- DashboardService
  - Add shims `_get_pipeline_service`, `_get_*_registry` to route to existing components.
  - Add module-level aliases for `requests`, `_to_url`, `provision_dashboard`,
    `publisher_from_config`, `ObservabilityService` for tests that patch them.
  - Add `list_models_with_performance` that composes registry results with performance summaries.
- MLSignalActorFacade
  - Provide `_create_strategy` wrapper that uses `SignalStrategyComponent` and falls back to threshold.
- Acceptance criteria:
  - Facade tests in `ml/tests/facades` and `ml/tests/unit/dashboard` no longer fail due to missing attributes.

### Phase 3 - Enforce prediction bounds and NaN handling (P1)

- Clamp predictions to [-1, 1] and confidence to [0, 1] in `_predict`.
- Replace NaN/Inf with safe defaults and record fallback metrics if required.
- Ensure feature-time metrics are recorded even when `_compute_features` is invoked directly.
- Acceptance criteria:
  - Property tests in `ml/tests/property/test_signal_actor_bounds.py` pass under xdist.
  - No hot-path allocation regressions.
- Status:
  - Implemented: adaptive threshold clamping, legacy override setters, and safe logging for threshold clamping.

### Phase 4 - Implement missing dashboard endpoints (P1)

- Add `/api/market/tickers` with a minimal data path (DataStore or stubbed response).
- Add `/api/registry/models/performance` that returns `{"models": [...]}`.
- Add `/api/training/streaming/state` returning `DashboardService.get_streaming_training_state()`.
- Acceptance criteria:
  - `ml/tests/unit/dashboard/test_dashboard_api.py` and
    `ml/dashboard/tests/test_streaming_monitor.py` pass.

### Phase 5 - Fix builder aliasing and timestamp column (P1)

- Replace `TFTDatasetBuilderFacade` alias with a thin wrapper class (same signature).
- Ensure FeatureStore path preserves `timestamp` column for contract tests.
- Acceptance criteria:
  - `ml/tests/facades/test_tft_builder_parity.py::test_e32_contract_output_columns` passes.

### Phase 6 - Orchestrator runtime attachment (P2)

- Inject orchestrator `_attach_runtime` into StageController so attachment updates the orchestrator fields.
- Alternatively, have StageController return the manager and sync back to orchestrator.
- Acceptance criteria:
  - `ml/tests/unit/orchestration/test_pipeline_orchestrator.py::test_pipeline_orchestrator_attach_runtime_sets_components` passes.

### Phase 7 - Logger exc_info compatibility (P2)

- Ensure logger wrappers tolerate `exc_info` and extra kwargs.
- Prefer centralized adapter or `_SafeLogger` usage where non-stdlib loggers exist.
- Acceptance criteria:
  - Strategy integration tests no longer fail with `unexpected keyword argument 'exc_info'`.
- Status:
  - Implemented: safe logger usage for threshold clamping and smart-order fallback without type errors.

## Validation checklist (for implementation work)

- Types: `poetry run mypy ml --strict`
- Lint: `poetry ruff check ml`
- Focused tests per area touched:
  - DB fixtures: `poetry run pytest -k database ml/tests/fixtures`
  - Actor bounds: `poetry run pytest -k signal_actor_bounds ml/tests/property`
  - Dashboard: `poetry run pytest -k dashboard ml/tests/unit/dashboard ml/dashboard/tests`
  - Orchestrator: `poetry run pytest -k pipeline_orchestrator ml/tests/unit/orchestration`
- Coverage: run `coverage report` and keep ML modules >= 90%.

## Risks and mitigations

- Changing DB fixtures could hide genuine test isolation issues. Mitigate by keeping
  `cloned_test_database` per-test and preserving pollution-detection tests.
- Compatibility shims risk masking regressions. Mitigate by limiting shims to private
  legacy names and forwarding to canonical components.
- Hot-path changes must remain allocation-free. Mitigate by clamping with simple `min/max`
  and by reusing pre-allocated buffers.
