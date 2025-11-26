# ML Architecture Migration Runbook ("The Heart Transplant")

**Objective:** Unify the `nautilus_trader` codebase by transplanting the clean, decomposed architecture from the `nautilus_trader-phase0` branch into the main branch.

**Core Philosophy:**

1.  **Transplant Code, Preserve Tests:** We want the *logic* from Phase0 but the *stability* (fixtures/tests) from Main.
2.  **Fix Forward:** Expect tests to break immediately after transplant. Fix them by adapting the *test code* to match the new components, or adapting the *components* to match the existing fixtures.

**Status Tracking:**

-   **Current Session:** fb907011-9104-47e1-8916-3827331e5722
-   **Last Completed Step:** Phase 6.1
-   **Next Step:** Phase 6.2

---

## Consolidation Alignment (post Phase0)

-   Scope: focus on the code transplant and consolidation; ignore unrelated chores in this runbook.
-   Canonical rule: facades/components are the source of truth; legacy/god classes remain only as thin shims behind `ML_USE_LEGACY_*` flags until final deletion.
-   Public API: export canonicals via each domain `__init__.py`; migrate callers to package-level imports; keep shims internal.
-   Parity + flags: keep legacy and facade side-by-side with parity tests; default to new paths, legacy flags for rollback; no deletes until both modes pass.
-   Test assimilation: copy Phase0 TDD suites (unit/contract/integration/parity/E2E) into main and adapt to `ml/tests/fixtures` guidance; add/extend `ml/tests/parity/` to cover legacy vs facade.
-   Orchestration note: placeholder components were removed; real implementations live in root modules (`config_resolver.py`, `dataset_builder.py`, `discovery_client.py`, `ingestion_coordinator.py`, `registry_synchronizer.py`, `runtime_attacher.py`, `training_coordinator.py`). Facade should import these; the legacy god class stays only for the legacy flag path.
-   Validation cadence per domain: `pytest -k <domain>` → `poetry run mypy ml --strict` → `poetry ruff check ml`; keep ML coverage ≥90%.

---

## Pick-up Checklist (quick start)

1. Baseline: `make pytest-ml` (or scoped `pytest -k <domain>`), `poetry run mypy ml --strict`, `poetry ruff check ml`; note failures in `reports/baseline_*.txt`.
2. Select domain (registry, actors, stores, orchestration, features, scheduler, training, dashboards).
3. Port Phase0 tests for that domain into `ml/tests` (unit/integration/parity/E2E) and align fixtures per `ml/tests/fixtures/FIXTURE_GUIDE.md`.
4. Ensure canonical wiring: facade/component exported via domain `__init__.py`; legacy file is shim behind `ML_USE_LEGACY_*` flag; callers use package-level imports.
5. Parity runs: `pytest -k <domain>` with flag off (new) and on (legacy); confirm identical assertions/parity outputs.
6. Record status in `reports/` (pass/fail, notes); only delete shims after gates below are met.

## Next Tasks (detailed)

1. Run full parity suites to surface drift:
   - `pytest ml/tests/facades` (already green) and rerun if code changes.
   - `pytest ml/tests/unit/core` and `ml/tests/unit/data ml/tests/unit/training ml/tests/unit/strategies ml/tests/unit/stores ml/tests/unit/orchestration` (expect type/protocol failures to triage).
2. Sweep orchestration parity/feature-flag behavior:
   - `pytest ml/tests/unit/orchestration/test_orchestrator_feature_flags.py`
   - Verify `use_legacy_orchestrator` honors both `ML_USE_LEGACY_ORCHESTRATOR` and `ML_USE_LEGACY_PIPELINE_ORCHESTRATOR`.
3. MLPipelineOrchestrator assimilation:
   - Decide legacy vs facade default; ensure `ml/orchestration/__init__.py` exports canonical.
   - Wire facade to real root modules; keep legacy god class only behind flag.
   - Resolve skipped integration tests or mark as structural with rationale.
4. Data module split (Phase 3.5 pending):
   - Split `ml/data/__init__.py` per Phase0 plan; keep legacy imports working.
   - Run `pytest -k data` and adapt fixtures/imports.
5. Static + lint gauntlet (expect hours):
   - `poetry run mypy ml --strict` and resolve: missing types from recent transplants, protocol conformance, TYPE_CHECKING imports.
   - `poetry ruff check ml` and fix lint/style/import cycles.
6. Coverage check after fixes:
   - `coverage run -m pytest ml/tests` then `coverage report --fail-under=90` (ML target).
7. Flag parity validation for remaining domains:
   - For each `ML_USE_LEGACY_*`, run targeted pytest with flag on/off (registry, stores, actors, scheduler, trainer, dashboard).
8. Update `__init__.py` exports:
   - Ensure canonicals exported; legacy shims internal; add back-compat aliases where needed (e.g., `MLIntegrationManager`).
9. Document progress:
   - Tick God-Class checklist items when parity + tests + flags verified.
   - Log failures/gaps in `reports/` for handoff.

## God-Class Assimilation Checklist (verify transplant + tests)

- [ ] MLPipelineOrchestrator (canonical facade + root modules; legacy god class only under flag)
- [x] DataStore (DataStoreFacade + components; legacy shim)
- [x] FeatureEngineer (facade + components; legacy shim)
- [x] BaseMLInferenceActor (facade/components; legacy shim)
- [x] MLSignalActor (facade/components; legacy path)
- [x] ModelRegistry (ModelRegistryFacade + components; legacy shim)
- [x] DataRegistry (DataRegistryFacade + components; legacy shim)
- [x] FeatureStore (FeatureStoreFacade + components; legacy shim)
- [x] TFTDatasetBuilder (facade/components; legacy flag)
- [x] MLIntegrationManager (facade/components; legacy shim; imports/export fixed)
- [x] MLTradingStrategy Base (facade/components; legacy shim)
- [ ] Data module (__init__ split pending)
- [x] Training Base (BaseMLTrainerFacade + components; legacy shim)
- [x] DataScheduler (DataSchedulerFacade + components; legacy flag)
- [x] Dashboard Service (facade/components; legacy shim)
- [x] Dashboard App (blueprint app facade; legacy flag)

## Feature-Flag Matrix (canonical vs legacy)

- `ML_USE_LEGACY_DATA_STORE`: canonical `ml/stores/data_store_facade.py`; legacy `ml/stores/data_store.py` shim.
- `ML_USE_LEGACY_FEATURE_STORE`: canonical `ml/stores/feature_store_facade.py`; legacy `ml/stores/feature_store.py` shim.
- `ML_USE_LEGACY_MODEL_REGISTRY`: canonical `ml/registry/model_registry_facade.py`; legacy `ml/registry/model_registry.py` shim.
- `ML_USE_LEGACY_DATA_REGISTRY`: canonical `ml/registry/data_registry_facade.py`; legacy `ml/registry/data_registry.py` shim.
- `ML_USE_LEGACY_PIPELINE_ORCHESTRATOR`: canonical `ml/orchestration/pipeline_orchestrator_facade.py` + root modules; legacy `ml/orchestration/pipeline_orchestrator.py`.
- `ML_USE_LEGACY_BASE_ACTOR`: canonical `ml/actors/base.py`; legacy `ml/actors/base_legacy.py` shim.
- `ML_USE_LEGACY_ML_SIGNAL_ACTOR`: canonical `ml/actors/signal.py` (or `signal_facade.py` unified); legacy path guarded in same module.
- `ML_USE_LEGACY_FEATURE_ENGINEER`: canonical `ml/features/facade.py`; legacy `ml/features/engineering.py` shim.
- `ML_USE_LEGACY_TFT_BUILDER`, `ML_USE_LEGACY_SCHEDULER`, `ML_USE_LEGACY_DASHBOARD_SERVICE`, `ML_USE_LEGACY_DASHBOARD_APP`, `ML_USE_LEGACY_TRAINER` follow the same pattern; defaults should favor canonicals.

## Outstanding Work (keep visible)

- Phase 6.2: replace `_TradingDayCalendar` with `pandas_market_calendars` implementation.
- Phase 6.3: enforce zero-copy prediction buffer (bounded `deque` or preallocated array).
- Phase 6.4: Polars core/Pandas edge sweep in `ml/features` and `ml/data`.

## Phase0 Test Import Map (where to pull from)

- Registry: copy Phase0 registry unit/integration/parity tests; patch imports to `ml.tests.fixtures` and main paths.
- Actors: copy Base/Signal actor tests (unit, contract, integration, parity) and align fixtures; ensure feature flags parameterized.
- Stores: copy DataStore/FeatureStore tests (CRUD, parity, integration with registries).
- Orchestration: copy real-root-module tests (not placeholder components); target `config_resolver`, `dataset_builder`, `discovery_client`, `ingestion_coordinator`, `registry_synchronizer`, `runtime_attacher`, `training_coordinator`.
- Features/TFT/Scheduler/Training/Dashboard: copy corresponding Phase0 suites; adapt fixtures per `ml/tests/fixtures/FIXTURE_GUIDE.md`.

## Deletion Gates (before removing legacy files)

- Parity tests passing with flags on/off for the domain.
- Callers migrated to package-level imports; no direct legacy imports in CI grep.
- Defaults point to canonical; legacy only behind flags.
- Coverage maintained (ML ≥90%); mypy/ruff clean.
- Results logged in `reports/` with command outputs.

## Orchestration Status (recap)

- Placeholders removed; real modules live at `ml/orchestration/{config_resolver,dataset_builder,discovery_client,ingestion_coordinator,registry_synchronizer,runtime_attacher,training_coordinator}.py`.
- Facade should import real modules; legacy god class exists only for `ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1`.
- Option B chosen: new root modules extracted from the god class; component placeholders and related tests deleted; continue to run parity for facade vs legacy.

---

## 🛑 Critical Instructions for Agents

1.  **Read-Only Phase0:** NEVER modify files in `nautilus_trader-phase0`.
2.  **Test First:** Run `pytest` before starting to establish a baseline.
3.  **Expect Breakage:** After a transplant step, tests *will* fail. This is normal. Your job is to make them pass again.

---

## Phase 1: Registry Transplant (Low Risk)
**Goal:** Replace the Hybrid/Legacy Registry with the pure Component Registry.

### 1.1 Backup & Prep

-   [x] **Command:** `cp -r ml/registry ml/registry_backup_phase1`
-   [x] **Verify:** `ls -d ml/registry_backup_phase1`

### 1.2 Transplant Components

-   [x] **Action:** Copy `ml/registry/persistence/` from Phase0 to Main.
-   `cp -r /home/nate/projects/nautilus_trader-phase0/ml/registry/persistence ml/registry/`
-   [x] **Action:** Copy [ml/registry/model_registry.py](file:///home/nate/projects/nautilus_trader/ml/registry/model_registry.py) from Phase0 to Main.
-   `cp /home/nate/projects/nautilus_trader-phase0/ml/registry/model_registry.py ml/registry/`

### 1.3 Cleanup Legacy

-   [x] **Action:** Delete [ml/registry/model_registry_legacy.py](file:///home/nate/projects/nautilus_trader/ml/registry/model_registry_legacy.py).
-   `rm ml/registry/model_registry_legacy.py`

### 1.4 Adaptation & Verification (The "How")

-   [x] **Command:** `pytest ml/tests/unit_tests/registry`
-   [x] **Troubleshooting Protocol:**
-   **ImportError (Fixtures):** If tests fail importing fixtures from `ml.tests.fixtures`, check `ml/tests/fixtures/` in Main. The Phase0 code might use a different path. Update the import in the *transplanted file* or the *test file* to point to the correct location in Main.
-   **AttributeError (Config):** The Phase0 components might expect slightly different config objects. Check `ml/config/` in Phase0 vs Main. If needed, port the config changes from Phase0.

---

## Phase 2: Orchestrator Transplant (Medium Risk)
**Goal:** Replace the God Class Orchestrator with the Decomposed Orchestrator.

### 2.1 Backup & Prep

-   [x] **Command:** `cp -r ml/orchestration ml/orchestration_backup_phase2`

### 2.2 Transplant Components

-   [x] **Action:** Copy `ml/orchestration/components/` from Phase0 to Main.
-   `cp -r /home/nate/projects/nautilus_trader-phase0/ml/orchestration/components ml/orchestration/`
-   [x] **Action:** Copy `ml/orchestration/pipeline_orchestrator.py` from Phase0 to Main.
-   `cp /home/nate/projects/nautilus_trader-phase0/ml/orchestration/pipeline_orchestrator.py ml/orchestration/`

### 2.3 Cleanup Legacy

-   [x] **Action:** Delete `ml/orchestration/pipeline_orchestrator_legacy.py`.
-   `rm ml/orchestration/pipeline_orchestrator_legacy.py`

### 2.4 Adaptation & Verification

-   [x] **Command:** `pytest ml/tests/unit_tests/orchestration`
-   [x] **Troubleshooting Protocol:**
-   **Missing Component:** If the Orchestrator fails to find `IngestionCoordinator`, ensure the `ml/orchestration/components/__init__.py` exposes it.
-   **Signature Mismatch:** If the `run()` method signature differs, update the calling code in `ml/tests/unit_tests/orchestration/test_pipeline_orchestrator.py`.

---

## Phase 3: Store Transplant (High Risk)
**Goal:** Decompose the DataStore and FeatureStore.

### 3.1 Backup & Prep

-   [x] **Command:** `cp -r ml/stores ml/stores_backup_phase3`

### 3.2 Transplant Components

-   [x] **Action:** Copy `ml/stores/components/` from Phase0 to Main.
-   `cp -r /home/nate/projects/nautilus_trader-phase0/ml/stores/components ml/stores/`
-   [x] **Action:** Copy `ml/stores/data_store.py` from Phase0 to Main.
-   `cp /home/nate/projects/nautilus_trader-phase0/ml/stores/data_store.py ml/stores/`

### 3.3 Cleanup Legacy

-   [x] **Action:** Delete `ml/stores/data_store_legacy.py`.
-   `rm ml/stores/data_store_legacy.py`

### 3.4 Adaptation & Verification

-   [x] **Command:** `pytest ml/tests/unit_tests/stores`
-   [x] **Troubleshooting Protocol:**
-   **Database Schema:** Phase0 might assume a newer SQL schema. Check `ml/stores/migrations` in Phase0. If needed, apply the schema changes to Main's migration files.

---

## Phase 4: Actor Transplant (High Risk)
**Goal:** Decompose the BaseMLInferenceActor.

### 4.1 Backup & Prep

-   [x] **Command:** `cp -r ml/actors ml/actors_backup_phase4`

### 4.2 Transplant Components

-   [x] **Action:** Copy `ml/actors/components/` from Phase0 to Main.
-   `cp -r /home/nate/projects/nautilus_trader-phase0/ml/actors/components ml/actors/`
-   [x] **Action:** Copy `ml/actors/base.py` from Phase0 to Main.
-   `cp /home/nate/projects/nautilus_trader-phase0/ml/actors/base.py ml/actors/`

### 4.3 Adaptation & Verification

-   [x] **Command:** `pytest ml/tests/unit_tests/actors`

---

## Phase 5: Feature Engineering Transplant (High Risk)
**Goal:** Decompose the FeatureEngineer God Class.

### 5.1 Backup & Prep

-   [x] **Command:** `cp -r ml/features ml/features_backup_phase5`

### 5.2 Transplant Components

-   [x] **Action:** Copy `ml/features/components/` from Phase0 to Main.
-   `cp -r /home/nate/projects/nautilus_trader-phase0/ml/features/components ml/features/`
-   [x] **Action:** Copy `ml/features/facade.py` from Phase0 to Main.
-   `cp /home/nate/projects/nautilus_trader-phase0/ml/features/facade.py ml/features/`

### 5.3 Cleanup Legacy

-   [x] **Action:** Delete `ml/features/engineering.py`.
-   `rm ml/features/engineering.py` (Renamed to engineering_legacy.py)

### 5.4 Adaptation & Verification

-   [x] **Command:** `pytest ml/tests/unit_tests/features`

---

## Phase 6: Critical Manual Fixes (Post-Transplant)
**Goal:** Resolve data integrity and performance issues.

### 6.1 Vectorization Fix (The "How")

-   [x] **Problem:** `update_batch_vectorized` uses a Python loop.
-   [x] **Solution:** Use Polars for true vectorization.
-   **Implementation Sketch:**

    ```python
    import polars as pl
    def update_batch_vectorized(self, ...):
        # Convert inputs to Polars DataFrame
        df = pl.DataFrame({
            "open": open_prices,
            "high": high_prices,
            ...
        })
        # Apply expressions (lazy is better for chaining)
        result = df.lazy().with_columns([
            pl.col("close").rolling_mean(window_size).alias("ma")
        ]).collect()
        return result.to_dicts()
    ```

### 6.2 Calendar Fix (The "How")

-   **Problem:** `_TradingDayCalendar` returns `True` for everything.
-   **Solution:** Use `pandas_market_calendars`.
-   **Implementation Sketch:**

    ```python
    from pandas_market_calendars import get_calendar
    class RealTradingCalendar:
        def __init__(self, exchange="NYSE"):
            self.cal = get_calendar(exchange)
        def is_trading_day(self, timestamp):
            return self.cal.valid_days(start_date=timestamp, end_date=timestamp).size > 0
    ```

### 6.3 Memory Leak Fix (The "How")

-   **Problem:** `MLSignalActor` leaks memory.
-   **Solution:** Enforce zero-copy.
-   **Check:** Ensure `prediction_buffer.py` uses `collections.deque(maxlen=N)` or a pre-allocated `numpy` array with a pointer, NOT `list.append()`.

### 6.4 Dataframe Unification (The "How")

-   **Problem:** Mixed Pandas/Polars usage.
-   **Solution:** "Polars Core, Pandas Edge".
-   **Action:** Grep for `pd.DataFrame` in `ml/features` and `ml/data`. Replace with `pl.DataFrame` unless the method explicitly returns to an external API that requires Pandas (e.g., some plotting libs or legacy Nautilus interfaces).
