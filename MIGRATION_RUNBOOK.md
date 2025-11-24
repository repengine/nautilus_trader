# ML Architecture Migration Runbook ("The Heart Transplant")

**Objective:** Unify the `nautilus_trader` codebase by transplanting the clean, decomposed architecture from the `nautilus_trader-phase0` branch into the main branch.

**Core Philosophy:**

1.  **Transplant Code, Preserve Tests:** We want the *logic* from Phase0 but the *stability* (fixtures/tests) from Main.
2.  **Fix Forward:** Expect tests to break immediately after transplant. Fix them by adapting the *test code* to match the new components, or adapting the *components* to match the existing fixtures.

**Status Tracking:**

-   **Current Session:** fb907011-9104-47e1-8916-3827331e5722
-   **Last Completed Step:** Phase 3.4
-   **Next Step:** Phase 4.1

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

-   [ ] **Command:** `cp -r ml/actors ml/actors_backup_phase4`

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
-   `rm ml/features/engineering.py`

### 5.4 Adaptation & Verification

-   [x] **Command:** `pytest ml/tests/unit_tests/features`

---

## Phase 6: Critical Manual Fixes (Post-Transplant)
**Goal:** Resolve data integrity and performance issues.

### 6.1 Vectorization Fix (The "How")

-   **Problem:** `update_batch_vectorized` uses a Python loop.
-   **Solution:** Use Polars for true vectorization.
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
