# Legacy Removal Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove remaining legacy env flags and fallback paths, standardize canonical imports, and add schema contract tests while restoring validate_wave and coverage gates.

**Architecture:** Eliminate runtime feature flag selection by deleting legacy flag helpers and collapsing facades into thin shims that re-export canonical modules. Replace legacy parity tests with facade-only contract tests, and add schema/template invariants grounded in `ml/schema.py` to guard identifier resolution and dataset mappings.

**Tech Stack:** Python 3.11, pytest, hypothesis (optional for property tests), mypy, ruff, coverage.

---

### Task 1: Inventory sweep and update legacy plan status

**Files:**
- Modify: `LEGACY_REMOVAL_PLAN.md`

**Step 1: Run the inventory sweep**
Run: `rg "ML_USE_LEGACY" ml` and `rg "_legacy" ml`
Expected: list of remaining legacy env flags and code paths.

**Step 2: Update the plan inventory**
Update `LEGACY_REMOVAL_PLAN.md` sections:
- Replace the “Legacy-Named Files Still Present” and “Flag Name Aliases Still Present” lists with actual current hits.
- Add a short “Remaining Removal Targets” list including the files still holding legacy flags (e.g. `ml/orchestration/feature_flags.py`, `ml/data/scheduler_facade.py`, `ml/data/tft_dataset_builder_facade.py`, `ml/dashboard/__init__.py`, `ml/training/base_facade.py`, `ml/strategies/base_facade.py`, `ml/config/feature_flags.py`, `ml/data/scheduler.py.bak`).

**Step 3: Commit**
Run:
```bash
git add LEGACY_REMOVAL_PLAN.md
git commit -m "docs: refresh legacy removal inventory"
```

---

### Task 2: Remove config-level legacy flags module

**Files:**
- Delete: `ml/config/feature_flags.py`
- Modify: `ml/tests/unit/config/test_feature_flags.py`

**Step 1: Write the failing test update**
Update `ml/tests/unit/config/test_feature_flags.py` to remove imports and test cases for `use_legacy_*` helpers. If no tests remain after removal, delete the file.

**Step 2: Run tests to confirm expected failure**
Run: `poetry run pytest ml/tests/unit/config/test_feature_flags.py -q`
Expected: FAIL (missing imports or empty test module).

**Step 3: Remove the legacy flag module**
Delete `ml/config/feature_flags.py` and update any remaining references.

**Step 4: Run tests again**
Run: `poetry run pytest ml/tests/unit/config/test_feature_flags.py -q`
Expected: PASS or “file not found” if deleted (adjust pytest target accordingly).

**Step 5: Commit**
```bash
git add ml/config/feature_flags.py ml/tests/unit/config/test_feature_flags.py
git commit -m "refactor: drop config-level legacy flags"
```

---

### Task 3: Remove orchestration legacy flags and fallbacks

**Files:**
- Delete: `ml/orchestration/feature_flags.py`
- Modify: `ml/orchestration/__init__.py`
- Modify: `ml/orchestration/pipeline_orchestrator_facade.py`
- Modify: `ml/orchestration/pipeline_orchestrator.py`
- Modify: `ml/orchestration/ingestion_coordinator.py`
- Modify: `ml/tests/unit/orchestration/test_orchestrator_parity.py`
- Modify: `ml/tests/facades/test_pipeline_orchestrator_parity.py`

**Step 1: Update tests to facade-only expectations**
Remove env-var toggle assertions and keep component-path invariants. Example replacement snippet for `test_pipeline_orchestrator_parity.py`:
```python
health = orchestrator.get_health_status()
assert health["implementation"] == "component-based"
```
(Adjust for the final `get_health_status` schema after flag removal.)

**Step 2: Run the targeted tests (expect failure)**
Run: `poetry run pytest ml/tests/unit/orchestration/test_orchestrator_parity.py -q`
Expected: FAIL (imports/flags removed).

**Step 3: Remove orchestration flag selection**
- In `ml/orchestration/__init__.py`, remove `use_legacy_orchestrator` import and conditional selection. Always export `MLPipelineOrchestrator` from `ml/orchestration/pipeline_orchestrator.py`.
- Convert `ml/orchestration/pipeline_orchestrator_facade.py` to a thin shim:
```python
from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator as MLPipelineOrchestratorFacade
__all__ = ["MLPipelineOrchestratorFacade"]
```
- Remove `use_legacy_orchestrator` references and legacy branches from `ml/orchestration/pipeline_orchestrator.py` and `ml/orchestration/ingestion_coordinator.py`.

**Step 4: Re-run the targeted tests**
Run: `poetry run pytest ml/tests/unit/orchestration/test_orchestrator_parity.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add ml/orchestration/__init__.py ml/orchestration/pipeline_orchestrator_facade.py \
  ml/orchestration/pipeline_orchestrator.py ml/orchestration/ingestion_coordinator.py \
  ml/tests/unit/orchestration/test_orchestrator_parity.py ml/tests/facades/test_pipeline_orchestrator_parity.py \
  ml/orchestration/feature_flags.py
git commit -m "refactor: drop orchestration legacy flags"
```

---

### Task 4: Scheduler legacy cleanup and shim

**Files:**
- Modify: `ml/data/scheduler.py`
- Delete: `ml/data/scheduler.py.bak`
- Modify: `ml/data/scheduler_facade.py`
- Modify: `ml/tests/unit/data/test_scheduler_facade.py`

**Step 1: Update tests to remove legacy flag assertions**
Delete tests that assert `use_legacy_scheduler()` behavior. Replace with a smoke test asserting `create_data_scheduler()` returns a `DataScheduler` instance.

**Step 2: Run tests (expect failure)**
Run: `poetry run pytest ml/tests/unit/data/test_scheduler_facade.py -q`
Expected: FAIL (legacy helper removed).

**Step 3: Remove legacy toggles and align shims**
- Remove `as_legacy_cython=True` in `ml/data/scheduler.py` DBN loader calls.
- Delete `ml/data/scheduler.py.bak`.
- Convert `ml/data/scheduler_facade.py` into a shim that re-exports `DataScheduler` and `create_data_scheduler` without env flags:
```python
from ml.data.scheduler import DataScheduler

def create_data_scheduler(...):
    return DataScheduler(...)

DataSchedulerFacade = DataScheduler
__all__ = ["DataScheduler", "DataSchedulerFacade", "create_data_scheduler"]
```

**Step 4: Re-run tests**
Run: `poetry run pytest ml/tests/unit/data/test_scheduler_facade.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add ml/data/scheduler.py ml/data/scheduler.py.bak ml/data/scheduler_facade.py \
  ml/tests/unit/data/test_scheduler_facade.py
git commit -m "refactor: drop scheduler legacy toggles"
```

---

### Task 5: TFT dataset builder legacy cleanup and shim

**Files:**
- Modify: `ml/data/tft_dataset_builder_facade.py`
- Modify: `ml/tests/unit/data/test_tft_dataset_builder_facade.py`
- Modify: `ml/tests/facades/test_tft_builder_parity.py`

**Step 1: Update tests to remove legacy env flag coverage**
Remove `ML_USE_LEGACY_TFT_BUILDER` toggle tests and keep parity checks focused on the facade class.

**Step 2: Run targeted tests (expect failure)**
Run: `poetry run pytest ml/tests/unit/data/test_tft_dataset_builder_facade.py -q`
Expected: FAIL (legacy helper removed).

**Step 3: Convert facade to shim**
Replace `use_legacy_builder` and the env-flag docstrings with a shim that re-exports the canonical builder:
```python
from ml.data.tft_dataset_builder import TFTDatasetBuilder

TFTDatasetBuilderFacade = TFTDatasetBuilder
__all__ = ["SchemaValidationError", "TFTDatasetBuilderFacade"]
```
Keep `SchemaValidationError` re-export if external callers depend on it.

**Step 4: Re-run tests**
Run: `poetry run pytest ml/tests/unit/data/test_tft_dataset_builder_facade.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add ml/data/tft_dataset_builder_facade.py ml/tests/unit/data/test_tft_dataset_builder_facade.py \
  ml/tests/facades/test_tft_builder_parity.py
git commit -m "refactor: drop tft builder legacy toggle"
```

---

### Task 6: Dashboard legacy cleanup and shims

**Files:**
- Modify: `ml/dashboard/__init__.py`
- Modify: `ml/dashboard/app_facade.py`
- Modify: `ml/dashboard/service_facade.py`
- Modify: `ml/tests/unit/dashboard/test_app_facade.py`
- Modify: `ml/tests/facades/test_app_parity.py`
- Modify: `ml/tests/facades/test_dashboard_parity.py`

**Step 1: Update tests to remove env-var toggles**
Delete tests that assert `ML_USE_LEGACY_DASHBOARD_APP` or `ML_USE_LEGACY_DASHBOARD_SERVICE` toggles.

**Step 2: Run targeted tests (expect failure)**
Run: `poetry run pytest ml/tests/unit/dashboard/test_app_facade.py -q`
Expected: FAIL (legacy helper removed).

**Step 3: Remove feature flag selection**
- `ml/dashboard/__init__.py`: always export `DashboardService` from `ml/dashboard/service.py`.
- `ml/dashboard/app_facade.py`: convert to a shim that re-exports `create_app` from `ml/dashboard/app.py` and remove `use_legacy_dashboard_app`.
- `ml/dashboard/service_facade.py`: keep class as alias to `DashboardService` or convert to re-export shim; remove legacy references in docstrings.

**Step 4: Re-run tests**
Run: `poetry run pytest ml/tests/unit/dashboard/test_app_facade.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add ml/dashboard/__init__.py ml/dashboard/app_facade.py ml/dashboard/service_facade.py \
  ml/tests/unit/dashboard/test_app_facade.py ml/tests/facades/test_app_parity.py \
  ml/tests/facades/test_dashboard_parity.py
git commit -m "refactor: drop dashboard legacy flags"
```

---

### Task 7: Remove trainer/strategy/actor legacy helpers

**Files:**
- Modify: `ml/training/base_facade.py`
- Modify: `ml/strategies/base_facade.py`
- Modify: `ml/tests/unit/training/test_base_trainer_facade.py`
- Modify: `ml/tests/unit/strategies/test_base_ml_strategy_facade.py`
- Modify: `ml/tests/facades/test_strategy_base_parity.py`
- Modify: `ml/tests/unit/actors/test_mlsignal_actor_facade.py`

**Step 1: Update tests to remove env-flag coverage**
Delete test cases that only exercise `use_legacy_trainer`, `_use_legacy_strategy_base`, or `ML_USE_LEGACY_SIGNAL_ACTOR` toggles.

**Step 2: Run targeted tests (expect failure)**
Run: `poetry run pytest ml/tests/unit/training/test_base_trainer_facade.py -q`
Expected: FAIL (helper removed).

**Step 3: Remove helper functions**
- Remove `use_legacy_trainer` from `ml/training/base_facade.py` and update docstrings.
- Remove `_use_legacy_strategy_base` from `ml/strategies/base_facade.py` and drop from `__all__`.
- Clean any env-var mentions in related docs/comments.

**Step 4: Re-run tests**
Run: `poetry run pytest ml/tests/unit/training/test_base_trainer_facade.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add ml/training/base_facade.py ml/strategies/base_facade.py \
  ml/tests/unit/training/test_base_trainer_facade.py \
  ml/tests/unit/strategies/test_base_ml_strategy_facade.py \
  ml/tests/facades/test_strategy_base_parity.py \
  ml/tests/unit/actors/test_mlsignal_actor_facade.py
git commit -m "refactor: drop trainer/strategy legacy helpers"
```

---

### Task 8: Remove legacy BaseMLInferenceActor shim

**Files:**
- Delete: `ml/actors/base_legacy.py`
- Modify: `ml/tests/unit/actors/test_facade_parity.py`

**Step 1: Update tests to stop importing base_legacy**
Replace `from ml.actors.base_legacy import BaseMLInferenceActor` with the canonical `ml.actors.base` import.

**Step 2: Run targeted tests (expect failure)**
Run: `poetry run pytest ml/tests/unit/actors/test_facade_parity.py -q`
Expected: FAIL (module removed).

**Step 3: Remove the shim module**
Delete `ml/actors/base_legacy.py`.

**Step 4: Re-run tests**
Run: `poetry run pytest ml/tests/unit/actors/test_facade_parity.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add ml/actors/base_legacy.py ml/tests/unit/actors/test_facade_parity.py
git commit -m "refactor: remove base_legacy shim"
```

---

### Task 9: Schema contract tests for identifier templates

**Files:**
- Create: `ml/tests/unit/schema/test_schema_identifier_templates.py`

**Step 1: Write failing tests**
Add tests asserting `validate_identifier_template` requires `{instrument_id}` and that schema defaults map to expected templates. Example:
```python
from ml.schema import DEFAULT_BAR_IDENTIFIER_TEMPLATE
from ml.schema import schema_to_identifier_template
from ml.schema import validate_identifier_template


def test_validate_identifier_template_requires_instrument_id() -> None:
    with pytest.raises(ValueError):
        validate_identifier_template("missing", label="schema template")


def test_schema_template_defaults_match_bar_template() -> None:
    assert schema_to_identifier_template("bars") == DEFAULT_BAR_IDENTIFIER_TEMPLATE
```

**Step 2: Run the new tests (expect failure)**
Run: `poetry run pytest ml/tests/unit/schema/test_schema_identifier_templates.py -q`
Expected: FAIL (tests not implemented / missing import).

**Step 3: Implement the tests with proper imports**
Ensure all tests import from `ml.schema` and use `DatasetType` where needed. Keep names aligned with `nautilus-test-writer` naming conventions.

**Step 4: Re-run the tests**
Run: `poetry run pytest ml/tests/unit/schema/test_schema_identifier_templates.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add ml/tests/unit/schema/test_schema_identifier_templates.py
git commit -m "test: add schema identifier template contracts"
```

---

### Task 10: Stabilize validate_wave tests

**Files:**
- Modify: `ml/scripts/validate_wave.py`
- Modify: `ml/tests/unit/scripts/test_validate_wave.py`

**Step 1: Add failing test coverage for current failure**
If the failures stem from changed defaults, add a test case in `test_validate_wave.py` that captures the new expected behavior.

**Step 2: Run targeted tests (expect failure)**
Run: `poetry run pytest ml/tests/unit/scripts/test_validate_wave.py -q`
Expected: FAIL (current regression).

**Step 3: Fix validate_wave behavior**
Adjust `ml/scripts/validate_wave.py` to align with `validation_bundle` defaults (or update the tests if defaults are correct). Keep logging with `exc_info=True` when catching errors per AGENTS.md.

**Step 4: Re-run tests**
Run: `poetry run pytest ml/tests/unit/scripts/test_validate_wave.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add ml/scripts/validate_wave.py ml/tests/unit/scripts/test_validate_wave.py
git commit -m "fix: stabilize validate_wave checks"
```

---

### Task 11: Final validation and coverage gate

**Files:**
- (No file changes; validation only)

**Step 1: Run type checks and lint**
Run: `poetry run mypy ml --strict`
Expected: PASS.

**Step 2: Run ruff**
Run: `poetry run ruff check ml`
Expected: PASS.

**Step 3: Run targeted pytest shards**
Run: `poetry run pytest -k "orchestration or scheduler or tft_dataset_builder or dashboard or trainer or strategy" -q`
Expected: PASS.

**Step 4: Run coverage**
Run:
```bash
coverage run -m pytest ml/tests/
coverage report
```
Expected: ML coverage >= 90%.

**Step 5: Commit the validation checkpoint (optional)**
```bash
git status -sb
# No changes expected; if any incidental updates exist, commit with "chore: validation cleanup".
```

---

Plan complete and saved to `docs/plans/2026-01-16-legacy-removal-phase1.md`.
Two execution options:
1. Subagent-Driven (this session) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open a new session with executing-plans, batch execution with checkpoints

Which approach?
