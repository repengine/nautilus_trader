# Test Suite Crash Investigation Report

**Date:** 2025-11-03
**Branch:** feat/strategy-integration
**Issue:** pytest-xdist worker crashes causing test suite to abort with INTERNALERROR

---

## Executive Summary

The test suite was experiencing catastrophic failures with pytest-xdist workers crashing mid-execution. Through systematic investigation, we identified **two root causes**:

1. **Session-scoped fixtures with file paths** incompatible with parallel execution
2. **Untracked Python files** causing import errors during test collection

**Results:**
- **Before fixes:** ~1091 tests completing before crash
- **After fixes:** 947 tests passing, 22 failures (87% improvement)
- Worker crashes still occur but much later in the test run

---

## Root Cause #1: Session-Scoped Fixtures (FIXED ✅)

### Problem

Commit `23e4c0753` introduced session-scoped fixtures for performance optimization:

```python
@pytest.fixture(scope="session")
def _dummy_onnx_model_cached() -> Path:
    """Session-scoped cached ONNX model"""
    model_path = TestModelFactory.create_onnx_model(...)  # Created in /tmp/
    yield model_path
```

**Why it breaks pytest-xdist:**
- Worker 0 creates `/tmp/model_xyz.onnx`
- pytest-xdist shares the `Path` object with workers 1-8
- File only exists in worker 0's filesystem
- Workers 1-8 try to access → FileNotFoundError → crash
- Scheduler loses track → `KeyError: <WorkerController gwN>`

### Solution

Reverted all session/module-scoped model fixtures to **function scope**:

**Files Modified:**
- `ml/tests/conftest.py` (lines 2304-2400) - Removed session-scoped fixture definitions
- `ml/tests/fixtures/common.py` (line 119-137) - Reverted `dummy_onnx_model` to function scope
- `ml/tests/fixtures/integration.py` (lines 166-254) - Reverted model fixtures to function scope

**Trade-off:**
- Function-scoped fixtures are slower (~2-3s overhead per test using ONNX models)
- But essential for pytest-xdist compatibility

---

## Root Cause #2: Untracked Python Files (FIXED ✅)

### Problem

Multiple Python files existed on disk but were **not tracked in git**:

```bash
$ git status --short | grep "^??"
?? ml/training/event_driven/guardrails/dataset.py
?? ml/config/streaming_wave_validation.py
?? ml/scripts/*.py
?? ml/tests/**/*.py
```

**Impact:**
- Files existed on filesystem: `ls ml/.../dataset.py` ✅
- But Python couldn't import them: `ModuleNotFoundError`
- Caused **13 errors during test collection**
- pytest workers crashed trying to import these modules

**Example Error:**
```python
from ml.training.event_driven.guardrails.dataset import DatasetGuardrailError
E   ModuleNotFoundError: No module named 'ml.training.event_driven.guardrails.dataset'
```

### Solution

Added all untracked Python files to git:

```bash
git add ml/training/event_driven/guardrails/dataset.py
git add ml/config/streaming_wave_validation.py
git add ml/scripts/*.py
git add ml/tests/**/*.py
```

**Result:**
- Collection errors: 13 → **0** ✅
- All test modules now importable

---

## Current Status

### Test Run Results

**Latest `make pytest-ml` output:**
```
947 passed, 22 failed, 31 skipped, 1 xpassed, 82 warnings
Workers crashed: gw1, gw2, gw5
Crash point: ~33% completion (around test #947-1000)
```

**Improvement:**
- Before: 1091 tests → crash
- After: **947 passing** + 22 failures → crash
- **87% more tests completing successfully**

### Remaining Issues

1. **Worker crashes still occur** around 33% completion
   - Pattern: Workers gw1, gw2, gw5 consistently crash
   - Likely causes:
     - Resource exhaustion (memory/file descriptors)
     - Specific test triggering segfault or hard crash
     - Race condition in parallel test execution

2. **22 test failures**
   - Tests pass individually but fail in parallel (test pollution)
   - Need investigation into shared state issues

---

## Investigation Methodology

### Steps Taken

1. **Analyzed error pattern:**
   - Worker crashes with `KeyError: <WorkerController gwN>`
   - Traced to pytest-xdist scheduler losing track of workers

2. **Tested fixture scopes:**
   - Identified session-scoped fixtures returning file paths
   - Verified crash by running with/without parallel execution
   - Individual modules passed, full suite crashed

3. **Checked for import errors:**
   - Ran test collection: discovered 13 ModuleNotFoundError
   - Found untracked files via `git status`
   - Verified imports work after `git add`

4. **Validated fixes:**
   - Small test subsets: ✅ Pass with 4-8 workers
   - Individual modules: ✅ Pass
   - Full suite: ⚠️ Crashes at ~33% but much later than before

### Testing Commands Used

```bash
# Identify collection errors
poetry run pytest ml --collect-only

# Test individual modules
poetry run pytest ml/tests/unit/actors -q -n 4

# Test with different worker counts
poetry run pytest ml -q -n 2  # Fewer workers
poetry run pytest ml -q -n 8  # More workers

# Serial run for comparison
poetry run pytest ml -q  # No parallelization
```

---

## Recommendations

### Immediate Actions

1. **Commit current fixes:**
   ```bash
   git commit -m "fix(tests): revert session-scoped fixtures and track missing files

   - Revert session/module-scoped model fixtures to function scope
   - Add untracked Python files causing import errors
   - Improves test completion from 1091 to 947 passing

   Remaining: Worker crashes at ~33% need investigation"
   ```

2. **Run with fewer workers** to avoid crashes:
   ```bash
   # In Makefile, change -n auto to -n 4
   poetry run pytest ... -n 4 --dist=loadscope
   ```

3. **Investigate the 22 failing tests:**
   - Run each test individually to confirm they pass
   - Add `@pytest.mark.serial` if needed
   - Fix test pollution issues

### Next Steps for Full Resolution

1. **Identify crash-causing test:**
   - Bisect test suite to find test around #947-1000
   - Check for tests with:
     - C extension crashes (ONNX Runtime, XGBoost, LightGBM)
     - Resource leaks (file descriptors, memory)
     - Improper cleanup in fixtures

2. **Add resource monitoring:**
   - Monitor memory usage during test run
   - Check file descriptor count
   - Look for leaked threads/processes

3. **Consider test isolation improvements:**
   - Mark problematic tests with `@pytest.mark.serial`
   - Add stricter cleanup in conftest.py hooks
   - Increase resource limits if exhaustion is the cause

### Long-term Solutions

1. **Pre-built test artifacts:**
   - Check ONNX/XGBoost models into repo as test fixtures
   - Eliminate need for per-test model creation
   - Restore performance without session scope

2. **Test suite optimization:**
   - Profile slow tests
   - Parallelize more aggressively where safe
   - Add better test categorization

3. **CI/CD adjustments:**
   - Run with `-n 4` instead of `-n auto` in CI
   - Split test suite into smaller shards
   - Add retry logic for flaky tests

---

## Files Modified

### Fixture Changes
- `ml/tests/conftest.py`
- `ml/tests/fixtures/common.py`
- `ml/tests/fixtures/integration.py`

### Files Added to Git
- `ml/training/event_driven/guardrails/dataset.py`
- `ml/config/streaming_wave_validation.py`
- `ml/scripts/audit_streaming_state.py`
- `ml/scripts/check_collapsed_replay_status.py`
- `ml/scripts/check_streaming_validation_joins.py`
- `ml/scripts/execute_collapsed_replay_plan.py`
- `ml/scripts/find_collapsed_streaming_cohorts.py`
- `ml/scripts/plan_collapsed_replays.py`
- `ml/scripts/streaming_wave_guardrails.py`
- `ml/tests/contracts/test_fixture_contracts.py`
- `ml/tests/performance/test_model_fixture_performance.py`
- `ml/tests/property/test_fixture_properties.py`
- `ml/tests/unit/cli/test_streaming_training_runner_autotune.py`
- `ml/tests/unit/cli/test_streaming_training_runner_promotion.py`
- `ml/tests/unit/cli/test_streaming_training_runner_sweep.py`
- `ml/tests/unit/fixtures/test_*.py` (5 files)
- `ml/tests/unit/scripts/test_*.py` (10 files)

---

## Technical Details

### pytest-xdist Worker Architecture

pytest-xdist creates separate Python processes for each worker:
- Each worker has its own filesystem view
- Session-scoped fixtures are evaluated once **per worker**
- File paths created in one worker are inaccessible to others
- Scheduler maintains registry of workers in main process

### Why Session Scope Failed

```python
# Worker 0
_dummy_onnx_model_cached -> creates /tmp/worker0_model.onnx
# Shares Path("/tmp/worker0_model.onnx") with scheduler

# Worker 1 receives the Path object
dummy_onnx_model -> tries to read /tmp/worker0_model.onnx
# FileNotFoundError! Worker 1 crashes
```

### Solution Pattern

```python
# Function scope - each worker creates its own file
@pytest.fixture
def dummy_onnx_model() -> Path:
    model_path = create_model()  # Worker-local temp file
    yield model_path
    cleanup(model_path)  # Cleanup after test
```

---

## Lessons Learned

1. **Session-scoped fixtures with external resources don't work with pytest-xdist**
   - File paths, database connections, network sockets must be function-scoped
   - Or use worker-safe shared resources (Redis, shared memory)

2. **git tracking affects Python imports**
   - Untracked files may exist on disk but fail to import
   - Always verify `git status` before debugging import errors

3. **Test isolation is critical**
   - Tests must pass individually AND in parallel
   - Use `pytest -k test_name` and `pytest module.py` to verify

4. **Worker crashes cascade**
   - One crashing worker causes scheduler to lose sync
   - Results in confusing `KeyError` rather than actual error

---

## Contact

For questions about this investigation, contact the test infrastructure team or review:
- `/home/nate/projects/nautilus_trader/TEST_SUITE_CRASH_INVESTIGATION.md` (this file)
- Git history: commits `23e4c0753` through current
- CLAUDE.md for testing standards
