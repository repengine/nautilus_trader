# Handoff: Test Suite Investigation - Next Claude Session

**Date:** 2025-10-27
**Branch:** feat/strategy-integration
**Current State:** 57 failures, 2,936 passed (97.5% pass rate)
**Previous Session Duration:** ~4 hours
**Status:** Investigation complete, hypothesis invalidated, ready for fresh approach

---

## Quick Start (Read This First!)

```bash
# 1. Verify current baseline
pytest ml/tests --tb=no -q 2>&1 | tail -5
# Expected: 57 failed, 2936 passed

# 2. Read this handoff completely before starting
cat HANDOFF_NEXT_CLAUDE.md

# 3. Read session summary for full context
cat ml/tests/SESSION_SUMMARY_AND_LESSONS.md

# 4. Start with Option A (below) - Debug ONE test
```

---

## Executive Summary

**Previous session attempted comprehensive investigation + systematic fixes based on fixture pollution hypothesis. This approach FAILED.**

**Key Discovery:** Tests pass individually but fail in full suite (100% individual pass rate observed).

**Why Previous Approach Failed:**
1. Trusted outdated handoff document with wrong assumptions
2. Investigated hypothetical pollution sources instead of actual errors
3. Implemented fix (clear `_SCHEMA_INITIALIZED`) that had zero impact
4. Never looked at actual error messages from failing tests

**What Next Claude Should Do:**
1. **DEBUG ACTUAL ERRORS** - Run ONE failing test with traceback
2. **Fix THAT specific error** - Not theoretical pollution
3. **Measure impact** - Verify failure count reduces
4. **Repeat** - Incremental, measured progress

---

## Current State Details

### Test Results (Verified 2025-10-27)
```
57 failed, 2936 passed, 22 skipped, 91 deselected
Runtime: 38 minutes (2294 seconds)
Pass Rate: 97.5%
```

### Code Changes Made This Session
```
File: ml/tests/conftest.py
Lines: 1370-1373 (4 lines added)
Change: Clear _SCHEMA_INITIALIZED dictionary in test_database fixture
Impact: ZERO (57 failures before and after)
Status: May revert or keep as defensive measure (your call)
```

### Key Insight: Isolation Pattern Confirmed

**Tests pass individually (100%) but fail in full suite:**

```bash
# Evidence from previous session:
# Individual tests: 13/13 PASS (100%)
# Module tests: 64/64 PASS (100%)
# Full suite: 57 FAIL (2% fail rate)
```

**This confirms:** Test ordering dependencies exist, BUT the root cause is NOT what we thought.

---

## What Previous Session Tried

### Investigation Phase (60 min - 81KB of reports)

**4 Parallel Explore Agents Generated:**
1. FIXTURE_USAGE_ANALYSIS.md - Proved only 2 files use store_bundle (invalidated F8 strategy)
2. ERROR_CATEGORIZATION.md - Categorized actual errors (we should have read this more carefully!)
3. ISOLATION_ANALYSIS.md - Confirmed tests pass individually
4. STATE_POLLUTION_SOURCES.md - Identified 13 theoretical pollution sources

**Reports Location:** `/home/nate/projects/nautilus_trader/ml/tests/`

### Implementation Phase (90 min - Phase 2 → 3 → 4 workflow)

**Task 1.1: Clear _SCHEMA_INITIALIZED**
- Hypothesis: Module-level dict causes schema corruption pollution
- Implementation: ✅ PASS (clean code, 4 lines)
- Static Validation: ✅ PASS (ruff, mypy clean)
- Integration Validation: ❌ FAIL (zero impact on failures)

**Conclusion:** The hypothesis was wrong. _SCHEMA_INITIALIZED is not causing the 57 failures.

---

## Why Tests Pass Individually But Fail In Suite

**Three Possible Explanations:**

### Theory 1: Shared State Pollution (What We Assumed)
- Tests share module-scoped fixtures
- First test corrupts state, subsequent tests fail
- **Status:** Partially disproven (Task 1.1 had no impact)

### Theory 2: Resource Exhaustion Over Time
- Connection pool exhaustion after 2000+ tests
- Memory leaks accumulating
- File handle limits
- **Status:** Not investigated yet

### Theory 3: Actual Code Bugs Masked By Test Order
- Tests have real bugs but compensating errors cancel out
- Early tests set up state that later tests depend on
- Import side effects
- **Status:** Most likely - need to debug actual errors

---

## Critical: What ERROR_CATEGORIZATION.md Actually Says

**We generated this report but didn't act on it!** Here are the ACTUAL error types:

### High-Priority Errors (Should debug these first)

1. **AttributeError / Missing Methods (5 failures)**
   - `_StubRegistry` missing `flush()` method
   - ONNX mocks returning wrong types
   - Actor registration failures
   - **These are real code bugs!**

2. **Assertion Failures - Data Validation (6 failures)**
   - Feature metadata unexpected flags
   - Multi-signal confidence bounds violations
   - Metrics calculation errors (0.0 vs 10.0)
   - **These are real logic bugs!**

3. **Engine/Registry Shared State (3 failures)**
   - EngineManager not returning cached instances
   - Multiple engine instances for same URL
   - **This might be fixture pollution - worth investigating**

4. **Critical Blockers:**
   - Cython ABI `KeyError: '__pyx_vtable__'` (1 failure)
   - `RuntimeError: no current event loop in MainThread` (1 failure)
   - **These block entire test categories!**

---

## Recommended Investigation Strategy

### Option A: Debug ONE Failure - Start Here (RECOMMENDED)

**Goal:** Fix one specific failure, measure impact, learn pattern, repeat.

```bash
# Step 1: Pick ONE failing test from registry cluster (related failures)
TEST="ml/tests/unit/registry/test_deployment_manager.py::TestRegistryDeployment::test_registry_basic_deploy"

# Step 2: Run with FULL traceback
pytest $TEST -xvs 2>&1 | tee debug_output.txt

# Step 3: Read the ACTUAL error
# Look for:
# - AttributeError? → Missing method/property
# - AssertionError? → Wrong value/behavior
# - ImportError? → Missing dependency
# - RuntimeError? → Async/event loop issue

# Step 4: Fix THAT specific error
# Edit the code that's actually broken

# Step 5: Verify fix works
pytest $TEST -xvs
# Should now pass

# Step 6: Run full suite
pytest ml/tests --tb=no -q 2>&1 | tail -3
# Count failures - should be 56 or less

# Step 7: Commit if improved
git add <files>
git commit -m "fix(tests): resolve <specific issue>"

# Step 8: Repeat with next failure
```

**Expected Result:** Each fix reduces failures by 1-4 tests (related failures).

**Time Estimate:** 15-30 minutes per failure, 10-20 hours total for all 57.

### Option B: Focus on Registry Cluster (4 Related Failures)

**Hypothesis:** All 4 registry failures have common root cause.

```bash
# All 4 failures are in these files:
# - test_deployment_manager.py (2 failures)
# - test_registry_first_export.py (2 failures)

# Step 1: Run registry tests individually
pytest ml/tests/unit/registry/test_deployment_manager.py -xvs 2>&1 | tee registry_debug.txt

# Step 2: Analyze common patterns
grep -E "(Error|Traceback|FAILED)" registry_debug.txt

# Step 3: Fix common root cause
# Likely: Missing method, wrong mock, import issue

# Step 4: Verify all 4 fixed
pytest ml/tests/unit/registry/ -v
pytest ml/tests/unit/training/test_registry_first_export.py -v

# Step 5: Measure impact
pytest ml/tests --tb=no -q 2>&1 | tail -3
# Should be 53 failures (4 less)
```

**Expected Result:** 4 failures resolved with one fix.

**Time Estimate:** 1-2 hours.

### Option C: Debug Event Loop Blocker (Critical Priority)

**If asyncio error blocks entire test categories:**

```bash
# Find the event loop error
pytest ml/tests --tb=long -x 2>&1 | grep -A 10 "RuntimeError.*event loop"

# This might be blocking many tests
# Fix this first, might resolve 10+ failures
```

### Option D: Investigate Resource Exhaustion

**Test the theory that failures are from exhaustion, not pollution:**

```bash
# Run first 1000 tests
pytest ml/tests --maxfail=1000 -x 2>&1 | tee first_1000.txt
grep FAILED first_1000.txt | wc -l

# If failures only appear after N tests, it's exhaustion
# If failures appear early, it's actual bugs
```

---

## What NOT To Do

### ❌ Don't Trust Previous Investigation Reports Blindly

The reports are well-written but based on wrong hypothesis. Use them as reference, not gospel.

### ❌ Don't Start With Comprehensive Fixes

We tried this. It doesn't work without understanding actual errors.

### ❌ Don't Implement More Theoretical Pollution Fixes

Task 1.1 (clear _SCHEMA_INITIALIZED) had zero impact. The other 3 "critical fixes" will likely fail too.

### ❌ Don't Assume Fixture Pollution Is The Problem

Only 2 test files use store_bundle. The 57 failures are spread across many unrelated files.

---

## Quick Reference: 57 Failing Tests

**Last 4 failures from test run (representative sample):**
```
FAILED ml/tests/unit/training/test_registry_first_export.py::TestCreateModelManifestStub::test_auto_detect_architecture
FAILED ml/tests/unit/training/test_registry_first_export.py::TestEndToEndTrainingIntegration::test_complete_training_to_registry_flow
FAILED ml/tests/unit/registry/test_deployment_manager.py::TestRegistryDeployment::test_registry_basic_deploy
FAILED ml/tests/unit/registry/test_deployment_manager.py::TestRegistryDeployment::test_registry_hot_reload
```

**To get full list:**
```bash
pytest ml/tests --tb=no -q 2>&1 | grep FAILED > current_failures.txt
cat current_failures.txt
```

---

## Resources Available

### Investigation Reports (May Use As Reference)
```
ml/tests/SYNTHESIS_AND_ACTION_PLAN.md - Overall strategy (based on wrong hypothesis)
ml/tests/FIXTURE_USAGE_ANALYSIS.md - Fixture dependency analysis
ml/tests/ERROR_CATEGORIZATION.md - ACTUAL error types (READ THIS!)
ml/tests/ISOLATION_ANALYSIS.md - Proves individual vs suite pattern
ml/tests/STATE_POLLUTION_SOURCES.md - Theoretical pollution sources
ml/tests/SESSION_SUMMARY_AND_LESSONS.md - Complete session retrospective
```

### Implementation Reports (Task 1.1)
```
reports/implementations/task_1_1_schema_initialized_report.md
reports/validations/task_1_1_static_validation_report.md
reports/validations/task_1_1_integration_validation_report.md
```

### Task Definitions
```
tasks/fixture_refactoring/task_1_1_clear_schema_initialized.md
tasks/fixture_refactoring/task_f8_1_pipeline_tasks_migration.md (abandoned)
```

---

## Decision: Keep or Revert Task 1.1?

**Task 1.1 Change:**
```python
# File: ml/tests/conftest.py, lines 1370-1373
# CRITICAL: Clear schema initialization tracking to prevent state poisoning
from ml.tests.fixtures.database_fixtures import _SCHEMA_INITIALIZED
_SCHEMA_INITIALIZED.clear()
_logger.debug("Schema initialization tracking cleared")
```

**Arguments to Keep:**
- ✅ Defensive measure (prevents future pollution)
- ✅ Clean code, passes all validation
- ✅ No performance impact
- ✅ No regressions introduced

**Arguments to Revert:**
- ❌ Zero measurable impact on current failures
- ❌ Adds complexity without solving problem
- ❌ Based on wrong hypothesis

**Recommendation:** Keep it (defensive), but don't implement Tasks 1.2-1.4 until you verify they'll help.

---

## Success Criteria for Next Session

### Minimum Success
- ✅ Debug and fix 1 specific failure with traceback
- ✅ Verify fix reduces failure count (57 → 56 or less)
- ✅ Understand actual error pattern
- ✅ Commit with clear explanation

### Good Success
- ✅ Fix registry cluster (4 failures → 0)
- ✅ Establish pattern for fixing remaining failures
- ✅ Document actual root causes found
- ✅ Create plan for remaining ~53 failures

### Excellent Success
- ✅ Fix 10+ failures by finding common patterns
- ✅ Identify and fix critical blockers (event loop, Cython ABI)
- ✅ Reduce failures to 40 or less
- ✅ Document repeatable fix process

---

## Commands to Start Investigation

```bash
# 1. Establish current baseline
pytest ml/tests --tb=no -q 2>&1 | tee baseline.txt | tail -5

# 2. Get full failure list
pytest ml/tests --tb=no -q 2>&1 | grep FAILED > failures.txt

# 3. Pick first registry failure
TEST=$(grep "test_deployment_manager" failures.txt | head -1 | cut -d' ' -f2)
echo "Debugging: $TEST"

# 4. Run with full traceback
pytest $TEST -xvs 2>&1 | tee debug.txt

# 5. Read the actual error
less debug.txt
# Scroll to bottom, read the actual exception

# 6. Fix the code that's broken
# (not the fixtures, not the pollution - the actual broken code)

# 7. Verify
pytest $TEST -xvs
# Should pass

# 8. Measure impact
pytest ml/tests --tb=no -q 2>&1 | tail -3
# Should see fewer failures
```

---

## Questions to Answer During Investigation

1. **Do failures cluster by error type?**
   - All AttributeErrors together?
   - All registry failures related?
   - Helps prioritize fixes

2. **Do failures appear early or late in suite?**
   - Early = actual bugs
   - Late = exhaustion/pollution
   - Run: `pytest ml/tests -x` to see first failure

3. **Are there critical blockers?**
   - Event loop errors?
   - Import errors?
   - Fix these first (high impact)

4. **Do any fixes resolve multiple failures?**
   - Registry fix → 4 tests pass?
   - Pattern recognition matters

---

## Final Advice

**START SMALL. DEBUG ACTUAL ERRORS. MEASURE PROGRESS.**

Don't repeat the previous session's mistake of comprehensive investigation before understanding the actual problems.

Run ONE test. See the error. Fix the error. Measure. Commit. Repeat.

The 57 failures are likely 57 small bugs, not one big pollution issue.

---

## Contact Information for Questions

If you get stuck or need clarification:

1. Read ml/tests/ERROR_CATEGORIZATION.md for actual error types
2. Read ml/tests/SESSION_SUMMARY_AND_LESSONS.md for what went wrong
3. Check git log for recent commits that might have introduced regressions
4. Consider that some failures might be expected/acceptable (ask user)

---

**Good luck! Focus on actual errors, not theoretical pollution.**

**Remember:** Every test that passes individually but fails in suite has a SPECIFIC reason. Find it. Fix it. Move on.
