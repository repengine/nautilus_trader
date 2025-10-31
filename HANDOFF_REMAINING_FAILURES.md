# Investigation Results: Test Suite Flakiness and Remaining Failures

**Date:** 2025-10-31
**Investigator:** Claude Code
**Starting Point:** 12 failures, 2898 passed (99.6% pass rate)
**Baseline Commit:** f925feba2

## Executive Summary

The test suite exhibits **severe flakiness** due to test isolation issues. Of the 4 "consistently failing" tests identified by comparing the handoff document with my baseline:

- **All 4 tests PASS when run individually** ✅
- **All 4 tests FAIL when run in full suite** ❌

This confirms the critical finding from the handoff document: **"12 of 13 tests pass individually but fail in full suite"**.

## What I Fixed

### 1. Enum Identity Issue in test_deployment_manager ✅

**Files Modified:**
- `ml/tests/unit/registry/test_deployment_manager.py`

**Changes:**
```python
# BEFORE (fails with module reloading)
assert model_info.deployment_status == DeploymentStatus.ACTIVE
assert model_v2_info.deployment_status == DeploymentStatus.ACTIVE
assert model_v1_info.deployment_status == DeploymentStatus.RETIRED

# AFTER (stable)
assert model_info.deployment_status.value == "active"
assert model_v2_info.deployment_status.value == "active"
assert model_v1_info.deployment_status.value == "retired"
```

**Commit:** 78a7f576e

---

## Consistently Failing Tests (from intersection analysis)

I compared the handoff document's list of failures with my baseline run to find tests that appear in BOTH lists. These 4 tests are the most likely to be consistent failures:

### 1. test_registry_basic_deploy ✅ FIXED
**Test:** `ml/tests/unit/registry/test_deployment_manager.py::TestRegistryDeployment::test_registry_basic_deploy`
**Status:** Fixed with enum .value comparison
**Root Cause:** Enum identity check fails with module reloading
### 2. test_manifest_adapter_class_dynamic_threshold ⚠️ PASSES INDIVIDUALLY

**Test:** `ml/tests/unit/actors/test_signal_adapter_loading.py::test_manifest_adapter_class_dynamic_threshold`

**Individual Run:** PASSES ✅

```bash
pytest ml/tests/unit/actors/test_signal_adapter_loading.py::test_manifest_adapter_class_dynamic_threshold -xvs
# Result: PASSED
```

**Full Suite Run:** FAILS ❌
**Root Cause:** Test isolation issue - likely mock state pollution or module-level import caching

### 3. test_load_model_verifies_integrity ⚠️ PASSES INDIVIDUALLY

**Test:** `ml/tests/integration/registry/test_model_registry_security.py::TestModelRegistryIntegrity::test_load_model_verifies_integrity`

**Individual Run:** PASSES ✅

```bash
pytest ml/tests/integration/registry/test_model_registry_security.py::TestModelRegistryIntegrity::test_load_model_verifies_integrity -xvs
# Result: PASSED
```

**Full Suite Run:** FAILS ❌
**Root Cause:** Test isolation issue - likely ONNX mock state or registry pollution

### 4. test_run_pipeline_schedule_interval ⚠️ PASSES INDIVIDUALLY

**Test:** `ml/tests/unit/tasks/test_pipeline_scheduler.py::test_run_pipeline_schedule_interval`

**Individual Run:** PASSES ✅

```bash
pytest ml/tests/unit/tasks/test_pipeline_scheduler.py::test_run_pipeline_schedule_interval -xvs
# Result: PASSED
```

**Full Suite Run:** FAILS ❌
**Root Cause:** Test isolation issue - likely environment variable or config pollution

---

## Critical Findings

### Finding 1: Test Suite is Highly Flaky 🚨

Running the full suite twice produced COMPLETELY different failure sets:

**Baseline Run (12 failures):**
- test_feature_store_honors_env_topic_scheme_and_prefix
- test_load_model_verifies_integrity
- test_batch_size_property_preservation
- test_store_metrics_snapshot_aggregates_real_data
- test_manifest_adapter_class_dynamic_threshold
- test_metrics_manager_counter_and_gauge
- test_secure_onnx_load_tampered_model_non_strict
- test_ensure_macro_ready_combines_results
- test_registry_basic_deploy
- test_run_pipeline_schedule_interval
- test_run_pipeline_schedule_sets_environment
- test_infer_batch_falls_back_to_per_row_on_ort_error

**After Enum Fix Run (18 failures):**
- test_12_concurrent_writes
- test_fresh_store_bundle_consistency[1]
- test_fresh_store_bundle_consistency[3]
- test_telemetry_emission[emergency_stop-ml_dashboard_actions_total]
- test_time_range_boundary_invariant
- test_actor_side_domain_event_bridge_publishes
- test_manifest_adapter_class_dynamic_threshold
- test_metrics_manager_counter_and_gauge
- test_load_model_verifies_integrity
- test_auto_deploy_student_targets_signal_actor
- test_task_uses_tier1_symbols
- test_l2_task_populate_l2_efficient_builds_loader_config
- test_pipeline_runner_run_pipeline_initialises_runner
- test_secure_onnx_load_with_session_options
- test_secure_onnx_load_tampered_model_non_strict
- test_run_pipeline_schedule_interval
- test_run_pipeline_schedule_sets_environment
- (+ more)

**Only 3 tests appear in both lists:**
- test_manifest_adapter_class_dynamic_threshold
- test_load_model_verifies_integrity
- test_run_pipeline_schedule_interval

### Finding 2: Enum Pattern is Proven ✅

The enum .value comparison pattern works reliably:
- Applied successfully in commit f925feba2 (9 tests fixed)
- Applied successfully in commit 78a7f576e (1 test fixed)
- **Total: 10 tests fixed with this pattern**

### Finding 3: Individual Test Runs are USELESS ❌

**Do NOT trust individual test runs.** The handoff was RIGHT about this:
- All 4 "consistent" failures pass individually
- Only full suite execution reveals the real issues
- Validation MUST use `make pytest-ml`

---

## Root Cause Categories

Based on the tests that fail in suite but pass individually, the issues are:

### 1. Mock State Pollution
**Affected Tests:**
- test_manifest_adapter_class_dynamic_threshold
- test_load_model_verifies_integrity
- test_telemetry_emission tests
- test_metrics_manager_counter_and_gauge

**Pattern:** Previous tests leave mocks configured that interfere with later tests

### 2. Module-Level Import Caching
**Affected Tests:**
- test_actor_side_domain_event_bridge_publishes
- test_auto_deploy_student_targets_signal_actor

**Pattern:** Module-level imports cache objects/classes that have stale state

### 3. Global Registry/Cache Pollution
**Affected Tests:**
- test_run_pipeline_schedule_interval
- test_run_pipeline_schedule_sets_environment
- test_feature_store_honors_env_topic_scheme_and_prefix

**Pattern:** Global state (Prometheus metrics, config, env vars) not cleaned between tests

### 4. Database/Store State Pollution
**Affected Tests:**
- test_12_concurrent_writes
- test_fresh_store_bundle_consistency
- test_store_metrics_snapshot_aggregates_real_data

**Pattern:** Database or store state leaks between tests

---

## Recommendations for Next Investigator

### Priority 1: Establish Stable Baseline (1-2 hours)

Run the full suite **3 times** and identify tests that fail in ALL 3 runs:

```bash
for i in 1 2 3; do
    echo "=== RUN $i ==="
    make pytest-ml 2>&1 | tee /tmp/run_$i.log
    grep "^FAILED" /tmp/run_$i.log > /tmp/failures_$i.txt
done

# Find tests that fail in ALL 3 runs
comm -12 <(sort /tmp/failures_1.txt) /tmp/failures_2.txt | \
  comm -12 - /tmp/failures_3.txt > /tmp/truly_consistent_failures.txt
```

**Only fix tests in truly_consistent_failures.txt**

### Priority 2: Look for More Enum Identity Issues (30 min)

Search for other enum identity checks:

```bash
# Find all enum comparisons in tests
grep -r "assert.*==.*\." ml/tests/ | grep -E "Status|Requirements|Role|Stage|Source" > /tmp/enum_checks.txt

# Look for patterns like:
# assert x.status == SomeStatus.ACTIVE
# assert y.role == ModelRole.INFERENCE
```

Apply the `.value` pattern to these.

### Priority 3: Investigate Mock State (2-3 hours)

For tests that pass individually but fail in suite:

1. Read the test code
2. Look for `@patch` or `Mock()` usage
3. Check if previous tests in the same file use related mocks
4. Add cleanup fixtures or use `mocker` fixture from pytest-mock

### Priority 4: Check Global State Cleanup (1-2 hours)

Review `ml/tests/conftest.py` for cleanup fixtures:
- Are Prometheus metrics being cleared?
- Are environment variables being restored?
- Are module-level caches being reset?

Look at commit `e2b34bbd2` for examples of conditional cleanup.

### Priority 5: Database Isolation (2-3 hours)

For database-related failures:
- Check if transactions are being rolled back
- Verify test database is being cleaned between tests
- Look for @pytest.mark.serial marker usage

---

## Tools and Commands

### Find Common Failures
```bash
comm -12 <(sort /tmp/handoff_failures.txt) <(sort /tmp/my_failures.txt)
```

### Run Full Suite
```bash
make pytest-ml | tee /tmp/suite_run.log
grep "^FAILED" /tmp/suite_run.log | wc -l  # Count failures
```

### Find Enum Comparisons
```bash
grep -n "== [A-Z][a-zA-Z]*\.[A-Z_]" ml/tests/unit/path/to/test.py
```

### Check Test in Isolation
```bash
pytest ml/tests/path/to/test.py::test_name -xvs
```

---

## What NOT to Do ❌

1. **Don't fix tests based on individual runs** - they lie
2. **Don't run the suite just once** - it's too flaky
3. **Don't batch multiple unrelated fixes** - makes debugging harder
4. **Don't trust that fixing one test won't break others** - always validate with full suite

---

## Current Status

**Commit:** 78a7f576e
**Tests Fixed:** 1 (test_registry_basic_deploy)
**Tests Still Failing:** ~11-17 (varies by run)
**Pass Rate:** ~99.4-99.6%

**Next Steps:**
1. Run suite 3 times to establish stable baseline
2. Fix truly consistent failures only
3. Focus on enum identity issues first (proven pattern)
4. Investigate mock/global state for others

---

## References

- **Previous successful pattern:** Commit f925feba2 (enum .value fixes)
- **Test isolation cleanup:** Commit e2b34bbd2
- **Original handoff:** HANDOFF_NEXT_CLAUDE.md (from previous session)
- **Baseline commit:** f925feba2

---

*Investigation Date: 2025-10-31*
*Investigator: Claude Code*
*Conclusion: Severe test isolation issues; individual test runs cannot be trusted; focus on stable baseline first*
