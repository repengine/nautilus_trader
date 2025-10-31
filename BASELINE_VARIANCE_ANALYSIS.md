# Baseline Variance Analysis: 4 Consecutive Test Runs
**Date:** 2025-10-31
**Commit:** 78a7f576e (with enum fix applied)
**No changes between runs**

## Summary Statistics

| Run | Failures | Passed | Pass Rate |
|-----|----------|--------|-----------|
| 1   | 13       | 2907   | 99.55%    |
| 2   | 15       | 2905   | 99.49%    |
| 3   | 8        | 2912   | 99.73%    |
| 4   | 13       | 2907   | 99.55%    |

**Variance:** 8-15 failures (87.5% variance)
**Mean:** 12.25 failures
**This confirms SEVERE test flakiness**

## Truly Consistent Failures (4/4 runs)

Only **2 tests** failed in ALL 4 runs:

1. `ml/tests/integration/test_feature_store_integration.py::TestFeatureStoreIntegration::test_feature_store_config_propagation`
2. `ml/tests/unit/fixtures/test_fresh_store_bundle.py::test_fresh_store_bundle_consistency[3]`

## Nearly Consistent Failures (3/4 runs)

4 tests failed in 3 out of 4 runs:

1. `ml/tests/unit/fixtures/test_fresh_store_bundle.py::test_fresh_store_bundle_consistency[1]` (runs 1, 2, 3)
2. `ml/tests/unit/fixtures/test_fresh_store_bundle.py::test_fresh_store_bundle_consistency[2]` (runs 1, 3, 4)
3. `ml/tests/integration/registry/test_model_registry_security.py::TestModelRegistryIntegrity::test_register_model_permission_error` (runs 1, 2, 4)
4. `ml/tests/contracts/test_base_actor_initialization.py::TestBaseMLInferenceActorInitialization::test_model_loader_initialization_with_default_loader` (runs 1, 2, 3)

## Recommended Action Plan

### Phase 1: Fix Truly Consistent Failures (PRIORITY)
Fix only the 2 tests that failed in ALL 4 runs:
1. test_feature_store_config_propagation
2. test_fresh_store_bundle_consistency[3]

### Phase 2: Fix Nearly Consistent (3/4)
After Phase 1 is validated, address the 4 tests failing in 3/4 runs.

### Phase 3: Re-establish Baseline
Run suite 4 more times to see if fixing consistent failures reduces flakiness.

### Phase 4: Root Cause Analysis
If flakiness persists, investigate test isolation issues:
- Mock state pollution
- Global Prometheus registry cleanup
- Database state leaks
- Module import caching

## Validation Strategy

**DO NOT trust individual test runs.**
Always validate fixes with:
```bash
for i in 1 2 3; do
    echo "=== RUN $i ==="
    make pytest-ml 2>&1 | tee /tmp/validation_run_$i.log
done
```

A fix is ONLY successful if the test passes in ALL 3 validation runs.
