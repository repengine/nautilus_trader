# Task 1.3: Store Initialization cast() Additions

## Context
**Phase**: Protocol Remediation - Task Group 1 (Critical)
**Task ID**: 1.3
**Depends On**: Task 1.1 (concrete types defined), Task 1.2 (dataclass types defined)
**Estimated Effort**: 1 hour 10 minutes
**Priority**: P0 (CRITICAL)

## Scope
Add explicit `cast()` calls in the `_init_stores()` method to resolve mypy assignment compatibility warnings when fallback stores (FileFeatureStore, DummyStore) are assigned to typed attributes.

**Current State** (lines 364-415 in `ml/core/integration.py`):
```python
# From Task 1.1, we now have typed attributes:
class MLIntegrationManager:
    feature_store: FeatureStore
    model_store: ModelStore
    # ... other typed stores

# But _init_stores() does this:
self.feature_store = FileFeatureStore(...)  # mypy error: incompatible types
self.feature_store = DummyStore(...)        # mypy error: incompatible types
```

**MyPy Errors** (38 assignment compatibility warnings):
```
error: Incompatible types in assignment (expression has type "FileFeatureStore", variable has type "FeatureStore")
error: Incompatible types in assignment (expression has type "DummyStore", variable has type "FeatureStore")
```

**Target State**:
```python
from typing import cast

# In _init_stores():
self.feature_store = cast(FeatureStore, FileFeatureStore(...))
self.feature_store = cast(FeatureStore, DummyStore(...))
```

## Required Reading
- [x] `reports/audit/stage2-2/INTEGRATION_GENERIC_TYPES_REMEDIATION.md` (Violation 3 section, lines 196-308)
- [x] `AGENT_TASK_FRAMEWORK.md` (5-phase workflow)
- [x] `CRITICAL_SAFEGUARDS.md` (Type safety and validation requirements)
- [x] `CLAUDE.md` (Pattern 2: Protocol-First Interface Design - cast() usage for legacy bridging)
- [x] Task 1.1 & 1.2 completion reports (build on established type foundation)

## Definition of Done
- [ ] cast() calls added for all fallback store assignments in _init_stores()
- [ ] All 3 initialization paths handled: PostgreSQL (likely no cast needed), file fallback, JSON/dummy fallback
- [ ] Tests designed BEFORE implementation (TDD)
- [ ] Tests verify mypy understands assignments after cast()
- [ ] All tests PASS with 100% pass rate (not just collect)
- [ ] MyPy strict passes with 0 errors or shows dramatic improvement (resolve 38 warnings)
- [ ] Ruff check passes with 0 violations
- [ ] No circular import errors
- [ ] 100% backward compatible (zero runtime changes - cast() is type-only)

## Files to Modify
- [x] `/home/nate/projects/nautilus_trader/ml/core/integration.py` (lines 1-60: add `cast` import)
- [x] `/home/nate/projects/nautilus_trader/ml/core/integration.py` (lines 364-415: add cast() calls)
  - Note: TYPE_CHECKING imports already present from Task 1.1

## Testing Requirements

### Unit Tests (NEW - TDD)
Create: `ml/tests/unit/core/test_store_initialization_casts.py`

**Test Cases**:
1. `test_postgres_store_initialization_no_cast_needed`
   - When PostgreSQL available, stores are correct type already
   - Verify no cast() needed for PostgreSQL path

2. `test_file_store_fallback_uses_cast`
   - When file backend used, cast() ensures type compatibility
   - Verify FileFeatureStore → FeatureStore via cast()

3. `test_dummy_store_fallback_uses_cast`
   - When no backend available, cast() ensures type compatibility
   - Verify DummyStore → FeatureStore via cast()

4. `test_mypy_accepts_casted_assignments`
   - Use typing.get_type_hints() to verify attributes have correct types
   - Simulate mypy's view of the assignments

5. `test_all_stores_have_cast_in_fallback_paths`
   - Property test: verify all 8 stores have cast() in fallback code
   - Covers FeatureStore, ModelStore, StrategyStore, DataStore + 4 registries

6. `test_runtime_behavior_unchanged_by_cast`
   - cast() is compile-time only
   - Verify store methods work identically with/without cast()

### Integration Tests
7. `test_init_stores_with_file_backend_returns_typed_stores`
   - Call _init_stores() with file backend config
   - Verify all stores have correct runtime types
   - Verify type hints work correctly

## Implementation Steps

### Phase 1: Test Design (20 minutes)
1. Review Violation 3 in remediation document (lines 196-308)
2. Review Task 1.1 test patterns (apply flexible assertions)
3. Design tests focusing on:
   - cast() presence in code
   - mypy assignment compatibility
   - runtime behavior parity
4. Write test skeletons with `@pytest.mark.skip` decorator
5. Document expected behavior in test docstrings
6. Generate test design report

### Phase 2: Implementation (20 minutes)
1. Add `cast` import at top of integration.py
2. Identify all fallback store assignments in _init_stores() (lines 364-415)
3. Add cast() to FileFeatureStore assignments: `cast(FeatureStore, FileFeatureStore(...))`
4. Add cast() to DummyStore assignments: `cast(FeatureStore, DummyStore(...))`
5. Repeat for all 8 stores and 4 registries (12 total)
6. Remove `@pytest.mark.skip` from tests
7. Run tests locally - verify 100% pass rate
8. Generate implementation report

### Phase 3: Static Validation (10 minutes)
1. Run: `poetry run mypy ml/core/integration.py --strict`
2. **Expected**: 38 assignment errors reduced to 0 (or near 0)
3. Run: `ruff check ml/core/integration.py`
4. Run: `python -c "import ml.core.integration; print('✓')"`
5. Generate static validation report
6. **Decision**: PASS → Phase 4 | FAIL → Return to Phase 2

### Phase 4: Integration Validation (15 minutes)
1. Run: `pytest ml/tests/unit/core/test_store_initialization_casts.py -v`
2. **⚠️ CRITICAL**: Verify output shows "X passed" NOT "X collected"
3. **⚠️ CRITICAL**: 100% pass rate required (learned from Task 1.1)
4. Run existing tests: `pytest ml/tests/unit/core -v`
5. Verify no regressions
6. Generate integration validation report
7. **Decision**: PASS → APPROVED | FAIL → Return to Phase 2

### Phase 5: System Validation
**SKIP** - Not required for type annotation changes (cast() is compile-time only)

## Rollback Plan
```bash
git checkout ml/core/integration.py
git checkout ml/tests/unit/core/test_store_initialization_casts.py
```

## Success Metrics
- cast() calls added: 12+ calls (8 stores + 4 registries across multiple fallback paths)
- MyPy strict: 0 errors (reduction of 38 errors from Task 1.1 baseline)
- Ruff: 0 violations
- All unit tests: 100% pass rate (tests EXECUTED, not just collected)
- Pattern 2 compliance: +10 points improvement (65% → 75%)
- Zero runtime behavior changes (cast() is type-only)

## Risk Assessment
- **Runtime Risk**: NONE (cast() is compile-time only, zero behavior changes)
- **Type Safety**: HUGE IMPROVEMENT (resolves all assignment compatibility warnings)
- **Backward Compatibility**: 100% (cast() doesn't affect runtime)
- **Import Cycles**: NONE (cast imported from typing, no new module dependencies)

## Lessons from Task 1.1 & 1.2
- ✅ Tests should verify behavior ("mypy accepts assignment") not exact code structure
- ✅ Handle runtime type variations (DummyStore, FileStore, PostgreSQL store)
- ✅ 100% pass rate is mandatory
- ✅ Test that cast() doesn't change runtime behavior
- ✅ Apply flexible assertion patterns

## Validation Checklist
- [ ] Test design report generated
- [ ] Tests written BEFORE implementation (TDD)
- [ ] Implementation report generated
- [ ] Static validation report shows PASS with mypy errors reduced
- [ ] Integration validation report shows PASS with 100% pass rate
- [ ] MyPy output shows dramatic improvement (38 errors → 0)
- [ ] No circular import errors
- [ ] All existing tests still pass

---

**Status**: Ready for Phase 1 (Test Design Agent)
**Next Agent**: test-design-agent
