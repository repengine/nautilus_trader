# Task 1.1: MLIntegrationManager Store/Registry Type Annotations

## Context
**Phase**: Protocol Remediation - Task Group 1 (Critical)
**Task ID**: 1.1
**Depends On**: None
**Estimated Effort**: 1 hour 15 minutes
**Priority**: P0 (CRITICAL)

## Scope
Replace 8 generic `object` type annotations with concrete store/registry types in the `MLIntegrationManager` class to enable full type safety for all ML component access.

**Current State** (lines 110-118 in `ml/core/integration.py`):
```python
class MLIntegrationManager:
    feature_store: object
    model_store: object
    strategy_store: object
    data_store: object | None
    feature_registry: object
    model_registry: object
    strategy_registry: object
    data_registry: object
```

**Target State**:
```python
if TYPE_CHECKING:
    from ml.stores.feature_store import FeatureStore
    from ml.stores.model_store import ModelStore
    # ... (other imports)

class MLIntegrationManager:
    feature_store: FeatureStore
    model_store: ModelStore
    strategy_store: StrategyStore
    data_store: DataStore | None
    feature_registry: FeatureRegistry
    model_registry: ModelRegistry
    strategy_registry: StrategyRegistry
    data_registry: DataRegistry
```

## Required Reading
- [x] `reports/audit/stage2-2/INTEGRATION_GENERIC_TYPES_REMEDIATION.md` (Violation 1 section)
- [x] `AGENT_TASK_FRAMEWORK.md` (Phase 1: Test Design requirements)
- [x] `CRITICAL_SAFEGUARDS.md` (TDD and validation requirements)
- [x] `CLAUDE.md` (Pattern 2: Protocol-First Interface Design)

## Definition of Done
- [ ] TYPE_CHECKING block added with 8 concrete type imports
- [ ] All 8 `object` annotations replaced with concrete types
- [ ] Tests designed BEFORE implementation (TDD)
- [ ] Tests verify type annotations via `typing.get_type_hints()`
- [ ] All tests PASS (not just collect - MUST show "X passed")
- [ ] MyPy strict passes with 0 errors or shows dramatic improvement
- [ ] Ruff check passes with 0 violations
- [ ] No circular import errors
- [ ] 100% backward compatible (zero runtime changes)

## Files to Modify
- [x] `/home/nate/projects/nautilus_trader/ml/core/integration.py` (lines 1-60: add TYPE_CHECKING imports)
- [x] `/home/nate/projects/nautilus_trader/ml/core/integration.py` (lines 110-118: replace annotations)

## Testing Requirements

### Unit Tests (NEW - TDD)
Create: `ml/tests/unit/core/test_integration_manager_types.py`

**Test Cases**:
1. `test_integration_manager_has_typed_store_attributes`
   - Use `typing.get_type_hints()` to verify each store attribute has correct type
   - Assert `FeatureStore`, `ModelStore`, etc. (not `object`)

2. `test_integration_manager_mypy_understands_store_methods`
   - Mock test: Create MLIntegrationManager with mock stores
   - Verify type checker sees correct methods on `mgr.feature_store`

3. `test_type_checking_imports_dont_affect_runtime`
   - Import MLIntegrationManager
   - Verify no import errors
   - Verify class can be instantiated

### Property Tests
4. `test_all_store_attributes_have_concrete_types`
   - Use hypothesis to generate MLIntegrationManager instances
   - Verify all store attributes have non-`object` type hints

### Contract Tests
5. `test_stores_implement_expected_protocols`
   - Verify FeatureStore implements expected protocol methods
   - Verify type hints match actual runtime store implementations

## Implementation Steps

### Phase 1: Test Design (30 minutes)
1. Review Violation 1 section in remediation document
2. Design comprehensive test cases covering:
   - Type hint extraction via `typing.get_type_hints()`
   - MyPy integration validation
   - Runtime import safety
   - Protocol conformance
3. Write test skeletons with `@pytest.mark.skip` decorator
4. Document expected behavior in test docstrings
5. Generate `reports/tests/protocol_remediation/task_1_1_test_design_report.md`

### Phase 2: Implementation (20 minutes)
1. Add TYPE_CHECKING guard at top of integration.py
2. Import 8 concrete store/registry types
3. Replace 8 `object` annotations with concrete types
4. Update class docstring with type information
5. Remove `@pytest.mark.skip` from tests
6. Generate `reports/implementations/protocol_remediation/task_1_1_implementation_report.md`

### Phase 3: Static Validation (10 minutes)
1. Run: `poetry run mypy ml/core/integration.py --strict`
2. Run: `ruff check ml/core/integration.py`
3. Run: `python -c "import ml.core.integration; print('✓')"`
4. Generate `reports/validations/protocol_remediation/task_1_1_static_validation_report.md`
5. **Decision**: PASS → Phase 4 | FAIL → Return to Phase 2

### Phase 4: Integration Validation (15 minutes)
1. Run: `pytest ml/tests/unit/core/test_integration_manager_types.py -v`
2. **⚠️ CRITICAL**: Verify output shows "X passed" NOT "X collected"
3. Run: `pytest ml/tests/unit/core -v` (all existing tests must still pass)
4. Verify type hints: `python -c "from typing import get_type_hints; from ml.core.integration import MLIntegrationManager; hints = get_type_hints(MLIntegrationManager); assert hints['feature_store'].__name__ == 'FeatureStore'"`
5. Generate `reports/validations/protocol_remediation/task_1_1_integration_validation_report.md`
6. **Decision**: PASS → APPROVED | FAIL → Return to Phase 2

### Phase 5: System Validation
**SKIP** - Not required for type annotation changes

## Rollback Plan
```bash
git checkout ml/core/integration.py
git checkout ml/tests/unit/core/test_integration_manager_types.py
```

## Success Metrics
- Type hint extraction test: PASS
- MyPy strict: 0 errors or dramatic improvement from baseline
- Ruff: 0 violations
- All unit tests: 100% pass rate (tests EXECUTED, not just collected)
- IDE autocomplete: Works for `mgr.feature_store.write_features()`
- Pattern 2 compliance: +10 points improvement

## Risk Assessment
- **Runtime Risk**: NONE (only type annotations, zero behavior changes)
- **Type Safety**: HUGE IMPROVEMENT (enables full IDE + mypy support)
- **Backward Compatibility**: 100% (annotations are compile-time only)
- **Import Cycles**: LOW (TYPE_CHECKING guard prevents runtime imports)

## Validation Checklist
- [ ] Test design report generated
- [ ] Tests written BEFORE implementation
- [ ] Implementation report generated
- [ ] Static validation report shows PASS
- [ ] Integration validation report shows PASS (tests EXECUTED)
- [ ] MyPy output improved from baseline
- [ ] No circular import errors
- [ ] All existing tests still pass

---

**Status**: Ready for Phase 1 (Test Design Agent)
**Next Agent**: test-design-agent
