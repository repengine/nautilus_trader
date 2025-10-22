# Task 1.2: ActorStoresRegistries Dataclass Type Annotations

## Context
**Phase**: Protocol Remediation - Task Group 1 (Critical)
**Task ID**: 1.2
**Depends On**: Task 1.1 (reuses TYPE_CHECKING imports)
**Estimated Effort**: 50 minutes
**Priority**: P0 (CRITICAL)

## Scope
Replace 8 generic `object` field annotations with concrete store/registry types in the `ActorStoresRegistries` dataclass to enable full type safety for dependency injection patterns.

**Current State** (lines 1635-1642 in `ml/core/integration.py`):
```python
@dataclass(slots=True)
class ActorStoresRegistries:
    """Simple container for actor-attached stores and registries."""

    feature_store: object              # ❌ Should be FeatureStore
    model_store: object                # ❌ Should be ModelStore
    strategy_store: object             # ❌ Should be StrategyStore
    data_store: object                 # ❌ Should be DataStore
    feature_registry: object           # ❌ Should be FeatureRegistry
    model_registry: object             # ❌ Should be ModelRegistry
    strategy_registry: object          # ❌ Should be StrategyRegistry
    data_registry: object              # ❌ Should be DataRegistry
    persistence_config: object | None
    connection_string: str | None
```

**Target State**:
```python
@dataclass(slots=True)
class ActorStoresRegistries:
    """Simple container for actor-attached stores and registries."""

    feature_store: FeatureStore        # ✅ Concrete type
    model_store: ModelStore            # ✅ Concrete type
    strategy_store: StrategyStore      # ✅ Concrete type
    data_store: DataStore              # ✅ Concrete type
    feature_registry: FeatureRegistry  # ✅ Concrete type
    model_registry: ModelRegistry      # ✅ Concrete type
    strategy_registry: StrategyRegistry # ✅ Concrete type
    data_registry: DataRegistry        # ✅ Concrete type
    persistence_config: PersistenceConfig | None  # ✅ Concrete type
    connection_string: str | None      # ✅ Already correct
```

## Required Reading
- [x] `reports/audit/stage2-2/INTEGRATION_GENERIC_TYPES_REMEDIATION.md` (Violation 2 section, lines 108-195)
- [x] `AGENT_TASK_FRAMEWORK.md` (5-phase workflow)
- [x] `CRITICAL_SAFEGUARDS.md` (TDD and validation requirements)
- [x] `CLAUDE.md` (Pattern 2: Protocol-First Interface Design)
- [x] Task 1.1 completion reports (same pattern applies)

## Definition of Done
- [ ] All 8 `object` field annotations replaced with concrete types
- [ ] `persistence_config` type changed from `object | None` to `PersistenceConfig | None`
- [ ] Tests designed BEFORE implementation (TDD)
- [ ] Tests verify dataclass field types via `dataclasses.fields()`
- [ ] All tests PASS with 100% pass rate (not just collect)
- [ ] MyPy strict passes with 0 errors or shows dramatic improvement
- [ ] Ruff check passes with 0 violations
- [ ] No circular import errors
- [ ] 100% backward compatible (zero runtime changes)

## Files to Modify
- [x] `/home/nate/projects/nautilus_trader/ml/core/integration.py` (lines 1635-1643: replace field annotations)
  - Note: TYPE_CHECKING imports already added in Task 1.1
  - May need to add `PersistenceConfig` import

## Testing Requirements

### Unit Tests (NEW - TDD)
Create: `ml/tests/unit/core/test_actor_stores_registries_types.py`

**Test Cases**:
1. `test_feature_store_field_has_concrete_type`
   - Use `dataclasses.fields()` to extract field metadata
   - Verify field type is FeatureStore (not object)

2. `test_all_store_registry_fields_have_concrete_types`
   - Iterate over all 8 store/registry fields
   - Verify none have object type
   - Property test style

3. `test_persistence_config_has_concrete_type`
   - Verify persistence_config field is PersistenceConfig | None
   - Not object | None

4. `test_dataclass_can_be_instantiated_with_typed_fields`
   - Create mock stores/registries
   - Instantiate ActorStoresRegistries
   - Verify type checking works

5. `test_dataclass_fields_accessible_via_type_hints`
   - Use `typing.get_type_hints()` on dataclass
   - Verify all fields have proper types

### Integration Tests
6. `test_init_ml_stores_and_registries_returns_typed_dataclass`
   - Call `init_ml_stores_and_registries()`
   - Verify return type is ActorStoresRegistries with concrete field types

## Implementation Steps

### Phase 1: Test Design (20 minutes)
1. Review Violation 2 in remediation document (lines 108-195)
2. Review Task 1.1 test patterns (similar approach)
3. Design tests using `dataclasses.fields()` for field inspection
4. Write test skeletons with `@pytest.mark.skip` decorator
5. Document expected behavior in test docstrings
6. Generate test design report

### Phase 2: Implementation (15 minutes)
1. Verify TYPE_CHECKING imports exist (from Task 1.1)
2. Add PersistenceConfig import if needed
3. Replace 8 `object` field annotations with concrete types
4. Replace `persistence_config: object | None` with `PersistenceConfig | None`
5. Update dataclass docstring with type examples
6. Remove `@pytest.mark.skip` from tests
7. Generate implementation report

### Phase 3: Static Validation (5 minutes)
1. Run: `poetry run mypy ml/core/integration.py --strict`
2. Run: `ruff check ml/core/integration.py`
3. Run: `python -c "import ml.core.integration; print('✓')"`
4. Generate static validation report
5. **Decision**: PASS → Phase 4 | FAIL → Return to Phase 2

### Phase 4: Integration Validation (10 minutes)
1. Run: `pytest ml/tests/unit/core/test_actor_stores_registries_types.py -v`
2. **⚠️ CRITICAL**: Verify output shows "X passed" NOT "X collected"
3. **⚠️ CRITICAL**: 100% pass rate required (learned from Task 1.1)
4. Run existing tests: `pytest ml/tests/unit/core -v`
5. Verify dataclass fields via `dataclasses.fields()`
6. Generate integration validation report
7. **Decision**: PASS → APPROVED | FAIL → Return to Phase 2

### Phase 5: System Validation
**SKIP** - Not required for type annotation changes

## Rollback Plan
```bash
git checkout ml/core/integration.py
git checkout ml/tests/unit/core/test_actor_stores_registries_types.py
```

## Success Metrics
- Dataclass field extraction test: PASS
- MyPy strict: 0 errors or improvement from baseline
- Ruff: 0 violations
- All unit tests: 100% pass rate (tests EXECUTED, not just collected)
- Dependency injection pattern: Typed and safe
- Pattern 2 compliance: +10 points improvement

## Risk Assessment
- **Runtime Risk**: NONE (only type annotations, zero behavior changes)
- **Type Safety**: HUGE IMPROVEMENT (enables typed dependency injection)
- **Backward Compatibility**: 100% (annotations are compile-time only)
- **Import Cycles**: NONE (TYPE_CHECKING already in place from Task 1.1)

## Lessons from Task 1.1
- ✅ Tests should verify "not object" rather than exact type names
- ✅ Handle runtime aliasing (e.g., FeatureStoreLegacy)
- ✅ Handle Python version differences (Union vs UnionType)
- ✅ 100% pass rate is mandatory
- ✅ Test behavior, not implementation details

## Validation Checklist
- [ ] Test design report generated
- [ ] Tests written BEFORE implementation (TDD)
- [ ] Implementation report generated
- [ ] Static validation report shows PASS
- [ ] Integration validation report shows PASS with 100% pass rate
- [ ] MyPy output improved from baseline
- [ ] No circular import errors
- [ ] All existing tests still pass

---

**Status**: Ready for Phase 1 (Test Design Agent)
**Next Agent**: test-design-agent
