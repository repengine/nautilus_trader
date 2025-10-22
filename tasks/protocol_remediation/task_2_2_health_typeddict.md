# Task 2.2: Health Aggregation TypedDict Definitions

## Context
**Phase**: Protocol Remediation - Task Group 2 (High Priority API Type Safety)
**Task ID**: 2.2
**Depends On**: None (independent task)
**Estimated Effort**: 1 hour 30 minutes
**Priority**: P1 (HIGH)

## Scope
Replace generic `dict[str, object]` return type in `aggregate_health()` method with structured TypedDict definitions to enable type-safe health monitoring and IDE autocomplete for health summary keys.

**Current State** (lines 983-1060 in `ml/core/integration.py`):
```python
def aggregate_health(self) -> dict[str, object]:
    """
    Aggregate component health into domain and system summaries.

    Returns
    -------
    dict[str, object]
        A structured health summary with keys:
        - components: per-component health and metrics (when available)
        - domains: aggregated health per domain (data, features, model, strategy)
        - system: overall status with list of unhealthy components
    """

    def _comp_health(comp: object) -> dict[str, object]:
        healthy = True
        health: dict[str, object] | None = None
        # ...
        return {"healthy": healthy, "health": health or {}, "metrics": metrics or {}}

    # ...
    return {"components": components, "domains": domains, "system": system}
```

**Target State**:
```python
from typing import TypedDict

# Add TypedDict definitions at module level
class ComponentHealthStatus(TypedDict, total=False):
    """Health status for a single component."""
    healthy: bool
    health: dict[str, object]
    metrics: dict[str, float]

class DomainHealth(TypedDict):
    """Health status for a domain (e.g., features, model, strategy)."""
    components: list[str]
    healthy: bool

class HealthDomains(TypedDict, total=False):
    """All domain health statuses."""
    data: DomainHealth
    features: DomainHealth
    model: DomainHealth
    strategy: DomainHealth

class SystemHealth(TypedDict):
    """Overall system health status."""
    healthy: bool
    unhealthy: list[str]

class HealthSummary(TypedDict):
    """Complete health summary for the ML integration system."""
    components: dict[str, ComponentHealthStatus]
    domains: HealthDomains
    system: SystemHealth


def aggregate_health(self) -> HealthSummary:
    """
    Aggregate component health into domain and system summaries.

    Returns
    -------
    HealthSummary
        Typed dictionary with structured health information
    """

    def _comp_health(comp: object) -> ComponentHealthStatus:
        # ... implementation
        return ComponentHealthStatus(
            healthy=healthy,
            health=health or {},
            metrics=metrics or {},
        )

    # ...
    return HealthSummary(
        components=components,
        domains=domains,
        system=system,
    )
```

## Required Reading
- [x] `reports/audit/stage2-2/INTEGRATION_GENERIC_TYPES_REMEDIATION.md` (Violation 6, lines 587-753)
- [x] `AGENT_TASK_FRAMEWORK.md` (5-phase workflow)
- [x] `CRITICAL_SAFEGUARDS.md` (TDD and validation requirements)
- [x] `CLAUDE.md` (Type Safety & Annotations section)
- [x] Task 1.1-1.3 & 2.1 completion reports (apply proven patterns)

## Definition of Done
- [ ] 5 TypedDict definitions added (ComponentHealthStatus, DomainHealth, HealthDomains, SystemHealth, HealthSummary)
- [ ] aggregate_health() return type changed from `dict[str, object]` to `HealthSummary`
- [ ] _comp_health() return type changed to `ComponentHealthStatus`
- [ ] Tests designed BEFORE implementation (TDD)
- [ ] Tests verify TypedDict structure and key access
- [ ] All tests PASS with 100% pass rate (not just collect)
- [ ] MyPy strict passes with 0 errors
- [ ] Ruff check passes with 0 violations
- [ ] No circular import errors
- [ ] 100% backward compatible (runtime dict structure unchanged)

## Files to Modify
- [x] `/home/nate/projects/nautilus_trader/ml/core/integration.py` (lines 1-100: add TypedDict imports and definitions)
- [x] `/home/nate/projects/nautilus_trader/ml/core/integration.py` (line 983: update aggregate_health return type)
- [x] `/home/nate/projects/nautilus_trader/ml/core/integration.py` (line 997: update _comp_health return type)

## TypedDict Definitions Required

### 1. ComponentHealthStatus
```python
from typing import TypedDict

class ComponentHealthStatus(TypedDict, total=False):
    """Health status for a single component."""
    healthy: bool
    health: dict[str, object]
    metrics: dict[str, float]
```

### 2. DomainHealth
```python
class DomainHealth(TypedDict):
    """Health status for a domain (e.g., features, model, strategy)."""
    components: list[str]
    healthy: bool
```

### 3. HealthDomains
```python
class HealthDomains(TypedDict, total=False):
    """All domain health statuses."""
    data: DomainHealth
    features: DomainHealth
    model: DomainHealth
    strategy: DomainHealth
```

### 4. SystemHealth
```python
class SystemHealth(TypedDict):
    """Overall system health status."""
    healthy: bool
    unhealthy: list[str]
```

### 5. HealthSummary
```python
class HealthSummary(TypedDict):
    """Complete health summary for the ML integration system."""
    components: dict[str, ComponentHealthStatus]
    domains: HealthDomains
    system: SystemHealth
```

## Testing Requirements

### Unit Tests (NEW - TDD)
Create: `ml/tests/unit/core/test_health_typeddict.py`

**Test Cases**:
1. `test_aggregate_health_returns_health_summary`
   - Call aggregate_health()
   - Verify return type has correct structure

2. `test_health_summary_has_required_keys`
   - Verify "components", "domains", "system" keys present

3. `test_component_health_status_structure`
   - Verify ComponentHealthStatus has healthy, health, metrics

4. `test_domain_health_structure`
   - Verify DomainHealth has components, healthy

5. `test_system_health_structure`
   - Verify SystemHealth has healthy, unhealthy

6. `test_health_domains_contains_all_domains`
   - Verify data, features, model, strategy domains

7. `test_typeddict_enables_ide_autocomplete`
   - Use typing.get_type_hints() to verify types
   - Verify TypedDict structure accessible

8. `test_backward_compatibility_dict_access`
   - Verify dict["key"] access still works
   - Verify .get() method works

9. `test_aggregate_health_with_healthy_components`
   - Mock healthy components
   - Verify health summary correct

10. `test_aggregate_health_with_unhealthy_components`
    - Mock unhealthy components
    - Verify system.unhealthy list populated

### Integration Tests
11. `test_health_summary_json_serializable`
    - Verify health summary can be JSON serialized
    - Important for APIs and monitoring

## Implementation Steps

### Phase 1: Test Design (25 minutes)
1. Review Violation 6 in remediation document (lines 587-753)
2. Review Task Group 1 test patterns (apply flexible assertions)
3. Design tests covering:
   - TypedDict structure
   - Key access
   - Backward compatibility
   - JSON serialization
4. Write test skeletons with `@pytest.mark.skip` decorator
5. Document expected behavior in test docstrings
6. Generate test design report

### Phase 2: Implementation (30 minutes)
1. Add TypedDict import at top of integration.py
2. Add 5 TypedDict definitions after imports (lines 100-150)
3. Update aggregate_health() return type to HealthSummary
4. Update _comp_health() return type to ComponentHealthStatus
5. Update method implementation to construct TypedDict instances
6. Remove `@pytest.mark.skip` from tests
7. Run tests locally - verify 100% pass rate
8. Generate implementation report

### Phase 3: Static Validation (15 minutes)
1. Run: `poetry run mypy ml/core/integration.py --strict`
2. Run: `ruff check ml/core/integration.py`
3. Run: `python -c "import ml.core.integration; print('✓')"`
4. Generate static validation report
5. **Decision**: PASS → Phase 4 | FAIL → Return to Phase 2

### Phase 4: Integration Validation (20 minutes)
1. Run: `pytest ml/tests/unit/core/test_health_typeddict.py -v`
2. **⚠️ CRITICAL**: Verify output shows "X passed" NOT "X collected"
3. **⚠️ CRITICAL**: 100% pass rate required
4. Run existing tests: `pytest ml/tests/unit/core/test_integration_health.py -v`
5. Verify no regressions
6. Generate integration validation report
7. **Decision**: PASS → APPROVED | FAIL → Return to Phase 2

### Phase 5: System Validation
**SKIP** - Not required for TypedDict additions (zero runtime behavior changes)

## Rollback Plan
```bash
git checkout ml/core/integration.py
git checkout ml/tests/unit/core/test_health_typeddict.py
```

## Success Metrics
- TypedDict definitions added: 5
- Methods updated: 2 (aggregate_health, _comp_health)
- MyPy strict: 0 errors
- Ruff: 0 violations
- All unit tests: 100% pass rate (tests EXECUTED, not just collected)
- Pattern 2 compliance: +10 points improvement (85% → 95%) ✅ **TARGET ACHIEVED**
- IDE autocomplete: Works for health["system"]["healthy"]
- Zero runtime behavior changes (TypedDict is structural typing)

## Risk Assessment
- **Runtime Risk**: NONE (TypedDict doesn't change runtime dict behavior)
- **Type Safety**: HUGE IMPROVEMENT (enables type checking and IDE support)
- **Backward Compatibility**: 100% (TypedDict is structural, dict access unchanged)
- **API Compatibility**: 100% (return value structure identical at runtime)

## Lessons from Task Group 1 & 2.1
- ✅ 100% pass rate is mandatory
- ✅ Tests should verify behavior (TypedDict structure) not implementation details
- ✅ Handle runtime dict compatibility (TypedDict is structural)
- ✅ Apply flexible assertion patterns
- ✅ Test execution, not just collection

## Validation Checklist
- [ ] Test design report generated
- [ ] Tests written BEFORE implementation (TDD)
- [ ] Implementation report generated
- [ ] Static validation report shows PASS
- [ ] Integration validation report shows PASS with 100% pass rate
- [ ] MyPy output shows 0 errors
- [ ] No circular import errors
- [ ] All existing tests still pass

---

**Status**: Ready for Phase 1 (Test Design Agent)
**Next Agent**: test-design-agent
