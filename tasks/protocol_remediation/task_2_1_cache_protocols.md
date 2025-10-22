# Task 2.1: Cache Component Protocol Definitions

## Context
**Phase**: Protocol Remediation - Task Group 2 (High Priority API Type Safety)
**Task ID**: 2.1
**Depends On**: None (independent of Task Group 1)
**Estimated Effort**: 1 hour 30 minutes
**Priority**: P1 (HIGH)

## Scope
Add protocol definitions for cache components in `ml/core/cache.py` to enable Protocol-First design for hot-path components. Currently, cache classes (LockFreeRingBuffer, PreAllocatedFeatureCache, ReservoirSampler, MultiChannelRingBuffer) are concrete implementations with NO protocol definitions.

**Current State** (lines 22-738 in `ml/core/cache.py`):
```python
# Concrete classes without protocols
class LockFreeRingBuffer:
    """Lock-free ring buffer for high-performance history tracking."""
    # ...implementation

class PreAllocatedFeatureCache:
    """Pre-allocated cache for feature vectors with zero-allocation hot path."""
    # ...implementation

class ReservoirSampler:
    """Reservoir sampling for maintaining representative sample."""
    # ...implementation
```

**Target State**:
```python
from typing import Protocol, runtime_checkable

# Add protocol definitions FIRST (Pattern 2: Protocol-First)
@runtime_checkable
class RingBufferProtocol(Protocol):
    """Protocol for ring buffer implementations."""
    def append(self, value: float) -> None: ...
    def get_last(self, n: int = 1) -> npt.NDArray[np.float64]: ...
    # ... other methods

@runtime_checkable
class FeatureCacheProtocol(Protocol):
    """Protocol for feature cache implementations."""
    def get_current_buffer(self) -> npt.NDArray[np.float32]: ...
    # ... other methods

@runtime_checkable
class SamplerProtocol(Protocol):
    """Protocol for sampling implementations."""
    def add_sample(self, value: float) -> None: ...
    # ... other methods

# Update concrete classes to implement protocols
class LockFreeRingBuffer(RingBufferProtocol):
    """Lock-free ring buffer for high-performance history tracking."""
    # ...implementation (unchanged)
```

## Required Reading
- [x] `reports/audit/stage2-2/INTEGRATION_GENERIC_TYPES_REMEDIATION.md` (Violation 8, lines 861-1006)
- [x] `AGENT_TASK_FRAMEWORK.md` (5-phase workflow)
- [x] `CRITICAL_SAFEGUARDS.md` (TDD and validation requirements)
- [x] `CLAUDE.md` (Pattern 2: Protocol-First Interface Design)
- [x] Task 1.1-1.3 completion reports (apply proven patterns)

## Definition of Done
- [ ] 3 protocol definitions added (RingBufferProtocol, FeatureCacheProtocol, SamplerProtocol)
- [ ] 3 concrete classes updated to inherit from protocols
- [ ] Protocols added to module __all__ exports
- [ ] Tests designed BEFORE implementation (TDD)
- [ ] Tests verify protocol conformance via runtime_checkable
- [ ] All tests PASS with 100% pass rate (not just collect)
- [ ] MyPy strict passes with 0 errors
- [ ] Ruff check passes with 0 violations
- [ ] No circular import errors
- [ ] 100% backward compatible (protocol inheritance has zero runtime cost)

## Files to Modify
- [x] `/home/nate/projects/nautilus_trader/ml/core/cache.py` (lines 1-50: add Protocol imports and definitions)
- [x] `/home/nate/projects/nautilus_trader/ml/core/cache.py` (line ~100: update LockFreeRingBuffer)
- [x] `/home/nate/projects/nautilus_trader/ml/core/cache.py` (line ~300: update PreAllocatedFeatureCache)
- [x] `/home/nate/projects/nautilus_trader/ml/core/cache.py` (line ~500: update ReservoirSampler)
- [x] `/home/nate/projects/nautilus_trader/ml/core/cache.py` (last lines: update __all__ exports)

## Protocol Definitions Required

### 1. RingBufferProtocol
```python
@runtime_checkable
class RingBufferProtocol(Protocol):
    """Protocol for ring buffer implementations."""

    def append(self, value: float) -> None:
        """Add value to buffer (overwrites oldest if full)."""
        ...

    def get_last(self, n: int = 1) -> npt.NDArray[np.float64]:
        """Get last n values as array."""
        ...

    def get_window(self, start: int, length: int) -> npt.NDArray[np.float64]:
        """Get window of values."""
        ...

    def reset(self) -> None:
        """Reset buffer to empty state."""
        ...

    @property
    def count(self) -> int:
        """Return number of elements."""
        ...
```

### 2. FeatureCacheProtocol
```python
@runtime_checkable
class FeatureCacheProtocol(Protocol):
    """Protocol for feature cache implementations."""

    def get_current_buffer(self) -> npt.NDArray[np.float32]:
        """Get current feature buffer."""
        ...

    def store_current_features(self) -> None:
        """Store current features in history."""
        ...

    def prepare_onnx_input(self, use_normalized: bool = True) -> npt.NDArray[np.float32]:
        """Prepare ONNX input buffer."""
        ...

    def reset(self) -> None:
        """Reset cache."""
        ...

    @property
    def n_features(self) -> int:
        """Number of features."""
        ...
```

### 3. SamplerProtocol
```python
@runtime_checkable
class SamplerProtocol(Protocol):
    """Protocol for sampling implementations."""

    def add_sample(self, value: float) -> None:
        """Add sample to reservoir."""
        ...

    def get_percentile(self, q: float) -> float:
        """Get percentile from current sample."""
        ...

    def reset(self) -> None:
        """Reset sampler."""
        ...

    @property
    def count(self) -> int:
        """Current number of samples in reservoir."""
        ...
```

## Testing Requirements

### Unit Tests (NEW - TDD)
Create: `ml/tests/unit/core/test_cache_protocols.py`

**Test Cases**:
1. `test_lock_free_ring_buffer_implements_protocol`
   - Use isinstance() with runtime_checkable to verify protocol conformance
   - Verify LockFreeRingBuffer conforms to RingBufferProtocol

2. `test_ring_buffer_protocol_methods_present`
   - Verify protocol methods: append, get_last, get_window, reset
   - Verify protocol property: count

3. `test_preallocated_feature_cache_implements_protocol`
   - Verify PreAllocatedFeatureCache conforms to FeatureCacheProtocol

4. `test_feature_cache_protocol_methods_present`
   - Verify protocol methods: get_current_buffer, store_current_features, prepare_onnx_input, reset
   - Verify protocol property: n_features

5. `test_reservoir_sampler_implements_protocol`
   - Verify ReservoirSampler conforms to SamplerProtocol

6. `test_sampler_protocol_methods_present`
   - Verify protocol methods: add_sample, get_percentile, reset
   - Verify protocol property: count

7. `test_protocols_exported_in_all`
   - Verify __all__ includes RingBufferProtocol, FeatureCacheProtocol, SamplerProtocol

8. `test_mock_implementations_conform_to_protocols`
   - Create mock classes implementing each protocol
   - Verify mocks pass isinstance() checks

9. `test_protocol_inheritance_has_zero_runtime_cost`
   - Measure memory/performance of class with/without protocol
   - Verify no overhead from protocol inheritance

10. `test_type_hints_work_with_protocols`
    - Use typing.get_type_hints() to verify protocol types
    - Verify type checkers understand protocol inheritance

### Integration Tests
11. `test_protocol_based_dependency_injection`
    - Create function accepting RingBufferProtocol
    - Verify LockFreeRingBuffer works via protocol

## Implementation Steps

### Phase 1: Test Design (25 minutes)
1. Review Violation 8 in remediation document (lines 861-1006)
2. Review Task 1.1-1.3 test patterns (apply flexible assertions)
3. Design tests covering:
   - Protocol conformance (isinstance checks)
   - Protocol method presence
   - __all__ exports
   - Mock implementations
   - Type hints
4. Write test skeletons with `@pytest.mark.skip` decorator
5. Document expected behavior in test docstrings
6. Generate test design report

### Phase 2: Implementation (30 minutes)
1. Add Protocol and runtime_checkable imports to cache.py
2. Add 3 protocol definitions at top of file (after imports, before classes)
3. Update LockFreeRingBuffer to inherit from RingBufferProtocol
4. Update PreAllocatedFeatureCache to inherit from FeatureCacheProtocol
5. Update ReservoirSampler to inherit from SamplerProtocol
6. Update __all__ to export 3 new protocols
7. Remove `@pytest.mark.skip` from tests
8. Run tests locally - verify 100% pass rate
9. Generate implementation report

### Phase 3: Static Validation (15 minutes)
1. Run: `poetry run mypy ml/core/cache.py --strict`
2. Run: `ruff check ml/core/cache.py`
3. Run: `python -c "import ml.core.cache; print('✓')"`
4. Generate static validation report
5. **Decision**: PASS → Phase 4 | FAIL → Return to Phase 2

### Phase 4: Integration Validation (20 minutes)
1. Run: `pytest ml/tests/unit/core/test_cache_protocols.py -v`
2. **⚠️ CRITICAL**: Verify output shows "X passed" NOT "X collected"
3. **⚠️ CRITICAL**: 100% pass rate required (learned from Task 1.1)
4. Run existing tests: `pytest ml/tests/unit/core -v`
5. Verify no regressions
6. Generate integration validation report
7. **Decision**: PASS → APPROVED | FAIL → Return to Phase 2

### Phase 5: System Validation
**SKIP** - Not required for protocol additions (zero runtime changes)

## Rollback Plan
```bash
git checkout ml/core/cache.py
git checkout ml/tests/unit/core/test_cache_protocols.py
```

## Success Metrics
- Protocol definitions added: 3 (RingBufferProtocol, FeatureCacheProtocol, SamplerProtocol)
- Classes updated: 3 (LockFreeRingBuffer, PreAllocatedFeatureCache, ReservoirSampler)
- Protocols exported in __all__: 3
- MyPy strict: 0 errors
- Ruff: 0 violations
- All unit tests: 100% pass rate (tests EXECUTED, not just collected)
- Pattern 2 compliance: +10 points improvement (75% → 85%)
- Zero runtime overhead (protocol inheritance has no cost)

## Risk Assessment
- **Runtime Risk**: NONE (protocol inheritance has zero runtime cost)
- **Type Safety**: HUGE IMPROVEMENT (enables Protocol-First design for hot-path components)
- **Backward Compatibility**: 100% (protocols don't change class behavior)
- **Testability**: IMPROVED (easier to create mock implementations)
- **Hot Path**: NO IMPACT (protocols are compile-time only)

## Lessons from Task Group 1
- ✅ 100% pass rate is mandatory
- ✅ Tests should verify behavior (protocol conformance) not implementation details
- ✅ Handle runtime type variations (use isinstance with runtime_checkable)
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
