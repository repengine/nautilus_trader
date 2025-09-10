# ADR-002: Protocol-First Interface Design

## Status
**ACCEPTED** - 2024-01-15

## Context

The Nautilus Trader ML system requires flexible component interfaces that support:

- **Duck Typing**: Test implementations should work seamlessly with production code
- **Structural Typing**: Components should be validated by capability rather than inheritance
- **Fallback Strategies**: Different implementations (PostgreSQL, File, Dummy) should be interchangeable
- **Testing**: Easy mocking and stubbing without complex inheritance hierarchies
- **Type Safety**: Strong static type checking while maintaining flexibility

Traditional inheritance-based interfaces create tight coupling and make testing difficult. Abstract base classes require all implementations to inherit from a common base, limiting flexibility.

## Decision

**Use `typing.Protocol` for ALL component interfaces, enabling structural typing without implementation coupling.**

### Core Principle
Components must implement required methods with correct signatures, but don't need to inherit from a specific base class. This enables:

- Duck typing support for testing
- Flexible fallback implementations
- Clear contracts without coupling
- Runtime type checking with `isinstance()`

### Implementation Standards

```python
from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class ComponentProtocol(Protocol):
    """Protocol defines interface contract without inheritance requirement."""

    def required_method(self, param: str) -> dict[str, Any]:
        """Method signature defines the contract."""
        ...

    def get_health_status(self) -> dict[str, Any]:
        """Standard health reporting (required for all components)."""
        ...
```

### Universal Protocol Requirements
All ML components MUST implement `MLComponentProtocol`:

```python
@runtime_checkable
class MLComponentProtocol(Protocol):
    def get_health_status(self) -> dict[str, Any]: ...
    def get_performance_metrics(self) -> dict[str, float]: ...
    def validate_configuration(self) -> list[str]: ...
```

## Consequences

### Positive

- **Flexible Implementation**: Components can use any implementation approach
- **Easy Testing**: Test doubles work seamlessly without inheritance
- **Fallback Support**: Dummy/File/PostgreSQL implementations are interchangeable
- **Type Safety**: Static type checking validates contracts at build time
- **Runtime Validation**: `isinstance()` checks work with structural typing
- **Loose Coupling**: No forced inheritance relationships between components

### Negative

- **Documentation Burden**: Protocol methods must be well-documented since no base implementation exists
- **No Shared Implementation**: Common functionality must be provided via mixins or composition
- **Runtime Discovery**: IDE support may be limited compared to class inheritance
- **Protocol Proliferation**: Many protocols needed for fine-grained interfaces

### Risks

- **Interface Drift**: Protocols might diverge from actual implementations over time
- **Runtime Errors**: Protocol violations only caught at runtime (mitigated by tests)
- **Complexity**: Developers must understand protocol system vs traditional inheritance

## Implementation Details

### Protocol Definition Pattern

```python
from typing import Protocol, runtime_checkable, Any, Dict, List

@runtime_checkable
class FeatureStoreProtocol(Protocol):
    """Protocol for feature store implementations."""

    def write_features(
        self,
        instrument_id: str,
        features: Dict[str, float],
        ts_event: int,
        ts_init: int
    ) -> None:
        """Write features to store."""
        ...

    def get_latest_features(
        self,
        instrument_id: str,
        ts_event: int
    ) -> Dict[str, float] | None:
        """Get most recent features for instrument."""
        ...
```

### Implementation Examples

```python
# Production implementation - no inheritance required
class PostgreSQLFeatureStore:
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def write_features(self, instrument_id: str, features: Dict[str, float],
                      ts_event: int, ts_init: int) -> None:
        # PostgreSQL implementation
        pass

    def get_latest_features(self, instrument_id: str, ts_event: int) -> Dict[str, float] | None:
        # PostgreSQL implementation
        pass

# Test implementation - also no inheritance required
class DummyFeatureStore:
    def __init__(self):
        self.features_cache: Dict[str, Dict[str, float]] = {}

    def write_features(self, instrument_id: str, features: Dict[str, float],
                      ts_event: int, ts_init: int) -> None:
        self.features_cache[instrument_id] = features

    def get_latest_features(self, instrument_id: str, ts_event: int) -> Dict[str, float] | None:
        return self.features_cache.get(instrument_id)

# Both implementations automatically conform to protocol
assert isinstance(PostgreSQLFeatureStore(...), FeatureStoreProtocol)
assert isinstance(DummyFeatureStore(), FeatureStoreProtocol)
```

### Consumer Pattern

```python
def process_features(store: FeatureStoreProtocol, instrument_id: str) -> bool:
    """Function uses protocol-based typing - works with ANY conforming implementation."""
    try:
        features = store.get_latest_features(instrument_id, time.time_ns())
        return features is not None
    except Exception:
        return False

# Works with both implementations seamlessly
postgres_store = PostgreSQLFeatureStore("postgresql://...")
dummy_store = DummyFeatureStore()

assert process_features(postgres_store, "EUR/USD")
assert process_features(dummy_store, "EUR/USD")
```

### Mixin Support for Shared Functionality

```python
class MLComponentMixin:
    """Mixin providing default implementations for common protocol methods."""

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "component": self.__class__.__name__,
            "status": "ok",
            "timestamp": time.time()
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        return {}

    def validate_configuration(self) -> List[str]:
        return []

# Implementation can inherit mixin for convenience
class PostgreSQLFeatureStore(MLComponentMixin):
    # Only need to implement domain-specific methods
    # get_health_status() provided by mixin
    pass
```

## Validation and Compliance

### Static Type Checking

```python
# mypy validates protocol conformance
def configure_store(store: FeatureStoreProtocol) -> None:
    # Type checker ensures store has required methods
    features = store.get_latest_features("EUR/USD", 0)

# This would cause mypy error if implementation missing methods
bad_store = SomeIncompleteImplementation()
configure_store(bad_store)  # ❌ mypy error if protocol not satisfied
```

### Runtime Validation

```python
def validate_protocol_compliance(instance: Any, protocol: type) -> List[str]:
    """Validate instance conforms to protocol at runtime."""
    issues = []

    # Check runtime protocol compliance
    if not isinstance(instance, protocol):
        issues.append(f"Instance does not conform to {protocol.__name__}")
        return issues

    # Additional method signature validation
    protocol_methods = [name for name in dir(protocol) if not name.startswith('_')]
    for method_name in protocol_methods:
        if not hasattr(instance, method_name):
            issues.append(f"Missing method: {method_name}")
        elif not callable(getattr(instance, method_name)):
            issues.append(f"Attribute {method_name} is not callable")

    return issues

# Usage in tests
def test_store_protocol_compliance():
    store = PostgreSQLFeatureStore(connection_string)
    issues = validate_protocol_compliance(store, FeatureStoreProtocol)
    assert len(issues) == 0, f"Protocol compliance issues: {issues}"
```

### Automated Compliance Testing

```python
import pytest
from typing import get_type_hints

class ProtocolComplianceTestSuite:
    """Automated tests for protocol compliance."""

    @pytest.mark.parametrize("implementation", [
        PostgreSQLFeatureStore,
        FileFeatureStore,
        DummyFeatureStore,
    ])
    def test_feature_store_protocol_compliance(self, implementation):
        """Test all feature store implementations conform to protocol."""
        # Create instance (may need test-specific parameters)
        if implementation == PostgreSQLFeatureStore:
            instance = implementation("sqlite:///:memory:")  # Test DB
        else:
            instance = implementation()

        # Validate protocol compliance
        assert isinstance(instance, FeatureStoreProtocol)

        # Test required methods exist and are callable
        assert hasattr(instance, 'write_features')
        assert callable(instance.write_features)
        assert hasattr(instance, 'get_latest_features')
        assert callable(instance.get_latest_features)

    def test_method_signatures_match_protocol(self):
        """Test implementation method signatures match protocol."""
        import inspect

        # Get protocol method signature
        protocol_method = FeatureStoreProtocol.write_features
        protocol_sig = inspect.signature(protocol_method)

        # Test each implementation
        for impl_class in [PostgreSQLFeatureStore, DummyFeatureStore]:
            impl_method = getattr(impl_class, 'write_features')
            impl_sig = inspect.signature(impl_method)

            # Compare signatures (allowing for self parameter)
            assert len(impl_sig.parameters) == len(protocol_sig.parameters) + 1  # +1 for self
```

## Migration Strategy

### Phase 1: Define Core Protocols

- Create `MLComponentProtocol` for universal interface
- Define domain-specific protocols (FeatureStoreProtocol, etc.)
- Implement protocol validation utilities

### Phase 2: Update Implementations

- Modify existing classes to conform to protocols (no inheritance changes needed)
- Add runtime protocol checks in integration points
- Update type hints to use protocols instead of concrete classes

### Phase 3: Consumer Updates

- Update function signatures to accept protocol types
- Add protocol validation in component initialization
- Update tests to use protocol-based assertions

### Phase 4: Enforcement

- Add mypy rules to enforce protocol usage
- Integration tests validate all implementations conform
- Documentation updated to emphasize protocol-first design

## Alternatives Considered

### Alternative 1: Abstract Base Classes (ABC)
**Rejected** - Requires inheritance, limits implementation flexibility

### Alternative 2: Implicit Duck Typing
**Rejected** - No static type checking, runtime errors difficult to debug

### Alternative 3: Explicit Interface Classes
**Rejected** - Similar to ABC but without enforcement, leads to drift

### Alternative 4: Dependency Injection Framework
**Rejected** - Adds complexity, doesn't solve typing issues

## Related ADRs

- ADR-001: 4-Store + 4-Registry Mandatory Pattern
- ADR-003: Hot/Cold Path Separation Strategy
- ADR-004: Progressive Fallback Implementation

## References

- [Python Protocol Documentation](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [PEP 544 - Protocols: Structural subtyping](https://peps.python.org/pep-0544/)
- [ML Component Protocols Implementation](../../common/protocols.py)
