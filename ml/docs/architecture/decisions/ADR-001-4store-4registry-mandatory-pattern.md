# ADR-001: 4-Store + 4-Registry Mandatory Pattern

## Status
**ACCEPTED** - 2024-01-15

## Context

The Nautilus Trader ML system requires consistent data lifecycle management across all components. Previously, different actors and components used varying approaches to data storage and registry management, leading to:

- **Inconsistent data persistence**: Some components persisted to PostgreSQL, others to files, some only in memory
- **Missing data lineage**: No unified tracking of data flow between components
- **Incomplete fallback strategies**: Components failed differently when dependencies were unavailable
- **Fragmented monitoring**: Health checks and metrics collection were inconsistent
- **Complex initialization**: Each component required custom setup logic

## Decision

**All ML actors MUST use exactly 4 stores and 4 registries through mandatory inheritance from `BaseMLInferenceActor`.**

### The 4 Mandatory Stores

1. **FeatureStore**: Persists feature values for training/inference parity
2. **ModelStore**: Persists predictions and model performance metrics
3. **StrategyStore**: Persists strategy state and trading decisions
4. **DataStore**: Unified facade with contract validation and event emission

### The 4 Mandatory Registries

1. **FeatureRegistry**: Feature schema validation and lifecycle management
2. **ModelRegistry**: Model deployment tracking and A/B testing
3. **StrategyRegistry**: Strategy compatibility and requirement validation
4. **DataRegistry**: Dataset manifest management and lineage tracking

### Implementation Requirements

- **Automatic Initialization**: All stores/registries initialized via `MLIntegrationManager`
- **Progressive Fallback**: PostgreSQL → DummyStore with warnings when unavailable
- **Unified Health Monitoring**: All components report health through common protocol
- **Event Emission**: Cross-domain events automatically generated and correlated

## Consequences

### Positive

- **Consistent Data Lifecycle**: All components follow identical data management patterns
- **Complete Observability**: Every data operation is tracked and auditable
- **Reliable Fallback**: System continues operating when PostgreSQL unavailable
- **Simplified Development**: New actors get stores automatically, no custom setup
- **Unified Monitoring**: Single health check covers all data persistence
- **Cross-Domain Coordination**: Automatic event emission enables domain orchestration

### Negative

- **Resource Overhead**: Each actor initializes all 4 stores (even if unused)
- **Memory Usage**: Larger memory footprint per actor due to mandatory components
- **Startup Time**: Additional initialization time for all stores/registries
- **Testing Complexity**: Tests must account for all 4 stores being present

### Risks

- **Single Point of Failure**: If `BaseMLInferenceActor` has bugs, affects all components
- **Performance Impact**: Additional indirection through stores may impact hot path
- **Migration Burden**: Existing actors must be refactored to use new pattern

## Implementation Details

### Base Actor Requirements

```python
from ml.actors.base import BaseMLInferenceActor

class YourCustomActor(BaseMLInferenceActor):
    def __init__(self, config: YourCustomActorConfig):
        super().__init__(config)  # REQUIRED: Initializes all 4 stores + 4 registries

        # Stores automatically available:
        # - self.feature_store
        # - self.model_store
        # - self.strategy_store
        # - self.data_store

        # Registries automatically available:
        # - self.feature_registry
        # - self.model_registry
        # - self.strategy_registry
        # - self.data_registry
```

### Fallback Behavior

- **Primary Mode**: Full PostgreSQL-backed stores with complete persistence
- **Fallback Mode**: DummyStore implementations with warnings logged
- **Graceful Degradation**: System continues operating, monitoring shows degraded state

### Health Monitoring Integration

```python
def check_actor_health(actor: BaseMLInferenceActor) -> dict:
    return {
        'stores': {
            'feature_store': actor.feature_store.get_health_status(),
            'model_store': actor.model_store.get_health_status(),
            'strategy_store': actor.strategy_store.get_health_status(),
            'data_store': actor.data_store.get_health_status(),
        },
        'registries': {
            'feature_registry': actor.feature_registry.get_health_status(),
            'model_registry': actor.model_registry.get_health_status(),
            'strategy_registry': actor.strategy_registry.get_health_status(),
            'data_registry': actor.data_registry.get_health_status(),
        }
    }
```

## Compliance Validation

### Automated Checks

- Static analysis to ensure `BaseMLInferenceActor` inheritance
- Runtime validation that all 4 stores are initialized and non-None
- Health check integration tests verify all stores respond correctly
- Performance tests ensure fallback behavior works under load

### Migration Strategy

1. **Phase 1**: Implement `BaseMLInferenceActor` with automatic initialization
2. **Phase 2**: Migrate existing actors one by one to inherit from base class
3. **Phase 3**: Add validation rules to prevent non-compliant actors
4. **Phase 4**: Remove old initialization patterns and enforce compliance

## Alternatives Considered

### Alternative 1: Optional Store Selection
**Rejected** - Would lead back to inconsistency and missing data lineage

### Alternative 2: Dependency Injection Pattern
**Rejected** - More complex setup, doesn't guarantee all stores are available

### Alternative 3: Factory Pattern for Store Creation
**Rejected** - Still allows inconsistency in which stores are used

### Alternative 4: Single Universal Store
**Rejected** - Would violate domain separation and complicate schema evolution

## Related ADRs

- ADR-002: Protocol-First Interface Design
- ADR-003: Hot/Cold Path Separation Strategy
- ADR-004: Progressive Fallback Implementation
- ADR-005: Centralized Metrics Bootstrap Pattern

## References

- [Domain Bookkeeping Architecture](../domain_bookkeeping.md)
- [ML Integration Architecture](../ml_integration_architecture.md)
- [BaseMLInferenceActor Implementation](../../actors/base.py)
