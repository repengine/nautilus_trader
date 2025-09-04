# ML Architecture Documentation

This directory contains comprehensive architectural documentation for the Nautilus Trader ML integration system. The documentation addresses cross-domain insights and provides implementation guidance for reliable, high-performance ML operations.

## Core Architecture Documents

### [ML Integration Architecture](ml_integration_architecture.md)
Complete overview of the ML integration framework including:
- Cross-domain integration patterns and data flow diagrams
- Performance contracts and SLA definitions  
- Error propagation and fallback strategies
- Security boundaries and access patterns
- Monitoring and observability systems

### [Universal Patterns Implementation Guide](universal_patterns_guide.md)  
Detailed implementation guidance for the 5 Universal ML Architecture Patterns:
- Pattern 1: Mandatory 4-Store + 4-Registry Integration
- Pattern 2: Protocol-First Interface Design
- Pattern 3: Hot/Cold Path Separation
- Pattern 4: Progressive Fallback Chains
- Pattern 5: Centralized Metrics Bootstrap

### [Cross-Domain Configuration Guide](cross_domain_configuration.md)
Unified configuration management strategy covering:
- Hierarchical configuration architecture
- Environment variable consolidation
- Multi-level validation patterns
- Deployment configuration best practices

### [Integration Testing Strategy](integration_testing_strategy.md)
Comprehensive testing approaches for ML pipelines:
- End-to-end pipeline testing patterns
- Cross-domain integration tests
- Performance and load testing strategies
- Resilience and fault injection testing
- Continuous integration patterns

## Architectural Decision Records (ADRs)

The `decisions/` directory contains detailed ADRs documenting key architectural choices:

### [ADR-001: 4-Store + 4-Registry Mandatory Pattern](decisions/ADR-001-4store-4registry-mandatory-pattern.md)
**Status: ACCEPTED**

Mandates that all ML actors use exactly 4 stores and 4 registries through inheritance from `BaseMLInferenceActor`. Ensures consistent data lifecycle management, automatic component initialization, and progressive fallback to dummy implementations.

**Key Benefits:**
- Consistent data persistence across all components
- Automatic fallback when PostgreSQL unavailable  
- Unified health monitoring and event emission
- Simplified actor development

### [ADR-002: Protocol-First Interface Design](decisions/ADR-002-protocol-first-interface-design.md)  
**Status: ACCEPTED**

Requires use of `typing.Protocol` for all component interfaces, enabling structural typing without implementation coupling. Supports duck typing for testing and clear contracts without forced inheritance.

**Key Benefits:**
- Flexible implementation approaches
- Easy testing with mock objects
- Seamless fallback implementations
- Strong type safety with loose coupling

### [ADR-003: Hot/Cold Path Separation Strategy](decisions/ADR-003-hot-cold-path-separation.md)
**Status: ACCEPTED**

Enforces strict separation between hot path (real-time, <5ms P99) and cold path (offline, batch) operations with different performance contracts and implementation strategies.

**Key Benefits:**
- Predictable trading performance
- Optimal resource utilization
- Clear performance boundaries
- Deployment flexibility

### [ADR-004: Progressive Fallback Implementation](decisions/ADR-004-progressive-fallback-implementation.md)
**Status: ACCEPTED**

Implements progressive fallback chains for all external dependencies, providing graceful degradation rather than hard failures. Includes automatic recovery detection and transparent operation across fallback levels.

**Key Benefits:**
- Continuous operation during outages
- Automated recovery without manual intervention
- Robust deployment across environments
- Comprehensive failure handling

### [ADR-005: Centralized Metrics Bootstrap Pattern](decisions/ADR-005-centralized-metrics-bootstrap.md)
**Status: ACCEPTED**

Prohibits direct `prometheus_client` imports, requiring use of centralized `ml.common.metrics_bootstrap` for all metrics operations. Prevents registry conflicts and ensures consistent naming conventions.

**Key Benefits:**
- No metric registry conflicts
- Consistent naming and labeling
- Memory leak prevention
- Testing support and isolation

## Implementation Relationships

The architecture documents work together to provide complete implementation guidance:

```
ML Integration Architecture (Overview)
├── Universal Patterns Guide (Implementation Details)  
├── Cross-Domain Configuration (System Configuration)
├── Integration Testing Strategy (Validation Approaches)
└── ADRs (Detailed Decisions)
    ├── ADR-001: 4-Store + 4-Registry (Data Management)
    ├── ADR-002: Protocol-First Design (Interface Contracts)
    ├── ADR-003: Hot/Cold Separation (Performance)
    ├── ADR-004: Progressive Fallback (Reliability)
    └── ADR-005: Metrics Bootstrap (Observability)
```

## Key Cross-Domain Insights Addressed

These documents specifically address the integration gaps identified in cross-domain analysis:

### Data Flow Integration
- **Problem**: Inconsistent data flow between domains
- **Solution**: Unified event correlation and lineage tracking across all 4 domains
- **Implementation**: Event correlation framework with automatic cascade generation

### Configuration Management  
- **Problem**: Fragmented configuration across components
- **Solution**: Hierarchical configuration with environment variable consolidation
- **Implementation**: Multi-level validation and environment-specific optimization

### Performance Optimization
- **Problem**: Mixed hot/cold path operations causing unpredictable latency
- **Solution**: Strict path separation with different performance contracts
- **Implementation**: Zero-allocation hot path with pre-allocated arrays

### Fallback Strategies
- **Problem**: Binary failure modes causing system-wide outages
- **Solution**: Progressive fallback chains with automatic recovery
- **Implementation**: 4-level fallback hierarchy with monitoring integration

### Testing Integration
- **Problem**: Inadequate end-to-end and cross-domain testing  
- **Solution**: Comprehensive integration testing strategy
- **Implementation**: Pipeline testing framework with performance validation

## Usage Guidelines

### For New ML Components
1. **Start with**: [Universal Patterns Guide](universal_patterns_guide.md) for implementation requirements
2. **Reference**: Relevant ADRs for detailed architectural decisions
3. **Configure**: Using [Cross-Domain Configuration Guide](cross_domain_configuration.md)
4. **Test**: Following [Integration Testing Strategy](integration_testing_strategy.md)

### For System Integration
1. **Plan**: Using [ML Integration Architecture](ml_integration_architecture.md) overview
2. **Implement**: Following all 5 Universal Patterns from the implementation guide
3. **Validate**: Using comprehensive testing strategy
4. **Monitor**: Through centralized metrics and health checks

### For Operations and Deployment
1. **Configure**: Environment-specific settings using configuration guide
2. **Deploy**: With fallback strategies from ADR-004
3. **Monitor**: Using metrics bootstrap pattern from ADR-005
4. **Troubleshoot**: Using cross-domain event correlation and health checks

## Compliance and Validation

All ML components must comply with the architectural patterns documented here. Use the validation frameworks provided in each guide to ensure compliance:

- **Pattern Compliance**: Automated validation in Universal Patterns Guide
- **Configuration Validation**: Multi-level validation in Configuration Guide  
- **Performance Validation**: SLA testing in Integration Testing Strategy
- **Metrics Compliance**: Bootstrap validation in ADR-005

## Related Documentation

- [Domain Bookkeeping Architecture](domain_bookkeeping.md) - Domain-specific data management
- [Registry Architecture](registry_architecture.md) - Model and feature registry systems
- [Teacher Student Architecture](teacher_student_architecture.md) - Model distillation patterns
- [Unified Observability](unified_observability.md) - System-wide monitoring

This architecture documentation provides the foundation for building reliable, high-performance ML systems that integrate seamlessly with Nautilus Trader's trading platform.