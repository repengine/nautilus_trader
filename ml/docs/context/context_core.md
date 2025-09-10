# Context: Core Module

## Overview

The ml/core/ module provides essential infrastructure for high-performance ML operations in Nautilus Trader. It implements zero-allocation data structures, centralized database connection management, and automatic system integration. The core module follows strict hot/cold path separation with sub-5ms latency targets for hot path operations.

**Key Design Principles:**

- **Zero-Allocation Hot Path**: Pre-allocated buffers and memory views to eliminate GC pressure
- **Thread-Safe Singleton Patterns**: Centralized resource management to prevent pool exhaustion
- **Automatic Integration**: Complete wiring of all ML components (stores + registries)
- **Progressive Fallback**: Graceful degradation when optional dependencies are unavailable
- **Production-Ready**: Comprehensive health monitoring and observability integration

## Architecture

The core module consists of three main components:

```
ml/core/
├── cache.py              # High-performance data structures for hot path
├── db_engine.py          # Singleton database engine management
├── integration.py        # Automatic ML system integration
└── __init__.py          # Module exports and public API
```

### Hot Path Optimization Strategy

The module enforces strict performance budgets:

- **Hot path operations**: <5ms P99 latency, zero allocations
- **Feature computation**: <500μs per inference cycle
- **Buffer operations**: <10μs append/retrieve operations
- **Memory stability**: Zero growth over 24h continuous operation

## Key Components

### LockFreeRingBuffer (`cache.py`)

High-performance ring buffer for maintaining rolling windows of prediction history and feature data.

**Features:**

- Zero-allocation O(1) append operations
- Lock-free design for concurrent access
- Memory view access for zero-copy operations
- Automatic wrap-around handling
- Built-in statistical functions (mean, std, percentile)

**Usage Patterns:**

```python
# Maintain prediction history for model drift detection
buffer = LockFreeRingBuffer(size=1000, dtype=np.float32)
buffer.append(prediction_value)  # Zero allocation
recent_predictions = buffer.get_last(100)  # Returns view when possible
```

### PreAllocatedFeatureCache (`cache.py`)

Pre-allocated cache system for feature vectors with zero-copy ONNX integration.

**Features:**

- Pre-allocated buffers for current, normalized, and historical features
- Memory views for zero-copy access
- ONNX-ready input buffer preparation
- Ring buffer history management
- In-place feature normalization support

**Critical for:**

- Sub-millisecond inference in production actors
- Memory-stable long-running processes
- ONNX Runtime integration with minimal overhead

### ReservoirSampler (`cache.py`)

Reservoir sampling implementation for maintaining representative samples from streaming data.

**Features:**

- Algorithm R implementation for uniform random sampling
- Efficient percentile calculation from fixed-size sample
- Memory-efficient alternative to storing complete history
- Multiple percentile calculation in single pass

**Use Cases:**

- Feature distribution monitoring for drift detection
- Model performance tracking with bounded memory usage
- Historical percentile calculation for normalization

### EngineManager (`db_engine.py`)

Thread-safe singleton manager for SQLAlchemy database engines with intelligent connection pooling.

**Problem Solved:**

- Prevents "too many clients already" PostgreSQL errors
- Eliminates connection pool exhaustion during hypothesis testing
- Centralizes database connection lifecycle management
- Provides environment-aware pool sizing

**Features:**

- One engine instance per connection string (singleton pattern)
- Automatic test environment detection with conservative pooling
- Thread-safe engine creation and disposal
- Connection pool health monitoring
- Graceful cleanup for test teardown

**Pool Configuration:**

- **Test environments**: pool_size=2, max_overflow=3 (conservative)
- **Production environments**: pool_size=10, max_overflow=20 (scalable)
- **Health checks**: pool_pre_ping=True, pool_recycle=3600s

### MLIntegrationManager (`integration.py`)

Comprehensive system integration manager that automatically wires all ML components together.

**Mandatory 4-Store + 4-Registry Integration:**

- **Stores**: FeatureStore, ModelStore, StrategyStore, DataStore
- **Registries**: FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry
- **Automatic initialization** with progressive fallback to dummy implementations
- **Health monitoring** across all components
- **Protocol compliance validation** with configurable strictness

**Key Responsibilities:**

1. **Database Management**: Auto-start PostgreSQL, run migrations, health checks
2. **Component Wiring**: Initialize all stores and registries with proper configuration
3. **Partition Management**: Set up time-based partitioning for high-volume tables
4. **Observability Pipeline**: Full observability service with background persistence and event correlation
5. **Actor Creation**: Factory method for creating pre-wired ML actors
6. **Protocol Compliance**: Runtime validation of MLComponentProtocol implementation across all components
7. **Message Bus Integration**: Event emission and external system integration via configurable publishers

**Production Features:**

- Environment variable configuration (ML_AUTO_START_DB, ML_AUTO_MIGRATE)
- Comprehensive health aggregation across domains (data, features, model, strategy)
- Event emission and cascade support for cross-domain coordination
- Background observability flush with configurable intervals and multiple sink support
- Protocol compliance validation with runtime checks and automatic fallback strategies
- Message bus publisher configuration with topic normalization and routing policies

## Dependencies

### Internal Dependencies

- `ml.stores.*`: All ML store implementations for data persistence
- `ml.registry.*`: All ML registry implementations for metadata management
- `ml.common.protocols`: MLComponentProtocol for standardized interfaces
- `ml.common.cascade`: Event cascade utilities for cross-domain coordination
- `ml.observability.*`: Optional monitoring service integration

### External Dependencies

- **SQLAlchemy**: Database engine management and connection pooling
- **NumPy**: High-performance array operations and memory views
- **PostgreSQL**: Primary persistence backend (with fallback support)
- **Docker**: Optional container management for database auto-start

### Performance Dependencies

- **ONNX Runtime**: Zero-allocation inference (via prepared buffers)
- **Memory Views**: Zero-copy data access patterns
- **Pre-allocated Arrays**: Elimination of GC pressure in hot paths

## Usage Patterns

### Basic System Initialization

```python
from ml.core.integration import MLIntegrationManager

# Initialize complete ML system
integration = MLIntegrationManager(
    auto_start_postgres=True,
    auto_migrate=True,
    ensure_healthy=True
)

# All stores and registries are now available
feature_store = integration.feature_store
model_registry = integration.model_registry
```

### High-Performance Feature Caching

```python
from ml.core.cache import PreAllocatedFeatureCache

# Initialize pre-allocated cache
cache = PreAllocatedFeatureCache(n_features=50, history_size=1000)

# Zero-allocation feature computation
features = cache.get_current_buffer()  # Returns pre-allocated array
# ... compute features in-place ...
cache.store_current_features()  # Zero-copy storage

# ONNX inference preparation
onnx_input = cache.prepare_onnx_input(use_normalized=True)
prediction = onnx_session.run(None, {'input': onnx_input})
```

### Database Engine Management

```python
from ml.core.db_engine import EngineManager

# Get singleton engine (auto-creates if needed)
engine = EngineManager.get_engine(connection_string)

# All subsequent calls return the same instance
engine2 = EngineManager.get_engine(connection_string)
assert engine is engine2

# Cleanup during teardown
EngineManager.dispose_all()
```

### Actor Integration with Observability

```python
# Create actor with automatic store integration and observability
actor = integration.create_integrated_actor(
    actor_class=MyMLActor,
    config=actor_config
)

# Actor automatically has access to all stores, registries, and observability
# via BaseMLInferenceActor inheritance

# Observability pipeline can be initialized separately
integration.initialize_observability_pipeline()

# Start background persistence with database sink
integration.start_observability_flush(
    base_path=Path("/data/observability"),
    interval_seconds=60.0,
    file_format="jsonl",
    sink="db",  # Direct PostgreSQL persistence
    db_connection_string="postgresql://user:pass@localhost/nautilus"
)
```

### Event Correlation and Message Bus

```python
# Emit correlated events across domains
correlation_id = integration.emit_cascade(
    event_type="prediction_generated",
    instrument_id="EURUSD.SIM",
    metadata={"model_version": "1.2.3", "confidence": 0.87}
)

# Events automatically routed to configured message bus
# Topic: ml.features.computed.EURUSD.SIM
# Topic: ml.model.prediction_generated.EURUSD.SIM
# Topic: ml.strategy.signal_emitted.EURUSD.SIM
```

## Integration Points

### Nautilus Trader Core

- **Timestamp Compliance**: All operations use nanosecond timestamps (ts_event, ts_init)
- **Data Model Integration**: Direct compatibility with Nautilus data types
- **Actor Framework**: Seamless integration with Nautilus actor lifecycle

### ML System Components

- **Feature Pipeline**: Provides caching infrastructure for feature computation
- **Model Inference**: ONNX-ready buffer preparation and memory management
- **Strategy Execution**: Zero-allocation prediction history and signal generation
- **Monitoring**: Automatic Prometheus metrics registration and health checks

### Storage Layer

- **Automatic Partitioning**: Time-based partitioning for high-volume ML data
- **Connection Pooling**: Shared engine instances across all ML stores
- **Health Monitoring**: Real-time connection pool and storage health tracking
- **Migration Management**: Automatic database schema evolution

### Observability Pipeline Integration

The core module provides comprehensive observability infrastructure through deep integration with the observability module:

**ObservabilityService Integration:**

- **Automatic Initialization**: Observability service created when `initialize_observability_pipeline()` called
- **Background Persistence**: Configurable flush intervals (default 60s) with structured data export
- **Database Sink**: Direct persistence to PostgreSQL tables for high-performance analytics
- **File Sink**: JSONL/CSV export for external data pipeline integration

**Event Correlation System:**

- **UUID4 Generation**: Deterministic correlation IDs for end-to-end request tracing
- **Cross-Domain Lineage**: Parent-child event relationships spanning data/features/models/strategies
- **Event Cascade Support**: Automatic correlation preservation via `emit_cascade()` method
- **Distributed Tracing**: Integration points prepared for OpenTelemetry/Jaeger systems

**Health Monitoring Integration:**

- **Component Health Aggregation**: Real-time health scores from all MLComponentProtocol-compliant components
- **Domain-Level Reporting**: Separate health tracking for data, features, model, and strategy domains
- **Alert Threshold Management**: Configurable thresholds with subsystem breakdown
- **Performance Metrics**: Latency watermarks, throughput metrics, and quality scores

**Message Bus Protocol:**

- **Publisher Interface**: MessagePublisherProtocol for event emission to external systems
- **Topic Standardization**: Canonical `ml.{domain}.{operation}.{instrument_id}` topic structure
- **Publishing Modes**: Both batch and per-row publishing with configurable routing
- **Noop Fallback**: Safe default implementation when message bus unavailable

## Implementation Notes

### Memory Management

- **Pre-allocation Strategy**: All buffers allocated at initialization to prevent GC pressure
- **Memory View Usage**: Zero-copy access patterns wherever possible
- **Ring Buffer Design**: Automatic wrap-around without memory reallocations
- **ONNX Integration**: Direct buffer reuse for inference without copying

### Thread Safety

- **Lock-Free Algorithms**: Ring buffer operations without synchronization overhead
- **Singleton Patterns**: Thread-safe engine creation with double-checked locking
- **Atomic Operations**: Safe concurrent access to shared resources
- **Connection Pooling**: SQLAlchemy's thread-safe connection management

### Error Handling

- **Progressive Fallback**: Graceful degradation when optional components unavailable
- **Health Monitoring**: Continuous validation of all system components
- **Connection Resilience**: Auto-reconnection and pool pre-ping validation
- **Protocol Validation**: Runtime checking of component interface compliance

### Performance Characteristics

- **Hot Path Latency**: <5ms P99 for complete inference cycle
- **Memory Footprint**: Stable over 24h continuous operation
- **Connection Efficiency**: Shared pools prevent resource exhaustion
- **Cache Hit Rates**: Memory views eliminate unnecessary data copying

### Testing Considerations

- **Environment Detection**: Automatic test-mode pool sizing
- **Cleanup Automation**: Proper disposal methods for test isolation
- **Hypothesis Compatibility**: Resource-efficient patterns for property-based testing
- **Health Validation**: Comprehensive component status checking for test assertions
