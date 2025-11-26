"""
ML Core Integration Layer.

This module provides the essential infrastructure for Nautilus Trader's ML system,
implementing the Universal ML Architecture Patterns for reliable, high-performance
machine learning operations.

## Core Components

### Integration Management
- `MLIntegrationManager`: Automatic wiring of all ML components with progressive fallback
- `ActorStoresRegistries`: Structured initialization of stores and registries for actors
- `init_ml_stores_and_registries`: Factory function for ML component initialization with dependency injection
- `init_actor_stores_and_registries`: (Deprecated) Legacy alias for backward compatibility

### High-Performance Data Structures
- `LockFreeRingBuffer`: Zero-allocation ring buffer for hot path time series operations
- `PreAllocatedFeatureCache`: Pre-allocated feature vectors with memoryview access
- `ReservoirSampler`: Statistical sampling for efficient percentile calculation

### Database Infrastructure
- `EngineManager`: Thread-safe singleton manager for SQLAlchemy engines with connection pooling

### Message Bus Integration
- `attach_publisher_from_env`: Environment-driven message bus publisher attachment

## Universal ML Architecture Patterns

This module implements all five Universal ML Architecture Patterns:

1. **Mandatory 4-Store + 4-Registry Integration**: All ML actors automatically initialize
   with FeatureStore, ModelStore, StrategyStore, DataStore and their corresponding registries.

2. **Protocol-First Interface Design**: Components implement `MLComponentProtocol` for
   structural typing and duck typing support without implementation coupling.

3. **Hot/Cold Path Separation**: High-performance data structures maintain <5ms P99
   latency for hot path operations while keeping heavy I/O in cold paths.

4. **Progressive Fallback Chains**: All external dependencies have fallback strategies
   (PostgreSQL -> DummyStore, Registry loading -> Direct file loading).

5. **Centralized Metrics Bootstrap**: All metrics use `ml.common.metrics_bootstrap`
   to prevent registry conflicts and ensure consistent naming.

## Feature Flag

Set `ML_USE_LEGACY_INTEGRATION_MANAGER=1` to use the legacy monolithic implementation.
By default (flag not set or set to "0"), the decomposed facade implementation is used.

## Usage

### Basic Integration
```python
from ml.core import MLIntegrationManager

# Automatic component wiring with progressive fallback
integration = MLIntegrationManager(
    auto_start_postgres=True,
    auto_migrate=True,
    ensure_healthy=True
)

# All stores and registries are now available
integration.feature_store.write_features(...)
integration.model_registry.register_model(...)
```

### Hot Path Data Structures
```python
from ml.core import LockFreeRingBuffer, PreAllocatedFeatureCache

# Zero-allocation price history
price_buffer = LockFreeRingBuffer(size=1000, dtype=np.float32)
price_buffer.append(100.5)  # O(1) operation

# Pre-allocated feature computation
feature_cache = PreAllocatedFeatureCache(n_features=10, history_size=1000)
current_buffer = feature_cache.get_current_buffer()  # Zero-copy access
```

### Dependency Injection for ML Components
```python
from ml.core import init_ml_stores_and_registries

# Progressive fallback initialization for any ML component
stores_registries = init_ml_stores_and_registries(config)
# Returns all 4 stores + 4 registries with automatic fallback handling

# Use with dependency injection pattern
class FeatureEngineer:
    def __init__(self, config, stores=None):
        self.stores = stores or init_ml_stores_and_registries(config)
```

## Performance Guarantees

- **Hot Path**: <5ms P99 latency for all ring buffer and cache operations
- **Zero Allocations**: Pre-allocated buffers with memoryview access for hot paths
- **Progressive Fallback**: Graceful degradation when PostgreSQL unavailable
- **Thread Safety**: All components support concurrent access patterns

## Environment Configuration

- `ML_AUTO_START_DB=1`: Automatically start PostgreSQL container
- `ML_AUTO_MIGRATE=1`: Run database migrations on startup
- `ML_ALLOW_DUMMY=1`: Enable dummy store fallback mode
- `ML_STRICT_PROTOCOL_VALIDATION=1`: Strict protocol compliance enforcement
- `ML_USE_LEGACY_INTEGRATION_MANAGER=1`: Use legacy implementation instead of facade

"""

import os

# Core integration components

# High-performance data structures (hot path optimized)
from ml.core.cache import LockFreeRingBuffer
from ml.core.cache import PreAllocatedFeatureCache
from ml.core.cache import ReservoirSampler

# Database infrastructure
from ml.core.db_engine import EngineManager


# Integration classes are imported lazily to avoid import cycles
# Feature flag determines which implementation to use


def _use_legacy_integration_manager() -> bool:
    """
    Check if legacy mode is enabled via environment variable.

    Returns
    -------
    bool
        True if ML_USE_LEGACY_INTEGRATION_MANAGER is set to '1'.

    """
    return os.getenv("ML_USE_LEGACY_INTEGRATION_MANAGER", "0") == "1"


__all__ = [
    "ActorStoresRegistries",
    "EngineManager",
    "LockFreeRingBuffer",
    "MLIntegrationManager",
    "PreAllocatedFeatureCache",
    "ReservoirSampler",
    "init_actor_stores_and_registries",  # Deprecated alias
    "init_ml_stores_and_registries",
]


def __getattr__(name: str) -> object:
    """
    Lazy import integration symbols to avoid import-time cycles.

    Uses feature flag ML_USE_LEGACY_INTEGRATION_MANAGER to determine
    which implementation module to import from.

    """
    if name in {
        "ActorStoresRegistries",
        "MLIntegrationManager",
        "init_actor_stores_and_registries",
        "init_ml_stores_and_registries",
    }:
        if _use_legacy_integration_manager():
            # Use legacy monolithic implementation
            from ml.core import integration as _module
        else:
            # Use decomposed facade implementation (default)
            from ml.core import integration_facade as _module

        return getattr(_module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
