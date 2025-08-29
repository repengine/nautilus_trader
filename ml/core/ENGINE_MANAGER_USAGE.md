# EngineManager Usage Guide

## Overview

The `EngineManager` class provides a thread-safe singleton pattern for managing SQLAlchemy database engines with proper connection pooling. This prevents connection pool exhaustion issues that can occur when multiple ML stores are created, especially during hypothesis testing.

## Problem Statement

Previously, each ML store (FeatureStore, ModelStore, StrategyStore) created its own SQLAlchemy engine with its own connection pool. During testing, particularly with hypothesis property-based tests, this could lead to:

- "Too many clients already" errors from PostgreSQL
- Connection pool exhaustion
- Resource leaks
- Test failures due to database connection limits

## Solution

The `EngineManager` implements a singleton pattern that ensures each unique connection string gets exactly one engine instance, preventing the creation of multiple connection pools to the same database.

## Key Features

1. **Singleton Pattern**: One engine per connection string
2. **Thread-Safe**: Safe for concurrent access
3. **Connection Pooling**: Configurable pool sizes with test-aware defaults
4. **Health Checks**: Pool pre-ping for connection validation
5. **Cleanup**: Proper disposal methods for test teardown

## Usage

### Basic Usage

```python
from ml.core import EngineManager

# Get or create an engine (singleton)
engine = EngineManager.get_engine("postgresql://user:pass@localhost/db")

# Subsequent calls return the same instance
engine2 = EngineManager.get_engine("postgresql://user:pass@localhost/db")
assert engine is engine2  # Same instance
```

### Test Configuration

For test environments, use conservative pool settings:

```python
# Test configuration with small pool
test_engine = EngineManager.get_engine(
    "postgresql://test_user:pass@localhost/test_db",
    pool_size=2,        # Small pool for tests
    max_overflow=3,     # Limited overflow
    pool_pre_ping=True  # Health checks
)
```

### Production Configuration

For production, use larger pool sizes:

```python
# Production configuration
prod_engine = EngineManager.get_engine(
    "postgresql://prod_user:pass@prod_host/nautilus",
    pool_size=10,       # Larger pool for production
    max_overflow=20,    # More overflow allowed
    pool_recycle=3600   # Recycle connections after 1 hour
)
```

### Cleanup

Always clean up in test teardown:

```python
def teardown_function():
    """Clean up all database connections."""
    EngineManager.dispose_all()
```

## Integration with ML Stores

To integrate the EngineManager with ML stores, patch the `create_engine` function:

```python
import ml.stores.feature_store

# Patch to use EngineManager
ml.stores.feature_store.create_engine = lambda conn_str, **kwargs: EngineManager.get_engine(
    conn_str,
    pool_size=2,
    max_overflow=3,
)

# Now all FeatureStore instances will share the same engine
fs1 = FeatureStore(connection_string, config)
fs2 = FeatureStore(connection_string, config)
# fs1.engine is fs2.engine  # True
```

## Test Environment Detection

The EngineManager automatically detects test environments and applies conservative settings:

- Test detection: Looks for "test", "temp", "tmp", or ":memory:" in connection string
- Automatically limits pool_size to max 2 for tests
- Automatically limits max_overflow to max 3 for tests

## Monitoring

Check pool status and health:

```python
# Get pool status
status = EngineManager.get_pool_status("postgresql://user:pass@localhost/db")
print(f"Connections in use: {status['checked_out']}/{status['total']}")

# Check if engine exists
if EngineManager.has_engine("postgresql://user:pass@localhost/db"):
    print("Engine exists")

# Get total engine count
count = EngineManager.get_engine_count()
print(f"Managing {count} database engine(s)")
```

## Best Practices

1. **Always use EngineManager** for database connections in ML stores
2. **Set appropriate pool sizes** based on environment (test vs production)
3. **Clean up properly** with `dispose_all()` in test teardown
4. **Monitor pool usage** to detect connection leaks
5. **Use pool_pre_ping=True** for reliability

## Migration Guide

To migrate existing stores to use EngineManager:

1. Import EngineManager: `from ml.core import EngineManager`
2. Replace `create_engine(conn_str)` with `EngineManager.get_engine(conn_str)`
3. Add cleanup in teardown: `EngineManager.dispose_all()`
4. Adjust pool settings based on environment

## Troubleshooting

### "Too many clients already" errors

- Ensure all stores use EngineManager
- Check pool settings are appropriate for environment
- Verify `dispose_all()` is called in teardown
- Monitor with `get_pool_status()`

### Connection timeouts

- Enable `pool_pre_ping=True`
- Set appropriate `pool_recycle` value
- Check database server settings

### Memory leaks

- Ensure proper cleanup with `dispose_all()`
- Check for circular references in store instances
- Monitor engine count with `get_engine_count()`