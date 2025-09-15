# Bus Publishing Standardization Summary

This document summarizes the standardization work completed for bus publishing gating and error handling across all ML stores.

## Issues Identified and Fixed

### 1. Inconsistent Publishing Gating

**Problem**: Different stores had inconsistent gating logic for bus publishing:
- **FeatureStore**: Checked both `_enable_publishing AND publisher is not None`
- **DataStore**: Only checked `publisher is not None`

**Solution**: Standardized all stores to use consistent gating pattern:
```python
if self._enable_publishing and self.publisher is not None:
    # Publish to message bus
```

**Files Modified**:
- `/home/nate/projects/nautilus_trader/ml/stores/data_store.py`

### 2. Topic Building Consistency

**Status**: ✅ Already Standardized

**Finding**: All stores already properly use `MessageBusConfig.from_env()` scheme/prefix through the `BusPublisherMixin._init_bus_publishing()` method:

- **BusPublisherMixin** reads from `MessageBusConfig.from_env()` and sets:
  - `self._topic_scheme` (default: "domain_op")
  - `self._topic_prefix` (default: "events.ml")

- **All stores** use these attributes consistently when building topics:
  ```python
  topic = build_topic_for_stage(
      stage,
      instrument_id,
      scheme=self._topic_scheme,
      prefix=self._topic_prefix,
  )
  ```

### 3. Hot-Path Budget Preservation

**Status**: ✅ Already Implemented

**Finding**: All stores properly implement non-blocking, best-effort publishing:

- **Error Handling**: All publishing operations are wrapped in try-except blocks
- **Non-blocking**: Exceptions are caught and logged without re-raising
- **Best-effort**: Publishing failures don't impact store operations

Example from FeatureStore:
```python
try:
    self.publisher.publish(topic, payload)
except Exception:
    logger.debug("FeatureStore publish failed", exc_info=True)
```

Example from DataStore:
```python
try:
    self.publisher.publish(topic, payload)
except Exception:
    logger.exception("Message bus publish failed for topic %s", topic)
```

Example from mixins (batch operations):
```python
try:
    publisher.publish(topic, payload)
except Exception:
    logger.debug("Batch publish failed", exc_info=True)
```

## Architecture Overview

### Store Publishing Patterns

1. **FeatureStore & DataStore**: Direct publishing with standardized gating
2. **ModelStore & StrategyStore**: Service-layer delegation to `ModelEventService`/`StrategyEventService` which use registry-based event emission

### Gating Logic (Now Standardized)

All stores follow this pattern:
```python
if self._enable_publishing and self.publisher is not None:
    try:
        # Build topic using env-driven scheme/prefix
        topic = build_topic_for_stage(
            stage, instrument_id,
            scheme=self._topic_scheme,  # From MessageBusConfig.from_env()
            prefix=self._topic_prefix   # From MessageBusConfig.from_env()
        )
        # Publish with best-effort error handling
        self.publisher.publish(topic, payload)
    except Exception:
        logger.debug/exception("Publishing failed", exc_info=True)
```

### Environment Configuration

Publishing behavior is controlled by these environment variables:

```bash
# Enable/disable publishing
ML_BUS_ENABLE=true|false

# Topic naming scheme
ML_BUS_SCHEME=domain_op|stage_first

# Topic prefix
ML_BUS_TOPIC_PREFIX=events.ml

# Redis backend (optional)
ML_BUS_BACKEND=redis
ML_BUS_REDIS_URL=redis://localhost:6379/0
ML_BUS_REDIS_STREAM=ml-events
```

## Testing

Comprehensive unit tests were added in:
`/home/nate/projects/nautilus_trader/ml/tests/unit/stores/test_bus_publishing_standardization.py`

### Test Coverage

1. **Gating Logic**: Verifies all combinations of `_enable_publishing` and `publisher` existence
2. **Environment Configuration**: Tests `MessageBusConfig.from_env()` parsing
3. **Error Handling**: Verifies non-blocking behavior when publishing fails
4. **Performance**: Smoke test for hot-path budget preservation
5. **Consistency**: Verifies all stores use the same topic scheme/prefix

### Test Results

```bash
# All tests pass successfully
python -m pytest ml/tests/unit/stores/test_bus_publishing_standardization.py -v
```

## Benefits Achieved

1. **Consistency**: All stores now have identical gating logic and topic building
2. **Reliability**: Publishing failures don't impact store operations
3. **Performance**: Hot-path budget preserved with best-effort publishing
4. **Configurability**: Environment-driven configuration for all publishing behavior
5. **Testability**: Comprehensive test coverage ensures behavior drift is prevented

## Compliance with Universal ML Architecture Patterns

This standardization supports the following patterns:

- **Pattern 3 (Hot/Cold Path Separation)**: Publishing is non-blocking and preserves hot-path performance budgets
- **Pattern 4 (Progressive Fallback)**: Publishing gracefully degrades when message bus is unavailable
- **Pattern 5 (Centralized Metrics Bootstrap)**: All stores use `BusPublisherMixin` for consistent configuration

## Future Maintenance

To maintain this standardization:

1. **New Stores**: Must inherit from `BusPublisherMixin` and call `_init_bus_publishing()`
2. **Publishing Code**: Must use the standardized gating pattern
3. **Testing**: All new publishing functionality should include tests that verify gating behavior
4. **Code Reviews**: Ensure no direct `publisher.publish()` calls without proper gating

The test suite serves as a regression test to catch any drift in bus publishing behavior across stores.