# ML Module API Method Mapping

## Overview
This document maps the correct method names and APIs for the ML module, addressing discrepancies between test specifications and actual implementation.

## Actor Methods

### BaseMLInferenceActor (base.py)

**Available Methods:**
- `on_bar(bar: Bar) -> None` - Process a bar and generate predictions
- `get_health_status() -> dict[str, Any]` - Get comprehensive health and statistics
- `on_start() -> None` - Initialize actor (called by Nautilus framework)
- `on_stop() -> None` - Cleanup actor (called by Nautilus framework)

**NOT Available (Common Mistakes):**
- ❌ `get_statistics()` - Use `get_health_status()` instead
- ❌ `update_model()` - Models are managed internally
- ❌ `hot_swap_model()` - Use internal `_reload_model()` or ModelSwapper
- ❌ `process_features()` - Not a public method (features processed internally via `_compute_features()`)

### MLSignalActor (signal.py)

**Available Methods:**
- All methods from BaseMLInferenceActor
- `get_signal_statistics() -> dict[str, Any]` - Extended statistics including signal-specific metrics
- `reset_signal_state() -> None` - Reset signal generation state

**Internal Components (not directly accessible):**
- `_model_swapper: ModelSwapper` - Handles atomic model swapping internally
- `_performance_monitor: PerformanceMonitor` - Tracks performance metrics

### OptimizedMLSignalActor (signal.py)

**Available Methods:**
- All methods from MLSignalActor
- `get_performance_stats() -> dict[str, Any]` - Detailed performance statistics

## Feature Engineering

### FeatureEngineer (features/engineering.py)

**Available Methods:**
- `compute_features_batch(data: np.ndarray) -> np.ndarray` - Batch feature computation
- `compute_features_realtime(bar_data: dict) -> np.ndarray` - Real-time feature computation
- `calculate_features_online(current_bar: dict, indicator_manager: IndicatorManager, scaler: Any) -> np.ndarray`

**NOT Available:**
- ❌ `compute_features()` - Use `compute_features_batch()` or `compute_features_realtime()`
- ❌ `compute()` - Not a valid method

## Model Management

### ModelRegistry (registry/local_registry.py)

**Available Methods:**
- `register_model(model_info: dict) -> str` - Register a new model version
- `get_latest_version(model_name: str) -> str | None` - Get latest model version
- `rollback(steps: int) -> bool` - Rollback to previous version

**NOT Available:**
- ❌ `get_current_version()` - Use `get_latest_version(model_name)`

## Correct Usage Examples

### Processing Bars
```python
# ✅ CORRECT
actor = MLSignalActor(config)
actor.on_start()  # Initialize
actor.on_bar(bar)  # Process bar - features computed internally

# ❌ WRONG
actor.process_features(features)  # Not a public method
actor._compute_features(bar)  # Don't call internal methods directly
```

### Getting Status
```python
# ✅ CORRECT
health_status = actor.get_health_status()
bars_processed = health_status["bars_processed"]
predictions_made = health_status["predictions_made"]

# ❌ WRONG
stats = actor.get_statistics()  # Method doesn't exist
```

### Model Updates (Internal Only)
```python
# ✅ CORRECT - Models are reloaded internally based on config
# The actor checks for model updates periodically if configured

# ❌ WRONG - These methods don't exist
actor.update_model(new_model)
actor.hot_swap_model(new_model)
```

## Test Writing Guidelines

### Use Public APIs Only
```python
# ✅ CORRECT Test
def test_actor_processes_bars():
    actor = MLSignalActor(config)
    actor.on_start()

    # Send bars
    for bar in bars:
        actor.on_bar(bar)

    # Check observable behavior
    health = actor.get_health_status()
    assert health["bars_processed"] == len(bars)

# ❌ WRONG Test
def test_actor_internal_state():
    actor = MLSignalActor(config)

    # Don't access private attributes
    assert actor._model is not None  # Bad!
    assert actor._feature_buffer.shape == (10,)  # Bad!
```

### Mock Correctly
```python
# ✅ CORRECT Mock
class MockModel:
    def predict(self, features):
        """Match the actual interface."""
        return np.array([[0.8]]), np.array([[0.9]])  # prediction, confidence

# ❌ WRONG Mock
class MockModel:
    def process_features(self, features):  # Wrong method name!
        return 0.5
```

## Common Patterns

### 1. Actor Lifecycle
```python
actor = MLSignalActor(config)
actor.on_start()  # Initialize (loads model, subscribes to bars)
# ... actor processes bars via on_bar() ...
actor.on_stop()   # Cleanup
```

### 2. Feature Computation (Internal)
```python
# This happens internally in the actor:
# 1. Bar arrives via on_bar()
# 2. Features computed via _compute_features()
# 3. Prediction made via _predict()
# 4. Signal potentially generated
```

### 3. Health Monitoring
```python
health = actor.get_health_status()
# Returns dict with:
# - status: "healthy", "degraded", or "unhealthy"
# - bars_processed: int
# - predictions_made: int
# - model_loaded: bool
# - last_prediction_time: float
# - error_rate: float
# - circuit_breaker_state: str
```

## Migration Guide

### From Test Specifications to Implementation

| Test Specification | Actual Implementation |
|-------------------|----------------------|
| `actor.get_statistics()` | `actor.get_health_status()` |
| `actor.process_features(features)` | Internal only - happens in `on_bar()` via `_compute_features()` |
| `actor.update_model(model)` | Models reload automatically based on config |
| `actor.hot_swap_model(model)` | Handled internally by ModelSwapper |
| `registry.get_current_version()` | `registry.get_latest_version(model_name)` |
| `engineer.compute()` | `engineer.compute_features_batch()` or `compute_features_realtime()` |

## Notes

1. **Internal Methods**: Methods starting with `_` are internal and should not be called directly in tests
2. **Observable Behavior**: Tests should focus on observable behavior via public methods
3. **Model Management**: Model loading/updating is handled internally based on configuration
4. **Feature Engineering**: Feature computation happens automatically during `on_bar()` processing
5. **Thread Safety**: The actors are designed to run in Nautilus's single-threaded event loop

## References

- Actual implementation: `/ml/actors/base.py`, `/ml/actors/signal.py`
- Test examples: `/ml/tests/unit/actors/`
- Integration tests: `/ml/tests/integration/`
