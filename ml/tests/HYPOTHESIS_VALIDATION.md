# Hypothesis Testing Validation for ML Module

## Summary
This document validates the Hypothesis property-based testing approach against the actual ML module implementation.

## ✅ Valid Patterns That Match Implementation

### 1. Actor Testing
- **Classes**: `MLSignalActor`, `OptimizedMLSignalActor`, `SimpleMLSignalActor` exist
- **Methods**: `on_bar()`, `_compute_features()`, `get_health_status()` are real
- **Config**: `MLActorConfig` with proper parameters exists

### 2. Feature Engineering
- **Class**: `FeatureEngineer` exists in `ml/features/engineering.py`
- **Methods**: `calculate_features_batch()`, `calculate_features_online()`
- **Config**: `FeatureConfig` with indicator periods exists

### 3. Model Registry
- **Classes**: `LocalModelRegistry`, `MLflowModelRegistry` exist
- **Methods**: `register_model()`, `get_latest_version()`, rollback support
- **Thread-safe**: Uses threading.Lock for concurrent access

## ❌ Corrections Made

### 1. Method Names
**Before (Wrong)**:
```python
actor.process_features(features)  # Doesn't exist
actor.update_model(model)  # Doesn't exist
```

**After (Correct)**:
```python
actor.on_bar(bar)  # Actual method
actor.hot_swap_model(model)  # For atomic swaps
actor._compute_features(bar)  # Internal feature computation
```

### 2. Class Usage
**Before (Wrong)**:
```python
engineer = FeatureEngineer()  # Missing config
registry = ModelRegistry()  # Abstract class
```

**After (Correct)**:
```python
config = FeatureConfig(sma_periods=[10, 20])
engineer = FeatureEngineer(config)

registry = LocalModelRegistry(Path("/tmp/registry"))
# or
registry = MLflowModelRegistry(tracking_uri)
```

### 3. Configuration
**Before (Wrong)**:
```python
config = MLActorConfig()  # Missing required params
```

**After (Correct)**:
```python
config = MLActorConfig(
    model_path="model.onnx",  # Required
    warm_up_period=10,  # Required
    n_features=20  # Required for some actors
)
```

## 🎯 Key Properties to Test

### 1. Performance Invariants (Actual Requirements)
```python
# From ml/actors/signal.py docstring
P99_FEATURE_COMPUTATION = 500  # microseconds
P99_MODEL_INFERENCE = 2000  # microseconds
P99_END_TO_END = 5000  # microseconds
MEMORY_STABLE_24H = True
ZERO_ALLOCATIONS_HOT_PATH = True
```

### 2. Actual Data Flow
```python
Bar → on_bar() → _compute_features() → predict() → MLSignal → publish
```

### 3. Real Circuit Breaker States
```python
# From ml/actors/base.py
CircuitBreakerState.CLOSED  # Normal operation
CircuitBreakerState.OPEN  # Blocking calls
CircuitBreakerState.HALF_OPEN  # Testing recovery
```

## 📝 Valid Hypothesis Strategies for This System

### 1. Bar Generation (Matches Nautilus Bar)
```python
@st.composite
def nautilus_bar_strategy(draw):
    """Generate valid Nautilus Bar objects."""
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.identifiers import InstrumentId

    low = draw(st.floats(0.01, 1000))
    high = draw(st.floats(low, low * 1.1))
    open_price = draw(st.floats(low, high))
    close = draw(st.floats(low, high))
    volume = draw(st.integers(0, 1000000))

    return Bar(
        bar_type=test_bar_type,
        open=Price(open_price, precision=2),
        high=Price(high, precision=2),
        low=Price(low, precision=2),
        close=Price(close, precision=2),
        volume=Quantity(volume, precision=0),
        ts_event=draw(st.integers(0, 10**18)),
        ts_init=draw(st.integers(0, 10**18))
    )
```

### 2. Model Config Generation
```python
@st.composite
def ml_actor_config_strategy(draw):
    """Generate valid MLActorConfig."""
    from ml.config.base import MLActorConfig

    return MLActorConfig(
        model_path=draw(st.sampled_from(["model.onnx", "model.json"])),
        warm_up_period=draw(st.integers(10, 100)),
        n_features=draw(st.integers(5, 50)),
        prediction_threshold=draw(st.floats(0.5, 0.9)),
        confidence_threshold=draw(st.floats(0.6, 0.95)),
        max_inference_latency_ms=draw(st.floats(1.0, 5.0))
    )
```

### 3. Feature Window Testing
```python
@given(
    window_size=st.integers(10, 1000),
    n_bars=st.integers(100, 10000)
)
def test_feature_window_bounded(window_size, n_bars):
    """Test that feature windows respect bounds."""
    from collections import deque

    # Actual implementation uses deque
    window = deque(maxlen=window_size)

    for i in range(n_bars):
        window.append(create_test_bar())

        # Property: Never exceeds window_size
        assert len(window) <= window_size

        # Property: After filling, maintains exact size
        if i >= window_size:
            assert len(window) == window_size
```

## 🚀 Recommended Approach

1. **Start with existing test patterns** in `ml/tests/unit/`
2. **Add Hypothesis gradually** to existing tests
3. **Focus on critical properties**:
   - Latency bounds
   - Memory stability
   - Thread safety
   - Data consistency
4. **Use actual imports** from ml module, not pseudocode
5. **Verify against implementation** before writing tests

## Example: Actual Working Test

```python
# ml/tests/unit/test_ml_actors_hypothesis.py
from hypothesis import given, strategies as st, settings
import numpy as np
import time

from ml.actors.signal import MLSignalActor
from ml.config.base import MLActorConfig
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.test_kit.stubs.data import TestDataStubs

class TestMLActorProperties:
    """Property-based tests for ML actors."""

    @given(
        n_bars=st.integers(100, 1000),
        warm_up=st.integers(10, 50)
    )
    @settings(max_examples=10, deadline=5000)  # 5 second deadline
    def test_warmup_period_respected(self, n_bars, warm_up):
        """Property: No predictions during warmup period."""
        config = MLActorConfig(
            model_path="test_model.onnx",
            warm_up_period=warm_up
        )
        actor = MLSignalActor(config)

        predictions_during_warmup = 0

        for i in range(n_bars):
            bar = TestDataStubs.bar()
            actor.on_bar(bar)

            if i < warm_up:
                # Should not have predictions yet
                assert actor._prediction_count == 0
            else:
                # Should start predicting after warmup
                assert actor._is_warmed_up
```

This approach ensures tests match the actual implementation rather than theoretical interfaces.
