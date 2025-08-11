# ML Testing Protocol

## Overview

This document defines the comprehensive testing protocol for the Nautilus Trader ML module. It establishes standards for test design, implementation, and maintenance to ensure robust, maintainable, and meaningful tests.

## Core Testing Principles

### 1. Test Behavior, Not Implementation
- **DO**: Test observable outcomes and public contracts
- **DON'T**: Test private attributes or internal state
- **Example**:
  ```python
  # ❌ BAD: Testing implementation details
  assert actor._bars_processed == 10
  assert actor._feature_buffer.shape == (1, 20)

  # ✅ GOOD: Testing observable behavior
  stats = actor.get_statistics()
  assert stats["bars_processed"] == 10
  assert actor.get_health_status()["status"] == "healthy"
  ```

### 2. Use Real Components Where Possible
- **DO**: Use minimal but real models for integration tests
- **DON'T**: Over-mock to the point tests become meaningless
- **Example**:
  ```python
  # ❌ BAD: Mock that doesn't represent real behavior
  mock_model = Mock(return_value=0.5)

  # ✅ GOOD: Minimal real model
  model = TestModelFactory.create_minimal_xgboost_model(n_features=5)
  ```

### 3. Test the Full Stack
- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests for complete workflows

## Test Categories

### 1. Contract Tests (`test_*_contracts.py`)
**Purpose**: Define behavioral contracts all implementations must follow

**Requirements**:
- Test public interfaces only
- Focus on invariants and guarantees
- Implementation-agnostic
- Use Hypothesis for property-based testing

**Example Structure**:
```python
from hypothesis import given, strategies as st, assume
import numpy as np

class TestActorContracts:
    """Behavioral contracts all ML actors must satisfy."""

    def test_actor_must_handle_warmup_period(self, any_actor):
        """Actors MUST buffer data during warmup without predictions."""
        # Test the CONTRACT, not the implementation

    def test_actor_must_publish_valid_signals(self, any_actor):
        """Actors MUST publish signals with required fields."""
        # Verify signal structure, not how it's generated

    @given(
        n_bars=st.integers(min_value=1, max_value=1000),
        bar_values=st.floats(min_value=0.01, max_value=10000, allow_nan=False)
    )
    def test_actor_preserves_temporal_order(self, any_actor, n_bars, bar_values):
        """Property: Actor must process bars in order and maintain causality."""
        bars = [create_bar(value=bar_values) for _ in range(n_bars)]

        timestamps = []
        for bar in bars:
            any_actor.on_bar(bar)
            if any_actor.has_signal():
                signal = any_actor.get_latest_signal()
                timestamps.append(signal.ts_event)

        # Property: Signals must be temporally ordered
        assert timestamps == sorted(timestamps)
```

### 2. Unit Tests (`unit/test_*.py`)
**Purpose**: Test individual components in isolation

**Requirements**:
- Fast execution (<100ms per test)
- Minimal dependencies
- Mock external services
- Test edge cases

**Example Structure**:
```python
class TestFeatureEngineer:
    """Unit tests for FeatureEngineer."""

    def test_calculate_features_with_valid_data(self):
        """Test feature calculation with normal inputs."""

    def test_calculate_features_with_missing_data(self):
        """Test handling of missing values."""

    def test_calculate_features_with_extreme_values(self):
        """Test handling of outliers and edge cases."""
```

### 3. Integration Tests (`integration/test_*.py`)
**Purpose**: Test component interactions and data flow

**Requirements**:
- Test realistic workflows
- Use real components (not mocks)
- Verify data consistency across components
- May be slower than unit tests

**Example Structure**:
```python
class TestMLPipeline:
    """Integration tests for ML pipeline."""

    def test_end_to_end_signal_generation(self):
        """Test complete flow from bars to signals."""

    def test_feature_parity_training_inference(self):
        """Verify features match between training and inference."""
```

### 4. Performance Tests (`test_performance.py`)
**Purpose**: Validate latency and throughput requirements

**Requirements**:
- Measure actual timings
- Test under realistic load
- Verify hot path performance
- Check memory usage

**Latency Requirements**:
```python
PERFORMANCE_REQUIREMENTS = {
    "feature_computation": 500,  # microseconds
    "model_inference": 2000,      # microseconds
    "end_to_end_signal": 5000,    # microseconds
}
```

## Test Data Management

### 1. Test Model Factory
All tests requiring ML models should use the centralized factory:

```python
# ml/tests/fixtures/model_factory.py
class TestModelFactory:
    @staticmethod
    def create_minimal_xgboost_model(
        n_features: int,
        model_type: str = "classification",
        output_path: Path = None
    ) -> Path:
        """Create minimal but valid XGBoost model."""

    @staticmethod
    def create_onnx_model(
        n_features: int,
        n_outputs: int = 1,
        output_path: Path = None
    ) -> Path:
        """Create minimal ONNX model for testing."""
```

### 2. Test Data Fixtures
Centralized fixtures for consistent test data:

```python
# ml/tests/conftest.py
@pytest.fixture
def sample_bars(n_bars: int = 100) -> list[Bar]:
    """Generate sample bar data."""

@pytest.fixture
def sample_features(n_samples: int = 100, n_features: int = 10) -> np.ndarray:
    """Generate sample feature matrix."""

@pytest.fixture
def test_model_path(tmp_path, n_features: int = 10) -> Path:
    """Create and return path to test model."""
```

### 3. Data Validation
All test data should be validated:

```python
def validate_test_data(data: Any) -> None:
    """Validate test data meets requirements."""
    assert not np.any(np.isnan(data)), "Test data contains NaN"
    assert not np.any(np.isinf(data)), "Test data contains inf"
    assert data.shape[0] > 0, "Test data is empty"
```

## Property-Based Testing with Hypothesis

### Core Properties for ML Systems

#### 1. Numerical Stability Properties
```python
from hypothesis import given, strategies as st, assume
import numpy as np

class NumericalStabilityProperties:
    """Properties ensuring numerical stability across ML components."""

    @given(
        features=st.arrays(
            dtype=np.float32,
            shape=st.tuples(
                st.integers(1, 100),  # samples
                st.integers(1, 50),   # features
            ),
            elements=st.one_of(
                st.floats(-1e6, 1e6, allow_nan=False),
                st.just(0.0),  # Test zero handling
                st.floats(min_value=1e-10, max_value=1e-8),  # Small values
                st.floats(min_value=1e8, max_value=1e10),    # Large values
            )
        )
    )
    def test_model_handles_extreme_values(self, model, features):
        """Property: Models must handle extreme values without NaN/Inf."""
        predictions = model.predict(features)

        assert not np.any(np.isnan(predictions)), "Model produced NaN"
        assert not np.any(np.isinf(predictions)), "Model produced Inf"
        assert predictions.shape[0] == features.shape[0], "Shape mismatch"

    @given(
        data=st.lists(
            st.floats(allow_nan=True, allow_infinity=True),
            min_size=10,
            max_size=1000
        )
    )
    def test_feature_engineering_handles_dirty_data(self, data):
        """Property: Feature engineering must sanitize dirty data."""
        # Use actual FeatureEngineer from ml/features/engineering.py
        from ml.features.engineering import FeatureEngineer, FeatureConfig

        config = FeatureConfig(
            sma_periods=[10, 20],
            ema_periods=[12, 26],
            use_returns=True,
            use_log_returns=False
        )
        engineer = FeatureEngineer(config)

        # FeatureEngineer has compute_features_batch for batch processing
        features = engineer.compute_features_batch(np.array(data))

        # Property: Output must be clean
        assert not np.any(np.isnan(features)), "Features contain NaN"
        assert not np.any(np.isinf(features)), "Features contain Inf"
```

#### 2. Temporal Consistency Properties
```python
class TemporalConsistencyProperties:
    """Properties ensuring temporal consistency in trading systems."""

    @given(
        timestamps=st.lists(
            st.integers(min_value=0, max_value=10**18),
            min_size=2,
            max_size=100,
            unique=True
        ).map(sorted)  # Ensure temporal order
    )
    def test_no_lookahead_bias(self, strategy, timestamps):
        """Property: Strategies cannot use future information."""
        for i, ts in enumerate(timestamps):
            features = strategy.compute_features_at_time(ts)

            # Property: Features only use data up to current time
            feature_timestamps = strategy.get_feature_timestamps(features)
            assert all(ft <= ts for ft in feature_timestamps)

    @given(
        window_size=st.integers(min_value=10, max_value=1000),
        n_updates=st.integers(min_value=100, max_value=10000)
    )
    def test_rolling_window_consistency(self, window_size, n_updates):
        """Property: Rolling windows maintain fixed size after filling."""
        window = RollingWindow(window_size)

        for i in range(n_updates):
            window.update(i)

            if i >= window_size:
                assert len(window) == window_size
                assert window.oldest() == i - window_size + 1
                assert window.newest() == i
```

#### 3. Concurrency and Atomicity Properties
```python
class ConcurrencyProperties:
    """Properties ensuring thread-safety and atomicity."""

    @given(
        n_concurrent_updates=st.integers(min_value=2, max_value=10),
        updates_per_thread=st.integers(min_value=10, max_value=100)
    )
    def test_model_swap_atomicity(self, n_concurrent_updates, updates_per_thread):
        """Property: Model swaps must be atomic."""
        from ml.actors.signal import MLSignalActor
        from ml.config.base import MLActorConfig

        config = MLActorConfig(
            model_path="test_model.onnx",
            warm_up_period=10
        )
        actor = MLSignalActor(config)

        def swap_model(version):
            # MLSignalActor has hot_swap_model method for atomic swaps
            for i in range(updates_per_thread):
                new_model = create_test_model(version=f"{version}.{i}")
                actor.hot_swap_model(new_model)

        threads = []
        for v in range(n_concurrent_updates):
            t = threading.Thread(target=swap_model, args=(v,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Property: Actor must have exactly one valid model
        assert actor._model is not None
        # Property: No partial updates
        assert actor.get_health_status()["status"] == "healthy"

    @given(
        queue_size=st.integers(min_value=10, max_value=100),
        n_producers=st.integers(min_value=1, max_value=5),
        items_per_producer=st.integers(min_value=50, max_value=200)
    )
    def test_bounded_queue_never_exceeds_limit(self, queue_size, n_producers, items_per_producer):
        """Property: Bounded queues never exceed their limit."""
        queue = BoundedQueue(max_size=queue_size)

        def produce_items(producer_id):
            for i in range(items_per_producer):
                queue.try_add(f"{producer_id}:{i}")

        threads = [
            threading.Thread(target=produce_items, args=(i,))
            for i in range(n_producers)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Property: Queue size never exceeds limit
        assert len(queue) <= queue_size
```

#### 4. Performance Invariant Properties
```python
class PerformanceInvariantProperties:
    """Properties ensuring performance requirements are maintained."""

    @given(
        n_iterations=st.integers(min_value=100, max_value=1000),
        feature_dim=st.integers(min_value=10, max_value=100)
    )
    def test_hot_path_latency_invariant(self, n_iterations, feature_dim):
        """Property: Hot path latency must stay under 5ms P99."""
        # Use actual ML actor from the system
        from ml.actors.signal import MLSignalActor
        from ml.config.base import MLActorConfig

        config = MLActorConfig(
            model_path="test_model.onnx",
            warm_up_period=10,
            n_features=feature_dim
        )
        actor = MLSignalActor(config)

        # Test actual on_bar method
        latencies = []
        for _ in range(n_iterations):
            bar = create_test_bar()
            start = time.perf_counter()
            actor.on_bar(bar)  # Actual method name
            latencies.append(time.perf_counter() - start)

        p99 = np.percentile(latencies, 99)
        assert p99 < 0.005, f"P99 latency {p99*1000:.1f}ms exceeds 5ms"

    @given(
        n_operations=st.integers(min_value=1000, max_value=10000)
    )
    def test_zero_allocation_invariant(self, n_operations):
        """Property: Hot path must have zero allocations."""
        from ml.actors.signal import OptimizedMLSignalActor
        from ml.config.base import MLActorConfig

        # Use OptimizedMLSignalActor which has zero-allocation optimizations
        config = MLActorConfig(
            model_path="test_model.onnx",
            warm_up_period=10,
            use_lock_free_buffers=True,
            enable_zero_copy=True
        )
        actor = OptimizedMLSignalActor(config)

        # Warm up
        for _ in range(100):
            actor.on_bar(create_test_bar())

        # Measure allocations
        import tracemalloc
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(n_operations):
            actor.on_bar(create_test_bar())

        snapshot2 = tracemalloc.take_snapshot()
        stats = snapshot2.compare_to(snapshot1, 'lineno')

        # Property: No significant allocations in hot path
        hot_path_allocations = [
            stat for stat in stats
            if 'on_bar' in str(stat) or '_compute_features' in str(stat)
        ]

        for stat in hot_path_allocations:
            assert stat.size_diff < 1000, f"Hot path allocated {stat.size_diff} bytes"
```

### Hypothesis Strategies for ML Testing

#### Custom Strategies for Trading Data
```python
# ml/tests/strategies.py
from hypothesis import strategies as st
import numpy as np

# Bar data strategy
bar_strategy = st.builds(
    Bar,
    open=st.floats(min_value=0.01, max_value=10000),
    high=st.floats(min_value=0.01, max_value=10000),
    low=st.floats(min_value=0.01, max_value=10000),
    close=st.floats(min_value=0.01, max_value=10000),
    volume=st.integers(min_value=0, max_value=1000000),
    ts_event=st.integers(min_value=0, max_value=10**18)
).filter(lambda bar: bar.low <= bar.high and
         bar.low <= bar.open <= bar.high and
         bar.low <= bar.close <= bar.high)

# Feature matrix strategy
feature_matrix_strategy = st.arrays(
    dtype=np.float32,
    shape=st.tuples(
        st.integers(min_value=1, max_value=1000),  # samples
        st.integers(min_value=1, max_value=100),   # features
    ),
    elements=st.floats(
        min_value=-100,
        max_value=100,
        allow_nan=False,
        allow_infinity=False,
        width=32
    )
)

# Model configuration strategy
model_config_strategy = st.builds(
    MLModelConfig,
    model_type=st.sampled_from(['xgboost', 'lightgbm', 'onnx']),
    n_features=st.integers(min_value=1, max_value=100),
    warmup_bars=st.integers(min_value=10, max_value=1000),
    prediction_threshold=st.floats(min_value=0.0, max_value=1.0)
)
```

### Hypothesis Testing Patterns

#### Pattern 1: Stateful Testing
```python
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
from ml.actors.signal import MLSignalActor
from ml.registry.local_registry import LocalModelRegistry
from pathlib import Path

class MLPipelineStateMachine(RuleBasedStateMachine):
    """Test ML pipeline state transitions using actual components."""

    def __init__(self):
        super().__init__()
        # Use real components from the ML module
        from ml.config.base import MLActorConfig
        config = MLActorConfig(
            model_path="test_model.onnx",
            warm_up_period=10
        )
        self.actor = MLSignalActor(config)
        self.registry = LocalModelRegistry(Path("/tmp/test_registry"))
        self.n_bars_processed = 0
        self.model_versions = []

    @rule(bar=bar_strategy)
    def process_bar(self, bar):
        """Process a bar through the actor."""
        self.actor.on_bar(bar)
        self.n_bars_processed += 1

    @rule()
    def register_model(self):
        """Register a new model version."""
        if self.n_bars_processed > 100:
            # Register model with the actual registry
            model_info = {
                "name": "test_model",
                "version": f"v{len(self.model_versions)}",
                "metrics": {"accuracy": 0.95}
            }
            version_id = self.registry.register_model(model_info)
            self.model_versions.append(version_id)

    @invariant()
    def actor_always_healthy(self):
        """Invariant: Actor always in healthy state."""
        health = self.actor.get_health_status()
        assert health["status"] in ["healthy", "warming_up"]

    @invariant()
    def registry_consistent(self):
        """Invariant: Registry maintains consistency."""
        if self.model_versions:
            latest = self.registry.get_latest_version("test_model")
            assert latest is not None

# Run the state machine test
TestMLPipeline = MLPipelineStateMachine.TestCase
```

#### Pattern 2: Compositional Testing
```python
@composite
def feature_pipeline_strategy(draw):
    """Generate valid feature pipeline configurations."""
    n_features = draw(st.integers(min_value=5, max_value=50))

    # Draw correlated components
    indicators = draw(st.lists(
        st.sampled_from(['SMA', 'EMA', 'RSI', 'MACD']),
        min_size=1,
        max_size=n_features // 2
    ))

    lookback_periods = draw(st.lists(
        st.integers(min_value=5, max_value=100),
        min_size=len(indicators),
        max_size=len(indicators)
    ))

    return FeaturePipelineConfig(
        indicators=indicators,
        lookback_periods=lookback_periods,
        total_features=n_features
    )
```

## Mock Guidelines

### When to Mock
- External services (APIs, databases)
- File I/O in unit tests
- Time-dependent operations
- Heavy computations in unit tests

### When NOT to Mock
- Core business logic
- Data transformations
- Critical calculations
- Integration test components

### Mock Implementation Standards
```python
class MockMLActor(Actor):
    """Mock actor for testing."""

    def __init__(self, config):
        super().__init__(config)
        # Initialize with realistic defaults
        self._setup_realistic_behavior()

    def get_statistics(self) -> dict[str, Any]:
        """Implement ALL public methods."""
        return {
            "bars_processed": self.bar_count,
            "status": "healthy"
        }

    def _setup_realistic_behavior(self):
        """Configure mock to behave realistically."""
        # Return values should be realistic
        # Timing should approximate real behavior
```

## Security Testing

### Model Loading Security
```python
def test_reject_pickle_models():
    """MUST reject pickle files for security."""
    with pytest.raises(ValueError, match="security"):
        loader.load_model("model.pkl")

def test_accept_safe_formats():
    """MUST accept JSON/ONNX formats."""
    loader.load_model("model.json")  # Should work
    loader.load_model("model.onnx")  # Should work
```

### Input Validation
```python
def test_validate_untrusted_input():
    """MUST validate all external inputs."""
    # Test injection attempts
    # Test malformed data
    # Test extreme values
```

## Performance Testing Protocol

### 1. Latency Tests
```python
def test_inference_latency():
    """Test inference meets latency requirements."""
    times = []
    for _ in range(100):
        start = time.perf_counter_ns()
        actor.predict(features)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000  # ms
        times.append(elapsed)

    p99 = np.percentile(times, 99)
    assert p99 < 5.0, f"P99 latency {p99}ms exceeds 5ms requirement"
```

### 2. Memory Tests
```python
def test_no_memory_leaks():
    """Test for memory leaks in hot path."""
    import tracemalloc

    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    # Run operations
    for _ in range(1000):
        actor.on_bar(bar)

    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    # Check for significant memory growth
    for stat in top_stats[:10]:
        assert stat.size_diff < 1_000_000  # Less than 1MB growth
```

### 3. Zero-Allocation Tests
```python
def test_hot_path_zero_allocation():
    """Verify hot path has zero allocations."""
    # Use memory_profiler or custom allocation tracking
    # Verify feature buffers are reused
    # Check no new arrays created per prediction
```

## Test Coverage Requirements

### Minimum Coverage Targets
- **Overall ML module**: ≥ 90%
- **Critical paths** (actors, strategies): ≥ 95%
- **Utility functions**: ≥ 80%
- **Experimental features**: ≥ 70%

### Coverage Enforcement
```yaml
# .coveragerc
[run]
source = ml/
omit =
    ml/tests/*
    ml/experimental/*

[report]
fail_under = 90
show_missing = True
skip_covered = False
```

## Test Execution Guidelines

### 1. Test Organization
```bash
ml/tests/
├── unit/                 # Fast, isolated tests
│   ├── test_actors.py
│   ├── test_features.py
│   └── test_models.py
├── integration/          # Component interaction tests
│   ├── test_pipeline.py
│   └── test_backtest.py
├── contracts/           # Behavioral contract tests
│   ├── test_actor_contracts.py
│   └── test_model_contracts.py
├── performance/         # Performance tests
│   └── test_latency.py
├── fixtures/            # Shared test utilities
│   ├── model_factory.py
│   └── data_generators.py
└── conftest.py         # Pytest configuration
```

### 2. Test Execution Order
```bash
# 1. Run unit tests first (fast feedback)
pytest ml/tests/unit/ -v

# 2. Run contract tests (verify interfaces)
pytest ml/tests/contracts/ -v

# 3. Run integration tests (verify interactions)
pytest ml/tests/integration/ -v

# 4. Run performance tests (verify requirements)
pytest ml/tests/performance/ -v

# Full test suite
pytest ml/tests/ -v --cov=ml --cov-report=html
```

### 3. Continuous Integration
```yaml
# .github/workflows/ml-tests.yml
- name: Run ML Tests
  run: |
    # Fast fail on unit tests
    pytest ml/tests/unit/ --fail-fast

    # Run full suite if units pass
    pytest ml/tests/ --cov=ml --cov-fail-under=90

    # Run performance tests
    pytest ml/tests/performance/ --benchmark-only
```

## Test Maintenance

### 1. Test Review Checklist
- [ ] Tests focus on behavior, not implementation
- [ ] No hardcoded test data in test files
- [ ] Uses appropriate fixtures
- [ ] Mocks are justified and realistic
- [ ] Performance requirements verified
- [ ] Security considerations tested
- [ ] Edge cases covered
- [ ] Documentation updated

### 2. Test Refactoring Guidelines
When refactoring tests:
1. Preserve behavioral coverage
2. Improve readability
3. Reduce duplication via fixtures
4. Update documentation
5. Verify performance unchanged

### 3. Test Debugging Protocol
```python
# Enable detailed logging for debugging
@pytest.mark.debug
def test_complex_scenario(caplog):
    """Use caplog for debugging test failures."""
    with caplog.at_level(logging.DEBUG):
        # Test code
        pass

    # Examine logs on failure
    if failed:
        print(caplog.text)
```

## Common Pitfalls to Avoid

### 1. Testing Implementation Details
```python
# ❌ BAD: Brittle test tied to implementation
assert actor._internal_buffer.size == 100
assert actor._state_machine.current == "READY"

# ✅ GOOD: Testing observable behavior
assert actor.is_ready()
assert len(actor.get_buffered_data()) == 100
```

### 2. Inadequate Test Isolation
```python
# ❌ BAD: Tests depend on execution order
class TestActor:
    actor = None  # Shared state!

    def test_1_init(self):
        self.actor = Actor()

    def test_2_process(self):
        self.actor.process()  # Fails if test_1 didn't run

# ✅ GOOD: Each test is independent
class TestActor:
    def test_init(self):
        actor = Actor()
        assert actor.is_initialized()

    def test_process(self):
        actor = Actor()
        actor.process()
        assert actor.processed
```

### 3. Overmocking
```python
# ❌ BAD: Mock everything, test nothing
@patch('ml.actors.signal.MLSignalActor.on_bar')
@patch('ml.actors.signal.MLSignalActor.predict')
@patch('ml.actors.signal.MLSignalActor.publish_signal')
def test_actor(mock1, mock2, mock3):
    # This tests mocks, not the actor!

# ✅ GOOD: Test real behavior with minimal mocking
def test_actor():
    actor = MLSignalActor(config)
    actor.on_bar(test_bar)
    signals = actor.get_published_signals()
    assert len(signals) > 0
```

## Test Quality Metrics

### 1. Test Effectiveness Metrics
- **Mutation Score**: >80% (using mutmut)
- **Defect Detection Rate**: Track bugs found by tests
- **False Positive Rate**: <5% flaky tests
- **Execution Time**: Unit tests <10s, Integration <60s

### 2. Test Maintainability Metrics
- **Test Code Ratio**: 1:1 with production code
- **Fixture Reuse**: >50% tests use shared fixtures
- **Documentation Coverage**: 100% of test classes documented
- **Assertion Density**: 2-5 assertions per test

## Appendix: Test Templates

### Unit Test Template
```python
"""Unit tests for [Component Name]."""

import pytest
import numpy as np
from ml.component import Component

class TestComponent:
    """Test suite for Component."""

    @pytest.fixture
    def component(self):
        """Create component instance for testing."""
        return Component(test_config)

    def test_normal_operation(self, component):
        """Test component under normal conditions."""
        result = component.process(valid_input)
        assert result.is_valid()
        assert result.value > 0

    def test_edge_case(self, component):
        """Test component with edge case input."""
        result = component.process(edge_input)
        assert result.handled_gracefully()

    def test_error_handling(self, component):
        """Test component error handling."""
        with pytest.raises(ValueError):
            component.process(invalid_input)
```

### Integration Test Template
```python
"""Integration tests for [Feature Name]."""

import pytest
from ml.pipeline import Pipeline

class TestFeatureIntegration:
    """Integration tests for feature."""

    @pytest.fixture
    def pipeline(self):
        """Create full pipeline for testing."""
        return Pipeline.create_test_pipeline()

    def test_end_to_end_flow(self, pipeline):
        """Test complete data flow through pipeline."""
        input_data = create_test_data()
        result = pipeline.process(input_data)

        # Verify data transformations
        assert result.shape == expected_shape
        assert result.dtype == expected_dtype

        # Verify business logic
        assert result.meets_requirements()
```

### Performance Test Template
```python
"""Performance tests for [Component Name]."""

import pytest
import time
import numpy as np

class TestComponentPerformance:
    """Performance test suite."""

    @pytest.mark.benchmark
    def test_latency_requirement(self, benchmark):
        """Test component meets latency requirements."""
        component = Component()
        data = create_test_data()

        result = benchmark(component.process, data)

        # Verify latency
        assert benchmark.stats['mean'] < 0.001  # 1ms
        assert benchmark.stats['max'] < 0.005   # 5ms

    @pytest.mark.memory
    def test_memory_usage(self):
        """Test component memory usage."""
        import tracemalloc

        tracemalloc.start()
        component = Component()

        # Process multiple iterations
        for _ in range(1000):
            component.process(data)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Verify memory usage
        assert peak < 100_000_000  # 100MB limit
```

---

*Last Updated: 2024*
*Version: 1.0*
