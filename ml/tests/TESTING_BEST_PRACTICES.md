# ML Module Testing Best Practices Guide

This comprehensive guide establishes testing standards and best practices for the Nautilus Trader ML module, ensuring robust, maintainable, and meaningful tests that support high-quality machine learning systems.

## Table of Contents

1. [Core Testing Principles](#core-testing-principles)
2. [Testing Standards & Conventions](#testing-standards--conventions)
3. [Test Categories & Organization](#test-categories--organization)
4. [Unit Testing Guidelines](#unit-testing-guidelines)
5. [Integration Testing Patterns](#integration-testing-patterns)
6. [Mock Objects & Test Isolation](#mock-objects--test-isolation)
7. [Test Data Management & Fixtures](#test-data-management--fixtures)
8. [Performance Testing for Hot Path Code](#performance-testing-for-hot-path-code)
9. [Property-Based Testing with Hypothesis](#property-based-testing-with-hypothesis)
10. [Coverage Requirements & Measurement](#coverage-requirements--measurement)
11. [CI/CD Integration](#cicd-integration)
12. [Common Anti-Patterns to Avoid](#common-anti-patterns-to-avoid)

## Core Testing Principles

### 1. Test Behavior, Not Implementation

Focus on observable outcomes and public contracts rather than internal implementation details.

```python
# ❌ BAD: Testing implementation details
def test_actor_internal_state():
    actor = MLSignalActor(config)
    actor.on_bar(test_bar)
    assert actor._bars_processed == 1  # Internal state
    assert actor._feature_buffer.shape == (1, 20)  # Implementation detail

# ✅ GOOD: Testing observable behavior
def test_actor_processes_bars_correctly():
    actor = MLSignalActor(config)
    actor.on_bar(test_bar)
    
    stats = actor.get_statistics()
    assert stats["bars_processed"] == 1
    assert actor.get_health_status()["status"] == "healthy"
    assert actor.has_generated_signal() is True
```

### 2. Use Real Components Where Feasible

Prefer minimal but real models and components over excessive mocking.

```python
# ❌ BAD: Over-mocking that doesn't represent real behavior
def test_model_inference_mocked():
    mock_model = Mock(return_value=np.array([0.5]))
    actor = MLSignalActor(config, model=mock_model)
    # This test doesn't validate real model behavior

# ✅ GOOD: Using real minimal model
def test_model_inference_real():
    model_path = TestModelFactory.create_minimal_xgboost_model(n_features=5)
    config = MLActorConfig(model_path=model_path, ...)
    actor = MLSignalActor(config)
    
    result = actor.predict(features)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert 0.0 <= result[0] <= 1.0
```

### 3. Maintain Test Hierarchy

- **Unit Tests**: Fast, isolated, focused on single components
- **Integration Tests**: Component interactions and data flows
- **End-to-End Tests**: Complete workflows and system behavior

## Testing Standards & Conventions

### Naming Conventions

```python
# Test file naming: test_<module_name>.py
# ml/features/engineering.py → ml/tests/unit/features/test_engineering.py

class TestFeatureEngineer:
    """Test class naming: Test<ClassName>"""
    
    def test_calculate_features_with_valid_data_returns_correct_shape(self):
        """Test method naming: test_<action>_<condition>_<expected_outcome>"""
        pass
    
    def test_calculate_features_when_insufficient_data_raises_value_error(self):
        """Clear, descriptive test names that explain the scenario"""
        pass
```

### Test Structure (AAA Pattern)

```python
def test_feature_engineer_calculates_technical_indicators():
    # ARRANGE: Set up test data and dependencies
    config = FeatureConfig(return_periods=[1, 5, 10])
    engineer = FeatureEngineer(config)
    df = create_test_price_data(n_samples=100)
    
    # ACT: Execute the code under test
    features, metadata = engineer.calculate_features(df)
    
    # ASSERT: Verify the expected outcomes
    assert features.shape[0] == df.shape[0]
    assert features.shape[1] == len(config.get_feature_names())
    assert not features.isnull().any().any()
    assert metadata["feature_count"] == features.shape[1]
```

### Documentation Standards

```python
def test_signal_actor_handles_circuit_breaker_activation():
    """
    Test that signal actor properly activates circuit breaker on consecutive failures.
    
    Verifies:
    - Circuit breaker activates after threshold failures
    - Actor stops processing during circuit break
    - Circuit breaker resets after timeout period
    - Metrics are properly updated during state transitions
    """
    # Test implementation...
```

## Test Categories & Organization

### Directory Structure

```
ml/tests/
├── conftest.py                    # Global fixtures
├── conftest_hypothesis.py        # Hypothesis configuration
├── fixtures/                     # Test model and data factories
│   └── model_factory.py
├── benchmarks/                   # Performance benchmarks
├── contracts/                    # Behavioral contract tests
│   ├── test_actor_contracts.py
│   ├── test_registry_contracts.py
│   └── test_strategy_contracts.py
├── integration/                  # Cross-component tests
├── performance/                  # Hot path performance tests
├── property/                     # Hypothesis property tests
├── unit/                        # Isolated component tests
│   ├── actors/
│   ├── features/
│   ├── registry/
│   └── strategies/
└── data/                        # Test data and model artifacts
```

### Contract Tests

Define behavioral contracts that all implementations must satisfy:

```python
# ml/tests/contracts/test_actor_contracts.py
class TestMLActorContracts:
    """Behavioral contracts all ML actors must satisfy."""
    
    @pytest.fixture(params=['MLSignalActor', 'CustomMLActor'])
    def any_ml_actor(self, request):
        """Parametrized fixture providing different actor implementations."""
        actor_class = get_actor_class(request.param)
        config = create_valid_config_for_actor(actor_class)
        return actor_class(config)
    
    def test_actor_must_handle_warmup_period(self, any_ml_actor):
        """CONTRACT: Actors must buffer data during warmup without predictions."""
        warmup_bars = 10
        
        for i in range(warmup_bars):
            bar = TestDataStubs.bar_5decimal(ts_event=i)
            any_ml_actor.on_bar(bar)
            
            # During warmup, no signals should be generated
            assert not any_ml_actor.has_generated_signal()
        
        # After warmup, signals should be possible
        post_warmup_bar = TestDataStubs.bar_5decimal(ts_event=warmup_bars)
        any_ml_actor.on_bar(post_warmup_bar)
        # Signal generation is now possible (implementation dependent)
```

## Unit Testing Guidelines

### Testing Individual Components

```python
class TestFeatureEngineer:
    """Unit tests for FeatureEngineer component."""
    
    def setup_method(self):
        """Setup run before each test method."""
        self.config = FeatureConfig(
            return_periods=[1, 5, 10],
            momentum_periods=[10, 20],
            rsi_period=14,
        )
        self.engineer = FeatureEngineer(self.config)
    
    def test_calculate_features_batch_mode_success(self):
        """Test successful feature calculation in batch mode."""
        # Create valid test data
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0] * 50,
            'high': [101.0, 102.0, 103.0] * 50,
            'low': [99.0, 100.0, 101.0] * 50,
            'close': [100.5, 101.5, 102.5] * 50,
            'volume': [1000000.0] * 150,
        })
        
        features, metadata = self.engineer.calculate_features(df, mode='batch')
        
        # Verify output structure
        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == df.shape[0]
        assert features.shape[1] == len(self.config.get_feature_names())
        
        # Verify data quality
        assert not features.isnull().all().any()  # No completely null columns
        assert features.dtypes.apply(lambda x: x.kind in 'biufc').all()  # Numeric types
        
        # Verify metadata
        assert metadata['mode'] == 'batch'
        assert metadata['feature_count'] == features.shape[1]
    
    def test_calculate_features_with_insufficient_data_raises_error(self):
        """Test that insufficient data raises appropriate error."""
        insufficient_df = pd.DataFrame({
            'close': [100.0, 101.0],  # Only 2 samples
            'volume': [1000000.0, 1000000.0],
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            self.engineer.calculate_features(insufficient_df)
    
    def test_feature_names_consistency(self):
        """Test that feature names are consistent across calls."""
        df = create_test_price_data(n_samples=100)
        
        features1, _ = self.engineer.calculate_features(df)
        features2, _ = self.engineer.calculate_features(df)
        
        assert features1.columns.tolist() == features2.columns.tolist()
        assert features1.columns.tolist() == self.config.get_feature_names()
```

### Testing Error Conditions

```python
def test_model_loading_with_invalid_path_raises_error():
    """Test model loading failure with invalid path."""
    config = MLActorConfig(
        model_path="/nonexistent/path/model.onnx",
        component_id="TEST_ACTOR",
    )
    
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        MLSignalActor(config)

def test_feature_calculation_with_nan_data_handles_gracefully():
    """Test graceful handling of NaN data in features."""
    df = create_test_price_data(n_samples=100)
    df.loc[50:55, 'close'] = np.nan  # Inject NaN values
    
    engineer = FeatureEngineer(FeatureConfig())
    features, metadata = engineer.calculate_features(df)
    
    # Should handle NaN appropriately (forward fill, etc.)
    assert not features.isnull().all().any()
    assert metadata['nan_count'] > 0  # Should track NaN handling
```

## Integration Testing Patterns

### Store Integration Tests

```python
class TestFeatureStoreIntegration:
    """Integration tests for FeatureStore with database."""
    
    @pytest.fixture
    def feature_store(self, database_url):
        """Setup feature store with test database."""
        store = FeatureStore(database_url)
        yield store
        store.cleanup()  # Cleanup after test
    
    def test_store_and_retrieve_features(self, feature_store):
        """Test complete store and retrieval cycle."""
        # Generate test features
        features = create_test_features(n_samples=10)
        metadata = FeatureMetadata(
            instrument_id="EURUSD.SIM",
            feature_set="test_features_v1",
            ts_event=1234567890,
        )
        
        # Store features
        feature_store.store_features(features, metadata)
        
        # Retrieve features
        retrieved = feature_store.get_features(
            instrument_id="EURUSD.SIM",
            feature_set="test_features_v1",
            ts_start=1234567880,
            ts_end=1234567900,
        )
        
        # Verify data integrity
        pd.testing.assert_frame_equal(features, retrieved.features)
        assert retrieved.metadata.instrument_id == metadata.instrument_id
```

### Actor-to-Strategy Communication

```python
def test_signal_actor_to_strategy_communication():
    """Test actor publishes signals that strategies receive."""
    # Setup components
    signal_config = MLSignalActorConfig(
        model_path=TestModelFactory.create_minimal_xgboost_model(),
        bar_type="EURUSD.SIM-1-MINUTE-BID-EXTERNAL",
    )
    signal_actor = MLSignalActor(signal_config)
    
    strategy_config = SimpleMLStrategyConfig(
        signal_source=signal_actor.component_id,
    )
    strategy = SimpleMLStrategy(strategy_config)
    
    # Connect via test message bus
    message_bus = TestMessageBus()
    signal_actor.register_message_bus(message_bus)
    strategy.register_message_bus(message_bus)
    
    # Send market data
    test_bar = TestDataStubs.bar_5decimal()
    signal_actor.on_bar(test_bar)
    
    # Process messages
    message_bus.process_all_messages()
    
    # Verify signal was received
    assert len(strategy.received_signals) == 1
    signal = strategy.received_signals[0]
    assert signal.instrument_id == test_bar.bar_type.instrument_id
    assert signal.ts_event >= test_bar.ts_event
```

### Training to Inference Parity

```python
def test_training_inference_feature_parity():
    """Ensure training features exactly match inference features."""
    config = FeatureConfig(return_periods=[1, 5, 10])
    
    # Create identical test data
    test_data = create_test_price_data(n_samples=100)
    
    # Calculate training features (batch mode)
    engineer = FeatureEngineer(config)
    training_features, _ = engineer.calculate_features(test_data, mode='batch')
    
    # Calculate inference features (online mode)
    indicator_mgr = IndicatorManager(config)
    
    # Warm up indicators with historical data
    for i in range(len(test_data) - 1):
        bar = create_bar_from_row(test_data.iloc[i])
        indicator_mgr.update_from_bar(bar)
    
    # Calculate final features online
    current_bar = row_to_dict(test_data.iloc[-1])
    inference_features = engineer.calculate_features_online(
        current_bar=current_bar,
        indicator_manager=indicator_mgr,
    )
    
    # Verify exact parity (strict tolerance for ML)
    training_final = training_features.iloc[-1].values
    np.testing.assert_allclose(
        training_final,
        inference_features,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Training and inference features must match exactly"
    )
```

## Mock Objects & Test Isolation

### Strategic Mocking Guidelines

Mock external dependencies while keeping core ML logic real:

```python
class TestMLSignalActor:
    """Test MLSignalActor with strategic mocking."""
    
    def test_actor_handles_model_loading_failure(self):
        """Test graceful handling of model loading failures."""
        # Mock the model loader to simulate failure
        with patch('ml.actors.signal.load_onnx_model') as mock_loader:
            mock_loader.side_effect = RuntimeError("Model corrupted")
            
            config = MLSignalActorConfig(model_path="test_model.onnx")
            
            with pytest.raises(RuntimeError, match="Model corrupted"):
                MLSignalActor(config)
    
    def test_actor_publishes_metrics_on_prediction(self):
        """Test that actor publishes Prometheus metrics."""
        # Use real model but mock metrics registry
        model_path = TestModelFactory.create_minimal_xgboost_model()
        
        with patch('ml.common.metrics.PREDICTION_COUNTER') as mock_counter:
            config = MLSignalActorConfig(model_path=model_path)
            actor = MLSignalActor(config)
            
            # Process a bar to trigger prediction
            bar = TestDataStubs.bar_5decimal()
            actor.on_bar(bar)
            
            # Verify metrics were updated
            mock_counter.inc.assert_called_once()
```

### Test Isolation Patterns

```python
class TestFeatureStore:
    """Isolated tests for FeatureStore."""
    
    def setup_method(self):
        """Create isolated test database for each test."""
        self.test_db_url = create_test_database()
        self.store = FeatureStore(self.test_db_url)
    
    def teardown_method(self):
        """Clean up after each test."""
        self.store.close()
        cleanup_test_database(self.test_db_url)
    
    def test_store_features_isolated(self):
        """Test feature storage in isolation."""
        # This test runs with a fresh database
        features = create_test_features()
        
        self.store.store_features(features)
        
        # Verify storage
        count = self.store.get_feature_count()
        assert count == len(features)
```

## Test Data Management & Fixtures

### Using the Test Model Factory

```python
# ml/tests/fixtures/model_factory.py provides safe, minimal models
def test_with_minimal_xgboost_model():
    """Test using factory-created minimal model."""
    model_path = TestModelFactory.create_minimal_xgboost_model(
        n_features=10,
        model_type='classification'
    )
    
    # Model is valid and safe to use
    model = joblib.load(model_path)
    test_data = np.random.randn(5, 10)
    predictions = model.predict(test_data)
    
    assert predictions.shape == (5,)
    # Cleanup is automatic - factory uses temp files

def test_with_onnx_model():
    """Test with ONNX model for inference."""
    onnx_path = TestModelFactory.create_onnx_model(
        n_features=5,
        n_outputs=1
    )
    
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    
    input_data = np.random.randn(1, 5).astype(np.float32)
    outputs = session.run(None, {'input': input_data})
    
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 1)
```

### Global Fixtures

```python
# ml/tests/conftest.py
@pytest.fixture(scope="session")
def test_database():
    """Session-scoped test database."""
    db_url = setup_test_database()
    yield db_url
    cleanup_test_database(db_url)

@pytest.fixture
def feature_config():
    """Standard feature configuration for tests."""
    return FeatureConfig(
        return_periods=[1, 5, 10],
        momentum_periods=[10, 20],
        rsi_period=14,
        bollinger_period=20,
        bollinger_std=2.0,
    )

@pytest.fixture
def sample_price_data():
    """Sample price data for testing."""
    np.random.seed(42)  # Reproducible
    n_samples = 1000
    
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    return pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.001,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.uniform(900000, 1100000, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
    })
```

## Performance Testing for Hot Path Code

### Zero-Allocation Tests

```python
# ml/tests/performance/test_hot_path_fixes.py
class TestHotPathPerformance:
    """Test performance requirements for hot path code."""
    
    def test_feature_engineer_returns_view_not_copy(self):
        """Verify FeatureEngineer returns views for zero allocation."""
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        
        # Set up test data
        current_bar = {
            'open': 100.0, 'high': 101.0, 'low': 99.0, 
            'close': 100.5, 'volume': 1000000.0
        }
        
        # Calculate features (hot path)
        features = engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=create_warmed_indicator_manager(),
        )
        
        # Verify it's a view, not a copy (zero allocation)
        assert np.shares_memory(features, engineer.feature_buffer)
        assert not features.isnull().any()
    
    @pytest.mark.benchmark(group="inference")
    def test_model_inference_latency(self, benchmark):
        """Benchmark model inference latency (<5ms requirement)."""
        model_path = TestModelFactory.create_minimal_xgboost_model()
        actor = MLSignalActor(MLActorConfig(model_path=model_path))
        
        test_features = np.random.randn(1, 10).astype(np.float32)
        
        # Benchmark the hot path
        result = benchmark(actor._predict, test_features)
        
        # Verify performance requirement
        assert benchmark.stats.mean < 0.005  # <5ms mean latency
        assert isinstance(result, np.ndarray)
```

### Memory Usage Tests

```python
def test_feature_buffer_memory_stability():
    """Test that feature buffers don't grow over time."""
    import psutil
    import gc
    
    config = FeatureConfig()
    engineer = FeatureEngineer(config)
    
    # Measure initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Process many bars (simulate long-running)
    for i in range(1000):
        bar_data = create_random_bar_data()
        engineer.calculate_features_online(bar_data)
        
        if i % 100 == 0:
            gc.collect()  # Force cleanup
    
    # Check memory hasn't grown significantly
    final_memory = process.memory_info().rss
    memory_growth = (final_memory - initial_memory) / initial_memory
    
    assert memory_growth < 0.1  # <10% memory growth allowed
```

## Property-Based Testing with Hypothesis

### Feature Engineering Properties

```python
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

class TestFeatureEngineerProperties:
    """Property-based tests for feature engineering."""
    
    @given(
        n_samples=st.integers(min_value=100, max_value=1000),
        return_periods=st.lists(
            st.integers(min_value=1, max_value=50),
            min_size=1, max_size=5, unique=True
        ),
    )
    @settings(max_examples=50, deadline=5000)
    def test_feature_count_consistency_property(self, n_samples, return_periods):
        """Property: Feature count must be consistent across batch/online modes."""
        # Ensure we have enough data
        assume(n_samples > max(return_periods) + 50)
        
        config = FeatureConfig(return_periods=sorted(return_periods))
        engineer = FeatureEngineer(config)
        
        # Generate test data
        df = create_synthetic_price_data(n_samples)
        
        # Calculate batch features
        features_batch, _ = engineer.calculate_features(df, mode='batch')
        
        # Calculate online features for last sample
        indicator_mgr = create_warmed_indicator_manager(df[:-1], config)
        current_bar = df.iloc[-1].to_dict()
        features_online = engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_mgr,
        )
        
        # Property: Same number of features
        batch_count = features_batch.shape[1]
        online_count = len(features_online)
        assert batch_count == online_count, f"Feature count mismatch: batch={batch_count}, online={online_count}"
    
    @given(
        close_prices=st.lists(
            st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
            min_size=100, max_size=500
        ),
        rsi_period=st.integers(min_value=2, max_value=50),
    )
    @settings(max_examples=30)
    def test_rsi_bounds_property(self, close_prices, rsi_period):
        """Property: RSI must always be between 0 and 100."""
        assume(len(close_prices) > rsi_period + 20)
        
        config = FeatureConfig(rsi_period=rsi_period)
        engineer = FeatureEngineer(config)
        
        df = create_test_df_from_closes(close_prices)
        features, _ = engineer.calculate_features(df, mode='batch')
        
        # Find RSI column
        rsi_cols = [col for col in features.columns if 'rsi' in col.lower()]
        assume(len(rsi_cols) > 0)
        
        for col in rsi_cols:
            rsi_values = features[col].dropna()
            assert (rsi_values >= 0).all(), f"RSI values below 0 found in {col}"
            assert (rsi_values <= 100).all(), f"RSI values above 100 found in {col}"
```

### Stateful Testing

```python
class TestActorStateMachine(RuleBasedStateMachine):
    """Stateful testing for ML actors."""
    
    def __init__(self):
        super().__init__()
        self.actor = MLSignalActor(create_test_config())
        self.bars_sent = 0
        self.signals_received = []
    
    @rule(bar_price=st.floats(min_value=50.0, max_value=200.0))
    def send_bar(self, bar_price):
        """Send a bar to the actor."""
        bar = create_test_bar(price=bar_price, ts_event=self.bars_sent)
        self.actor.on_bar(bar)
        self.bars_sent += 1
        
        # Collect any new signals
        if self.actor.has_generated_signal():
            signal = self.actor.get_latest_signal()
            self.signals_received.append(signal)
    
    @invariant()
    def signals_are_temporally_ordered(self):
        """Invariant: Signals must be in temporal order."""
        if len(self.signals_received) > 1:
            timestamps = [s.ts_event for s in self.signals_received]
            assert timestamps == sorted(timestamps), "Signals out of temporal order"
    
    @invariant()
    def no_signals_during_warmup(self):
        """Invariant: No signals should be generated during warmup."""
        if self.bars_sent < self.actor.warm_up_period:
            assert len(self.signals_received) == 0, "Signal generated during warmup"

# Run the stateful test
TestActorStateMachineTest = TestActorStateMachine.TestCase
```

## Coverage Requirements & Measurement

### Coverage Thresholds

- **ML Module**: ≥90% coverage (enforced by pre-commit hooks)
- **General Code**: ≥80% coverage
- **Integration Tests**: ≥80% coverage of critical paths

### Running Coverage

```bash
# Run tests with coverage reporting
pytest ml/tests/ --cov=ml --cov-report=html --cov-report=term-missing

# Check coverage for specific modules
pytest ml/tests/unit/features/ --cov=ml.features --cov-report=term-missing

# Generate detailed HTML report
pytest ml/tests/ --cov=ml --cov-report=html:htmlcov
open htmlcov/index.html
```

### Coverage Configuration

```ini
# pyproject.toml
[tool.coverage.run]
source = ["ml"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
fail_under = 90  # For ML module
```

### Measuring Coverage

```python
# Example coverage test
def test_feature_engineer_coverage():
    """Verify all public methods are tested."""
    from ml.features.engineering import FeatureEngineer
    
    public_methods = [
        method for method in dir(FeatureEngineer)
        if not method.startswith('_') and callable(getattr(FeatureEngineer, method))
    ]
    
    # Verify all public methods have corresponding tests
    test_methods = [
        method for method in dir(TestFeatureEngineer)
        if method.startswith('test_')
    ]
    
    for method in public_methods:
        matching_tests = [t for t in test_methods if method in t]
        assert len(matching_tests) > 0, f"No tests found for {method}"
```

## CI/CD Integration

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-ml-test-coverage
        name: check ML test coverage
        description: Ensures new ML modules have ≥90% test coverage
        entry: .pre-commit-hooks/check_test_coverage.py
        language: python
        files: ^ml/.*\.py$
        pass_filenames: true
```

### GitHub Actions

```yaml
# .github/workflows/ml-tests.yml
name: ML Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          uv sync --all-groups --all-extras
      
      - name: Run ML unit tests
        run: |
          pytest ml/tests/unit/ -v --cov=ml --cov-fail-under=90
      
      - name: Run ML integration tests
        run: |
          pytest ml/tests/integration/ -v
      
      - name: Run property tests
        run: |
          HYPOTHESIS_PROFILE=ci pytest ml/tests/property/ -v
```

### Make Targets

```makefile
# Makefile ML test targets
pytest-ml:  #-- Run all ML tests
	pytest ml/tests/ -v

pytest-ml-coverage:  #-- Run ML tests with coverage
	pytest ml/tests/ --cov=ml --cov-fail-under=90 --cov-report=html

pytest-ml-unit:  #-- Run only ML unit tests
	pytest ml/tests/unit/ -v

pytest-ml-integration:  #-- Run ML integration tests  
	pytest ml/tests/integration/ -v

pytest-ml-property:  #-- Run property tests
	HYPOTHESIS_PROFILE=dev pytest ml/tests/property/ -v
```

## Common Anti-Patterns to Avoid

### 1. Testing Implementation Details

```python
# ❌ DON'T: Test private attributes
def test_actor_bad():
    actor = MLSignalActor(config)
    actor.on_bar(test_bar)
    assert actor._internal_counter == 1  # Brittle
    assert len(actor._predictions_cache) == 1  # Implementation detail

# ✅ DO: Test observable behavior
def test_actor_good():
    actor = MLSignalActor(config)
    actor.on_bar(test_bar)
    
    stats = actor.get_statistics()
    assert stats['bars_processed'] == 1
    assert actor.has_generated_signal()
```

### 2. Over-Mocking

```python
# ❌ DON'T: Mock everything
def test_overmocked():
    mock_model = Mock(return_value=[0.7])
    mock_features = Mock(return_value=np.array([1, 2, 3]))
    mock_scaler = Mock(return_value=np.array([0.1, 0.2, 0.3]))
    
    # Test becomes meaningless - no real logic tested

# ✅ DO: Mock only external dependencies
def test_selective_mocking():
    # Use real model and features, mock only I/O
    with patch('ml.data.external_api.fetch_data') as mock_fetch:
        mock_fetch.return_value = test_data
        
        actor = MLSignalActor(real_config)
        result = actor.process_data()
        # Real logic is tested
```

### 3. Non-Deterministic Tests

```python
# ❌ DON'T: Use random data without seeding
def test_non_deterministic():
    data = np.random.randn(100, 10)  # Different every run
    result = process_data(data)
    assert result.shape == (100,)  # May fail randomly

# ✅ DO: Use seeded random data or fixed test data
def test_deterministic():
    np.random.seed(42)
    data = np.random.randn(100, 10)  # Same every run
    result = process_data(data)
    assert result.shape == (100,)
    # Or better yet:
    data = create_fixed_test_data()
```

### 4. Insufficient Error Testing

```python
# ❌ DON'T: Only test happy path
def test_incomplete():
    result = calculate_features(good_data)
    assert result is not None

# ✅ DO: Test error conditions
def test_comprehensive():
    # Test happy path
    result = calculate_features(good_data)
    assert result.shape == expected_shape
    
    # Test error conditions
    with pytest.raises(ValueError):
        calculate_features(empty_data)
    
    with pytest.raises(TypeError):
        calculate_features(wrong_type_data)
    
    # Test edge cases
    result = calculate_features(minimal_data)
    assert result is not None
```

### 5. Ignoring Performance in Tests

```python
# ❌ DON'T: Ignore performance requirements
def test_ignores_performance():
    # This test might be slow but doesn't verify requirements
    result = expensive_calculation(large_data)
    assert result is not None

# ✅ DO: Include performance assertions
def test_includes_performance():
    import time
    
    start = time.time()
    result = hot_path_calculation(test_data)
    duration = time.time() - start
    
    assert result is not None
    assert duration < 0.005  # <5ms requirement for hot path
```

### 6. Weak Assertions

```python
# ❌ DON'T: Use weak assertions
def test_weak_assertions():
    result = calculate_features(data)
    assert result is not None  # Too weak
    assert len(result) > 0  # Still weak

# ✅ DO: Use specific assertions
def test_strong_assertions():
    result = calculate_features(data)
    
    # Specific shape and type
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(data), expected_features)
    assert result.dtype == np.float32
    
    # Value ranges
    assert np.all(result >= -5.0)  # Reasonable bounds
    assert np.all(result <= 5.0)
    assert not np.any(np.isnan(result))  # No NaN values
```

---

## Conclusion

This guide establishes comprehensive testing standards for the Nautilus Trader ML module. By following these practices, you ensure:

- **Reliability**: Tests catch regressions and validate correctness
- **Maintainability**: Tests evolve with code changes
- **Performance**: Hot path requirements are validated
- **Coverage**: High test coverage with meaningful tests
- **Integration**: Seamless CI/CD integration

Remember: Good tests are not just about coverage numbers—they're about building confidence in your ML systems and catching issues before they reach production.

For specific examples, refer to the existing test files mentioned throughout this guide. When in doubt, prioritize testing behavior over implementation and always include both happy path and error condition tests.