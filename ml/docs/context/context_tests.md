# ML Tests Context Documentation

## Executive Summary

The Nautilus Trader ML testing infrastructure implements a comprehensive multi-layered testing strategy that ensures robust, maintainable, and meaningful test coverage across the entire ML module. The testing framework leverages 130+ test files organized into distinct categories (unit, integration, contracts, property-based, and benchmarks) with sophisticated fixtures, utilities, and protocols that enforce behavioral contracts while maintaining implementation flexibility.

## Core Testing Architecture

### Test Organization Structure

The ML testing infrastructure is organized into a hierarchical structure that reflects both the module architecture and testing patterns:

```
ml/tests/
├── unit/                    # Component-level isolation tests
│   ├── actors/             # ML actor unit tests
│   ├── data/               # Data processing tests
│   ├── features/           # Feature engineering tests
│   ├── strategies/         # Strategy component tests
│   └── training/           # Model training tests
├── integration/            # Component interaction tests
│   ├── test_e2e_*.py      # End-to-end workflows
│   ├── test_*_demo.py     # Demonstration scenarios
│   └── test_utils.py      # Integration test utilities
├── contracts/              # Behavioral contract tests
│   ├── test_actor_contracts.py
│   ├── test_strategy_contracts.py
│   └── test_registry_behavioral.py
├── property/               # Property-based tests with Hypothesis
├── benchmarks/             # Performance and latency tests
├── fixtures/               # Test model and data factories
└── data/                   # Test data storage
```

### Testing Philosophy and Principles

The ML testing framework is built on several key principles documented in `TESTING_PROTOCOL.md`:

1. **Test Behavior, Not Implementation**: Tests focus on observable outcomes and public contracts rather than internal state
2. **Use Real Components**: Minimal but real models for integration tests instead of excessive mocking
3. **Full Stack Coverage**: Unit tests for components, integration for interactions, E2E for workflows
4. **Contract-Based Testing**: Define what implementations MUST do, not HOW they do it
5. **Property-Based Testing**: Use Hypothesis to verify invariants across random inputs

## Test Categories and Patterns

### Contract Tests (`contracts/`)

Contract tests define behavioral requirements that all implementations must satisfy:

```python
class TestActorContracts:
    """Behavioral contracts all ML actors must satisfy."""
    
    def test_actor_publishes_ml_signal_on_bar(self):
        """Actor MUST publish MLSignal when receiving bar data."""
        # Verifies signal structure and required fields
        
    def test_actor_includes_model_id_in_signal(self):
        """Every signal MUST identify its source model."""
        # Ensures traceability and model attribution
```

**Key Patterns**:
- Implementation-agnostic tests
- Focus on invariants and guarantees
- Public interface validation only
- Property-based testing with Hypothesis

### Unit Tests (`unit/`)

Unit tests verify individual components in isolation with minimal dependencies:

```python
class TestFeatureEngineer:
    """Unit tests for FeatureEngineer component."""
    
    def test_calculate_features_with_valid_data(self):
        """Test feature calculation with normal inputs."""
        
    def test_calculate_features_with_missing_data(self):
        """Test handling of missing values."""
        
    def test_calculate_features_with_extreme_values(self):
        """Test handling of outliers and edge cases."""
```

**Requirements**:
- Fast execution (<100ms per test)
- Mock external services
- Test edge cases and error conditions
- Minimal but complete test data

### Integration Tests (`integration/`)

Integration tests verify component interactions and data flow through the system:

```python
class TestMLPipeline:
    """Integration tests for ML pipeline."""
    
    def test_end_to_end_signal_generation(self):
        """Test complete flow from bars to signals."""
        
    def test_feature_parity_training_inference(self):
        """Verify features match between training and inference."""
```

**Key Features**:
- Use real components (not mocks)
- Test realistic workflows
- Verify data consistency across components
- May be slower than unit tests

### Property-Based Tests (`property/`)

Property-based tests use Hypothesis to verify invariants across random inputs:

```python
@given(
    n_bars=st.integers(min_value=1, max_value=1000),
    bar_values=st.floats(min_value=0.01, max_value=10000)
)
def test_actor_preserves_temporal_order(self, any_actor, n_bars, bar_values):
    """Property: Actor must process bars in order and maintain causality."""
    # Verifies temporal ordering invariant
```

**Coverage Areas**:
- Feature contract validation
- Fractional differencing properties
- Temporal ordering guarantees
- Numerical stability properties

### End-to-End Tests

E2E tests validate complete workflows from data ingestion to signal generation:

- `test_data_registry_e2e.py`: Full data registry lifecycle
- `test_e2e_signal_actor_featurestore.py`: Signal generation with feature storage
- `test_strategy_store_e2e.py`: Strategy execution and persistence

## Test Fixtures and Utilities

### Core Fixtures (`conftest.py`)

The testing framework provides comprehensive fixtures at multiple levels:

```python
# Root-level fixtures (ml/tests/conftest.py)
@pytest.fixture
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    
@pytest.fixture
def model_registry_dir() -> Path:
    """Provide path to test model registry."""

# Integration-level fixtures (ml/tests/integration/conftest.py)
@pytest.fixture
def test_instrument() -> CurrencyPair:
    """Provide test EURUSD instrument."""
    
@pytest.fixture
def generate_test_bars() -> list[Bar]:
    """Generate realistic test Bar objects."""
    
@pytest.fixture
def mock_parquet_catalog() -> ParquetDataCatalog:
    """Create mock ParquetDataCatalog with test data."""
```

### Test Utilities (`integration/test_utils.py`)

Comprehensive utilities for test data generation and validation:

```python
def generate_realistic_ohlcv(
    instrument_id: InstrumentId,
    n_bars: int = 1000,
    volatility: float = 0.02
) -> list[Bar]:
    """Generate realistic OHLCV data with specified characteristics."""

def create_correlated_multi_instrument_data(
    instruments: dict[str, float],
    correlation_matrix: np.ndarray | None = None
) -> dict[InstrumentId, list[Bar]]:
    """Create correlated bar data for multiple instruments."""

def validate_feature_parity(
    batch_features: np.ndarray,
    online_features: np.ndarray,
    tolerance: float = 1e-10
) -> tuple[bool, dict]:
    """Validate batch and online feature calculations match."""
```

### Model Factory (`fixtures/model_factory.py`)

Factory for creating minimal but valid test models:

```python
class TestModelFactory:
    """Factory for creating minimal but valid test models."""
    
    @staticmethod
    def create_minimal_xgboost_model(
        n_features: int = 10,
        model_type: str = "classification"
    ) -> Path:
        """Create minimal valid XGBoost model for testing."""
        
    @staticmethod
    def create_onnx_model(
        n_features: int = 10,
        n_outputs: int = 1
    ) -> Path:
        """Create minimal ONNX model for testing."""
        
    @staticmethod
    def validate_model(model_path: Path) -> dict:
        """Validate that a test model is properly formed."""
```

**Key Features**:
- Minimal models for fast tests
- Production-safe formats (no pickle)
- Proper metadata inclusion
- Support for XGBoost, LightGBM, ONNX, sklearn

## Test Data Management

### Test Data Directory Structure

```
ml/tests/data/
├── model_registry/         # Test model storage
│   └── models/
│       ├── xgb_v1.json
│       └── xgb_v2.json
├── model_registry_rollout/ # Rollout testing models
│   └── models/
│       ├── prod.onnx
│       └── new.onnx
└── __init__.py             # Data directory utilities
```

### Data Generation Patterns

The testing framework provides multiple patterns for test data generation:

1. **Realistic Market Data**: Generate OHLCV bars with realistic price movements
2. **Correlated Instruments**: Create multi-instrument data with correlation matrices
3. **ML Signals**: Generate mock ML signals correlated with bar data
4. **Feature Data**: Create synthetic feature matrices for model training

## Testing Protocols and Standards

### Coverage Requirements

- **General Python Code**: ≥80% coverage
- **ML Modules**: ≥90% coverage
- **Hot Path Code**: 100% coverage with performance benchmarks

### Test Naming Conventions

```python
# Unit tests
test_{function}_when_{condition}_returns_{expected}
test_{component}_handles_{scenario}

# Integration tests
test_end_to_end_{workflow}
test_{component1}_integrates_with_{component2}

# Contract tests
test_{component}_must_{requirement}
test_{component}_preserves_{invariant}

# Property tests
test_{property}_holds_for_{inputs}
```

### Performance Testing Requirements

- Hot path functions: P99 latency < 5ms
- Feature computation: < 1ms per feature
- Model inference: < 10ms for ONNX models
- Batch operations: Linear scaling with data size

## Running Tests

### Basic Test Execution

```bash
# Run all ML tests
pytest ml/tests -v

# Run specific test category
pytest ml/tests/unit -v
pytest ml/tests/integration -v
pytest ml/tests/contracts -v

# Run with coverage reporting
pytest ml/tests --cov=ml --cov-report=html

# Run property-based tests with more examples
pytest ml/tests/property --hypothesis-profile=dev
```

### Test Profiles

```bash
# Fast tests only (unit tests)
pytest ml/tests/unit -m "not slow"

# Integration tests with real data
pytest ml/tests/integration --integration

# Contract validation
pytest ml/tests/contracts --strict

# Performance benchmarks
pytest ml/tests/benchmarks --benchmark-only
```

### CI/CD Integration

Tests are integrated into the CI/CD pipeline with:

1. **Pre-commit Hooks**: Linting, formatting, type checking
2. **GitHub Actions**: Full test suite on PR
3. **Coverage Gates**: Enforce minimum coverage thresholds
4. **Performance Regression**: Detect latency increases

## Test Development Guidelines

### Writing New Tests

1. **Choose the Right Category**:
   - Unit: Single component, fast, isolated
   - Integration: Multiple components, data flow
   - Contract: Behavioral requirement, invariant
   - Property: Random inputs, mathematical properties

2. **Use Appropriate Fixtures**:
   - Leverage existing fixtures before creating new ones
   - Create minimal but realistic test data
   - Use factory patterns for complex objects

3. **Follow Testing Patterns**:
   - Arrange-Act-Assert for unit tests
   - Given-When-Then for behavioral tests
   - Property assertions for invariants

### Common Test Patterns

```python
# Pattern 1: Testing with real components
def test_feature_computation_with_real_data(self):
    # Use TestModelFactory for minimal models
    model = TestModelFactory.create_minimal_xgboost_model()
    
    # Use generate_realistic_ohlcv for test data
    bars = generate_realistic_ohlcv(instrument_id, n_bars=100)
    
    # Test actual behavior
    features = engineer.compute_features(bars)
    assert features.shape == (100, expected_features)

# Pattern 2: Contract validation
def test_store_must_persist_all_signals(self):
    # Define the contract
    signals = generate_test_signals(n=100)
    
    # Execute
    store.write_signals(signals)
    
    # Verify contract
    retrieved = store.read_signals()
    assert len(retrieved) == len(signals)
    assert all(s.ts_event for s in retrieved)

# Pattern 3: Property-based testing
@given(
    n_features=st.integers(min_value=1, max_value=100),
    n_samples=st.integers(min_value=10, max_value=1000)
)
def test_feature_scaling_preserves_order(n_features, n_samples):
    # Property: scaling preserves relative ordering
    data = np.random.randn(n_samples, n_features)
    scaled = scaler.fit_transform(data)
    
    # Verify property holds
    for i in range(n_features):
        original_order = np.argsort(data[:, i])
        scaled_order = np.argsort(scaled[:, i])
        assert np.array_equal(original_order, scaled_order)
```

## Test Maintenance

### Regular Maintenance Tasks

1. **Update Test Data**: Refresh test data quarterly
2. **Review Mocks**: Ensure mocks reflect current interfaces
3. **Performance Baselines**: Update performance benchmarks
4. **Coverage Analysis**: Review uncovered code paths
5. **Flaky Test Investigation**: Fix non-deterministic tests

### Test Documentation

Each test file should include:
- Module docstring explaining test purpose
- Class docstrings for test groups
- Method docstrings using Given-When-Then format
- Comments for complex test logic

## Advanced Testing Features

### Hypothesis Strategies

Custom strategies for ML-specific data generation:

```python
# Custom strategy for valid feature names
valid_feature_names = st.text(
    alphabet=string.ascii_lowercase + "_",
    min_size=3,
    max_size=20
)

# Strategy for correlation matrices
@st.composite
def correlation_matrices(draw, n):
    # Generate valid correlation matrix
    ...
```

### Test Markers

```python
# Mark slow tests
@pytest.mark.slow
def test_large_dataset_processing():
    pass

# Mark tests requiring external services
@pytest.mark.external
def test_databento_integration():
    pass

# Mark performance benchmarks
@pytest.mark.benchmark
def test_inference_latency():
    pass
```

## Best Practices

1. **Test Independence**: Each test should be independent and idempotent
2. **Clear Assertions**: Use descriptive assertion messages
3. **Minimal Test Data**: Use the smallest data that exercises the behavior
4. **Avoid Over-Mocking**: Mock only external dependencies
5. **Test Public APIs**: Focus on public interfaces, not implementation
6. **Document Intent**: Clear test names and docstrings
7. **Performance Awareness**: Monitor test suite execution time
8. **Regular Cleanup**: Remove obsolete tests and update fixtures

## Summary

The ML testing infrastructure provides comprehensive coverage through multiple testing layers, sophisticated fixtures and utilities, and clear protocols for test development and maintenance. The framework ensures robust validation of ML components while maintaining flexibility for implementation changes and supporting rapid development cycles.