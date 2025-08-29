# ML Tests Context Documentation

**Version**: 4.1  
**Last Updated**: 2024-01-10  
**Status**: Optimized test infrastructure with markers, consolidation, and verification

## ⚠️ Important Database Requirement

**The ML system requires PostgreSQL.** The SQL migrations use PostgreSQL-specific features (partitioning, PL/pgSQL functions, triggers) that are incompatible with SQLite. This is a fundamental architectural requirement.

## Executive Summary

This document summarizes the ML testing architecture and conventions. The test infrastructure has been consolidated into a single, comprehensive `conftest.py` that uses the `EngineManager` singleton for proper connection pooling. Multiple testing approaches (property-based, metamorphic, contract, and pairwise) ensure thorough coverage while minimizing test count.

### Current State (January 2024)

- **Infrastructure**: Consolidated conftest.py with session-scoped fixtures and proper cleanup
- **Connection Management**: EngineManager prevents pool exhaustion (2 connections + 3 overflow)
- **Test Organization**: Well-structured directories with pytest markers for categorization
- **Test Optimization**: 57% reduction in PostgreSQL tests through consolidation
- **Testing Approaches**: Property-based (Hypothesis), metamorphic, contract (Pandera), pairwise
- **Pass Rate**: Significantly improved from 45-50% to >95% after infrastructure fixes
- **Type Safety**: Zero mypy errors after comprehensive type annotation fixes
- **Test Markers**: All 131 test files properly marked for optimal execution strategies

## Test Infrastructure

### Consolidated Configuration (`conftest.py`)

The test configuration has been unified into a single `conftest.py` that provides:

```python
# Session-scoped database engine (prevents connection exhaustion)
@pytest.fixture(scope="session")
def database_engine() -> Generator[Engine, None, None]:
    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,  # Conservative for tests
        max_overflow=3,  # Limited overflow
        pool_pre_ping=True,  # Test connections
        pool_recycle=300,  # 5-minute recycle
    )
    yield engine
    EngineManager.dispose_all()

# Transaction-isolated test sessions
@pytest.fixture
def database_session(database_session_factory):
    """Isolated session with automatic rollback"""
    # Uses nested transactions for complete isolation
```

### Hypothesis Profiles

Three testing profiles for different environments:

- **CI Profile**: Fast (50 examples, 5s deadline, deterministic)
- **Dev Profile**: Thorough (200 examples, no deadline)
- **Debug Profile**: Minimal (10 examples, verbose output)

## Testing Approaches

### 1. Property-Based Testing (`property/`)

Using Hypothesis to verify invariants:

```python
@given(
    instrument_id=instrument_ids(),
    features=feature_values(),
    ts_events=st.lists(nanosecond_timestamps(), min_size=1, unique=True)
)
def test_timestamp_monotonicity_invariant(self, ...):
    """Timestamps must always increase monotonically"""
```

Key invariants tested:
- Timestamp monotonicity
- Feature immutability after write
- Partition consistency
- Data integrity across operations

### 2. Metamorphic Testing (`metamorphic/`)

Testing relationships under controlled transformations:

```python
def test_price_scaling_invariance(self):
    """Returns should be unchanged when prices are scaled"""
    scaled_features = engineer.compute_features(scaled_bars)
    np.testing.assert_allclose(
        original_features['returns'],
        scaled_features['returns']
    )
```

Metamorphic relations tested:
- Price scaling invariance
- Time reversal properties
- Noise addition robustness

### 3. Contract Testing (`contracts/`)

Using Pandera for schema validation:

```python
class FeatureInputSchema(pa.DataFrameModel):
    instrument_id: Series[str] = pa.Field()
    ts_event: Series[int] = pa.Field(ge=0)
    ts_init: Series[int] = pa.Field(ge=0)
    feature_values: Series[object] = pa.Field()
    
    @pa.check("ts_event")
    def ts_event_monotonic(cls, series):
        return series.is_monotonic_increasing
```

### 4. Pairwise Testing (`combinatorial/`)

Reducing combinatorial explosion with AllPairs:

```python
# 8,748 possible combinations → 15 test cases (99.8% reduction)
pairwise_configs = list(AllPairs([
    return_periods, momentum_periods, volume_periods,
    volatility_windows, use_log_returns, detrend_returns
]))
```

## Test Organization

```
ml/tests/
├── conftest.py              # Consolidated configuration
├── test_smoke.py           # Quick validation tests
├── property/               # Property-based tests
├── metamorphic/           # Metamorphic relation tests
├── contracts/             # Schema contract tests
├── combinatorial/         # Pairwise combination tests
├── unit/                  # Unit tests by domain
│   ├── actors/
│   ├── stores/
│   ├── features/
│   └── strategies/
├── integration/           # Integration tests
│   └── conftest.py       # Integration-specific fixtures
├── e2e/                  # End-to-end tests
├── performance/          # Performance benchmarks
└── tools/                # Test utilities and analysis

```

## Running Tests

### Quick Validation
```bash
# Smoke tests - verify basic functionality
python -m pytest ml/tests/test_smoke.py -xvs

# Unit tests only (fast, mocked)
python -m pytest ml/tests/unit -x --tb=short
```

### Property-Based Tests
```bash
# Run with CI profile (fast)
HYPOTHESIS_PROFILE=ci python -m pytest ml/tests/property -x

# Run with dev profile (thorough)
HYPOTHESIS_PROFILE=dev python -m pytest ml/tests/property
```

### Integration Tests
```bash
# Fast integration tests
python -m pytest ml/tests/integration -x -m "not slow"

# Full integration suite
python -m pytest ml/tests/integration
```

### Performance Tests
```bash
# Benchmark hot path operations
python -m pytest ml/tests/performance/test_ml_hot_path_benchmarks.py --benchmark-only
```

## Connection Management

### EngineManager Pattern

All database connections go through the EngineManager singleton:

```python
from ml.core.db_engine import EngineManager

# Get or create engine (reuses existing)
engine = EngineManager.get_engine(connection_string)

# Dispose specific engine
EngineManager.dispose_engine(connection_string)

# Dispose all engines (cleanup)
EngineManager.dispose_all()
```

### Preventing Connection Exhaustion

1. **Session-scoped fixtures**: Single engine for entire test session
2. **Conservative pooling**: 2 base + 3 overflow connections
3. **Automatic cleanup**: `cleanup_engines` fixture disposes after each test
4. **Transaction isolation**: Tests use nested transactions with rollback
5. **Connection monitoring**: `connection_monitor` fixture detects leaks

## Common Patterns

### Mock Stores for Unit Tests

```python
@pytest.fixture
def mock_feature_store():
    mock_store = MagicMock()
    mock_store.write_features = MagicMock(return_value=True)
    mock_store.get_latest_features = MagicMock(return_value={})
    return mock_store
```

### Isolated SQLite for Hypothesis

```python
@pytest.fixture
def hypothesis_database_session():
    """In-memory SQLite for rapid property test generation"""
    engine = create_engine("sqlite:///:memory:", poolclass=NullPool)
    # ... setup and teardown
```

### Test Data Factories

```python
from ml.tests.fixtures.model_factory import create_test_model
from ml.tests.fixtures.mock_services import create_mock_fred_client

model = create_test_model("xgboost")
fred_client = create_mock_fred_client(test_data)
```

## Debugging Failed Tests

### Connection Issues
```bash
# Monitor PostgreSQL connections
watch -n1 "psql -c 'SELECT count(*) FROM pg_stat_activity;'"

# Check EngineManager pool status
python -c "from ml.core.db_engine import EngineManager; print(EngineManager.get_pool_status('...'))"
```

### Hypothesis Failures
```python
# Use debug profile for verbose output
HYPOTHESIS_PROFILE=debug python -m pytest failing_test.py -xvs

# Reproduce with seed
python -m pytest --hypothesis-seed=12345
```

### Performance Issues
```bash
# Profile test execution
python -m pytest --profile test_slow.py

# Benchmark specific operations
python -m pytest test_file.py::test_function --benchmark-only
```

## Best Practices

1. **Use appropriate fixtures**: Mock stores for unit tests, real stores for integration
2. **Leverage property testing**: Find edge cases automatically with Hypothesis
3. **Test contracts**: Validate data shapes with Pandera schemas
4. **Reduce combinations**: Use pairwise testing for configuration spaces
5. **Monitor connections**: Use connection_monitor for database-heavy tests
6. **Clean up properly**: Ensure all resources are released in teardown

## Known Issues and Workarounds

### PostgreSQL Required
- SQLite is not supported due to PostgreSQL-specific features
- Use Docker for local development if PostgreSQL not installed

### Parallel Test Execution
- Limited to CPU/2 workers to prevent database overwhelm
- Use `-n auto` flag with pytest-xdist for parallel execution

### Memory Usage
- Hypothesis tests can consume significant memory
- Use smaller max_examples in CI environments

## Recent Improvements (January 2024)

### Test Marker Implementation
- Applied pytest markers to all 131 test files
- Database tests marked with `@pytest.mark.serial` to prevent connection exhaustion
- Parallel-safe tests marked for concurrent execution
- Created verification scripts to ensure marker compliance

### PostgreSQL Test Consolidation
- Merged 3 redundant PostgreSQL test files into 1 parameterized file
- Achieved 57% line reduction (223 → 96 lines)
- Preserved all test scenarios while eliminating duplication
- Fixed broken fixtures that were causing test failures

### Infrastructure Consolidation
- Unified multiple conftest files into single source of truth
- Implemented session-scoped database fixtures
- Added automatic cleanup to prevent connection leaks
- Integrated Hypothesis profiles for different test environments

## Future Improvements

1. **Test Coverage**: Increase from current ~80% to >90%
2. **Mutation Testing**: Add mutmut for test effectiveness validation
3. **Fuzz Testing**: Extend property tests with fuzzing strategies
4. **Performance Regression**: Automated benchmark comparisons
5. **Test Impact Analysis**: Run only affected tests on code changes