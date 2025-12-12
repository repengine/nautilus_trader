# Test Design Report: Phase 1.1 Wire FeatureCalculator to Facade

**Design Date:** 2025-12-01T12:00:00Z
**Designer:** Test Design Agent (Phase 1)
**Task Reference:** Task 1.1: Wire FeatureCalculator to Facade

## Test Strategy Overview

This test design ensures the successful wiring of `FeatureCalculator` component to the `FeatureEngineer` facade. The core challenge is proving that after wiring, the facade methods (`calculate_features`, `calculate_features_batch`, `calculate_features_online`, `compute_features`) delegate to `self.calculator` instead of `self._legacy_impl` while maintaining **perfect numerical parity**.

The testing strategy employs three key approaches:

1. **Parity Tests (CRITICAL)**: Verify facade produces IDENTICAL results to legacy implementation using `np.testing.assert_allclose(rtol=1e-10)`. These are the most important tests - they prove the wiring is correct by ensuring no behavioral change.

2. **Component Delegation Tests (Category 14)**: Verify that after wiring, the facade's public methods actually invoke `self.calculator` methods rather than `self._legacy_impl`. These use mocking to verify the call path.

3. **Performance Tests (HOT PATH)**: Verify `calculate_features_online` maintains P99 < 5ms latency requirement. The wiring must not introduce performance regression.

Additionally, we test feature flag behavior (ML_USE_LEGACY_FEATURE_ENGINEER), backward compatibility, and error handling to ensure the refactoring is complete and safe.

## Test Files Created/Modified

### Parity Tests (CRITICAL)
- `ml/tests/facades/test_feature_engineer_calculator_wiring_parity.py`
  - Test cases:
    - `test_facade_calculate_features_batch_matches_legacy`
    - `test_facade_calculate_features_online_matches_legacy`
    - `test_facade_compute_features_matches_legacy`
    - `test_facade_calculate_features_unified_batch_matches_legacy`
    - `test_facade_calculate_features_unified_online_matches_legacy`
    - `test_parity_with_scaler_fitting`
    - `test_parity_across_config_variations`
    - `test_parity_with_100_bars`
  - Coverage target: 100% for delegation paths
  - Key assertions: `np.testing.assert_allclose(legacy_result, facade_result, rtol=1e-10)`

### Component Delegation Tests (Category 14)
- `ml/tests/facades/test_feature_calculator_delegation.py`
  - Test cases:
    - `test_calculate_features_batch_delegates_to_calculator`
    - `test_calculate_features_online_delegates_to_calculator`
    - `test_compute_features_delegates_to_calculator`
    - `test_calculate_features_unified_delegates_to_calculator`
    - `test_legacy_impl_not_called_after_wiring`
  - Coverage target: 100% for delegation verification
  - Key assertions: Mock call verification, _legacy_impl NOT called

### HOT PATH Performance Tests
- `ml/tests/facades/test_feature_engineer_facade_performance.py`
  - Test cases:
    - `test_calculate_features_online_p99_under_5ms`
    - `test_facade_overhead_under_10_percent_vs_calculator`
    - `test_feature_buffer_reused_zero_allocations`
  - Coverage target: HOT PATH validation
  - Key assertions: P99 < 5ms, overhead < 10%, memory allocation < 1000 bytes/call

### Feature Flag Tests
- `ml/tests/facades/test_feature_engineer_feature_flags.py`
  - Test cases:
    - `test_legacy_flag_true_uses_legacy_path`
    - `test_legacy_flag_false_uses_component_path`
    - `test_both_paths_produce_identical_results`
  - Coverage target: 100% for feature flag logic
  - Key assertions: Correct delegation based on env var

### Error Handling Tests
- `ml/tests/facades/test_feature_engineer_error_handling.py`
  - Test cases:
    - `test_calculate_features_online_without_indicator_manager_raises`
    - `test_calculate_features_with_invalid_mode_raises`
    - `test_calculate_features_batch_with_empty_dataframe`
    - `test_compute_features_with_empty_bars_list`
  - Coverage target: 100% for error conditions
  - Key assertions: Correct exception types and messages

### Integration Tests
- `ml/tests/integration/features/test_feature_calculator_facade_integration.py`
  - Test cases:
    - `test_end_to_end_batch_workflow`
    - `test_end_to_end_online_workflow`
    - `test_scaler_fitted_in_batch_applied_in_online`
  - Coverage target: 90% for integration paths
  - Key assertions: Complete workflow success

## Detailed Test Cases

### Happy Path Tests

#### test_facade_calculate_features_batch_matches_legacy
**Purpose:** Verify `facade.calculate_features_batch()` produces identical output to legacy
**Input:** DataFrame with 100 bars of OHLCV data
**Expected Behavior:** Feature DataFrame and scaler match legacy implementation exactly
**Assertions:**
- Shape parity: `assert legacy_df.shape == facade_df.shape`
- Value parity: `np.testing.assert_allclose(legacy_df.to_numpy(), facade_df.to_numpy(), rtol=1e-10)`
- Scaler parity: If scaler fitted, mean/std match
**Fixtures Used:** `sample_ohlcv_dataframe`, `feature_config`

#### test_facade_calculate_features_online_matches_legacy
**Purpose:** Verify `facade.calculate_features_online()` produces identical output to legacy
**Input:** Current bar dict with pre-warmed IndicatorManager (50 bars history)
**Expected Behavior:** Feature array matches legacy implementation exactly
**Assertions:**
- Shape parity: `assert legacy_features.shape == facade_features.shape`
- Dtype parity: `assert legacy_features.dtype == facade_features.dtype == np.float32`
- Value parity: `np.testing.assert_allclose(legacy_features, facade_features, rtol=1e-10)`
**Fixtures Used:** `current_bar_dict`, `indicator_manager_with_history`, `feature_config`

#### test_facade_compute_features_matches_legacy
**Purpose:** Verify `facade.compute_features()` (legacy shim) produces identical output
**Input:** List of 50 Bar objects
**Expected Behavior:** Dict[str, float] with same keys and values as legacy
**Assertions:**
- Key parity: `assert set(legacy_result.keys()) == set(facade_result.keys())`
- Value parity: For each key, `np.allclose(legacy_result[k], facade_result[k], rtol=1e-10)`
**Fixtures Used:** `test_data_factory` (for generating bars), `feature_config`

#### test_calculate_features_batch_delegates_to_calculator
**Purpose:** Verify facade.calculate_features_batch() calls self.calculator, NOT self._legacy_impl
**Input:** Mock calculator, real facade
**Expected Behavior:** Calculator's _calculate_features_batch called, legacy_impl NOT called
**Assertions:**
- `mock_calculator._calculate_features_batch.assert_called_once()`
- `mock_legacy_impl.calculate_features_batch.assert_not_called()`
**Fixtures Used:** `feature_config`, `sample_ohlcv_dataframe`

#### test_calculate_features_online_p99_under_5ms
**Purpose:** Verify HOT PATH performance requirement (P99 < 5ms)
**Input:** Pre-warmed indicator manager, 1000 iterations
**Expected Behavior:** P99 latency < 5000 microseconds
**Assertions:**
- `assert p99_latency_us < 5000`
- `assert bytes_per_call < 1000`
**Fixtures Used:** `feature_config`, `prepared_indicator_manager`, `sample_bar_dict`

### Error Condition Tests

#### test_calculate_features_online_without_indicator_manager_raises
**Purpose:** Verify ValueError when indicator_manager is None in online mode
**Input:** Current bar dict, mode="online", indicator_manager=None
**Expected Behavior:** Raises ValueError with clear message
**Assertions:**
- `pytest.raises(ValueError, match="indicator_manager is required")`
**Fixtures Used:** `feature_config`, `current_bar_dict`

#### test_calculate_features_with_invalid_mode_raises
**Purpose:** Verify ValueError when mode is not "batch" or "online"
**Input:** DataFrame, mode="invalid"
**Expected Behavior:** Raises ValueError with valid modes listed
**Assertions:**
- `pytest.raises(ValueError, match="Invalid mode.*batch.*online")`
**Fixtures Used:** `feature_config`, `sample_ohlcv_dataframe`

#### test_compute_features_with_empty_bars_list
**Purpose:** Verify correct error handling for empty input
**Input:** Empty list `[]`
**Expected Behavior:** Raises ValueError
**Assertions:**
- `pytest.raises(ValueError, match="empty|No bars")`
**Fixtures Used:** `feature_config`

### Edge Case Tests

#### test_parity_with_single_bar
**Purpose:** Verify parity at minimum viable input
**Input:** List with 1 Bar object
**Expected Behavior:** Both implementations handle gracefully, results match
**Assertions:**
- No exceptions raised
- Results match within tolerance
**Fixtures Used:** `test_data_factory`, `feature_config`

#### test_parity_with_extreme_prices
**Purpose:** Verify numerical stability at extreme values
**Input:** Bars with prices near float limits, very small volumes
**Expected Behavior:** No NaN/Inf in output, results match
**Assertions:**
- `assert not np.isnan(features).any()`
- `assert not np.isinf(features).any()`
- Value parity maintained
**Fixtures Used:** Custom extreme data generation

#### test_parity_with_flat_prices
**Purpose:** Verify handling of zero-return edge case
**Input:** Bars where all OHLC are identical
**Expected Behavior:** Zero returns, zero volatility, results match
**Assertions:**
- Return features are 0.0
- Volatility features are 0.0
- Value parity maintained
**Fixtures Used:** Custom flat price data

### Backward Compatibility Tests

#### test_legacy_api_still_works
**Purpose:** Verify old usage patterns remain functional
**Input:** Old-style API calls (compute_features with Bar list)
**Expected Behavior:** Same results as before wiring
**Assertions:**
- Return type unchanged (dict[str, float])
- Feature names unchanged
- Values within tolerance of legacy
**Fixtures Used:** `test_data_factory`, `feature_config`

#### test_scaler_fitting_unchanged
**Purpose:** Verify scaler fitting/application behavior preserved
**Input:** DataFrame, fit_scaler=True, scaler_fit_ratio=0.7
**Expected Behavior:** Scaler fitted on 70% of data, applied to 100%
**Assertions:**
- `scaler.n_samples_seen_ == int(len(df) * 0.7)`
- Output DataFrame scaled correctly
**Fixtures Used:** `sample_ohlcv_dataframe`, `feature_config`

### Feature Flag Parity Tests

#### test_legacy_flag_produces_identical_results
**Purpose:** Verify ML_USE_LEGACY_FEATURE_ENGINEER=1 and =0 produce same output
**Input:** Same data through both paths
**Expected Behavior:** IDENTICAL numerical results
**Assertions:**
- `np.testing.assert_allclose(legacy_result, component_result, rtol=1e-10)`
**Fixtures Used:** `sample_ohlcv_dataframe`, `feature_config`, `monkeypatch`

## Fixtures and Test Data Requirements

### Existing Fixtures to Use

From `ml/tests/fixtures/`:
- `sample_bars_dataframe_factory`: Generate deterministic OHLCV DataFrames
- `sample_bar_series_config_factory`: Configure bar generation
- `test_data_factory`: Session-scoped factory for Bar objects
- `base_feature_config`: Base MLFeatureConfig
- `isolated_prometheus_registry`: Metrics isolation

### Custom Fixtures to Create

```python
@pytest.fixture
def feature_config():
    """Standard FeatureConfig for calculator wiring tests."""
    return FeatureConfig(
        return_periods=[1, 2, 5],
        momentum_periods=[1, 3],
        volume_ma_periods=[10, 20],
        ema_fast=12,
        ema_slow=26,
        rsi_period=14,
        bb_period=20,
        bb_std=2.0,
        atr_period=14,
        enable_returns=True,
        enable_momentum=True,
        enable_volatility=True,
        enable_technical=True,
    )


@pytest.fixture
def sample_ohlcv_dataframe():
    """DataFrame with 100 bars of synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="1min")
    close_prices = 100.0 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.abs(np.random.randn(100) * 0.3)
    low_prices = close_prices - np.abs(np.random.randn(100) * 0.3)
    open_prices = close_prices + np.random.randn(100) * 0.2
    volumes = np.random.uniform(900000, 1100000, 100)
    return pd.DataFrame({
        "timestamp": dates,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })


@pytest.fixture
def indicator_manager_with_history(feature_config):
    """IndicatorManager pre-warmed with 50 bars of history."""
    from ml.features.engineering import IndicatorManager
    manager = IndicatorManager(feature_config)
    for i in range(50):
        manager.update_from_values(
            close=100.0 + i * 0.1,
            high=101.0 + i * 0.1,
            low=99.0 + i * 0.1,
            volume=1000000.0 + i * 1000,
        )
    return manager


@pytest.fixture
def current_bar_dict():
    """Single bar as dict for online mode."""
    return {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000000.0,
    }
```

### Hypothesis Strategies

```python
from hypothesis import strategies as st, given, assume

# Strategy for valid OHLCV bars
valid_prices = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)
valid_volumes = st.floats(min_value=1.0, max_value=1e12, allow_nan=False, allow_infinity=False)

@st.composite
def ohlcv_bar(draw):
    """Generate valid OHLCV bar dict."""
    close = draw(valid_prices)
    high = draw(st.floats(min_value=close, max_value=close * 1.1, allow_nan=False))
    low = draw(st.floats(min_value=close * 0.9, max_value=close, allow_nan=False))
    open_price = draw(st.floats(min_value=low, max_value=high, allow_nan=False))
    volume = draw(valid_volumes)
    return {"open": open_price, "high": high, "low": low, "close": close, "volume": volume}


@st.composite
def ohlcv_dataframe(draw, min_rows=10, max_rows=500):
    """Generate valid OHLCV DataFrame."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    bars = [draw(ohlcv_bar()) for _ in range(n_rows)]
    return pd.DataFrame(bars)
```

### Mock Objects Required

```python
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_calculator():
    """Mock FeatureCalculator for delegation verification."""
    mock = MagicMock()
    mock._calculate_features_batch.return_value = (
        pd.DataFrame({"feature_1": [1.0]}),
        None
    )
    mock._calculate_features_online.return_value = np.array([1.0], dtype=np.float32)
    mock.compute_features.return_value = {"feature_1": 1.0}
    mock.n_features = 10
    mock.feature_buffer = np.zeros(20, dtype=np.float32)
    return mock


@pytest.fixture
def mock_legacy_impl():
    """Mock legacy FeatureEngineer for verifying it's NOT called."""
    mock = MagicMock()
    mock.calculate_features_batch.return_value = (
        pd.DataFrame({"feature_1": [2.0]}),  # Different value to detect wrong path
        None
    )
    return mock
```

## Coverage Expectations

**Target Coverage:** 95% for facade delegation paths, 90% for ML modules overall

**Critical Paths Covered:**
1. `FeatureEngineer.calculate_features()` - unified entry point
2. `FeatureEngineer.calculate_features_batch()` - batch mode delegation
3. `FeatureEngineer.calculate_features_online()` - HOT PATH delegation
4. `FeatureEngineer.compute_features()` - legacy compatibility shim
5. Feature flag toggle paths (ML_USE_LEGACY_FEATURE_ENGINEER)

**Known Coverage Gaps:**
- Internal FeatureCalculator methods (already tested in `test_feature_calculator.py`)
- Store persistence (not relevant to this wiring task)
- Registry operations (not relevant to this wiring task)

**Hot Path Performance:**
- `calculate_features_online`: P99 < 5ms (5000 microseconds)
- Facade overhead vs calculator direct: < 10%
- Memory allocation per call: < 1000 bytes

## Test Execution Plan

### Test Order
1. Parity tests (CRITICAL - prove correctness)
2. Delegation tests (verify wiring)
3. Error handling tests (verify robustness)
4. Performance tests (verify HOT PATH)
5. Feature flag tests (verify toggle behavior)
6. Integration tests (end-to-end)

### Pytest Markers
- `@pytest.mark.parity`: Parity verification tests
- `@pytest.mark.unit`: Fast, isolated tests
- `@pytest.mark.performance`: Performance benchmarks
- `@pytest.mark.slow`: Takes >1 second
- `@pytest.mark.skip(reason="Test design - implementation pending")`: Initially failing

### Commands to Run
```bash
# Run all tests for this task
pytest ml/tests/facades/test_feature_engineer_calculator_wiring_parity.py \
       ml/tests/facades/test_feature_calculator_delegation.py \
       ml/tests/facades/test_feature_engineer_facade_performance.py \
       ml/tests/facades/test_feature_engineer_feature_flags.py \
       ml/tests/facades/test_feature_engineer_error_handling.py -v

# Run with coverage
pytest ml/tests/facades/test_feature_engineer_*.py \
       --cov=ml.features.facade \
       --cov=ml.features.common.feature_calculator \
       --cov-report=term-missing

# Run parity tests only (CRITICAL)
pytest ml/tests/facades/test_feature_engineer_calculator_wiring_parity.py -v

# Run performance tests
pytest ml/tests/facades/test_feature_engineer_facade_performance.py -v --benchmark-only
```

## Handoff Notes for Implementation Agent

### Contract to Satisfy

The implementation MUST modify `ml/features/facade.py` to delegate these methods to `self.calculator`:

1. **`calculate_features(data, mode, ...)`** (lines 490-557)
   - When `mode="batch"`: Call `self.calculator._calculate_features_batch(...)`
   - When `mode="online"`: Call `self.calculator._calculate_features_online(...)`
   - Current: `self._legacy_impl.calculate_features(...)`

2. **`calculate_features_batch(df, fit_scaler, scaler_fit_ratio)`** (lines 559-600)
   - Call `self.calculator._calculate_features_batch(df, fit_scaler, scaler_fit_ratio)`
   - Current: `self._legacy_impl.calculate_features_batch(...)`

3. **`calculate_features_online(current_bar, indicator_manager, scaler)`** (lines 602-651)
   - Call `self.calculator._calculate_features_online(current_bar, indicator_manager, scaler)`
   - Current: `self._legacy_impl.calculate_features_online(...)`
   - **HOT PATH** - P99 < 5ms

4. **`compute_features(bars)`** (lines 428-464)
   - Call `self.calculator.compute_features(bars)`
   - Current: `self._legacy_impl.compute_features(...)`

### Key Invariants

1. **Numerical Parity (CRITICAL)**: `np.testing.assert_allclose(legacy, facade, rtol=1e-10)` must pass
2. **Type Parity**: Return types must match legacy (DataFrame, np.ndarray, dict)
3. **HOT PATH**: P99 < 5ms for online mode
4. **Zero Allocations**: feature_buffer must be reused across calls
5. **Scaler Behavior**: Same fitting/application logic

### Error Handling Requirements

- `calculate_features(mode="online")` without `indicator_manager`: Raise `ValueError("indicator_manager is required")`
- `calculate_features(mode="invalid")`: Raise `ValueError("Invalid mode: {mode}. Must be 'batch' or 'online'")`
- `compute_features([])`: Raise `ValueError("bars list cannot be empty")`

### Performance Requirements

- `calculate_features_online` P99 < 5ms (5000 microseconds)
- Facade overhead < 10% vs direct calculator call
- Memory allocation < 1000 bytes per call

### Backward Compatibility Constraints

- All public method signatures unchanged
- Return types unchanged
- Feature names order unchanged
- Scaler fitting behavior unchanged
- Legacy `compute_features(bars)` still works with Bar objects

### Special Considerations

1. **Feature Flag**: `ML_USE_LEGACY_FEATURE_ENGINEER` env var may still control behavior during gradual rollout
2. **Scaler Sharing**: Scaler fitted in batch mode must work correctly when passed to online mode
3. **Indicator Manager State**: Online mode relies on pre-warmed indicator manager
4. **feature_buffer Property**: Must return calculator's feature_buffer for zero-allocation access

### APIs to Verify with Codex

1. `FeatureEngineer.calculate_features(data, mode, indicator_manager, fit_scaler, scaler_fit_ratio, scaler)`
2. `FeatureEngineer.calculate_features_batch(df, fit_scaler, scaler_fit_ratio)`
3. `FeatureEngineer.calculate_features_online(current_bar, indicator_manager, scaler)`
4. `FeatureEngineer.compute_features(bars)`
5. `FeatureCalculator._calculate_features_batch(data, fit_scaler, scaler_fit_ratio)`
6. `FeatureCalculator._calculate_features_online(current_bar, indicator_manager, scaler)`
7. `FeatureCalculator.compute_features(bars)`
8. `FeatureEngineer.feature_buffer` property

## Validation Checklist

Before handing off to implementation:
- [x] All test files have clear, descriptive names
- [x] All test functions have docstrings explaining purpose
- [x] All assertions have clear failure messages
- [x] Fixtures are properly scoped (function/module/session)
- [x] Property tests have appropriate Hypothesis strategies
- [x] Integration tests use `clean_postgres_db` for isolation (N/A - no DB tests)
- [x] E2E tests cover complete workflows
- [x] Error tests verify specific exception types and messages
- [x] Edge case tests cover boundaries and null cases
- [x] Coverage expectations are realistic and justified
- [x] Performance tests have clear benchmarks
- [x] All tests are initially marked `@pytest.mark.skip` or designed to fail
