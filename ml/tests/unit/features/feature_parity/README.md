# Feature Parity Validation Tests

This directory contains comprehensive tests to ensure perfect feature parity between batch (training) and online (inference) feature computations with < 1e-10 tolerance.

## Overview

Feature parity validation is critical for ML model performance in production. These tests guarantee that:

1. **Batch processing** (used during training) produces identical features to
2. **Online processing** (used during live inference)

Any discrepancy beyond the tolerance threshold indicates a bug that could cause model performance degradation in production.

## Test Structure

```
feature_parity/
├── __init__.py                    # Package initialization
├── utils.py                       # Core testing utilities
├── test_parity_technical.py       # Technical indicator parity tests
├── test_parity_microstructure.py  # Microstructure feature parity tests
├── test_parity_trade_flow.py      # Trade flow feature parity tests
└── test_parity_edge_cases.py      # Edge case and robustness tests
```

## Core Utilities (`utils.py`)

### ParityTestUtils

- `assert_features_equal()`: Validates feature parity within tolerance
- `compare_feature_vectors()`: Compares batch DataFrame vs online feature arrays
- `measure_computation_time()`: Performance timing utilities
- `validate_performance()`: Ensures hot path performance requirements

### TestDataGenerators

- `generate_normal_ohlcv()`: Standard market data scenarios
- `generate_trending_data()`: Trending market conditions
- `generate_volatile_data()`: High volatility scenarios
- `generate_gapped_data()`: Price gaps and discontinuities
- `generate_with_microstructure_data()`: Bid/ask and size data
- `generate_with_trade_data()`: Trade-level data

### PerformanceProfiler

- `profile_feature_computation()`: Latency measurements
- `validate_latency_requirements()`: P99 < 5ms validation
- `get_performance_summary()`: Performance reporting

## Test Categories

### Technical Indicator Tests (`test_parity_technical.py`)
Tests parity for core technical indicators:

- **SMA/EMA**: Moving averages with different periods
- **RSI**: Relative Strength Index with normalization
- **Bollinger Bands**: Upper/middle/lower bands and derived features
- **MACD**: MACD line, signal line, and histogram
- **ATR**: Average True Range for volatility
- **Volume Indicators**: Volume ratios and patterns

**Key Tests:**

- `test_sma_feature_parity_normal_data()`: Basic SMA validation
- `test_ema_feature_parity_trending_data()`: EMA responsiveness
- `test_rsi_feature_parity_volatile_data()`: RSI in volatile conditions
- `test_bollinger_bands_feature_parity()`: Multi-scenario BB testing
- `test_macd_feature_parity_with_signals()`: Complete MACD system
- `test_atr_feature_parity_with_gaps()`: ATR with price gaps

### Microstructure Tests (`test_parity_microstructure.py`)
Tests market microstructure features with fallback handling:

- **Spread Metrics**: Bid/ask spreads, relative spreads
- **Size Imbalance**: Bid/ask size ratios and statistics
- **Mid-Price Returns**: Mid-price volatility and autocorrelation
- **OHLCV Fallbacks**: Approximations when microstructure data unavailable

**Key Tests:**

- `test_microstructure_parity_with_bid_ask_data()`: Full microstructure data
- `test_microstructure_parity_ohlcv_fallback()`: Fallback calculations
- `test_spread_metrics_parity_volatile_data()`: Volatile spread scenarios
- `test_size_imbalance_parity()`: Size imbalance variations

### Trade Flow Tests (`test_parity_trade_flow.py`)
Tests trade-level flow features with fallback scenarios:

- **Trade Imbalance**: Buy/sell volume ratios
- **VWAP**: Volume-weighted average price calculation
- **Trade Intensity**: Trade frequency normalization
- **Price Impact**: Average price impact per trade
- **OHLCV Fallbacks**: Approximations without trade data

**Key Tests:**

- `test_trade_flow_parity_with_trade_data()`: Full trade-level data
- `test_trade_flow_parity_ohlcv_fallback()`: Fallback scenarios
- `test_trade_imbalance_parity_directional_data()`: Directional biases
- `test_vwap_calculation_parity()`: VWAP accuracy validation

### Edge Case Tests (`test_parity_edge_cases.py`)
Tests robustness under challenging conditions:

- **Extreme Values**: Very large/small prices and volumes
- **Numerical Precision**: Values near machine epsilon
- **Zero Handling**: Zero volumes and price spreads
- **Constant Data**: No price movement scenarios
- **Initialization**: Insufficient data for indicators
- **Stress Scenarios**: Combined challenging conditions

**Key Tests:**

- `test_extremely_small_values_parity()`: Precision handling
- `test_extremely_large_values_parity()`: Overflow prevention
- `test_zero_values_handling_parity()`: Zero division protection
- `test_constant_price_data_parity()`: No volatility scenarios
- `test_numerical_precision_edge_cases()`: Machine epsilon limits

## Performance Requirements

All tests validate that online feature computation meets strict performance requirements:

- **P99 Latency**: < 5ms (configurable)
- **Mean Latency**: < 2ms for standard features
- **Memory**: Pre-allocated buffers, no dynamic allocation
- **Stability**: No memory leaks over 24h periods

## Tolerance Settings

- **Default Tolerance**: 1e-10 (MLConstants.FEATURE_PARITY_TOLERANCE)
- **Relaxed Tolerance**: 1e-9 for statistical features
- **Precision Tolerance**: 1e-8 for extreme numerical conditions

## Running Tests

### Individual Test Categories

```bash
# Technical indicators only
pytest ml/tests/unit/feature_parity/test_parity_technical.py -v

# Microstructure features only
pytest ml/tests/unit/feature_parity/test_parity_microstructure.py -v

# Trade flow features only
pytest ml/tests/unit/feature_parity/test_parity_trade_flow.py -v

# Edge cases only
pytest ml/tests/unit/feature_parity/test_parity_edge_cases.py -v
```

### All Parity Tests

```bash
# Run complete parity validation suite
pytest ml/tests/unit/feature_parity/ -v
```

### Specific Test Cases

```bash
# Test specific indicator
pytest ml/tests/unit/feature_parity/test_parity_technical.py::TestTechnicalIndicatorParity::test_rsi_feature_parity_volatile_data -v

# Test performance requirements
pytest ml/tests/unit/feature_parity/ -k performance -v

# Test edge cases only
pytest ml/tests/unit/feature_parity/test_parity_edge_cases.py -v
```

## Expected Outcomes

### Passing Tests
When feature parity is perfect:

```
✓ All features match within 1e-10 tolerance
✓ Performance requirements met (P99 < 5ms)
✓ No NaN or infinite values generated
✓ Consistent results across multiple runs
```

### Failing Tests (Parity Violations)
When parity is violated, tests provide detailed diagnostics:

```
AssertionError: Feature parity violation:
  Maximum difference: 7.91e-02 (tolerance: 1.00e-10)
  At index: (19, 22)
  Batch value: 0.5
  Online value: 0.579
  Feature name: price_position_20
```

This indicates the specific feature, index, and magnitude of the violation.

## Integration with CI/CD

These tests are designed for continuous integration:

1. **Pre-commit**: Run subset of parity tests
2. **Pull Request**: Full parity validation suite
3. **Nightly**: Extended edge case and performance testing
4. **Release**: Complete validation with performance profiling

## Debugging Parity Violations

When tests fail:

1. **Identify Feature**: Check error message for specific feature name
2. **Check Implementation**: Compare batch vs online calculation logic
3. **Verify Data**: Ensure identical input data to both paths
4. **Test Isolation**: Run specific feature test in isolation
5. **Add Logging**: Temporarily add debug output to trace calculations

## Coverage Requirements

- **Technical Features**: 95%+ coverage of all indicator combinations
- **Microstructure Features**: 90%+ coverage including fallback scenarios
- **Trade Flow Features**: 90%+ coverage with and without trade data
- **Edge Cases**: 100% coverage of numerical edge conditions

## Maintenance

This test suite should be updated when:

1. **New Features Added**: Add corresponding parity tests
2. **Indicators Modified**: Update affected test cases
3. **Performance Requirements Change**: Update validation thresholds
4. **New Edge Cases Discovered**: Add specific test scenarios

The parity test suite is critical infrastructure that prevents ML model performance degradation in production by catching feature computation discrepancies early in development.
