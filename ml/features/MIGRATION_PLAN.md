# Migration Plan: FeatureEngineerV2 to Current ML Infrastructure

## Executive Summary

This document outlines the migration plan for integrating advanced features from the OLD FeatureEngineerV2 implementation into the current ML infrastructure while maintaining backward compatibility and adhering to Nautilus Trader's architectural requirements.

## Current State Analysis

### Current Implementation (`ml/features/engineering.py`)
**Strengths:**

- Clean implementation using msgspec for configuration
- Proper hot/cold path separation
- Uses Nautilus native indicators
- Feature parity validation with < 1e-10 tolerance
- Memory-bounded with proper cleanup
- Pre-allocated numpy arrays for hot path

**Core Features:**

- Basic technical indicators (RSI, BB, ATR, MACD, EMA, SMA)
- Price returns and momentum
- Volatility calculations
- Volume ratios
- Price position features

### OLD Implementation (`OLD/trade/nautilus_ml/features/feature_engineering.py`)
**Advanced Features to Migrate:**

1. **Microstructure Features** (Optional, cold path only)
   - Bid-ask spread statistics
   - Size imbalance metrics
   - Mid-price return autocorrelation

2. **Trade Flow Features** (Optional, cold path only)
   - Trade flow imbalance
   - VWAP calculations
   - Trade intensity metrics
   - Price impact analysis

3. **Feature Store Integration**
   - Feature versioning
   - Feature registry support
   - Computation metrics tracking
   - Quality validation metrics

4. **Causal Feature Selection**
   - EconML integration for causal inference
   - Treatment effect estimation
   - Confounder identification

5. **Additional Nautilus Indicators**
   - Can add: Aroon, CCI, CMO, Stochastics, OBV, Keltner Channels
   - Already available in Nautilus but not used

## Migration Strategy

### Phase 1: Core Feature Enhancement (Priority: HIGH)
**Timeline: 1-2 days**

#### 1.1 Add Optional Advanced Features

```python
# Extend FeatureConfig with backward compatibility
class FeatureConfig(MLFeatureConfig, kw_only=True, frozen=True):
    # Existing fields...

    # New optional features (default False for backward compatibility)
    include_microstructure: bool = False
    include_trade_flow: bool = False
    include_advanced_indicators: bool = False

    # Additional indicator periods
    aroon_period: int = 25
    cci_period: int = 20
    stoch_period: int = 14
    obv_ma_period: int = 20
```

#### 1.2 Implement Microstructure Features (Cold Path Only)

- Add `calculate_microstructure_features()` method
- Only process when `include_microstructure=True`
- Use Polars for batch processing
- Ensure features are added to feature_names list

#### 1.3 Implement Trade Flow Features (Cold Path Only)

- Add `calculate_trade_flow_features()` method
- Only process when `include_trade_flow=True`
- Require additional data (trades, quotes)
- Maintain feature parity between batch/online

### Phase 2: Feature Store Integration (Priority: MEDIUM)
**Timeline: 2-3 days**

#### 2.1 Create Feature Store Abstraction

```python
class FeatureStore(ABC):
    @abstractmethod
    def store_features(self, features: np.ndarray, metadata: dict) -> None:
        pass

    @abstractmethod
    def get_features(self, entity_ids: list[str], timestamp: int) -> np.ndarray:
        pass
```

#### 2.2 Add Feature Versioning

- Track feature configuration versions
- Store feature computation metadata
- Enable feature rollback capability

#### 2.3 Computation Metrics Tracking

- Track latency percentiles (P50, P95, P99)
- Monitor feature quality metrics
- Alert on performance degradation

### Phase 3: Advanced Indicators (Priority: LOW)
**Timeline: 1 day**

#### 3.1 Add Additional Nautilus Indicators

```python
# Add to IndicatorManager
from nautilus_trader.indicators.aroon import Aroon
from nautilus_trader.indicators.cci import CommodityChannelIndex
from nautilus_trader.indicators.stochastics import Stochastics
from nautilus_trader.indicators.obv import OnBalanceVolume
from nautilus_trader.indicators.keltner_channel import KeltnerChannel
```

#### 3.2 Ensure Proper Initialization

- Add to indicator_specs generation
- Update feature_names list
- Validate in both batch and online paths

### Phase 4: Causal Feature Selection (Priority: OPTIONAL)
**Timeline: 3-5 days**

**Note:** This requires external dependencies (EconML) and should be optional.

#### 4.1 Create Optional Causal Module

```python
# ml/features/causal.py (optional module)
try:
    from econml import CausalForest
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

class CausalFeatureSelector:
    def select_features(self, data, treatment, outcome):
        if not ECONML_AVAILABLE:
            raise ImportError("EconML required for causal feature selection")
        # Implementation...
```

## Implementation Details

### 1. Backward Compatibility Requirements

- All new features MUST be optional (default disabled)
- Existing API must remain unchanged
- Performance must not degrade for basic features
- Test coverage must remain > 90%

### 2. Performance Constraints

#### Hot Path (Real-time Inference)

- Feature computation: < 500μs
- No Polars/Pandas operations
- Pre-allocated numpy arrays only
- No dynamic memory allocation

#### Cold Path (Training)

- Can use Polars for efficiency
- Batch processing optimizations
- Memory management for large datasets

### 3. Testing Requirements

#### Feature Parity Tests

```python
def test_microstructure_feature_parity():
    """Test batch/online consistency for microstructure features."""
    config = FeatureConfig(include_microstructure=True)
    validator = FeatureParityValidator(config)

    # Generate test data with quotes
    df = generate_test_data_with_quotes()

    # Validate parity
    report = validator.validate_parity(df)
    assert report["parity_passed"]
    assert report["max_difference"] < 1e-10
```

#### Performance Tests

```python
def test_feature_computation_latency():
    """Ensure feature computation meets latency requirements."""
    config = FeatureConfig(
        include_microstructure=True,
        include_trade_flow=True
    )
    validator = FeatureParityValidator(config)

    df = generate_large_test_data(n_samples=10000)
    report = validator.validate_performance(df)

    assert report["p99_latency_ms"] < 5.0  # 5ms P99 requirement
```

### 4. Migration Steps

#### Step 1: Create Feature Branch

```bash
git checkout -b ml-feature-enhancement
```

#### Step 2: Implement Core Changes

1. Extend FeatureConfig with optional fields
2. Add microstructure feature calculation
3. Add trade flow feature calculation
4. Update feature_names generation
5. Ensure proper array pre-allocation

#### Step 3: Add Comprehensive Tests

1. Feature parity tests for new features
2. Performance benchmarks
3. Backward compatibility tests
4. Edge case handling

#### Step 4: Update Documentation

1. Add docstrings for new methods
2. Update feature descriptions
3. Add usage examples
4. Document performance characteristics

#### Step 5: Performance Validation

```bash
# Run performance benchmarks
python scripts/check_code_quality.py ml/features/engineering.py

# Validate feature parity
pytest tests/ml/features/test_feature_parity.py -v

# Check test coverage
pytest --cov=ml.features --cov-report=term-missing
```

## Risk Assessment

### High Risk Areas

1. **Feature Parity**: Any discrepancy between batch/online will break models
   - **Mitigation**: Comprehensive parity testing with 1e-10 tolerance

2. **Performance Degradation**: New features could slow down inference
   - **Mitigation**: Optional features, pre-allocation, performance tests

3. **Memory Leaks**: Unbounded collections in hot path
   - **Mitigation**: Fixed-size buffers, regular cleanup

### Medium Risk Areas

1. **Dependency Conflicts**: EconML/other dependencies
   - **Mitigation**: Make causal features completely optional

2. **API Changes**: Breaking existing code
   - **Mitigation**: All changes are additive, defaults preserve behavior

## Recommended Implementation Order

1. **Week 1**: Core Feature Enhancement
   - Add microstructure features (2 days)
   - Add trade flow features (2 days)
   - Comprehensive testing (1 day)

2. **Week 2**: Feature Store Integration (if needed)
   - Design abstraction layer (1 day)
   - Implement storage backend (2 days)
   - Add versioning support (2 days)

3. **Future**: Optional Enhancements
   - Additional indicators (as needed)
   - Causal feature selection (if EconML dependency acceptable)

## Success Criteria

1. **Functional Requirements**
   - ✅ All existing tests pass
   - ✅ New features have > 90% test coverage
   - ✅ Feature parity < 1e-10 tolerance
   - ✅ Backward compatibility maintained

2. **Performance Requirements**
   - ✅ P99 latency < 5ms for feature computation
   - ✅ No memory leaks over 24h operation
   - ✅ Zero allocations in hot path

3. **Quality Requirements**
   - ✅ MyPy: 0 errors
   - ✅ Ruff: 0 violations
   - ✅ Documentation complete
   - ✅ Examples provided

## Conclusion

The migration can be completed incrementally without disrupting existing functionality. The core enhancements (microstructure and trade flow features) provide the most value and should be prioritized. Feature store integration and causal selection can be deferred based on actual requirements.

The key to success is maintaining strict separation between hot and cold paths, ensuring feature parity, and keeping all enhancements optional to preserve backward compatibility.
