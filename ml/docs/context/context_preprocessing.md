# Context: Preprocessing Module

## Overview

The `ml/preprocessing/` directory implements advanced preprocessing and transformation utilities for financial time series data. The module provides sophisticated techniques from academic literature, particularly "Advances in Financial Machine Learning" by López de Prado, with a focus on preventing information leakage, ensuring point-in-time correctness, and applying proper stationarity transformations. All components are designed to integrate seamlessly with both Polars and Pandas dataframes while adhering to Nautilus Trader's data model standards.

The preprocessing module serves as a critical foundation for the ML pipeline, providing statistically sound data preparation techniques that maintain temporal ordering, prevent lookahead bias, and apply advanced transformations like fractional differencing to achieve stationarity while preserving memory in financial time series.

## Architecture

### Structural Organization

```
ml/preprocessing/
├── stationarity.py          # Advanced stationarity transformations and cross-validation
└── joins.py                 # Point-in-time correct joins and temporal utilities
```

The module follows a dual-dataframe approach, providing implementations for both Polars (preferred for performance) and Pandas (compatibility), with automatic detection and routing based on input dataframe types. All functions maintain strict temporal ordering and implement sophisticated financial ML techniques to prevent common pitfalls in time series modeling.

### Design Principles

1. **Point-in-Time Correctness**: All temporal operations ensure no future information leakage
2. **Dual DataFrame Support**: Native support for both Polars and Pandas with automatic routing
3. **Academic Rigor**: Implementation of proven techniques from financial ML literature
4. **Performance Optimization**: JIT compilation via Numba where available, efficient vectorized operations
5. **Nautilus Integration**: Timestamp handling in nanoseconds, instrument_id grouping support

## Key Components

### Stationarity Transformations (`stationarity.py`)

#### StationarityTransformer
Advanced transformer implementing fractional differencing for financial time series:

- **Fractional Differencing**: Achieves stationarity while preserving memory using López de Prado's method
- **Automatic d Selection**: Uses ADF test to find optimal differencing order (`find_optimal_d`)
- **Weight Computation**: JIT-compiled `_compute_weights_numba` for efficient fractional weights calculation
- **Threshold Filtering**: Drops small weights below configurable threshold for computational efficiency
- **Transformation Methods**: `fractional`, `standard`, or `auto` mode with optimal parameter detection

Key methods:

- `fractional_difference()`: Apply fractional differencing with configurable order d
- `fit_transform()`: Combined fitting and transformation with optional auto-d detection
- `inverse_transform()`: Approximate reconstruction of original series

#### MarketMicrostructureFeatures
Comprehensive market microstructure analytics implementation:

- **Roll's Spread Estimator**: Bid-ask spread estimation from price covariance
- **Kyle's Lambda**: Price impact measurement from signed volume regression
- **Amihud Illiquidity**: Market liquidity measure based on return-to-volume ratio
- **VPIN**: Volume-synchronized Probability of Informed Trading with volume bucket analysis

All methods handle edge cases (zero volume, insufficient data) and return robust estimates.

#### FeatureLagGenerator
Comprehensive lagged feature creation system:

- **Simple Lags**: Configurable lag periods with proper NaN handling
- **Rolling Statistics**: Rolling means and standard deviations with efficient convolution
- **Exponentially Weighted Features**: EWM calculations with configurable decay spans
- **Vectorized Implementation**: Efficient numpy-based computations for large datasets

#### DataNormalizer
Advanced normalization techniques resistant to financial data characteristics:

- **Robust Scaling**: Uses median and MAD instead of mean/std for outlier resistance
- **Rank Transformation**: Converts to uniform then normal distribution via percentiles
- **Box-Cox Transformation**: Power transformation with automatic lambda estimation
- **Invertible Operations**: All transformations support approximate inverse transformation

#### PurgedCrossValidator
Industry-standard purged walk-forward cross-validation for financial ML:

- **Purge Gap**: Configurable gap between train/test to prevent information leakage
- **Embargo Period**: Percentage-based embargo after test sets
- **Walk-Forward Logic**: Proper temporal ordering for time series validation
- **sklearn Compatibility**: Compatible with sklearn cross-validation interfaces

### Point-in-Time Joins (`joins.py`)

#### asof_join()
Dual-implementation as-of join ensuring temporal correctness:

- **Backward/Forward/Nearest**: Configurable join directions with tolerance
- **By-Group Support**: Group-wise joins (e.g., by instrument_id)
- **Automatic Routing**: Detects dataframe type and routes to appropriate implementation
- **Tolerance Handling**: Time-based tolerance for match flexibility

Implementation variants:

- `_asof_join_polars()`: Polars-optimized implementation using join_asof
- `_asof_join_pandas()`: Pandas implementation using merge_asof

#### embargo_window()
Event-based embargo window application:

- **Event Timestamps**: List of timestamps requiring embargo periods
- **Configurable Windows**: Before/after embargo periods in nanoseconds
- **Boolean Marking**: Adds `embargo` column marking affected periods
- **Dual Implementation**: Efficient implementations for both dataframe types

#### validate_no_lookahead()
Critical validation function preventing future information usage:

- **Timestamp Validation**: Ensures features don't exceed target timestamps
- **Automatic Detection**: Works with any timestamp column names
- **Exception Raising**: Clear error messages when lookahead bias detected

#### create_lag_features()
Point-in-time correct lag feature creation:

- **Group-Aware Lagging**: Proper lagging within instrument groups
- **Temporal Ordering**: Ensures proper timestamp-based ordering before lagging
- **Multiple Lags**: Support for multiple lag periods simultaneously
- **NaN Handling**: Proper NaN filling for initial periods without data

## Dependencies

### Internal Dependencies

- **ml._imports**: Lazy loading system for Polars (`HAS_POLARS`) and Pandas (`HAS_PANDAS`)
- **nautilus_trader.core.data**: Nanosecond timestamp handling standards
- **ml.common.metrics_bootstrap**: Prometheus metrics integration (indirectly via actors)

### External Dependencies

- **numpy**: Core numerical computations and array operations
- **polars**: High-performance dataframe operations (lazy-loaded)
- **pandas**: Traditional dataframe operations for compatibility (lazy-loaded)
- **scipy.stats**: Statistical functions (ADF test, Box-Cox, normal distribution)
- **statsmodels**: Augmented Dickey-Fuller test for stationarity detection
- **numba**: JIT compilation for performance-critical operations (optional)

## Usage Patterns

### Stationarity Transformation

```python
from ml.preprocessing.stationarity import StationarityTransformer

# Fractional differencing with auto d-selection
transformer = StationarityTransformer(method="auto")
stationary_series = transformer.fit_transform(price_series, auto_d=True)

# Manual fractional differencing
transformer = StationarityTransformer(method="fractional", d=0.4)
stationary_series = transformer.fractional_difference(price_series)
```

### Point-in-Time Joins

```python
from ml.preprocessing.joins import asof_join, embargo_window

# Join market data with corporate events (no lookahead)
joined_data = asof_join(
    market_df, events_df,
    on="ts_event",
    by="instrument_id",
    direction="backward"
)

# Apply embargo around earnings releases
embargoed_data = embargo_window(
    df, earnings_timestamps,
    embargo_before_ns=3600_000_000_000,  # 1 hour before
    embargo_after_ns=7200_000_000_000    # 2 hours after
)
```

### Cross-Validation

```python
from ml.preprocessing.stationarity import PurgedCrossValidator

# Purged walk-forward CV
cv = PurgedCrossValidator(
    n_splits=5,
    purge_gap=10,      # 10 samples gap
    embargo_pct=0.1    # 10% embargo
)

for train_idx, test_idx in cv.split(X):
    # Training and testing with no information leakage
    pass
```

## Integration Points

### Feature Pipeline Integration

- **FeatureEngineer**: Uses stationarity transformers in preprocessing pipelines
- **Pipeline Specifications**: TransformSpec integration for declarative preprocessing
- **Batch/Online Parity**: Ensures identical transformations in training and inference

### ML Actor Integration

- **BaseMLInferenceActor**: Preprocessing transformations in data preparation phase
- **Signal Generation**: Stationary features for improved model performance
- **Real-time Processing**: Hot-path optimized transformations for inference

### Data Store Integration

- **Temporal Validation**: Joins ensure proper ts_event/ts_init handling
- **Instrument Grouping**: By-instrument processing maintains data isolation
- **Schema Compliance**: All operations preserve required Nautilus timestamp fields

### Training Pipeline Integration

- **Cross-Validation**: PurgedCrossValidator prevents overfitting in time series models
- **Data Preparation**: Stationarity transformation before model training
- **Feature Engineering**: Lag generation and microstructure features for model input

## Implementation Notes

### Performance Optimizations

- **JIT Compilation**: Numba acceleration for fractional weight computation where available
- **Vectorization**: All operations use vectorized numpy/polars operations
- **Memory Efficiency**: In-place operations and pre-allocated arrays where possible
- **Lazy Evaluation**: Polars lazy evaluation patterns for large dataset processing

### Numerical Stability

- **Weight Thresholding**: Drops small fractional differencing weights for stability
- **Division by Zero**: Proper handling in microstructure calculations
- **NaN Propagation**: Consistent NaN handling across all transformations
- **Precision Handling**: Uses float64 throughout for numerical accuracy

### Temporal Correctness

- **Strict Ordering**: All operations maintain proper temporal ordering
- **No Lookahead**: Multiple validation layers prevent future information usage
- **Embargo Enforcement**: Event-based embargo windows for information quarantine
- **Point-in-Time Joins**: Guaranteed historical data usage only

### Error Handling and Validation

- **Input Validation**: Comprehensive parameter validation with descriptive errors
- **Data Quality Checks**: Automatic detection of insufficient data for operations
- **Framework Availability**: Graceful handling when Polars/Pandas unavailable
- **Statistical Validity**: ADF test validation for stationarity assessment

### Academic Compliance

- **López de Prado Methods**: Faithful implementation of AFML techniques
- **Statistical Rigor**: Proper statistical tests and validation procedures
- **Literature References**: Code comments reference relevant academic sources
- **Best Practices**: Incorporates industry best practices for financial ML preprocessing

The preprocessing module provides the statistical foundation for robust financial ML applications, ensuring that all data transformations maintain temporal integrity while applying sophisticated techniques for stationarity, feature engineering, and cross-validation. Its dual-dataframe approach and integration with Nautilus Trader's data standards make it essential for any ML pipeline working with financial time series data.
