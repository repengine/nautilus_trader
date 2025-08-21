# Feature Engineering Context Document

## Executive Summary

The `ml/features/` directory contains a sophisticated, production-ready feature engineering system that guarantees mathematical parity between batch (training) and real-time (inference) computation paths. This is one of the most mature and complete modules in the ML infrastructure, implementing hot/cold path separation, zero-allocation online processing, and comprehensive validation with <1e-10 tolerance.

### Key Architectural Principles

1. **Perfect Feature Parity**: Guaranteed identical features between batch and online computation
2. **Hot/Cold Path Separation**: Optimized paths for training vs inference workloads
3. **Zero-Allocation Online Processing**: Pre-allocated buffers for real-time performance
4. **Comprehensive Validation**: Built-in parity validation with detailed reporting
5. **Nautilus Integration**: Seamless integration with Nautilus Trader indicators

## Module Structure

```
ml/features/
├── __init__.py              # Public API exports
├── engineering.py           # Core feature engineering classes (2,436 lines)
├── validation.py           # Parity validation system (680 lines)
├── pipeline.py             # Declarative pipeline framework (508 lines)
├── microstructure.py       # L2/L3 microstructure features (948 lines)
├── feature_export.py       # Registry integration utilities (53 lines)
└── materialize_cli.py      # Feature materialization CLI (109 lines)
```

## Core Components

### 1. FeatureEngineer Class (`engineering.py`)

The primary feature computation engine with dual-mode operation:

#### Cold Path (Batch Processing)

- **Purpose**: Training data preparation and validation
- **Optimization**: Vectorized operations using Polars/Pandas
- **Memory**: Dynamic allocation acceptable
- **Performance**: Optimized for throughput

```python
features_df, scaler = engineer.calculate_features(
    df, mode="batch", fit_scaler=True
)
```

#### Hot Path (Online Processing)

- **Purpose**: Real-time inference during trading
- **Optimization**: Pre-allocated arrays, zero dynamic allocation
- **Memory**: Fixed buffer with `FEATURE_BUFFER_PAD` space
- **Performance**: <5ms P99 latency requirement

```python
features = engineer.calculate_features(
    current_bar, mode="online",
    indicator_manager=mgr, scaler=scaler
)
```

### 2. IndicatorManager Class

Manages stateful Nautilus indicators for consistent calculations:

- **Indicators Supported**: SMA, EMA, RSI, Bollinger Bands, ATR, MACD
- **State Management**: Price history with `PRICE_HISTORY_MAXLEN` limit
- **Dual Update**: Both Bar objects and raw OHLCV values
- **Vectorized Batch**: `update_batch_vectorized()` for training

### 3. FeatureConfig Class

Configuration system with comprehensive validation:

```python
@dataclass(kw_only=True, frozen=True)
class FeatureConfig(MLFeatureConfig):
    # Technical indicators
    rsi_period: int = 14              # [2, 100]
    bb_period: int = 20               # [2, 100]
    bb_std: float = 2.0               # [0.5, 5.0]

    # Moving averages
    ema_fast: int = 12                # [2, 50]
    ema_slow: int = 26                # [10, 200], > ema_fast

    # Feature toggles
    include_microstructure: bool = False
    include_trade_flow: bool = False
```

## Feature Categories

### 1. Price-Based Features

- **Returns**: Multiple periods (1, 5, 10, 20 bars)
- **Momentum**: Same as returns for consistency
- **Volatility**: Rolling standard deviation (5, 20 periods)

### 2. Volume Features

- **Volume Ratios**: Current volume vs moving averages
- **Normalization**: Against configurable MA periods

### 3. Technical Indicators

- **RSI**: Normalized to [-1, 1] range for ML
- **Bollinger Bands**: Width and position metrics
- **ATR**: Normalized by current price
- **EMA Cross**: Fast/slow EMA relationships
- **MACD**: Line, signal, and difference
- **Price Position**: Location within 20-day range

### 4. Microstructure Features (L2/L3)
Advanced order book and trade flow analysis:

#### L2 Order Book Features

- **Spread Metrics**: Basic, weighted, effective spreads
- **Imbalance**: Level-wise and weighted imbalances
- **Depth Analysis**: Concentration, VWAP, slope
- **Shape Features**: Skewness, kurtosis, liquidity zones

#### L3 Trade Flow Features

- **Trade Imbalance**: Volume, dollar, count imbalances
- **VWAP Features**: Price deviation, variance, sided VWAP
- **Intensity**: Trade rate, clustering, size statistics
- **Price Impact**: Kyle's lambda, temporary vs permanent

## Hot Path Optimizations

### 1. Pre-Allocation Strategy

```python
# Feature buffer sized with padding
buffer_size = n_features + SystemConstants.FEATURE_BUFFER_PAD
self.feature_buffer = np.zeros(buffer_size, dtype=np.float32)
```

### 2. Zero-Copy Returns

```python
# Return view of buffer (zero allocation)
return self.feature_buffer[:feature_idx]
```

### 3. Memory Management

- **Price History**: Limited to `PRICE_HISTORY_MAXLEN` (default: 1000)
- **Indicator State**: Reused across calculations
- **Buffer Reuse**: Same buffer for all feature calculations

### 4. Safe Division
All division operations use `safe_divide()` with configurable defaults to prevent NaN/Inf propagation.

## Parity Validation System

### FeatureParityValidator Class

Comprehensive validation ensuring identical results between batch and online modes:

#### Validation Process

1. **Batch Computation**: Process full dataset with indicators
2. **Online Simulation**: Step-by-step processing matching real-time
3. **State Alignment**: Warm up online indicators to match batch state
4. **Comparison**: Element-wise difference validation
5. **Reporting**: Detailed per-feature analysis

#### Critical Implementation Details

- **Indicator Warmup**: Online indicators warmed to same state as batch
- **Feature Buffer Copying**: Copy features since online returns views
- **Shape Validation**: Ensure array dimensions match exactly
- **Tolerance Checking**: Default `1e-10` for numerical precision

#### Validation Report

```python
{
    "parity_passed": bool,
    "max_difference": float,
    "tolerance": float,
    "failing_features": list[str],
    "feature_differences": dict,  # Per-feature analysis
    "validation_time": float,
    "n_samples_validated": int
}
```

### Performance Validation

- **Latency Testing**: P99 latency measurement
- **Target**: <5ms for online feature computation
- **Metrics**: Mean, std, percentiles across iterations

## Pipeline Framework

### Declarative Transform System

```python
@dataclass
class PipelineSpec:
    transforms: list[TransformSpec]

class TransformSpec:
    name: str
    params: dict[str, Any]
```

### Transform Catalog

- **Core Transforms**: returns, momentum, volatility, volume_ratio, core_indicators
- **Advanced Transforms**: microstructure, trade_flow, keltner, obv
- **TFT Transforms**: calendar, event_schedule, macro_indicators, static_covariates

### Data Requirements Gating

```python
class DataRequirements(Enum):
    L1_ONLY = "L1_ONLY"        # OHLCV only
    L1_L2 = "L1_L2"            # + Order book L2
    L1_L2_L3 = "L1_L2_L3"      # + Trade data L3
```

## Integration Points

### 1. Registry Integration

- **FeatureManifest**: Schema hashing and versioning
- **Feature Registry**: Local registration and retrieval
- **Materialization**: CLI tool for feature export

### 2. Store Integration
Features integrate with the mandatory store triad:

- **FeatureStore**: Persists computed feature values
  - Automatic persistence from DataScheduler
  - TFTDatasetBuilder reads from FeatureStore
  - Training/inference parity through shared storage
- **ModelStore**: Stores predictions using features
- **StrategyStore**: Trading decisions based on features

### 3. Metrics Integration
Comprehensive Prometheus metrics:

- **Computation Time**: By instrument, feature type, mode
- **Feature Quality**: Null rates, outliers, drift detection
- **Cache Performance**: Hit rates and latencies

## Performance Benchmarks

### Hot Path Requirements

- **P99 Latency**: <5ms for online feature computation
- **Memory**: Zero dynamic allocation during inference
- **Throughput**: Support high-frequency trading workloads

### Cold Path Optimization

- **Vectorization**: Polars/Pandas for batch processing
- **Parallelization**: Multiple indicators updated simultaneously
- **Memory Efficiency**: Streaming processing for large datasets

## Current Implementation Status

### Completed Features ✅

- [x] Core technical indicators (RSI, BB, ATR, EMA, MACD)
- [x] Price and volume features
- [x] Perfect parity validation system
- [x] Hot/cold path separation
- [x] Pre-allocated online processing
- [x] Comprehensive configuration validation
- [x] L2/L3 microstructure features
- [x] Pipeline framework with transform catalog
- [x] Registry integration
- [x] Performance validation
- [x] Metrics integration
- [x] FeatureStore integration via DataScheduler
- [x] TFTDatasetBuilder with dual-source support (FeatureStore/direct)

### Advanced Features 🔄

- [ ] Fractional differencing integration (StationarityTransformer)
- [ ] Additional technical indicators (Keltner, OBV)
- [ ] Cross-sectional features
- [ ] Feature selection and importance analysis

### TFT-Specific Features 📋

- [x] Calendar features (time-based, cyclical encoding)
- [x] Event schedule features (earnings, Fed meetings)
- [x] Macro indicator features (VIX, rates, yield curve)
- [x] Static covariate features (instrument metadata)

## Critical Implementation Notes

### 1. Parity Validation Critical Path
The parity validation system is essential for production deployment:

- **Indicator State**: Must exactly match between batch and online
- **Numerical Precision**: 1e-10 tolerance prevents accumulation errors
- **Buffer Management**: Online features must be copied, not referenced

### 2. Hot Path Performance
Zero-allocation design enables low-latency inference:

- **Pre-allocation**: All arrays allocated at initialization
- **Buffer Reuse**: Same buffer for all feature computations
- **View Returns**: Return buffer views to avoid copying

### 3. Feature Normalization
Consistent normalization across modes:

- **RSI**: Normalized to [-1, 1] from Nautilus [0, 1]
- **Price Features**: Normalized by current price
- **Volume Features**: Normalized by moving averages

### 4. Error Handling
Robust error handling for production:

- **Safe Division**: Prevents NaN/Inf propagation
- **Bounds Checking**: Configuration validation
- **Graceful Degradation**: Default values for missing data

## Integration Examples

### Basic Usage

```python
# Configuration
config = FeatureConfig(
    rsi_period=14,
    include_microstructure=True
)

# Engineer
engineer = FeatureEngineer(config)

# Batch processing (training)
features_df, scaler = engineer.calculate_features(
    df, mode="batch", fit_scaler=True
)

# Online processing (inference)
indicator_mgr = IndicatorManager(config)
# ... warm up indicators ...
features = engineer.calculate_features(
    current_bar, mode="online",
    indicator_manager=indicator_mgr,
    scaler=scaler
)
```

### TFT Dataset Builder with FeatureStore

```python
from ml.data.tft_dataset_builder import TFTDatasetBuilder

# Create builder
builder = TFTDatasetBuilder(
    catalog=catalog,
    symbols=["SPY", "QQQ"],
    feature_store=feature_store  # Optional: reads from store if available
)

# Build dataset (automatically uses FeatureStore if connected)
dataset = builder.build_training_dataset(
    start_date="2024-01-01",
    end_date="2024-12-31",
    prediction_horizon=15
)
```

### Parity Validation

```python
validator = FeatureParityValidator(config)
report = validator.validate_parity(df)
assert report["parity_passed"]
assert report["max_difference"] < 1e-10
```

### Performance Testing

```python
perf_report = validator.validate_performance(df)
assert perf_report["performance_passed"]
assert perf_report["p99_latency_ms"] < 5.0
```

## Dependencies

### Required

- **numpy**: Core numerical operations
- **msgspec**: Configuration serialization
- **nautilus_trader**: Indicator implementations

### Optional

- **polars**: High-performance DataFrame operations (preferred)
- **pandas**: Fallback DataFrame operations
- **sklearn**: Feature scaling (StandardScaler)

## Quality Metrics

### Code Quality

- **Type Safety**: Complete type annotations
- **Documentation**: Comprehensive docstrings
- **Testing**: Extensive unit and parity tests
- **Linting**: Ruff compliant, formatted with black

### Feature Quality

- **Null Rate**: Monitored per feature
- **Outlier Rate**: IQR-based detection
- **Drift Detection**: Comparative statistics
- **Performance**: Latency and throughput monitoring

This feature engineering system represents a production-ready, high-performance solution for ML feature computation with guaranteed consistency between training and inference environments.
## Cross-Module References

- **Data Pipeline**: See `context_data.md` for data ingestion and collection
- **Feature Engineering**: See `context_features.md` for feature computation
- **Stores**: See `context_stores.md` for persistence layer
- **Training**: See `context_training.md` for model training pipelines
- **Registry**: See `context_registry.md` for lifecycle management
- **Strategies**: See `context_strategies.md` for trading strategy framework
- **Deployment**: See `context_deployment.md` for containerization
- **Monitoring**: See `context_monitoring.md` for observability
- **Actors**: See `context_actors.md` for inference actors
- **Models**: See `context_models.md` for model implementations
