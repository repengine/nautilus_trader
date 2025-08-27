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
├── engineering.py           # Core feature engineering classes (2,452 lines)
├── validation.py           # Parity validation system
├── pipeline.py             # Declarative pipeline framework (508 lines)
├── microstructure.py       # L2/L3 microstructure features
├── feature_export.py       # Registry integration utilities
└── materialize_cli.py      # Feature materialization CLI
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
- **State Management**: Price history with `PRICE_HISTORY_MAXLEN` limit (default 1000)
- **Dual Update Methods**:
  - `update_from_bar(bar)`: Updates from Nautilus Bar objects
  - `update_from_values()`: Updates from raw OHLCV values (hot path convenience)
- **Vectorized Batch**: `update_batch_vectorized()` for efficient training data processing

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

- **Returns**: Multiple periods (default: 1, 5, 10, 20 bars)
- **Momentum**: Multiple periods (default: 5, 10, 20 bars)
- **Volatility**: Rolling standard deviation (5, 20 periods)
- **High-Low Spread**: Normalized by close price

### 2. Volume Features

- **Volume Ratios**: Current volume vs moving averages (default: 5, 10, 20 periods)
- **Normalization**: Against configurable MA periods

### 3. Technical Indicators

- **RSI**: Normalized to [-1, 1] range from Nautilus [0, 1] for ML compatibility
- **RSI Thresholds**: Overbought (>70) and oversold (<30) binary indicators
- **Bollinger Bands**: Width (upper-lower)/middle and position within bands
- **ATR**: Normalized by current price
- **EMA Features**:
  - Fast EMA distance from price
  - Slow EMA distance from price
  - EMA cross (fast-slow)/slow
- **MACD**: Line, signal, and difference (all normalized by price)
- **Price Position**: Location within 20-day high-low range

### 4. Microstructure Features (L2/L3)
Advanced order book and trade flow analysis:

#### L2 Order Book Features (when include_microstructure=True)

- **Spread Metrics**:
  - `spread_mean`: Average spread over lookback window
  - `spread_std`: Spread standard deviation
  - `spread_relative`: Relative spread normalized by price
- **Size Imbalance**:
  - `size_imbalance_mean`: Average bid-ask size imbalance
  - `size_imbalance_std`: Imbalance volatility
- **Mid-Price Dynamics**:
  - `mid_return_std`: Mid-price return volatility
  - `mid_return_autocorr`: Mid-price return autocorrelation

#### L3 Trade Flow Features (when include_trade_flow=True)

- **Trade Flow Metrics**:
  - `trade_flow_imbalance`: Buy-sell volume imbalance
  - `vwap`: Volume-weighted average price
  - `trade_intensity`: Trading rate normalized by average
  - `avg_price_impact`: Average price impact per trade

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

The pipeline framework provides a declarative way to define feature transformations with automatic schema computation and data requirements gating.

#### Canonical Source of Feature Names
The ordered feature names and pipeline signature are derived from the declarative pipeline (`PipelineRunner`) built from `FeatureConfig`. To avoid drift:
- `FeatureConfig.get_feature_names()` delegates to a `PipelineRunner`
- `FeatureEngineer.generate_feature_manifest()` uses the same path
- This ensures manifests, stores, and training use identical schemas

#### Pipeline Building
```python
# Helper function builds pipeline from config
def build_pipeline_spec_from_feature_config(cfg: FeatureConfig) -> PipelineSpec:
    transforms = [
        TransformSpec(name="returns", params={"periods": cfg.return_periods}),
        TransformSpec(name="momentum", params={"periods": cfg.momentum_periods}),
        TransformSpec(name="volatility", params={}),
        TransformSpec(name="volume_ratio", params={"periods": cfg.volume_ma_periods}),
        TransformSpec(name="core_indicators", params={}),
    ]
    if cfg.include_microstructure:
        transforms.append(TransformSpec(name="microstructure", params={}))
    if cfg.include_trade_flow:
        transforms.append(TransformSpec(name="trade_flow", params={}))
    return PipelineSpec(transforms=transforms)
```

#### Core Classes
```python
@dataclass
class PipelineSpec:
    transforms: list[TransformSpec]

@dataclass
class TransformSpec:
    name: str
    params: dict[str, Any]
```

### Transform Catalog

Core transforms (always available):
- **returns**: Price returns over configurable periods
- **momentum**: Price momentum indicators
- **volatility**: Rolling volatility calculations
- **volume_ratio**: Volume relative to moving averages
- **core_indicators**: RSI, Bollinger Bands, ATR, EMA, MACD indicators

Advanced transforms (require L1_L2 data):
- **microstructure**: Order book spread, imbalance, and depth features
- **trade_flow**: Trade flow imbalance, VWAP, intensity metrics
- **keltner**: Keltner channel width and position
- **obv**: On-Balance Volume normalized

TFT-specific transforms:
- **calendar**: Time-based features with cyclic/fourier/onehot encoding
- **event_schedule**: Earnings, Fed meetings, economic releases, options expiry
- **macro_indicators**: VIX, DXY, treasury yields, term spread, fed funds rate
- **static_covariates**: Instrument metadata (tick size, lot size, exchange, etc.)

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
- [x] Price returns and momentum features
- [x] Volume ratio features
- [x] Volatility calculations
- [x] Perfect parity validation system
- [x] Hot/cold path separation with zero-allocation
- [x] Pre-allocated feature buffers for online processing
- [x] Comprehensive configuration validation
- [x] Simplified L2/L3 microstructure features for hot path
- [x] Simplified trade flow features for hot path
- [x] Pipeline framework with transform catalog
- [x] Feature manifest generation and registry integration
- [x] Performance validation with latency benchmarks
- [x] Feature quality validation (null rates, outliers, drift)
- [x] Feature materialization CLI
- [x] Convenience API for hot path (direct OHLCV input)

### Advanced Features (Pipeline Transforms) ✅

- [x] Keltner channel transform (registered, gated by L1_L2)
- [x] On-Balance Volume transform (registered, gated by L1_L2)
- [x] Full microstructure transform (registered, gated by L1_L2)
- [x] Full trade flow transform (registered, gated by L1_L2)

### TFT-Specific Features (Pipeline Transforms) ✅

- [x] Calendar features transform (cyclic/fourier/onehot encoding)
- [x] Event schedule features transform (earnings, Fed meetings, options expiry)
- [x] Macro indicators transform (VIX, DXY, treasury yields, fed funds)
- [x] Static covariates transform (instrument metadata)

### Pending Features 🔄

- [ ] Fractional differencing integration with StationarityTransformer
- [ ] Cross-sectional features across multiple instruments
- [ ] Feature selection and importance analysis tools
- [ ] Full L2/L3 data integration with L2MicrostructureFeatures class

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

- **RSI**: Normalized to [-1, 1] from Nautilus [0, 1] using formula: `(rsi - 0.5) * 2.0`
- **MACD**: All components normalized by current price
- **Price Features**: Normalized by current price using safe_divide
- **Volume Features**: Normalized by moving averages
- **Bollinger Bands**: Width normalized by middle band
- **ATR**: Normalized by current price

### 4. Error Handling
Robust error handling for production:

- **Safe Division**: `safe_divide()` helper prevents NaN/Inf propagation
- **Bounds Checking**: Configuration validation in `__post_init__()`
- **Graceful Degradation**: Default values for missing data
- **Runtime Assertions**: RSI normalization bounds checked at runtime

### 5. Internal Implementation Details

Key internal methods in FeatureEngineer:
- `_calculate_return_features()`: Computes returns and momentum
- `_calculate_volatility_features()`: Rolling volatility over 5/20 periods
- `_calculate_indicator_features()`: All technical indicators
- `_calculate_microstructure_features_online()`: Simplified L2 features for hot path
- `_calculate_trade_flow_features_online()`: Simplified L3 features for hot path
- `_extract_price_arrays()`: Handles both Polars and Pandas DataFrames
- `_apply_scaler()`: Fits StandardScaler on training portion only (prevents lookahead)

Helper utilities:
- `build_pipeline_spec_from_feature_config()`: Single source of truth for feature ordering
- `_dummy_context_manager`: Used when metrics collector is not available

## Integration Examples

### Basic Usage

```python
# Configuration
config = FeatureConfig(
    rsi_period=14,
    include_microstructure=True,
    include_trade_flow=False,
    validate_quality=True
)

# Engineer
engineer = FeatureEngineer(config)

# Batch processing (training)
features_df, scaler = engineer.calculate_features(
    df, mode="batch", fit_scaler=True, scaler_fit_ratio=0.7
)

# Online processing (inference) - Option 1: with IndicatorManager
indicator_mgr = IndicatorManager(config)
# ... warm up indicators ...
features = engineer.calculate_features(
    current_bar, mode="online",
    indicator_manager=indicator_mgr,
    scaler=scaler
)

# Online processing (inference) - Option 2: using convenience API
features = engineer.calculate_features_online(
    close_price=100.5,
    high_price=101.0,
    low_price=100.0,
    volume=1000000.0,
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

### Feature Manifest Generation

```python
# Generate feature manifest for registry
manifest = engineer.generate_feature_manifest(
    name="core_technical_features",
    version="1.0.0",
    role=FeatureRole.PRIMARY,
    data_requirements=DataRequirements.L1_ONLY,
    parity_tolerance=1e-10
)

# Register with FeatureRegistry
from ml.registry.feature_registry import FeatureRegistry
registry = FeatureRegistry(Path("ml/registry"))
feature_set_id = registry.register_feature_set(manifest)
```

### Feature Export and Materialization

```python
from ml.features.feature_export import register_feature_set_from_engineer

# Register feature set from engineer configuration
feature_set_id = register_feature_set_from_engineer(
    registry_path=Path("ml/registry"),
    name="production_features",
    version="2.0.0",
    role=FeatureRole.PRIMARY,
    data_requirements=DataRequirements.L1_L2,
    feature_config=config,
    parity_report={"tolerance": 1e-10, "parity_passed": True}
)

# Materialize features to CSV
# Command-line usage:
# python -m ml.features.materialize_cli \
#     --feature_registry_dir ml/registry \
#     --feature_set_id ${feature_set_id} \
#     --input_csv data/features_raw.csv \
#     --output_csv data/features_materialized.csv
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

- **Type Safety**: Complete type annotations with overloads for API flexibility
- **Documentation**: Comprehensive docstrings with examples
- **Testing**: Extensive unit and parity validation tests
- **Linting**: Ruff compliant, formatted with black

### Feature Quality (when validate_quality=True)

The `validate_feature_quality()` method provides comprehensive metrics:
- **Null Rate**: Percentage of NaN values per feature
- **Zero Rate**: Percentage of zero values per feature
- **Unique Ratio**: Ratio of unique values to total rows
- **Inf Rate**: Percentage of infinite values (float features only)
- **Outlier Rate**: IQR-based detection (1.5 * IQR threshold)

Quality validation is performed via:
- `_calculate_feature_qualities()`: Batch quality metrics computation
- `_calculate_column_metrics()`: Per-column analysis
- `_calculate_outlier_rate()`: IQR-based outlier detection

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
