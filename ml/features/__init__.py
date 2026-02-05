"""
Nautilus Trader ML Feature Engineering Module.

This module provides comprehensive feature engineering capabilities with guaranteed
parity between batch (training) and real-time (inference) computation paths, following
the Universal ML Architecture Patterns.

## Architecture Overview

The feature engineering system is designed around hot/cold path separation (Pattern 3):

### Hot Path Components (Real-time, <5ms P99 latency)
- `FeatureEngineer.calculate_features_online()`: Zero-allocation feature computation
- `IndicatorManager.update_from_bar()`: Pre-allocated indicator updates
- Hot path uses pre-initialized indicators and reusable buffers

### Cold Path Components (Batch processing, heavy I/O allowed)
- `FeatureEngineer.calculate_features_batch()`: DataFrame-based batch computation
- `FeatureParityValidator`: Validation between hot/cold paths
- `L2MicrostructureFeatures`: Advanced order book feature computation
- `L3TradeFlowFeatures`: Trade flow analysis
- Feature materialization and export utilities

## Core Components

### Feature Engineering
- `FeatureConfig`: Configuration for feature computation
- `FeatureEngineer`: Main feature computation engine with hot/cold path separation
- `IndicatorManager`: Manages Nautilus technical indicators

### Pipeline Framework
- `PipelineSpec`: Declarative feature pipeline specification
- `PipelineRunner`: Compile-time pipeline validation and execution
- `FeatureTransform`: Protocol for pluggable feature transforms

### Validation & Quality
- `FeatureParityValidator`: Validates hot/cold path consistency
- `validate_feature_parity()`: Convenience function for parity validation
- `FeatureParityError`: Exception for parity validation failures

### Advanced Features (Cold Path Only)
- `L2MicrostructureFeatures`: Order book depth and spread analysis
- `L3TradeFlowFeatures`: Trade flow imbalance and impact measures

### Aggregation Utilities
- `L2Aggregator`: Per-minute L2 order book aggregation
- `MicrostructureAggregator`: L1/L2 microstructure aggregation

## Usage Examples

### Hot Path (Real-time Trading Actor)
```python
from ml.features import FeatureConfig, FeatureEngineer, IndicatorManager

# Initialize once at actor startup
config = FeatureConfig(
    enable_returns=True,
    enable_volatility=True,
    return_periods=[1, 5, 10]
)
engineer = FeatureEngineer(config)
indicator_mgr = IndicatorManager(config)

def on_bar(self, bar: Bar) -> None:
    # Update indicators (hot path - pre-allocated)
    indicator_mgr.update_from_bar(bar)

    # Extract bar data
    current_bar = {
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": float(bar.volume)
    }

    # Compute features (hot path - zero allocations)
    features = engineer.calculate_features_online(
        current_bar, indicator_mgr, scaler=None
    )

    # Use features for prediction...
```

### Cold Path (Training Pipeline)
```python
import polars as pl
from ml.features import FeatureConfig, FeatureEngineer, validate_feature_parity

# Load training data
df = pl.read_parquet("market_data.parquet")

# Configure features
config = FeatureConfig(
    enable_returns=True,
    enable_volatility=True,
    enable_momentum=True,
    return_periods=[1, 5, 10, 20]
)

# Batch feature computation
engineer = FeatureEngineer(config)
features_df, scaler = engineer.calculate_features_batch(df, fit_scaler=True)

# Validate hot/cold path parity
report = validate_feature_parity(
    df=df,
    config=config,
    tolerance=1e-10,
    start_idx=50,
    end_idx=200
)
print(f"Parity validation: {'PASSED' if report['parity_passed'] else 'FAILED'}")
```

### Pipeline Framework
```python
from ml.features.pipeline import PipelineSpec, TransformSpec, PipelineRunner
from ml.registry.base import DataRequirements

# Define feature pipeline
spec = PipelineSpec(transforms=[
    TransformSpec(name="returns", params={"periods": [1, 5, 10]}),
    TransformSpec(name="volatility", params={}),
    TransformSpec(name="core_indicators", params={})
])

# Compile and validate pipeline
runner = PipelineRunner(spec, allowable=DataRequirements.L1_ONLY)
feature_names = runner.compute_feature_names()
signature = runner.compute_signature()
```

## Performance Requirements

Hot path components MUST meet these SLA requirements:
- P99 latency < 5ms for `calculate_features_online()`
- Zero memory allocations after warmup
- Pre-allocated arrays for all computations

Cold path components have no performance constraints and may use:
- Heavy I/O operations
- Large DataFrame operations
- Complex mathematical computations
- Model training and evaluation

## Pattern Compliance

This module follows all Universal ML Architecture Patterns:

1. **Pattern 1**: 4-Store + 4-Registry Integration via `BaseMLInferenceActor`
2. **Pattern 2**: Protocol-First Interface Design for feature transforms
3. **Pattern 3**: Hot/Cold Path Separation with strict performance SLAs
4. **Pattern 4**: Progressive Fallback Chains for external dependencies
5. **Pattern 5**: Centralized Metrics Bootstrap for monitoring

"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ml.features.facade import FeatureConfig as FeatureConfig
    from ml.features.facade import FeatureEngineer as FeatureEngineer
    from ml.features.facade import IndicatorManager as IndicatorManager

# Use lazy imports to avoid circular import issues
__all__ = [
    "CorrelationState",
    "EWMABetaState",
    "EarningsCalendarTransformSpec",
    "EarningsGrowthTransformSpec",
    "EarningsMomentumTransformSpec",
    "EarningsSurpriseTransformSpec",
    "FeatureConfig",
    "FeatureEngineer",
    "FeatureParityError",
    "FeatureParityValidator",
    "FeatureTransform",
    "IndicatorManager",
    "L2Aggregator",
    "L2MicrostructureFeatures",
    "L3TradeFlowFeatures",
    "MacroDataCache",
    "MacroFeatureTransform",
    "MicrostructureAggregator",
    "PipelineRunner",
    "PipelineSpec",
    "PipelineStreamContext",
    "PipelineStreamExecutor",
    "TransformSpec",
    "ZScoreSpreadState",
    "aggregate_l2_minute_pl",
    "aggregate_microstructure_minute_pl",
    "compute_correlation_batch",
    "compute_correlation_incremental",
    "compute_ewma_beta_batch",
    "compute_ewma_beta_incremental",
    "compute_zscore_spread_batch",
    "compute_zscore_spread_incremental",
    "create_macro_transform_from_config",
    "register_feature_set_from_engineer",
    "register_transform",
    "validate_feature_parity",
]


def __getattr__(name: str) -> object:
    """
    Lazy import implementation to avoid circular imports.

    FeatureEngineer, FeatureConfig, and IndicatorManager are loaded from the component-
    based facade implementation.

    """
    if name in {"FeatureConfig", "FeatureEngineer", "IndicatorManager"}:
        from ml.features.facade import FeatureConfig
        from ml.features.facade import FeatureEngineer
        from ml.features.facade import IndicatorManager

        mapping = {
            "FeatureConfig": FeatureConfig,
            "FeatureEngineer": FeatureEngineer,
            "IndicatorManager": IndicatorManager,
        }
        return mapping[name]
    elif name == "FeatureTransform":
        from ml.features.pipeline import FeatureTransform

        return FeatureTransform
    elif name == "PipelineSpec":
        from ml.features.pipeline import PipelineSpec

        return PipelineSpec
    elif name == "PipelineRunner":
        from ml.features.pipeline import PipelineRunner

        return PipelineRunner
    elif name == "PipelineStreamContext":
        from ml.features.pipeline_stream import PipelineStreamContext

        return PipelineStreamContext
    elif name == "PipelineStreamExecutor":
        from ml.features.pipeline_stream import PipelineStreamExecutor

        return PipelineStreamExecutor
    elif name == "TransformSpec":
        from ml.features.pipeline import TransformSpec

        return TransformSpec
    elif name == "register_transform":
        from ml.features.pipeline import register_transform

        return register_transform
    elif name == "FeatureParityValidator":
        from ml.features.validation import FeatureParityValidator

        return FeatureParityValidator
    elif name == "FeatureParityError":
        from ml.features.validation import FeatureParityError

        return FeatureParityError
    elif name == "validate_feature_parity":
        from ml.features.validation import validate_feature_parity

        return validate_feature_parity
    elif name == "L2MicrostructureFeatures":
        from ml.features.microstructure import L2MicrostructureFeatures

        return L2MicrostructureFeatures
    elif name == "L3TradeFlowFeatures":
        from ml.features.microstructure import L3TradeFlowFeatures

        return L3TradeFlowFeatures
    elif name == "L2Aggregator":
        from ml.features.l2_aggregate import L2Aggregator

        return L2Aggregator
    elif name == "aggregate_l2_minute_pl":
        from ml.features.l2_aggregate import aggregate_l2_minute_pl

        return aggregate_l2_minute_pl
    elif name == "MicrostructureAggregator":
        from ml.features.micro_aggregate import MicrostructureAggregator

        return MicrostructureAggregator
    elif name == "aggregate_microstructure_minute_pl":
        from ml.features.micro_aggregate import aggregate_microstructure_minute_pl

        return aggregate_microstructure_minute_pl
    elif name == "register_feature_set_from_engineer":
        from ml.features.feature_export import register_feature_set_from_engineer

        return register_feature_set_from_engineer
    elif name == "MacroDataCache":
        from ml.features.macro_cache import MacroDataCache

        return MacroDataCache
    elif name == "MacroFeatureTransform":
        from ml.features.macro_transforms import MacroFeatureTransform

        return MacroFeatureTransform
    elif name == "create_macro_transform_from_config":
        from ml.features.macro_transforms import create_macro_transform_from_config

        return create_macro_transform_from_config
    elif name == "CorrelationState":
        from ml.features.cross_asset import CorrelationState

        return CorrelationState
    elif name == "EWMABetaState":
        from ml.features.cross_asset import EWMABetaState

        return EWMABetaState
    elif name == "EarningsCalendarTransformSpec":
        from ml.features.earnings import EarningsCalendarTransformSpec

        return EarningsCalendarTransformSpec
    elif name == "EarningsGrowthTransformSpec":
        from ml.features.earnings import EarningsGrowthTransformSpec

        return EarningsGrowthTransformSpec
    elif name == "EarningsMomentumTransformSpec":
        from ml.features.earnings import EarningsMomentumTransformSpec

        return EarningsMomentumTransformSpec
    elif name == "EarningsSurpriseTransformSpec":
        from ml.features.earnings import EarningsSurpriseTransformSpec

        return EarningsSurpriseTransformSpec
    elif name == "ZScoreSpreadState":
        from ml.features.cross_asset import ZScoreSpreadState

        return ZScoreSpreadState
    elif name == "compute_correlation_incremental":
        from ml.features.cross_asset import compute_correlation_incremental

        return compute_correlation_incremental
    elif name == "compute_correlation_batch":
        from ml.features.cross_asset import compute_correlation_batch

        return compute_correlation_batch
    elif name == "compute_ewma_beta_incremental":
        from ml.features.cross_asset import compute_ewma_beta_incremental

        return compute_ewma_beta_incremental
    elif name == "compute_ewma_beta_batch":
        from ml.features.cross_asset import compute_ewma_beta_batch

        return compute_ewma_beta_batch
    elif name == "compute_zscore_spread_incremental":
        from ml.features.cross_asset import compute_zscore_spread_incremental

        return compute_zscore_spread_incremental
    elif name == "compute_zscore_spread_batch":
        from ml.features.cross_asset import compute_zscore_spread_batch

        return compute_zscore_spread_batch
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
