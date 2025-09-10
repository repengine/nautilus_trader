# Context: Config Module

## Overview

The `ml/config/` directory implements a comprehensive configuration system for ML components using msgspec-based structured configuration classes. The module enforces the config-driven development mandate from CLAUDE.md, providing type-safe, validated, hierarchical configurations for all ML actors, training pipelines, and runtime components. All configurations are immutable (frozen=True), support environment variable overrides, and integrate seamlessly with Nautilus Trader's configuration system.

The config module follows a strict hierarchy where base configurations define common patterns, framework-specific configs (XGBoost, LightGBM) extend training configurations, and actor configs compose multiple configuration components. This architecture ensures consistency across the ML pipeline while providing flexibility for specialized use cases.

## Architecture

### Structural Organization

```
ml/config/
├── __init__.py              # Centralized exports and public API
├── base.py                  # Base configuration classes and core ML configs
├── shared.py                # Shared configuration components (GPU, Optuna, advanced training)
├── constants.py             # System constants, time constants, and ML constants
├── actors.py                # Actor-specific configuration classes
├── xgboost.py              # XGBoost training configuration with validation
├── lightgbm.py             # LightGBM training with GOSS, DART, EFB configs
├── registry.py             # Registry-related configuration classes
├── runtime.py              # ONNX Runtime configuration and session helpers
├── adapters.py             # Configuration utilities and protocol-based helpers
├── events.py               # Canonical event constants for ML pipeline stages
├── version.py              # Package version helper utilities
├── loader.py               # Typed configuration loader with layered merge
├── defaults.py             # Default configuration constructors
├── names.py                # Canonical metric names and Prometheus labels
├── scheduler_config.py     # Data scheduler and Databento configuration
├── observability.py        # Observability system configuration with database and file sinks
├── transformers.py         # Configuration for Transformer models (TFT, N-BEATS, CVML)
├── databases.py           # Database-specific configurations for PostgreSQL and TimescaleDB
└── ensemble.py            # Ensemble model configuration with voting and stacking strategies
```

### Configuration Hierarchy

1. **Base Layer**: `NautilusConfig`-derived msgspec structs with frozen=True
2. **Domain Layer**: ML-specific base classes (MLFeatureConfig, MLInferenceConfig, MLTrainingConfig)
3. **Framework Layer**: XGBoost, LightGBM training configurations with framework-specific parameters
4. **Actor Layer**: Composite configurations for ML actors and strategies
5. **Runtime Layer**: ONNX, optimization, and hardware acceleration configurations

## Key Components

### Base Configuration Classes (`base.py`)

#### MLFeatureConfig
Type-safe feature engineering configuration with validation:

- `lookback_window: PositiveInt = 100` - Historical bars for feature computation
- `indicators: dict[str, dict[str, Any]] | None` - Indicator configurations
- `normalize_features: bool = True` - Feature normalization toggle
- `fill_missing_with: float = 0.0` - Missing data imputation value
- `average_volume: PositiveFloat = 1000000.0` - Volume normalization baseline

#### MLInferenceConfig
Comprehensive inference configuration with model loading options:

- Supports both file-based (`model_path`) and registry-based (`model_id`) model loading
- `prediction_threshold: NonNegativeFloat = 0.5` - Confidence threshold for valid predictions
- `max_inference_latency_ms: PositiveFloat = 5.0` - Hot-path latency budget enforcement
- `use_manifest_features: bool = True` - Registry-based feature schema loading
- `use_dummy_stores: bool = False` - Testing vs production store selection

#### MLActorConfig
Enhanced production-ready ML actor configuration:

- `model_id: str` - Required for model tracking and deployment
- `circuit_breaker_config: CircuitBreakerConfig` - Fault tolerance configuration
- `enable_health_monitoring: bool = True` - Health status monitoring toggle
- `max_feature_latency_ms: PositiveFloat = 0.5` - Feature computation latency budget
- `allow_non_onnx_in_dev: bool = False` - Security enforcement for production deployments

#### MLStrategyConfig
Trading strategy configuration extending Nautilus StrategyConfig:

- `ml_signal_source: str` - Actor ID for ML signal consumption
- `position_size_pct: PositiveFloat = 0.1` - Risk management parameter
- `min_confidence: NonNegativeFloat = 0.7` - Signal confidence threshold
- `execute_trades: bool = False` - Production safety flag for signal-only mode

### Framework-Specific Training Configurations

#### XGBoostTrainingConfig (`xgboost.py`)
Comprehensive XGBoost configuration with advanced features:

- Core boosting parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`
- Regularization: `reg_alpha`, `reg_lambda`, `gamma`, `min_child_weight`
- Hardware optimization: `tree_method`, `gpu_id` with GPU validation
- Advanced features: SHAP computation, monotonic constraints, multi-asset support
- Cross-sectional features for multi-asset models with sector mapping

#### LightGBMTrainingConfig (`lightgbm.py`)
Advanced LightGBM configuration with specialized boosting strategies:

- **GOSS Config**: Gradient-based One-Side Sampling for dataset reduction
- **DART Config**: Dropout regularization for tree ensembles
- **EFB Config**: Exclusive Feature Bundling for performance optimization
- Native categorical feature support and memory optimization flags
- GPU acceleration with OpenCL platform configuration

### Shared Components (`shared.py`)

#### OptunaConfig
Hyperparameter optimization configuration:

- `enabled: bool = False` - Toggle for Optuna integration
- `n_trials: int = 100` - Optimization trial budget
- `direction: str = "maximize"` - Optimization direction (maximize/minimize)
- `metric: str = "sharpe_ratio"` - Target metric for optimization
- Database persistence support with study storage

#### GPU Configuration Hierarchy

- `BaseGPUConfig`: Common GPU settings with device validation
- `XGBoostGPUConfig`: XGBoost-specific GPU parameters (max_bin, predictor)
- `LightGBMGPUConfig`: LightGBM-specific OpenCL configuration (platform_id, gpu_use_dp)

#### AdvancedTrainingConfig
Cross-framework advanced training features:

- Feature importance decay tracking with configurable thresholds
- Cross-validation strategies: "time_series", "blocked", "purged", "standard"
- ONNX export configuration with automatic path generation
- Prometheus metrics integration toggle

#### ObservabilityConfig (`observability.py`)
Comprehensive observability system configuration:

- **Database Persistence**: PostgreSQL table configuration for latency watermarks, metrics, correlation data, and health scores
- **File Export Options**: JSONL and CSV output with configurable directory structures
- **Flush Scheduling**: Background persistence with configurable intervals (default 60s)
- **Sink Selection**: Database-first with file backup, or file-only for external pipeline integration
- **Health Monitoring**: Configurable thresholds for component health aggregation and alerting

#### TransformerConfig (`transformers.py`)
Advanced transformer model configurations:

- **TFTConfig**: Temporal Fusion Transformer with attention mechanisms and variable selection
- **NBEATSConfig**: Neural Basis Expansion Analysis for seasonal and trend decomposition
- **CVMLConfig**: Computer Vision for Market Liquidity with CNN architectures for order book analysis
- **AttentionConfig**: Multi-head attention parameters with positional encoding options
- **EncoderDecoderConfig**: Sequence-to-sequence architecture for multi-horizon forecasting

#### DatabaseConfig (`databases.py`)
Database-specific optimization configurations:

- **PostgreSQLConfig**: Connection pooling, timeout settings, and performance tuning parameters
- **TimescaleDBConfig**: Hypertable management, retention policies, and continuous aggregation
- **PartitionConfig**: Time-based partitioning strategies with automatic maintenance
- **IndexConfig**: BRIN, BTREE, and GIN index configuration for ML table optimization
- **ReplicationConfig**: High-availability setup with read replica configuration

#### EnsembleConfig (`ensemble.py`)
Multi-model ensemble configuration:

- **VotingConfig**: Hard and soft voting strategies with weight optimization
- **StackingConfig**: Meta-learner configuration with cross-validation integration
- **BaggingConfig**: Bootstrap aggregation with parallel training support
- **BlendingConfig**: Linear combination strategies with regularization
- **DiversityConfig**: Model diversity enforcement through parameter constraints

### Actor Configurations (`actors.py`)

#### MLSignalActorConfig
Unified signal actor configuration with strategy composition:

- `signal_strategy: Literal["threshold", "extremes", "momentum", "ensemble", "adaptive"]`
- `optimization_config: OptimizationConfig` - Performance optimization settings
- `strategy_config: StrategyConfig` - Strategy-specific parameters
- Registry integration: `feature_set_id`, `use_registry_features`
- Backward compatibility mapping for legacy test configurations

### Runtime and System Constants

#### Constants Module (`constants.py`)
Centralized constants following the config-driven development mandate:

- `TimeConstants`: Nanosecond conversions and trading calendar constants
- `TechnicalIndicatorPeriods`: Standard periods for technical indicators
- `MLConstants`: Data windows, confidence thresholds, feature engineering limits
- `FeatureColumns`: Standard column names as enums for type safety
- `IndicatorNames`: Naming patterns for technical indicators

#### Runtime Configuration (`runtime.py`)
ONNX Runtime optimization configuration:

- `OnnxRuntimeConfig`: Graph optimization levels, execution modes, provider selection
- `to_session_options()`: Helper function converting config to ONNX SessionOptions
- Provider management: CPU/CUDA execution provider selection

### Configuration Loading and Validation

#### Loader Module (`loader.py`)
Layered configuration loading with priority hierarchy:

1. Code defaults (lowest priority)
2. File-based configuration (YAML/JSON)
3. Environment variables (ML_ prefix)
4. CLI overrides (highest priority)

Environment integration supports JSON blob loading via `{PREFIX}_JSON` variables.

## Dependencies

### Internal Dependencies

- **ml.common.protocols**: Universal ML Component Protocol compliance
- **ml._imports**: Lazy loading of ML frameworks with availability checks
- **nautilus_trader.common.config**: Base configuration classes and validation types
- **nautilus_trader.model**: Nautilus data types (InstrumentId, BarType, ComponentId)

### External Dependencies

- **msgspec**: High-performance serialization with schema validation
- **ML Framework Libraries**: XGBoost, LightGBM (lazy-loaded via ml._imports)
- **onnxruntime**: ONNX model inference (lazy-loaded)
- **optuna**: Hyperparameter optimization (lazy-loaded)

## Usage Patterns

### Configuration Creation

```python
from ml.config import XGBoostTrainingConfig, OptunaConfig, AdvancedTrainingConfig, ObservabilityConfig

# Framework-specific training configuration
training_config = XGBoostTrainingConfig(
    data_source="historical_data.parquet",
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    enable_shap=True,
    optuna_config=OptunaConfig(
        enabled=True,
        n_trials=50,
        metric="sharpe_ratio"
    ),
    advanced_config=AdvancedTrainingConfig(
        cv_strategy="purged",
        export_onnx=True
    )
)

# Observability configuration with database sink
observability_config = ObservabilityConfig(
    enable_db_sink=True,
    enable_file_sink=True,
    flush_interval_seconds=60.0,
    file_format="jsonl",
    db_connection_string="postgresql://user:pass@localhost/nautilus"
)
```

### Actor Configuration

```python
from ml.config import MLSignalActorConfig, OptimizationConfig, MLFeatureConfig

# Signal actor with complete configuration
actor_config = MLSignalActorConfig(
    model_path="models/xgb_model.onnx",
    model_id="xgb_signal_v1",
    bar_type=bar_type,
    instrument_id=instrument_id,
    signal_strategy="adaptive",
    feature_config=MLFeatureConfig(
        lookback_window=200,
        normalize_features=True
    ),
    optimization_config=OptimizationConfig(
        level="optimized",
        enable_zero_copy=True
    )
)
```

### Environment Override

```bash
export ML_JSON='{"prediction_threshold": 0.8, "max_inference_latency_ms": 3.0}'
```

## Integration Points

### Nautilus Trader Integration

- Extends `NautilusConfig` for seamless integration with Nautilus configuration system
- Uses Nautilus validation types: `PositiveInt`, `NonNegativeFloat`, `PositiveFloat`
- Integrates with `InstrumentId`, `BarType`, and `ComponentId` from Nautilus model types
- Supports Nautilus actor lifecycle through `ActorConfig` mapping utilities

### ML Pipeline Integration

- **Registry System**: Configuration validation against registry schemas
- **Store Systems**: Database connection configuration for persistence layers
- **Actor System**: BaseMLInferenceActor automatic configuration consumption
- **Training Pipeline**: Framework-agnostic configuration interfaces

### Monitoring Integration

- **Prometheus Metrics**: Standardized metric names and labels via `names.py`
- **Health Monitoring**: Configuration-driven health check thresholds
- **Performance Tracking**: Latency budgets and SLA configuration

### Data Pipeline Integration

- **Databento Configuration**: Schema, dataset, and collection configuration
- **Scheduler Configuration**: Symbol universe and collection timing
- **Feature Store**: Database connection and pipeline specification
- **Observability Pipeline**: Database sink configuration and background persistence scheduling

## Implementation Notes

### Configuration Validation
All configuration classes implement `__post_init__()` methods for comprehensive validation:

- Range validation for numeric parameters
- Enum validation for categorical parameters
- Cross-parameter constraint validation
- Framework availability checking
- Hardware capability validation (GPU, OpenCL)

### Immutability and Safety

- All configurations use `frozen=True` for immutability after construction
- Type annotations are mandatory for all fields
- Default values are provided for optional parameters
- Legacy alias support maintains backward compatibility

### Environment Integration

- Structured environment variable loading via `loader.py`
- JSON blob support for complex configuration overrides
- Layered priority system for configuration sources
- Development vs production configuration separation

### Framework Integration

- Lazy loading of ML frameworks via `ml._imports`
- Framework-specific parameter validation
- Hardware capability detection and fallback
- Export format standardization (ONNX, native formats)

### Error Handling

- Comprehensive validation error messages
- Framework availability warnings
- Hardware compatibility checking
- Graceful degradation for missing dependencies

The config module serves as the foundation for all ML component configuration in Nautilus Trader, ensuring type safety, validation, and consistency across the entire ML pipeline while maintaining flexibility for specialized use cases and framework-specific optimizations.
