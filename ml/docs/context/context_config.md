# Context: Config Module

## Overview

The `ml/config/` directory implements a msgspec-based configuration system for ML components in Nautilus Trader. The module provides type-safe, immutable configuration classes for actors, training pipelines, and runtime components, with partial support for environment variable overrides and integration with Nautilus Trader's configuration system.

**Current Implementation Status**: **~75% Complete** - Core configuration classes are implemented with comprehensive validation, but several advanced features documented elsewhere are not yet fully realized. Environment override support is limited to 3 out of 12 major configuration classes.

The config module follows a hierarchical structure where base configurations define common patterns, framework-specific configs extend training configurations, and specialized configs handle runtime concerns. While the architecture supports the intended 4-store + 4-registry integration pattern, enforcement mechanisms are not fully implemented.

## Architecture

### Actual File Structure

```
ml/config/
├── __init__.py              # Public API with 28 exported classes and utilities
├── base.py                  # Base ML configurations (568 lines)
├── actors.py                # Actor-specific configurations with backward compatibility
├── shared.py                # GPU configs, Optuna, and advanced training features
├── constants.py             # System constants and enums
├── events.py                # Event enums (Stage, Source, EventStatus)
├── xgboost.py              # XGBoost training configuration with validation
├── lightgbm.py             # LightGBM with GOSS, DART, EFB specialized configs
├── registry.py             # Registry configuration classes
├── runtime.py              # ONNX Runtime configuration
├── loader.py               # Configuration loading utilities (74 lines)
├── observability.py        # Observability configuration with from_env() support
├── bus.py                  # Message bus configuration with environment parsing
├── actor_bus.py            # Actor-side message bus configuration
├── scheduler_config.py     # Data scheduler and Databento configuration
├── adapters.py             # Configuration utilities (115 lines)
├── defaults.py             # Configuration defaults
├── names.py                # Name constants and mappings
├── universes.py            # Symbol universe configurations
└── version.py              # Version constants
```

### Configuration Categories

- **Core Configs**: 8 base classes in `base.py` covering features, inference, actors, strategies
- **Training Configs**: Framework-specific XGBoost and LightGBM configurations with GPU support
- **Runtime Configs**: ONNX runtime, observability, message bus configurations
- **Specialized Configs**: Event enums, registry policies, scheduler configurations
- **Utility Functions**: Configuration loading, validation, and environment integration

### Configuration Hierarchy

1. **Base Layer**: `NautilusConfig`-derived msgspec structs with `frozen=True` for immutability
2. **Core ML Layer**: Base classes (`MLFeatureConfig`, `MLInferenceConfig`, `MLTrainingConfig`, `MLActorConfig`)
3. **Framework Layer**: XGBoost and LightGBM training configurations with specialized features
4. **Actor Layer**: `MLSignalActorConfig` with optimization and strategy components
5. **Runtime Layer**: ONNX runtime, observability, message bus configurations
6. **Environment Layer**: **LIMITED** - Only `ObservabilityConfig`, `MessageBusConfig`, and `DataCollectorConfig` support `from_env()` methods

## Key Components (Current Implementation)

### Base Configuration Classes (`base.py`)

#### MLFeatureConfig
**Status**: ✅ **Fully Implemented** - Complete feature engineering configuration:

- `lookback_window: PositiveInt = 100` - Historical bars for feature computation
- `indicators: dict[str, dict[str, Any]] | None` - Indicator configurations
- `feature_names: list[str] | None` - Explicit feature list (auto-computed if None)
- `normalize_features: bool = True` - Feature normalization toggle
- `fill_missing_with: float = 0.0` - Missing data imputation value
- `average_volume: PositiveFloat = 1000000.0` - Volume normalization baseline

#### MLInferenceConfig
**Status**: ✅ **Fully Implemented** - Comprehensive inference configuration with dual model loading:

- **Model Loading**: Supports file-based (`model_path`) OR registry-based (`model_id`) model loading
- `prediction_threshold: NonNegativeFloat = 0.5` - Confidence threshold for valid predictions
- `max_inference_latency_ms: PositiveFloat = 5.0` - Hot-path latency budget enforcement
- `batch_size: PositiveInt = 1` - Batch size for model inference
- `warm_up_period: NonNegativeInt = 50` - Bars before predictions start
- `use_manifest_features: bool = True` - Registry-based feature schema loading
- `use_dummy_stores: bool = False` - Testing vs production store selection
- **Validation**: Complete `__post_init__()` with mutual exclusion checks

#### MLActorConfig
**Status**: ⚠️ **Partially Implemented** - Core ML actor configuration with missing components:

- **Model Management**: `model_path: str`, `model_id: str` (required for tracking)
- **Performance**: `max_inference_latency_ms: PositiveFloat = 5.0`, `max_feature_latency_ms: PositiveFloat = 0.5`
- **Health Monitoring**: `enable_health_monitoring: bool = True`, `health_config: HealthMonitorConfig`
- **Circuit Breaker**: `circuit_breaker_config: CircuitBreakerConfig` for fault tolerance
- **Hot Reload**: `enable_hot_reload: bool = False`, `model_check_interval: PositiveInt = 300`
- **Security**: `allow_non_onnx_in_dev: bool = False` (ONNX-only enforcement in production)
- **Store Integration**: `db_connection: str | None`, `use_dummy_stores: bool = False`
- **⚠️ Missing**: `HealthMonitorConfig` referenced but not defined, no validation enforcement

#### MLStrategyConfig
**Status**: ✅ **Fully Implemented** - Trading strategy configuration extending Nautilus StrategyConfig:

- **Signal Source**: `ml_signal_source: str` - Actor ID for ML signal consumption
- **Risk Management**: `position_size_pct: PositiveFloat = 0.1`, `max_positions: PositiveInt = 1`
- **Confidence**: `min_confidence: NonNegativeFloat = 0.7` - Signal confidence threshold
- **Risk Controls**: `stop_loss_pct: NonNegativeFloat = 0.02`, `take_profit_pct: NonNegativeFloat = 0.04`
- **Production Safety**: `execute_trades: bool = False` - Signal-only mode by default
- **Store Integration**: `use_strategy_store: bool = True`, `persist_all_signals: bool = False`

#### MultiModelStrategyConfig
**Status**: ✅ **Fully Implemented** - Configuration for strategies consuming multiple models:

- `target_model_ids: list[str]` - List of model IDs to consume
- `aggregation_mode: Literal["voting", "weighted_average", "best"]` - Signal aggregation method
- `model_weights: dict[str, float] | None` - Weights for weighted_average mode
- `required_models: PositiveInt = 1` - Minimum models before trading

#### ModelDeploymentConfig & CanaryDeploymentConfig
**Status**: ✅ **Fully Implemented** - Advanced deployment configurations with validation:

- **Deployment**: `deployment_target`, `rollout_strategy`, `rollout_percentage`
- **Canary**: `initial_traffic_percentage`, `auto_promote`, `error_threshold_percentage`
- **Health Checks**: `health_check_interval`, `auto_rollback_on_error`
- **Validation**: Complete percentage range validation in `__post_init__()`

### Framework-Specific Training Configurations

#### XGBoostTrainingConfig (`xgboost.py`)
**Status**: ✅ **Fully Implemented** - Comprehensive XGBoost configuration extending MLTrainingConfig:

**Core Parameters:**

- `n_estimators: PositiveInt = 100`, `max_depth: PositiveInt = 6`, `learning_rate: PositiveFloat = 0.3`
- `subsample: PositiveFloat = 1.0`, `colsample_bytree: PositiveFloat = 1.0`, `colsample_bylevel: PositiveFloat = 1.0`

**Regularization:**

- `reg_alpha: NonNegativeFloat = 0.0`, `reg_lambda: NonNegativeFloat = 1.0`, `gamma: NonNegativeFloat = 0.0`
- `min_child_weight: NonNegativeFloat = 1.0`

**Hardware & Objectives:**

- `tree_method: str = "hist"` (GPU via "gpu_hist"), `objective: str = "binary:logistic"`, `eval_metric: str = "auc"`
- ✅ GPU validation and environment checking via `validate_environment()`

**Advanced Features:**

- `enable_shap: bool = False` - SHAP value computation for interpretability
- `monotonic_constraints: dict[str, int] | None` - Feature monotonicity constraints
- `multi_asset: bool = False` - Multi-asset training with cross-sectional features
- `sector_map: dict[str, str] | None` - Asset-to-sector mapping for multi-asset models

**Integration Components:**

- `gpu_config: XGBoostGPUConfig | None` - GPU acceleration settings
- `optuna_config: OptunaConfig | None` - Hyperparameter optimization
- `advanced_config: AdvancedTrainingConfig | None` - Cross-validation and ONNX export

#### LightGBMTrainingConfig (`lightgbm.py`)
**Status**: ✅ **Fully Implemented** - Advanced LightGBM configuration with specialized boosting strategies:

**Core Parameters:**

- `n_estimators: PositiveInt = 100`, `max_depth: PositiveInt = 6`, `learning_rate: PositiveFloat = 0.1`
- `num_leaves: PositiveInt = 31`, `min_child_samples: PositiveInt = 20`, `min_child_weight: NonNegativeFloat = 1e-3`

**Advanced Boosting Strategies:**

- **GOSSConfig**: ✅ `enabled: bool`, `top_rate: float = 0.2`, `other_rate: float = 0.1` - Gradient-based One-Side Sampling with validation
- **DARTConfig**: ✅ `enabled: bool`, `drop_rate: float = 0.1`, `max_drop: int = 50` - Dropout regularization for trees with validation
- **EFBConfig**: ✅ `enabled: bool = True`, `max_conflict_rate: float = 0.0` - Exclusive Feature Bundling optimization

**Memory & Performance:**

- `force_col_wise: bool = False`, `force_row_wise: bool = False` - Memory layout optimization
- `categorical_features: list[str] = []` - Native categorical feature support

**Hardware Integration:**

- `gpu_config: LightGBMGPUConfig | None` - OpenCL GPU acceleration with platform selection
- ✅ Environment validation for GPU availability

### Shared Components (`shared.py`)

#### OptunaConfig
**Status**: ✅ **Fully Implemented** - Comprehensive hyperparameter optimization configuration:

- `enabled: bool = False`, `n_trials: int = 100`, `direction: str = "maximize"`
- `metric: str = "sharpe_ratio"` - Target metric (sharpe_ratio, accuracy, auc, rmse, mae, r2)
- `pruner: str = "median"` - Pruning algorithm (median, percentile, hyperband, none)
- `sampler: str = "tpe"` - Sampling algorithm (tpe, random, cmaes, grid)
- `timeout: int | None = None`, `study_name: str | None`, `storage_url: str | None` - Persistence options
- ✅ Complete validation with enum checking in `__post_init__()`

#### GPU Configuration Hierarchy

**BaseGPUConfig**: ✅ **Fully Implemented** - Foundation for GPU acceleration

- `enabled: bool = False`, `device_id: int = 0`, `validate_gpu: bool = True`

**XGBoostGPUConfig**: ✅ **Fully Implemented** - XGBoost-specific GPU parameters

- `max_bin: int = 256` - Histogram construction bins
- `predictor: str = "gpu_predictor"` - GPU inference predictor type

**LightGBMGPUConfig**: ✅ **Fully Implemented** - LightGBM OpenCL configuration

- `platform_id: int = -1` - OpenCL platform (-1 = auto-detect)
- `gpu_use_dp: bool = False` - Double precision GPU math

#### AdvancedTrainingConfig
**Status**: ✅ **Fully Implemented** - Cross-framework advanced training features:

- **Feature Monitoring**: `track_feature_decay: bool = True`, `feature_decay_threshold: float = 0.3`
- **Cross-Validation**: `cv_strategy: str = "time_series"` (time_series, blocked, purged, standard)
- `cv_folds: int = 5`, `purge_gap: int = 10` - CV configuration
- **ONNX Export**: `export_onnx: bool = False`, `onnx_output_path: str | None`
- **Monitoring**: `enable_monitoring: bool = True` - Prometheus metrics integration

### Runtime and System Configurations

#### ObservabilityConfig (`observability.py`)
**Status**: ✅ **Fully Implemented** - Off hot-path observability configuration with environment integration:

- **Sink Selection**: `sink: Literal["file", "db"] = "file"` - Output destination
- **File Options**: `base_path: str = "./observability"`, `file_format: str = "jsonl"`
- **Database**: `db_connection_string: str | None` - SQLAlchemy connection URL
- **Scheduling**: `interval_seconds: PositiveFloat = 60.0` - Background flush interval
- **Environment Integration**: ✅ Complete `from_env()` class method with ML_OBS_* environment variables
- **Async Options**: `async_enabled: bool = False`, `async_queue_maxsize: int = 4096`

#### OnnxRuntimeConfig (`runtime.py`)
**Status**: ⚠️ **Placeholder Implementation** - ONNX Runtime configuration:

- **⚠️ Current**: Empty placeholder class for backward compatibility
- **⚠️ Missing**: All documented fields (graph_optimization_level, execution_mode, providers, etc.)
- **⚠️ Missing**: `to_session_options()` helper method

#### MessageBusConfig (`bus.py`)
**Status**: ✅ **Fully Implemented** - Optional ML message bus configuration with environment parsing:

- **Control**: `enabled: bool = False`, `backend: BusBackend = "noop"` (noop, redis)
- **Topics**: `scheme: TopicScheme = "domain_op"` (domain_op, stage_first), `topic_prefix: str`
- **Redis**: `redis_url: str`, `redis_stream: str`, `redis_maxlen: int | None`
- **Environment**: ✅ Complete `from_env()` with ML_BUS_* environment variables

#### DataCollectorConfig (`base.py`)
**Status**: ✅ **Fully Implemented** - Enhanced data collector configuration:

- `data_dir: str = "./data/tier1"`, `storage_limit_gb: PositiveFloat = 500.0`
- `end_date_iso: str | None` - Optional collection end date
- **Environment Mapping**: ✅ Complete ML_DATA_TIER1_DIR, ML_STORAGE_LIMIT_GB with legacy fallbacks

### Data Pipeline Configurations

#### SchedulerConfig (`scheduler_config.py`)
Data scheduler and Databento collection configuration:

**Core Settings:**

- `symbols: list[str]` - Default tier-1 symbols (SPY, QQQ, IWM, AAPL, etc.)
- `collection_time: str = "04:00"` - Daily collection time (24-hour format)
- `retention_days: int = 90` - Historical data retention period

**Data Sources:**

- `databento: DatabentoConfig` - Databento-specific configuration
- `enable_l2_depth: bool = False`, `enable_trades: bool = False`, `enable_quotes: bool = False`

**Reliability:**

- `max_retries: int = 3`, `retry_delay_seconds: float = 5.0`
- `feature_store_enabled: bool = True`, `feature_store_connection: str | None`

#### DatabentoConfig
Databento API configuration:

- `dataset: str = "GLBX.MDP3"` - Dataset identifier
- `schema: str = "ohlcv-1m"` - Data schema (ohlcv-1m, trades, mbp-1)
- `stype_in: str = "raw_symbol"` - Symbol type for input
- `use_temporary_files: bool = True`, `temp_data_dir: str`
- `price_precision: int | None`, `api_key: str | None`

#### UniverseConfig
Symbol universe management:

- `priority_symbols: list[str]` - High-priority symbols for deep data
- `sector_etfs: list[str]` - Sector ETF symbols (XLF, XLK, XLE, etc.)
- `volatility_symbols: list[str]` - Volatility instruments (VXX, UVXY, SVXY)
- `commodity_symbols: list[str]` - Commodities and bonds (TLT, GLD, SLV, USO)
- `expansion_mode: Literal["conservative", "moderate", "aggressive"]` - Universe expansion strategy

### Actor Configurations (`actors.py`)

#### MLSignalActorConfig
Unified ML signal actor configuration extending MLActorConfig:

**Signal Strategy:**

- `signal_strategy: Literal["threshold", "extremes", "momentum", "ensemble", "adaptive"] = "threshold"`
- `signal_policy: Literal["threshold", "extremes", "momentum", "ensemble", "adaptive"] | None = None` (alias)
- `adaptive_window: PositiveInt = 20`, `min_signal_separation_bars: PositiveInt = 3`
- `feature_importance_threshold: NonNegativeFloat = 0.01`
- `enable_regime_detection: bool = True`

**Performance Optimization:**

- `optimization_config: OptimizationConfig | None` - Performance settings
- `level: Literal["standard", "optimized"]`, `enable_zero_copy: bool`
- `pre_allocate_buffers: bool = True`, `reservoir_sample_size: PositiveInt = 1000`

**Strategy Configuration:**

- `strategy_config: StrategyConfig | None` - Strategy-specific parameters
- `extremes_top_pct: float = 0.1`, `momentum_lookback: PositiveInt = 5`
- `ensemble_weights: dict[str, float] | None`, `adaptive_volatility_factor: float = 2.0`

**Registry Integration:**

- `feature_set_id: str | None`, `registry_path: str | None`, `use_registry_features: bool = False`
- `use_feature_store: bool = False`, `persist_features: bool = True`

**Hot Reload & Runtime:**

- `enable_hot_reload: bool = False`, `hot_reload_interval: PositiveInt = 300`
- `onnx_runtime_config: OnnxRuntimeConfig | None`

**Backward Compatibility:**

- Legacy alias mapping in `__post_init__()`: `optimization` → `optimization_config`, `strategy` → `strategy_config`
- Conservative threshold merging for legacy tests

### Runtime and System Constants

#### Constants Module (`constants.py`)
Centralized constants following the config-driven development mandate:

**Export Formats & Versions:**

- `ExportFormats`: ONNX, LIGHTGBM, XGBOOST, PICKLE, JSON with file extensions
- `Versions`: ONNX_OPSET=17, framework minimum versions, default manifest versions
- `Providers`: CPU/CUDA execution providers for ONNX Runtime

**Time & Trading Constants:**

- `TimeConstants`: Nanosecond conversions (NS_IN_SECOND=1B), trading calendar (252 trading days/year)
- `TechnicalIndicatorPeriods`: Standard periods (MA_FAST=5, RSI_DEFAULT=14, BB_DEFAULT=20)

**ML & System Constants:**

- `MLConstants`: Data windows (DEFAULT_LOOKBACK_DAYS=252), confidence thresholds, latency budgets (MAX_INFERENCE_LATENCY_MS=5.0)
- `SystemConstants`: Queue sizes, buffer limits, memory safety pads for hot path
- `FeatureColumns`: Enum-based column names for type safety (OPEN, HIGH, LOW, CLOSE, VOLUME, RETURNS)
- `IndicatorNames`: Standardized naming patterns with helper methods (`lag_feature()`, `feature_index()`)

### Configuration Loading and Validation

#### Loader Module (`loader.py`)
Typed configuration loader with layered merge system:

**Priority Hierarchy:**

1. **Code defaults** (lowest priority) - Default values in configuration classes
2. **File-based configuration** (YAML/JSON) - `load_from_file(path, type, default)`
3. **Environment variables** (ML_ prefix) - `merge_env(prefix, type, base)`
4. **CLI overrides** (highest priority) - Call-site parameter overrides

**Environment Integration:**

- JSON blob support: `{PREFIX}_JSON` environment variables
- Best-effort shallow merge for msgspec structs
- Type-safe decoding with fallbacks to defaults

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

## Usage Patterns (Current Implementation)

### Configuration Creation Examples

#### Framework Training Configuration

```python
from ml.config import XGBoostTrainingConfig, OptunaConfig, AdvancedTrainingConfig

# ✅ XGBoost with advanced features (FULLY IMPLEMENTED)
training_config = XGBoostTrainingConfig(
    data_source="data/historical_features.parquet",
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    enable_shap=True,
    multi_asset=True,
    sector_map={"SPY": "broad_market", "XLF": "financials"},
    optuna_config=OptunaConfig(
        enabled=True,
        n_trials=50,
        metric="sharpe_ratio",
        direction="maximize"
    ),
    advanced_config=AdvancedTrainingConfig(
        cv_strategy="time_series",  # ✅ Available: time_series, blocked, purged, standard
        cv_folds=5,
        export_onnx=True,
        track_feature_decay=True
    )
)
```

#### Production ML Actor Configuration

```python
from ml.config import MLSignalActorConfig, OptimizationConfig, StrategyConfig, MLFeatureConfig

# ⚠️ Production actor config (PARTIALLY IMPLEMENTED)
actor_config = MLSignalActorConfig(
    model_path="models/xgb_model.onnx",
    model_id="xgb_signal_v2.1",
    bar_type=bar_type,
    instrument_id=instrument_id,
    signal_strategy="adaptive",   # ✅ Available: threshold, extremes, momentum, ensemble, adaptive
    # Performance optimization (✅ IMPLEMENTED)
    optimization_config=OptimizationConfig(
        level="optimized",
        enable_zero_copy=True,
        pre_allocate_buffers=True
    ),
    # Strategy configuration (✅ IMPLEMENTED)
    strategy_config=StrategyConfig(
        adaptive_volatility_factor=2.0,
        min_threshold=0.1,
        max_threshold=0.95
    ),
    # Feature configuration (✅ IMPLEMENTED)
    feature_config=MLFeatureConfig(
        lookback_window=200,
        normalize_features=True,
        feature_names=["price_sma_5", "volume_sma_20", "rsi"]
    ),
    # Store integration fields present but no validation enforcement
    db_connection="postgresql://user:pass@localhost:5432/nautilus",
    use_dummy_stores=False,
    # Health monitoring (⚠️ HealthMonitorConfig referenced but not defined)
    enable_health_monitoring=True,
    # health_config=HealthMonitorConfig(...)  # ❌ NOT IMPLEMENTED
)
```

#### Environment-Based Configuration

```python
from ml.config import ObservabilityConfig, MessageBusConfig

# ✅ Environment-driven configuration (LIMITED SUPPORT)
obs_config = ObservabilityConfig.from_env()  # ✅ Uses 8 ML_OBS_* environment variables
bus_config = MessageBusConfig.from_env()     # ✅ Uses 7 ML_BUS_* environment variables

# ❌ NO ENVIRONMENT SUPPORT for major configs:
# MLActorConfig.from_env()         # Not implemented
# MLFeatureConfig.from_env()       # Not implemented
# XGBoostTrainingConfig.from_env() # Not implemented
```

### Environment Variable Patterns

#### JSON Override Pattern

```bash
# Complex configuration via JSON
export ML_JSON='{"prediction_threshold": 0.8, "max_inference_latency_ms": 3.0, "enable_hot_reload": true}'

# Observability configuration
export ML_OBS_SINK="db"
export ML_OBS_DB_URL="postgresql://prod:secret@db.example.com:5432/nautilus"
export ML_OBS_INTERVAL_SECONDS="30.0"
# Optional async worker (off hot-path)
export ML_OBS_ASYNC_ENABLE="true"
export ML_OBS_ASYNC_QUEUE_MAX="8192"
export ML_OBS_ASYNC_COMPONENT="obs_async_worker"

# Message bus configuration
export ML_BUS_ENABLE="true"
export ML_BUS_BACKEND="redis"
export ML_BUS_REDIS_URL="redis://redis.example.com:6379/0"
```

#### Data Collection Configuration

```bash
# Data tier configuration
export ML_DATA_TIER1_DIR="/mnt/fast-ssd/tier1"
export ML_STORAGE_LIMIT_GB="1000.0"
export ML_END_DATE="2024-12-31"
```

## Integration Points

### Nautilus Trader Integration

- **Base Types**: Extends `NautilusConfig` for seamless integration with Nautilus configuration system
- **Validation**: Uses Nautilus validation types (`PositiveInt`, `NonNegativeFloat`, `PositiveFloat`)
- **Model Types**: Integrates with `InstrumentId`, `BarType`, and `ComponentId` from Nautilus model types
- **Actor Lifecycle**: `ActorConfig` mapping utilities via `adapters.py`

### 4-Store + 4-Registry Integration (Mandatory Pattern)

**MANDATORY Integration**: All ML actors MUST use the 4-store + 4-registry pattern via BaseMLInferenceActor:

**4 Stores (Data Persistence):**

1. **FeatureStore**: Feature values for training/inference parity and model validation
2. **ModelStore**: Predictions, performance metrics, and model tracking
3. **StrategyStore**: Strategy decisions, position tracking, and trade analysis
4. **DataStore**: Unified facade with contract validation and event emission

**4 Registries (Metadata Management):**

1. **FeatureRegistry**: Feature schema validation, versioning, and lifecycle management
2. **ModelRegistry**: Model deployment tracking, A/B testing, and version management
3. **StrategyRegistry**: Strategy compatibility validation and requirement checking
4. **DataRegistry**: Dataset manifest management and data lineage tracking

**Configuration Integration:**

- `db_connection: str | None` - Database connection for store initialization
- `use_dummy_stores: bool = False` - Testing vs production mode selection
- Progressive fallback: PostgreSQL → DummyStore (warnings logged)
- Protocol-based interfaces for type safety without implementation coupling

### Environment Configuration Patterns

**Environment Override Systems:**

- **Class Methods**: `ObservabilityConfig.from_env()`, `MessageBusConfig.from_env()`
- **Environment Mapping**: `_ENV_MAPPING` class variables with field-to-variable mapping
- **Legacy Support**: `_LEGACY_ENV_MAPPING` for backward compatibility
- **JSON Blobs**: `{PREFIX}_JSON` variables for complex configuration overrides

**Production Deployment Patterns:**

- Environment-specific configuration layering
- Database connection management with fallbacks
- Security enforcement (ONNX-only in production)
- Hot-reload capabilities with preservation of state

### Framework Integration Patterns

**Lazy Loading**: Via `ml._imports` with availability checks (`HAS_XGBOOST`, `HAS_LIGHTGBM`)

- Framework-specific parameter validation
- Hardware capability detection and fallback
- Export format standardization (ONNX preference for production)

**Hot Path Performance**:

- Latency budgets enforced via configuration (`max_inference_latency_ms`)
- Pre-allocated buffers and zero-copy optimizations
- Circuit breaker patterns for fault tolerance

## Implementation Notes

### Configuration Validation and Safety

**Comprehensive Validation**: All configuration classes implement `__post_init__()` methods with:

- Range validation for numeric parameters (e.g., `0.0 < learning_rate <= 1.0`)
- Enum validation for categorical parameters
- Cross-parameter constraint validation (e.g., `num_leaves < 2^max_depth`)
- Framework availability checking via `ml._imports`
- Hardware capability validation (GPU, OpenCL) via `validate_environment()`

**Immutability Patterns**:

- All configurations use `frozen=True` for immutability after construction
- Type annotations are mandatory for all fields (`PositiveInt`, `NonNegativeFloat`)
- Default values provided for optional parameters
- Legacy alias support with backward compatibility mappings in `__post_init__()`

### Production Deployment Considerations

**Security Enforcement**:

- `allow_non_onnx_in_dev: bool = False` - Enforces ONNX-only models in production
- Database connection validation with fallback to DummyStore
- Environment variable sanitization and type checking

**Performance Optimization**:

- Pre-allocated buffers for hot path: `pre_allocate_buffers: bool = True`
- Latency budget enforcement: `max_inference_latency_ms`, `max_feature_latency_ms`
- Zero-copy optimizations: `enable_zero_copy: bool`
- Circuit breaker patterns for fault tolerance

**Environment Configuration Strategies**:

- Progressive environment override system with priority layers
- `from_env()` class methods for environment-driven configuration
- JSON blob support for complex overrides: `{PREFIX}_JSON`
- Legacy environment variable support with graceful migration

### Framework Integration and Validation

**Lazy Loading Architecture**:

- ML frameworks loaded via `ml._imports` with `HAS_*` availability flags
- Framework-specific parameter validation with sensible defaults
- Hardware capability detection with automatic fallback strategies
- Export format preferences (ONNX for production, native for development)

**Error Handling and Diagnostics**:

- Descriptive validation error messages with parameter context
- Framework availability warnings with installation instructions
- Hardware compatibility checking with fallback recommendations
- Graceful degradation for missing dependencies with functional alternatives

### 4-Store + 4-Registry Integration Requirements

**Mandatory Store Integration**: All ML actors extending BaseMLInferenceActor automatically initialize:

- Protocol-based interfaces for type safety without coupling
- Progressive fallback: PostgreSQL → DummyStore with warnings
- Connection string management with environment overrides
- Automatic store health checking and monitoring

**Configuration-Driven Behavior**:

- `use_dummy_stores: bool` controls testing vs production mode
- `db_connection: str | None` configures store persistence backend
- Store-specific configuration passed through actor config
- Registry integration for schema validation and versioning

The config module serves as the production-ready foundation for all ML component configuration in Nautilus Trader, ensuring type safety, validation, and consistency across the entire ML pipeline while maintaining the mandatory 4-store + 4-registry integration pattern and supporting flexible deployment scenarios from development to production.

## Implementation Status Summary

### Current Completion Assessment

**Overall Status**: **~75% Complete** (Updated from previous 100% claim)

#### ✅ **Fully Implemented** (9 components)
- **Core Configurations**: `MLFeatureConfig`, `MLInferenceConfig`, `MLStrategyConfig`, `MultiModelStrategyConfig`
- **Training Configurations**: `XGBoostTrainingConfig`, `LightGBMTrainingConfig` with GOSS/DART/EFB
- **Shared Components**: `OptunaConfig`, `AdvancedTrainingConfig`, All GPU configurations
- **System Configurations**: `ObservabilityConfig`, `MessageBusConfig`, `DataCollectorConfig`
- **Advanced Features**: `ModelDeploymentConfig`, `CanaryDeploymentConfig`

#### ⚠️ **Partially Implemented** (2 components)
- **MLActorConfig**: Core structure complete, missing `HealthMonitorConfig` definition and validation enforcement
- **Actor Configurations**: `MLSignalActorConfig` with backward compatibility mappings, but circular import issues

#### ❌ **Placeholder/Missing** (3 components)
- **OnnxRuntimeConfig**: Empty placeholder class - all documented functionality missing
- **Centralized Metrics**: `ml.common.metrics_bootstrap` exists but not integrated with config classes
- **Protocol Enforcement**: Protocol interfaces exist but no runtime validation

### Universal ML Architecture Pattern Compliance

#### Pattern 1: 4-Store + 4-Registry Integration
**Status**: ⚠️ **Partial Compliance**
- ✅ Configuration fields present (`db_connection`, `use_dummy_stores`)
- ❌ No enforcement of BaseMLInferenceActor inheritance
- ❌ No validation in configuration `__post_init__()` methods

#### Pattern 2: Protocol-First Interface Design
**Status**: ❌ **Not Implemented**
- ✅ Basic Protocol type hints in `adapters.py`
- ❌ No ML-specific protocol enforcement
- ❌ No runtime protocol compliance checking

#### Pattern 3: Hot/Cold Path Separation
**Status**: ✅ **Implemented**
- ✅ Latency budget configurations (`max_inference_latency_ms`, `max_feature_latency_ms`)
- ✅ Performance optimization settings in configurations

#### Pattern 4: Progressive Fallback Chains
**Status**: ⚠️ **Partial Compliance**
- ✅ `CircuitBreakerConfig` implemented with validation
- ✅ Dummy store configuration options
- ❌ No automatic fallback configuration generation

#### Pattern 5: Centralized Metrics Bootstrap
**Status**: ❌ **Major Violation**
- ✅ `ml.common.metrics_bootstrap` module exists (111 lines)
- ❌ 14+ files still contain direct `prometheus_client` imports
- ❌ Configuration classes have no metrics integration

### Environment Variable Support

#### ✅ **Implemented** (3 configurations)
- `ObservabilityConfig.from_env()` - 8 ML_OBS_* variables
- `MessageBusConfig.from_env()` - 7 ML_BUS_* variables
- `DataCollectorConfig` - 3 ML_DATA_* variables with legacy fallbacks

#### ❌ **Missing** (9 major configurations)
- `MLActorConfig`, `MLFeatureConfig`, `MLInferenceConfig` - No environment override support
- `XGBoostTrainingConfig`, `LightGBMTrainingConfig` - No environment integration
- `MLSignalActorConfig` - No environment parsing despite complexity

### Code Quality Issues

#### Type Safety Concerns
- Circular import workarounds in `actors.py` (lines 17-24)
- TYPE_CHECKING guards to avoid runtime cycles

#### Immutability Violations
- Direct `object.__setattr__()` usage in frozen dataclasses
- Legacy field mapping violates immutability guarantees

#### Missing Documentation
- `HealthMonitorConfig` referenced but not defined
- Placeholder classes with no implementation
- File structure discrepancies vs documentation

### Recommendations for Next Implementation Phase

#### High Priority (Essential for Universal Pattern Compliance)
1. **Implement missing `HealthMonitorConfig`** and integrate with `MLActorConfig`
2. **Add 4-store validation** to configuration `__post_init__()` methods
3. **Complete `OnnxRuntimeConfig`** implementation with all documented fields
4. **Remove direct prometheus_client imports** and enforce centralized metrics usage

#### Medium Priority (Enhanced Environment Support)
1. **Add `from_env()` methods** to remaining 9 major configuration classes
2. **Implement protocol enforcement** beyond basic type hints
3. **Resolve circular import dependencies** in actor configurations

#### Low Priority (Quality Improvements)
1. **Fix immutability violations** while preserving backward compatibility
2. **Add comprehensive validation** to all configuration classes
3. **Update documentation** to match actual implementation

The configuration system provides a solid foundation with comprehensive type safety and framework integration, but requires focused effort on Universal Pattern compliance and environment variable support to reach production readiness.
