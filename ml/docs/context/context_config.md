# Context: Config Module

## Overview

The `ml/config/` directory implements a comprehensive, production-ready configuration system for ML components using msgspec-based structured configuration classes. The module enforces the config-driven development mandate from CLAUDE.md, providing type-safe, validated, hierarchical configurations for all ML actors, training pipelines, and runtime components. All configurations are immutable (frozen=True), support environment variable overrides, and integrate seamlessly with Nautilus Trader's configuration system.

The config module follows a strict hierarchy where base configurations define common patterns, framework-specific configs (XGBoost, LightGBM) extend training configurations, and actor configs compose multiple configuration components. This architecture ensures consistency across the ML pipeline while providing flexibility for specialized use cases and maintaining the 4-store + 4-registry integration pattern required by the ML platform.

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
├── loader.py               # Typed configuration loader with layered merge
├── scheduler_config.py     # Data scheduler and Databento configuration
├── observability.py        # Observability system configuration with database and file sinks
├── bus.py                  # Message bus configuration and environment parsing
└── actor_bus.py            # Actor-side message bus configuration
```

### Configuration Examples (JSON)

```
ml/config/ (JSON Examples)
├── build_runner_example.json        # TFT dataset building configuration
├── pipeline_config_example.json     # Data pipeline configuration
├── databento_safe_config.json       # Databento API configuration
├── metrics_example.json             # Prometheus metrics configuration
├── promotion_gates_example.json     # Model promotion criteria
├── universe_tier*.json              # Symbol universe definitions
└── universe_proxies.json            # Symbol proxy mappings
```

### Configuration Hierarchy

1. **Base Layer**: `NautilusConfig`-derived msgspec structs with frozen=True
2. **Domain Layer**: ML-specific base classes (MLFeatureConfig, MLInferenceConfig, MLTrainingConfig, MLActorConfig)
3. **Framework Layer**: XGBoost, LightGBM training configurations with framework-specific parameters
4. **Actor Layer**: MLSignalActorConfig for composite ML actor configurations with strategy and optimization components
5. **Runtime Layer**: ONNX runtime, observability, message bus, and hardware acceleration configurations
6. **Environment Layer**: Progressive environment variable override system with `from_env()` class methods

## Key Components

### Base Configuration Classes (`base.py`)

#### MLFeatureConfig
Type-safe feature engineering configuration with validation:

- `lookback_window: PositiveInt = 100` - Historical bars for feature computation
- `indicators: dict[str, dict[str, Any]] | None` - Indicator configurations
- `feature_names: list[str] | None` - Explicit feature list (auto-computed if None)
- `normalize_features: bool = True` - Feature normalization toggle
- `fill_missing_with: float = 0.0` - Missing data imputation value
- `average_volume: PositiveFloat = 1000000.0` - Volume normalization baseline

#### MLInferenceConfig
Comprehensive inference configuration with dual model loading modes:

- **Model Loading**: Supports file-based (`model_path`) OR registry-based (`model_id`) model loading
- `prediction_threshold: NonNegativeFloat = 0.5` - Confidence threshold for valid predictions
- `max_inference_latency_ms: PositiveFloat = 5.0` - Hot-path latency budget enforcement
- `batch_size: PositiveInt = 1` - Batch size for model inference
- `warm_up_period: NonNegativeInt = 50` - Bars before predictions start
- `use_manifest_features: bool = True` - Registry-based feature schema loading
- `use_dummy_stores: bool = False` - Testing vs production store selection

#### MLActorConfig (Enhanced Production Features)
Core ML actor configuration with mandatory store integration:

- **Model Management**: `model_path: str`, `model_id: str` (required for tracking)
- **Performance**: `max_inference_latency_ms: PositiveFloat = 5.0`, `max_feature_latency_ms: PositiveFloat = 0.5`
- **Health Monitoring**: `enable_health_monitoring: bool = True`, `health_config: HealthMonitorConfig`
- **Circuit Breaker**: `circuit_breaker_config: CircuitBreakerConfig` for fault tolerance
- **Hot Reload**: `enable_hot_reload: bool = False`, `model_check_interval: PositiveInt = 300`
- **Security**: `allow_non_onnx_in_dev: bool = False` (ONNX-only enforcement in production)
- **Store Integration**: `db_connection: str | None`, `use_dummy_stores: bool = False`

#### MLStrategyConfig (Trading Strategy Integration)
Trading strategy configuration extending Nautilus StrategyConfig:

- **Signal Source**: `ml_signal_source: str` - Actor ID for ML signal consumption
- **Risk Management**: `position_size_pct: PositiveFloat = 0.1`, `max_positions: PositiveInt = 1`
- **Confidence**: `min_confidence: NonNegativeFloat = 0.7` - Signal confidence threshold
- **Risk Controls**: `stop_loss_pct: NonNegativeFloat = 0.02`, `take_profit_pct: NonNegativeFloat = 0.04`
- **Production Safety**: `execute_trades: bool = False` - Signal-only mode by default
- **Store Integration**: `use_strategy_store: bool = True`, `persist_all_signals: bool = False`

#### New: MultiModelStrategyConfig
Configuration for strategies consuming multiple models:

- `target_model_ids: list[str]` - List of model IDs to consume
- `aggregation_mode: Literal["voting", "weighted_average", "best"]` - Signal aggregation method
- `model_weights: dict[str, float] | None` - Weights for weighted_average mode
- `required_models: PositiveInt = 1` - Minimum models before trading

#### New: ModelDeploymentConfig & CanaryDeploymentConfig
Advanced deployment configurations:

- **Deployment**: `deployment_target`, `rollout_strategy`, `rollout_percentage`
- **Canary**: `initial_traffic_percentage`, `auto_promote`, `error_threshold_percentage`
- **Health Checks**: `health_check_interval`, `auto_rollback_on_error`

### Framework-Specific Training Configurations

#### XGBoostTrainingConfig (`xgboost.py`)
Comprehensive XGBoost configuration extending MLTrainingConfig:

**Core Parameters:**

- `n_estimators: PositiveInt = 100`, `max_depth: PositiveInt = 6`, `learning_rate: PositiveFloat = 0.3`
- `subsample: PositiveFloat = 1.0`, `colsample_bytree: PositiveFloat = 1.0`, `colsample_bylevel: PositiveFloat = 1.0`

**Regularization:**

- `reg_alpha: NonNegativeFloat = 0.0`, `reg_lambda: NonNegativeFloat = 1.0`, `gamma: NonNegativeFloat = 0.0`
- `min_child_weight: NonNegativeFloat = 1.0`

**Hardware & Objectives:**

- `tree_method: str = "hist"` (GPU via "gpu_hist"), `objective: str = "binary:logistic"`, `eval_metric: str = "auc"`
- GPU validation and environment checking via `validate_environment()`

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
Advanced LightGBM configuration with specialized boosting strategies:

**Core Parameters:**

- `n_estimators: PositiveInt = 100`, `max_depth: PositiveInt = 6`, `learning_rate: PositiveFloat = 0.1`
- `num_leaves: PositiveInt = 31`, `min_child_samples: PositiveInt = 20`, `min_child_weight: NonNegativeFloat = 1e-3`

**Advanced Boosting Strategies:**

- **GOSSConfig**: `enabled: bool`, `top_rate: float = 0.2`, `other_rate: float = 0.1` - Gradient-based One-Side Sampling
- **DARTConfig**: `enabled: bool`, `drop_rate: float = 0.1`, `max_drop: int = 50` - Dropout regularization for trees
- **EFBConfig**: `enabled: bool = True`, `max_conflict_rate: float = 0.0` - Exclusive Feature Bundling optimization

**Memory & Performance:**

- `force_col_wise: bool = False`, `force_row_wise: bool = False` - Memory layout optimization
- `categorical_features: list[str] = []` - Native categorical feature support

**Hardware Integration:**

- `gpu_config: LightGBMGPUConfig | None` - OpenCL GPU acceleration with platform selection
- Environment validation for GPU availability

### Shared Components (`shared.py`)

#### OptunaConfig
Comprehensive hyperparameter optimization configuration:

- `enabled: bool = False`, `n_trials: int = 100`, `direction: str = "maximize"`
- `metric: str = "sharpe_ratio"` - Target metric (sharpe_ratio, accuracy, auc, rmse, mae, r2)
- `pruner: str = "median"` - Pruning algorithm (median, percentile, hyperband, none)
- `sampler: str = "tpe"` - Sampling algorithm (tpe, random, cmaes, grid)
- `timeout: int | None = None`, `study_name: str | None`, `storage_url: str | None` - Persistence options

#### GPU Configuration Hierarchy

**BaseGPUConfig**: Foundation for GPU acceleration

- `enabled: bool = False`, `device_id: int = 0`, `validate_gpu: bool = True`

**XGBoostGPUConfig**: XGBoost-specific GPU parameters

- `max_bin: int = 256` - Histogram construction bins
- `predictor: str = "gpu_predictor"` - GPU inference predictor type

**LightGBMGPUConfig**: LightGBM OpenCL configuration

- `platform_id: int = -1` - OpenCL platform (-1 = auto-detect)
- `gpu_use_dp: bool = False` - Double precision GPU math

#### AdvancedTrainingConfig
Cross-framework advanced training features:

- **Feature Monitoring**: `track_feature_decay: bool = True`, `feature_decay_threshold: float = 0.3`
- **Cross-Validation**: `cv_strategy: str = "time_series"` (time_series, blocked, purged, standard)
- `cv_folds: int = 5`, `purge_gap: int = 10` - CV configuration
- **ONNX Export**: `export_onnx: bool = False`, `onnx_output_path: str | None`
- **Monitoring**: `enable_monitoring: bool = True` - Prometheus metrics integration

### Runtime and System Configurations

#### ObservabilityConfig (`observability.py`)
Off hot-path observability configuration with environment integration:

- **Sink Selection**: `sink: Literal["file", "db"] = "file"` - Output destination
- **File Options**: `base_path: str = "./observability"`, `file_format: str = "jsonl"`
- **Database**: `db_connection_string: str | None` - SQLAlchemy connection URL
- **Scheduling**: `interval_seconds: PositiveFloat = 60.0` - Background flush interval
- **Environment Integration**: `from_env()` class method with ML_OBS_* environment variables

#### OnnxRuntimeConfig (`runtime.py`)
ONNX Runtime optimization configuration for inference:

- `graph_optimization_level: GraphOptLevel = "all"` - Optimization level (disable, basic, extended, all)
- `execution_mode: ExecutionMode = "sequential"` - Execution mode (sequential, parallel)
- `providers: list[str]` - Execution providers (CPU, CUDA)
- `intra_threads: int | None`, `inter_threads: int | None` - Thread configuration
- Helper: `to_session_options()` converts config to ONNX SessionOptions

#### MessageBusConfig (`bus.py`)
Optional ML message bus configuration with environment parsing:

- **Control**: `enabled: bool = False`, `backend: BusBackend = "noop"` (noop, redis)
- **Topics**: `scheme: TopicScheme = "domain_op"` (domain_op, stage_first), `topic_prefix: str`
- **Redis**: `redis_url: str`, `redis_stream: str`, `redis_maxlen: int | None`
- **Environment**: `from_env()` with ML_BUS_* environment variables

#### DataCollectorConfig (`base.py`)
Enhanced data collector configuration:

- `data_dir: str = "./data/tier1"`, `storage_limit_gb: PositiveFloat = 500.0`
- `end_date_iso: str | None` - Optional collection end date
- **Environment Mapping**: ML_DATA_TIER1_DIR, ML_STORAGE_LIMIT_GB with legacy fallbacks

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

## Usage Patterns

### Modern Configuration Creation

#### Framework Training Configuration

```python
from ml.config import XGBoostTrainingConfig, OptunaConfig, AdvancedTrainingConfig

# XGBoost with advanced features
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
        cv_strategy="purged",
        cv_folds=5,
        export_onnx=True,
        track_feature_decay=True
    )
)
```

#### Production ML Actor Configuration

```python
from ml.config import MLSignalActorConfig, OptimizationConfig, StrategyConfig, MLFeatureConfig

# Production-ready signal actor
actor_config = MLSignalActorConfig(
    model_path="models/xgb_model.onnx",
    model_id="xgb_signal_v2.1",
    bar_type=bar_type,
    instrument_id=instrument_id,
    signal_strategy="adaptive",   # or set `signal_policy="adaptive"`
    # Performance optimization
    optimization_config=OptimizationConfig(
        level="optimized",
        enable_zero_copy=True,
        pre_allocate_buffers=True
    ),
    # Strategy configuration
    strategy_config=StrategyConfig(
        adaptive_volatility_factor=2.0,
        min_threshold=0.1,
        max_threshold=0.95
    ),
    # Feature configuration
    feature_config=MLFeatureConfig(
        lookback_window=200,
        normalize_features=True,
        feature_names=["price_sma_5", "volume_sma_20", "rsi"]
    ),
    # Store integration (MANDATORY)
    db_connection="postgresql://user:pass@localhost:5432/nautilus",
    use_dummy_stores=False,  # Production mode
    # Health monitoring
    enable_health_monitoring=True,
    health_config=HealthMonitorConfig(
        critical_consecutive_failures=10,
        degraded_success_rate_threshold=0.9
    )
)
```

#### Environment-Based Configuration

```python
from ml.config import ObservabilityConfig, MessageBusConfig

# Environment-driven configuration
obs_config = ObservabilityConfig.from_env()  # Uses ML_OBS_* environment variables
bus_config = MessageBusConfig.from_env()     # Uses ML_BUS_* environment variables
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

## Implementation Review Addendum

### Ground Truth Analysis vs Documentation Claims

**Overall Assessment**: The ml/config implementation shows **substantial discrepancies** between documentation claims and actual code implementation, with significant gaps in Universal ML Architecture Pattern compliance.

### 1. Universal ML Architecture Pattern Compliance Issues

#### Pattern 1: 4-Store + 4-Registry Integration - ❌ **PARTIAL COMPLIANCE**

**Documentation Claims**: "MANDATORY Integration: All ML actors MUST use the 4-store + 4-registry pattern via BaseMLInferenceActor"

**Ground Truth Issues**:

- **File**: `/home/nate/projects/nautilus_trader/ml/config/base.py:222-223`
  - `MLActorConfig` includes optional store fields (`db_connection: str | None = None`, `use_dummy_stores: bool = False`) but **does NOT enforce** BaseMLInferenceActor inheritance
  - No validation in `__post_init__()` to ensure mandatory store integration
- **File**: `/home/nate/projects/nautilus_trader/ml/config/actors.py:61-93`
  - `MLSignalActorConfig` extends `MLActorConfig` but **lacks mandatory store validation**
  - Contains optional fields like `use_feature_store: bool = False` contradicting "mandatory" claims

#### Pattern 2: Protocol-First Interface Design - ❌ **NOT IMPLEMENTED**

**Documentation Claims**: "Use typing.Protocol for all component interfaces"

**Ground Truth Issues**:

- **File**: `/home/nate/projects/nautilus_trader/ml/config/adapters.py:41-51`
  - Uses basic `Protocol` for type hints but **no actual ML protocol enforcement**
  - Missing `MLComponentProtocol` implementation referenced in documentation
- No evidence of runtime protocol compliance checking mentioned in docs

#### Pattern 5: Centralized Metrics Bootstrap - ❌ **MAJOR VIOLATION**

**Documentation Claims**: "NEVER import prometheus_client directly. Use ml.common.metrics_bootstrap"

**Ground Truth Issues**:

- Found **36 files** containing `prometheus_client` imports throughout the codebase
- **NO centralized metrics_bootstrap module found** in ml/config/ or ml/common/
- Configuration classes provide **no metrics integration** despite documentation claims

### 2. Configuration System Implementation Gaps

#### Environment Integration Issues

**Documentation Claims**: "Progressive environment override system with `from_env()` class methods"

**Ground Truth**:

- **PARTIAL**: Only `ObservabilityConfig` and `MessageBusConfig` implement `from_env()` methods
- **Missing**: Base `MLActorConfig`, `MLFeatureConfig`, `XGBoostTrainingConfig` have **no environment override capability**
- **File**: `/home/nate/projects/nautilus_trader/ml/config/base.py:226-286` - Only `DataCollectorConfig` implements environment mapping

#### Configuration Validation Inconsistencies

**Documentation Claims**: "All configuration classes implement `__post_init__()` methods with comprehensive validation"

**Ground Truth Issues**:

- **File**: `/home/nate/projects/nautilus_trader/ml/config/base.py:101-111` - `MLInferenceConfig.__post_init__()` validates model loading but **missing hardware/framework validation**
- **File**: `/home/nate/projects/nautilus_trader/ml/config/shared.py:52-83` - `OptunaConfig` has proper validation
- **File**: `/home/nate/projects/nautilus_trader/ml/config/base.py:313-320` - `OnnxRuntimeConfig`, `OptimizationConfig`, `MLSignalActorConfig` are **placeholder classes with no implementation**

### 3. Framework Integration Issues

#### Lazy Loading Implementation

**Documentation Claims**: "ML frameworks loaded via `ml._imports` with `HAS_*` availability flags"

**Ground Truth**:

- **VERIFIED**: XGBoost and LightGBM configs properly use lazy imports
- **File**: `/home/nate/projects/nautilus_trader/ml/config/xgboost.py:334-372` - Environment validation implemented
- **File**: `/home/nate/projects/nautilus_trader/ml/config/lightgbm.py:486-545` - Environment validation implemented

#### GPU Configuration

**Documentation Claims**: "Hardware capability detection with automatic fallback strategies"

**Ground Truth**:

- **IMPLEMENTED**: Both XGBoost and LightGBM configs include GPU validation
- **ISSUE**: No fallback configuration classes for when GPU unavailable

### 4. Documentation Accuracy Issues

#### Completion Percentages

**Documentation Claims**: "Configuration system with environment overrides (100% complete)"

**Ground Truth**: **~60-70% complete** based on actual implementation:

- Environment overrides: 3/12 major config classes
- Validation completeness: 8/15 config classes have proper `__post_init__()`
- Protocol implementation: 0/5 Universal Patterns properly implemented

#### Missing Components

**Documentation Lists but Not Implemented**:

- `HealthMonitorConfig` referenced in `MLActorConfig` but **not defined in codebase**
- `CircuitBreakerConfig` implemented (lines 113-144) but **not integrated** in main configs
- `SchedulerConfig` and `DatabentoConfig` use `@dataclass` instead of `NautilusConfig`

### 5. File Structure vs Documentation

**Documentation Claims**:

```
ml/config/
├── adapters.py             # Configuration utilities and protocol-based helpers
├── loader.py               # Typed configuration loader with layered merge
└── actor_bus.py            # Actor-side message bus configuration
```

**Ground Truth Issues**:

- **Missing Files**: No `scheduler_config.py`, `events.py`, `names.py`, `defaults.py` mentioned in documentation
- **Extra Files**: Found `version.py`, `actor_bus.py`, `events.py` not documented
- **File**: `/home/nate/projects/nautilus_trader/ml/config/adapters.py:115` - Only 115 lines vs claimed comprehensive utilities

### 6. Specific Code Quality Issues

#### Type Safety Issues

**File**: `/home/nate/projects/nautilus_trader/ml/config/actors.py:17-24`

```python
if TYPE_CHECKING:
    from ml.actors.signal import OptimizationLevel as _OptimizationLevel
    from ml.actors.signal import SignalStrategy as _SignalStrategy
else:
    _OptimizationLevel = object  # type: ignore[misc,assignment]
    _SignalStrategy = object  # type: ignore[misc,assignment]
```

- **Issue**: Circular import workaround indicates poor module separation

#### Immutability Violations

**File**: `/home/nate/projects/nautilus_trader/ml/config/base.py:138-144`

```python
def __post_init__(self) -> None:
    if self.half_open_attempts is not None:
        object.__setattr__(self, "success_threshold", int(self.half_open_attempts))
```

- **Issue**: Modifying frozen dataclass violates immutability guarantees

### 7. Integration Pattern Violations

#### 4-Store Pattern Enforcement

**Documentation Claims**: "Configuration Integration: `use_dummy_stores: bool` controls testing vs production mode"

**Ground Truth Issues**:

- **File**: `/home/nate/projects/nautilus_trader/ml/config/base.py:223` - Field exists but **no enforcement mechanism**
- **File**: `/home/nate/projects/nautilus_trader/ml/config/actors.py:92` - Default `use_dummy_stores: bool = False` provides no validation

### Recommendations for Remediation

1. **Implement Missing Components**:
   - Create `ml.common.metrics_bootstrap` module
   - Implement `MLComponentProtocol` interface
   - Add environment override methods to all major config classes

2. **Fix Universal Pattern Compliance**:
   - Add BaseMLInferenceActor inheritance validation
   - Remove direct prometheus_client imports
   - Implement protocol-first interfaces

3. **Documentation Updates**:
   - Reduce completion claims from 100% to realistic 60-70%
   - Document actual file structure
   - Remove references to unimplemented components

4. **Code Quality Improvements**:
   - Resolve circular import dependencies
   - Fix immutability violations
   - Add comprehensive validation to all config classes

**Summary**: The ml/config domain shows a **70% implementation gap** compared to documentation claims, with critical failures in Universal ML Architecture Pattern compliance and missing core infrastructure components.
