# Context: ML Configuration System

## Overview

The `ml/config/` directory implements a **msgspec-based configuration system** for ML components in Nautilus Trader, providing type-safe, immutable configuration classes for actors, training pipelines, runtime components, and orchestration workflows. The system enforces **config-driven development** through frozen dataclasses with comprehensive validation, environment variable overrides, and integration with Nautilus Trader's core configuration infrastructure.

**Implementation Status**: **~80% Complete** - Core configurations are production-ready with validation and environment support, but some advanced features (ONNX runtime config, universal pattern enforcement) remain incomplete. The system successfully deployed in production with 95-instrument datasets and full macro feature pipelines.

**Total Code Volume**: 4,433 lines across 26 Python modules + 3 TOML orchestrator configs + 6 JSON descriptor files.

## Architecture

### File Structure (Current Reality)

```
ml/config/
├── __init__.py                   # 254 lines - Public API with 54 exports
├── events.py                     # 62 lines - Event enums (Stage, Source, EventStatus)
├── base.py                       # 587 lines - Core ML configurations
├── actors.py                     # 173 lines - Actor configurations with backward compat
├── shared.py                     # 238 lines - GPU, Optuna, advanced training
├── constants.py                  # 271 lines - System constants and enums
├── loader.py                     # 74 lines - Typed config loading utilities
├── xgboost.py                    # 241 lines - XGBoost training config
├── lightgbm.py                   # 302 lines - LightGBM with GOSS/DART/EFB
├── registry.py                   # 34 lines - Registry config
├── observability.py              # 112 lines - Observability with from_env()
├── bus.py                        # 104 lines - Message bus with from_env()
├── actor_bus.py                  # Actor-side bus config
├── runtime.py                    # 23 lines - ONNX runtime placeholder
├── scheduler_config.py           # Data scheduler and Databento config
├── market_data.py                # Market feed descriptors and bindings
├── streaming_pipeline.py         # 192 lines - Streaming training configs (NEW)
├── playground.py                 # Backtest and risk model defaults
├── adapters.py                   # 115 lines - Nautilus adapter utilities
├── defaults.py                   # Configuration defaults
├── names.py                      # Name constants and mappings
├── universes.py                  # Symbol universe configurations
├── dataset_ids.py                # Dataset ID constants
├── databento_policy.py           # Databento retry and rate-limit policies
├── coverage.py                   # Coverage tracking config
├── version.py                    # Version constants
└── orchestrator/
    ├── production_full.toml      # 183 lines - 95-symbol full macro pipeline
    ├── spy_production_full_macro.toml  # 171 lines - 30-macro-series config
    ├── spy_with_revisions.toml   # 133 lines - Vintage age conversion
    └── validate_config.py        # TOML validation utilities
```

### Configuration Categories

1. **Core Configs** (base.py, 587 lines)
   - MLFeatureConfig, MLInferenceConfig, MLActorConfig, MLTrainingConfig
   - MLStrategyConfig, MultiModelStrategyConfig, DataCollectorConfig
   - CircuitBreakerConfig, HealthMonitorConfig, ModelDeploymentConfig

2. **Event System** (events.py, 62 lines)
   - Stage enum: DATASET_PLANNED, DATA_INGESTED, FEATURE_COMPUTED, MODEL_TRAINING_STARTED, PREDICTION_EMITTED, SIGNAL_EMITTED
   - Source enum: LIVE, BATCH, HISTORICAL, BACKFILL
   - EventStatus enum: SUCCESS, FAILED, PARTIAL, DEFERRED
   - **Used in 91 files** across stores, registries, orchestration, training

3. **Training Configs** (xgboost.py 241 lines, lightgbm.py 302 lines, shared.py 238 lines)
   - XGBoostTrainingConfig with GPU, SHAP, multi-asset, monotonic constraints
   - LightGBMTrainingConfig with GOSS/DART/EFB advanced boosting
   - OptunaConfig for hyperparameter optimization
   - AdvancedTrainingConfig with CV strategies and ONNX export

4. **Runtime Configs** (observability.py 112 lines, bus.py 104 lines)
   - ObservabilityConfig with file/db sinks and async workers
   - MessageBusConfig with Redis Streams and topic schemes
   - OnnxRuntimeConfig (placeholder - **INCOMPLETE**)

5. **Streaming Pipeline** (streaming_pipeline.py 192 lines - **NEW**)
   - DatasetServiceConfig for dataset planning microservice
   - StreamingWorkerConfig for bounded streaming training
   - TrainingOrchestratorConfig for event-driven orchestration

6. **Orchestration** (orchestrator/ directory)
   - TOML-based pipeline configurations for production
   - Multi-instrument universes (95 symbols in production_full.toml)
   - Comprehensive macro feature sets (30 series across 6 dimensions)

## Core Configuration Classes

### Event System (events.py)

**Purpose**: Canonical event constants preventing ad-hoc string literals. Values persist to database and must match schema constraints.

**Implementation** (lines 14-60):

```python
class Stage(str, Enum):
    """Processing stages for ML pipeline events."""
    DATASET_PLANNED = "DATASET_PLANNED"
    DATA_INGESTED = "INGESTED"
    CATALOG_WRITTEN = "CATALOG_WRITTEN"
    FEATURE_COMPUTED = "FEATURE_COMPUTED"
    MODEL_INFERRED = "MODEL_INFERRED"          # Back-compat alias
    MODEL_TRAINING_STARTED = "MODEL_TRAINING_STARTED"
    MODEL_TRAINING_COMPLETED = "MODEL_TRAINING_COMPLETED"
    WORKER_HEARTBEAT = "WORKER_HEARTBEAT"
    PREDICTION_EMITTED = "PREDICTION_EMITTED"
    SIGNAL_EMITTED = "SIGNAL_EMITTED"

class Source(str, Enum):
    """Allowed event sources persisted by the registry."""
    LIVE = "live"
    BATCH = "batch"                             # Back-compat alias
    HISTORICAL = "historical"
    BACKFILL = "backfill"

class EventStatus(str, Enum):
    """Standardized status values for emitted events."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    DEFERRED = "deferred"
```

**Usage Pattern**:
```python
from ml.config import EventStage, EventSource, EventStatus

# Build topics via ml.common.message_topics.build_topic_for_stage
topic = build_topic_for_stage(EventStage.FEATURE_COMPUTED, Source.LIVE)
# Topic: "ml.features.live.computed" (domain_op scheme)

# Emit events with status
event = FeatureEvent(
    stage=EventStage.FEATURE_COMPUTED,
    source=EventSource.LIVE,
    status=EventStatus.SUCCESS.value,  # Use .value when persisting
)
```

**Integration**: Used in 91 files including:
- All store implementations (data_store.py, feature_store.py, model_store.py)
- Registry event managers (data_registry.py, event_manager.py)
- Training orchestration (pipeline_orchestrator.py, streaming workers)
- Test contracts (409 total occurrences across codebase)

### MLFeatureConfig (base.py lines 27-55)

**Status**: ✅ **Fully Implemented** - Complete feature engineering configuration

```python
class MLFeatureConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for ML feature engineering."""
    lookback_window: PositiveInt = 120
    indicators: dict[str, dict[str, Any]] | None = None
    feature_names: list[str] | None = None
    normalize_features: bool = True
    fill_missing_with: float = 0.0
    average_volume: PositiveFloat = 1000000.0
```

**Key Features**:
- **Lookback window**: Historical bars for feature computation (default 120)
- **Indicators**: Dict of indicator configs (e.g., `{"sma_5": {"period": 5}}`)
- **Feature names**: Explicit feature list (auto-computed if None)
- **Normalization**: Feature scaling toggle with imputation value
- **Volume baseline**: Average volume for normalization

**No validation** in `__post_init__()` - relies on type annotations only.

### MLInferenceConfig (base.py lines 57-111)

**Status**: ✅ **Fully Implemented** - Dual model loading with validation

```python
class MLInferenceConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for ML inference components."""
    model_path: str | None = None              # File-based loading
    model_id: str | None = None                # Registry-based loading
    registry_path: str | None = None
    prediction_threshold: NonNegativeFloat = 0.5
    max_inference_latency_ms: PositiveFloat = 5.0
    feature_config: MLFeatureConfig | None = None
    batch_size: PositiveInt = 1
    warm_up_period: NonNegativeInt = 50
    use_manifest_features: bool = True
    use_dummy_stores: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.model_path and not self.model_id:
            raise ValidationError("Either model_path or model_id must be provided")
        if self.model_id and not self.registry_path:
            raise ValidationError("registry_path is required when using model_id")
        if self.model_path and self.model_id:
            raise ValidationError("Cannot specify both model_path and model_id")
```

**Validation Strategy** (lines 101-111):
- Mutual exclusion: model_path XOR model_id (not both)
- Registry requirement: model_id requires registry_path
- Comprehensive error messages with parameter context

### MLActorConfig (base.py lines 146-229)

**Status**: ✅ **Configurable via Environment** – runtime builder + validation helpers

```python
class MLActorConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for ML actors with enhanced production features."""
    model_path: str
    model_id: str                              # Required for tracking
    bar_type: BarType
    instrument_id: InstrumentId
    prediction_threshold: NonNegativeFloat = 0.5
    max_inference_latency_ms: PositiveFloat = 5.0
    feature_config: MLFeatureConfig | None = None
    batch_size: PositiveInt = 1
    warm_up_period: NonNegativeInt = 50
    publish_signals: bool = True
    signal_data_type: str = "MLSignal"
    log_predictions: bool = False
    enable_hot_reload: bool = False
    model_check_interval: PositiveInt = 300
    preserve_state_on_reload: bool = True
    circuit_breaker_config: CircuitBreakerConfig | None = None
    enable_health_monitoring: bool = True
    health_config: HealthMonitorConfig | None = None  # ⚠️ Defined below
    max_feature_latency_ms: PositiveFloat = 0.5
    component_id: ComponentId | None = None
    log_events: bool = True
    log_commands: bool = True

    # Security and integration
    allow_non_onnx_in_dev: bool = False        # ONNX-only enforcement
    db_connection: str | None = None
    use_dummy_stores: bool = False

    # Async persistence
    enable_async_persistence: bool = True
    persistence_queue_size: PositiveInt = 10000
    persistence_flush_interval: PositiveFloat = 1.0
    persistence_batch_size: PositiveInt = 100
```

**Environment overrides**: `MLActorConfig.from_env()` now maps runtime settings (MODEL_PATH/ID,
INSTRUMENT_ID, BAR_TYPE, persistence toggles, logging flags) with PostgreSQL resolution delegated
to `collect_postgres_candidates`. Hot-path safety options (`ML_ENABLE_ASYNC_PERSISTENCE`,
`ML_PUBLISH_SIGNALS`, `ML_ALLOW_NON_ONNX_IN_DEV`) align with actor entrypoint semantics.

**Validation**: Health monitor thresholds remain delegated to `HealthMonitorConfig`; additional
validation occurs in downstream actors.

**HealthMonitorConfig** (base.py lines 293-314):

```python
class HealthMonitorConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration thresholds for ML actor health monitor."""
    critical_consecutive_failures: PositiveInt = 10
    degraded_success_rate_threshold: NonNegativeFloat = 0.9
    degraded_consecutive_failures: PositiveInt = 3
    degraded_latency_violations: PositiveInt = 100
```

**Status**: ✅ **Validated** – `__post_init__()` clamps thresholds within [0, 1].

### MLStrategyConfig (base.py lines 334-386)

**Status**: ✅ **Fully Implemented** - Trading strategy configuration

```python
class MLStrategyConfig(StrategyConfig, kw_only=True, frozen=True):
    """Configuration for ML-based trading strategies."""
    instrument_id: InstrumentId
    ml_signal_source: str                      # Actor ID or data source
    position_size_pct: PositiveFloat = 0.1
    min_confidence: NonNegativeFloat = 0.7
    max_positions: PositiveInt = 1
    stop_loss_pct: NonNegativeFloat = 0.02
    take_profit_pct: NonNegativeFloat = 0.04
    use_strategy_store: bool = True
    strategy_store_config: dict[str, Any] | None = None
    persist_all_signals: bool = False
    execute_trades: bool = False               # Signal-only mode by default

    # Protocol-first component configs (TYPE_CHECKING only)
    sizing_config: _SizingConfig | None = None
    risk_config: _RiskConfig | None = None
    execution_config: _ExecutionConfig | None = None
    portfolio_config: _PortfolioConfig | None = None
    analytics_config: _AnalyticsConfig | None = None
    circuit_breaker_config: CircuitBreakerConfig | None = None
```

**Pattern**: Uses TYPE_CHECKING imports (lines 581-586) to avoid circular dependencies:

```python
if TYPE_CHECKING:  # typing-only to avoid runtime import cycles
    from ml.strategies.analytics import AnalyticsConfig as _AnalyticsConfig
    from ml.strategies.execution import ExecutionConfig as _ExecutionConfig
    # ... other imports
```

### CircuitBreakerConfig (base.py lines 113-144)

**Status**: ✅ **Fully Implemented** - Progressive fallback pattern support

```python
class CircuitBreakerConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for circuit breaker pattern."""
    failure_threshold: PositiveInt = 5
    recovery_timeout: PositiveInt = 60
    success_threshold: PositiveInt = 3
    half_open_attempts: PositiveInt | None = None  # Legacy alias

    def __post_init__(self) -> None:
        """Normalize legacy aliases while preserving immutability."""
        if (self.half_open_attempts is not None and
            self.half_open_attempts != self.success_threshold):
            object.__setattr__(self, "success_threshold", int(self.half_open_attempts))
```

**Pattern**: Uses `object.__setattr__()` to mutate frozen dataclass during `__post_init__()` for backward compatibility.

## Actor Configuration (actors.py)

### MLSignalActorConfig (actors.py lines 61-166)

**Status**: ✅ **Fully Implemented** - Unified signal actor configuration with extensive backward compatibility

```python
class MLSignalActorConfig(MLActorConfig, kw_only=True, frozen=True):
    """Unified configuration for ML Signal Actor with all features."""
    signal_strategy: Literal["threshold", "extremes", "momentum", "ensemble", "adaptive"] | _SignalStrategy = "threshold"
    signal_policy: ... | None = None           # Alias for clarity
    adaptive_window: PositiveInt = 20
    min_signal_separation_bars: PositiveInt = 3
    feature_importance_threshold: NonNegativeFloat = 0.01
    enable_regime_detection: bool = True
    optimization_config: OptimizationConfig | None = None
    strategy_config: StrategyConfig | None = None
    enable_hot_reload: bool = False
    hot_reload_interval: PositiveInt = 300
    custom_strategy: Any | None = None
    actor_id: str | None = None

    # Feature registry integration
    feature_set_id: str | None = None
    registry_path: str | None = None
    use_registry_features: bool = False

    # ONNX runtime
    onnx_runtime_config: OnnxRuntimeConfig | None = None

    # FeatureStore integration
    use_feature_store: bool = False
    db_connection: str | None = msgspec.field(default_factory=_default_actor_db_connection)
    persist_features: bool = True
    pipeline_spec: Any | None = None
    use_dummy_stores: bool = False

    # Back-compat alias fields
    optimization: OptimizationConfig | None = None
    strategy: StrategyConfig | None = None
```

`db_connection` now defaults to the first candidate discovered by
`collect_postgres_candidates(ConnectionRole.PRIMARY)`, which honours `ML_DB_CONNECTION`,
`DATABASE_URL`, and other environment overrides before falling back to Compose-style
localhost URLs. When none of these are present the field remains `None`, allowing dummy
store configurations to operate without a database.

**Backward Compatibility** (lines 104-166):

```python
def __post_init__(self) -> None:
    """Map backward-compat alias fields to canonical fields while frozen."""
    # Map signal_policy -> signal_strategy
    if self.signal_policy is not None:
        object.__setattr__(self, "signal_strategy", self.signal_policy)

    # Map optimization -> optimization_config
    if self.optimization is not None and self.optimization_config is None:
        object.__setattr__(self, "optimization_config", self.optimization)

    # Map strategy -> strategy_config
    if self.strategy is not None and self.strategy_config is None:
        object.__setattr__(self, "strategy_config", self.strategy)

    # Map legacy strategy.strategy_type to signal_strategy
    if self.strategy is not None and self.strategy.strategy_type:
        object.__setattr__(self, "signal_strategy", self.strategy.strategy_type)

    # Merge legacy thresholds (threshold_long, threshold_short) -> prediction_threshold
    # Uses conservative max() to preserve strictest threshold
    if self.strategy and (self.strategy.threshold_long or self.strategy.threshold_short):
        merged = max(abs(self.strategy.threshold_long or 0.0),
                     abs(self.strategy.threshold_short or 0.0))
        object.__setattr__(self, "prediction_threshold", merged)
```

**Circular Import Avoidance** (lines 17-23):

```python
if TYPE_CHECKING:
    from ml.actors.signal import OptimizationLevel as _OptimizationLevel
    from ml.actors.signal import SignalStrategy as _SignalStrategy
else:  # pragma: no cover
    _OptimizationLevel = object  # type: ignore
    _SignalStrategy = object     # type: ignore
```

### OptimizationConfig (actors.py lines 26-41)

```python
class OptimizationConfig(NautilusConfig, kw_only=True, frozen=True):
    """Performance optimization configuration for signal actors."""
    level: Literal["standard", "optimized"] | _OptimizationLevel = "standard"
    enable_zero_copy: bool = False
    enable_model_warm_up: bool = False
    warm_up_iterations: PositiveInt = 100
    pre_allocate_buffers: bool = True
    use_lock_free_buffers: bool = False
    reservoir_sample_size: PositiveInt = 1000
    # Back-compat aliases
    feature_cache_size: PositiveInt | None = None
    enable_profiling: bool = False
```

### StrategyConfig (actors.py lines 43-58)

```python
class StrategyConfig(NautilusConfig, kw_only=True, frozen=True):
    """Strategy-specific configuration for signal generation."""
    extremes_top_pct: float = 0.1
    momentum_lookback: PositiveInt = 5
    ensemble_weights: dict[str, float] | None = None
    adaptive_volatility_factor: float = 2.0
    min_threshold: float = 0.1
    max_threshold: float = 0.95
    update_frequency: PositiveInt = 10
    # Back-compat aliases
    strategy_type: str | None = None
    threshold_long: float | None = None
    threshold_short: float | None = None
```

## Training Configurations

### XGBoostTrainingConfig (xgboost.py lines 26-160)

**Status**: ✅ **Fully Implemented** - Comprehensive XGBoost configuration

```python
class XGBoostTrainingConfig(MLTrainingConfig, kw_only=True, frozen=True):
    """Comprehensive configuration for XGBoost model training."""
    # Core parameters
    n_estimators: PositiveInt = 100
    max_depth: PositiveInt = 6
    learning_rate: PositiveFloat = 0.3
    min_child_weight: NonNegativeFloat = 1.0
    subsample: PositiveFloat = 1.0
    colsample_bytree: PositiveFloat = 1.0
    colsample_bylevel: PositiveFloat = 1.0
    gamma: NonNegativeFloat = 0.0
    reg_alpha: NonNegativeFloat = 0.0
    reg_lambda: NonNegativeFloat = 1.0

    # Missing value handling
    handle_missing: bool = True
    missing_value: float = float("nan")
    scale_pos_weight: PositiveFloat | None = None

    # Hardware settings
    tree_method: str = "hist"                  # "hist" for CPU, "gpu_hist" for GPU
    gpu_id: NonNegativeInt = 0

    # Training objective
    objective: str = "binary:logistic"
    eval_metric: str = "auc"

    # Advanced features
    enable_shap: bool = False
    monotonic_constraints: dict[str, int] | None = None

    # Multi-asset configuration
    multi_asset: bool = False
    sector_map: dict[str, str] | None = None
    cross_sectional_features: bool = True

    # Legacy hyperparameter optimization (backward compat)
    optimize_hyperparams: bool = False
    n_trials: PositiveInt = 100
    optimization_metric: str = "sharpe_ratio"

    # Advanced configuration components
    gpu_config: XGBoostGPUConfig | None = None
    optuna_config: OptunaConfig | None = None
    mlflow_config: None = None                 # deprecated
    advanced_config: AdvancedTrainingConfig | None = None
```

**Convenience Properties** (lines 141-160): Delegate to advanced_config for backward compat.

**Environment overrides**: `XGBoostTrainingConfig.from_env()` hydrates training, GPU, Optuna, and advanced settings from `ML_XGB_*`, `ML_OPTUNA_*`, and `ML_TRAIN_*` variables (e.g., `ML_XGB_DATA_SOURCE`, `ML_XGB_TREE_METHOD`, `ML_OPTUNA_TRIALS`), enabling twelve-factor tuning without editing manifests.

### LightGBMTrainingConfig (lightgbm.py)

**Status**: ✅ **Fully Implemented** - Advanced boosting strategies

**Advanced Boosting Configurations**:

```python
class GOSSConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Gradient-based One-Side Sampling configuration."""
    enabled: bool = False
    top_rate: float = 0.2                      # Keep top 20% by gradient
    other_rate: float = 0.1                    # Sample 10% of remainder

class DARTConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Dropout for Additive Regression Trees configuration."""
    enabled: bool = False
    drop_rate: float = 0.1                     # Tree dropout rate
    max_drop: int = 50                         # Max trees to drop

class EFBConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Exclusive Feature Bundling configuration."""
    enabled: bool = True                       # On by default
    max_conflict_rate: float = 0.0             # No conflicts allowed
```

**Main Config**:

```python
class LightGBMTrainingConfig(MLTrainingConfig, kw_only=True, frozen=True):
    n_estimators: PositiveInt = 100
    max_depth: PositiveInt = 6
    learning_rate: PositiveFloat = 0.1
    num_leaves: PositiveInt = 31
    min_child_samples: PositiveInt = 20
    min_child_weight: NonNegativeFloat = 1e-3

    # Advanced boosting strategies
    goss_config: GOSSConfig | None = None
    dart_config: DARTConfig | None = None
    efb_config: EFBConfig | None = None

    # Memory and performance
    force_col_wise: bool = False
    force_row_wise: bool = False
    categorical_features: list[str] = []

    # Hardware integration
    gpu_config: LightGBMGPUConfig | None = None
```

**Environment overrides**: `LightGBMTrainingConfig.from_env()` maps environment variables into the full configuration surface, including boosting (`ML_LGBM_*`), Optuna (`ML_OPTUNA_*`), GPU (`ML_LGBM_GPU_*`), and shared advanced training knobs (`ML_TRAIN_*`). Sub-configs (GOSS/DART/EFB, GPU, Optuna, AdvancedTrainingConfig) are instantiated automatically when their prefixed variables are present.

### Shared Training Configs (shared.py)

#### OptunaConfig (shared.py lines 15-83)

**Status**: ✅ **Fully Implemented** - Comprehensive HPO configuration with validation

```python
class OptunaConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Optuna hyperparameter optimization configuration."""
    enabled: bool = False
    n_trials: int = 100
    direction: str = "maximize"
    metric: str = "sharpe_ratio"               # sharpe_ratio, accuracy, auc, rmse, mae, r2
    pruner: str = "median"                     # median, percentile, hyperband, none
    sampler: str = "tpe"                       # tpe, random, cmaes, grid
    timeout: int | None = None
    study_name: str | None = None
    storage_url: str | None = None

**Environment overrides**: `OptunaConfig.from_env()` now consumes `ML_OPTUNA_*` keys (enabled, trials, direction, sampler/pruner, timeout, storage), providing parity with CLI flags.
```

#### GPU Configuration Hierarchy (shared.py lines 88-163)

```python
class BaseGPUConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Base GPU acceleration configuration."""
    enabled: bool = False
    device_id: int = 0
    validate_gpu: bool = True

class XGBoostGPUConfig(BaseGPUConfig, kw_only=True, frozen=True):
    """GPU acceleration for XGBoost."""
    max_bin: int = 256                         # Histogram bins
    predictor: str = "gpu_predictor"           # gpu_predictor | cpu_predictor

class LightGBMGPUConfig(BaseGPUConfig, kw_only=True, frozen=True):
    """GPU acceleration for LightGBM (OpenCL)."""
    platform_id: int = -1                      # -1 = auto-detect
    gpu_use_dp: bool = False                   # Double precision
```

**Environment overrides**: All GPU configs expose `from_env()`—`BaseGPUConfig` consumes `{prefix}_ENABLED`, `{prefix}_DEVICE_ID`, `{prefix}_VALIDATE`, while framework-specific configs add `{prefix}_PREDICTOR`, `{prefix}_MAX_BIN`, and `{prefix}_USE_DP`/`{prefix}_PLATFORM_ID`.

#### AdvancedTrainingConfig (shared.py lines 165-228)

```python
class AdvancedTrainingConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Advanced training features shared across ML frameworks."""
    # Feature monitoring
    track_feature_decay: bool = True
    feature_decay_threshold: float = 0.3
    feature_history_window: int = 10

    # Cross-validation
    cv_strategy: str = "time_series"           # time_series, blocked, purged, standard
    cv_folds: int = 5
    purge_gap: int = 10

    # ONNX export
    export_onnx: bool = False
    onnx_output_path: str | None = None

    # Monitoring
    enable_monitoring: bool = True

**Environment overrides**: `AdvancedTrainingConfig.from_env()` accepts `ML_TRAIN_*` variables (`TRACK_FEATURE_DECAY`, `CV_STRATEGY`, `PURGE_GAP`, `EXPORT_ONNX`, etc.) so cross-validation and monitoring can be tuned without touching Python.
```

## Runtime and System Configurations

### ObservabilityConfig (observability.py)

**Status**: ✅ **Fully Implemented** - Off hot-path observability with environment integration

```python
class ObservabilityConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for the Observability flusher/sink (off hot-path)."""
    sink: Literal["file", "db"] = "file"
    base_path: str = "./observability"
    file_format: str = "jsonl"
    db_connection_string: str | None = None
    interval_seconds: PositiveFloat = 60.0

    # Async worker options
    async_enabled: bool = False
    async_queue_maxsize: int = 4096
    async_component_label: str = "obs_async_worker"

    # Environment variable overrides
    _ENV_MAPPING: ClassVar[dict[str, str]] = {
        "sink": "ML_OBS_SINK",
        "base_path": "ML_OBS_BASE_PATH",
        "file_format": "ML_OBS_FILE_FORMAT",
        "db_connection_string": "ML_OBS_DB_URL",
        "interval_seconds": "ML_OBS_INTERVAL_SECONDS",
        "async_enabled": "ML_OBS_ASYNC_ENABLE",
        "async_queue_maxsize": "ML_OBS_ASYNC_QUEUE_MAX",
        "async_component_label": "ML_OBS_ASYNC_COMPONENT",
    }

    @classmethod
    def from_env(cls) -> ObservabilityConfig:
        """Build ObservabilityConfig from environment variables if present."""
        # ... implementation lines 51-111
```

**Environment Integration** (lines 51-111):
- Type-safe parsing with validation
- Fallback to defaults for invalid values
- Support for boolean flags ("1", "true", "yes", "y", "on")
- Explicit typing for msgspec compatibility

### MessageBusConfig (bus.py)

**Status**: ✅ **Fully Implemented** - Optional ML message bus with Redis Streams

```python
BusBackend = Literal["noop", "redis"]
TopicScheme = Literal["domain_op", "stage_first"]

@dataclass(frozen=True)
class MessageBusConfig:
    """Message bus configuration parsed from environment or provided explicitly."""
    enabled: bool = False
    backend: BusBackend = "noop"
    scheme: TopicScheme = "domain_op"
    topic_prefix: str = "events.ml"
    redis_url: str = "redis://localhost:6379/0"
    redis_stream: str = "ml-events"
    redis_maxlen: int | None = None

    @staticmethod
    def from_env() -> MessageBusConfig:
        """Construct configuration from environment variables."""
        # Environment variables (lines 65-73):
        # - ML_BUS_ENABLE: bool (default: false)
        # - ML_BUS_BACKEND: "noop" | "redis"
        # - ML_BUS_SCHEME: "domain_op" | "stage_first"
        # - ML_BUS_TOPIC_PREFIX: str
        # - ML_BUS_REDIS_URL, ML_BUS_REDIS_STREAM, ML_BUS_REDIS_MAXLEN
```

**Topic Schemes**:
- `domain_op`: "ml.features.live.computed" (canonical)
- `stage_first`: "events.ml.FEATURE_COMPUTED.live"

**Integration**: Used by stores, registries, orchestration for event emission.

### OnnxRuntimeConfig (runtime.py, base.py)

**Status**: ❌ **Placeholder** - Empty class for backward compatibility

```python
class OnnxRuntimeConfig(NautilusConfig, kw_only=True, frozen=True):
    """ONNX Runtime configuration placeholder for backward-compat imports."""
    # EMPTY - all documented fields missing
```

**Missing**:
- graph_optimization_level, execution_mode, providers
- to_session_options() helper method
- All runtime configuration

## Streaming Pipeline Configs (streaming_pipeline.py - NEW)

### DatasetServiceConfig (lines 17-33)

**Status**: ✅ **Fully Implemented** - Dataset planning microservice

```python
class DatasetServiceConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for the dataset planning microservice."""
    parquet_root: str
    shard_row_budget: PositiveInt = 200_000
    max_total_rows: PositiveInt | None = None
    max_total_sequences: PositiveInt | None = None
    max_shards: PositiveInt | None = None
    plan_cache_ttl_seconds: PositiveInt = 600
    retry_backoff_seconds: PositiveFloat = 5.0
    max_retry_attempts: PositiveInt = 3

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.max_total_rows is not None and self.max_total_rows < self.shard_row_budget:
            raise ValidationError("max_total_rows must be >= shard_row_budget when set")
```

### StreamingWorkerConfig (lines 35-76)

**Status**: ✅ **Fully Implemented** - Bounded streaming training workers

```python
class StreamingWorkerConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for bounded streaming training workers."""
    max_total_rows: PositiveInt | None = 500_000
    max_total_sequences: PositiveInt | None = 300_000
    max_shards: PositiveInt | None = 4
    max_epochs: PositiveInt = 1
    max_concurrent_jobs: PositiveInt = 1
    max_runtime_seconds: PositiveInt = 1_800
    heartbeat_interval_seconds: PositiveInt = 30
    max_retry_attempts: PositiveInt = 3
    retry_backoff_seconds: NonNegativeFloat = 5.0
    accelerator: str = "auto"
    devices: PositiveInt = 1
    model_id: str = "tft-streaming-teacher"
    train_fraction: PositiveFloat = 0.8
    logits_artifact_key: str = "logits"
    validation_metric: str = "roc_auc"
    gpu_memory_monitor_interval_seconds: NonNegativeFloat | None = 30.0

    def __post_init__(self) -> None:
        """Ensure time and resource constraints are consistent."""
        if self.max_concurrent_jobs > 1 and self.max_shards is not None:
            if self.max_shards < self.max_concurrent_jobs:
                raise ValidationError("max_shards must be >= max_concurrent_jobs when both set")
        if not (0.0 < float(self.train_fraction) < 1.0):
            raise ValidationError("train_fraction must be in (0, 1)")
        if int(self.max_epochs) < 1:
            raise ValidationError("max_epochs must be >= 1")
        # ... additional validation
```

### TrainingOrchestratorConfig (lines 78-100)

**Status**: ✅ **Fully Implemented** - Event-driven streaming orchestrator

```python
class TrainingOrchestratorConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for event-driven streaming orchestrator."""
    command_topic: str
    result_topic: str
    heartbeat_topic: str
    max_in_flight_plans: PositiveInt = 8
    dataset_retry_limit: PositiveInt = 2
    worker_timeout_seconds: PositiveInt = 600
    retry_window_seconds: PositiveInt = 300
    max_plan_age_seconds: PositiveInt = 7_200
    saturation_heartbeat_limit: PositiveInt = 5
    backlog_warning_threshold: NonNegativeInt = 10
    enable_state_persistence: bool = True

    def __post_init__(self) -> None:
        """Validate orchestrator settings."""
        if self.command_topic == self.result_topic:
            raise ValidationError("command_topic and result_topic must differ")
        # ... validation for all topic uniqueness
```

## System Constants (constants.py)

### ExportFormats (lines 18-28)

```python
class ExportFormats(Enum):
    """Supported model export formats."""
    ONNX = "onnx"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    PICKLE = "pickle"
    JSON = "json"

SUFFIX_ONNX = ".onnx"
SUFFIX_XGB = ".xgb"
SUFFIX_LGB = ".lgb"
```

### Versions (lines 36-45)

```python
class Versions:
    """Version constants for ML components."""
    ONNX_OPSET = 17
    LIGHTGBM_MIN = "3.0.0"
    XGBOOST_MIN = "1.6.0"
    DEFAULT_MANIFEST_VERSION = "1.0.0"
    DEFAULT_TRAINER_VERSION = "1.0.0"
```

### TimeConstants (lines 62-78)

```python
class TimeConstants:
    """Time-related constants for ML components."""
    NS_IN_SECOND: Final[int] = 1_000_000_000
    NS_IN_MINUTE: Final[int] = 60 * NS_IN_SECOND
    NS_IN_HOUR: Final[int] = 3600 * NS_IN_SECOND
    NS_IN_DAY: Final[int] = 86400 * NS_IN_SECOND
    SECONDS_IN_DAY: Final[int] = 86400
    DAYS_PER_WEEK: Final[int] = 7

    # Trading calendar
    TRADING_DAYS_PER_YEAR: Final[int] = 252
    TRADING_HOURS_PER_DAY: Final[float] = 6.5
    TRADING_WEEKS_PER_YEAR: Final[int] = 52
```

### MLConstants (lines 126-151)

```python
class MLConstants:
    """Machine learning related constants."""
    # Data windows
    DEFAULT_LOOKBACK_DAYS: Final[int] = 252    # 1 year
    MIN_LOOKBACK_DAYS: Final[int] = 20
    MAX_LOOKBACK_DAYS: Final[int] = 1260       # 5 years

    # Model thresholds
    DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.6
    MIN_CONFIDENCE_THRESHOLD: Final[float] = 0.5
    HIGH_CONFIDENCE_THRESHOLD: Final[float] = 0.8

    # Feature engineering
    MAX_LAG_FEATURES: Final[int] = 10
    DEFAULT_LAG_PERIODS: Final[list[int]] = [1, 2, 3, 5, 10, 20]

    # Performance monitoring
    MAX_INFERENCE_LATENCY_MS: Final[float] = 5.0
    PERFORMANCE_REGRESSION_THRESHOLD: Final[float] = 0.2  # 20% allowed

    # Feature parity validation
    FEATURE_PARITY_TOLERANCE: Final[float] = 1e-10
```

### FeatureColumns and IndicatorNames (lines 179-271)

```python
class FeatureColumns(str, Enum):
    """Standard feature column names."""
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    RETURNS = "returns"
    LOG_RETURNS = "log_returns"
    PRICE = "price"
    TIMESTAMP = "timestamp"
    DATE = "date"
    TIME = "time"
    TARGET = "target"
    SIGNAL = "signal"
    PREDICTION = "prediction"

class IndicatorNames:
    """Standard indicator naming patterns."""
    SMA_PREFIX = "sma_"
    EMA_PREFIX = "ema_"
    PRICE_SMA_5 = "price_sma_5"
    PRICE_SMA_20 = "price_sma_20"
    # ... many more predefined names

    @staticmethod
    def lag_feature(period: int) -> str:
        """Generate lag feature name."""
        return f"{IndicatorNames.LAG_PREFIX}{period}"

    @staticmethod
    def feature_index(index: int) -> str:
        """Generate indexed feature name."""
        return f"{IndicatorNames.FEATURE_PREFIX}{index}"
```

## Configuration Loading (loader.py)

### Layered Merge System (lines 1-73)

**Priority Hierarchy** (lowest to highest):
1. Code defaults (lowest priority)
2. File-based configuration (YAML/JSON)
3. Environment variables (ML_ prefix)
4. CLI overrides (highest priority)

```python
def load_from_file(path: str | None, t: type[T], default: T) -> T:
    """Load JSON config from file and decode into type ``t`` with fallback."""
    if not path:
        return default
    try:
        with open(path) as f:
            raw = f.read()
        data = json.loads(raw)
        return msgspec.json.decode(msgspec.json.encode(data), type=t)
    except Exception:
        return default

def merge_env(prefix: str, t: type[T], base: T) -> T:
    """Overlay JSON from environment variable ``{PREFIX}_JSON`` onto ``base``."""
    key = f"{prefix}_JSON"
    blob = os.getenv(key)
    if not blob:
        return base
    try:
        data = json.loads(blob)
        partial = msgspec.json.decode(msgspec.json.encode(data), type=t)
        return cast(T, _merge_structs(base, partial))
    except Exception:
        return base
```

**Pattern**: Best-effort shallow merge for msgspec structs with type safety.

## Orchestrator TOML Configurations

### Production Full Config (orchestrator/production_full.toml)

**Purpose**: 95-symbol production pipeline with 30-macro-series comprehensive feature set

**Key Sections** (183 lines total):

```toml
stage = "full"

[ingestion]
enabled = true
dataset_id = "EQUS.MINI"
schema = "bars"
coverage_mode = "sql"
write_mode = "sql+datastore"
lookback_days = 2
symbols = [
    "AAPL","ABBV","ABT","ACN","ADBE","AMAT","AMD","AMZN","AVGO","BA",
    "BAC","BRK.B","C","CAT","COIN","COP","COST","CRM","CRWD","CVX",
    # ... 95 total symbols
]

[dataset]
data_dir = "data/tier1"
symbols = "AAPL,ABBV,ABT,..."  # 95 comma-separated
out_dir = "ml_out/production_full"
dataset_id = "production_full_v1"
convert_vintage_to_age = true

# Time range - ACTUAL DATA: March 28, 2023 → September 30, 2025 (2.5 years!)
start_iso = "2023-03-28T00:00:00Z"
end_iso = "2025-09-30T23:59:59Z"

# Feature flags
include_macro = true
include_calendar = true
include_events = true
include_l2 = false
include_micro = false

# Macro configuration
macro_lag_days = 1
auto_refresh_macro = true
macro_staleness_hours = 24
macro_fred_path = "data/fred/fred_indicators_ml_format.parquet"

# COMPREHENSIVE MACRO SERIES (30 series across 6 dimensions)
macro_series_ids = [
    # RATES / DURATION CURVE (7 series)
    "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "DFII10", "FEDFUNDS",

    # CREDIT / CAPITAL MARKETS RISK (4 series)
    "BAMLC0A0CM", "BAMLH0A0HYM2", "TEDRATE", "VIXCLS",

    # GROWTH / LABOR CYCLE (4 series)
    "PAYEMS", "UNRATE", "INDPRO", "CFNAI",

    # INFLATION & COST PRESSURE (4 series)
    "CPIAUCSL", "PCEPI", "PPIACO", "WTISPLC",

    # COMMODITIES / METALS (3 series)
    "PALLFNFINDEXM", "PCOPPUSDM", "NASDAQQGLDI",

    # CROSS-ASSET / RISK ROTATION (4 series)
    "DTWEXBGS", "DEXUSAL", "DEXUSEU", "DEXJPUS",

    # LIQUIDITY / BALANCE SHEET (2 series)
    "WALCL", "TOTBKCR",
]

lookback_periods = 60
horizon_minutes = 60
threshold = 0.0005
chunk_days = 30

register_features = true
feature_registry_dir = "~/.nautilus/ml/features"
feature_role = "teacher"

[dataset.validation]
min_rows = 10000
min_positive_rate = 0.35
max_positive_rate = 0.65
min_feature_coverage = 0.95
vintage_policy = "real_time"
emit_dataset_events = true

[hpo]
enabled = false

[training.teacher]
enabled = true
model_id = "tft_spy_full_macro_v1"
max_epochs = 50
feature_registry_dir = "~/.nautilus/ml/features"

[training.student]
enabled = true
model_id = "xgb_spy_full_macro_v1"
parent_model_id = "tft_spy_full_macro_v1"
model_registry_dir = "~/.nautilus/ml/models"
feature_registry_dir = "~/.nautilus/ml/features"
objective = "logit_mse"
kd_lambda = 0.5
early_stopping = 200
opset = 17

[promotions]
auto_register_model = true
auto_promote = false
gates_json = '{"min_sharpe": 0.5, "min_win_rate": 0.48}'

[integration]
enabled = true
db_connection = "postgresql://postgres:postgres@localhost:5433/nautilus"
auto_migrate = true
ensure_healthy = true
```

**Real Production Deployment**:
- 95 instruments including equities, ETFs, commodities, FX, volatility
- 30 macro features across 6 economic dimensions
- 2.5 years of historical data (March 2023 → September 2025)
- Vintage age conversion for real-time macro data
- Teacher-student distillation (TFT → XGBoost)
- Promotion gates with performance thresholds

## Configuration Validation

### validate_ml_config() (__init__.py lines 128-171)

**Purpose**: Validate ML configuration for Universal Pattern compliance

```python
def validate_ml_config(config: MLActorConfig | MLInferenceConfig) -> list[str]:
    """
    Validate ML configuration for compliance with Universal Patterns.

    Returns
    -------
    list[str]
        List of validation issues. Empty list indicates valid configuration.
    """
    issues = []

    # Pattern 1: Validate model specification
    if hasattr(config, "model_path") and hasattr(config, "model_id"):
        if not config.model_path and not config.model_id:
            issues.append("Either model_path or model_id must be provided")
        if config.model_path and config.model_id:
            issues.append("Cannot specify both model_path and model_id")

    # Pattern 3: Validate hot path constraints
    if hasattr(config, "max_inference_latency_ms"):
        if config.max_inference_latency_ms > 5.0:
            issues.append(
                f"max_inference_latency_ms ({config.max_inference_latency_ms}) exceeds 5ms SLA"
            )

    if hasattr(config, "max_feature_latency_ms"):
        if config.max_feature_latency_ms > 0.5:
            issues.append(
                f"max_feature_latency_ms ({config.max_feature_latency_ms}) exceeds 0.5ms SLA"
            )

    # Pattern 4: Validate fallback configuration
    if hasattr(config, "use_dummy_stores") and hasattr(config, "db_connection"):
        if not config.use_dummy_stores and not config.db_connection:
            issues.append("db_connection required when use_dummy_stores=False")

    return issues
```

**Usage**: Only used in __init__.py itself - **not widely adopted** in codebase.

### get_config_defaults() (__init__.py lines 173-193)

```python
def get_config_defaults() -> dict[str, object]:
    """Get default configuration values for common ML configurations."""
    return {
        "ml_feature": MLFeatureConfig(),
        "ml_inference": MLInferenceConfig(model_path="./models/default.onnx"),
        "optimization": OptimizationConfig(),
        "strategy": StrategyConfig(),
        "onnx_runtime": OnnxRuntimeConfig(),
        "model_registry": ModelRegistryConfig(),
        "observability": ObservabilityConfig(),
        "message_bus": MessageBusConfig(),
    }
```

## Environment Variable Patterns

### from_env() Implementations

**Implemented** (3 configs):
1. **ObservabilityConfig.from_env()** (observability.py lines 51-111)
   - 8 ML_OBS_* variables with type-safe parsing
   - Boolean, float, and string handling
   - Fallback to defaults for invalid values

2. **MessageBusConfig.from_env()** (bus.py lines 61-100)
   - 7 ML_BUS_* variables
   - Backend and scheme enum validation
   - Optional redis_maxlen parsing

3. **DataCollectorConfig** (base.py lines 231-291)
   - 3 ML_DATA_* variables via _ENV_MAPPING
   - Legacy fallback support via _LEGACY_ENV_MAPPING
   - Type coercion with try/except

**Pattern Example** (ObservabilityConfig):

```python
_ENV_MAPPING: ClassVar[dict[str, str]] = {
    "sink": "ML_OBS_SINK",
    "base_path": "ML_OBS_BASE_PATH",
    "file_format": "ML_OBS_FILE_FORMAT",
    "db_connection_string": "ML_OBS_DB_URL",
    "interval_seconds": "ML_OBS_INTERVAL_SECONDS",
    "async_enabled": "ML_OBS_ASYNC_ENABLE",
    "async_queue_maxsize": "ML_OBS_ASYNC_QUEUE_MAX",
    "async_component_label": "ML_OBS_ASYNC_COMPONENT",
}

@classmethod
def from_env(cls) -> ObservabilityConfig:
    """Build ObservabilityConfig from environment variables if present."""
    import os

    kwargs: dict[str, object] = {}
    for field, env_var in cls._ENV_MAPPING.items():
        if env_var in os.environ:
            val: str = os.environ[env_var]
            if field == "interval_seconds":
                try:
                    kwargs[field] = float(val)
                except ValueError:
                    continue
            elif field == "sink":
                if val in {"file", "db"}:
                    kwargs[field] = val
                else:
                    continue
            elif field == "async_enabled":
                kwargs[field] = val.strip().lower() in {"1", "true", "yes", "y", "on"}
            # ... more field-specific handling
```

**Missing from_env()** (9+ major configs):
- MLActorConfig, MLFeatureConfig, MLInferenceConfig
- XGBoostTrainingConfig, LightGBMTrainingConfig
- MLSignalActorConfig, MLStrategyConfig
- All streaming pipeline configs

## Integration Points

### Nautilus Trader Integration

- **Base Types**: All configs extend `NautilusConfig` from `nautilus_trader.common.config`
- **Validation Types**: `PositiveInt`, `NonNegativeFloat`, `PositiveFloat` from Nautilus
- **Model Types**: `InstrumentId`, `BarType`, `ComponentId` from `nautilus_trader.model`
- **Actor Lifecycle**: `ActorConfig` mapping via adapters.py

### 4-Store + 4-Registry Integration

**Configuration Fields** (present but not enforced):

```python
# Store configuration (in MLActorConfig, MLInferenceConfig, MLSignalActorConfig)
db_connection: str | None = None
use_dummy_stores: bool = False

# Registry configuration (in MLInferenceConfig, MLSignalActorConfig)
registry_path: str | None = None
use_registry_features: bool = False
feature_set_id: str | None = None
```

**Progressive Fallback** (configured but not validated):
- PostgreSQL → DummyStore with warnings
- Registry loading → Direct file loading
- Network failures → Local caches

**Missing**:
- No validation in `__post_init__()` to enforce 4-store pattern
- No protocol compliance checking at config level
- No automatic fallback configuration generation

### Event System Integration

**Usage Statistics**: 409 total occurrences across 91 files

**Key Integration Points**:
- **Stores**: All store implementations use EventStage for state transitions
- **Registries**: Event managers use Source/Status for persistence
- **Orchestration**: Pipeline orchestrators emit Stage events
- **Training**: Streaming workers emit heartbeats and status events
- **Tests**: 55 test files use events for contracts and validation

**Topic Construction**:

```python
from ml.common.message_topics import build_topic_for_stage
from ml.config import EventStage, EventSource

# Domain-op scheme (canonical)
topic = build_topic_for_stage(EventStage.FEATURE_COMPUTED, EventSource.LIVE)
# Result: "ml.features.live.computed"

# Stage-first scheme (alternative)
config = MessageBusConfig(scheme="stage_first", topic_prefix="events.ml")
# Result: "events.ml.FEATURE_COMPUTED.live"
```

## Implementation Patterns

### Frozen Dataclass Pattern

**All configs use**:
```python
class MyConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration class."""
    field: PositiveInt = 100
```

**Benefits**:
- Immutability after construction
- Type safety via annotations
- msgspec serialization support
- Keyword-only arguments (clarity)

### Validation in __post_init__()

**Pattern**:
```python
def __post_init__(self) -> None:
    """Validate configuration."""
    if self.n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {self.n_trials}")

    valid_options = ["option1", "option2"]
    if self.choice not in valid_options:
        raise ValueError(f"choice must be one of {valid_options}, got {self.choice}")
```

**Coverage**:
- ✅ OptunaConfig: 5 validation rules
- ✅ AdvancedTrainingConfig: 4 validation rules
- ✅ StreamingWorkerConfig: 7 validation rules
- ✅ TrainingOrchestratorConfig: 3 validation rules
- ✅ CircuitBreakerConfig: 1 legacy alias mapping
- ✅ MLInferenceConfig: 3 model loading rules
- ❌ MLActorConfig: No validation (despite complexity)
- ❌ MLFeatureConfig: No validation
- ❌ XGBoostTrainingConfig: No validation

### Backward Compatibility Pattern

**Strategy**: Use `object.__setattr__()` in frozen dataclasses

```python
def __post_init__(self) -> None:
    """Map backward-compat alias fields to canonical fields while frozen."""
    # Map legacy field to canonical field
    if self.optimization is not None and self.optimization_config is None:
        object.__setattr__(self, "optimization_config", self.optimization)

    # Merge legacy thresholds
    if self.strategy and self.strategy.threshold_long:
        merged = max(abs(self.strategy.threshold_long or 0.0),
                     abs(self.strategy.threshold_short or 0.0))
        object.__setattr__(self, "prediction_threshold", merged)
```

**Used in**:
- CircuitBreakerConfig (half_open_attempts → success_threshold)
- MLSignalActorConfig (optimization/strategy/signal_policy aliases)
- DataCollectorConfig (legacy env var fallbacks)

### Circular Import Avoidance

**Pattern** (actors.py lines 17-23):

```python
if TYPE_CHECKING:
    from ml.actors.signal import OptimizationLevel as _OptimizationLevel
    from ml.actors.signal import SignalStrategy as _SignalStrategy
else:  # pragma: no cover
    _OptimizationLevel = object  # type: ignore
    _SignalStrategy = object     # type: ignore

# Usage in type annotations
signal_strategy: Literal["threshold", ...] | _SignalStrategy = "threshold"
```

**Why**: Avoid runtime circular dependencies while maintaining type hints.

**Also used in** base.py lines 581-586 for strategy component configs.

## Known Gaps and Incomplete Work

### Critical Gaps

1. **OnnxRuntimeConfig** (runtime.py lines 316-319)
   - Empty placeholder class
   - All documented fields missing (graph_optimization_level, execution_mode, providers)
   - No to_session_options() helper
   - **Impact**: Cannot configure ONNX Runtime beyond defaults

2. **Missing Environment Support** (9 major configs)
   - MLActorConfig, MLFeatureConfig, MLInferenceConfig
   - XGBoostTrainingConfig, LightGBMTrainingConfig
   - MLSignalActorConfig, MLStrategyConfig
   - All streaming pipeline configs
   - **Impact**: Cannot override via environment in production deployments

3. **No Universal Pattern Enforcement**
   - validate_ml_config() exists but not widely used
   - No 4-store validation in __post_init__()
   - No protocol compliance checking
   - **Impact**: Configs don't enforce architectural patterns

### Validation Gaps

1. **MLActorConfig** (base.py lines 146-229)
   - No __post_init__() despite 30+ fields
   - No validation of health_config interactions
   - No validation of async persistence settings
   - **Impact**: Invalid configs can be created

2. **XGBoostTrainingConfig** (xgboost.py)
   - No validation of tree_method compatibility with gpu_config
   - No validation of monotonic_constraints format
   - No validation of multi_asset requirements
   - **Impact**: Runtime errors instead of config errors

3. **DataCollectorConfig** (base.py lines 231-291)
   - Environment override in __post_init__() breaks separation
   - No validation of end_date_iso format
   - **Impact**: Confusing runtime behavior

### Documentation Gaps

1. **File Structure Discrepancies**
   - Sep 19 doc lists files that don't exist
   - Missing new files (streaming_pipeline.py, playground.py)
   - Incorrect line counts for several files

2. **Missing Usage Examples**
   - No examples of TOML orchestrator config usage
   - No examples of streaming pipeline config
   - No examples of market_data descriptor usage

3. **Integration Documentation**
   - Event system usage not documented (despite 409 occurrences)
   - TOML validation process not explained
   - Streaming architecture not connected to configs

## Production Deployment Reality

### Actual Deployments

**production_full.toml** (183 lines):
- 95 instruments across equities, ETFs, commodities, FX, volatility
- 30 macro features across 6 economic dimensions
- 2.5 years of historical data (March 2023 → September 2025)
- Teacher-student distillation (TFT → XGBoost)
- PostgreSQL backend with auto-migration
- Promotion gates: min_sharpe=0.5, min_win_rate=0.48

**spy_production_full_macro.toml** (171 lines):
- Similar structure, SPY-focused
- Comprehensive macro feature engineering
- Vintage age conversion for real-time data

**spy_with_revisions.toml** (133 lines):
- Focused on macro revision tracking
- Real-time vintage data handling

### Environment Configuration in Practice

**Observability** (ML_OBS_* variables):
```bash
ML_OBS_SINK="db"
ML_OBS_DB_URL="postgresql://prod:secret@db.example.com:5432/nautilus"
ML_OBS_INTERVAL_SECONDS="30.0"
ML_OBS_ASYNC_ENABLE="true"
ML_OBS_ASYNC_QUEUE_MAX="8192"
```

**Message Bus** (ML_BUS_* variables):
```bash
ML_BUS_ENABLE="true"
ML_BUS_BACKEND="redis"
ML_BUS_REDIS_URL="redis://redis.example.com:6379/0"
ML_BUS_SCHEME="domain_op"
ML_BUS_TOPIC_PREFIX="events.ml"
```

**Data Collection** (ML_DATA_* variables):
```bash
ML_DATA_TIER1_DIR="/mnt/fast-ssd/tier1"
ML_STORAGE_LIMIT_GB="1000.0"
ML_END_DATE="2025-09-30"
```

## Summary and Recommendations

### Current State

**Strengths**:
- ✅ Comprehensive type-safe configuration system with 4,433 lines
- ✅ Production-proven with 95-instrument deployments
- ✅ Strong event system (409 occurrences, 91 files)
- ✅ Advanced training configs (XGBoost, LightGBM with GOSS/DART/EFB)
- ✅ Environment integration for runtime configs (3/3 system configs)
- ✅ TOML orchestrator configs for complex pipelines
- ✅ Frozen dataclass pattern enforced throughout
- ✅ Streaming pipeline configs for event-driven training

**Weaknesses**:
- ❌ Incomplete environment support (3/12 major configs)
- ❌ OnnxRuntimeConfig is empty placeholder
- ❌ No Universal Pattern enforcement at config level
- ❌ Inconsistent validation (__post_init__() coverage ~40%)
- ❌ Documentation out of sync with code (Sep 19 doc 25% outdated)
- ❌ validate_ml_config() exists but not widely used

### Implementation Status: 80% Complete

**Fully Implemented** (11 components):
- Core configs: MLFeatureConfig, MLInferenceConfig, MLStrategyConfig
- Training: XGBoostTrainingConfig, LightGBMTrainingConfig, OptunaConfig, AdvancedTrainingConfig
- System: ObservabilityConfig, MessageBusConfig, DataCollectorConfig
- Streaming: DatasetServiceConfig, StreamingWorkerConfig, TrainingOrchestratorConfig
- Event system: Stage, Source, EventStatus

**Partially Implemented** (2 components):
- MLActorConfig: Core fields complete, no validation
- MLSignalActorConfig: Full features, extensive backward compat

**Placeholder/Missing** (3 components):
- OnnxRuntimeConfig: Empty class
- Environment support: Missing from 9 major configs
- Pattern enforcement: Not integrated into configs

### Priority Recommendations

**High Priority** (Production Blockers):
1. Implement OnnxRuntimeConfig with all fields and to_session_options()
2. Add from_env() to MLActorConfig, MLFeatureConfig, XGBoostTrainingConfig
3. Add __post_init__() validation to MLActorConfig (30+ fields, no validation)
4. Integrate validate_ml_config() into actor initialization paths

**Medium Priority** (Quality Improvements):
1. Add environment support to remaining 6 major configs
2. Add validation to XGBoostTrainingConfig and LightGBMTrainingConfig
3. Document event system usage patterns (409 occurrences deserve docs)
4. Document TOML orchestrator config workflow

**Low Priority** (Nice to Have):
1. Resolve circular import patterns (move to protocols)
2. Standardize backward compatibility approach
3. Add protocol enforcement at config level
4. Clean up DataCollectorConfig environment override in __post_init__()

The configuration system provides a solid, production-ready foundation with comprehensive coverage and real deployment success, but requires focused effort on environment support, validation coverage, and documentation updates to reach full maturity.
