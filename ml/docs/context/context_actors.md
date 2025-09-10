# ML Actors Context Documentation

## Executive Summary

The ML actors framework provides a production-ready foundation for real-time machine learning inference and signal generation within Nautilus Trader. The architecture follows strict hot/cold path separation, ensuring sub-millisecond performance in production environments while maintaining comprehensive observability and fault tolerance.

**✨ ENHANCEMENT:** All actors now fully implement the 5 Universal ML Architecture Patterns defined in CLAUDE.md, ensuring consistent behavior across the ML platform.

Operational notes:

- Timestamps: Actors should emit UNIX nanoseconds for `ts_event`/`ts_init`. Stores defensively normalize smaller units (seconds/ms/us) to ns with a warning. See `context_stores.md` → "Timestamp Policy & Normalization".
- DB preflight: Verify required DB functions and current partition exist before startup. See `context_deployment.md` → "DB Preflight (recommended)".
- **📝 ADDITION:** Security: Non-ONNX model formats are restricted in production environments unless explicitly enabled via ML_TEST_ALLOW_NON_ONNX or ML_ALLOW_NON_ONNX_IN_TESTS environment variables.

**Key Components:**

- **BaseMLInferenceActor**: Abstract foundation class with mandatory store integration and production features
- **MLSignalActor**: Production signal generation actor with multiple built-in strategies
- **ONNXMLInferenceActor**: ONNX-optimized inference actor for lowest latency **✨ ENHANCEMENT:** Now includes CPU provider optimizations and configurable session options
- **EnhancedMLInferenceActor**: **🔄 UPDATE:** Minimal test-focused implementation showcasing hot-path optimization with zero-allocation feature computation
- **PickleMLInferenceActor**: **⚠️ CORRECTION:** DEPRECATED - Raises SecurityError to prevent insecure model loading
- **Hot Path Optimization**: Zero-allocation inference with <5ms end-to-end latency targets
- **Mandatory 4-Store + 4-Registry Pattern**: Complete data lifecycle management with automatic initialization

## Architecture Overview

### Actor Hierarchy

```
BaseMLInferenceActor (Abstract)
├── Mandatory Features:
│   ├── Store Integration (FeatureStore, ModelStore, StrategyStore, DataStore)
│   ├── Registry Integration (FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry)
│   ├── Health Monitoring & Circuit Breaker
│   ├── Model Hot-Reloading with State Preservation
│   ├── Prometheus Metrics Integration
│   └── Registry-Based Model/Feature Loading
│
├── Abstract Methods:
│   ├── _load_model()
│   ├── _initialize_features()
│   ├── _compute_features(bar) -> features | None
│   └── _predict(features) -> (prediction, confidence)
│
├── Concrete Implementations:
│   ├── MLSignalActor: Production signal generation with configurable strategies **✨ ENHANCEMENT:** Features adaptive thresholds, ensemble strategies, and hot-reload capability
│   ├── ONNXMLInferenceActor: ONNX-optimized inference for sub-millisecond latency **📝 ADDITION:** Supports configurable runtime providers and session options
│   ├── EnhancedMLInferenceActor: **🔄 UPDATE:** Minimal test implementation with zero-allocation feature computation and null store protocols
│   └── PickleMLInferenceActor: DEPRECATED - raises SecurityError (security risk)
│
└── Notes:
    └── **⚠️ CORRECTION:** Pickle formats are intentionally disallowed for production actors (security)
        — Production environments enforce ONNX-only unless ML_TEST_ALLOW_NON_ONNX is set
        — **📝 ADDITION:** Development environments can use allow_non_onnx_in_dev config flag
```

### Signal Generation Architecture

The `MLSignalActor` provides a sophisticated signal generation system with pluggable strategies:

**Built-in Strategies:**

- **ThresholdSignalStrategy**: Simple confidence-based filtering with static threshold **📝 ADDITION:** Uses configurable confidence threshold for binary signal generation
- **ExtremesStrategy**: Percentile-based signal generation using prediction extremes **✨ ENHANCEMENT:** Implements lock-free ring buffer for zero-allocation extremes detection
- **MomentumStrategy**: Trend-based signal enhancement using prediction momentum **📝 ADDITION:** Configurable lookback period and momentum threshold detection
- **EnsembleStrategy**: Weighted voting across multiple strategies **✨ ENHANCEMENT:** Supports configurable strategy weights and dynamic ensemble scoring
- **AdaptiveStrategy**: Dynamic threshold adjustment based on market volatility and prediction variance **📝 ADDITION:** Includes market regime detection and signal strength calculation

**Signal Data Classes:**

- **MLSignal**: **✨ ENHANCEMENT:** Unified signal data class extending NautilusData with complete tracking:
  - `instrument_id`: The instrument identifier (InstrumentId type)
  - `model_id`: Unique model identifier for tracking **📝 ADDITION:** Required field for model lineage
  - `prediction`: Model prediction value (float)
  - `confidence`: Confidence score (0.0 to 1.0) **📝 ADDITION:** Used for threshold filtering
  - `features`: Optional feature array for debugging (npt.NDArray[np.float32] | None)
  - `metadata`: Optional additional metadata (dict[str, Any] | None) **📝 ADDITION:** Used for adaptive strategy context
  - `ts_event`: Event timestamp in nanoseconds (int) **⚠️ CORRECTION:** Property accessor for immutable timestamp
  - `ts_init`: Initialization timestamp in nanoseconds (int) **⚠️ CORRECTION:** Property accessor for immutable timestamp
- **AdaptiveSignal**: **🔄 UPDATE:** Type alias for MLSignal (backward compatibility) - no longer separate class

## Hot Path Performance Requirements

### Latency Targets

- **P99 Feature Computation**: <500μs
- **P99 Model Inference**: <2ms
- **P99 End-to-End Signal**: <5ms
- **Memory Stability**: Zero allocations in hot path, stable over 24h operation

### Hot Path Implementation

```python
def on_bar(self, bar: Bar) -> None:
    """Hot path: Zero allocations, bounded computation time."""
    # 1. Circuit breaker check (fast exit)
    if self._circuit_breaker and not self._circuit_breaker.can_execute():
        return

    # 2. Feature computation using pre-allocated buffers
    start_time = time.perf_counter()
    features = self._compute_features(bar)  # <500μs target

    # 3. Model inference (ONNX optimized)
    prediction, confidence = self._predict(features)  # <2ms target

    # 4. Signal generation (strategy-based)
    signal = self._signal_strategy.generate_signal(...)  # <1ms target

    # 5. Store persistence (batched writes; flushed on stop)
    #    FeatureStore.write_features(...) and ModelStore.write_prediction(...)
```

### Zero-Allocation Techniques

- **Pre-allocated Feature Buffers**: `np.zeros(n_features, dtype=np.float32)` **📝 ADDITION:** Sized dynamically from FeatureEngineer.n_features
- **Rolling Windows**: Fixed-size deques with maxlen constraints **✨ ENHANCEMENT:** Also supports lock-free ring buffers for OPTIMIZED level
- **Buffer Reuse**: Feature vectors returned as views, not copies **⚠️ CORRECTION:** EnhancedMLInferenceActor ensures view semantics in _compute_features
- **ONNX Runtime Optimization**: CPU-optimized sessions with minimal memory footprint **📝 ADDITION:** Configurable providers and session options via OnnxRuntimeConfig
- **📝 ADDITION:** **Lock-Free Ring Buffers**: Available in OPTIMIZED mode via ml.core.cache components (LockFreeRingBuffer, PreAllocatedFeatureCache)
- **📝 ADDITION:** **Reservoir Sampling**: Bounded memory performance monitoring with configurable sample sizes

## Store Integration Patterns

### Mandatory 4-Store + 4-Registry Pattern

Every ML actor **MUST** initialize and use four stores and four registries for complete data lifecycle management:

```python
def _init_stores_and_registries(self) -> None:
    """MANDATORY: Initialize all stores and registries - no optional parameters allowed."""
    # STORES: Complete data lifecycle management
    # 1. FeatureStore: For training/inference parity
    self._feature_store = FeatureStore(connection_string=db_connection)

    # 2. ModelStore: For prediction tracking and performance analysis
    self._model_store = ModelStore(persistence_config=persistence_config)

    # 3. StrategyStore: For trading decisions and signal analysis
    self._strategy_store = StrategyStore(persistence_config=persistence_config)

    # 4. DataStore: Unified facade with contract validation and event emission
    self._data_store = DataStore(registry=self._data_registry, connection_string=db_connection)

    # REGISTRIES: Component lifecycle and schema management
    # 1. FeatureRegistry: Feature schema validation and lifecycle
    self._feature_registry = FeatureRegistry(registry_path, persistence_config)

    # 2. ModelRegistry: Model deployment tracking and A/B testing
    self._model_registry = ModelRegistry(registry_path, persistence_config)

    # 3. StrategyRegistry: Strategy compatibility and requirements
    self._strategy_registry = StrategyRegistry(registry_path, persistence_config)

    # 4. DataRegistry: Dataset manifest management and lineage tracking
    self._data_registry = DataRegistry(registry_path, persistence_config)
```

### Data Persistence Flow

1. **Feature Storage**: Every computed feature vector is persisted with instrument_id, ts_event, ts_init
2. **Prediction Storage**: All predictions stored with inference_time_ms, confidence, and feature_dict
3. **Signal Storage**: Generated signals with strategy metadata, risk metrics, and execution parameters
4. **Contract Validation**: DataStore validates all data against registered schemas before persistence
5. **Event Emission**: DataRegistry tracks processing events and lineage for complete observability
6. **Schema Management**: Registries enforce compatibility between components and track evolution

### Store Usage in Hot Path

```python
# MANDATORY: Store features for parity tracking
feature_dict = {f"feature_{i}": float(v) for i, v in enumerate(features)}
self._feature_store.write_features(
    feature_set_id=getattr(self._config, "feature_set_id", "default"),  # **⚠️ CORRECTION:** Safe attribute access
    instrument_id=str(bar.bar_type.instrument_id),
    features=feature_dict,
    ts_event=bar.ts_event,
    ts_init=bar.ts_init,
)

# MANDATORY: Store prediction for performance tracking
self._model_store.write_prediction(
    model_id=self._model_id,  # **📝 ADDITION:** Determined from metadata, training_metadata, or path fallback
    instrument_id=str(bar.bar_type.instrument_id),
    prediction=float(prediction),
    confidence=float(confidence),
    features=feature_dict,
    inference_time_ms=inference_time,
    ts_event=bar.ts_event,
)

# **📝 ADDITION:** MANDATORY: Store signals for strategy analysis
self._strategy_store.write_signal(
    strategy_id=str(self.id),
    instrument_id=str(bar.bar_type.instrument_id),
    signal_type="buy" if prediction > 0 else "sell",
    strength=abs(prediction),
    model_predictions={self._model_id: prediction},
    risk_metrics={"confidence": confidence},
    execution_params={"threshold": adaptive_threshold},
    ts_event=bar.ts_event,
)
```

## Event Bus Integration

### Message Publishing

```python
def _publish_signal(self, signal: MLSignal) -> None:
    """Publish ML signal to the message bus."""
    self.publish_data(
        DataType(MLSignal, metadata={"source": self.id.value}),
        signal,
    )
```

### Event Handling

- **on_bar(bar)**: Primary hot path event handler
- **on_start()**: Model loading, feature initialization, subscription setup
- **on_stop()**: Store flushing and performance statistics

### Data Subscription

```python
def on_start(self) -> None:
    """Initialize and subscribe to market data."""
    self.subscribe_bars(self._config.bar_type)  # Subscribe to configured bar type
```

## Registry Integration

### Model Registry Integration

```python
def _try_load_from_registry(self) -> bool:
    """Load model and metadata from ModelRegistry."""
    if hasattr(self._config, "model_id") and self._config.model_id:
        registry = ModelRegistry(registry_path)
        model_info = registry.get_model(self._config.model_id)

        # Load model and extract manifest metadata
        self._model = registry.load_model(self._config.model_id)
        self._model_metadata = extract_manifest_metadata(model_info.manifest)

        # Validate feature schema compatibility
        assert_features_compatible(manifest, actual_names, actual_dtypes)
        return True
    return False
```

### Feature Registry Integration

```python
# Validate features against registry manifest
if config.use_registry_features and config.feature_set_id:
    freg = FeatureRegistry(Path(config.registry_path))
    feature_info = freg.get_feature_set(config.feature_set_id)
    manifest = feature_info.manifest

    expected = list(manifest.feature_names)
    actual = self._feature_engineer.config.get_feature_names()
    if expected != actual:
        raise ValueError(f"Feature schema mismatch with manifest")
```

## Production Features

### Health Monitoring

```python
class HealthMonitor:
    """Tracks prediction success rates, latency violations, system status."""

    status: HealthStatus  # HEALTHY, DEGRADED, or UNHEALTHY
    model_loaded: bool
    indicators_initialized: bool
    consecutive_failures: int
    total_predictions: int
    failed_predictions: int
    total_latency_violations: int

    def update_prediction_success(self) -> None:
        self.consecutive_failures = 0
        self.total_predictions += 1

    def update_prediction_failure(self) -> None:
        self.consecutive_failures += 1
        self.failed_predictions += 1
        self._update_health_status()

    def get_success_rate(self) -> float:
        """Calculate overall prediction success rate."""
```

### Circuit Breaker Protection

```python
class CircuitBreaker:
    """Prevents cascade failures with configurable thresholds."""

    _state: CircuitBreakerState  # CLOSED, OPEN, or HALF_OPEN
    _failure_count: int
    _success_count: int
    _recovery_timeout: float

    def can_execute(self) -> bool:
        if self._state == CircuitBreakerState.OPEN:
            if current_time >= self._next_attempt:
                self._state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self) -> None:
        """Record successful operation and potentially close circuit."""

    def record_failure(self) -> None:
        """Record failure and potentially open circuit."""
```

### Model Hot-Reloading

```python
def _check_model_updates(self, event: Any) -> None:
    """Periodic model version checks with atomic swapping."""
    current_version = self._model_loader.get_model_version(self._config.model_path)
    if current_version != self._model_version:
        # Backup indicator state
        if self._config.preserve_state_on_reload:
            self._backup_indicator_state()

        # Atomic model reload
        self._reload_model()

        # Restore indicator state
        if self._config.preserve_state_on_reload:
            self._restore_indicator_state()
```

## Performance Optimization

### Optimization Levels

```python
class OptimizationLevel(Enum):
    STANDARD = "standard"     # Default performance
    OPTIMIZED = "optimized"   # Advanced optimizations with lock-free buffers
```

### Model Loading Support

The framework supports multiple model formats through automatic detection:

- **ONNX** (.onnx): Optimized runtime with CPU providers
- **XGBoost** (.json): Native XGBoost JSON format
- **Joblib** (.joblib): Scikit-learn and general Python models
- **Pickle** (.pkl, .pickle): DEPRECATED - raises SecurityError for security

### ONNX Runtime Optimization

```python
def _load_optimized_onnx_model(self) -> None:
    """Load ONNX model with CPU optimizations."""
    session_options, providers = to_session_options(self._config.onnx_runtime_config)
    model = ort.InferenceSession(
        self._config.model_path,
        sess_options=session_options,
        providers=providers,  # CPU optimized providers
    )
```

### Performance Monitoring

```python
class PerformanceMonitor:
    """Non-blocking performance tracking with reservoir sampling."""

    feature_times: list[float]
    inference_times: list[float]
    total_times: list[float]
    reservoir_size: int
    prediction_count: int
    signal_count: int
    error_count: int

    def record_timing(self, feature_time_ns, inference_time_ns, total_time_ns):
        # Bounded memory with reservoir sampling
        if len(self.feature_times) > self.reservoir_size:
            self.feature_times = self.feature_times[-self.reservoir_size:]

    def get_latency_percentiles(self) -> dict[str, dict[float, float]]:
        percentiles = [50.0, 90.0, 95.0, 99.0]
        return {
            "feature_computation": {p: np.percentile(self.feature_times, p) for p in percentiles},
            "inference": {p: np.percentile(self.inference_times, p) for p in percentiles},
            "total": {p: np.percentile(self.total_times, p) for p in percentiles}
        }

    def get_current_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
```

### Model Swapper

```python
class ModelSwapper:
    """Atomic model swapping for hot reload without disrupting inference."""

    _current_model: Any
    _current_metadata: dict[str, Any]
    _swap_pending: bool

    def prepare_swap(self, model: Any, metadata: dict[str, Any]) -> None:
        """Prepare new model for atomic swap."""

    def execute_swap(self) -> bool:
        """Execute atomic model swap."""
```

## Metrics and Observability

### Prometheus Metrics

```python
# Core metrics (from base.py)
ml_predictions_total = Counter(
    "nautilus_ml_predictions_total",
    "Total number of ML predictions made",
    ["actor_id", "model_name"],
)

ml_prediction_latency = Histogram(
    "nautilus_ml_prediction_latency_seconds",
    "Latency of ML predictions in seconds",
    ["actor_id", "model_name"],
)

ml_signal_confidence = Histogram(
    "nautilus_ml_signal_confidence",
    "Distribution of ML signal confidence scores",
    ["actor_id", "model_name"],
)

# Additional signal actor metrics (from signal.py)
prediction_distribution = Histogram(
    "nautilus_ml_prediction_distribution",
    "Distribution of model predictions",
    ["actor_id"],
)

confidence_distribution = Histogram(
    "nautilus_ml_confidence_distribution",
    "Distribution of prediction confidence scores",
    ["actor_id"],
)

signal_generation_time = Histogram(
    "nautilus_ml_signal_generation_seconds",
    "Signal generation latency in seconds",
    ["actor_id", "strategy"],
)

feature_time_by_set = Histogram(
    "nautilus_ml_feature_time_by_set_seconds",
    "Feature computation latency by feature_set_id",
    ["actor_id", "feature_set_id"],
)

signals_generated = Counter(
    "nautilus_ml_signals_generated_total",
    "Total number of signals generated",
    ["actor_id", "strategy", "signal_type"],
)

adaptive_threshold = Histogram(
    "nautilus_ml_adaptive_threshold",
    "Adaptive threshold values",
    ["actor_id"],
)

market_regime = Counter(
    "nautilus_ml_market_regime_total",
    "Market regime detection counts",
    ["actor_id", "regime"],
)
```

### Metrics Recording

```python
# Record in hot path
self._inference_latency_metric.labels(
    actor_id=str(self.id),
    model_name=Path(self._config.model_path).stem,
).observe(inference_time / 1000)

self._inference_count_metric.labels(
    actor_id=str(self.id),
    model_name=Path(self._config.model_path).stem,
).inc()
```

## Configuration Architecture

### Base Configuration (MLActorConfig)

```python
class MLActorConfig(NautilusConfig):
    """Base configuration for all ML actors."""

    # Model configuration
    model_path: str                   # Path to model file
    model_id: str | None = None       # Model identifier for registry

    # Market data configuration
    bar_type: BarType
    instrument_id: InstrumentId | None = None

    # Prediction configuration
    prediction_threshold: float = 0.5
    max_inference_latency_ms: float = 5.0
    max_feature_latency_ms: float = 0.5

    # Feature configuration
    feature_config: MLFeatureConfig | None = None

    # Warm-up and batching
    batch_size: int = 1
    warm_up_period: int = 50          # Bars before predictions start

    # Signal publishing
    publish_signals: bool = True
    log_predictions: bool = False

    # Hot reload configuration
    enable_hot_reload: bool = False
    model_check_interval: int = 300   # Seconds between checks
    preserve_state_on_reload: bool = True

    # Health and resilience
    circuit_breaker_config: CircuitBreakerConfig | None = None
    enable_health_monitoring: bool = True
    health_config: HealthMonitorConfig | None = None

    # Actor configuration
    component_id: ComponentId | None = None
    log_events: bool = True
    log_commands: bool = True
```

### Signal Actor Configuration (MLSignalActorConfig)

```python
class MLSignalActorConfig(MLActorConfig):
    """Extended configuration for signal generation actors."""

    # Signal generation
    signal_strategy: Literal["threshold", "extremes", "momentum", "ensemble", "adaptive"] = "threshold"
    adaptive_window: int = 20
    min_signal_separation_bars: int = 3
    feature_importance_threshold: float = 0.01
    enable_regime_detection: bool = True

    # Performance optimization
    optimization_config: OptimizationConfig | None = None
    strategy_config: StrategyConfig | None = None
    onnx_runtime_config: OnnxRuntimeConfig | None = None

    # Hot reload
    enable_hot_reload: bool = False
    hot_reload_interval: int = 300

    # Custom strategy
    custom_strategy: Any | None = None  # Custom SignalGenerationStrategy

    # Registry integration
    feature_set_id: str | None = None
    registry_path: str | None = None
    use_registry_features: bool = False

    # Store integration
    use_feature_store: bool = False
    db_connection: str = "postgresql://postgres:postgres@localhost:5432/nautilus"
    persist_features: bool = True
    pipeline_spec: Any | None = None

    # Test mode
    use_dummy_stores: bool = False
    actor_id: str | None = None  # For test identification
```

### Optimization Configuration

```python
class OptimizationConfig(NautilusConfig):
    """Performance optimization settings."""

    level: Literal["standard", "optimized"] = "standard"
    enable_zero_copy: bool = False
    enable_model_warm_up: bool = False
    warm_up_iterations: int = 100
    pre_allocate_buffers: bool = True
    use_lock_free_buffers: bool = False
    reservoir_sample_size: int = 1000
```

### Strategy Configuration

```python
class StrategyConfig(NautilusConfig):
    """Strategy-specific parameters."""

    # Extremes strategy
    extremes_top_pct: float = 0.1

    # Momentum strategy
    momentum_lookback: int = 5

    # Ensemble strategy
    ensemble_weights: dict[str, float] | None = None

    # Adaptive strategy
    adaptive_volatility_factor: float = 2.0
    min_threshold: float = 0.1
    max_threshold: float = 0.95
    update_frequency: int = 10
```

## Current Implementation Status

### Completed Features ✅

- **Base Actor Framework**: Abstract base class with mandatory store integration
- **Multiple Actor Implementations**:
  - MLSignalActor: Production signal generation with configurable strategies
  - ONNXMLInferenceActor: ONNX-optimized inference actor
  - EnhancedMLInferenceActor: Complete demonstration with technical indicators
- **Signal Generation Strategies**: All 5 strategies implemented:
  - ThresholdSignalStrategy: Static threshold filtering
  - ExtremesStrategy: Percentile-based extremes detection
  - MomentumStrategy: Trend-based momentum signals
  - EnsembleStrategy: Weighted voting ensemble
  - AdaptiveStrategy: Dynamic threshold with volatility adjustment
- **Model Format Support**:
  - **✨ ENHANCEMENT:** ONNX (.onnx) with optimized runtime and configurable providers
  - **📝 ADDITION:** XGBoost (.json) with both Booster.load_model() and JSON fallback support
  - **📝 ADDITION:** Joblib (.joblib) with direct joblib.load() and metadata extraction
  - **⚠️ CORRECTION:** Pickle (.pkl) DEPRECATED with ValueError (not SecurityError) and migration guidance
- **Hot Path Optimization**:
  - Zero-allocation feature computation
  - Pre-allocated numpy buffers
  - Lock-free buffer support (optional)
  - Model warm-up capability
- **Registry Integration**:
  - Model manifest validation
  - Feature schema compatibility checks
  - Automatic model/feature loading from registry
- **Store Integration**:
  - Mandatory FeatureStore persistence
  - ModelStore for predictions and metrics
  - StrategyStore for signals and decisions
  - Automatic store initialization with fallback to SQLite
- **Performance Monitoring**:
  - PerformanceMonitor with reservoir sampling
  - Comprehensive latency tracking
  - Signal generation metrics
  - Market regime detection
- **Resilience Features**:
  - Circuit breaker with configurable thresholds
  - Health monitoring with status tracking
  - Model hot-reloading with state preservation
  - ModelSwapper for atomic model updates

### Testing Coverage ✅

- **Unit Tests**: Comprehensive coverage for all strategies and edge cases
- **Integration Tests**: E2E testing with real Nautilus components
- **Mock Support**: Full support for testing with mock models
- **Dummy Stores**: Test mode with DummyStore for isolated testing

### Security & Safety ✅

- **⚠️ CORRECTION:** **Pickle Deprecation**: PickleMLInferenceActor raises SecurityError (stub implementation)
- **📝 ADDITION:** **Production Model Security**: Non-ONNX formats restricted unless ML_TEST_ALLOW_NON_ONNX environment variable set
- **Model Format Validation**: Strict format checking with safe loading **✨ ENHANCEMENT:** Includes comprehensive error messages and migration guidance
- **Input Validation**: Aggressive validation with descriptive exceptions **📝 ADDITION:** Includes feature schema compatibility checks
- **Memory Safety**: Bounded buffers and controlled allocations **✨ ENHANCEMENT:** Lock-free alternatives available in OPTIMIZED mode
- **Thread Safety**: Atomic model swapping and store operations **📝 ADDITION:** Single-threaded actor model ensures hot-path safety

## Critical Implementation Details

### Feature Engineering Integration

```python
class MLSignalActor(BaseMLInferenceActor):
    def __init__(self, config):
        # **✨ ENHANCEMENT:** Feature engineering setup with comprehensive validation
        self._feature_engineer = FeatureEngineer(self._feature_config)
        self._indicator_manager = IndicatorManager(self._feature_config)

        # **📝 ADDITION:** Validate against model manifest if loaded from registry
        model_names = getattr(self, "_manifest_feature_names", [])
        if model_names:
            actual_names = self._feature_engineer.config.get_feature_names()
            # **📝 ADDITION:** Create temporary manifest for feature compatibility check
            tmp_manifest = ModelManifest(
                model_id="__validation__",
                role=ModelRole.STUDENT,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="unknown",
                feature_schema=dict.fromkeys(model_names, "float32"),
                feature_schema_hash=getattr(self, "_manifest_feature_schema_hash", ""),
            )
            actual_dtypes = ["float32"] * len(actual_names)  # **⚠️ CORRECTION:** Hot path uses float32
            assert_features_compatible(tmp_manifest, actual_names, actual_dtypes)

        # **✨ ENHANCEMENT:** Validate against feature registry with comprehensive error handling
        if config.use_registry_features and config.feature_set_id:
            try:
                freg = FeatureRegistry(Path(config.registry_path))
                feature_info = freg.get_feature_set(config.feature_set_id)
                manifest = feature_info.manifest if feature_info else None
            except Exception as e:
                manifest = None
                self.log.warning(f"Feature registry load failed: {e}")

            if manifest is not None:
                expected = list(manifest.feature_names)
                actual = self._feature_engineer.config.get_feature_names()
                if expected != actual:
                    raise ValueError(
                        f"Feature schema mismatch with manifest: expected {len(expected)} names "
                        f"(hash={manifest.schema_hash}), got {len(actual)}"
                    )
                self._feature_set_id = manifest.feature_set_id
```

### Thread Safety

- **Store Operations**: Thread-safe with connection pooling
- **Metrics Recording**: Prometheus client handles concurrent access
- **Model Swapping**: Atomic operations with memory barriers
- **Buffer Access**: Single-threaded actor model ensures safety

### Memory Management

- **Pre-allocated Buffers**: Fixed-size numpy arrays for features
- **Bounded Histories**: Rolling windows with maxlen constraints
- **Reservoir Sampling**: Constant memory for performance tracking
- **Model Cleanup**: Explicit deletion during hot-reload

### Error Handling

```python
def _generate_prediction_protected(self, bar, features):
    try:
        prediction, confidence = self._predict(features)
        # ... success path
        if self._circuit_breaker:
            self._circuit_breaker.record_success()
    except Exception as e:
        self.log.error(f"Prediction failed: {e}")
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        if self._health_monitor:
            self._health_monitor.update_prediction_failure()
```

## Usage Patterns

### Basic Signal Actor

```python
from nautilus_trader.model.data import BarType
from ml.actors import MLSignalActor, MLSignalActorConfig

config = MLSignalActorConfig(
    component_id="ml_signal",
    model_path="models/my_model.onnx",
    signal_strategy="threshold",  # Can use string or enum
    prediction_threshold=0.75,
    bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-BID-EXTERNAL"),
    warm_up_period=50,  # Bars before predictions start
)

actor = MLSignalActor(config)
```

### Advanced Configuration with Optimization

```python
from ml.actors import OptimizationConfig, StrategyConfig
from ml.config.runtime import OnnxRuntimeConfig  # **📝 ADDITION:** ONNX runtime configuration

config = MLSignalActorConfig(
    component_id="advanced_ml_signal",
    model_path="models/ensemble_model.onnx",
    signal_strategy="ensemble",
    optimization_config=OptimizationConfig(
        level="optimized",
        enable_model_warm_up=True,
        warm_up_iterations=100,
        use_lock_free_buffers=True,
        reservoir_sample_size=1000,
    ),
    strategy_config=StrategyConfig(
        ensemble_weights={
            "threshold": 0.4,
            "extremes": 0.3,
            "momentum": 0.3,
        },
        extremes_top_pct=0.1,
        momentum_lookback=5,
    ),
    # **📝 ADDITION:** ONNX runtime configuration for CPU optimization
    onnx_runtime_config=OnnxRuntimeConfig(
        graph_optimization_level="ORT_ENABLE_ALL",
        execution_mode="ORT_SEQUENTIAL",
        intra_threads=1,
        inter_threads=1,
    ),
    enable_hot_reload=True,
    enable_regime_detection=True,
    adaptive_window=20,
    min_signal_separation_bars=3,
    # **📝 ADDITION:** Feature store integration
    use_feature_store=True,
    persist_features=True,
    feature_set_id="ensemble_features_v1",
)

actor = MLSignalActor(config)
```

### Custom Strategy Implementation

```python
from ml.actors.signal import SignalGenerationStrategy

class CustomStrategy(SignalGenerationStrategy):
    def generate_signal(self, bar, prediction, confidence, features, context):
        # Access context data
        adaptive_threshold = context.get("adaptive_threshold", 0.7)
        market_regime = context.get("market_regime", "unknown")

        # Custom signal logic
        if confidence >= adaptive_threshold:
            return MLSignal(
                instrument_id=bar.bar_type.instrument_id,
                model_id=context.get("model_id", "custom"),
                prediction=prediction,
                confidence=confidence,
                features=features if context.get("log_predictions") else None,
                metadata={"regime": market_regime},
                ts_event=bar.ts_event,
                ts_init=context["timestamp_ns"],
            )
        return None

config = MLSignalActorConfig(
    component_id="custom_signal",
    model_path="models/custom_model.onnx",
    custom_strategy=CustomStrategy(),
)
```

### Registry Integration

```python
config = MLSignalActorConfig(
    component_id="registry_signal",
    model_id="xgboost_v2.1.0",  # Load from registry instead of path
    registry_path="ml/models",
    feature_set_id="microstructure_features_v1",
    use_registry_features=True,
    bar_type=BarType.from_str("SPY.XNAS-1-MINUTE-BID-EXTERNAL"),
)

actor = MLSignalActor(config)
```

### Test Configuration with Dummy Stores

```python
config = MLSignalActorConfig(
    component_id="test_signal",
    model_path="tests/fixtures/test_model.onnx",
    signal_strategy="threshold",
    prediction_threshold=0.5,
    use_dummy_stores=True,  # Use DummyStore for testing
    bar_type=BarType.from_str("TEST.SIM-1-MINUTE-BID-EXTERNAL"),
)

actor = MLSignalActor(config)
```

This ML actors framework provides a production-ready foundation for real-time machine learning in trading systems, with strict performance requirements, comprehensive observability, and robust fault tolerance.

## **📝 ADDITION:** Enhanced Architecture Patterns

### Model Prediction Compatibility Layer

The framework now includes enhanced model compatibility for various model types:

```python
def _predict(self, features: npt.NDArray[np.float32]) -> tuple[float, float]:
    """Enhanced prediction with comprehensive model type support."""

    # **📝 ADDITION:** Mock support for testing environments
    if isinstance(self._model, Mock | MagicMock):
        # Handle test mock models with proper prediction interface

    # **📝 ADDITION:** Unified model wrapper support
    elif hasattr(self._model, "predict") and hasattr(self._model, "metadata"):
        result = self._model.predict(features)
        return result[0], result[1]

    # **📝 ADDITION:** Raw ONNX model support
    elif hasattr(self._model, "run") and "input_names" in self._model_metadata:
        features_2d = features.reshape(1, -1).astype(np.float32)
        input_name = self._model_metadata["input_names"][0]
        outputs = self._model.run(None, {input_name: features_2d})

    # **📝 ADDITION:** XGBoost Booster support with DMatrix
    elif hasattr(self._model, "num_features") and hasattr(self._model, "get_score"):
        from ml._imports import xgb
        features_2d = features.reshape(1, -1)
        dmatrix = xgb.DMatrix(features_2d)
        predictions = self._model.predict(dmatrix)
        return float(predictions[0]), 0.5
```

### **📝 ADDITION:** Store Progressive Fallback Implementation

```python
def _init_stores_and_registries(self) -> None:
    """Progressive fallback chain: PostgreSQL → SQLite → DummyStore."""

    # Try PostgreSQL connection first
    try:
        test_engine = EngineManager.get_engine(
            "postgresql://postgres:postgres@localhost:5432/nautilus"
        )
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        # PostgreSQL available - use production stores
        self._feature_store = FeatureStore(connection_string=db_connection)

    except Exception:
        # PostgreSQL unavailable - fall back to DummyStore with warning
        self.log.warning("PostgreSQL not available; using DummyStore (no persistence)")
        self._feature_store = DummyStore()
```

### **📝 ADDITION:** Feature Store Compute Integration

```python
def _compute_features(self, bar: Bar) -> npt.NDArray[np.float32] | None:
    """Feature computation with store delegation and fallback."""

    # **✨ ENHANCEMENT:** Prefer FeatureStore.compute_realtime when available
    try:
        if hasattr(self, "_feature_store") and self._feature_store is not None:
            features = self._feature_store.compute_realtime(
                bar=bar,
                store=self._persist_features
            )
            if isinstance(features, np.ndarray) and features.size > 0:
                return features
    except Exception as exc:
        # Fall back to local feature engineering on store failure
        self.log.debug("FeatureStore compute_realtime failed; falling back", exc_info=exc)

    # **⚠️ CORRECTION:** Always use local FeatureEngineer as fallback
    # Implementation continues with indicator manager and feature calculation...
```

### **📝 ADDITION:** Performance Monitoring Integration

The actors now include comprehensive runtime statistics tracking:

```python
def get_signal_statistics(self) -> dict[str, Any]:
    """Lightweight runtime statistics for testing and diagnostics."""
    stats = {
        "bars_processed": int(getattr(self, "_bars_processed", 0)),
        "prediction_history_size": len(getattr(self, "_prediction_history", [])),
        "confidence_history_size": len(getattr(self, "_confidence_history", [])),
    }

    # **📝 ADDITION:** Merge PerformanceMonitor stats if available
    if hasattr(self, "_performance_monitor") and self._performance_monitor is not None:
        pm_stats = self._performance_monitor.get_current_stats()
        stats.update(pm_stats)

    return stats
```

## Universal Pattern Compliance

The ML actors framework fully implements all 5 universal ML architecture patterns:

### ✅ Pattern 1: Mandatory 4-Store + 4-Registry Integration

- All actors inherit from BaseMLInferenceActor with automatic component initialization
- Property accessors provide clean interface: `.feature_store`, `.data_store`, `.feature_registry`, etc.
- Health monitoring includes all 8 components
- Progressive fallback to DummyStore/DummyRegistry when PostgreSQL unavailable

### ✅ Pattern 2: Protocol-First Interface Design

- Store protocols in `ml/stores/protocols.py` enable structural typing
- DummyStore conforms to all protocols for testing compatibility
- Type safety without circular dependencies

### ✅ Pattern 3: Hot/Cold Path Separation

- Hot path: <5ms P99 latency with zero-allocation patterns
- Cold path: Model loading, registry operations, health monitoring
- Pre-allocated feature arrays and model sessions

### ✅ Pattern 4: Progressive Fallback Chains

- PostgreSQL → DummyStore with warning logs
- Registry loading → Direct file loading via model_path
- Configuration errors → Safe defaults with operational alerts

### ✅ Pattern 5: Centralized Metrics Bootstrap

- All Prometheus metrics via `ml.common.metrics_bootstrap`
- Zero direct prometheus_client imports
- Safe metric registration across module reloads

This comprehensive pattern adherence ensures consistent architecture, reliable performance, and maintainable integration across all ML components.

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

## Universal Component Protocol

All actors implement the universal component interface defined in `ml/common/protocols.py`:

- `get_health_status() -> dict[str, Any]`: health summary (safe to call off the hot path)
- `get_performance_metrics() -> dict[str, float]`: lightweight diagnostic metrics
- `validate_configuration() -> list[str]`: configuration validation issues (empty if valid)

Notes:

- These methods must not be called in the hot path; use them in setup, health endpoints, or scheduled checks.
- Protocol compliance is validated by the Integration Manager (warn by default; strict mode via `ML_STRICT_PROTOCOL_VALIDATION`).

### Runtime Statistics

The MLSignalActor exposes a lightweight, non–hot‑path method `get_signal_statistics()` for tests and diagnostics. It returns:

- bars_processed: total bars seen by the actor **📝 ADDITION:** Tracked by BaseMLInferenceActor, falls back to 0 if missing
- prediction_history_size, confidence_history_size: current lengths of rolling histories
- PerformanceMonitor summary: prediction_count, signal_count, error_count, average and p99 latencies **✨ ENHANCEMENT:** Includes comprehensive performance metrics with reservoir sampling

This method is safe to call outside the hot path and is intended for assertions in tests and sanity checks.

## **📝 ADDITION:** Centralized Metrics Bootstrap Pattern

### Metrics Bootstrap Integration

All actors now use the centralized metrics bootstrap pattern instead of direct prometheus imports:

```python
# ❌ OLD: Direct prometheus usage
from prometheus_client import Counter, Histogram

# ✅ NEW: Centralized bootstrap
from ml.common.metrics_bootstrap import get_counter, get_histogram

ml_predictions_total = get_counter(
    "nautilus_ml_predictions_total",
    "Total number of ML predictions made",
    ["actor_id", "model_name"]
)
```

**Benefits:**

- Prevents metric registry conflicts during module reloads
- Safe for testing environments with metric cleanup
- Consistent naming and labeling across components

### **📝 ADDITION:** Universal MLComponentProtocol Integration

All actors now implement the universal component protocol via `MLComponentMixin`:

```python
class BaseMLInferenceActor(MLComponentMixin, NautilusActor, ABC):
    """Base actor with universal protocol compliance."""

    def get_health_status(self) -> dict[str, Any]:
        """Enhanced health reporting with store status."""
        base_status = super().get_health_status()
        base_status.update({
            "stores_initialized": self._stores_healthy(),
            "model_loaded": self._model is not None,
            "circuit_breaker_state": self._circuit_breaker.state if self._circuit_breaker else "disabled"
        })
        return base_status
```

**Protocol Methods:**

- `get_health_status()`: Comprehensive health with store/model status
- `get_performance_metrics()`: Lightweight diagnostic metrics
- `validate_configuration()`: Configuration validation issues

### Store Protocol Evolution

```python
class FeatureStoreProtocol(Protocol):
    def compute_realtime(
        self,
        bar: Any,
        store: bool = ...,
        indicator_manager: Any | None = ...
    ) -> Any:
        """Delegate feature computation with optional persistence."""
```

**Integration Pattern:**

```python
def _compute_features(self, bar: Bar) -> npt.NDArray[np.float32] | None:
    # Prefer FeatureStore delegation when available
    try:
        features = self._feature_store.compute_realtime(
            bar=bar,
            store=self._persist_features
        )
        if isinstance(features, np.ndarray) and features.size > 0:
            return features
    except Exception as exc:
        self.log.debug("FeatureStore failed, falling back to local computation")

    # Fallback to local feature engineering
    return self._feature_engineer.calculate_features_online(...)
```

## **📝 ADDITION:** Configuration Enhancement Summary

### MLSignalActorConfig Extensions

The configuration has been extended with several new options:

```python
class MLSignalActorConfig(MLActorConfig):
    # **📝 ADDITION:** ONNX runtime optimization
    onnx_runtime_config: OnnxRuntimeConfig | None = None

    # **📝 ADDITION:** Feature store integration
    use_feature_store: bool = False
    persist_features: bool = True
    feature_set_id: str | None = None
    pipeline_spec: Any | None = None

    # **📝 ADDITION:** Test mode configuration
    use_dummy_stores: bool = False
    actor_id: str | None = None  # For test identification

    # **⚠️ CORRECTION:** Backward compatibility mappings in __post_init__()
    optimization: OptimizationConfig | None = None  # Maps to optimization_config
    strategy: StrategyConfig | None = None          # Maps to strategy_config
```

**Key Features:**

- Automatic field mapping for backward compatibility
- Test mode configuration with dummy stores
- ONNX runtime provider configuration
- Feature persistence control

### **🔄 UPDATE:** EnhancedMLInferenceActor Architecture Change

The EnhancedMLInferenceActor has been redesigned as a **minimal test-focused implementation** rather than a complete demonstration:

- **Purpose**: Testing zero-allocation feature computation patterns
- **Store Integration**: Uses null store protocols to avoid external dependencies
- **Feature Computation**: Guarantees view semantics with pre-allocated buffers
- **Usage**: Primarily for performance tests and hot-path validation

```python
def _compute_features(self, bar: Bar) -> npt.NDArray[np.float32] | None:
    """Guarantees zero-allocation feature computation with buffer reuse."""
    features = self._engineer.calculate_features_online(...)
    self._feature_buffer[:size] = features
    return self._feature_buffer[:size]  # Returns view, not copy
```

#### Test Configuration Patterns

```python
# **📝 ADDITION:** Test configuration with dummy stores
config = MLSignalActorConfig(
    component_id="test_signal",
    model_path="tests/fixtures/test_model.onnx",
    use_dummy_stores=True,  # **📝 ADDITION:** Avoids external dependencies
    actor_id="test_actor_1",  # **📝 ADDITION:** For test identification
    bar_type=BarType.from_str("TEST.SIM-1-MINUTE-BID-EXTERNAL"),
)
```
