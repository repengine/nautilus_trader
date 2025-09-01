# ML Actors Context Documentation

## Executive Summary

The ML actors framework provides a production-ready foundation for real-time machine learning inference and signal generation within Nautilus Trader. The architecture follows strict hot/cold path separation, ensuring sub-millisecond performance in production environments while maintaining comprehensive observability and fault tolerance.

Operational notes:

- Timestamps: Actors should emit UNIX nanoseconds for `ts_event`/`ts_init`. Stores defensively normalize smaller units (seconds/ms/us) to ns with a warning. See `context_stores.md` → "Timestamp Policy & Normalization".
- DB preflight: Verify required DB functions and current partition exist before startup. See `context_deployment.md` → "DB Preflight (recommended)".

**Key Components:**

- **BaseMLInferenceActor**: Abstract foundation class with mandatory store integration and production features
- **MLSignalActor**: Production signal generation actor with multiple built-in strategies
- **ONNXMLInferenceActor**: ONNX-optimized inference actor for lowest latency
- **EnhancedMLInferenceActor**: Complete implementation showcasing all production features
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
│   ├── MLSignalActor: Production signal generation with configurable strategies
│   ├── ONNXMLInferenceActor: ONNX-optimized inference for sub-millisecond latency
│   ├── EnhancedMLInferenceActor: Complete feature demonstration with technical indicators
│   └── PickleMLInferenceActor: DEPRECATED - raises SecurityError (security risk)
│
└── Notes:
    └── Pickle formats are intentionally disallowed for production actors (security)
        — export ONNX, joblib, or framework-native formats instead
```

### Signal Generation Architecture

The `MLSignalActor` provides a sophisticated signal generation system with pluggable strategies:

**Built-in Strategies:**

- **ThresholdSignalStrategy**: Simple confidence-based filtering with static threshold
- **ExtremesStrategy**: Percentile-based signal generation using prediction extremes
- **MomentumStrategy**: Trend-based signal enhancement using prediction momentum
- **EnsembleStrategy**: Weighted voting across multiple strategies
- **AdaptiveStrategy**: Dynamic threshold adjustment based on market volatility and prediction variance

**Signal Data Classes:**

- **MLSignal**: Unified signal data class containing:
  - `instrument_id`: The instrument identifier
  - `model_id`: Unique model identifier for tracking
  - `prediction`: Model prediction value
  - `confidence`: Confidence score (0.0 to 1.0)
  - `features`: Optional feature array for debugging
  - `metadata`: Optional additional metadata
  - `ts_event`: Event timestamp in nanoseconds
  - `ts_init`: Initialization timestamp in nanoseconds
- **AdaptiveSignal**: Type alias for MLSignal (backward compatibility)

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

- **Pre-allocated Feature Buffers**: `np.zeros(n_features, dtype=np.float32)`
- **Rolling Windows**: Fixed-size deques with maxlen constraints
- **Buffer Reuse**: Feature vectors returned as views, not copies
- **ONNX Runtime Optimization**: CPU-optimized sessions with minimal memory footprint

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
    feature_set_id=self._config.feature_set_id,
    instrument_id=str(bar.bar_type.instrument_id),
    features=feature_dict,
    ts_event=bar.ts_event,
    ts_init=bar.ts_init,
)

# MANDATORY: Store prediction for performance tracking
self._model_store.write_prediction(
    model_id=self._model_id,
    instrument_id=str(bar.bar_type.instrument_id),
    prediction=float(prediction),
    confidence=float(confidence),
    features=feature_dict,
    inference_time_ms=inference_time,
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
  - ONNX (.onnx) with optimized runtime
  - XGBoost (.json) native format
  - Joblib (.joblib) for scikit-learn
  - Pickle (.pkl) DEPRECATED with SecurityError
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

- **Pickle Deprecation**: PickleMLInferenceActor raises SecurityError
- **Model Format Validation**: Strict format checking with safe loading
- **Input Validation**: Aggressive validation with descriptive exceptions
- **Memory Safety**: Bounded buffers and controlled allocations
- **Thread Safety**: Atomic model swapping and store operations

## Critical Implementation Details

### Feature Engineering Integration

```python
class MLSignalActor(BaseMLInferenceActor):
    def __init__(self, config):
        # Feature engineering setup
        self._feature_engineer = FeatureEngineer(self._feature_config)
        self._indicator_manager = IndicatorManager(self._feature_config)

        # Validate against model manifest if available
        model_names = getattr(self, "_manifest_feature_names", [])
        if model_names:
            actual_names = self._feature_engineer.config.get_feature_names()
            assert_features_compatible(tmp_manifest, actual_names, actual_dtypes)

        # Validate against feature registry if configured
        if config.use_registry_features and config.feature_set_id:
            freg = FeatureRegistry(Path(config.registry_path))
            feature_info = freg.get_feature_set(config.feature_set_id)
            manifest = feature_info.manifest
            expected = list(manifest.feature_names)
            actual = self._feature_engineer.config.get_feature_names()
            if expected != actual:
                raise ValueError(f"Feature schema mismatch")
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
    enable_hot_reload=True,
    enable_regime_detection=True,
    adaptive_window=20,
    min_signal_separation_bars=3,
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
- bars_processed: total bars seen by the actor
- prediction_history_size, confidence_history_size: current lengths of rolling histories
- PerformanceMonitor summary: prediction_count, signal_count, error_count, average and p99 latencies

This method is safe to call outside the hot path and is intended for assertions in tests and sanity checks.
