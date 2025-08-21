# ML Actors Context Documentation

## Executive Summary

The ML actors framework provides a production-ready foundation for real-time machine learning inference and signal generation within Nautilus Trader. The architecture follows strict hot/cold path separation, ensuring sub-millisecond performance in production environments while maintaining comprehensive observability and fault tolerance.

**Key Components:**

- **BaseMLInferenceActor**: Foundation class with mandatory store integration and production features
- **MLSignalActor**: Specialized actor for signal generation with configurable strategies
- **Hot Path Optimization**: Zero-allocation inference with <5ms end-to-end latency targets
- **Mandatory Store Triad**: FeatureStore, ModelStore, and StrategyStore for complete data persistence

## Architecture Overview

### Actor Hierarchy

```
BaseMLInferenceActor (Abstract)
├── Mandatory Features:
│   ├── Store Integration (FeatureStore, ModelStore, StrategyStore)
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
├── Concrete Implementation:
│   └── MLSignalActor (extends BaseMLInferenceActor; production signal actor with ONNX-optimized loading)
│
└── Notes:
    └── Pickle formats are intentionally disallowed for production actors (security)
        — export ONNX or framework-native formats instead
```

### Signal Generation Architecture

The `MLSignalActor` provides a sophisticated signal generation system with pluggable strategies:

**Built-in Strategies:**

- **ThresholdSignalStrategy**: Simple confidence-based filtering
- **ExtremesStrategy**: Percentile-based signal generation
- **MomentumStrategy**: Trend-based signal enhancement
- **EnsembleStrategy**: Weighted voting across multiple strategies
- **AdaptiveStrategy**: Dynamic threshold adjustment based on market volatility

**Signal Types:**

- **MLSignal**: Unified signal data class with instrument_id, model_id, prediction, confidence, and optional features
- **AdaptiveSignal**: Alias for MLSignal (backward compatibility)

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

### Mandatory Store Triad

Every ML actor **MUST** initialize and use three stores for complete data persistence:

```python
def _init_stores_and_registries(self) -> None:
    """MANDATORY: Initialize all stores - no optional parameters allowed."""
    # 1. FeatureStore: For training/inference parity
    self._feature_store = FeatureStore(connection_string=db_connection)

    # 2. ModelStore: For prediction tracking and performance analysis
    self._model_store = ModelStore(persistence_config=persistence_config)

    # 3. StrategyStore: For trading decisions and signal analysis
    self._strategy_store = StrategyStore(persistence_config=persistence_config)
```

### Data Persistence Flow

1. **Feature Storage**: Every computed feature vector is persisted with instrument_id, ts_event, ts_init
2. **Prediction Storage**: All predictions stored with inference_time_ms, confidence, and feature_dict
3. **Signal Storage**: Generated signals with strategy metadata, risk metrics, and execution parameters

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

    def update_prediction_success(self) -> None:
        self.consecutive_failures = 0
        self.total_predictions += 1

    def update_prediction_failure(self) -> None:
        self.consecutive_failures += 1
        self.failed_predictions += 1
        self._update_health_status()
```

### Circuit Breaker Protection

```python
class CircuitBreaker:
    """Prevents cascade failures with configurable thresholds."""

    def can_execute(self) -> bool:
        if self._state == CircuitBreakerState.OPEN:
            if current_time >= self._next_attempt:
                self._state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        return True
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
    OPTIMIZED = "optimized"   # Advanced optimizations
```

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
```

## Metrics and Observability

### Prometheus Metrics

```python
# Canonical metric names (via ml.config.names)
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

### Base Configuration (key fields)

```python
class MLActorConfig(NautilusConfig):
    model_path: str
    model_id: str                     # required for tracking
    bar_type: BarType
    instrument_id: InstrumentId
    prediction_threshold: float = 0.5
    max_inference_latency_ms: float = 5.0
    feature_config: MLFeatureConfig | None = None
    batch_size: int = 1
    warm_up_period: int = 50          # indicator warmup
    publish_signals: bool = True
    log_predictions: bool = False
    enable_hot_reload: bool = False
    model_check_interval: int = 300
    preserve_state_on_reload: bool = True
    circuit_breaker_config: CircuitBreakerConfig | None = None
    enable_health_monitoring: bool = True
    health_config: HealthMonitorConfig | None = None
    max_feature_latency_ms: float = 0.5
    component_id: ComponentId | None = None
    log_events: bool = True
    log_commands: bool = True
```

### Signal Actor Configuration

```python
class MLSignalActorConfig(MLActorConfig):
    signal_strategy: SignalStrategy = SignalStrategy.THRESHOLD
    adaptive_window: int = 20
    min_signal_separation_bars: int = 3
    optimization_config: OptimizationConfig | None = None
    strategy_config: StrategyConfig | None = None
    feature_set_id: str | None = None
    use_registry_features: bool = False
    use_feature_store: bool = False
```

## Current Implementation Status

### Completed Features ✅

- **Base Actor Framework**: Complete with mandatory stores and health monitoring
- **Signal Generation**: All 5 strategies implemented (threshold, extremes, momentum, ensemble, adaptive)
- **Hot Path Optimization**: Zero-allocation feature computation with pre-allocated buffers
- **ONNX Integration**: Optimized runtime with CPU providers
- **Registry Integration**: Model and feature manifest validation
- **Store Integration**: Mandatory FeatureStore, ModelStore, StrategyStore persistence
- **Performance Monitoring**: Comprehensive metrics with reservoir sampling
- **Circuit Breaker**: Fault tolerance with configurable thresholds
- **Model Hot-Reloading**: Atomic swapping with state preservation

### Testing Coverage ✅

- **Unit Tests**: Comprehensive coverage for all strategies and edge cases
- **Integration Tests**: E2E testing with real Nautilus components
- **Property-Based Tests**: Hypothesis testing for signal consistency
- **Performance Tests**: Hot path latency validation and memory leak detection

### Security & Safety ✅

- **Pickle Deprecation**: PickleMLInferenceActor raises SecurityError
- **Model Format Validation**: Strict format checking with safe loading
- **Input Validation**: Aggressive validation with descriptive exceptions
- **Memory Safety**: Bounded buffers and controlled allocations

## Critical Implementation Details

### Feature Engineering Integration

```python
class MLSignalActor(BaseMLInferenceActor):
    def __init__(self, config):
        # Feature engineering with validation
        self._feature_engineer = FeatureEngineer(self._feature_config)

        # Validate against model manifest if available
        if model_names:
            assert_features_compatible(tmp_manifest, actual_names, actual_dtypes)

        # Validate against feature registry if configured
        if config.use_registry_features:
            freg = FeatureRegistry(Path(config.registry_path))
            feature_info = freg.get_feature_set(config.feature_set_id)
            # Schema validation logic...
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
config = MLSignalActorConfig(
    component_id="ml_signal",
    model_path="models/my_model.onnx",
    signal_strategy=SignalStrategy.THRESHOLD,
    prediction_threshold=0.75,
    bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-BID-EXTERNAL"),
)

actor = MLSignalActor(config)
```

### Advanced Configuration

```python
config = MLSignalActorConfig(
    component_id="advanced_ml_signal",
    model_path="models/ensemble_model.onnx",
    signal_strategy=SignalStrategy.ENSEMBLE,
    optimization_config=OptimizationConfig(
        level=OptimizationLevel.OPTIMIZED,
        enable_model_warm_up=True,
        use_lock_free_buffers=True,
    ),
    strategy_config=StrategyConfig(
        ensemble_weights={"threshold": 0.4, "extremes": 0.3, "momentum": 0.3},
    ),
    enable_hot_reload=True,
    enable_regime_detection=True,
)
```

### Custom Strategy

```python
class CustomStrategy(SignalGenerationStrategy):
    def generate_signal(self, bar, prediction, confidence, features, context):
        # Custom logic...
        return MLSignal(...) if meets_criteria else None

config = MLSignalActorConfig(
    component_id="custom_signal",
    model_path="models/custom_model.onnx",
    custom_strategy=CustomStrategy(),
)
```

This ML actors framework provides a production-ready foundation for real-time machine learning in trading systems, with strict performance requirements, comprehensive observability, and robust fault tolerance.
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
