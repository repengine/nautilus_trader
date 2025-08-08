# BaseInferenceActor Implementation Plan

## Executive Summary

This document outlines the plan to implement a high-performance BaseInferenceActor for the new ML module, migrating key functionality from the OLD GenericInferenceActor while adhering to Nautilus Trader's strict performance requirements and ML integration architecture guidelines.

## Key Requirements

### Performance Requirements (from CLAUDE.md)

- Feature computation: < 500μs
- Model inference: < 2ms
- End-to-end signal: < 5ms
- Memory stable over 24h
- Zero allocations in hot path
- P99 latency must be < 5ms

### Architecture Requirements

- Follow Nautilus Actor pattern
- Support real-time inference in hot path
- Use pre-allocated numpy arrays
- No pandas in hot path
- No blocking operations in event handlers
- Use ONNX for model serialization when possible
- Feature parity with 1e-10 tolerance

## Migration Analysis

### What to Migrate from GenericInferenceActor

1. **Core Features to Preserve**:
   - Model hot-reload capability with version checking
   - Offline/online mode support (MLflow optional)
   - Health check functionality
   - Indicator history preservation during reload
   - Retry logic for model loading
   - Performance tracking and metrics

2. **Features to Enhance**:
   - Better separation of concerns (base vs concrete implementations)
   - More efficient feature buffer management
   - Improved error handling with circuit breaker pattern
   - Better integration with Nautilus metrics system

3. **Features to Remove/Refactor**:
   - MLflow direct integration (move to separate loader)
   - Complex metadata handling (simplify)
   - Redundant configuration options

### Integration with Existing BaseMLInferenceActor

The current `BaseMLInferenceActor` already provides:

- Basic actor lifecycle (on_start, on_bar, on_stop)
- Feature buffer pre-allocation
- Performance tracking
- MLSignal publishing
- Abstract methods for customization

We need to enhance it with:

- Model hot-reload capability
- Health monitoring
- Indicator management integration
- Better error recovery

## Implementation Strategy

### 1. Enhanced BaseMLInferenceActor

```python
class BaseMLInferenceActor(Actor, ABC):
    """Enhanced base class with production-ready features."""

    def __init__(self, config: MLActorConfig):
        super().__init__(config)

        # Core components
        self._model = None
        self._feature_buffer = None
        self._indicator_manager = None

        # Hot reload support
        self._model_version = None
        self._last_model_check = 0
        self._model_check_interval = config.model_check_interval

        # Health monitoring
        self._health_status = HealthStatus()
        self._consecutive_failures = 0
        self._circuit_breaker = CircuitBreaker(config.circuit_breaker_config)

        # Performance metrics
        self._inference_histogram = Histogram(buckets=[0.5, 1, 2, 5, 10])
        self._feature_computation_histogram = Histogram(buckets=[0.1, 0.25, 0.5, 1])
```

### 2. Model Loading Architecture

```python
class ModelLoader(ABC):
    """Abstract base for model loading strategies."""

    @abstractmethod
    def load_model(self, path: str) -> tuple[Any, dict[str, Any]]:
        """Load model and return (model, metadata)."""
        ...

    @abstractmethod
    def check_version(self, path: str) -> str:
        """Check model version without loading."""
        ...

class PickleModelLoader(ModelLoader):
    """Load pickle/joblib models."""

class ONNXModelLoader(ModelLoader):
    """Load ONNX models with optimized runtime."""

class MLflowModelLoader(ModelLoader):
    """Load models from MLflow (optional dependency)."""
```

### 3. Feature Engineering Integration

```python
class BaseMLInferenceActor(Actor, ABC):

    def _initialize_features(self):
        """Initialize feature engineering components."""
        # Use existing FeatureEngineer and IndicatorManager
        self._feature_engineer = FeatureEngineer(self._config.feature_config)
        self._indicator_manager = IndicatorManager(self._config.feature_config)

        # Pre-allocate buffers
        n_features = len(self._feature_engineer.get_feature_names())
        self._feature_buffer = np.zeros(n_features, dtype=np.float32)

    def _compute_features(self, bar: Bar) -> np.ndarray | None:
        """Compute features with <500μs latency."""
        start = time.perf_counter()

        # Update indicators
        self._indicator_manager.update_from_bar(bar)

        if not self._indicator_manager.all_initialized():
            return None

        # Get current bar data
        current_bar = {
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': float(bar.volume)
        }

        # Calculate features in-place
        features = self._feature_engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=self._indicator_manager,
            scaler=self._scaler
        )

        # Track performance
        latency = (time.perf_counter() - start) * 1000
        self._feature_computation_histogram.observe(latency)

        if latency > 0.5:  # 500μs threshold
            self.log.warning(f"Feature computation exceeded 500μs: {latency:.2f}ms")

        return features
```

### 4. Hot Reload Implementation

```python
def _schedule_model_checks(self):
    """Schedule periodic model version checks."""
    if self._config.enable_hot_reload:
        self.clock.set_timer(
            name="model_version_check",
            interval=timedelta(seconds=self._model_check_interval),
            callback=self._check_model_updates
        )

def _check_model_updates(self, event):
    """Check for model updates and hot-reload if needed."""
    try:
        new_version = self._model_loader.check_version(self._config.model_path)

        if new_version != self._model_version:
            self.log.info(f"New model version detected: {new_version}")

            # Preserve state
            old_indicator_manager = self._indicator_manager

            # Reload model
            self._load_model()

            # Restore state if configured
            if self._config.preserve_state_on_reload:
                self._preserve_indicator_history(old_indicator_manager)

    except Exception as e:
        self.log.error(f"Model check failed: {e}")
```

### 5. Health Monitoring

```python
class HealthStatus:
    """Track actor health metrics."""

    def __init__(self):
        self.is_healthy = True
        self.model_loaded = False
        self.indicators_initialized = False
        self.last_prediction_time = 0
        self.consecutive_failures = 0
        self.total_predictions = 0
        self.failed_predictions = 0

    def to_dict(self) -> dict[str, Any]:
        """Export health status as dict."""
        return {
            "healthy": self.is_healthy,
            "model_loaded": self.model_loaded,
            "indicators_initialized": self.indicators_initialized,
            "uptime_seconds": time.time() - self.start_time,
            "success_rate": self.get_success_rate(),
            "consecutive_failures": self.consecutive_failures
        }
```

### 6. Concrete Implementations

```python
class PickleMLInferenceActor(BaseMLInferenceActor):
    """Inference actor for pickle/joblib models."""

    def __init__(self, config: MLActorConfig):
        super().__init__(config)
        self._model_loader = PickleModelLoader()

class ONNXMLInferenceActor(BaseMLInferenceActor):
    """Inference actor for ONNX models with optimized runtime."""

    def __init__(self, config: MLActorConfig):
        super().__init__(config)
        self._model_loader = ONNXModelLoader()

    def _predict(self, features: np.ndarray) -> tuple[float, float]:
        """ONNX-optimized prediction."""
        # Use ONNX Runtime for inference
        outputs = self._ort_session.run(
            None,
            {self._input_name: features.reshape(1, -1).astype(np.float32)}
        )
        return float(outputs[0][0]), float(outputs[1][0])
```

## Performance Optimization Strategies

### 1. Pre-allocation

- All numpy arrays pre-allocated during initialization
- Feature buffers reused across predictions
- No dynamic memory allocation in hot path

### 2. Indicator Optimization

- Use Nautilus's Rust/Cython indicators
- Batch indicator updates when possible
- Maintain bounded history (deque with maxlen)

### 3. Model Inference

- Prefer ONNX for lowest latency
- Batch predictions when possible
- Use model quantization for faster inference

### 4. Error Handling

- Circuit breaker to prevent cascade failures
- Exponential backoff for retries
- Graceful degradation on errors

## Test Requirements

### Unit Tests

1. **Initialization Tests**
   - Configuration validation
   - Buffer allocation
   - Model loading

2. **Feature Computation Tests**
   - Correctness vs batch processing
   - Performance benchmarks
   - Edge cases (missing data, NaN handling)

3. **Hot Reload Tests**
   - Version detection
   - State preservation
   - Error recovery

4. **Health Monitoring Tests**
   - Status tracking
   - Circuit breaker behavior
   - Metrics collection

### Integration Tests

1. **End-to-End Inference**
   - Full pipeline from bar to signal
   - Latency measurements
   - Memory stability over time

2. **Model Compatibility**
   - Pickle/joblib models
   - ONNX models
   - Scikit-learn models

3. **Feature Parity Tests**
   - Batch vs online consistency
   - 1e-10 tolerance validation
   - Indicator state consistency

### Performance Tests

1. **Latency Benchmarks**
   - Feature computation < 500μs
   - Model inference < 2ms
   - End-to-end < 5ms

2. **Memory Tests**
   - No memory leaks over 24h
   - Bounded memory usage
   - GC pressure monitoring

## Implementation Timeline

### Phase 1: Core Enhancement (Week 1)

- Enhance BaseMLInferenceActor with production features
- Implement ModelLoader abstraction
- Add health monitoring

### Phase 2: Hot Reload (Week 2)

- Implement model version checking
- Add state preservation
- Test hot reload scenarios

### Phase 3: Performance Optimization (Week 3)

- ONNX runtime integration
- Performance benchmarking
- Memory optimization

### Phase 4: Testing & Documentation (Week 4)

- Comprehensive test coverage
- Performance validation
- Documentation and examples

## Risk Mitigation

### Technical Risks

1. **Latency Violations**
   - Mitigation: Extensive benchmarking, profiling tools
   - Fallback: Simplified feature sets for critical paths

2. **Feature Parity Issues**
   - Mitigation: Automated parity tests, numerical validation
   - Fallback: Logging and alerting on deviations

3. **Memory Leaks**
   - Mitigation: Bounded collections, regular profiling
   - Fallback: Periodic actor restart capability

### Operational Risks

1. **Model Compatibility**
   - Mitigation: Versioning, compatibility tests
   - Fallback: Multiple loader implementations

2. **Hot Reload Failures**
   - Mitigation: Rollback capability, health checks
   - Fallback: Manual intervention procedures

## Conclusion

This implementation plan provides a clear path to create a production-ready BaseInferenceActor that:

- Meets all performance requirements (<5ms end-to-end)
- Integrates seamlessly with existing ML infrastructure
- Provides enterprise features (hot reload, health monitoring)
- Maintains perfect feature parity with training
- Follows Nautilus Trader best practices

The phased approach allows for incremental delivery while maintaining system stability and performance throughout the implementation.
