# MLSignalActor Implementation Plan - Phase 4.1

## Executive Summary
This plan details the implementation of a high-performance MLSignalActor for real-time ML inference in Nautilus Trader. The actor will achieve <2ms inference latency through strict hot path optimization, pre-allocated buffers, and ONNX runtime optimization.

## Current Status
✅ **Completed:**

- MLDataLoader for efficient data loading
- FeatureEngineer with comprehensive features
- UnifiedXGBoostTrainer and UnifiedLightGBMTrainer
- MLflowManager for experiment tracking
- Feature parity validation tests
- Base ML actor infrastructure (ml/actors/base.py)
- Signal actor with basic functionality (ml/actors/signal.py)

🔄 **Existing Infrastructure:**

- Base MLInferenceActor with hot reload support
- ONNX model loading with optimization
- Circuit breaker and health monitoring
- Pre-allocated feature buffers
- Prometheus metrics integration

## Implementation Requirements

### 1. Performance Requirements

- **Feature computation:** <500μs
- **Model inference:** <2ms
- **End-to-end signal:** <5ms
- **Memory:** Stable over 24h operation
- **Zero allocations:** In hot path

### 2. Architecture Overview

```python
MLSignalActor (Enhanced)
├── Hot Path (Real-time)
│   ├── Feature Computation (<500μs)
│   │   ├── Pre-allocated buffers
│   │   ├── Nautilus indicators (Rust/Cython)
│   │   └── In-place operations
│   ├── ONNX Inference (<2ms)
│   │   ├── Optimized session
│   │   ├── CPU provider
│   │   └── Sequential execution
│   └── Signal Publishing (<100μs)
│       ├── Message bus
│       └── Zero-copy where possible
└── Cold Path (Initialization)
    ├── Model Loading
    ├── Buffer Pre-allocation
    ├── Indicator Setup
    └── Monitoring Setup
```

## Phase 4.1: Core Enhancement Plan

### Task 1: Optimize ONNX Runtime Configuration
**Goal:** Achieve <2ms inference latency

```python
# Enhanced ONNX configuration
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session_options.intra_op_num_threads = 1  # Single thread for predictable latency
session_options.inter_op_num_threads = 1
session_options.enable_cpu_mem_arena = False  # Disable for lower latency
session_options.enable_mem_pattern = False
session_options.enable_mem_reuse = True

# Provider options for CPU optimization
providers = [
    ('CPUExecutionProvider', {
        'arena_extend_strategy': 'kSameAsRequested',
    })
]
```

### Task 2: Implement Zero-Copy Feature Pipeline
**Goal:** Eliminate memory allocations in hot path

```python
class OptimizedFeaturePipeline:
    def __init__(self, n_features: int):
        # Pre-allocate all buffers
        self.feature_buffer = np.empty(n_features, dtype=np.float32)
        self.feature_view = memoryview(self.feature_buffer)
        self.normalized_buffer = np.empty(n_features, dtype=np.float32)

        # Pre-allocate ONNX input
        self.onnx_input = np.empty((1, n_features), dtype=np.float32)

    def compute_features_inplace(self, bar: Bar, indicators: dict) -> memoryview:
        """Compute features directly into pre-allocated buffer."""
        # All operations write directly to self.feature_buffer
        # Return memoryview for zero-copy access
        return self.feature_view
```

### Task 3: Implement Advanced Signal Generation Strategies
**Goal:** Sophisticated signal generation with minimal overhead

```python
class SignalGenerator:
    """High-performance signal generation with multiple strategies."""

    def __init__(self, config: SignalConfig):
        # Pre-allocate strategy buffers
        self.threshold_buffer = np.empty(10, dtype=np.float32)
        self.momentum_buffer = np.empty(config.momentum_lookback, dtype=np.float32)
        self.adaptive_buffer = np.empty(config.adaptive_window, dtype=np.float32)

        # Pre-compile strategy functions
        self.strategies = {
            SignalStrategy.THRESHOLD: self._threshold_strategy,
            SignalStrategy.EXTREMES: self._extremes_strategy,
            SignalStrategy.MOMENTUM: self._momentum_strategy,
            SignalStrategy.ENSEMBLE: self._ensemble_strategy,
            SignalStrategy.ADAPTIVE: self._adaptive_strategy,
        }
```

### Task 4: Implement Lock-Free Ring Buffers
**Goal:** Efficient history tracking without allocations

```python
class RingBuffer:
    """Lock-free ring buffer for prediction history."""

    def __init__(self, size: int, dtype=np.float32):
        self.buffer = np.empty(size, dtype=dtype)
        self.size = size
        self.index = 0
        self.count = 0

    def append(self, value: float) -> None:
        """Add value to ring buffer (overwrites oldest)."""
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def get_window(self, n: int) -> np.ndarray:
        """Get last n values as view (no copy)."""
        if n > self.count:
            return self.buffer[:self.count]
        start = (self.index - n) % self.size
        if start + n <= self.size:
            return self.buffer[start:start + n]
        # Handle wrap-around
        return np.concatenate([self.buffer[start:], self.buffer[:n - (self.size - start)]])
```

### Task 5: Implement Performance Monitoring
**Goal:** Track performance without impacting hot path

```python
class PerformanceMonitor:
    """Non-blocking performance monitoring."""

    def __init__(self):
        # Use atomic counters for lock-free updates
        self.feature_time_ns = 0
        self.inference_time_ns = 0
        self.signal_time_ns = 0
        self.prediction_count = 0

        # Percentile tracking with reservoir sampling
        self.latency_reservoir = np.empty(1000, dtype=np.float64)
        self.reservoir_index = 0

    def record_latency(self, latency_ns: int) -> None:
        """Record latency using reservoir sampling."""
        if self.reservoir_index < 1000:
            self.latency_reservoir[self.reservoir_index] = latency_ns
            self.reservoir_index += 1
        else:
            # Reservoir sampling for uniform distribution
            j = np.random.randint(0, self.prediction_count)
            if j < 1000:
                self.latency_reservoir[j] = latency_ns
```

### Task 6: Implement Model Hot-Swapping
**Goal:** Zero-downtime model updates

```python
class ModelSwapper:
    """Atomic model swapping with state preservation."""

    def __init__(self):
        self.current_model = None
        self.next_model = None
        self.swap_pending = False

    def prepare_swap(self, new_model_path: str) -> None:
        """Load new model in background."""
        # Load model in separate thread
        self.next_model = self._load_model_async(new_model_path)
        self.swap_pending = True

    def execute_swap(self) -> None:
        """Atomically swap models between bars."""
        if self.swap_pending:
            old_model = self.current_model
            self.current_model = self.next_model
            self.next_model = None
            self.swap_pending = False
            # Cleanup old model
            if old_model:
                self._cleanup_model(old_model)
```

## Implementation Steps

### Phase 4.1.1: Core Optimization (Week 1)

1. **Day 1-2:** Enhance ONNX runtime configuration
   - Implement optimized session options
   - Add CPU provider tuning
   - Benchmark inference latency

2. **Day 3-4:** Implement zero-copy feature pipeline
   - Create pre-allocated buffer system
   - Implement in-place feature computation
   - Add memoryview for zero-copy access

3. **Day 5:** Performance validation
   - Create latency benchmarks
   - Validate <2ms inference requirement
   - Profile memory allocations

### Phase 4.1.2: Signal Generation (Week 2)

1. **Day 1-2:** Implement advanced signal strategies
   - Port existing strategies to optimized versions
   - Add ring buffer for history tracking
   - Implement lock-free updates

2. **Day 3-4:** Add adaptive thresholds
   - Implement market regime detection
   - Add volatility-based adjustments
   - Create dynamic threshold calculation

3. **Day 5:** Integration testing
   - Test all signal strategies
   - Validate signal quality
   - Performance benchmarking

### Phase 4.1.3: Monitoring & Hot-Swap (Week 3)

1. **Day 1-2:** Performance monitoring
   - Implement reservoir sampling
   - Add percentile tracking
   - Create Prometheus exporters

2. **Day 3-4:** Model hot-swapping
   - Implement atomic swap mechanism
   - Add state preservation
   - Create background loader

3. **Day 5:** System integration
   - Integration with existing monitoring
   - End-to-end testing
   - Documentation

## Testing Strategy

### Performance Tests

```python
def test_inference_latency():
    """Verify <2ms inference latency."""
    actor = MLSignalActor(config)
    features = np.random.randn(100).astype(np.float32)

    # Warm-up
    for _ in range(100):
        actor._predict(features)

    # Measure
    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        actor._predict(features)
        latencies.append(time.perf_counter_ns() - start)

    p99_latency_ms = np.percentile(latencies, 99) / 1e6
    assert p99_latency_ms < 2.0, f"P99 latency {p99_latency_ms}ms exceeds 2ms"
```

### Memory Tests

```python
def test_memory_stability():
    """Verify no memory growth over time."""
    import tracemalloc

    actor = MLSignalActor(config)
    tracemalloc.start()

    # Process many bars
    for _ in range(10000):
        bar = create_test_bar()
        actor.on_bar(bar)

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    # Verify no significant allocations in hot path
    for stat in top_stats[:10]:
        assert 'on_bar' not in str(stat), f"Memory allocation in hot path: {stat}"
```

### Integration Tests

```python
def test_end_to_end_latency():
    """Test complete signal generation pipeline."""
    actor = MLSignalActor(config)

    bars = load_test_bars()
    latencies = []

    for bar in bars:
        start = time.perf_counter_ns()
        actor.on_bar(bar)
        latencies.append(time.perf_counter_ns() - start)

    p99_latency_ms = np.percentile(latencies, 99) / 1e6
    assert p99_latency_ms < 5.0, f"End-to-end P99 {p99_latency_ms}ms exceeds 5ms"
```

## Monitoring & Observability

### Key Metrics

1. **Latency Metrics**
   - `nautilus_ml_feature_computation_seconds` - Feature computation time
   - `nautilus_ml_inference_latency_seconds` - Model inference time
   - `nautilus_ml_signal_generation_seconds` - Total signal generation time

2. **Throughput Metrics**
   - `nautilus_ml_predictions_per_second` - Prediction rate
   - `nautilus_ml_signals_generated_total` - Total signals generated
   - `nautilus_ml_bars_processed_total` - Total bars processed

3. **Health Metrics**
   - `nautilus_ml_actor_health` - Overall health status
   - `nautilus_ml_circuit_breaker_state` - Circuit breaker status
   - `nautilus_ml_model_version` - Current model version

### Grafana Dashboards

- Real-time latency monitoring
- Signal quality tracking
- Model performance comparison
- Resource utilization

## Risk Mitigation

### Performance Risks

1. **Risk:** Latency spikes during GC
   - **Mitigation:** Pre-allocate all buffers, avoid allocations

2. **Risk:** CPU contention
   - **Mitigation:** Single-threaded inference, CPU affinity

3. **Risk:** Model complexity
   - **Mitigation:** Model pruning, quantization if needed

### Operational Risks

1. **Risk:** Model corruption
   - **Mitigation:** Checksum validation, fallback models

2. **Risk:** Memory leaks
   - **Mitigation:** Bounded buffers, regular monitoring

3. **Risk:** Signal quality degradation
   - **Mitigation:** Continuous validation, circuit breakers

## Success Criteria

### Performance

- ✅ P99 feature computation <500μs
- ✅ P99 inference latency <2ms
- ✅ P99 end-to-end <5ms
- ✅ Zero allocations in hot path
- ✅ Memory stable over 24h

### Functionality

- ✅ All signal strategies working
- ✅ Hot reload without downtime
- ✅ Feature parity maintained
- ✅ Monitoring integrated
- ✅ Circuit breaker protection

### Quality

- ✅ >90% test coverage
- ✅ Performance benchmarks passing
- ✅ Integration tests green
- ✅ Documentation complete
- ✅ Code review approved

## Next Steps

1. **Immediate Actions:**
   - Review existing signal.py implementation
   - Identify optimization opportunities
   - Create performance baseline

2. **Week 1 Goals:**
   - Implement ONNX optimizations
   - Create zero-copy pipeline
   - Achieve <2ms inference

3. **Week 2 Goals:**
   - Enhance signal strategies
   - Add adaptive thresholds
   - Complete integration tests

4. **Week 3 Goals:**
   - Finalize monitoring
   - Implement hot-swapping
   - Complete documentation

## Appendix A: Code Organization

```
ml/actors/
├── signal_enhanced.py       # Enhanced MLSignalActor
├── performance.py           # Performance monitoring
├── buffers.py              # Ring buffer implementations
├── strategies.py           # Signal generation strategies
└── tests/
    ├── test_signal_performance.py
    ├── test_signal_strategies.py
    └── benchmarks/
        ├── latency_benchmark.py
        └── memory_benchmark.py
```

## Appendix B: Configuration Example

```yaml
# ml_signal_actor.yaml
actor:
  class: MLSignalActor
  config:
    model_path: "models/xgboost_v3.onnx"
    bar_type: "BTCUSDT.BINANCE-1-MINUTE"

    # Performance settings
    max_feature_latency_ms: 0.5
    max_inference_latency_ms: 2.0
    warm_up_period: 100

    # Signal generation
    signal_strategy: "adaptive"
    prediction_threshold: 0.7
    adaptive_window: 20
    adaptive_volatility_factor: 2.0
    min_signal_separation_bars: 3

    # Optimization
    enable_hot_reload: true
    hot_reload_interval: 300
    preserve_state_on_reload: true

    # Monitoring
    enable_health_monitoring: true
    enable_circuit_breaker: true
    circuit_breaker_threshold: 0.1
    circuit_breaker_window: 100
```

## Appendix C: Benchmark Results (Target)

```
Feature Computation Latency:
  P50: 200μs
  P95: 400μs
  P99: 500μs

Model Inference Latency:
  P50: 0.8ms
  P95: 1.5ms
  P99: 2.0ms

End-to-End Signal Generation:
  P50: 1.5ms
  P95: 3.0ms
  P99: 5.0ms

Memory Usage:
  Startup: 150MB
  After 1h: 152MB
  After 24h: 153MB

Throughput:
  Bars/sec: 1000+
  Signals/sec: 50-100
```

---

**Document Version:** 1.0
**Last Updated:** 2025-01-07
**Author:** ML Infrastructure Team
**Status:** Ready for Implementation
