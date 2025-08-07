# Optimized ML Signal Actor Implementation

This module implements the enhanced MLSignalActor with hot path optimization for <2ms inference latency, as outlined in Phase 4.1 of the Signal Actor Implementation Plan.

## Overview

The optimized ML Signal Actor achieves enterprise-grade performance for real-time trading through:

- **Zero-allocation hot path**: Pre-allocated buffers with memoryviews for zero-copy operations
- **Optimized ONNX runtime**: Single-threaded execution with CPU optimization
- **Lock-free data structures**: Ring buffers and reservoir sampling for efficient history tracking
- **Atomic model hot-swapping**: Zero-downtime model updates with state preservation
- **Circuit breaker protection**: Automatic fallback during performance degradation

## Performance Targets

| Metric | Target | Implementation |
|--------|--------|----------------|
| Feature Computation P99 | <500μs | Pre-allocated buffers, Nautilus indicators |
| Model Inference P99 | <2ms | Optimized ONNX runtime configuration |
| End-to-End P99 | <5ms | Complete zero-allocation pipeline |
| Memory Growth | Stable over 24h | Bounded ring buffers, no dynamic allocation |

## Architecture

```
OptimizedMLSignalActor
├── Hot Path (Real-time)
│   ├── Feature Computation (<500μs)
│   │   ├── PreAllocatedFeatureCache
│   │   ├── Zero-copy memoryviews
│   │   └── Nautilus indicators (Rust/Cython)
│   ├── ONNX Inference (<2ms)
│   │   ├── Optimized session options
│   │   ├── CPU provider tuning
│   │   └── Pre-allocated input buffers
│   └── Signal Generation (<100μs)
│       ├── LockFreeRingBuffer history
│       ├── ReservoirSampler percentiles
│       └── Adaptive threshold strategies
└── Cold Path (Initialization)
    ├── Model Loading & Warm-up
    ├── Buffer Pre-allocation
    ├── Performance Monitor Setup
    └── Hot-swap Infrastructure
```

## Key Components

### 1. Feature Cache (`feature_cache.py`)

**LockFreeRingBuffer**: O(1) append/access for prediction history

```python
buffer = LockFreeRingBuffer(size=1000, dtype=np.float32)
buffer.append(prediction)  # Zero-allocation append
recent = buffer.get_last(n=10)  # Zero-copy when possible
```

**ReservoirSampler**: Uniform sampling for percentile calculation

```python
sampler = ReservoirSampler(reservoir_size=1000)
sampler.add_sample(prediction)
p95 = sampler.get_percentile(95.0)  # Efficient percentile tracking
```

**PreAllocatedFeatureCache**: Zero-allocation feature pipeline

```python
cache = PreAllocatedFeatureCache(n_features=256)
buffer = cache.get_current_buffer()  # Pre-allocated feature buffer
onnx_input = cache.prepare_onnx_input()  # Zero-copy ONNX preparation
```

### 2. Signal Configuration (`signal_config.py`)

**OptimizedMLSignalActorConfig**: Enhanced configuration with performance tuning

```python
config = OptimizedMLSignalActorConfig(
    signal_strategy=SignalStrategy.ADAPTIVE,
    threshold_strategy=ThresholdStrategy.REGIME_AWARE,
    enable_model_warm_up=True,
    warm_up_iterations=100,
)
```

**Performance Configurations**:

- `ONNXOptimizationConfig`: Single-threaded execution, disabled memory arena
- `AdaptiveThresholdsConfig`: Market regime-aware threshold adjustment
- `HotPathOptimizationConfig`: Zero-copy and pre-allocation settings

### 3. Optimized Actor (`signal_optimized.py`)

**OptimizedMLSignalActor**: Main implementation with performance enhancements

```python
actor = OptimizedMLSignalActor(config)
# Automatically loads model with optimization
# Pre-allocates all buffers
# Sets up performance monitoring
```

**Key Optimizations**:

- `_compute_features_optimized()`: Uses pre-allocated buffers and memoryviews
- `_predict_optimized()`: ONNX inference with optimal session configuration
- `_generate_signal_optimized()`: Zero-allocation signal generation
- Hot model swapping with `ModelSwapper`

## Usage Examples

### Basic Optimized Actor

```python
from ml.actors import OptimizedMLSignalActor, OptimizedMLSignalActorConfig
from ml.actors.signal_config import SignalStrategy, ThresholdStrategy

config = OptimizedMLSignalActorConfig(
    actor_id="optimized_signal",
    bar_type="BTCUSDT.BINANCE-1-MINUTE",
    model_path="models/xgboost_optimized.onnx",
    signal_strategy=SignalStrategy.ADAPTIVE,
    threshold_strategy=ThresholdStrategy.REGIME_AWARE,

    # Performance tuning
    enable_model_warm_up=True,
    warm_up_iterations=100,
    adaptive_window=20,
    min_signal_separation_bars=3,
)

actor = OptimizedMLSignalActor(config)
```

### Performance Monitoring

```python
# Get comprehensive performance statistics
stats = actor.get_performance_stats()
print(f"P99 Inference Latency: {stats['latency_percentiles']['inference'][99.0]:.3f}ms")
print(f"Signal Rate: {stats['signal_rate']:.2%}")
print(f"Error Rate: {stats['error_rate']:.2%}")

# Check if performance targets are being met
latency_p99 = stats['latency_percentiles']['total'][99.0]
assert latency_p99 < 5.0, f"P99 latency {latency_p99:.3f}ms exceeds 5ms target"
```

### Hot Model Swapping

```python
# Model swapping happens automatically during safe points
# State is preserved across reloads
actor.reload_model("models/xgboost_v2_optimized.onnx")

# Check swap status
if actor._model_swapper.swap_pending:
    print("Model swap pending...")
```

## Configuration Options

### Signal Generation Strategies

- **THRESHOLD**: Simple confidence threshold
- **EXTREMES**: Top/bottom percentile predictions
- **MOMENTUM**: Prediction momentum analysis
- **ENSEMBLE**: Weighted combination of strategies
- **ADAPTIVE**: Market regime-aware adaptive thresholds

### Threshold Strategies

- **FIXED**: Static confidence threshold
- **PERCENTILE**: Rolling percentile-based threshold
- **VOLATILITY_ADJUSTED**: Volatility-based adjustment
- **REGIME_AWARE**: Market regime-specific thresholds

### Performance Tuning

```python
# ONNX Runtime Optimization
onnx_config = ONNXOptimizationConfig(
    graph_optimization_level="ORT_ENABLE_ALL",
    execution_mode="ORT_SEQUENTIAL",
    intra_op_num_threads=1,  # Single-threaded for predictable latency
    enable_cpu_mem_arena=False,  # Disabled for lower latency
)

# Hot Path Optimization
hotpath_config = HotPathOptimizationConfig(
    enable_zero_copy=True,
    pre_allocate_buffers=True,
    max_inference_latency_us=2000,  # 2ms target
    circuit_breaker_threshold=0.1,
)
```

## Testing

### Unit Tests

```bash
# Test core components
python -m pytest ml/tests/unit/test_signal_optimized.py -v

# Test specific components
python -m pytest ml/tests/unit/test_signal_optimized.py::TestLockFreeRingBuffer -v
```

### Performance Benchmarks

```bash
# Run performance benchmarks
python -m pytest ml/tests/benchmarks/test_signal_performance.py -v

# Run benchmarks directly
python ml/tests/benchmarks/test_signal_performance.py
```

### Memory Stability Tests

```bash
# Long-running memory stability test
python -m pytest ml/tests/benchmarks/test_signal_performance.py::TestOptimizedActorPerformance::test_memory_stability -v
```

## Performance Validation

The implementation includes comprehensive benchmarks that validate:

1. **Component Performance**:
   - Ring buffer operations: <1μs P99
   - Reservoir sampling: <5μs P99
   - Feature cache access: <1μs P99

2. **End-to-End Performance**:
   - Feature computation: <500μs P99
   - Model inference: <2ms P99
   - Signal generation: <5ms P99

3. **Memory Stability**:
   - No unbounded growth over 10,000+ iterations
   - Zero allocations in hot path
   - Bounded buffer sizes

## Integration with Nautilus Trader

The optimized actor is fully compatible with Nautilus Trader's architecture:

```python
# Use in backtest
from nautilus_trader.backtest import BacktestEngine

engine = BacktestEngine()
engine.add_actor(actor)

# Use in live trading
from nautilus_trader.live import LiveTradingNode

node = LiveTradingNode()
node.add_actor(actor)
```

## Monitoring & Observability

### Prometheus Metrics

- `nautilus_ml_optimized_signal_latency_seconds`: Component-wise latency tracking
- `nautilus_ml_adaptive_threshold`: Adaptive threshold values
- `nautilus_ml_signals_generated_total`: Signal generation counts

### Performance Dashboard
The actor exposes detailed performance statistics for monitoring:

- Real-time latency percentiles (P50, P95, P99)
- Signal quality metrics
- Memory usage tracking
- Circuit breaker status

## Troubleshooting

### High Latency

1. Check model complexity - consider quantization
2. Verify single-threaded execution
3. Monitor GC activity - ensure zero-allocation hot path
4. Check CPU affinity and power management

### Memory Growth

1. Verify bounded buffer sizes
2. Check for reference leaks in indicators
3. Monitor feature cache history limits
4. Review model loading/cleanup

### Signal Quality

1. Check adaptive threshold behavior
2. Monitor market regime detection
3. Validate feature parity between training/inference
4. Review signal separation settings

## Future Enhancements

- [ ] GPU acceleration for complex models
- [ ] Model ensemble with weighted averaging
- [ ] Advanced market regime detection
- [ ] Real-time feature importance analysis
- [ ] Distributed inference across multiple actors
