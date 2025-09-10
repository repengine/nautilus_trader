# Performance Tests

This directory contains performance-critical tests that verify ML components meet latency and memory requirements for real-time trading.

## Purpose

Performance tests ensure the ML pipeline can operate within the strict timing constraints of live trading. They catch performance regressions and validate optimization efforts.

## Performance Requirements

### Latency Targets

- **Feature computation**: < 500μs per bar
- **Model inference**: < 2ms per prediction
- **End-to-end signal**: < 5ms from bar to signal
- **Hot path operations**: Zero allocation guarantee

### Memory Requirements

- **Stable over 24h**: No memory leaks during extended runs
- **Bounded growth**: All data structures have maximum size limits
- **GC pressure**: Minimal allocation in hot paths

## Test Categories

### Hot Path Tests (`test_hot_path_fixes.py`)

- Zero-allocation guarantees
- Sub-millisecond feature computation
- Memory stability validation
- Cache efficiency verification

### Allocation Tests (`test_zero_allocation.py`)

- Memory allocation tracking
- Garbage collection impact
- Buffer reuse verification
- Object pooling effectiveness

### Benchmarks (`benchmark_hot_path.py`)

- Performance baseline establishment
- Regression detection (20% tolerance)
- Latency distribution analysis
- Throughput measurement

## Automated Performance Monitoring

Performance tests run automatically when ML inference or feature files change:

```bash
# Run performance tests
make test-performance

# Update performance baselines
make update-ml-baseline
```

## Performance Regression Policy

- **20% degradation**: Warning (investigate but may proceed)
- **>20% degradation**: Blocking (must fix before merge)
- **P99 > 5ms**: Hard failure (violates trading requirements)

## When to Add Performance Tests

Add performance tests when:

- Creating new inference pathways
- Modifying feature computation logic
- Adding new indicators or models
- Optimizing hot path operations
- Introducing caching or pooling

## Example Performance Test

```python
def test_feature_computation_latency():
    """Verify feature computation stays under 500μs."""
    start = time.perf_counter_ns()
    features = engineer.compute_features(bar)
    duration = time.perf_counter_ns() - start

    assert duration < 500_000  # 500μs in nanoseconds
```

Performance is critical for trading success - these tests ensure we maintain the speed needed for profitable operations.
