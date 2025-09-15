# ML Performance Guardrails

This directory contains performance guardrails for the ML pipeline that ensure production reliability requirements are maintained. These tests **FAIL CI** if performance regressions are detected.

## Overview

The performance guardrails validate:

1. **Feature Parity**: Online and offline feature computation produce identical results
2. **Zero Allocations**: Hot path has no memory allocations after warmup
3. **P99 Latency Budgets**: Critical performance thresholds are maintained
4. **Buffer Reuse**: Pre-allocated buffers are reused correctly
5. **Memory Stability**: No memory leaks over extended operation

## Performance Requirements

| Component | P99 Latency Budget | Zero Allocations | Parity Tolerance |
|-----------|-------------------|------------------|------------------|
| Feature Computation | <500μs | ✅ | <1e-6 |
| Model Inference | <2ms | ✅ | N/A |
| End-to-End Signal | <5ms | ✅ | N/A |

## Usage

### Local Development

```bash
# Run all performance guardrails
make pytest-ml-guardrails

# Run in strict mode (tighter requirements)
make pytest-ml-guardrails-strict

# Run only zero-allocation validation
make pytest-ml-zero-allocation

# Run with custom relax factor for noisy environments
ML_BENCH_RELAX=2.0 make pytest-ml-guardrails
```

### CI Integration

The guardrails are designed to integrate with CI pipelines:

```bash
# Standard CI run
python ml/tests/performance/ci_performance_guardrails.py

# Strict mode for production
python ml/tests/performance/ci_performance_guardrails.py --strict

# With performance report output
python ml/tests/performance/ci_performance_guardrails.py --report-file performance_report.json
```

### Exit Codes

- `0`: All guardrails passed
- `1`: Performance regressions detected (fails CI)
- `2`: Test execution failed

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_BENCH_RELAX` | `1.0` | Performance requirement relaxation factor |
| `ML_GUARDRAILS_STRICT` | `false` | Enable strict mode with tighter requirements |
| `PYTEST_XDIST_WORKER` | - | Detected automatically for parallel execution |
| `CI` | - | Detected automatically for CI environments |

### Actor Configuration

Feature parity smoke-checks are configurable in `MLSignalActorConfig`:

```python
config = MLSignalActorConfig(
    # ... other config ...
    enable_parity_smoke_check=True,          # Enable smoke-check
    parity_smoke_check_window_bars=200,      # Window size
    parity_tolerance=1e-6,                   # Drift tolerance
)
```

## Test Categories

### 1. Feature Computation Guardrails

**File**: `test_parity_buffer_guardrails.py::TestFeatureComputationGuardrails`

- P99 latency budget enforcement (<500μs)
- Zero allocation validation
- Throughput requirements (>2000 computations/second)

### 2. Feature Parity Guardrails

**File**: `test_parity_buffer_guardrails.py::TestFeatureParityGuardrails`

- Online vs offline computation parity
- Drift detection and reporting
- Metrics emission validation

### 3. Model Inference Guardrails

**File**: `test_parity_buffer_guardrails.py::TestModelInferenceGuardrails`

- ONNX inference P99 budget (<2ms)
- Model hot-swapping latency (<100ms)
- Inference zero allocation validation

### 4. End-to-End Guardrails

**File**: `test_parity_buffer_guardrails.py::TestEndToEndGuardrails`

- Complete signal generation pipeline (<5ms)
- Concurrent processing under load
- Memory stability over extended operation

### 5. Buffer Reuse Guardrails

**File**: `test_parity_buffer_guardrails.py::TestBufferReuseGuardrails`

- Pre-allocated buffer reuse validation
- Memory view correctness
- 24-hour memory stability simulation

### 6. Performance Regression Detection

**File**: `test_parity_buffer_guardrails.py::TestPerformanceRegressionGuardrails`

- Baseline performance comparison
- Regression threshold enforcement (20% degradation limit)
- Historical performance tracking

## Metrics

The guardrails integrate with existing Prometheus metrics:

### Parity Metrics

- `ml_feature_parity_checks_total`: Total parity checks executed
- `ml_feature_parity_drift`: Maximum feature difference detected

### Performance Metrics

All standard ML performance metrics are validated during guardrail execution.

## Failure Scenarios

### P99 Budget Exceeded

```
❌ Feature computation P99 latency 650.2μs exceeded budget 500.0μs
```

**Resolution**: Optimize feature computation or investigate system performance.

### Memory Allocation Detected

```
❌ Feature computation allocated 1024 bytes (10.2 per call), expected near-zero
```

**Resolution**: Eliminate allocations in hot path, ensure buffer reuse.

### Feature Parity Drift

```
❌ Feature parity drift 2.5e-5 exceeded tolerance 1.0e-6
```

**Resolution**: Investigate numerical stability, check for race conditions.

### Memory Leak Detected

```
❌ Memory leak detected: 25.3MB increase after 10k operations
```

**Resolution**: Check for retained references, validate garbage collection.

## Best Practices

### 1. Development Workflow

1. **Before Changes**: Run `make pytest-ml-guardrails` to establish baseline
2. **After Changes**: Run guardrails again to detect regressions
3. **Before PR**: Run `make pytest-ml-guardrails-strict` for thorough validation

### 2. Performance Optimization

- Focus on P99 latency, not average performance
- Ensure zero allocations in hot path after warmup
- Pre-allocate all buffers during initialization
- Use memory views instead of copying arrays

### 3. CI Integration

- Include guardrails in PR validation
- Use strict mode for release branches
- Generate performance reports for tracking
- Set appropriate timeout values for CI stability

### 4. Troubleshooting

- Use `ML_BENCH_RELAX=3.0` for noisy CI environments
- Check system load during performance tests
- Validate test isolation (no interference between tests)
- Consider CPU frequency scaling on CI systems

## Architecture

### Measurement Strategy

The guardrails use robust measurement techniques:

- **P99 Calculation**: Direct percentile computation from many samples
- **Warmup**: Eliminates JIT compilation and cache effects
- **Environment Detection**: Adapts to CI vs local environments
- **Isolation**: Each test runs independently

### Allocation Detection

Zero-allocation validation uses Python's `tracemalloc` module:

1. Warmup phase to fill all caches
2. Garbage collection to clean baseline
3. Memory snapshot before test execution
4. Memory snapshot after test execution
5. Analysis of allocation differences

### Buffer Reuse Validation

Buffer reuse is validated using NumPy's memory sharing detection:

```python
assert np.shares_memory(result, pre_allocated_buffer)
```

## Contributing

When adding new performance-critical code:

1. **Add Guardrails**: Create tests in the appropriate category
2. **Set Budgets**: Define realistic P99 latency budgets
3. **Validate Zero Allocations**: Ensure hot path has no allocations
4. **Test Parity**: Verify online/offline computation consistency
5. **Update Documentation**: Document new requirements and thresholds

### Example Test

```python
def test_new_component_p99_budget(self):
    """Ensure new component meets P99 latency budget."""

    def test_function():
        # Your performance-critical code here
        pass

    # Measure P99 latency
    p99_ns = measure_p99_latency_ns(test_function, iterations=1000)

    # Enforce budget (adjust as needed)
    assert_p99_budget(p99_ns, 1_000_000, "New component")  # 1ms budget
```

## Future Enhancements

- [ ] Automated performance baseline updates
- [ ] Performance trend analysis and alerts
- [ ] Integration with continuous profiling
- [ ] Cross-platform performance validation
- [ ] Automated optimization recommendations