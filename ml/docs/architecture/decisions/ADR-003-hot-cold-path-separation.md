# ADR-003: Hot/Cold Path Separation Strategy

## Status
**ACCEPTED** - 2024-01-15

## Context

The Nautilus Trader ML system must support both real-time trading operations and offline analytical tasks with vastly different performance requirements:

**Real-time Trading (Hot Path)**:
- Sub-5ms P99 latency requirements for trading signals  
- Memory-constrained environments (edge deployment)
- Zero allocation after warmup to avoid garbage collection pauses
- High throughput (thousands of operations per second)
- Predictable performance under load

**Offline Analytics (Cold Path)**:
- Model training can take hours
- Large dataset processing (GBs of historical data)
- Complex feature engineering with expensive operations
- Database migrations and schema changes
- Batch processing and backfill operations

Previously, the system mixed hot and cold path operations, causing:
- Unpredictable latencies in trading operations due to GC from heavy processing
- Resource contention between training and inference
- Complex codebase where performance-critical and analytical code were intermingled
- Difficulty in optimizing for different deployment scenarios

## Decision

**Enforce strict separation between hot path (real-time) and cold path (offline) operations with different performance contracts and implementation strategies.**

### Hot Path Requirements
- **Latency**: <5ms P99 for complete pipeline (data → features → prediction → signal)
- **Memory**: Zero allocations after warmup period
- **Deployment**: Optimized for edge/container deployment
- **Operations**: Feature computation, model inference, signal generation, risk checks

### Cold Path Requirements  
- **Throughput**: Optimized for batch processing large datasets
- **Resources**: Can use significant CPU/memory/disk
- **Operations**: Model training, feature backfill, data migrations, analytics
- **Flexibility**: Can use heavy libraries and complex algorithms

### Separation Enforcement
- **Code Organization**: Separate modules for hot/cold path implementations
- **Performance Testing**: Different SLAs and test suites for each path
- **Resource Allocation**: Isolated resource pools and deployment strategies
- **Implementation Patterns**: Different coding patterns optimized for each path

## Consequences

### Positive
- **Predictable Performance**: Hot path latency becomes deterministic and reliable
- **Optimal Resource Usage**: Each path optimized for its specific requirements  
- **Deployment Flexibility**: Hot path can be deployed on edge devices, cold path on powerful servers
- **Clear Boundaries**: Developers understand performance implications of their code
- **Testing Clarity**: Different test strategies and SLAs for different paths

### Negative
- **Code Duplication**: Some functionality implemented differently for hot/cold paths
- **Complexity**: Two different approaches to similar problems
- **Coordination Overhead**: Data sharing between paths requires careful design
- **Development Burden**: Developers must understand both paradigms

### Risks
- **Feature Parity**: Hot and cold path implementations might diverge
- **Data Consistency**: Synchronization issues between hot and cold processing
- **Resource Waste**: Under-utilization if paths are over-separated

## Implementation Details

### Hot Path Implementation Standards

#### Memory Management
```python
import numpy as np
from typing import Final

class HotPathFeatureComputer:
    """Hot path optimized feature computation."""
    
    def __init__(self, max_lookback: int = 20):
        # Pre-allocate ALL arrays during initialization
        self.MAX_LOOKBACK: Final = max_lookback
        self.price_buffer = np.zeros(max_lookback, dtype=np.float32)
        self.volume_buffer = np.zeros(max_lookback, dtype=np.float32)  
        self.feature_output = np.zeros(5, dtype=np.float32)
        self.buffer_index = 0
    
    def compute_features(self, price: float, volume: float) -> np.ndarray:
        """ZERO allocations - reuse pre-allocated arrays."""
        # Update circular buffer (no new allocations)
        idx = self.buffer_index % self.MAX_LOOKBACK
        self.price_buffer[idx] = price
        self.volume_buffer[idx] = volume
        self.buffer_index += 1
        
        # Compute in-place using pre-allocated output array
        self.feature_output[0] = price
        self.feature_output[1] = volume
        
        if self.buffer_index >= 5:
            # Vectorized operations on pre-allocated buffers
            start_idx = max(0, (self.buffer_index - 5) % self.MAX_LOOKBACK)
            self.feature_output[2] = np.mean(self.price_buffer[start_idx:start_idx+5])
        
        return self.feature_output  # Return view, not copy
```

#### Model Inference
```python
class HotPathModelInference:
    """Hot path model inference with pre-loaded models."""
    
    def __init__(self, model_path: str):
        # Load model ONCE at startup
        self.model = self._load_onnx_model(model_path)
        self.input_array = np.zeros((1, 5), dtype=np.float32)  # Pre-allocated
        
    def predict(self, features: np.ndarray) -> float:
        """Sub-millisecond inference with no allocations."""
        # Reuse pre-allocated input array
        self.input_array[0] = features
        
        # ONNX inference (pre-compiled, optimized)
        prediction = self.model.run(['output'], {'input': self.input_array})[0]
        
        return float(prediction[0, 1])  # Binary classification probability
```

#### Performance Contracts
```python
from ml.common.metrics_bootstrap import get_histogram

# Hot path metrics with sub-millisecond buckets
hot_path_latency = get_histogram(
    "ml_hot_path_latency_seconds",
    "Hot path operation latency",
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.05],  # 0.5ms to 50ms
    labels=["operation", "component"]
)

@hot_path_latency.time(labels={"operation": "feature_computation", "component": "technical_indicators"})
def compute_hot_path_features(self, bar: Bar) -> np.ndarray:
    """Automatically tracked hot path operation."""
    return self.feature_computer.compute_features(bar.close, bar.volume)
```

### Cold Path Implementation Standards

#### Batch Processing
```python
import pandas as pd
from pathlib import Path
import dask.dataframe as dd

class ColdPathFeatureEngineer:
    """Cold path feature engineering for training and backfill."""
    
    def compute_batch_features(self, data_path: Path) -> pd.DataFrame:
        """Heavy feature computation using all available resources."""
        
        # Load large datasets (OK in cold path)
        df = pd.read_parquet(data_path)  # Can be GBs
        
        # Complex feature engineering (expensive operations OK)
        features_df = pd.DataFrame(index=df.index)
        
        # Technical indicators with multiple timeframes  
        for period in [5, 10, 20, 50, 100, 200]:
            features_df[f'sma_{period}'] = df['close'].rolling(period).mean()
            features_df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features_df[f'std_{period}'] = df['close'].rolling(period).std()
        
        # Cross-sectional features (very expensive)
        features_df['percentile_rank'] = df.groupby('timestamp')['close'].rank(pct=True)
        
        # ML-based features (can use heavy models)
        features_df['anomaly_score'] = self._detect_anomalies(df)
        
        return features_df
    
    def backfill_features(self, start_date: str, end_date: str) -> None:
        """Large-scale feature backfill."""
        
        # Use distributed processing for large datasets
        ddf = dd.read_parquet(f"data/raw/{start_date}_{end_date}/*.parquet")
        
        # Compute features in parallel across partitions
        features_ddf = ddf.map_partitions(
            self._compute_partition_features, 
            meta=self._get_feature_schema()
        )
        
        # Write to feature store in batches
        features_ddf.to_parquet("features/batch_computed/", compression='snappy')
```

#### Model Training
```python
class ColdPathModelTrainer:
    """Cold path model training with full resource utilization."""
    
    def train_ensemble_model(self, training_data: Path) -> dict:
        """Resource-intensive model training."""
        
        # Load large training dataset
        df = pd.read_parquet(training_data)  # Can be 10GB+
        
        # Complex feature engineering
        features = self._engineer_training_features(df)
        targets = self._compute_targets(df)
        
        # Train multiple models (CPU/GPU intensive)
        models = {}
        
        # XGBoost with extensive hyperparameter search
        models['xgboost'] = self._train_xgboost(features, targets)
        
        # Neural network training (can take hours)
        models['neural_net'] = self._train_neural_network(features, targets)
        
        # Ensemble combination
        ensemble_model = self._create_ensemble(models, features, targets)
        
        # Extensive validation (cross-validation, walk-forward)
        validation_results = self._validate_model(ensemble_model, features, targets)
        
        return {
            'model': ensemble_model,
            'validation': validation_results,
            'feature_importance': self._analyze_features(ensemble_model),
        }
```

### Path Interaction Patterns

#### Data Handoff
```python
class HotColdPathBridge:
    """Manages data flow between hot and cold paths."""
    
    def __init__(self):
        self.hot_path_cache = {}  # Fast lookup for hot path
        self.cold_path_queue = []  # Batch processing queue
        
    def update_from_cold_path(self, model_id: str, model_artifacts: dict) -> None:
        """Update hot path with cold path results."""
        
        # Convert cold path model to hot path format
        hot_path_model = self._optimize_for_hot_path(model_artifacts['model'])
        
        # Update hot path cache atomically
        self.hot_path_cache[model_id] = {
            'model': hot_path_model,
            'metadata': model_artifacts['metadata'],
            'last_updated': time.time()
        }
        
        # Emit metrics for monitoring
        self._emit_model_update_metrics(model_id, model_artifacts)
    
    def queue_for_cold_path(self, data: dict) -> None:
        """Queue data for cold path processing."""
        self.cold_path_queue.append({
            'data': data,
            'queued_at': time.time(),
            'priority': self._calculate_priority(data)
        })
        
        # Trigger batch processing if queue is full
        if len(self.cold_path_queue) >= 1000:
            self._trigger_cold_path_processing()
```

### Performance Validation

#### Hot Path Testing
```python
import pytest
import time
import tracemalloc
import gc

class TestHotPathPerformance:
    """Rigorous performance testing for hot path."""
    
    def test_feature_computation_latency_sla(self):
        """Ensure feature computation meets latency SLA."""
        computer = HotPathFeatureComputer()
        
        # Warmup to eliminate JIT effects
        for _ in range(1000):
            computer.compute_features(1.1000, 10000.0)
        
        # Force garbage collection before test
        gc.collect()
        
        # Measure latency distribution
        latencies = []
        for _ in range(10000):
            start = time.perf_counter_ns()
            features = computer.compute_features(1.1000 + np.random.normal(0, 0.001), 10000.0)
            end = time.perf_counter_ns()
            latencies.append(end - start)
        
        # Validate SLA requirements
        p50_latency_us = np.percentile(latencies, 50) / 1000
        p99_latency_us = np.percentile(latencies, 99) / 1000
        
        assert p50_latency_us < 100, f"P50 latency {p50_latency_us}μs exceeds 100μs SLA"
        assert p99_latency_us < 1000, f"P99 latency {p99_latency_us}μs exceeds 1ms SLA"
    
    def test_zero_allocations_after_warmup(self):
        """Ensure no allocations in hot path after warmup."""
        computer = HotPathFeatureComputer()
        
        # Warmup
        for _ in range(100):
            computer.compute_features(1.1000, 10000.0)
        
        # Start memory tracking
        tracemalloc.start()
        gc.collect()  # Clean slate
        
        current, peak = tracemalloc.get_traced_memory()
        
        # Execute hot path operations
        for _ in range(1000):
            features = computer.compute_features(1.1000, 10000.0)
        
        current_after, peak_after = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Should have zero allocations
        allocations = current_after - current
        assert allocations == 0, f"Hot path allocated {allocations} bytes"
    
    def test_deterministic_performance(self):
        """Ensure performance is deterministic across runs."""
        computer = HotPathFeatureComputer()
        
        # Multiple performance runs
        run_latencies = []
        for run in range(10):
            run_times = []
            
            # Warmup each run
            for _ in range(100):
                computer.compute_features(1.1000, 10000.0)
            
            # Measure this run
            for _ in range(1000):
                start = time.perf_counter_ns()
                computer.compute_features(1.1000, 10000.0)
                end = time.perf_counter_ns()
                run_times.append(end - start)
            
            run_latencies.append(np.mean(run_times))
        
        # Performance should be consistent across runs
        performance_variance = np.var(run_latencies) / np.mean(run_latencies)
        assert performance_variance < 0.1, f"Performance variance too high: {performance_variance}"
```

#### Cold Path Testing
```python
class TestColdPathPerformance:
    """Validate cold path handles large-scale processing."""
    
    def test_large_dataset_processing(self):
        """Test processing large datasets efficiently."""
        
        # Generate large test dataset
        large_df = self._generate_test_data(rows=1_000_000)  # 1M rows
        
        engineer = ColdPathFeatureEngineer()
        
        start_time = time.time()
        features_df = engineer.compute_batch_features_from_dataframe(large_df)
        processing_time = time.time() - start_time
        
        # Should process within reasonable time
        rows_per_second = len(large_df) / processing_time
        assert rows_per_second > 1000, f"Processing rate too slow: {rows_per_second} rows/sec"
        
        # Should produce expected number of features
        assert len(features_df.columns) >= 50, "Insufficient features generated"
    
    def test_memory_efficient_processing(self):
        """Test cold path processes data without excessive memory usage."""
        
        # Monitor memory during processing
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engineer = ColdPathFeatureEngineer()
        
        # Process multiple chunks
        for chunk_id in range(10):
            chunk_data = self._generate_test_data(rows=100_000)
            features = engineer.compute_batch_features_from_dataframe(chunk_data)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be bounded
            assert memory_growth < 1024, f"Memory growth too high: {memory_growth} MB"
```

## Migration Strategy

### Phase 1: Identification and Classification
- Audit existing codebase to identify hot vs cold path operations
- Create performance profiles for all ML operations  
- Document current performance characteristics

### Phase 2: Hot Path Optimization
- Implement zero-allocation patterns for hot path operations
- Pre-load models and pre-allocate arrays
- Add performance SLA testing

### Phase 3: Cold Path Restructuring  
- Move heavy operations to dedicated cold path modules
- Implement batch processing patterns
- Add resource utilization monitoring

### Phase 4: Bridge Implementation
- Create data handoff mechanisms between paths
- Implement model deployment from cold to hot path
- Add coordination and monitoring systems

### Phase 5: Validation and Enforcement
- Performance testing in CI/CD pipeline
- Static analysis rules to prevent hot/cold path violations
- Documentation and training for development teams

## Alternatives Considered

### Alternative 1: Unified Performance Optimization
**Rejected** - Cannot simultaneously optimize for sub-millisecond latency and batch throughput

### Alternative 2: Microservice Separation  
**Rejected** - Adds network latency and deployment complexity for hot path

### Alternative 3: Dynamic Performance Modes
**Rejected** - Runtime switching adds complexity and potential performance degradation

### Alternative 4: JIT Compilation Approach
**Rejected** - Warmup time and unpredictable performance not suitable for trading

## Related ADRs
- ADR-001: 4-Store + 4-Registry Mandatory Pattern
- ADR-002: Protocol-First Interface Design  
- ADR-004: Progressive Fallback Implementation

## References
- [Performance Targets Documentation](../../monitoring/performance_targets.md)
- [Hot Path Implementation Examples](../../actors/signal.py)
- [Cold Path Implementation Examples](../../training/)