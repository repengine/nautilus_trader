# ML Core Utilities

This module contains core utilities and data structures for ML components in Nautilus Trader.

## Components

### cache.py
High-performance, zero-allocation data structures for hot path operations:

- `LockFreeRingBuffer`: Lock-free ring buffer for maintaining rolling windows of data
- `PreAllocatedFeatureCache`: Pre-allocated cache for feature vectors with zero-copy access
- `ReservoirSampler`: Reservoir sampling for maintaining representative samples

## Design Principles

1. **Zero Allocation**: All data structures pre-allocate memory and reuse buffers
2. **Hot Path Optimized**: Designed for < 5ms P99 latency in production
3. **Memory Views**: Return views instead of copies where possible
4. **Thread Safe**: Lock-free designs for concurrent access

## Usage

```python
from ml.core.cache import LockFreeRingBuffer, PreAllocatedFeatureCache

# Create a ring buffer for prediction history
buffer = LockFreeRingBuffer(size=1000, dtype=np.float32)
buffer.append(0.95)  # Zero allocation O(1) operation

# Pre-allocated feature cache
cache = PreAllocatedFeatureCache(n_features=50, history_size=1000)
features = cache.get_current_buffer()  # Returns pre-allocated buffer
# ... compute features in-place ...
cache.store_current_features()  # Zero-copy storage
```

## Performance Requirements

- Feature computation: < 500μs
- Buffer operations: < 10μs
- Memory stable over 24h continuous operation
- Zero allocations in hot path

```
