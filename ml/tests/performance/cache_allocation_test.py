#!/usr/bin/env python3
"""
Test cache allocation behavior to understand zero allocation claims.
"""

import gc
import logging
import tracemalloc
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any, cast

import numpy as np

from ml.core.cache import LockFreeRingBuffer
from ml.core.cache import PreAllocatedFeatureCache


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def track_allocations() -> Iterator[None]:
    """
    Track memory allocations precisely.
    """
    # Clear any existing traces
    tracemalloc.stop()
    gc.collect()

    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    yield

    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, "lineno")

    # Calculate total allocations
    total_allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

    # Show top allocations
    print("\nTop memory allocations:")
    for i, stat in enumerate(top_stats[:5]):
        if stat.size_diff > 0:
            print(f"  {i+1}. {stat.traceback.format()[-1].strip()}: +{stat.size_diff:,} bytes")

    print(f"\nTotal allocated: {total_allocated:,} bytes")
    tracemalloc.stop()

    cast(Any, track_allocations).total_bytes = total_allocated


def test_ring_buffer_allocations() -> dict[str, Any]:
    """
    Test ring buffer allocation behavior.
    """
    print("=" * 60)
    print("RING BUFFER ALLOCATION TEST")
    print("=" * 60)

    # Test 1: Buffer creation (expected to allocate)
    print("\n1. Buffer creation:")
    with track_allocations():
        buffer = LockFreeRingBuffer(1000)
    creation_bytes = int(getattr(track_allocations, "total_bytes", 0))
    print(f"Buffer creation allocated: {creation_bytes:,} bytes (expected)")

    # Test 2: Append operations (should be zero allocation)
    print("\n2. Append operations (hot path):")
    with track_allocations():
        for i in range(1000):
            buffer.append(float(i))
    append_bytes = int(getattr(track_allocations, "total_bytes", 0))
    print(f"1000 appends allocated: {append_bytes:,} bytes")

    # Test 3: Get operations (critical for zero allocation claim)
    print("\n3. Get operations (hot path):")
    with track_allocations():
        for _ in range(1000):
            result = buffer.get_last(10)
            _ = result[0]  # Access the data
    get_bytes = int(getattr(track_allocations, "total_bytes", 0))
    print(f"1000 get_last calls allocated: {get_bytes:,} bytes")

    # Test 4: Views vs copies
    print("\n4. Testing view behavior:")
    buffer.append(1.0)
    buffer.append(2.0)
    buffer.append(3.0)

    with track_allocations():
        view1 = buffer.get_last(3)
        view2 = buffer.get_last(3)
        # Check if they share memory
        shares_memory = np.shares_memory(view1, buffer._buffer)
    view_bytes = int(getattr(track_allocations, "total_bytes", 0))

    print(f"View operations allocated: {view_bytes:,} bytes")
    print(f"Views share memory with buffer: {shares_memory}")

    return {
        "append_zero_alloc": append_bytes == 0,
        "get_zero_alloc": get_bytes == 0,
        "view_zero_alloc": view_bytes == 0,
        "shares_memory": shares_memory,
    }


def test_feature_cache_allocations() -> dict[str, Any]:
    """
    Test feature cache allocation behavior.
    """
    print("\n" + "=" * 60)
    print("FEATURE CACHE ALLOCATION TEST")
    print("=" * 60)

    # Test 1: Cache creation
    print("\n1. Cache creation:")
    with track_allocations():
        cache = PreAllocatedFeatureCache(n_features=20, history_size=1000)
    creation_bytes = int(getattr(track_allocations, "total_bytes", 0))
    print(f"Cache creation allocated: {creation_bytes:,} bytes (expected)")

    # Test 2: Buffer access (should be zero allocation)
    print("\n2. Buffer access operations:")
    with track_allocations():
        for _ in range(1000):
            buffer = cache.get_current_buffer()
            buffer[:] = np.random.random(20).astype(np.float32)
    buffer_access_bytes = int(getattr(track_allocations, "total_bytes", 0))
    print(f"1000 buffer accesses allocated: {buffer_access_bytes:,} bytes")

    # Test 3: Store operations
    print("\n3. Store operations:")
    with track_allocations():
        for _ in range(1000):
            cache.store_current_features()
    store_bytes = int(getattr(track_allocations, "total_bytes", 0))
    print(f"1000 store operations allocated: {store_bytes:,} bytes")

    # Test 4: History access
    print("\n4. History access:")
    with track_allocations():
        for _ in range(1000):
            history = cache.get_feature_history(10)
            _ = history[0, 0]  # Access data
    history_bytes = int(getattr(track_allocations, "total_bytes", 0))
    print(f"1000 history accesses allocated: {history_bytes:,} bytes")

    return {
        "buffer_access_zero_alloc": buffer_access_bytes == 0,
        "store_zero_alloc": store_bytes == 0,
        "history_zero_alloc": history_bytes == 0,
    }


def test_real_world_usage_pattern() -> dict[str, Any]:
    """
    Test realistic usage pattern.
    """
    print("\n" + "=" * 60)
    print("REAL WORLD USAGE PATTERN TEST")
    print("=" * 60)

    # Setup
    buffer = LockFreeRingBuffer(100)
    cache = PreAllocatedFeatureCache(n_features=10, history_size=100)

    # Fill with initial data
    for i in range(50):
        buffer.append(float(i))
        cache.get_current_buffer()[:] = np.random.random(10).astype(np.float32)
        cache.store_current_features()

    print("\nSimulating realistic ML inference loop:")

    with track_allocations():
        for i in range(1000):
            # Typical hot path operations

            # 1. Add new price data
            buffer.append(float(i + 50))

            # 2. Get recent price history for features
            recent_prices = buffer.get_last(20)

            # 3. Compute features into pre-allocated buffer
            feature_buffer = cache.get_current_buffer()
            feature_buffer[0] = recent_prices[-1]  # Current price
            feature_buffer[1] = np.mean(recent_prices)  # Moving average
            feature_buffer[2] = np.std(recent_prices)  # Volatility
            # ... more feature computations

            # 4. Store features
            cache.store_current_features()

            # 5. Get feature history for model input
            feature_history = cache.get_feature_history(5)

            # 6. Simulate model inference (access data)
            _ = feature_history.mean()

    real_world_bytes = int(getattr(track_allocations, "total_bytes", 0))
    print(f"1000 realistic inference loops allocated: {real_world_bytes:,} bytes")

    return {
        "real_world_zero_alloc": real_world_bytes == 0,
        "real_world_low_alloc": real_world_bytes < 1000,  # Less than 1KB
    }


def main() -> int:
    """
    Run cache allocation tests.
    """
    print("TESTING ML CACHE ZERO ALLOCATION CLAIMS")
    print("=" * 80)
    print("This test examines the actual memory allocation behavior of ML caches")
    print("to verify the 'zero allocation hot path' claims.")

    try:
        # Run tests
        ring_results = test_ring_buffer_allocations()
        cache_results = test_feature_cache_allocations()
        real_world_results = test_real_world_usage_pattern()

        # Summary
        print("\n" + "=" * 60)
        print("ZERO ALLOCATION CLAIMS ASSESSMENT")
        print("=" * 60)

        print("\nRing Buffer:")
        print(
            f"  Append zero alloc:  {'✓ PASS' if ring_results['append_zero_alloc'] else '✗ FAIL'}",
        )
        print(f"  Get zero alloc:     {'✓ PASS' if ring_results['get_zero_alloc'] else '✗ FAIL'}")
        print(f"  Uses memory views:  {'✓ YES' if ring_results['shares_memory'] else '✗ NO'}")

        print("\nFeature Cache:")
        print(
            f"  Buffer access:      {'✓ PASS' if cache_results['buffer_access_zero_alloc'] else '✗ FAIL'}",
        )
        print(
            f"  Store operations:   {'✓ PASS' if cache_results['store_zero_alloc'] else '✗ FAIL'}",
        )
        print(
            f"  History access:     {'✓ PASS' if cache_results['history_zero_alloc'] else '✗ FAIL'}",
        )

        print("\nReal World Usage:")
        print(
            f"  Zero allocation:    {'✓ PASS' if real_world_results['real_world_zero_alloc'] else '✗ FAIL'}",
        )
        print(
            f"  Low allocation:     {'✓ PASS' if real_world_results['real_world_low_alloc'] else '✗ FAIL'}",
        )

        # Overall verdict
        all_zero = (
            ring_results["append_zero_alloc"]
            and ring_results["get_zero_alloc"]
            and cache_results["buffer_access_zero_alloc"]
            and cache_results["store_zero_alloc"]
            and cache_results["history_zero_alloc"]
        )

        print(
            f"\nOVERALL ZERO ALLOCATION CLAIM: {'✓ VALIDATED' if all_zero else '✗ PARTIALLY FAILS'}",
        )

        if not all_zero:
            print("\nNote: Some allocations may be unavoidable due to Python overhead,")
            print("NumPy temporary arrays, or wraparound handling. The key metric is")
            print("whether allocations are minimal and bounded.")

        return 0 if real_world_results["real_world_low_alloc"] else 1

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
