#!/usr/bin/env python3
"""
Quick ML Performance Test - Fast assessment of performance claims
"""

import gc
import logging
import statistics
import time
import tracemalloc
from contextlib import contextmanager

import numpy as np

# Direct imports to avoid complex dependencies
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import OptimizationLevel
from ml.config.actors import OptimizationConfig
from ml.features.config import FeatureConfig
from ml.features.indicators import IndicatorManager
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def measure_memory():
    """
    Simple memory measurement.
    """
    tracemalloc.start()
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()

    yield

    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
    tracemalloc.stop()
    measure_memory.allocated_bytes = allocated


def quick_feature_test(n_iterations: int = 1000) -> dict:
    """
    Quick feature computation test.
    """
    logger.info(f"Quick feature test ({n_iterations} iterations)...")

    config = FeatureConfig()
    engineer = FeatureEngineer(config)
    indicator_mgr = IndicatorManager(config)

    # Simple initialization
    for i in range(20):  # Reduced from 50
        price = 1.1000 + i * 0.0001
        indicator_mgr.price_history["closes"].append(price)
        indicator_mgr.price_history["volumes"].append(1000000)
        indicator_mgr.price_history["highs"].append(price + 0.0001)
        indicator_mgr.price_history["lows"].append(price - 0.0001)

    # Warm up (reduced)
    for _ in range(10):
        current_bar = {
            "open": 1.1000,
            "high": 1.1001,
            "low": 1.0999,
            "close": 1.1000,
            "volume": 1000000,
        }
        engineer.calculate_features_online(current_bar, indicator_mgr)

    # Performance test
    latencies = []

    with measure_memory():
        for i in range(n_iterations):
            current_bar = {
                "open": 1.1000,
                "high": 1.1001,
                "low": 1.0999,
                "close": 1.1000 + i * 0.00001,
                "volume": 1000000,
            }

            start = time.perf_counter()
            features = engineer.calculate_features_online(current_bar, indicator_mgr)
            end = time.perf_counter()

            latency_us = (end - start) * 1_000_000
            latencies.append(latency_us)

    allocated = getattr(measure_memory, "allocated_bytes", 0)

    return {
        "min_latency_us": min(latencies),
        "mean_latency_us": statistics.mean(latencies),
        "p99_latency_us": np.percentile(latencies, 99),
        "max_latency_us": max(latencies),
        "allocated_bytes": allocated,
        "passes_500us": np.percentile(latencies, 99) < 500,
        "passes_5ms": np.percentile(latencies, 99) < 5000,
        "passes_zero_alloc": allocated < 1000,
    }


def main():
    """
    Run quick performance test.
    """
    print("=" * 60)
    print("QUICK ML PERFORMANCE TEST")
    print("=" * 60)
    print()

    try:
        # Test with different iteration counts
        results = {}

        for n in [100, 500, 1000]:
            results[n] = quick_feature_test(n)

            print(f"Test with {n} iterations:")
            r = results[n]
            print(f"  Min latency:    {r['min_latency_us']:6.1f}μs")
            print(f"  Mean latency:   {r['mean_latency_us']:6.1f}μs")
            print(f"  P99 latency:    {r['p99_latency_us']:6.1f}μs ⭐")
            print(f"  Max latency:    {r['max_latency_us']:6.1f}μs")
            print(f"  Memory alloc:   {r['allocated_bytes']:,} bytes")
            print(f"  <500μs claim:   {'✓ PASS' if r['passes_500us'] else '✗ FAIL'}")
            print(f"  Zero alloc:     {'✓ PASS' if r['passes_zero_alloc'] else '✗ FAIL'}")
            print()

        # Overall assessment
        print("ASSESSMENT:")
        print("-" * 12)

        latest_result = results[1000]

        if latest_result["passes_500us"]:
            print("✅ Feature computation <500μs: PASS")
        else:
            print(f"❌ Feature computation <500μs: FAIL ({latest_result['p99_latency_us']:.1f}μs)")

        if latest_result["passes_zero_alloc"]:
            print("✅ Zero allocation claim: PASS")
        else:
            print(f"❌ Zero allocation claim: FAIL ({latest_result['allocated_bytes']:,} bytes)")

        print()

        if latest_result["passes_500us"] and latest_result["passes_zero_alloc"]:
            print("🎉 OVERALL: ML system meets core performance claims")
            return 0
        else:
            print("⚠️ OVERALL: ML system fails to meet some performance claims")
            return 1

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
