#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Benchmark to verify hot path performance improvements from zero-allocation fixes.
"""

import time
import tracemalloc

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager


def benchmark_feature_calculation(n_iterations=1000):
    """
    Benchmark feature calculation in hot path.
    """
    config = FeatureConfig()
    engineer = FeatureEngineer(config)
    indicator_mgr = IndicatorManager(config)

    # Initialize price history
    for i in range(50):
        indicator_mgr.price_history["closes"].append(100.0 + i * 0.1)
        indicator_mgr.price_history["volumes"].append(1000000.0)
        indicator_mgr.price_history["highs"].append(101.0 + i * 0.1)
        indicator_mgr.price_history["lows"].append(99.0 + i * 0.1)

    # Warm up
    for i in range(100):
        current_bar = {
            "open": 100.0 + i * 0.01,
            "high": 101.0 + i * 0.01,
            "low": 99.0 + i * 0.01,
            "close": 100.5 + i * 0.01,
            "volume": 1000000.0,
        }
        _ = engineer.calculate_features_online(current_bar, indicator_mgr)

    # Start memory tracking
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    # Benchmark timing
    start = time.perf_counter()

    for i in range(n_iterations):
        current_bar = {
            "open": 100.0 + i * 0.01,
            "high": 101.0 + i * 0.01,
            "low": 99.0 + i * 0.01,
            "close": 100.5 + i * 0.01,
            "volume": 1000000.0 + i * 100,
        }

        _ = engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_mgr,
            scaler=None,
        )

        # Update history (simulate real usage)
        indicator_mgr.price_history["closes"].append(current_bar["close"])
        indicator_mgr.price_history["volumes"].append(current_bar["volume"])
        indicator_mgr.price_history["highs"].append(current_bar["high"])
        indicator_mgr.price_history["lows"].append(current_bar["low"])

        # Keep history bounded
        for key in indicator_mgr.price_history:
            if len(indicator_mgr.price_history[key]) > 100:
                indicator_mgr.price_history[key] = indicator_mgr.price_history[key][-100:]

    elapsed = time.perf_counter() - start

    # Take memory snapshot
    snapshot2 = tracemalloc.take_snapshot()

    # Calculate memory difference
    top_stats = snapshot2.compare_to(snapshot1, "lineno")

    # Calculate allocations
    total_allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

    tracemalloc.stop()

    # Calculate metrics
    avg_latency_us = (elapsed / n_iterations) * 1_000_000
    allocations_per_call = total_allocated / n_iterations

    return {
        "n_iterations": n_iterations,
        "total_time_s": elapsed,
        "avg_latency_us": avg_latency_us,
        "total_allocated_bytes": total_allocated,
        "bytes_per_call": allocations_per_call,
    }


def main():
    print("Benchmarking hot path performance after zero-allocation fixes...")
    print("=" * 60)

    # Run benchmark
    results = benchmark_feature_calculation(n_iterations=10000)

    print(f"Iterations:           {results['n_iterations']:,}")
    print(f"Total time:           {results['total_time_s']:.3f} seconds")
    print(f"Avg latency:          {results['avg_latency_us']:.1f} μs")
    print(f"Memory allocated:     {results['total_allocated_bytes']:,} bytes total")
    print(f"Allocation per call:  {results['bytes_per_call']:.1f} bytes")
    print()

    # Check performance requirements
    if results["avg_latency_us"] < 500:
        print(" PASS: Feature computation < 500μs requirement")
    else:
        print(
            f" FAIL: Feature computation {results['avg_latency_us']:.1f}μs > 500μs requirement",
        )

    if results["bytes_per_call"] < 100:  # Allow small allocations for history management
        print(" PASS: Near-zero allocation achieved")
    else:
        print(f"  WARNING: {results['bytes_per_call']:.1f} bytes allocated per call")

    print()
    print("Performance summary:")
    print(f"  • Latency: {results['avg_latency_us']:.1f}μs per feature calculation")
    print(f"  • Memory:  {results['bytes_per_call']:.1f} bytes per call")
    print(f"  • Throughput: {1_000_000 / results['avg_latency_us']:.0f} calculations/second")


if __name__ == "__main__":
    main()
