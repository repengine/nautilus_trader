#!/usr/bin/env python3
"""
Performance Assessment Runner.

Runs key performance tests and displays results in a concise format.

"""

import sys
import time
from pathlib import Path


from typing import Callable, Any, cast


def run_quick_test() -> int:
    """
    Run the quick performance test.
    """
    print("Running quick ML performance test...")
    try:
        from quick_performance_test import main as quick_main
        quick = cast(Callable[[], int], quick_main)
        return quick()
    except Exception as e:
        print(f"Quick test failed: {e}")
        return 1


def run_existing_benchmark() -> int:
    """
    Run the existing benchmark test.
    """
    print("Running existing benchmark test...")
    try:
        from benchmark_hot_path import benchmark_feature_calculation

        result = benchmark_feature_calculation(5000)

        print("Existing Benchmark Results:")
        print(f"  Average latency: {result['avg_latency_us']:.1f}μs")
        print(f"  Memory per call: {result['bytes_per_call']:.1f} bytes")
        print(f"  Passes 500μs:    {'✓' if result['avg_latency_us'] * 2 < 500 else '✗'}")
        print(f"  Passes zero alloc: {'✓' if result['bytes_per_call'] < 10 else '✗'}")

        return 0 if result["avg_latency_us"] * 2 < 500 else 1
    except Exception as e:
        print(f"Benchmark test failed: {e}")
        return 1


def main() -> int:
    """
    Run performance assessment.
    """
    print("=" * 60)
    print("ML SYSTEM PERFORMANCE ASSESSMENT")
    print("=" * 60)
    print()

    start_time = time.time()

    # Run quick test
    quick_result = run_quick_test()
    print()

    # Run existing benchmark
    bench_result = run_existing_benchmark()
    print()

    elapsed = time.time() - start_time
    print(f"Assessment completed in {elapsed:.1f} seconds")
    print()

    # Summary
    print("SUMMARY:")
    print("-" * 8)
    if quick_result == 0 and bench_result == 0:
        print("✅ System meets core performance requirements")
        print("📊 See BRUTAL_PERFORMANCE_ASSESSMENT.md for detailed analysis")
        return 0
    else:
        print("⚠️  System has performance issues")
        print("📊 See BRUTAL_PERFORMANCE_ASSESSMENT.md for detailed analysis")
        return 1


if __name__ == "__main__":
    sys.exit(main())
