#!/usr/bin/env python
"""
Script to run ML hot path performance benchmarks and generate a report.

This script runs all performance benchmarks for ML hot path components
and generates a comprehensive report showing whether the system meets
production latency requirements.

Usage:
    python ml/tests/performance/run_benchmarks.py

"""

import subprocess
import sys
from pathlib import Path


from typing import List


def run_benchmarks() -> int:
    """
    Run all ML hot path benchmarks and generate report.
    """
    print("=" * 80)
    print("ML HOT PATH PERFORMANCE BENCHMARK REPORT")
    print("=" * 80)
    print()
    print("Performance Requirements:")
    print("  • P99 feature computation: <500μs")
    print("  • P99 model inference: <2ms")
    print("  • P99 end-to-end signal: <5ms")
    print("  • Zero allocations in hot path (<500 bytes/call)")
    print("  • Memory stable over 24h operation")
    print()
    print("Running benchmarks...")
    print()

    # Path to the benchmark file
    benchmark_file = Path(__file__).parent / "test_ml_hot_path_benchmarks.py"

    # Run benchmarks
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(benchmark_file),
        "-v",
        "--benchmark-only",
        "--benchmark-columns=min,max,mean,stddev,median,iqr,outliers,ops",
        "--benchmark-sort=name",
        "--benchmark-group-by=class",
        "--benchmark-warmup=on",
        "--benchmark-warmup-iterations=10",
        "--benchmark-max-time=2",  # Limit time per benchmark
        "-q",  # Quiet mode for cleaner output
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        print(result.stdout)

        if result.returncode != 0:
            print("\n❌ BENCHMARK FAILURES DETECTED:")
            print(result.stderr)
            print("\nSome performance requirements are not met!")
            return 1
        else:
            print("\n✅ All performance requirements satisfied!")
            print()
            print("Summary:")

            # Parse output to extract key metrics
            lines = result.stdout.split("\n")
            for line in lines:
                if "test_feature_computation_p99_latency" in line:
                    parts = line.split()
                    if len(parts) > 2:
                        max_latency = parts[2]  # Max column
                        print(f"  • Feature computation P99: {max_latency} (Target: <500μs)")
                elif "test_onnx_inference_p99_latency" in line:
                    parts = line.split()
                    if len(parts) > 2:
                        max_latency = parts[2]
                        print(f"  • ONNX inference P99: {max_latency} (Target: <2ms)")
                elif "test_signal_generation_e2e_latency" in line:
                    parts = line.split()
                    if len(parts) > 2:
                        max_latency = parts[2]
                        print(f"  • End-to-end signal P99: {max_latency} (Target: <5ms)")

            return 0

    except Exception as e:
        print(f"\n❌ Error running benchmarks: {e}")
        return 1


def run_memory_tests() -> int:
    """
    Run memory allocation and leak detection tests.
    """
    print("\n" + "=" * 80)
    print("MEMORY ALLOCATION TESTS")
    print("=" * 80)
    print()

    benchmark_file = Path(__file__).parent / "test_ml_hot_path_benchmarks.py"

    # Run memory tests
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(benchmark_file),
        "-k",
        "memory",
        "-v",
        "-q",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if "PASSED" in result.stdout:
            print("✅ Memory allocation tests passed")
            print("  • Feature computation allocates <500 bytes per call")
            print("  • No memory leaks detected over extended operation")
        else:
            print("❌ Memory allocation tests failed")
            print(result.stdout)

        return result.returncode

    except Exception as e:
        print(f"❌ Error running memory tests: {e}")
        return 1


def main() -> int:
    """
    Main entry point.
    """
    print("\nStarting ML Hot Path Performance Validation\n")

    # Run performance benchmarks
    benchmark_result = run_benchmarks()

    # Run memory tests
    memory_result = run_memory_tests()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    if benchmark_result == 0 and memory_result == 0:
        print("\n🎉 SUCCESS: All ML hot path components meet production requirements!")
        print("\nThe system is ready for production deployment with:")
        print("  • Sub-millisecond feature computation")
        print("  • Low-latency model inference")
        print("  • Minimal memory allocation")
        print("  • Stable memory usage")
        return 0
    else:
        print("\n⚠️  WARNING: Some requirements not met. Review the failures above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
