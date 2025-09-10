#!/usr/bin/env python3
"""
Test hot path performance for the feature parity fix.
"""

import sys

sys.path.insert(0, "/home/nate/projects/nautilus_trader")

import time
import numpy as np
from ml.features.engineering import FeatureConfig, FeatureEngineer


def test_hot_path_performance():
    """
    Test that online feature computation maintains <5ms P99 latency requirement.
    """
    print("=== Hot Path Performance Test ===\n")

    configs_to_test = [
        ("Default (L1)", FeatureConfig()),
        ("Microstructure (L1+L2)", FeatureConfig(include_microstructure=True)),
        ("Full (L1+L2+L3)", FeatureConfig(include_microstructure=True, include_trade_flow=True)),
    ]

    all_passed = True

    for config_name, config in configs_to_test:
        print(f"Testing {config_name}:")
        passed = test_config_performance(config)
        all_passed = all_passed and passed
        print()

    print("=" * 50)
    if all_passed:
        print("🎉 ALL PERFORMANCE TESTS PASSED!")
        print("Hot path latency < 5ms P99 maintained.")
    else:
        print("❌ PERFORMANCE REGRESSION DETECTED!")
        print("Hot path latency exceeds 5ms requirement.")
        sys.exit(1)


def test_config_performance(config):
    """
    Test performance for a specific configuration.
    """
    engineer = FeatureEngineer(config)

    # Generate test data
    rng = np.random.default_rng(42)
    base_price = 1.1000

    # Warm up the engineer with 50 bars
    print("  Warming up indicators...")
    for i in range(50):
        price = base_price + rng.uniform(-0.001, 0.001)
        high = price + abs(rng.uniform(0, 0.002))
        low = price - abs(rng.uniform(0, 0.002))
        volume = 1000000 + rng.integers(-100000, 100000)

        engineer.calculate_features_online(
            close_price=price,
            high_price=high,
            low_price=low,
            volume=float(volume),
        )

    # Performance test with 1000 iterations
    print("  Running performance test...")
    latencies = []
    n_iterations = 1000

    for i in range(n_iterations):
        # Generate realistic bar data
        price = base_price + rng.uniform(-0.001, 0.001)
        high = price + abs(rng.uniform(0, 0.002))
        low = price - abs(rng.uniform(0, 0.002))
        volume = 1000000 + rng.integers(-100000, 100000)

        # Measure latency
        start_time = time.perf_counter()
        features = engineer.calculate_features_online(
            close_price=price,
            high_price=high,
            low_price=low,
            volume=float(volume),
        )
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # Calculate statistics
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean_latency = np.mean(latencies)
    max_latency = np.max(latencies)

    print(f"  Performance results ({n_iterations} iterations):")
    print(f"    Mean:     {mean_latency:.3f}ms")
    print(f"    P50:      {p50:.3f}ms")
    print(f"    P95:      {p95:.3f}ms")
    print(f"    P99:      {p99:.3f}ms")
    print(f"    Max:      {max_latency:.3f}ms")
    print(f"    Features: {len(features)}")

    # Check performance requirements
    requirement_p99 = 5.0  # 5ms P99 requirement

    if p99 < requirement_p99:
        print(f"  ✅ Performance PASSED (P99 {p99:.3f}ms < {requirement_p99:.1f}ms)")
        return True
    else:
        print(f"  ❌ Performance FAILED (P99 {p99:.3f}ms >= {requirement_p99:.1f}ms)")
        return False


if __name__ == "__main__":
    test_hot_path_performance()
