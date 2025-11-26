#!/usr/bin/env python3
"""
Working feature engineering test script to validate claims.
"""

import datetime
import gc
import logging
import sys
import time
import tracemalloc

import numpy as np
import polars as pl

sys.path.insert(0, ".")

from ml.features import FeatureConfig, FeatureEngineer, IndicatorManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_data(n_bars=200):
    """
    Create test market data.
    """
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_bars)
    prices = [base_price]

    for ret in returns:
        prices.append(prices[-1] * (1 + ret))

    close_prices = np.array(prices[1:])
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    volumes = np.random.lognormal(15, 0.5, n_bars)

    timestamps = [
        datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i) for i in range(n_bars)
    ]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        },
    )


def run_comprehensive_test():
    """
    Run comprehensive performance and functionality tests.
    """
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE FEATURE ENGINEERING TEST")
    logger.info("=" * 60)

    # Test configuration
    config = FeatureConfig(
        return_periods=[1, 5, 10],
        momentum_periods=[5, 10],
        rsi_period=14,
        bb_period=20,
        volume_ma_periods=[5, 10],
    )

    engineer = FeatureEngineer(config)
    logger.info(f"Engineer created with {engineer.n_features} features")
    logger.info(f"Pre-allocated buffer shape: {engineer.feature_buffer.shape}")

    # Create test data
    df = create_test_data(500)
    logger.info(f"Created test data: {df.shape}")

    # Test 1: Basic functionality
    logger.info("\n1. Testing Basic Functionality")
    logger.info("-" * 40)

    try:
        start = time.perf_counter()
        features_df, scaler = engineer.calculate_features_batch(df, fit_scaler=True)
        batch_time = (time.perf_counter() - start) * 1000

        logger.info(f"✓ Batch processing: {batch_time:.2f}ms")
        logger.info(f"✓ Features generated: {features_df.shape}")
        logger.info(f"✓ Scaler fitted: {scaler is not None}")

        basic_passed = True

    except Exception as e:
        logger.error(f"✗ Basic functionality failed: {e}")
        basic_passed = False
        return

    # Test 2: Hot/Cold Path Separation & Performance
    logger.info("\n2. Testing Hot/Cold Path Performance")
    logger.info("-" * 40)

    indicator_mgr = IndicatorManager(config)

    # Warmup with first 50 bars
    logger.info("Warming up indicators...")
    for i in range(50):
        row = df.row(i, named=True)
        indicator_mgr.update_from_values(
            close=float(row["close"]),
            high=float(row["high"]),
            low=float(row["low"]),
            volume=float(row["volume"]),
        )

    # Performance test on hot path
    logger.info("Testing hot path performance...")
    online_latencies = []
    online_features = []

    for i in range(50, 350):  # Test 300 bars
        row = df.row(i, named=True)
        current_bar = {
            "close": float(row["close"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "volume": float(row["volume"]),
        }

        # Time the hot path
        start = time.perf_counter_ns()
        features = engineer.calculate_features_online(current_bar, indicator_mgr, scaler)
        end = time.perf_counter_ns()

        latency_ms = (end - start) / 1_000_000
        online_latencies.append(latency_ms)
        online_features.append(features.copy())

    # Performance statistics
    mean_latency = np.mean(online_latencies)
    p99_latency = np.percentile(online_latencies, 99)
    p95_latency = np.percentile(online_latencies, 95)
    max_latency = np.max(online_latencies)

    logger.info(f"✓ Online operations: {len(online_latencies)}")
    logger.info(f"✓ Mean latency: {mean_latency:.3f}ms")
    logger.info(f"✓ P95 latency: {p95_latency:.3f}ms")
    logger.info(f"✓ P99 latency: {p99_latency:.3f}ms")
    logger.info(f"✓ Max latency: {max_latency:.3f}ms")

    meets_5ms_sla = p99_latency < 5.0
    logger.info(f"✓ Meets <5ms P99 SLA: {meets_5ms_sla} ({'✓' if meets_5ms_sla else '✗'})")

    # Test 3: Mathematical Parity
    logger.info("\n3. Testing Mathematical Parity")
    logger.info("-" * 40)

    try:
        # Compare batch vs online features for overlapping period
        batch_subset = features_df[50 : 50 + len(online_features)]
        batch_values = batch_subset.select(pl.exclude("timestamp")).to_numpy()
        online_values = np.array(online_features)

        if batch_values.shape == online_values.shape:
            differences = np.abs(batch_values - online_values)
            max_diff = np.max(differences)
            mean_diff = np.mean(differences)

            logger.info(f"✓ Shapes match: {batch_values.shape}")
            logger.info(f"✓ Max difference: {max_diff:.2e}")
            logger.info(f"✓ Mean difference: {mean_diff:.2e}")

            parity_threshold = 1e-6  # Reasonable floating point tolerance
            parity_passed = max_diff < parity_threshold
            logger.info(
                f"✓ Parity (< {parity_threshold:.0e}): {parity_passed} ({'✓' if parity_passed else '✗'})",
            )

        else:
            logger.error(
                f"✗ Shape mismatch: batch {batch_values.shape} vs online {online_values.shape}",
            )
            parity_passed = False

    except Exception as e:
        logger.error(f"✗ Parity test failed: {e}")
        parity_passed = False

    # Test 4: Memory Allocation
    logger.info("\n4. Testing Memory Allocation")
    logger.info("-" * 40)

    # Reset engineer state
    engineer.reset()
    indicator_mgr.reset()

    # Warmup
    for i in range(30):
        row = df.row(i, named=True)
        indicator_mgr.update_from_values(
            close=float(row["close"]),
            high=float(row["high"]),
            low=float(row["low"]),
            volume=float(row["volume"]),
        )

    # Check buffer properties
    buffer_exists = hasattr(engineer, "feature_buffer") and isinstance(
        engineer.feature_buffer,
        np.ndarray,
    )
    buffer_id_before = id(engineer.feature_buffer) if buffer_exists else None

    logger.info(f"✓ Pre-allocated buffer exists: {buffer_exists}")
    if buffer_exists:
        logger.info(f"✓ Buffer shape: {engineer.feature_buffer.shape}")
        logger.info(f"✓ Buffer dtype: {engineer.feature_buffer.dtype}")

    # Memory tracking
    tracemalloc.start()
    gc.collect()
    memory_before = tracemalloc.get_traced_memory()[0]

    # Hot path operations
    for i in range(30, 130):  # 100 operations
        row = df.row(i, named=True)
        current_bar = {
            "close": float(row["close"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "volume": float(row["volume"]),
        }
        features = engineer.calculate_features_online(current_bar, indicator_mgr)

    memory_after = tracemalloc.get_traced_memory()[0]
    memory_growth = memory_after - memory_before
    tracemalloc.stop()

    # Check buffer reuse
    buffer_id_after = id(engineer.feature_buffer) if buffer_exists else None
    buffer_reused = buffer_id_before == buffer_id_after if buffer_exists else False

    logger.info(f"✓ Memory growth: {memory_growth} bytes")
    logger.info(f"✓ Buffer reused: {buffer_reused}")

    memory_stable = memory_growth < 5000  # Allow some growth but not excessive
    logger.info(
        f"✓ Memory stable (< 5KB growth): {memory_stable} ({'✓' if memory_stable else '✗'})",
    )

    # Test 5: Feature Quality
    logger.info("\n5. Testing Feature Quality")
    logger.info("-" * 40)

    feature_cols = [col for col in features_df.columns if col != "timestamp"]

    nan_counts = {}
    inf_counts = {}
    zero_counts = {}

    for col in feature_cols:
        values = features_df[col].to_numpy()
        nan_counts[col] = np.sum(np.isnan(values))
        inf_counts[col] = np.sum(np.isinf(values))
        zero_counts[col] = np.sum(values == 0.0)

    total_nans = sum(nan_counts.values())
    total_infs = sum(inf_counts.values())

    logger.info(f"✓ Features analyzed: {len(feature_cols)}")
    logger.info(f"✓ Total NaNs: {total_nans}")
    logger.info(f"✓ Total Infs: {total_infs}")

    quality_passed = total_nans == 0 and total_infs == 0
    logger.info(f"✓ Quality check: {quality_passed} ({'✓' if quality_passed else '✗'})")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    tests_passed = 0
    tests_total = 5

    if basic_passed:
        tests_passed += 1
        logger.info("✓ Basic functionality: PASS")
    else:
        logger.info("✗ Basic functionality: FAIL")

    if meets_5ms_sla:
        tests_passed += 1
        logger.info(f"✓ Performance (<5ms P99): PASS ({p99_latency:.3f}ms)")
    else:
        logger.info(f"✗ Performance (<5ms P99): FAIL ({p99_latency:.3f}ms)")

    if parity_passed:
        tests_passed += 1
        logger.info(f"✓ Mathematical parity: PASS (max diff: {max_diff:.2e})")
    else:
        logger.info("✗ Mathematical parity: FAIL")

    if memory_stable and buffer_reused:
        tests_passed += 1
        logger.info(f"✓ Memory efficiency: PASS ({memory_growth} bytes growth)")
    else:
        logger.info(f"✗ Memory efficiency: FAIL ({memory_growth} bytes growth)")

    if quality_passed:
        tests_passed += 1
        logger.info("✓ Feature quality: PASS")
    else:
        logger.info("✗ Feature quality: FAIL")

    logger.info(f"\nOverall: {tests_passed}/{tests_total} tests passed")

    # Claims verification
    logger.info("\n" + "=" * 60)
    logger.info("CLAIMS VERIFICATION")
    logger.info("=" * 60)

    verified_claims = []
    failed_claims = []

    if basic_passed:
        verified_claims.append("Feature engineering pipeline works")

    if buffer_exists:
        verified_claims.append("Pre-allocated buffers exist")

    if buffer_reused:
        verified_claims.append("Buffers are reused across calls")

    if meets_5ms_sla:
        verified_claims.append(f"<5ms P99 latency requirement (actual: {p99_latency:.3f}ms)")
    else:
        failed_claims.append(f"<5ms P99 latency requirement (actual: {p99_latency:.3f}ms)")

    if parity_passed:
        verified_claims.append("Mathematical parity between batch/online paths")
    else:
        failed_claims.append("Mathematical parity between batch/online paths")

    if memory_stable:
        verified_claims.append(f"Memory stable operation ({memory_growth} bytes growth)")
    else:
        failed_claims.append(f"Zero memory allocation (actual: {memory_growth} bytes growth)")

    if quality_passed:
        verified_claims.append("Feature quality (no NaNs/Infs)")
    else:
        failed_claims.append("Feature quality (contains NaNs/Infs)")

    logger.info("VERIFIED CLAIMS:")
    for claim in verified_claims:
        logger.info(f"  ✓ {claim}")

    if failed_claims:
        logger.info("\nFAILED CLAIMS:")
        for claim in failed_claims:
            logger.info(f"  ✗ {claim}")

    logger.info(
        f"\nVerification: {len(verified_claims)}/{len(verified_claims) + len(failed_claims)} claims verified",
    )

    return {
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "verified_claims": len(verified_claims),
        "total_claims": len(verified_claims) + len(failed_claims),
        "performance": {
            "mean_latency_ms": mean_latency,
            "p99_latency_ms": p99_latency,
            "meets_sla": meets_5ms_sla,
        },
        "parity": {
            "passed": parity_passed,
            "max_difference": max_diff if "max_diff" in locals() else None,
        },
        "memory": {
            "buffer_exists": buffer_exists,
            "buffer_reused": buffer_reused,
            "growth_bytes": memory_growth,
            "stable": memory_stable,
        },
    }


if __name__ == "__main__":
    results = run_comprehensive_test()
