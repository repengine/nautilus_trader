#!/usr/bin/env python3
"""
Simplified test script to validate ML feature engineering pipeline claims.
"""

import gc
import logging
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_data(n_bars=200):
    """
    Generate sample market data.
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

    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 12, 31),
                interval="1h",
                closed="left",
            )[:n_bars],
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        }
    )


def test_basic_functionality():
    """
    Test basic functionality.
    """
    logger.info("Testing basic functionality...")

    try:
        from ml.features.engineering import FeatureConfig, FeatureEngineer

        config = FeatureConfig(
            return_periods=[1, 5, 10],
            momentum_periods=[5, 10],
            rsi_period=14,
            bb_period=20,
            volume_ma_periods=[5, 10],
        )
        engineer = FeatureEngineer(config)

        # Test with small dataset
        df = generate_sample_data(100)

        # Test batch processing
        features_df, scaler = engineer.calculate_features_batch(df, fit_scaler=True)

        if features_df is not None and len(features_df) > 0:
            feature_count = len(features_df.columns) - 1  # -1 for timestamp
            logger.info(f"✓ Basic functionality works. Generated {feature_count} features")
            return {"passed": True, "feature_count": feature_count}
        else:
            logger.error("✗ No features generated")
            return {"passed": False, "error": "No features generated"}

    except Exception as e:
        logger.error(f"✗ Basic functionality failed: {e}")
        return {"passed": False, "error": str(e)}


def test_hot_cold_paths():
    """
    Test hot/cold path separation and parity.
    """
    logger.info("Testing hot/cold path separation...")

    try:
        from ml.features.engineering import FeatureConfig, FeatureEngineer, IndicatorManager

        config = FeatureConfig(
            return_periods=[1, 5],
            momentum_periods=[5],
            rsi_period=14,
            volume_ma_periods=[5],
        )
        engineer = FeatureEngineer(config)

        df = generate_sample_data(150)

        # Test batch path (cold)
        start_time = time.perf_counter()
        batch_features, scaler = engineer.calculate_features_batch(df, fit_scaler=True)
        batch_time = (time.perf_counter() - start_time) * 1000

        # Test online path (hot)
        indicator_mgr = IndicatorManager(config)
        online_features = []

        # Warmup indicators with first 30 bars
        for i in range(30):
            row = df.row(i, named=True)
            indicator_mgr.update_from_values(
                close=float(row["close"]),
                high=float(row["high"]),
                low=float(row["low"]),
                volume=float(row["volume"]),
            )

        # Time online computation
        online_times = []
        for i in range(30, 80):  # Test 50 bars
            row = df.row(i, named=True)
            current_bar = {
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": float(row["volume"]),
            }

            start = time.perf_counter()
            features = engineer.calculate_features_online(current_bar, indicator_mgr, scaler)
            end = time.perf_counter()

            online_time = (end - start) * 1000
            online_times.append(online_time)
            online_features.append(features.copy())

        avg_online_time = np.mean(online_times)
        p99_online_time = np.percentile(online_times, 99)

        # Test parity
        if len(online_features) > 0 and batch_features is not None:
            batch_subset = batch_features[30 : 30 + len(online_features)]
            batch_values = batch_subset.select(pl.exclude("timestamp")).to_numpy()
            online_values = np.array(online_features)

            if batch_values.shape == online_values.shape:
                max_diff = np.max(np.abs(batch_values - online_values))
                parity_passed = max_diff < 1e-6  # Reasonable tolerance

                logger.info(
                    f"✓ Hot/cold paths tested. Batch: {batch_time:.2f}ms, "
                    f"Online avg: {avg_online_time:.3f}ms, "
                    f"P99: {p99_online_time:.3f}ms, Max diff: {max_diff:.2e}"
                )

                return {
                    "batch_time_ms": batch_time,
                    "online_avg_ms": avg_online_time,
                    "online_p99_ms": p99_online_time,
                    "parity_passed": parity_passed,
                    "max_difference": max_diff,
                    "meets_5ms_sla": p99_online_time < 5.0,
                }

    except Exception as e:
        logger.error(f"✗ Hot/cold path test failed: {e}")
        return {"error": str(e)}


def test_memory_allocation():
    """
    Test zero-allocation claims.
    """
    logger.info("Testing memory allocation...")

    try:
        from ml.features.engineering import FeatureConfig, FeatureEngineer, IndicatorManager

        config = FeatureConfig(return_periods=[1, 5], volume_ma_periods=[5])
        engineer = FeatureEngineer(config)

        # Check for pre-allocated buffer
        has_buffer = hasattr(engineer, "feature_buffer") and isinstance(
            engineer.feature_buffer, np.ndarray
        )

        df = generate_sample_data(100)
        indicator_mgr = IndicatorManager(config)

        # Warmup
        for i in range(20):
            row = df.row(i, named=True)
            indicator_mgr.update_from_values(
                close=float(row["close"]),
                high=float(row["high"]),
                low=float(row["low"]),
                volume=float(row["volume"]),
            )

        # Test memory allocation during hot path
        tracemalloc.start()
        gc.collect()

        memory_before = tracemalloc.get_traced_memory()[0]

        # Run hot path operations
        for i in range(20, 50):
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

        logger.info(
            f"✓ Memory test completed. Pre-allocated buffer: {has_buffer}, "
            f"Memory growth: {memory_growth} bytes"
        )

        return {
            "has_pre_allocated_buffer": has_buffer,
            "memory_growth_bytes": memory_growth,
            "memory_stable": memory_growth < 1024,  # Less than 1KB growth
        }

    except Exception as e:
        logger.error(f"✗ Memory test failed: {e}")
        tracemalloc.stop()
        return {"error": str(e)}


def test_mathematical_correctness():
    """
    Test mathematical correctness.
    """
    logger.info("Testing mathematical correctness...")

    try:
        from ml.features.engineering import FeatureConfig, FeatureEngineer

        config = FeatureConfig(
            return_periods=[1, 5],
            rsi_period=14,
            volume_ma_periods=[5],
        )
        engineer = FeatureEngineer(config)

        df = generate_sample_data(100)
        features_df, _ = engineer.calculate_features_batch(df, fit_scaler=False)

        if features_df is not None:
            numeric_cols = [col for col in features_df.columns if col != "timestamp"]

            has_nans = False
            has_infs = False

            for col in numeric_cols:
                values = features_df[col].to_numpy()
                if np.any(np.isnan(values)):
                    has_nans = True
                if np.any(np.isinf(values)):
                    has_infs = True

            logger.info(
                f"✓ Mathematical correctness tested for {len(numeric_cols)} features. "
                f"NaNs: {has_nans}, Infs: {has_infs}"
            )

            return {
                "feature_count": len(numeric_cols),
                "has_nans": has_nans,
                "has_infs": has_infs,
                "passed": not (has_nans or has_infs),
            }

    except Exception as e:
        logger.error(f"✗ Mathematical correctness test failed: {e}")
        return {"error": str(e)}


def main():
    """
    Run all tests.
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE TEST")
    logger.info("=" * 60)

    results = {}

    # Run tests
    results["basic"] = test_basic_functionality()
    results["hot_cold"] = test_hot_cold_paths()
    results["memory"] = test_memory_allocation()
    results["math"] = test_mathematical_correctness()

    # Generate report
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    # Basic functionality
    basic = results.get("basic", {})
    if basic.get("passed"):
        logger.info(f"✓ Basic functionality: PASS ({basic.get('feature_count')} features)")
    else:
        logger.info(f"✗ Basic functionality: FAIL ({basic.get('error')})")

    # Hot/cold paths
    hot_cold = results.get("hot_cold", {})
    if "error" not in hot_cold:
        p99 = hot_cold.get("online_p99_ms", 0)
        parity = hot_cold.get("parity_passed", False)
        sla = hot_cold.get("meets_5ms_sla", False)
        logger.info(f"✓ Performance: P99 {p99:.3f}ms ({'✓' if sla else '✗'} <5ms SLA)")
        logger.info(
            f"✓ Parity: {'PASS' if parity else 'FAIL'} (diff: {hot_cold.get('max_difference', 0):.2e})"
        )
    else:
        logger.info(f"✗ Hot/cold paths: FAIL ({hot_cold.get('error')})")

    # Memory
    memory = results.get("memory", {})
    if "error" not in memory:
        buffer = memory.get("has_pre_allocated_buffer", False)
        stable = memory.get("memory_stable", False)
        growth = memory.get("memory_growth_bytes", 0)
        logger.info(f"✓ Pre-allocated buffer: {'YES' if buffer else 'NO'}")
        logger.info(f"✓ Memory stable: {'YES' if stable else 'NO'} ({growth} bytes growth)")
    else:
        logger.info(f"✗ Memory test: FAIL ({memory.get('error')})")

    # Math
    math_result = results.get("math", {})
    if "error" not in math_result:
        passed = math_result.get("passed", False)
        logger.info(f"✓ Mathematical correctness: {'PASS' if passed else 'FAIL'}")
    else:
        logger.info(f"✗ Mathematical correctness: FAIL ({math_result.get('error')})")

    logger.info("=" * 60)

    # Claims analysis
    logger.info("CLAIMS ANALYSIS:")
    logger.info("")

    verified_claims = []
    failed_claims = []

    if basic.get("passed"):
        verified_claims.append("Feature engineering pipeline works")
    else:
        failed_claims.append("Basic feature engineering functionality")

    if memory.get("has_pre_allocated_buffer"):
        verified_claims.append("Pre-allocated buffers exist")
    else:
        failed_claims.append("Pre-allocated buffer claims")

    if hot_cold.get("parity_passed"):
        verified_claims.append("Mathematical parity between batch/online")
    else:
        failed_claims.append("Perfect mathematical parity")

    if hot_cold.get("meets_5ms_sla"):
        verified_claims.append("<5ms P99 latency requirement")
    else:
        failed_claims.append(f"<5ms P99 latency (actual: {hot_cold.get('online_p99_ms', 0):.3f}ms)")

    if memory.get("memory_stable"):
        verified_claims.append("Memory stability in hot path")
    else:
        failed_claims.append("Zero memory allocation in hot path")

    logger.info("VERIFIED CLAIMS:")
    for claim in verified_claims:
        logger.info(f"  • {claim}")

    logger.info("")
    logger.info("FAILED CLAIMS:")
    for claim in failed_claims:
        logger.info(f"  • {claim}")

    logger.info("")
    logger.info(
        f"Overall: {len(verified_claims)}/{len(verified_claims) + len(failed_claims)} claims verified"
    )


if __name__ == "__main__":
    main()
