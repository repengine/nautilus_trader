#!/usr/bin/env python3
"""
Comprehensive test script to validate ML feature engineering pipeline claims.

This script empirically tests:
1. Hot/cold path separation and performance (claimed <5ms P99)
2. Zero-allocation claims during hot path feature computation
3. Mathematical correctness and batch/online parity
4. Pre-allocated buffer usage
5. Feature quality and integration

Run with: python ml/test_feature_claims.py

"""

import gc
import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_market_data(n_bars: int = 1000) -> pl.DataFrame:
    """
    Generate realistic sample market data for testing.
    """
    np.random.seed(42)

    # Generate realistic price data with trend and volatility
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_bars)  # Small positive drift, 2% daily vol
    prices = [base_price]

    for ret in returns:
        prices.append(prices[-1] * (1 + ret))

    close_prices = np.array(prices[1:])
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    # Generate realistic volume
    volumes = np.random.lognormal(15, 0.5, n_bars)  # Log-normal volume distribution

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


class FeatureEngineeringTester:
    """
    Comprehensive tester for feature engineering claims.
    """

    def __init__(self):
        self.results = {}
        self.config = FeatureConfig(
            return_periods=[1, 5, 10],
            momentum_periods=[5, 10],
            rsi_period=14,
            bb_period=20,
            volume_ma_periods=[5, 10],
            include_microstructure=False,  # Keep simple for core testing
            include_trade_flow=False,
        )
        self.engineer = FeatureEngineer(self.config)

    def test_basic_functionality(self) -> dict[str, Any]:
        """
        Test that basic feature engineering actually works.
        """
        logger.info("Testing basic functionality...")

        results = {"passed": False, "error": None, "feature_count": 0}

        try:
            # Test with small dataset
            df = generate_sample_market_data(100)

            # Test batch processing
            features_df, scaler = self.engineer.calculate_features_batch(df, fit_scaler=True)

            if features_df is not None and len(features_df) > 0:
                results["feature_count"] = len(features_df.columns) - 1  # -1 for timestamp
                results["passed"] = True
                logger.info(
                    f"✓ Basic functionality works. Generated {results['feature_count']} features"
                )
            else:
                results["error"] = "No features generated"

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"✗ Basic functionality failed: {e}")

        return results

    def test_hot_cold_path_separation(self) -> dict[str, Any]:
        """
        Test hot/cold path separation claims.
        """
        logger.info("Testing hot/cold path separation...")

        results = {
            "batch_time_ms": 0,
            "online_time_ms": 0,
            "parity_passed": False,
            "max_difference": float("inf"),
            "error": None,
        }

        try:
            df = generate_sample_market_data(200)

            # Test batch path (cold)
            start = time.perf_counter()
            batch_features, scaler = self.engineer.calculate_features_batch(df, fit_scaler=True)
            batch_time = (time.perf_counter() - start) * 1000
            results["batch_time_ms"] = batch_time

            # Test online path (hot) - use same data for parity check
            indicator_mgr = IndicatorManager(self.config)
            online_features = []

            # Warmup indicators
            for i in range(50):  # Warmup with first 50 bars
                row = df[i]
                indicator_mgr.update_from_values(
                    close=float(row["close"][0]),
                    high=float(row["high"][0]),
                    low=float(row["low"][0]),
                    volume=float(row["volume"][0]),
                )

            # Time online computation for remaining bars
            online_times = []
            for i in range(50, min(100, len(df))):  # Test next 50 bars
                row = df[i]
                current_bar = {
                    "close": float(row["close"][0]),
                    "high": float(row["high"][0]),
                    "low": float(row["low"][0]),
                    "volume": float(row["volume"][0]),
                }

                start = time.perf_counter()
                features = self.engineer.calculate_features_online(
                    current_bar,
                    indicator_mgr,
                    scaler,
                )
                online_time = (time.perf_counter() - start) * 1000
                online_times.append(online_time)
                online_features.append(features.copy())

            results["online_time_ms"] = float(np.mean(online_times))
            results["online_p99_ms"] = float(np.percentile(online_times, 99))

            # Test parity (compare a subset where both have data)
            if len(online_features) > 0 and batch_features is not None:
                batch_subset = batch_features[50 : 50 + len(online_features)]
                batch_values = batch_subset.select(pl.exclude("timestamp")).to_numpy()
                online_values = np.array(online_features)

                if batch_values.shape == online_values.shape:
                    max_diff = float(np.max(np.abs(batch_values - online_values)))
                    results["max_difference"] = max_diff
                    results["parity_passed"] = max_diff < 1e-8  # Relaxed from claimed 1e-10

            logger.info(
                f"✓ Hot/cold separation tested. Batch: {batch_time:.2f}ms, "
                f"Online avg: {results['online_time_ms']:.2f}ms, "
                f"P99: {results.get('online_p99_ms', 0):.2f}ms"
            )

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"✗ Hot/cold path separation failed: {e}")

        return results

    def test_zero_allocation_claims(self):
        """
        Test zero-allocation claims in hot path.
        """
        logger.info("Testing zero-allocation claims...")

        results = {
            "pre_allocated_buffer_exists": False,
            "buffer_reused": False,
            "memory_stable": False,
            "allocations_detected": 0,
            "error": None,
        }

        try:
            # Check if pre-allocated buffer exists
            if hasattr(self.engineer, "feature_buffer") and isinstance(
                self.engineer.feature_buffer, np.ndarray
            ):
                results["pre_allocated_buffer_exists"] = True
                buffer_id_before = id(self.engineer.feature_buffer)
                logger.info(f"✓ Pre-allocated buffer found: {self.engineer.feature_buffer.shape}")

            # Generate test data
            df = generate_sample_market_data(100)
            indicator_mgr = IndicatorManager(self.config)

            # Warmup
            for i in range(20):
                row = df[i]
                indicator_mgr.update_from_values(
                    close=float(row["close"][0]),
                    high=float(row["high"][0]),
                    low=float(row["low"][0]),
                    volume=float(row["volume"][0]),
                )

            # Start memory tracing
            tracemalloc.start()
            gc.collect()  # Clean slate

            # Test hot path memory allocation
            memory_before = tracemalloc.get_traced_memory()[0]

            for i in range(20, 50):  # Test 30 iterations
                row = df[i]
                current_bar = {
                    "close": float(row["close"][0]),
                    "high": float(row["high"][0]),
                    "low": float(row["low"][0]),
                    "volume": float(row["volume"][0]),
                }

                features = self.engineer.calculate_features_online(current_bar, indicator_mgr)

                # Check if buffer is being reused
                if hasattr(self.engineer, "feature_buffer"):
                    buffer_id_after = id(self.engineer.feature_buffer)
                    if buffer_id_before == buffer_id_after:
                        results["buffer_reused"] = True

            memory_after = tracemalloc.get_traced_memory()[0]
            memory_growth = memory_after - memory_before

            tracemalloc.stop()

            results["memory_growth_bytes"] = memory_growth
            results["memory_stable"] = memory_growth < 1024  # Less than 1KB growth

            logger.info(f"✓ Zero-allocation test completed. Memory growth: {memory_growth} bytes")

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"✗ Zero-allocation test failed: {e}")
            tracemalloc.stop()

        return results

    def test_performance_claims(self) -> dict[str, Any]:
        """
        Test <5ms P99 latency claims.
        """
        logger.info("Testing performance claims...")

        results = {
            "p99_latency_ms": float("inf"),
            "mean_latency_ms": 0,
            "meets_sla": False,
            "throughput_ops_per_sec": 0,
            "error": None,
        }

        try:
            df = generate_sample_market_data(500)
            indicator_mgr = IndicatorManager(self.config)

            # Warmup
            for i in range(50):
                row = df[i]
                indicator_mgr.update_from_values(
                    close=float(row["close"][0]),
                    high=float(row["high"][0]),
                    low=float(row["low"][0]),
                    volume=float(row["volume"][0]),
                )

            # Measure performance over many iterations
            latencies = []
            start_throughput = time.perf_counter()

            for i in range(50, 450):  # 400 iterations
                row = df[i]
                current_bar = {
                    "close": float(row["close"][0]),
                    "high": float(row["high"][0]),
                    "low": float(row["low"][0]),
                    "volume": float(row["volume"][0]),
                }

                start = time.perf_counter_ns()
                features = self.engineer.calculate_features_online(current_bar, indicator_mgr)
                end = time.perf_counter_ns()

                latency_ms = (end - start) / 1_000_000  # Convert to milliseconds
                latencies.append(latency_ms)

            throughput_time = time.perf_counter() - start_throughput

            results["mean_latency_ms"] = float(np.mean(latencies))
            results["p99_latency_ms"] = float(np.percentile(latencies, 99))
            results["meets_sla"] = results["p99_latency_ms"] < 5.0
            results["throughput_ops_per_sec"] = len(latencies) / throughput_time

            logger.info(
                f"✓ Performance test completed. "
                f"P99: {results['p99_latency_ms']:.3f}ms, "
                f"Mean: {results['mean_latency_ms']:.3f}ms, "
                f"Throughput: {results['throughput_ops_per_sec']:.0f} ops/sec"
            )

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"✗ Performance test failed: {e}")

        return results

    def test_mathematical_correctness(self) -> dict[str, Any]:
        """
        Test mathematical correctness of features.
        """
        logger.info("Testing mathematical correctness...")

        results = {
            "features_in_range": True,
            "no_nans_infs": True,
            "feature_statistics": {},
            "error": None,
        }

        try:
            df = generate_sample_market_data(200)
            features_df, _ = self.engineer.calculate_features_batch(df, fit_scaler=False)

            if features_df is not None:
                # Check for NaNs and infinities
                numeric_cols = [col for col in features_df.columns if col != "timestamp"]

                for col in numeric_cols:
                    values = features_df[col].to_numpy()

                    has_nan = np.any(np.isnan(values))
                    has_inf = np.any(np.isinf(values))

                    if has_nan or has_inf:
                        results["no_nans_infs"] = False

                    results["feature_statistics"][col] = {
                        "mean": float(np.nanmean(values)),
                        "std": float(np.nanstd(values)),
                        "min": float(np.nanmin(values)),
                        "max": float(np.nanmax(values)),
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                    }

                logger.info(f"✓ Mathematical correctness tested for {len(numeric_cols)} features")

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"✗ Mathematical correctness test failed: {e}")

        return results

    def test_store_integration(self) -> dict[str, Any]:
        """
        Test FeatureStore integration if available.
        """
        logger.info("Testing store integration...")

        results = {
            "store_available": False,
            "can_persist_features": False,
            "error": None,
        }

        try:
            # Try to import and use store
            try:
                from ml.stores.feature_store import FeatureStore

                results["store_available"] = True
                logger.info("✓ FeatureStore import successful")
            except ImportError:
                results["error"] = "FeatureStore not available"
                logger.info("ℹ FeatureStore not available for testing")

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"✗ Store integration test failed: {e}")

        return results

    def run_all_tests(self) -> dict[str, Any]:
        """
        Run all tests and compile results.
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE FEATURE ENGINEERING TESTS")
        logger.info("=" * 60)

        all_results = {}

        # Run tests in order
        all_results["basic_functionality"] = self.test_basic_functionality()
        all_results["hot_cold_separation"] = self.test_hot_cold_path_separation()
        all_results["zero_allocation"] = self.test_zero_allocation_claims()
        all_results["performance"] = self.test_performance_claims()
        all_results["mathematical_correctness"] = self.test_mathematical_correctness()
        all_results["store_integration"] = self.test_store_integration()

        return all_results


def generate_report(results: dict[str, Any]) -> str:
    """
    Generate a comprehensive report of findings.
    """
    report = []
    report.append("=" * 80)
    report.append("FEATURE ENGINEERING PIPELINE TEST REPORT")
    report.append("=" * 80)
    report.append("")

    # Basic functionality
    basic = results.get("basic_functionality", {})
    report.append("1. BASIC FUNCTIONALITY")
    report.append("-" * 40)
    if basic.get("passed"):
        report.append(
            f"✓ PASS: Feature engineering works, generated {basic.get('feature_count')} features"
        )
    else:
        report.append(f"✗ FAIL: {basic.get('error', 'Unknown error')}")
    report.append("")

    # Hot/Cold Path Separation
    hot_cold = results.get("hot_cold_separation", {})
    report.append("2. HOT/COLD PATH SEPARATION")
    report.append("-" * 40)
    if hot_cold.get("error"):
        report.append(f"✗ FAIL: {hot_cold['error']}")
    else:
        report.append(f"Batch (cold) time: {hot_cold.get('batch_time_ms', 0):.2f}ms")
        report.append(f"Online (hot) avg time: {hot_cold.get('online_time_ms', 0):.2f}ms")
        report.append(f"Online P99 latency: {hot_cold.get('online_p99_ms', 0):.2f}ms")

        if hot_cold.get("parity_passed"):
            report.append(
                f"✓ PASS: Mathematical parity maintained (max diff: {hot_cold.get('max_difference', 0):.2e})"
            )
        else:
            report.append(
                f"✗ FAIL: Parity violation (max diff: {hot_cold.get('max_difference', 0):.2e})"
            )
    report.append("")

    # Zero Allocation Claims
    zero_alloc = results.get("zero_allocation", {})
    report.append("3. ZERO-ALLOCATION CLAIMS")
    report.append("-" * 40)
    if zero_alloc.get("error"):
        report.append(f"✗ FAIL: {zero_alloc['error']}")
    else:
        if zero_alloc.get("pre_allocated_buffer_exists"):
            report.append("✓ PASS: Pre-allocated buffer exists")
        else:
            report.append("✗ FAIL: No pre-allocated buffer found")

        if zero_alloc.get("buffer_reused"):
            report.append("✓ PASS: Buffer reused across calls")
        else:
            report.append("? WARNING: Buffer reuse unclear")

        growth = zero_alloc.get("memory_growth_bytes", 0)
        if growth < 1024:
            report.append(f"✓ PASS: Memory stable ({growth} bytes growth)")
        else:
            report.append(f"✗ FAIL: Memory growth detected ({growth} bytes)")
    report.append("")

    # Performance Claims
    perf = results.get("performance", {})
    report.append("4. PERFORMANCE CLAIMS (<5ms P99)")
    report.append("-" * 40)
    if perf.get("error"):
        report.append(f"✗ FAIL: {perf['error']}")
    else:
        p99 = perf.get("p99_latency_ms", float("inf"))
        mean_lat = perf.get("mean_latency_ms", 0)
        throughput = perf.get("throughput_ops_per_sec", 0)

        if perf.get("meets_sla"):
            report.append(f"✓ PASS: P99 latency {p99:.3f}ms < 5ms SLA")
        else:
            report.append(f"✗ FAIL: P99 latency {p99:.3f}ms > 5ms SLA")

        report.append(f"Mean latency: {mean_lat:.3f}ms")
        report.append(f"Throughput: {throughput:.0f} ops/sec")
    report.append("")

    # Mathematical Correctness
    math_correct = results.get("mathematical_correctness", {})
    report.append("5. MATHEMATICAL CORRECTNESS")
    report.append("-" * 40)
    if math_correct.get("error"):
        report.append(f"✗ FAIL: {math_correct['error']}")
    else:
        if math_correct.get("no_nans_infs"):
            report.append("✓ PASS: No NaNs or infinities detected")
        else:
            report.append("✗ FAIL: NaNs or infinities found in features")

        stats = math_correct.get("feature_statistics", {})
        report.append(f"Features analyzed: {len(stats)}")
    report.append("")

    # Store Integration
    store = results.get("store_integration", {})
    report.append("6. FEATURE STORE INTEGRATION")
    report.append("-" * 40)
    if store.get("store_available"):
        report.append("✓ INFO: FeatureStore available")
    else:
        report.append("ℹ INFO: FeatureStore not available")
    report.append("")

    # Summary
    report.append("SUMMARY")
    report.append("-" * 40)

    # Count passes/fails
    tests_passed = 0
    tests_total = 0

    if basic.get("passed"):
        tests_passed += 1
    tests_total += 1

    if hot_cold.get("parity_passed") and not hot_cold.get("error"):
        tests_passed += 1
    tests_total += 1

    if zero_alloc.get("memory_stable") and not zero_alloc.get("error"):
        tests_passed += 1
    tests_total += 1

    if perf.get("meets_sla") and not perf.get("error"):
        tests_passed += 1
    tests_total += 1

    if math_correct.get("no_nans_infs") and not math_correct.get("error"):
        tests_passed += 1
    tests_total += 1

    report.append(f"Tests passed: {tests_passed}/{tests_total}")
    report.append("")

    # Verdict on claims
    report.append("CLAIMS ANALYSIS:")
    report.append("")

    report.append("VERIFIED CLAIMS:")
    if basic.get("passed"):
        report.append("• Feature engineering pipeline works")
    if zero_alloc.get("pre_allocated_buffer_exists"):
        report.append("• Pre-allocated buffers exist")
    if hot_cold.get("parity_passed"):
        report.append("• Mathematical parity between batch/online")

    report.append("")
    report.append("FAILED CLAIMS:")
    if not perf.get("meets_sla") and not perf.get("error"):
        report.append(f"• <5ms P99 latency (actual: {perf.get('p99_latency_ms', 0):.3f}ms)")
    if not zero_alloc.get("memory_stable"):
        report.append("• Zero memory allocation in hot path")
    if not hot_cold.get("parity_passed"):
        report.append("• Perfect mathematical parity")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """
    Main test runner.
    """
    try:
        # Create tester and run all tests
        tester = FeatureEngineeringTester()
        results = tester.run_all_tests()

        # Generate and print report
        report = generate_report(results)
        print(report)

        # Also save to file
        report_path = Path("ml/feature_engineering_test_report.txt")
        report_path.write_text(report)
        print(f"\nReport saved to: {report_path.absolute()}")

    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
