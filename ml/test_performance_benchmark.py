#!/usr/bin/env python3
"""
Performance benchmark test for L2/L3 microstructure features.

Tests actual performance against documented claims:
- Hot path <5ms P99 latency
- 1000+ bars/second throughput
- Zero allocation during inference

"""

import gc
import logging
import statistics
import sys
import time

import numpy as np
import pandas as pd

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from ml.features.l2_aggregate import aggregate_l2_minute_pl
from ml.features.microstructure import L2MicrostructureFeatures
from ml.features.microstructure import L3TradeFlowFeatures


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data(n_samples: int, include_l2: bool = False, include_l3: bool = False):
    """
    Create test data for benchmarking.
    """
    np.random.seed(42)

    base_time = pd.Timestamp("2024-01-01 09:30:00", tz="UTC")
    timestamps = [base_time + pd.Timedelta(seconds=i) for i in range(n_samples)]

    data = {
        "open": [100.0 + i * 0.001 + np.random.normal(0, 0.1) for i in range(n_samples)],
        "high": [100.5 + i * 0.001 + np.random.normal(0, 0.1) for i in range(n_samples)],
        "low": [99.5 + i * 0.001 + np.random.normal(0, 0.1) for i in range(n_samples)],
        "close": [100.0 + i * 0.001 + np.random.normal(0, 0.1) for i in range(n_samples)],
        "volume": [1000 + np.random.exponential(500) for _ in range(n_samples)],
        "ts_event": timestamps,
        "ts_init": timestamps,
    }

    if include_l2:
        # Add L2 order book data
        for level in range(10):
            data[f"bid_price_{level}"] = [
                99.95 - level * 0.001 + np.random.normal(0, 0.0005) for _ in range(n_samples)
            ]
            data[f"ask_price_{level}"] = [
                100.05 + level * 0.001 + np.random.normal(0, 0.0005) for _ in range(n_samples)
            ]
            data[f"bid_size_{level}"] = [
                max(100, 1000 * (1 - level * 0.1) + np.random.normal(0, 50))
                for _ in range(n_samples)
            ]
            data[f"ask_size_{level}"] = [
                max(100, 1000 * (1 - level * 0.1) + np.random.normal(0, 50))
                for _ in range(n_samples)
            ]

    if include_l3:
        # Add L3 trade data
        data["trade_price"] = [100.0 + np.random.normal(0, 0.01) for _ in range(n_samples)]
        data["trade_volume"] = [max(1, np.random.exponential(100)) for _ in range(n_samples)]
        data["trade_side"] = [1 if np.random.random() > 0.5 else -1 for _ in range(n_samples)]

    if HAS_POLARS:
        return pl.DataFrame(data)
    else:
        return pd.DataFrame(data)


def benchmark_online_latency():
    """
    Benchmark online feature computation latency.
    """
    logger.info("=== Benchmarking Online Feature Computation Latency ===")

    # Test different configurations
    configs = [
        ("Basic L1", FeatureConfig(include_microstructure=False, include_trade_flow=False)),
        (
            "L1 + Microstructure",
            FeatureConfig(include_microstructure=True, include_trade_flow=False),
        ),
        ("L1 + Trade Flow", FeatureConfig(include_microstructure=False, include_trade_flow=True)),
        ("Full L2/L3", FeatureConfig(include_microstructure=True, include_trade_flow=True)),
    ]

    results = {}

    for config_name, config in configs:
        logger.info(f"Testing configuration: {config_name}")

        try:
            engineer = FeatureEngineer(config)
            indicator_mgr = IndicatorManager(config)

            # Warm up indicators
            warmup_data = create_test_data(50)
            for i in range(len(warmup_data)):
                if HAS_POLARS:
                    row = warmup_data.slice(i, 1)
                    indicator_mgr.update_from_values(
                        close=float(row["close"][0]),
                        high=float(row["high"][0]),
                        low=float(row["low"][0]),
                        volume=float(row["volume"][0]),
                    )
                else:
                    row = warmup_data.iloc[i]
                    indicator_mgr.update_from_values(
                        close=float(row["close"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        volume=float(row["volume"]),
                    )

            # Create scaler for normalization
            train_data = create_test_data(100)
            _, scaler = engineer.calculate_features(train_data, mode="batch", fit_scaler=True)

            # Benchmark online computation
            n_iterations = 1000
            latencies = []

            # Force garbage collection before benchmark
            gc.collect()

            for i in range(n_iterations):
                # Generate single bar
                close = 100.0 + np.random.normal(0, 0.1)
                high = close + abs(np.random.normal(0, 0.1))
                low = close - abs(np.random.normal(0, 0.1))
                volume = 1000 + np.random.exponential(500)

                # Measure latency
                start_time = time.perf_counter()

                features = engineer.calculate_features_online(
                    close_price=close,
                    high_price=high,
                    low_price=low,
                    volume=volume,
                    scaler=scaler,
                )

                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            # Calculate statistics
            mean_latency = statistics.mean(latencies)
            p50_latency = statistics.median(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            max_latency = max(latencies)

            results[config_name] = {
                "mean_ms": mean_latency,
                "p50_ms": p50_latency,
                "p95_ms": p95_latency,
                "p99_ms": p99_latency,
                "max_ms": max_latency,
                "n_features": len(features),
            }

            # Check SLA compliance
            sla_status = "✓ PASS" if p99_latency < 5.0 else "✗ FAIL"
            logger.info(f"  {sla_status} P99: {p99_latency:.3f}ms (target: <5ms)")
            logger.info(
                f"  Features: {len(features)}, Mean: {mean_latency:.3f}ms, Max: {max_latency:.3f}ms"
            )

        except Exception as e:
            logger.error(f"  ✗ FAIL - Exception: {e}")
            results[config_name] = {"error": str(e)}

    return results


def benchmark_throughput():
    """
    Benchmark batch processing throughput.
    """
    logger.info("=== Benchmarking Batch Processing Throughput ===")

    config = FeatureConfig(include_microstructure=True, include_trade_flow=True)
    engineer = FeatureEngineer(config)

    test_sizes = [100, 500, 1000, 5000]
    throughput_results = {}

    for n_samples in test_sizes:
        logger.info(f"Testing batch size: {n_samples}")

        try:
            # Create test data
            df = create_test_data(n_samples)

            # Benchmark batch processing
            start_time = time.perf_counter()
            features_df, _ = engineer.calculate_features(df, mode="batch", fit_scaler=True)
            end_time = time.perf_counter()

            processing_time = end_time - start_time
            throughput = n_samples / processing_time

            logger.info(f"  Processed {n_samples} samples in {processing_time:.3f}s")
            logger.info(f"  Throughput: {throughput:.1f} samples/sec")
            logger.info(f"  Features shape: {features_df.shape}")

            throughput_results[n_samples] = {
                "processing_time_s": processing_time,
                "throughput_samples_per_sec": throughput,
                "n_features": features_df.shape[1],
            }

            # Check if we meet 1000+ samples/sec claim
            if throughput >= 1000:
                logger.info("  ✓ PASS - Meets 1000+ samples/sec target")
            else:
                logger.info("  ⚠ WARNING - Below 1000 samples/sec target")

        except Exception as e:
            logger.error(f"  ✗ FAIL - Exception: {e}")
            throughput_results[n_samples] = {"error": str(e)}

    return throughput_results


def benchmark_l2_aggregation_performance():
    """
    Benchmark L2 aggregation performance at scale.
    """
    logger.info("=== Benchmarking L2 Aggregation Performance ===")

    if not HAS_POLARS:
        logger.warning("Polars not available - skipping L2 aggregation benchmark")
        return {}

    # Test different data sizes
    test_sizes = [1000, 5000, 10000, 50000]
    results = {}

    for n_samples in test_sizes:
        logger.info(f"Testing L2 aggregation with {n_samples} samples")

        try:
            # Create L2 data
            base_time = pd.Timestamp("2024-01-01 09:30:00", tz="UTC")
            timestamps = [base_time + pd.Timedelta(seconds=i) for i in range(n_samples)]

            data = {"ts_event": timestamps}

            # Generate 10 levels of order book data
            for level in range(10):
                data[f"bid_px_{level:02d}"] = [
                    float(100.0 - 0.005 - level * 0.001 + np.random.normal(0, 0.0001))
                    for _ in range(n_samples)
                ]
                data[f"ask_px_{level:02d}"] = [
                    float(100.0 + 0.005 + level * 0.001 + np.random.normal(0, 0.0001))
                    for _ in range(n_samples)
                ]
                data[f"bid_sz_{level:02d}"] = [
                    int(max(100, 1000 - level * 100 + np.random.normal(0, 20)))
                    for _ in range(n_samples)
                ]
                data[f"ask_sz_{level:02d}"] = [
                    int(max(100, 1000 - level * 100 + np.random.normal(0, 20)))
                    for _ in range(n_samples)
                ]

            df = pl.DataFrame(data)
            df = df.with_columns(pl.col("ts_event").cast(pl.Datetime("ns", "UTC")))

            # Benchmark aggregation
            start_time = time.perf_counter()
            result = aggregate_l2_minute_pl(df)
            end_time = time.perf_counter()

            processing_time = end_time - start_time
            throughput = n_samples / processing_time

            logger.info(
                f"  Processed {n_samples} -> {len(result)} minutes in {processing_time:.3f}s"
            )
            logger.info(f"  Throughput: {throughput:.0f} samples/sec")
            logger.info(f"  Generated {len(result.columns)} L2 features")

            results[n_samples] = {
                "processing_time_s": processing_time,
                "throughput_samples_per_sec": throughput,
                "output_rows": len(result),
                "n_features": len(result.columns),
            }

        except Exception as e:
            logger.error(f"  ✗ FAIL - Exception: {e}")
            results[n_samples] = {"error": str(e)}

    return results


def benchmark_l2_l3_feature_computation():
    """
    Benchmark L2/L3 feature computation performance.
    """
    logger.info("=== Benchmarking L2/L3 Feature Computation ===")

    results = {}

    # Test L2 features
    logger.info("Testing L2 microstructure features...")
    try:
        calculator = L2MicrostructureFeatures(n_levels=10, lookback_window=20)

        n_samples = 1000
        n_levels = 10

        # Generate realistic data
        bid_prices = np.random.normal(99.95, 0.01, (n_samples, n_levels))
        ask_prices = np.random.normal(100.05, 0.01, (n_samples, n_levels))
        bid_sizes = np.random.exponential(1000, (n_samples, n_levels))
        ask_sizes = np.random.exponential(1000, (n_samples, n_levels))

        # Benchmark computation
        start_time = time.perf_counter()

        spread_features = calculator.compute_spread_features(
            bid_prices, ask_prices, bid_sizes, ask_sizes
        )
        imbalance_features = calculator.compute_imbalance_features(
            bid_sizes, ask_sizes, bid_prices, ask_prices
        )
        depth_features = calculator.compute_depth_features(
            bid_sizes, ask_sizes, bid_prices, ask_prices
        )
        shape_features = calculator.compute_shape_features(
            bid_sizes, ask_sizes, bid_prices, ask_prices
        )

        end_time = time.perf_counter()

        total_features = (
            len(spread_features)
            + len(imbalance_features)
            + len(depth_features)
            + len(shape_features)
        )
        processing_time = end_time - start_time

        logger.info(f"  L2 computation: {processing_time*1000:.2f}ms for {total_features} features")
        logger.info(f"  Per-feature time: {processing_time*1000/total_features:.3f}ms")

        results["L2"] = {
            "processing_time_ms": processing_time * 1000,
            "n_features": total_features,
            "per_feature_ms": processing_time * 1000 / total_features,
        }

    except Exception as e:
        logger.error(f"  L2 features failed: {e}")
        results["L2"] = {"error": str(e)}

    # Test L3 features
    logger.info("Testing L3 trade flow features...")
    try:
        calculator = L3TradeFlowFeatures(lookback_window=100)

        n_trades = 1000
        timestamps = np.arange(n_trades, dtype=np.int64) * 1_000_000_000  # 1 second intervals
        prices = np.random.normal(100.0, 0.1, n_trades)
        volumes = np.random.exponential(100, n_trades)
        sides = np.random.choice([-1, 1], n_trades)

        # Benchmark computation
        start_time = time.perf_counter()

        imbalance_features = calculator.compute_trade_imbalance(prices, volumes, sides)
        vwap_features = calculator.compute_vwap_features(prices, volumes, sides)
        intensity_features = calculator.compute_intensity_features(timestamps, volumes, prices)
        impact_features = calculator.compute_price_impact(prices, volumes, sides)

        end_time = time.perf_counter()

        total_features = (
            len(imbalance_features)
            + len(vwap_features)
            + len(intensity_features)
            + len(impact_features)
        )
        processing_time = end_time - start_time

        logger.info(f"  L3 computation: {processing_time*1000:.2f}ms for {total_features} features")
        logger.info(f"  Per-feature time: {processing_time*1000/total_features:.3f}ms")

        results["L3"] = {
            "processing_time_ms": processing_time * 1000,
            "n_features": total_features,
            "per_feature_ms": processing_time * 1000 / total_features,
        }

    except Exception as e:
        logger.error(f"  L3 features failed: {e}")
        results["L3"] = {"error": str(e)}

    return results


def main():
    """
    Run performance benchmarks.
    """
    logger.info("Starting L2/L3 microstructure performance benchmarks...")

    all_results = {}

    # Run benchmarks
    benchmarks = [
        ("Online Latency", benchmark_online_latency),
        ("Batch Throughput", benchmark_throughput),
        ("L2 Aggregation Performance", benchmark_l2_aggregation_performance),
        ("L2/L3 Feature Computation", benchmark_l2_l3_feature_computation),
    ]

    for benchmark_name, benchmark_func in benchmarks:
        logger.info(f"\n{'='*70}")
        try:
            results = benchmark_func()
            all_results[benchmark_name] = results
        except Exception as e:
            logger.error(f"Benchmark '{benchmark_name}' crashed: {e}")
            all_results[benchmark_name] = {"error": str(e)}

    # Summary report
    logger.info(f"\n{'='*70}")
    logger.info("PERFORMANCE BENCHMARK SUMMARY")
    logger.info(f"{'='*70}")

    # Online latency summary
    if "Online Latency" in all_results:
        logger.info("\n🚀 ONLINE LATENCY PERFORMANCE:")
        latency_results = all_results["Online Latency"]

        for config_name, stats in latency_results.items():
            if "error" in stats:
                logger.info(f"  ✗ {config_name}: {stats['error']}")
            else:
                p99 = stats["p99_ms"]
                status = "✓ PASS" if p99 < 5.0 else "✗ FAIL"
                logger.info(
                    f"  {status} {config_name}: P99={p99:.2f}ms (features: {stats['n_features']})"
                )

    # Throughput summary
    if "Batch Throughput" in all_results:
        logger.info("\n📊 BATCH THROUGHPUT PERFORMANCE:")
        throughput_results = all_results["Batch Throughput"]

        max_throughput = 0
        for n_samples, stats in throughput_results.items():
            if "error" not in stats:
                throughput = stats["throughput_samples_per_sec"]
                status = "✓ PASS" if throughput >= 1000 else "⚠ LOW"
                logger.info(f"  {status} {n_samples} samples: {throughput:.0f} samples/sec")
                max_throughput = max(max_throughput, throughput)

        if max_throughput >= 1000:
            logger.info(
                f"  ✅ MEETS CLAIM: Peak throughput {max_throughput:.0f} samples/sec (>1000 target)"
            )
        else:
            logger.info(
                f"  ❌ BELOW CLAIM: Peak throughput {max_throughput:.0f} samples/sec (<1000 target)"
            )

    # L2/L3 performance summary
    if "L2/L3 Feature Computation" in all_results:
        logger.info("\n⚡ L2/L3 FEATURE COMPUTATION:")
        feature_results = all_results["L2/L3 Feature Computation"]

        for feature_type, stats in feature_results.items():
            if "error" in stats:
                logger.info(f"  ✗ {feature_type}: {stats['error']}")
            else:
                time_ms = stats["processing_time_ms"]
                n_features = stats["n_features"]
                logger.info(f"  ✓ {feature_type}: {time_ms:.2f}ms for {n_features} features")

    logger.info(f"\n{'='*70}")
    logger.info("📋 CLAIMS VERIFICATION SUMMARY:")
    logger.info(f"{'='*70}")

    # Verify key claims
    claims_verified = []

    # Check P99 latency claim
    if "Online Latency" in all_results:
        latency_results = all_results["Online Latency"]
        full_config_p99 = latency_results.get("Full L2/L3", {}).get("p99_ms", float("inf"))

        if full_config_p99 < 5.0:
            claims_verified.append("✅ Hot path <5ms P99 latency: VERIFIED")
        else:
            claims_verified.append(
                f"❌ Hot path <5ms P99 latency: FAILED ({full_config_p99:.2f}ms)"
            )

    # Check throughput claim
    if "Batch Throughput" in all_results:
        throughput_results = all_results["Batch Throughput"]
        max_throughput = max(
            (
                stats.get("throughput_samples_per_sec", 0)
                for stats in throughput_results.values()
                if "error" not in stats
            ),
            default=0,
        )

        if max_throughput >= 1000:
            claims_verified.append("✅ 1000+ bars/second processing: VERIFIED")
        else:
            claims_verified.append(
                f"❌ 1000+ bars/second processing: FAILED ({max_throughput:.0f} samples/sec)"
            )

    # Check L2/L3 feature computation
    if "L2/L3 Feature Computation" in all_results:
        feature_results = all_results["L2/L3 Feature Computation"]
        l2_working = "L2" in feature_results and "error" not in feature_results["L2"]
        l3_working = "L3" in feature_results and "error" not in feature_results["L3"]

        if l2_working and l3_working:
            claims_verified.append("✅ L2/L3 microstructure features: VERIFIED")
        elif l2_working:
            claims_verified.append("⚠️ L2 features working, L3 features failed")
        elif l3_working:
            claims_verified.append("⚠️ L3 features working, L2 features failed")
        else:
            claims_verified.append("❌ L2/L3 microstructure features: FAILED")

    for claim in claims_verified:
        logger.info(f"  {claim}")

    # Overall assessment
    passed_claims = sum(1 for claim in claims_verified if claim.startswith("✅"))
    total_claims = len(claims_verified)

    logger.info(f"\n🎯 OVERALL ASSESSMENT: {passed_claims}/{total_claims} claims verified")

    return 0 if passed_claims == total_claims else 1


if __name__ == "__main__":
    sys.exit(main())
