#!/usr/bin/env python3
"""
Comprehensive validation of L2/L3 microstructure features claims.

This script tests the actual implementation against the documented capabilities to
identify gaps between claims and reality.

"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.l2_aggregate import L2Aggregator
from ml.features.l2_aggregate import aggregate_l2_minute_pl
from ml.features.micro_aggregate import MicrostructureAggregator
from ml.features.microstructure import L2MicrostructureFeatures
from ml.features.microstructure import L3TradeFlowFeatures


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_l2_order_book_aggregation():
    """
    Test L2 order book aggregation functionality.
    """
    logger.info("=== Testing L2 Order Book Aggregation ===")

    if not HAS_POLARS:
        logger.error("Polars not available - skipping L2 tests")
        return False

    # Create realistic L2 order book data
    n_samples = 1000
    n_levels = 10

    # Generate synthetic MBP-10 data
    base_time = pd.Timestamp("2024-01-01 09:30:00", tz="UTC")
    timestamps = [base_time + pd.Timedelta(seconds=i) for i in range(n_samples)]

    data = {
        "ts_event": timestamps,
    }

    # Generate order book levels
    mid_price = 100.0
    spread = 0.01

    for level in range(n_levels):
        # Bid prices decrease with level
        bid_prices = [
            mid_price - spread / 2 - level * 0.001 + np.random.normal(0, 0.0005)
            for _ in range(n_samples)
        ]
        ask_prices = [
            mid_price + spread / 2 + level * 0.001 + np.random.normal(0, 0.0005)
            for _ in range(n_samples)
        ]

        # Sizes decrease with level
        bid_sizes = [
            max(100, 1000 * (1 - level * 0.1) + np.random.normal(0, 50)) for _ in range(n_samples)
        ]
        ask_sizes = [
            max(100, 1000 * (1 - level * 0.1) + np.random.normal(0, 50)) for _ in range(n_samples)
        ]

        data[f"bid_px_{level:02d}"] = bid_prices
        data[f"ask_px_{level:02d}"] = ask_prices
        data[f"bid_sz_{level:02d}"] = bid_sizes
        data[f"ask_sz_{level:02d}"] = ask_sizes

    # Create DataFrame
    df = pl.DataFrame(data)

    try:
        # Test the aggregation function
        start_time = time.time()
        result = aggregate_l2_minute_pl(df)
        end_time = time.time()

        logger.info(f"L2 aggregation processed {n_samples} samples in {end_time - start_time:.3f}s")
        logger.info(f"Result shape: {result.shape}")

        if len(result) == 0:
            logger.error("L2 aggregation returned empty result")
            return False

        # Check expected columns exist
        expected_cols = [
            "timestamp",
            "midprice",
            "spread_bps",
            "microprice_bps",
            "depth_imbalance_top1",
            "depth_imbalance_top3",
            "depth_imbalance_top5",
            "depth_imbalance_top10",
            "dwp_bps_top1",
            "dwp_bps_top3",
            "dwp_bps_top5",
            "dwp_bps_top10",
            "bid_slope_top1",
            "bid_slope_top3",
            "bid_slope_top5",
            "bid_slope_top10",
            "ask_slope_top1",
            "ask_slope_top3",
            "ask_slope_top5",
            "ask_slope_top10",
        ]

        for col in expected_cols:
            if col not in result.columns:
                logger.error(f"Missing expected column: {col}")
                return False

        # Check data validity
        midprices = result["midprice"].to_numpy()
        spreads = result["spread_bps"].to_numpy()

        if np.any(midprices <= 0):
            logger.error("Invalid midprices found (<=0)")
            return False

        if np.any(spreads < 0):
            logger.error("Invalid spreads found (<0)")
            return False

        logger.info("✓ L2 order book aggregation works correctly")
        logger.info(f"  - Computed {len(expected_cols)} features")
        logger.info(f"  - Processed {n_samples} samples -> {len(result)} minute aggregates")
        logger.info(f"  - Average midprice: {np.mean(midprices):.4f}")
        logger.info(f"  - Average spread (bps): {np.mean(spreads):.2f}")

        return True

    except Exception as e:
        logger.error(f"L2 aggregation failed: {e}")
        return False


def test_l2_microstructure_features():
    """
    Test L2 microstructure feature computation.
    """
    logger.info("=== Testing L2 Microstructure Features ===")

    try:
        calculator = L2MicrostructureFeatures(n_levels=10, lookback_window=20)

        # Create realistic order book data
        n_samples = 100
        n_levels = 10

        mid_price = 100.0
        spread = 0.01

        bid_prices = np.zeros((n_samples, n_levels))
        ask_prices = np.zeros((n_samples, n_levels))
        bid_sizes = np.zeros((n_samples, n_levels))
        ask_sizes = np.zeros((n_samples, n_levels))

        for i in range(n_samples):
            # Add price movement
            mid_price += np.random.normal(0, 0.005)

            for level in range(n_levels):
                bid_prices[i, level] = mid_price - spread / 2 - level * 0.001
                ask_prices[i, level] = mid_price + spread / 2 + level * 0.001
                bid_sizes[i, level] = max(10, 1000 * (1 - level * 0.15) + np.random.normal(0, 50))
                ask_sizes[i, level] = max(10, 1000 * (1 - level * 0.15) + np.random.normal(0, 50))

        # Test individual feature groups
        start_time = time.time()

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

        end_time = time.time()

        total_features = (
            len(spread_features)
            + len(imbalance_features)
            + len(depth_features)
            + len(shape_features)
        )

        logger.info(f"L2 microstructure features computed in {end_time - start_time:.4f}s")
        logger.info(f"✓ Computed {total_features} L2 microstructure features:")
        logger.info(f"  - Spread features: {len(spread_features)}")
        logger.info(f"  - Imbalance features: {len(imbalance_features)}")
        logger.info(f"  - Depth features: {len(depth_features)}")
        logger.info(f"  - Shape features: {len(shape_features)}")

        # Validate feature values
        assert spread_features["spread"] > 0, "Spread should be positive"
        assert spread_features["spread_bps"] > 0, "Spread BPS should be positive"
        assert -1 <= imbalance_features["imbalance_l1"] <= 1, "Imbalance should be normalized"
        assert depth_features["bid_depth_total"] > 0, "Bid depth should be positive"
        assert depth_features["ask_depth_total"] > 0, "Ask depth should be positive"

        return True

    except Exception as e:
        logger.error(f"L2 microstructure features failed: {e}")
        return False


def test_l3_trade_flow_features():
    """
    Test L3 trade flow feature computation.
    """
    logger.info("=== Testing L3 Trade Flow Features ===")

    try:
        calculator = L3TradeFlowFeatures(lookback_window=100)

        # Create realistic trade data
        n_trades = 500

        # Generate trade data
        prices = []
        volumes = []
        sides = []
        timestamps = []

        base_price = 100.0
        base_time = (
            pd.Timestamp("2024-01-01 09:30:00", tz="UTC").value // 1_000_000_000
        )  # nanoseconds

        for i in range(n_trades):
            # Random price movement
            base_price += np.random.normal(0, 0.01)
            prices.append(base_price)

            # Random volume
            volumes.append(max(1, np.random.exponential(100)))

            # Random side (1 for buy, -1 for sell)
            sides.append(1 if np.random.random() > 0.5 else -1)

            # Sequential timestamps
            timestamps.append(base_time + i * 1_000_000_000)  # 1 second apart

        prices = np.array(prices)
        volumes = np.array(volumes)
        sides = np.array(sides)
        timestamps = np.array(timestamps)

        # Test individual feature groups
        start_time = time.time()

        imbalance_features = calculator.compute_trade_imbalance(prices, volumes, sides)
        vwap_features = calculator.compute_vwap_features(prices, volumes, sides)
        intensity_features = calculator.compute_intensity_features(timestamps, volumes, prices)
        impact_features = calculator.compute_price_impact(prices, volumes, sides)

        end_time = time.time()

        total_features = (
            len(imbalance_features)
            + len(vwap_features)
            + len(intensity_features)
            + len(impact_features)
        )

        logger.info(f"L3 trade flow features computed in {end_time - start_time:.4f}s")
        logger.info(f"✓ Computed {total_features} L3 trade flow features:")
        logger.info(f"  - Trade imbalance features: {len(imbalance_features)}")
        logger.info(f"  - VWAP features: {len(vwap_features)}")
        logger.info(f"  - Intensity features: {len(intensity_features)}")
        logger.info(f"  - Price impact features: {len(impact_features)}")

        # Validate feature values
        assert (
            -1 <= imbalance_features["trade_imbalance"] <= 1
        ), "Trade imbalance should be normalized"
        assert vwap_features["vwap"] > 0, "VWAP should be positive"
        assert intensity_features["trade_rate"] >= 0, "Trade rate should be non-negative"
        assert impact_features["avg_price_impact"] >= 0, "Price impact should be non-negative"

        return True

    except Exception as e:
        logger.error(f"L3 trade flow features failed: {e}")
        return False


def test_feature_engineer_integration():
    """
    Test FeatureEngineer integration with L2/L3 features.
    """
    logger.info("=== Testing FeatureEngineer L2/L3 Integration ===")

    try:
        # Test with microstructure features enabled
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=True,
            validate_quality=True,
        )

        engineer = FeatureEngineer(config)

        # Create test data with L1 OHLCV (fallback case)
        n_samples = 100
        data = {
            "open": [100 + i * 0.1 + np.random.normal(0, 0.5) for i in range(n_samples)],
            "high": [101 + i * 0.1 + np.random.normal(0, 0.5) for i in range(n_samples)],
            "low": [99 + i * 0.1 + np.random.normal(0, 0.5) for i in range(n_samples)],
            "close": [100.5 + i * 0.1 + np.random.normal(0, 0.5) for i in range(n_samples)],
            "volume": [1000 + np.random.exponential(500) for _ in range(n_samples)],
            "ts_event": pd.date_range(
                "2024-01-01 09:30:00", periods=n_samples, freq="1min", tz="UTC"
            ),
            "ts_init": pd.date_range(
                "2024-01-01 09:30:00", periods=n_samples, freq="1min", tz="UTC"
            ),
        }

        if HAS_POLARS:
            df = pl.DataFrame(data)
        else:
            df = pd.DataFrame(data)

        # Test batch processing
        start_time = time.time()
        features_df, scaler = engineer.calculate_features(df, mode="batch", fit_scaler=True)
        batch_time = time.time() - start_time

        logger.info(f"Batch feature computation: {batch_time:.3f}s for {n_samples} samples")
        logger.info(f"Feature shape: {features_df.shape}")

        # Get feature names
        feature_names = engineer.get_feature_names()
        logger.info(f"✓ Generated {len(feature_names)} features")

        # Check for microstructure features
        microstructure_features = [
            name
            for name in feature_names
            if any(term in name.lower() for term in ["spread", "imbalance", "vwap", "flow"])
        ]
        logger.info(f"  - Microstructure/trade flow features: {len(microstructure_features)}")

        if microstructure_features:
            logger.info(f"  - Sample microstructure features: {microstructure_features[:5]}")

        # Test online processing
        from ml.features.engineering import IndicatorManager

        indicator_mgr = IndicatorManager(config)

        # Warm up indicators
        for i in range(min(50, len(df))):
            if HAS_POLARS:
                row = df.slice(i, 1)
                indicator_mgr.update_from_values(
                    close=float(row["close"][0]),
                    high=float(row["high"][0]),
                    low=float(row["low"][0]),
                    volume=float(row["volume"][0]),
                )
            else:
                row = df.iloc[i]
                indicator_mgr.update_from_values(
                    close=float(row["close"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    volume=float(row["volume"]),
                )

        # Test online feature computation
        start_time = time.time()

        if HAS_POLARS:
            last_row = df.slice(-1, 1)
            online_features = engineer.calculate_features_online(
                close_price=float(last_row["close"][0]),
                high_price=float(last_row["high"][0]),
                low_price=float(last_row["low"][0]),
                volume=float(last_row["volume"][0]),
                scaler=scaler,
            )
        else:
            last_row = df.iloc[-1]
            online_features = engineer.calculate_features_online(
                close_price=float(last_row["close"]),
                high_price=float(last_row["high"]),
                low_price=float(last_row["low"]),
                volume=float(last_row["volume"]),
                scaler=scaler,
            )

        online_time = time.time() - start_time

        logger.info(f"Online feature computation: {online_time*1000:.2f}ms")
        logger.info(f"Online features shape: {online_features.shape}")

        # Check latency requirement
        latency_ms = online_time * 1000
        if latency_ms > 5.0:
            logger.warning(f"⚠ Online latency {latency_ms:.2f}ms exceeds 5ms target")
        else:
            logger.info(f"✓ Online latency {latency_ms:.2f}ms meets <5ms requirement")

        return True

    except Exception as e:
        logger.error(f"FeatureEngineer integration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_fallback_behavior():
    """
    Test fallback behavior when only L1 data is available.
    """
    logger.info("=== Testing Fallback Behavior (L1 only) ===")

    try:
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=True,
        )

        engineer = FeatureEngineer(config)

        # Create L1-only data (no L2 depth, no L3 trades)
        n_samples = 50
        data = {
            "open": [100 + i * 0.1 for i in range(n_samples)],
            "high": [101 + i * 0.1 for i in range(n_samples)],
            "low": [99 + i * 0.1 for i in range(n_samples)],
            "close": [100.5 + i * 0.1 for i in range(n_samples)],
            "volume": [1000 + i * 10 for i in range(n_samples)],
            "ts_event": pd.date_range(
                "2024-01-01 09:30:00", periods=n_samples, freq="1min", tz="UTC"
            ),
            "ts_init": pd.date_range(
                "2024-01-01 09:30:00", periods=n_samples, freq="1min", tz="UTC"
            ),
        }

        if HAS_POLARS:
            df = pl.DataFrame(data)
        else:
            df = pd.DataFrame(data)

        # Test that it still computes features
        features_df, _ = engineer.calculate_features(df, mode="batch")

        logger.info(
            f"✓ Fallback behavior works - computed {features_df.shape[1]} features from L1-only data"
        )

        feature_names = engineer.get_feature_names()
        microstructure_features = [
            name
            for name in feature_names
            if any(term in name.lower() for term in ["spread", "imbalance", "vwap", "flow"])
        ]

        if microstructure_features:
            logger.info(
                f"  - Generated {len(microstructure_features)} microstructure approximations"
            )
        else:
            logger.warning("  - No microstructure features generated in fallback mode")

        return True

    except Exception as e:
        logger.error(f"Fallback behavior test failed: {e}")
        return False


def test_data_ingestion_integration():
    """
    Test claimed data ingestion capabilities.
    """
    logger.info("=== Testing Data Ingestion Integration ===")

    try:
        # Test L2 aggregator (would work if data files existed)
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            aggregator = L2Aggregator(base_dir=base_dir)

            # Try to compute for a symbol (should handle missing data gracefully)
            result = aggregator.compute_for_symbol("TEST_SYMBOL")

            if len(result) == 0:
                logger.info("✓ L2Aggregator handles missing data gracefully")
            else:
                logger.info(f"✓ L2Aggregator returned {len(result)} rows")

            # Test micro aggregator
            micro_aggregator = MicrostructureAggregator(base_dir=base_dir)
            result = micro_aggregator.compute_for_symbol("TEST_SYMBOL")

            if len(result) == 0:
                logger.info("✓ MicrostructureAggregator handles missing data gracefully")
            else:
                logger.info(f"✓ MicrostructureAggregator returned {len(result)} rows")

        logger.info("✓ Data ingestion integration classes exist and handle edge cases")
        return True

    except Exception as e:
        logger.error(f"Data ingestion integration test failed: {e}")
        return False


def main():
    """
    Run all microstructure validation tests.
    """
    logger.info("Starting comprehensive L2/L3 microstructure validation...")

    # Check dependencies
    if not HAS_POLARS:
        logger.warning("Polars not available - some tests may be skipped")

    tests = [
        ("L2 Order Book Aggregation", test_l2_order_book_aggregation),
        ("L2 Microstructure Features", test_l2_microstructure_features),
        ("L3 Trade Flow Features", test_l3_trade_flow_features),
        ("FeatureEngineer Integration", test_feature_engineer_integration),
        ("Fallback Behavior", test_fallback_behavior),
        ("Data Ingestion Integration", test_data_ingestion_integration),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION SUMMARY:")
    logger.info(f"{'='*60}")

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:10} {test_name}")
        if result:
            passed += 1

    logger.info(f"{'='*60}")
    logger.info(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        logger.info("🎉 All microstructure validation tests PASSED!")
        return 0
    else:
        logger.warning(
            f"⚠ {total-passed} tests FAILED - gaps identified between claims and implementation"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
