#!/usr/bin/env python3
"""
Fix and test L2 aggregation with proper data types.
"""

import logging
import sys

import numpy as np
import pandas as pd

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.features.l2_aggregate import aggregate_l2_minute_pl


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_l2_aggregation_fixed():
    """
    Test L2 order book aggregation with proper data types.
    """
    logger.info("=== Testing L2 Order Book Aggregation (Fixed) ===")

    if not HAS_POLARS:
        logger.error("Polars not available - skipping L2 tests")
        return False

    # Create realistic L2 order book data with proper types
    n_samples = 1000
    n_levels = 10

    # Generate synthetic MBP-10 data
    base_time = pd.Timestamp("2024-01-01 09:30:00", tz="UTC")
    timestamps = [base_time + pd.Timedelta(seconds=i) for i in range(n_samples)]

    data = {
        "ts_event": timestamps,
    }

    # Generate order book levels with consistent float types
    mid_price = 100.0
    spread = 0.01

    np.random.seed(42)  # For reproducible results

    for level in range(n_levels):
        # Bid prices decrease with level - use consistent float64
        bid_prices = [
            float(mid_price - spread / 2 - level * 0.001 + np.random.normal(0, 0.0005))
            for _ in range(n_samples)
        ]
        ask_prices = [
            float(mid_price + spread / 2 + level * 0.001 + np.random.normal(0, 0.0005))
            for _ in range(n_samples)
        ]

        # Sizes decrease with level - ensure positive integers
        bid_sizes = [
            int(max(100, 1000 * (1 - level * 0.1) + np.random.normal(0, 50)))
            for _ in range(n_samples)
        ]
        ask_sizes = [
            int(max(100, 1000 * (1 - level * 0.1) + np.random.normal(0, 50)))
            for _ in range(n_samples)
        ]

        data[f"bid_px_{level:02d}"] = bid_prices
        data[f"ask_px_{level:02d}"] = ask_prices
        data[f"bid_sz_{level:02d}"] = bid_sizes
        data[f"ask_sz_{level:02d}"] = ask_sizes

    # Create DataFrame with explicit schema
    try:
        df = pl.DataFrame(data)

        # Ensure proper datetime type
        df = df.with_columns(pl.col("ts_event").cast(pl.Datetime("ns", "UTC")))

        logger.info(f"Created L2 DataFrame with shape: {df.shape}")
        logger.info(f"Sample columns: {df.columns[:10]}")

        # Test the aggregation function
        import time

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
        ]

        # Check for depth imbalance columns
        for k in [1, 3, 5, 10]:
            expected_cols.extend(
                [
                    f"depth_imbalance_top{k}",
                    f"dwp_bps_top{k}",
                    f"bid_slope_top{k}",
                    f"ask_slope_top{k}",
                ]
            )

        missing_cols = []
        for col in expected_cols:
            if col not in result.columns:
                missing_cols.append(col)

        if missing_cols:
            logger.error(f"Missing expected columns: {missing_cols}")
            return False

        # Check data validity
        midprices = result["midprice"].to_numpy()
        spreads = result["spread_bps"].to_numpy()

        if np.any(np.isnan(midprices)):
            logger.error("NaN midprices found")
            return False

        if np.any(midprices <= 0):
            logger.error("Invalid midprices found (<=0)")
            return False

        if np.any(np.isnan(spreads)):
            logger.error("NaN spreads found")
            return False

        if np.any(spreads < 0):
            logger.error("Invalid spreads found (<0)")
            return False

        # Check depth imbalances are in valid range
        for k in [1, 3, 5, 10]:
            imbalance_col = f"depth_imbalance_top{k}"
            if imbalance_col in result.columns:
                imbalances = result[imbalance_col].to_numpy()
                if np.any(np.abs(imbalances) > 1.1):  # Allow slight numerical error
                    logger.warning(f"Depth imbalances for top{k} outside expected [-1,1] range")

        logger.info("✓ L2 order book aggregation works correctly")
        logger.info(f"  - Computed {len(result.columns)} features")
        logger.info(f"  - Processed {n_samples} samples -> {len(result)} minute aggregates")
        logger.info(f"  - Average midprice: {np.mean(midprices):.4f}")
        logger.info(f"  - Average spread (bps): {np.mean(spreads):.2f}")
        logger.info("  - Sample feature values:")

        # Show sample of computed features
        first_row = result.slice(0, 1)
        for col in ["midprice", "spread_bps", "depth_imbalance_top1", "depth_imbalance_top5"]:
            if col in first_row.columns:
                val = first_row[col][0]
                logger.info(f"    {col}: {val}")

        return True

    except Exception as e:
        logger.error(f"L2 aggregation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_microstructure_aggregation():
    """
    Test microstructure aggregation.
    """
    logger.info("=== Testing Microstructure Aggregation ===")

    if not HAS_POLARS:
        logger.error("Polars not available")
        return False

    from ml.features.micro_aggregate import aggregate_microstructure_minute_pl

    try:
        # Create synthetic quote data
        n_samples = 500
        base_time = pd.Timestamp("2024-01-01 09:30:00", tz="UTC")
        timestamps = [base_time + pd.Timedelta(seconds=i) for i in range(n_samples)]

        np.random.seed(42)

        # Generate quotes with proper types
        quotes_data = {
            "ts_event": timestamps,
            "bid_px_00": [
                float(100.0 - 0.005 + np.random.normal(0, 0.001)) for _ in range(n_samples)
            ],
            "ask_px_00": [
                float(100.0 + 0.005 + np.random.normal(0, 0.001)) for _ in range(n_samples)
            ],
            "bid_sz_00": [int(max(100, 1000 + np.random.normal(0, 100))) for _ in range(n_samples)],
            "ask_sz_00": [int(max(100, 1000 + np.random.normal(0, 100))) for _ in range(n_samples)],
        }

        quotes_df = pl.DataFrame(quotes_data)

        # Generate trades with proper types
        trades_data = {
            "ts_event": timestamps[: n_samples // 2],  # Fewer trades than quotes
            "price": [float(100.0 + np.random.normal(0, 0.01)) for _ in range(n_samples // 2)],
            "size": [int(max(1, np.random.exponential(100))) for _ in range(n_samples // 2)],
            "side": ["BUY" if np.random.random() > 0.5 else "SELL" for _ in range(n_samples // 2)],
        }

        trades_df = pl.DataFrame(trades_data)

        # Test aggregation
        import time

        start_time = time.time()
        result = aggregate_microstructure_minute_pl(quotes_df, trades_df)
        end_time = time.time()

        logger.info(f"Microstructure aggregation took {end_time - start_time:.3f}s")
        logger.info(f"Result shape: {result.shape}")

        if len(result) == 0:
            logger.error("Microstructure aggregation returned empty result")
            return False

        # Check expected columns
        expected_cols = [
            "timestamp",
            "midprice",
            "spread_bps",
            "quote_imbalance",
            "trade_imbalance",
            "realized_vol",
        ]

        for col in expected_cols:
            if col not in result.columns:
                logger.error(f"Missing column: {col}")
                return False

        # Check data validity
        midprices = result["midprice"].to_numpy()
        if np.any(midprices <= 0):
            logger.error("Invalid midprices")
            return False

        logger.info("✓ Microstructure aggregation works correctly")
        logger.info(f"  - Generated {len(result.columns)} microstructure features")
        logger.info(f"  - Average midprice: {np.mean(midprices):.4f}")

        return True

    except Exception as e:
        logger.error(f"Microstructure aggregation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """
    Run L2/microstructure aggregation tests.
    """
    logger.info("Testing L2 and microstructure aggregation...")

    tests = [
        ("L2 Aggregation Fixed", test_l2_aggregation_fixed),
        ("Microstructure Aggregation", test_microstructure_aggregation),
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
    logger.info("AGGREGATION TEST SUMMARY:")
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

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
