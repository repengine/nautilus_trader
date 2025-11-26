"""
Integration tests for FeatureMetricsCollector component.

Tests all 4 methods working together with realistic data to ensure proper composition
and end-to-end functionality.
"""

import numpy as np
import polars as pl
import pytest

from ml.features.common.feature_metrics_collector import FeatureMetricsCollector


@pytest.mark.integration
def test_metrics_collector_integration(
    realistic_column_data: pl.Series,
    realistic_spread_data: dict[str, np.ndarray],
    realistic_trade_data: dict[str, np.ndarray],
) -> None:
    """
    Test all 4 methods work together with realistic data.

    Verify:
    - Column metrics: reasonable quality metrics for realistic data
    - Spread metrics: valid L2 spreads and imbalances
    - Trade metrics: valid trade flow and VWAP
    - All methods execute without exceptions
    - Results have correct types and structure
    """
    collector = FeatureMetricsCollector()

    # ========================================
    # Test 1: Column metrics with realistic data
    # ========================================
    col_metrics = collector._calculate_column_metrics(
        realistic_column_data, total_rows=len(realistic_column_data)
    )

    # Verify structure
    assert "null_rate" in col_metrics
    assert "zero_rate" in col_metrics
    assert "unique_ratio" in col_metrics
    assert "inf_rate" in col_metrics
    assert "outlier_rate" in col_metrics

    # Verify reasonable ranges
    assert 0.0 <= col_metrics["null_rate"] <= 1.0
    assert 0.0 <= col_metrics["zero_rate"] <= 1.0
    assert 0.0 <= col_metrics["unique_ratio"] <= 1.0
    assert 0.0 <= col_metrics["inf_rate"] <= 1.0
    assert 0.0 <= col_metrics["outlier_rate"] <= 1.0

    # For realistic data with ~3% nulls and some outliers
    assert col_metrics["null_rate"] < 0.1  # Less than 10% nulls
    assert col_metrics["outlier_rate"] < 0.1  # Less than 10% outliers

    # ========================================
    # Test 2: Spread metrics with realistic L2 data
    # ========================================
    spreads, rel_spreads, imbalances, mid_prices = collector._calculate_spread_metrics(
        realistic_spread_data["bid_prices"],
        realistic_spread_data["ask_prices"],
        realistic_spread_data["bid_sizes"],
        realistic_spread_data["ask_sizes"],
        start_idx=0,
        end_idx=len(realistic_spread_data["bid_prices"]) - 1,
    )

    # Verify structure
    assert len(spreads) > 0
    assert len(spreads) == len(rel_spreads)
    assert len(spreads) == len(imbalances)
    assert len(spreads) == len(mid_prices)

    # Verify all spreads are non-negative
    assert all(s >= 0 for s in spreads)
    assert all(rs >= 0 for rs in rel_spreads)

    # Verify imbalances in valid range
    assert all(-1.0 <= imb <= 1.0 for imb in imbalances)

    # Verify mid prices are reasonable
    assert all(m > 0 for m in mid_prices)

    # For liquid markets, relative spreads should be small (< 1%)
    assert all(rs < 0.01 for rs in rel_spreads)

    # ========================================
    # Test 3: Trade metrics with realistic trade data
    # ========================================
    flow_imb, vwap, intensity, impact, had_trades = collector._calculate_trade_metrics(
        realistic_trade_data["trade_prices"],
        realistic_trade_data["trade_volumes"],
        realistic_trade_data["trade_sides"],
        start_idx=0,
        end_idx=len(realistic_trade_data["trade_prices"]) - 1,
    )

    # Verify had_trades is True for realistic data
    assert had_trades is True

    # Verify flow imbalance in valid range
    assert -1.0 <= flow_imb <= 1.0

    # Verify VWAP is within price range
    min_price = realistic_trade_data["trade_prices"].min()
    max_price = realistic_trade_data["trade_prices"].max()
    assert min_price <= vwap <= max_price

    # Verify trade intensity is reasonable
    assert 0.0 <= intensity <= 5.0

    # Verify price impact is non-negative
    assert impact >= 0.0

    # ========================================
    # Test 4: Verify all methods compose correctly
    # ========================================
    # All methods should be callable without errors and return expected types
    assert isinstance(col_metrics, dict)
    assert isinstance(spreads, list)
    assert isinstance(flow_imb, float)
    assert isinstance(vwap, float)
    assert isinstance(had_trades, bool)
