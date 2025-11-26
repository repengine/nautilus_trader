"""
Unit tests for FeatureMetricsCollector component.

Tests the 4 metrics calculation methods extracted from FeatureEngineer:
- _calculate_column_metrics (quality metrics for data columns)
- _calculate_outlier_rate (IQR-based outlier detection)
- _calculate_spread_metrics (L2 spread and imbalance metrics)
- _calculate_trade_metrics (trade flow and VWAP metrics)
"""

import numpy as np
import polars as pl
import pytest

from ml.features.common.feature_metrics_collector import FeatureMetricsCollector


# ================================================================================================
# Test Class: _calculate_column_metrics
# ================================================================================================


class TestCalculateColumnMetrics:
    """Tests for _calculate_column_metrics method."""

    def test_calculate_column_metrics_basic(self) -> None:
        """
        Test basic quality metrics calculation for normal float column.

        Verify that all 5 metric keys are present and rates are in [0, 1].
        """
        collector = FeatureMetricsCollector()
        col_data = pl.Series(
            "test_col", [float(i) for i in range(1, 101)], dtype=pl.Float64
        )
        total_rows = 100

        metrics = collector._calculate_column_metrics(col_data, total_rows)

        # Assert all required keys present
        assert "null_rate" in metrics
        assert "zero_rate" in metrics
        assert "unique_ratio" in metrics
        assert "inf_rate" in metrics
        assert "outlier_rate" in metrics

        # Assert basic metrics for sequential data
        assert metrics["null_rate"] == 0.0  # No nulls
        assert metrics["zero_rate"] == 0.0  # No zeros (starts at 1)
        assert metrics["unique_ratio"] == 1.0  # All unique
        assert metrics["inf_rate"] == 0.0  # No infinities
        assert 0.0 <= metrics["outlier_rate"] <= 1.0  # Outlier rate in valid range

    def test_calculate_column_metrics_with_nulls(self) -> None:
        """
        Test null_rate calculation with missing values.

        Verify that null_rate correctly reflects 40% nulls.
        """
        collector = FeatureMetricsCollector()
        col_data = pl.Series(
            "col_nulls",
            [1.0, None, 3.0, None, 5.0, None, 7.0, None, 9.0, 10.0],
            dtype=pl.Float64,
        )
        total_rows = 10

        metrics = collector._calculate_column_metrics(col_data, total_rows)

        assert metrics["null_rate"] == 0.4  # 4 out of 10 nulls
        assert metrics["zero_rate"] >= 0.0  # Non-negative
        assert metrics["unique_ratio"] > 0.0  # Has unique values

    def test_calculate_column_metrics_with_zeros(self) -> None:
        """
        Test zero_rate calculation with many zero values.

        Verify that zero_rate correctly reflects 70% zeros.
        """
        collector = FeatureMetricsCollector()
        values = [0.0] * 70 + [float(i) for i in range(1, 31)]
        col_data = pl.Series("col_zeros", values, dtype=pl.Float64)
        total_rows = 100

        metrics = collector._calculate_column_metrics(col_data, total_rows)

        assert metrics["zero_rate"] == 0.7  # 70 out of 100 zeros
        assert metrics["unique_ratio"] < 1.0  # Not all unique (repeated zeros)

    def test_calculate_column_metrics_with_infs(self) -> None:
        """
        Test inf_rate calculation for float columns with infinities.

        Verify that inf_rate correctly detects positive and negative infinities.
        """
        collector = FeatureMetricsCollector()
        col_data = pl.Series(
            "col_infs",
            [1.0, 2.0, float("inf"), 4.0, float("-inf"), 6.0, 7.0, 8.0, 9.0, 10.0],
            dtype=pl.Float64,
        )
        total_rows = 10

        metrics = collector._calculate_column_metrics(col_data, total_rows)

        assert metrics["inf_rate"] == 0.2  # 2 out of 10 infinities

    def test_calculate_column_metrics_empty(self) -> None:
        """
        Test defensive behavior with empty series.

        Verify that all rates return 0.0 for empty data.
        """
        collector = FeatureMetricsCollector()
        col_data = pl.Series("empty", [], dtype=pl.Float64)
        total_rows = 0

        metrics = collector._calculate_column_metrics(col_data, total_rows)

        assert metrics["null_rate"] == 0.0
        assert metrics["zero_rate"] == 0.0
        assert metrics["unique_ratio"] == 0.0
        assert metrics["inf_rate"] == 0.0
        assert metrics["outlier_rate"] == 0.0


# ================================================================================================
# Test Class: _calculate_outlier_rate
# ================================================================================================


class TestCalculateOutlierRate:
    """Tests for _calculate_outlier_rate method."""

    def test_calculate_outlier_rate_normal(self) -> None:
        """
        Test IQR outlier detection with normal distribution + extreme outliers.

        Verify that the 5 extreme values are detected as outliers.
        """
        collector = FeatureMetricsCollector()

        # Generate normal distribution with fixed seed
        np.random.seed(42)
        normal_vals = np.random.normal(50, 10, 95).tolist()
        # Add extreme outliers
        outliers = [0.0, 0.0, 100.0, 100.0, 100.0]
        col_data = pl.Series("values", normal_vals + outliers, dtype=pl.Float64)
        total_rows = 100

        outlier_rate = collector._calculate_outlier_rate(col_data, total_rows)

        # Should detect ~5% outliers (allow tolerance for edge cases)
        assert 0.04 <= outlier_rate <= 0.06

    def test_calculate_outlier_rate_no_outliers(self) -> None:
        """
        Test zero outlier rate when all values in IQR range.

        Verify that a tight distribution has no outliers.
        """
        collector = FeatureMetricsCollector()
        col_data = pl.Series(
            "tight", [50.0 + i * 0.1 for i in range(10)], dtype=pl.Float64
        )
        total_rows = 10

        outlier_rate = collector._calculate_outlier_rate(col_data, total_rows)

        assert outlier_rate == 0.0

    def test_calculate_outlier_rate_edge_cases(self) -> None:
        """
        Test defensive handling of edge cases.

        Test 3 sub-cases:
        1. All same value (zero IQR) → 0.0
        2. Series with NaNs → defensive handling
        3. Empty series → 0.0
        """
        collector = FeatureMetricsCollector()

        # Sub-case 1: All same value (zero IQR)
        col_uniform = pl.Series("uniform", [5.0] * 10, dtype=pl.Float64)
        outlier_rate = collector._calculate_outlier_rate(col_uniform, 10)
        assert outlier_rate == 0.0

        # Sub-case 2: With NaNs (polars should handle gracefully)
        col_nans = pl.Series(
            "nans",
            [1.0, float("nan"), 3.0, float("nan"), 5.0],
            dtype=pl.Float64,
        )
        outlier_rate_nans = collector._calculate_outlier_rate(col_nans, 5)
        assert 0.0 <= outlier_rate_nans <= 1.0  # Should be defensive

        # Sub-case 3: Empty series
        col_empty = pl.Series("empty", [], dtype=pl.Float64)
        outlier_rate_empty = collector._calculate_outlier_rate(col_empty, 0)
        assert outlier_rate_empty == 0.0


# ================================================================================================
# Test Class: _calculate_spread_metrics
# ================================================================================================


class TestCalculateSpreadMetrics:
    """Tests for _calculate_spread_metrics method."""

    def test_calculate_spread_metrics_normal(self) -> None:
        """
        Test spread calculations with valid bid/ask data.

        Verify spreads, relative spreads, size imbalances, and mid prices.
        """
        collector = FeatureMetricsCollector()
        bid_prices = np.array([99.0, 99.5, 100.0, 100.5, 101.0])
        ask_prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        bid_sizes = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
        ask_sizes = np.array([80.0, 120.0, 180.0, 230.0, 270.0])
        start_idx = 0
        end_idx = 4

        spreads, relative_spreads, size_imbalances, mid_prices = (
            collector._calculate_spread_metrics(
                bid_prices, ask_prices, bid_sizes, ask_sizes, start_idx, end_idx
            )
        )

        # Verify lengths
        assert len(spreads) == 5
        assert len(relative_spreads) == 5
        assert len(size_imbalances) == 5
        assert len(mid_prices) == 5

        # Verify spreads (all should be 1.0)
        assert all(s == 1.0 for s in spreads)

        # Verify mid prices
        expected_mid_prices = [99.5, 100.0, 100.5, 101.0, 101.5]
        assert np.allclose(mid_prices, expected_mid_prices)

        # Verify relative spreads
        expected_rel_spreads = [1.0 / m for m in expected_mid_prices]
        assert np.allclose(relative_spreads, expected_rel_spreads)

        # Verify size imbalances are in valid range
        assert all(-1.0 <= imb <= 1.0 for imb in size_imbalances)

    def test_calculate_spread_metrics_invalid_bid_ask(self) -> None:
        """
        Test defensive behavior when bid > ask (invalid market data).

        Verify that invalid ticks are skipped.
        """
        collector = FeatureMetricsCollector()
        bid_prices = np.array([101.0, 99.0, 102.0])  # bid > ask at indices 0 and 2
        ask_prices = np.array([100.0, 100.0, 101.0])
        bid_sizes = np.array([100.0, 100.0, 100.0])
        ask_sizes = np.array([100.0, 100.0, 100.0])
        start_idx = 0
        end_idx = 2

        spreads, relative_spreads, _size_imbalances, mid_prices = (
            collector._calculate_spread_metrics(
                bid_prices, ask_prices, bid_sizes, ask_sizes, start_idx, end_idx
            )
        )

        # Only index 1 is valid (bid=99 < ask=100)
        assert len(spreads) == 1
        assert spreads[0] == 1.0
        assert len(relative_spreads) == 1
        assert len(mid_prices) == 1
        assert mid_prices[0] == 99.5

    def test_calculate_spread_metrics_zero_prices(self) -> None:
        """
        Test defensive behavior with zero bid or ask prices.

        Verify that zero prices are skipped.
        """
        collector = FeatureMetricsCollector()
        bid_prices = np.array([0.0, 99.0, 100.0])
        ask_prices = np.array([100.0, 0.0, 101.0])
        bid_sizes = np.array([100.0, 100.0, 100.0])
        ask_sizes = np.array([100.0, 100.0, 100.0])
        start_idx = 0
        end_idx = 2

        spreads, _relative_spreads, _size_imbalances, mid_prices = (
            collector._calculate_spread_metrics(
                bid_prices, ask_prices, bid_sizes, ask_sizes, start_idx, end_idx
            )
        )

        # Only index 2 is valid
        assert len(spreads) == 1
        assert spreads[0] == 1.0
        assert mid_prices[0] == 100.5

    def test_calculate_spread_metrics_zero_sizes(self) -> None:
        """
        Test size_imbalance calculation when bid/ask sizes are zero.

        Verify that zero total_size returns imbalance of 0.0.
        """
        collector = FeatureMetricsCollector()
        bid_prices = np.array([99.0, 99.0, 99.0])
        ask_prices = np.array([100.0, 100.0, 100.0])
        bid_sizes = np.array([0.0, 100.0, 0.0])
        ask_sizes = np.array([0.0, 100.0, 0.0])
        start_idx = 0
        end_idx = 2

        spreads, _relative_spreads, size_imbalances, _mid_prices = (
            collector._calculate_spread_metrics(
                bid_prices, ask_prices, bid_sizes, ask_sizes, start_idx, end_idx
            )
        )

        assert len(spreads) == 3
        assert all(s == 1.0 for s in spreads)
        # Indices 0 and 2: total_size=0 → imbalance=0.0
        # Index 1: total_size=200 → imbalance=0.0 (balanced)
        assert size_imbalances == [0.0, 0.0, 0.0]

    def test_calculate_spread_metrics_empty_window(self) -> None:
        """
        Test behavior when window is empty (start_idx > end_idx).

        Verify that all return lists are empty.
        """
        collector = FeatureMetricsCollector()
        bid_prices = np.array([99.0, 99.5, 100.0])
        ask_prices = np.array([100.0, 100.5, 101.0])
        bid_sizes = np.array([100.0, 100.0, 100.0])
        ask_sizes = np.array([100.0, 100.0, 100.0])
        start_idx = 2
        end_idx = 1  # Empty range

        spreads, relative_spreads, size_imbalances, mid_prices = (
            collector._calculate_spread_metrics(
                bid_prices, ask_prices, bid_sizes, ask_sizes, start_idx, end_idx
            )
        )

        assert spreads == []
        assert relative_spreads == []
        assert size_imbalances == []
        assert mid_prices == []


# ================================================================================================
# Test Class: _calculate_trade_metrics
# ================================================================================================


class TestCalculateTradeMetrics:
    """Tests for _calculate_trade_metrics method."""

    def test_calculate_trade_metrics_normal(self) -> None:
        """
        Test trade metrics with mixed buy/sell trades.

        Verify flow imbalance, VWAP, intensity, price impact calculations.
        """
        collector = FeatureMetricsCollector()
        trade_prices = np.array([100.0, 100.5, 99.5, 101.0, 100.0])
        trade_volumes = np.array([10.0, 20.0, 15.0, 25.0, 30.0])
        trade_sides = np.array([1.0, 1.0, -1.0, 1.0, -1.0])  # Buy, Buy, Sell, Buy, Sell
        start_idx = 0
        end_idx = 4

        (
            trade_flow_imbalance,
            vwap,
            trade_intensity,
            avg_price_impact,
            had_trades,
        ) = collector._calculate_trade_metrics(
            trade_prices, trade_volumes, trade_sides, start_idx, end_idx
        )

        # Buy volume = 10 + 20 + 25 = 55
        # Sell volume = 15 + 30 = 45
        # Total volume = 100
        # Flow imbalance = (55 - 45) / 100 = 0.1
        assert np.isclose(trade_flow_imbalance, 0.1)

        # VWAP = (100*10 + 100.5*20 + 99.5*15 + 101*25 + 100*30) / 100
        expected_vwap = (1000 + 2010 + 1492.5 + 2525 + 3000) / 100
        assert np.isclose(vwap, expected_vwap)

        # Trade count = 5, intensity = min(5 / 20.0, 5.0) = 0.25
        assert np.isclose(trade_intensity, 0.25)

        # Should have some price impact
        assert avg_price_impact > 0.0
        assert had_trades is True

    def test_calculate_trade_metrics_no_trades(self) -> None:
        """
        Test defensive behavior when no valid trades (zero volumes).

        Verify default values for all metrics.
        """
        collector = FeatureMetricsCollector()
        trade_prices = np.array([100.0, 100.0, 100.0])
        trade_volumes = np.array([0.0, 0.0, 0.0])  # All zero
        trade_sides = np.array([1.0, -1.0, 1.0])
        start_idx = 0
        end_idx = 2

        (
            trade_flow_imbalance,
            vwap,
            _trade_intensity,
            _avg_price_impact,
            had_trades,
        ) = collector._calculate_trade_metrics(
            trade_prices, trade_volumes, trade_sides, start_idx, end_idx
        )

        assert trade_flow_imbalance == 0.0
        assert vwap == 0.0
        # trade_intensity would be 1.0 (default when no trades)
        # avg_price_impact would be 0.0
        assert had_trades is False

    def test_calculate_trade_metrics_buy_only(self) -> None:
        """
        Test metrics when all trades are buys.

        Verify flow imbalance = 1.0 (maximum buy pressure).
        """
        collector = FeatureMetricsCollector()
        trade_prices = np.array([100.0, 101.0, 102.0])
        trade_volumes = np.array([10.0, 20.0, 30.0])
        trade_sides = np.array([1.0, 1.0, 1.0])  # All buys
        start_idx = 0
        end_idx = 2

        (
            trade_flow_imbalance,
            vwap,
            _trade_intensity,
            _avg_price_impact,
            had_trades,
        ) = collector._calculate_trade_metrics(
            trade_prices, trade_volumes, trade_sides, start_idx, end_idx
        )

        # Buy volume = 60, sell volume = 0
        # Flow imbalance = (60 - 0) / 60 = 1.0
        assert trade_flow_imbalance == 1.0

        # VWAP = (100*10 + 101*20 + 102*30) / 60 = 6080 / 60
        expected_vwap = 6080.0 / 60.0
        assert np.isclose(vwap, expected_vwap, rtol=1e-3)
        assert had_trades is True

    def test_calculate_trade_metrics_sell_only(self) -> None:
        """
        Test metrics when all trades are sells.

        Verify flow imbalance = -1.0 (maximum sell pressure).
        """
        collector = FeatureMetricsCollector()
        trade_prices = np.array([100.0, 99.0, 98.0])
        trade_volumes = np.array([10.0, 20.0, 30.0])
        trade_sides = np.array([-1.0, -1.0, -1.0])  # All sells
        start_idx = 0
        end_idx = 2

        (
            trade_flow_imbalance,
            vwap,
            _trade_intensity,
            _avg_price_impact,
            had_trades,
        ) = collector._calculate_trade_metrics(
            trade_prices, trade_volumes, trade_sides, start_idx, end_idx
        )

        # Buy volume = 0, sell volume = 60
        # Flow imbalance = (0 - 60) / 60 = -1.0
        assert trade_flow_imbalance == -1.0

        # VWAP = (100*10 + 99*20 + 98*30) / 60 = 5920 / 60
        expected_vwap = 5920.0 / 60.0
        assert np.isclose(vwap, expected_vwap, rtol=1e-3)
        assert had_trades is True

    def test_calculate_trade_metrics_empty_window(self) -> None:
        """
        Test behavior when window is empty.

        Verify all metrics default to zero/baseline.
        """
        collector = FeatureMetricsCollector()
        trade_prices = np.array([100.0, 101.0, 102.0])
        trade_volumes = np.array([10.0, 20.0, 30.0])
        trade_sides = np.array([1.0, 1.0, 1.0])
        start_idx = 2
        end_idx = 1  # Empty range

        (
            trade_flow_imbalance,
            vwap,
            _trade_intensity,
            _avg_price_impact,
            had_trades,
        ) = collector._calculate_trade_metrics(
            trade_prices, trade_volumes, trade_sides, start_idx, end_idx
        )

        assert trade_flow_imbalance == 0.0
        assert vwap == 0.0
        # trade_intensity would be 1.0 (default when no trades)
        # avg_price_impact would be 0.0
        assert had_trades is False
