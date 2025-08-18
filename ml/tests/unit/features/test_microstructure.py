"""
Unit tests for L2/L3 microstructure feature engineering.

Tests order book depth and trade flow feature extraction from market data.

"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml._imports import HAS_PANDAS
from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.features.microstructure import L2MicrostructureFeatures
from ml.features.microstructure import L3TradeFlowFeatures


pytestmark = pytest.mark.skipif(
    not HAS_POLARS or not HAS_PANDAS,
    reason="Requires polars and pandas",
)


class TestL2MicrostructureFeatures:
    """Test L2 order book microstructure features."""

    def test_spread_features_basic(self) -> None:
        """Test basic spread feature calculation."""
        calculator = L2MicrostructureFeatures(n_levels=5, lookback_window=5)

        # Create sample order book data
        bid_prices = np.array([
            [99.5, 99.4, 99.3, 99.2, 99.1],
            [99.6, 99.5, 99.4, 99.3, 99.2],
        ])
        ask_prices = np.array([
            [100.5, 100.6, 100.7, 100.8, 100.9],
            [100.4, 100.5, 100.6, 100.7, 100.8],
        ])
        bid_sizes = np.array([
            [100, 200, 300, 400, 500],
            [150, 250, 350, 450, 550],
        ])
        ask_sizes = np.array([
            [100, 200, 300, 400, 500],
            [150, 250, 350, 450, 550],
        ])

        features = calculator.compute_spread_features(
            bid_prices, ask_prices, bid_sizes, ask_sizes
        )

        # Check spread calculation
        assert abs(features["spread"] - 0.8) < 1e-10  # 100.4 - 99.6
        assert features["spread_bps"] > 0  # Should be positive
        assert "spread_mean" in features
        assert "spread_std" in features
        assert "weighted_spread" in features

    def test_imbalance_features_basic(self) -> None:
        """Test order book imbalance features."""
        calculator = L2MicrostructureFeatures(n_levels=5)

        # Create imbalanced order book
        bid_sizes = np.array([
            [200, 300, 400, 500, 600],  # More bid volume
            [250, 350, 450, 550, 650],
        ])
        ask_sizes = np.array([
            [100, 150, 200, 250, 300],  # Less ask volume
            [125, 175, 225, 275, 325],
        ])

        features = calculator.compute_imbalance_features(bid_sizes, ask_sizes)

        # Check imbalance calculations
        assert features["imbalance_l1"] > 0  # More bids than asks
        assert "imbalance_l1_mean" in features
        assert "imbalance_top5" in features
        assert "imbalance_total" in features

    def test_depth_features_basic(self) -> None:
        """Test order book depth features."""
        calculator = L2MicrostructureFeatures(n_levels=5)

        bid_prices = np.array([[99.5, 99.4, 99.3, 99.2, 99.1]])
        ask_prices = np.array([[100.5, 100.6, 100.7, 100.8, 100.9]])
        bid_sizes = np.array([[100, 200, 300, 400, 500]])
        ask_sizes = np.array([[150, 250, 350, 450, 550]])

        features = calculator.compute_depth_features(
            bid_sizes, ask_sizes, bid_prices, ask_prices
        )

        # Check depth calculations
        assert abs(features["bid_depth_total"] - 1500) < 1e-10  # Sum of bid sizes
        assert abs(features["ask_depth_total"] - 1750) < 1e-10  # Sum of ask sizes
        assert abs(features["depth_ratio"] - (1500 / 1750)) < 1e-10
        assert "bid_vwap" in features
        assert "ask_vwap" in features
        assert "vwap_spread" in features

    def test_shape_features_basic(self) -> None:
        """Test order book shape features."""
        calculator = L2MicrostructureFeatures(n_levels=5)

        bid_prices = np.array([[99.5, 99.4, 99.3, 99.2, 99.1]])
        ask_prices = np.array([[100.5, 100.6, 100.7, 100.8, 100.9]])
        bid_sizes = np.array([[100, 200, 300, 400, 500]])
        ask_sizes = np.array([[100, 200, 300, 400, 500]])

        features = calculator.compute_shape_features(
            bid_sizes, ask_sizes, bid_prices, ask_prices
        )

        # Check shape calculations
        assert "book_skewness" in features
        assert "bid_kurtosis" in features
        assert "ask_kurtosis" in features
        assert abs(features["bid_price_range"] - 0.4) < 1e-10  # 99.5 - 99.1
        assert abs(features["ask_price_range"] - 0.4) < 1e-10  # 100.9 - 100.5

    def test_compute_all_features_polars(self) -> None:
        """Test computing all features from Polars DataFrame."""
        calculator = L2MicrostructureFeatures(n_levels=3, lookback_window=2)

        # Create sample L2 data
        data = {
            "ts_event": [1000, 2000, 3000, 4000],
            "bid_price_0": [99.5, 99.6, 99.7, 99.8],
            "bid_price_1": [99.4, 99.5, 99.6, 99.7],
            "bid_price_2": [99.3, 99.4, 99.5, 99.6],
            "ask_price_0": [100.5, 100.4, 100.3, 100.2],
            "ask_price_1": [100.6, 100.5, 100.4, 100.3],
            "ask_price_2": [100.7, 100.6, 100.5, 100.4],
            "bid_size_0": [100, 110, 120, 130],
            "bid_size_1": [200, 210, 220, 230],
            "bid_size_2": [300, 310, 320, 330],
            "ask_size_0": [100, 110, 120, 130],
            "ask_size_1": [200, 210, 220, 230],
            "ask_size_2": [300, 310, 320, 330],
        }

        df = pl.DataFrame(data)
        features = calculator.compute_all_features(df)

        # Check feature arrays
        assert isinstance(features, dict)
        assert "spread" in features
        assert "imbalance_l1" in features
        assert "bid_depth_total" in features

        # Check array shapes (should have 2 values after lookback)
        for key, values in features.items():
            assert len(values) == 2, f"Feature {key} has wrong length"

    @given(
        n_levels=st.integers(min_value=2, max_value=10),
        n_samples=st.integers(min_value=10, max_value=50),
    )
    @settings(max_examples=20, deadline=5000)
    def test_spread_features_property_positive(
        self,
        n_levels: int,
        n_samples: int,
    ) -> None:
        """Property: spread should always be positive."""
        calculator = L2MicrostructureFeatures(n_levels=n_levels)

        # Generate random but valid order book
        base_bid = 100.0
        base_ask = 101.0

        bid_prices = np.zeros((n_samples, n_levels))
        ask_prices = np.zeros((n_samples, n_levels))

        for i in range(n_samples):
            for j in range(n_levels):
                bid_prices[i, j] = base_bid - j * 0.1
                ask_prices[i, j] = base_ask + j * 0.1

        bid_sizes = np.random.uniform(10, 1000, (n_samples, n_levels))
        ask_sizes = np.random.uniform(10, 1000, (n_samples, n_levels))

        features = calculator.compute_spread_features(
            bid_prices, ask_prices, bid_sizes, ask_sizes
        )

        # Spread should be positive
        assert features["spread"] > 0
        assert features["spread_bps"] > 0

    @given(
        bid_volume_multiplier=st.floats(min_value=0.1, max_value=10.0),
        ask_volume_multiplier=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=20, deadline=5000)
    def test_imbalance_features_property_range(
        self,
        bid_volume_multiplier: float,
        ask_volume_multiplier: float,
    ) -> None:
        """Property: imbalance should be in [-1, 1]."""
        calculator = L2MicrostructureFeatures(n_levels=5)

        # Create order book with controlled imbalance
        base_size = 100
        bid_sizes = np.array([[base_size * bid_volume_multiplier] * 5])
        ask_sizes = np.array([[base_size * ask_volume_multiplier] * 5])

        features = calculator.compute_imbalance_features(bid_sizes, ask_sizes)

        # Imbalance should be in valid range
        assert -1 <= features["imbalance_l1"] <= 1
        assert -1 <= features["imbalance_top5"] <= 1
        assert -1 <= features["imbalance_total"] <= 1


class TestL3TradeFlowFeatures:
    """Test L3 trade flow features."""

    def test_trade_imbalance_basic(self) -> None:
        """Test trade imbalance calculation."""
        calculator = L3TradeFlowFeatures(lookback_window=10)

        # Create sample trade data
        prices = np.array([100.0, 100.1, 100.2, 100.1, 100.0])
        volumes = np.array([100, 200, 150, 300, 250])
        sides = np.array([1, 1, -1, 1, -1])  # Buy, Buy, Sell, Buy, Sell

        features = calculator.compute_trade_imbalance(prices, volumes, sides)

        # Check imbalance calculations
        buy_volume = 100 + 200 + 300  # 600
        sell_volume = 150 + 250  # 400
        expected_imbalance = (600 - 400) / 1000  # 0.2

        assert abs(features["trade_imbalance"] - expected_imbalance) < 1e-10
        assert "dollar_imbalance" in features
        assert "trade_count_imbalance" in features
        assert "cumulative_flow" in features

    def test_vwap_features_basic(self) -> None:
        """Test VWAP calculation."""
        calculator = L3TradeFlowFeatures()

        prices = np.array([100.0, 101.0, 99.0, 100.5])
        volumes = np.array([100, 200, 150, 250])
        sides = np.array([1, 1, -1, 1])

        features = calculator.compute_vwap_features(prices, volumes, sides)

        # Check VWAP calculation
        expected_vwap = (100*100 + 101*200 + 99*150 + 100.5*250) / 700
        assert abs(features["vwap"] - expected_vwap) < 1e-10
        assert "price_vs_vwap" in features
        assert "buy_vwap" in features
        assert "sell_vwap" in features
        assert "vwap_spread" in features

    def test_intensity_features_basic(self) -> None:
        """Test trade intensity calculation."""
        calculator = L3TradeFlowFeatures()

        # Timestamps in nanoseconds (1 second apart)
        timestamps = np.array([0, 1_000_000_000, 2_000_000_000, 3_000_000_000])
        volumes = np.array([100, 200, 150, 250])
        prices = np.array([100.0, 101.0, 99.0, 100.5])

        features = calculator.compute_intensity_features(timestamps, volumes, prices)

        # Check intensity calculations
        assert abs(features["trade_rate"] - 4 / 3) < 1e-10  # 4 trades in 3 seconds
        assert abs(features["volume_rate"] - 700 / 3) < 1e-10  # 700 volume in 3 seconds
        assert "dollar_rate" in features
        assert "avg_trade_size" in features
        assert "trade_clustering" in features

    def test_price_impact_basic(self) -> None:
        """Test price impact calculation."""
        calculator = L3TradeFlowFeatures()

        # Prices that move with trades
        prices = np.array([100.0, 100.1, 100.2, 100.1, 100.0])
        volumes = np.array([100, 200, 150, 300, 250])
        sides = np.array([1, 1, 1, -1, -1])  # Buys push up, sells push down

        features = calculator.compute_price_impact(prices, volumes, sides)

        # Check impact calculations
        assert "avg_price_impact" in features
        assert "kyle_lambda" in features
        assert features["avg_price_impact"] >= 0  # Should be positive

    def test_compute_all_features_polars(self) -> None:
        """Test computing all features from Polars DataFrame."""
        calculator = L3TradeFlowFeatures(lookback_window=3)

        # Create sample L3 data
        data = {
            "ts_event": [1000, 2000, 3000, 4000, 5000],
            "price": [100.0, 100.1, 100.2, 100.1, 100.0],
            "volume": [100, 200, 150, 300, 250],
            "side": ["BUY", "BUY", "SELL", "BUY", "SELL"],
        }

        df = pl.DataFrame(data)
        features = calculator.compute_all_features(df)

        # Check feature arrays
        assert isinstance(features, dict)
        assert "trade_imbalance" in features
        assert "vwap" in features
        assert "trade_rate" in features
        assert "avg_price_impact" in features

        # Check array shapes (should have 2 values after lookback)
        for key, values in features.items():
            assert len(values) == 2, f"Feature {key} has wrong length"

    @given(
        n_trades=st.integers(min_value=10, max_value=100),
        buy_ratio=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=20, deadline=5000)
    def test_imbalance_property_range(
        self,
        n_trades: int,
        buy_ratio: float,
    ) -> None:
        """Property: trade imbalance should be in [-1, 1]."""
        calculator = L3TradeFlowFeatures()

        # Generate trades with controlled buy/sell ratio
        n_buys = int(n_trades * buy_ratio)
        n_sells = n_trades - n_buys

        prices = np.random.uniform(99, 101, n_trades)
        volumes = np.random.uniform(10, 1000, n_trades)
        sides = np.array([1] * n_buys + [-1] * n_sells)
        np.random.shuffle(sides)

        features = calculator.compute_trade_imbalance(prices, volumes, sides)

        # Imbalance should be in valid range
        assert -1 <= features["trade_imbalance"] <= 1
        assert -1 <= features["dollar_imbalance"] <= 1
        assert -1 <= features["trade_count_imbalance"] <= 1

    @given(
        n_trades=st.integers(min_value=5, max_value=50),
        price_base=st.floats(min_value=50, max_value=200),
    )
    @settings(max_examples=20, deadline=5000)
    def test_vwap_property_in_price_range(
        self,
        n_trades: int,
        price_base: float,
    ) -> None:
        """Property: VWAP should be within price range."""
        calculator = L3TradeFlowFeatures()

        # Generate trades around base price
        prices = np.random.uniform(price_base * 0.95, price_base * 1.05, n_trades)
        volumes = np.random.uniform(10, 1000, n_trades)

        features = calculator.compute_vwap_features(prices, volumes)

        # VWAP should be within min/max prices
        assert prices.min() <= features["vwap"] <= prices.max()

    @given(
        time_span_seconds=st.integers(min_value=1, max_value=100),
        n_trades=st.integers(min_value=2, max_value=100),
    )
    @settings(max_examples=20, deadline=5000)
    def test_intensity_property_positive_rates(
        self,
        time_span_seconds: int,
        n_trades: int,
    ) -> None:
        """Property: trade rates should be positive."""
        calculator = L3TradeFlowFeatures()

        # Generate evenly spaced timestamps
        timestamps = np.linspace(0, time_span_seconds * 1e9, n_trades).astype(np.int64)
        volumes = np.random.uniform(10, 1000, n_trades)
        prices = np.random.uniform(99, 101, n_trades)

        features = calculator.compute_intensity_features(timestamps, volumes, prices)

        # All rates should be positive
        assert features["trade_rate"] > 0
        assert features["volume_rate"] > 0
        assert features["dollar_rate"] > 0
        assert features["avg_trade_size"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
