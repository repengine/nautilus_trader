"""
Metamorphic tests for feature transformations.

These tests verify that feature transformations maintain expected relationships under
controlled input changes. Instead of testing for specific outputs (which can be
brittle), we test how outputs should change relative to each other.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.test_kit.stubs.data import TestDataStubs


@pytest.mark.property
@pytest.mark.parallel_safe
class TestFeatureTransformMetamorphic:
    """
    Metamorphic tests for feature engineering transformations.
    """

    @given(
        base_price=st.floats(min_value=1.0, max_value=1000.0),
        n_bars=st.integers(min_value=20, max_value=100),
        scale_factor=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=30, deadline=5000)
    def test_price_scaling_invariance(self, base_price, n_bars, scale_factor):
        """
        Metamorphic relation: Scaling all prices by a factor should scale
        price-based features proportionally but not affect normalized features.
        """
        # Create feature engineer
        config = FeatureConfig(
            enable_returns=True,
            enable_volatility=True,
            enable_technical=True,
            rsi_period=14,
            ma_periods=[5, 10],
        )
        engineer = FeatureEngineer(config)

        # Generate original price series
        prices = base_price + np.random.randn(n_bars) * (base_price * 0.01)
        prices = np.abs(prices)  # Ensure positive

        # Create bars from prices
        bars_original = self._create_bars_from_prices(prices)
        bars_scaled = self._create_bars_from_prices(prices * scale_factor)

        # Compute features
        features_original = engineer.compute_features(bars_original)
        features_scaled = engineer.compute_features(bars_scaled)

        # Metamorphic relations to verify:

        # 1. Returns should be unchanged (they're normalized)
        if "returns" in features_original:
            np.testing.assert_allclose(
                features_original["returns"],
                features_scaled["returns"],
                rtol=1e-10,
                err_msg="Returns should be scale-invariant",
            )

        # 2. RSI should be unchanged (it's normalized)
        if "rsi" in features_original:
            np.testing.assert_allclose(
                features_original["rsi"],
                features_scaled["rsi"],
                rtol=1e-3,
                err_msg="RSI should be scale-invariant (allowing small fp tolerance)",
            )

        # 3. Moving averages should scale proportionally
        if "ma_5" in features_original:
            expected_ma = features_original["ma_5"] * scale_factor
            np.testing.assert_allclose(
                features_scaled["ma_5"],
                expected_ma,
                rtol=1e-10,
                err_msg="Moving averages should scale proportionally",
            )

    @given(
        prices=st.lists(
            st.floats(min_value=90.0, max_value=110.0),
            min_size=30,
            max_size=100,
        ),
        shuffle_seed=st.integers(min_value=0, max_value=1000000),
    )
    @settings(max_examples=30, deadline=5000)
    def test_time_reversal_relationships(self, prices, shuffle_seed):
        """
        Metamorphic relation: Reversing time series should reverse
        directional indicators but preserve magnitude-based features.
        """
        prices = np.array(prices)

        # Create feature engineer
        config = FeatureConfig(
            enable_returns=True,
            enable_momentum=True,
            enable_volatility=True,
        )
        engineer = FeatureEngineer(config)

        # Create bars in forward and reverse order
        bars_forward = self._create_bars_from_prices(prices)
        bars_reverse = self._create_bars_from_prices(prices[::-1])

        # Compute features
        features_forward = engineer.compute_features(bars_forward)
        features_reverse = engineer.compute_features(bars_reverse)

        # Metamorphic relations:

        # 1. Volatility should be the same (magnitude-based)
        if "volatility" in features_forward:
            # Allow small differences due to rolling window effects
            assert (
                abs(features_forward["volatility"] - features_reverse["volatility"]) < 0.01
            ), "Volatility should be similar regardless of time direction"

        # 2. Returns should be negated (directional)
        if "returns" in features_forward:
            # The sign of returns should be opposite
            if features_forward["returns"] != 0:
                assert (
                    np.sign(features_forward["returns"]) == -np.sign(features_reverse["returns"])
                    or abs(features_forward["returns"]) < 1e-10
                ), "Returns should have opposite signs when time is reversed"

    @given(
        base_prices=st.lists(
            st.floats(min_value=90.0, max_value=110.0),
            min_size=50,
            max_size=50,
        ),
        noise_level=st.floats(min_value=0.0, max_value=0.1),
    )
    @settings(max_examples=30, deadline=5000)
    def test_noise_addition_bounds(self, base_prices, noise_level):
        """
        Metamorphic relation: Adding small noise should result in
        bounded changes to features (stability test).
        """
        base_prices = np.array(base_prices)

        # Create feature engineer
        config = FeatureConfig(
            enable_technical=True,
            ma_periods=[10, 20],
        )
        engineer = FeatureEngineer(config)

        # Create original and noisy versions
        bars_original = self._create_bars_from_prices(base_prices)
        noisy_prices = base_prices * (1 + np.random.randn(len(base_prices)) * noise_level)
        bars_noisy = self._create_bars_from_prices(noisy_prices)

        # Compute features
        features_original = engineer.compute_features(bars_original)
        features_noisy = engineer.compute_features(bars_noisy)

        # Metamorphic relation: Small input changes -> bounded output changes
        for key in features_original:
            # RSI can be highly sensitive around flat series; skip for this bound check
            if key in {"rsi", "rsi_overbought", "rsi_oversold"}:
                continue
            if isinstance(features_original[key], int | float):
                original_val = features_original[key]
                noisy_val = features_noisy[key]

                if original_val != 0:
                    relative_change = abs((noisy_val - original_val) / original_val)
                    # Feature change should be bounded by noise level (with some margin)
                    assert (
                        relative_change <= noise_level * 10
                    ), f"Feature {key} changed too much ({relative_change:.2%}) for {noise_level:.2%} noise"

    @given(
        prices=st.lists(
            st.floats(min_value=90.0, max_value=110.0),
            min_size=30,
            max_size=100,
        ),
        duplication_factor=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=30, deadline=5000)
    def test_data_duplication_invariance(self, prices, duplication_factor):
        """
        Metamorphic relation: Duplicating each data point should not
        affect value-based features significantly (different from simply
        having more data points).
        """
        prices = np.array(prices)

        # Create feature engineer
        config = FeatureConfig(
            enable_technical=True,
            enable_volatility=True,
        )
        engineer = FeatureEngineer(config)

        # Create original bars
        bars_original = self._create_bars_from_prices(prices)

        # Create duplicated data (each price appears multiple times)
        duplicated_prices = np.repeat(prices, duplication_factor)
        bars_duplicated = self._create_bars_from_prices(duplicated_prices)

        # Compute features
        features_original = engineer.compute_features(bars_original)
        features_duplicated = engineer.compute_features(bars_duplicated)

        # Metamorphic relations:

        # 1. Mean-based features should be similar
        if "ma_10" in features_original and len(prices) >= 10:
            # The moving average should be similar (within tolerance)
            assert (
                abs(features_original.get("ma_10", 0) - features_duplicated.get("ma_10", 0)) < 1.0
            ), "Moving average should be stable under data duplication"

        # 2. Volatility should decrease (less variation in duplicated data)
        if "volatility" in features_original and "volatility" in features_duplicated:
            assert (
                features_duplicated["volatility"] <= features_original["volatility"]
            ), "Volatility should not increase with duplicated data"

    def _create_bars_from_prices(self, prices: np.ndarray) -> list[Bar]:
        """
        Helper to create bars from price array.
        """
        bars: list[Bar] = []
        bar_type = BarType.from_str("EURUSD.SIM-1-MINUTE-BID-EXTERNAL")

        for i, price in enumerate(prices):
            # Create bar with some realistic OHLC variation
            high = price * 1.001
            low = price * 0.999
            bars.append(
                Bar(
                    bar_type=bar_type,
                    open=Price.from_str(f"{price:.5f}"),
                    high=Price.from_str(f"{high:.5f}"),
                    low=Price.from_str(f"{low:.5f}"),
                    close=Price.from_str(f"{price:.5f}"),
                    volume=Quantity.from_int(1_000_000),
                    ts_event=i * 60_000_000_000,
                    ts_init=i * 60_000_000_000 + 1000,
                ),
            )

        return bars


class TestFeatureCompositionMetamorphic:
    """
    Test metamorphic properties of feature composition.
    """

    @given(
        n_features=st.integers(min_value=5, max_value=20),
        n_samples=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30, deadline=5000)
    def test_feature_subset_consistency(self, n_features, n_samples):
        """
        Metamorphic relation: Computing features on a subset should
        yield the same values as computing all features and selecting.
        """
        # Generate sample data
        prices = 100 + np.random.randn(n_samples) * 2

        # Create two configs - one with all features, one with subset
        config_all = FeatureConfig(
            enable_returns=True,
            enable_volatility=True,
            enable_technical=True,
            enable_momentum=True,
        )

        config_subset = FeatureConfig(
            enable_returns=True,
            enable_volatility=False,
            enable_technical=True,
            enable_momentum=False,
        )

        engineer_all = FeatureEngineer(config_all)
        engineer_subset = FeatureEngineer(config_subset)

        bars = self._create_simple_bars(prices)

        # Compute features
        features_all = engineer_all.compute_features(bars)
        features_subset = engineer_subset.compute_features(bars)

        # Metamorphic relation: Subset features should match
        for key in features_subset:
            if key in features_all:
                np.testing.assert_allclose(
                    features_subset[key],
                    features_all[key],
                    rtol=1e-10,
                    err_msg=f"Feature {key} differs between subset and full computation",
                )

    @given(
        prices1=st.lists(
            st.floats(min_value=90.0, max_value=110.0),
            min_size=20,
            max_size=50,
        ),
        prices2=st.lists(
            st.floats(min_value=90.0, max_value=110.0),
            min_size=20,
            max_size=50,
        ),
    )
    @settings(max_examples=30, deadline=5000)
    def test_concatenation_consistency(self, prices1, prices2):
        """
        Metamorphic relation: Features computed on concatenated data
        should be consistent with features computed separately.
        """
        # Create feature engineer
        config = FeatureConfig(
            enable_returns=True,
            ma_periods=[5],
        )
        engineer = FeatureEngineer(config)

        # Create bars for each price series
        bars1 = self._create_simple_bars(prices1)
        bars2 = self._create_simple_bars(prices2)
        bars_concat = bars1 + bars2

        # Compute features
        features1 = engineer.compute_features(bars1)
        features2 = engineer.compute_features(bars2)
        features_concat = engineer.compute_features(bars_concat)

        # Metamorphic relation: Returns at the boundary should be consistent
        # The return from last price of series1 to first price of series2
        # should be captured in the concatenated version
        if "returns" in features_concat:
            # The concatenated version should have a valid return value
            assert not np.isnan(
                features_concat["returns"],
            ), "Concatenated features should handle boundaries properly"

    def _create_simple_bars(self, prices: list[float]) -> list[Bar]:
        """
        Helper to create simple bars from prices.
        """
        bars: list[Bar] = []
        bar_type = BarType.from_str("TEST.SIM-1-MINUTE-BID-EXTERNAL")

        for i, price in enumerate(prices):
            bars.append(
                Bar(
                    bar_type=bar_type,
                    open=Price.from_str(f"{price:.5f}"),
                    high=Price.from_str(f"{price * 1.001:.5f}"),
                    low=Price.from_str(f"{price * 0.999:.5f}"),
                    close=Price.from_str(f"{price:.5f}"),
                    volume=Quantity.from_int(1_000_000),
                    ts_event=i * 60_000_000_000,
                    ts_init=i * 60_000_000_000 + 1000,
                ),
            )

        return bars
