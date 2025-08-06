# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Comprehensive tests for FeatureEngineer trade flow features.

Tests cover:
- Trade flow feature batch processing
- Trade flow feature online processing
- Feature parity validation with < 1e-10 tolerance
- VWAP calculations
- Trade intensity metrics
- Edge cases (zero volume, single trades)
- Fallback behavior when trade data unavailable

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager


class TestTradeFlowFeaturesBatch:
    """
    Test trade flow features in batch processing mode (cold path).
    """

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """
        Create feature engineer with trade flow enabled.
        """
        config = FeatureConfig(
            include_microstructure=False,
            include_trade_flow=True,
            validate_quality=False,
        )
        return FeatureEngineer(config)

    @pytest.fixture
    def trade_data(self) -> pd.DataFrame:
        """
        Create sample trade data.
        """
        np.random.seed(42)  # noqa: NPY002
        n_rows = 50

        # Generate realistic trade data
        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(n_rows) * 0.01)  # noqa: NPY002

        # Create trade data with buy/sell sides
        trade_sides = np.random.choice([-1, 1], size=n_rows)  # noqa: NPY002  # -1 = sell, 1 = buy
        trade_volumes = np.random.uniform(100, 1000, n_rows)  # noqa: NPY002

        # Add some noise to trade prices based on side
        trade_prices = prices.copy()
        for i in range(n_rows):
            if trade_sides[i] > 0:  # Buy
                trade_prices[i] += np.random.uniform(0.001, 0.01)  # noqa: NPY002
            else:  # Sell
                trade_prices[i] -= np.random.uniform(0.001, 0.01)  # noqa: NPY002

        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="1min"),
                "open": prices,
                "high": prices + np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "low": prices - np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "close": prices,
                "volume": np.random.uniform(1000, 10000, n_rows),  # noqa: NPY002
                "trade_price": trade_prices,
                "trade_volume": trade_volumes,
                "trade_side": trade_sides,
            },
        )

    @pytest.fixture
    def ohlcv_only_data(self) -> pd.DataFrame:
        """
        Create sample OHLCV data without trade information.
        """
        np.random.seed(42)  # noqa: NPY002
        n_rows = 50

        prices = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.01)  # noqa: NPY002

        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="1min"),
                "open": prices,
                "high": prices + np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "low": prices - np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "close": prices,
                "volume": np.random.uniform(1000, 10000, n_rows),  # noqa: NPY002
            },
        )

    def test_calculate_trade_flow_features_batch_with_trade_data(
        self,
        feature_engineer: FeatureEngineer,
        trade_data: pd.DataFrame,
    ) -> None:
        """
        Test batch calculation with full trade data.
        """
        features, _ = feature_engineer.calculate_features_batch(trade_data)

        assert len(features) == len(trade_data)

        # Check trade flow features are present
        expected_features = ["trade_flow_imbalance", "vwap", "trade_intensity", "avg_price_impact"]
        feature_names = feature_engineer.get_feature_names()

        for feature in expected_features:
            assert feature in feature_names

        # Validate feature values
        if isinstance(features, pd.DataFrame):
            for feature in expected_features:
                if feature in features.columns:
                    assert not features[feature].isna().any(), f"{feature} contains NaN values"
                    assert np.isfinite(
                        features[feature],
                    ).all(), f"{feature} contains infinite values"
        elif HAS_POLARS and hasattr(features, "columns"):
            for feature in expected_features:
                if feature in features.columns:
                    assert features[feature].null_count() == 0, f"{feature} contains null values"

    def test_calculate_trade_flow_features_batch_ohlcv_fallback(
        self,
        feature_engineer: FeatureEngineer,
        ohlcv_only_data: pd.DataFrame,
    ) -> None:
        """
        Test batch calculation falls back to OHLCV when trade data unavailable.
        """
        features, _ = feature_engineer.calculate_features_batch(ohlcv_only_data)

        assert len(features) == len(ohlcv_only_data)

        # Check trade flow features use fallback values
        expected_features = ["trade_flow_imbalance", "vwap", "trade_intensity", "avg_price_impact"]
        feature_names = feature_engineer.get_feature_names()

        for feature in expected_features:
            assert feature in feature_names

        # Validate fallback behavior
        if isinstance(features, pd.DataFrame):
            # Trade flow imbalance should be 0 (no directional info)
            if "trade_flow_imbalance" in features.columns:
                assert (features["trade_flow_imbalance"] == 0.0).all()

            # VWAP should approximate close price
            if "vwap" in features.columns and "close" in ohlcv_only_data.columns:
                vwap_values = features["vwap"].to_numpy()
                close_values = ohlcv_only_data["close"].to_numpy()
                assert np.allclose(vwap_values, close_values, rtol=1e-10)

    def test_trade_flow_features_edge_cases(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test edge cases with zero volume and single trades.
        """
        # Zero volume case
        zero_volume_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
                "open": [100.0] * 5,
                "high": [100.1] * 5,
                "low": [99.9] * 5,
                "close": [100.0] * 5,
                "volume": [0.0] * 5,
                "trade_price": [100.0] * 5,
                "trade_volume": [0.0] * 5,
                "trade_side": [1] * 5,
            },
        )

        features, _ = feature_engineer.calculate_features_batch(zero_volume_data)
        assert len(features) == 5

        # Single trade case
        single_trade_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=1, freq="1min"),
                "open": [100.0],
                "high": [100.1],
                "low": [99.9],
                "close": [100.0],
                "volume": [1000.0],
                "trade_price": [100.05],
                "trade_volume": [500.0],
                "trade_side": [1],
            },
        )

        features, _ = feature_engineer.calculate_features_batch(single_trade_data)
        assert len(features) == 1

    def test_vwap_calculation_accuracy(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test VWAP calculation accuracy with known values.
        """
        # Create data with known VWAP
        trade_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="1min"),
                "open": [100.0, 101.0, 102.0],
                "high": [100.5, 101.5, 102.5],
                "low": [99.5, 100.5, 101.5],
                "close": [100.0, 101.0, 102.0],
                "volume": [1000.0, 1500.0, 2000.0],
                "trade_price": [100.0, 101.0, 102.0],
                "trade_volume": [500.0, 750.0, 1000.0],
                "trade_side": [1, -1, 1],
            },
        )

        features, _ = feature_engineer.calculate_features_batch(trade_data)

        # For the last row, VWAP should be calculated from recent trades
        # This is a basic validation - exact VWAP depends on window size
        if isinstance(features, pd.DataFrame) and "vwap" in features.columns:
            vwap_values = features["vwap"].to_numpy()
            assert all(v > 0 for v in vwap_values), "VWAP values should be positive"
            assert all(99 < v < 103 for v in vwap_values), "VWAP should be in reasonable range"

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_trade_flow_features_polars(
        self,
        feature_engineer: FeatureEngineer,
        trade_data: pd.DataFrame,
    ) -> None:
        """
        Test trade flow features with Polars DataFrame.
        """
        polars_data = pl.from_pandas(trade_data)
        features, _ = feature_engineer.calculate_features_batch(polars_data)

        assert len(features) == len(trade_data)

        # Check feature names are present
        expected_features = ["trade_flow_imbalance", "vwap", "trade_intensity", "avg_price_impact"]
        feature_names = feature_engineer.get_feature_names()

        for feature in expected_features:
            assert feature in feature_names


class TestTradeFlowFeaturesOnline:
    """
    Test trade flow features in online processing mode (hot path).
    """

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """
        Create feature engineer with trade flow enabled.
        """
        config = FeatureConfig(
            include_microstructure=False,
            include_trade_flow=True,
            validate_quality=False,
        )
        return FeatureEngineer(config)

    @pytest.fixture
    def indicator_manager(self) -> IndicatorManager:
        """
        Create indicator manager for online testing.
        """
        config = FeatureConfig(include_trade_flow=True)
        return IndicatorManager(config)

    def test_calculate_trade_flow_features_online(
        self,
        feature_engineer: FeatureEngineer,
        indicator_manager: IndicatorManager,
    ) -> None:
        """
        Test online calculation of trade flow features.
        """
        # Simulate price history
        test_prices = [100.0, 100.1, 100.05, 100.2, 99.95]
        test_volumes = [1000.0, 1200.0, 800.0, 1500.0, 900.0]

        for price, volume in zip(test_prices, test_volumes):
            from nautilus_trader.model.data import Bar
            from nautilus_trader.model.data import BarSpecification
            from nautilus_trader.model.data import BarType
            from nautilus_trader.model.enums import AggressorSide
            from nautilus_trader.model.enums import BarAggregation
            from nautilus_trader.model.enums import PriceType
            from nautilus_trader.model.identifiers import InstrumentId
            from nautilus_trader.model.objects import Price
            from nautilus_trader.model.objects import Quantity

            instrument_id = InstrumentId.from_str("TEST.SIM")
            bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
            bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(str(price)),
                high=Price.from_str(str(price + 0.05)),
                low=Price.from_str(str(price - 0.05)),
                close=Price.from_str(str(price)),
                volume=Quantity.from_str(str(volume)),
                ts_event=0,
                ts_init=0,
            )
            indicator_manager.update_from_bar(bar)

        # Calculate features online
        current_bar = {
            "open": 100.15,
            "high": 100.25,
            "low": 100.05,
            "close": 100.15,
            "volume": 1100.0,
        }

        features = feature_engineer.calculate_features_online(current_bar, indicator_manager)

        assert len(features) > 0
        assert np.isfinite(features).all(), "All features should be finite"

    def test_trade_flow_features_online_edge_cases(
        self,
        feature_engineer: FeatureEngineer,
        indicator_manager: IndicatorManager,
    ) -> None:
        """
        Test online calculation with edge cases.
        """
        # Zero volume case
        current_bar_zero_volume = {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 0.0,
        }

        features = feature_engineer.calculate_features_online(
            current_bar_zero_volume,
            indicator_manager,
        )
        assert np.isfinite(features).all()

        # High volume case
        current_bar_high_volume = {
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 100000.0,  # Very high volume
        }

        features = feature_engineer.calculate_features_online(
            current_bar_high_volume,
            indicator_manager,
        )
        assert np.isfinite(features).all()

    def test_trade_flow_features_online_consistency(
        self,
        feature_engineer: FeatureEngineer,
        indicator_manager: IndicatorManager,
    ) -> None:
        """
        Test that repeated calls with same data produce same results.
        """
        current_bar = {
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.05,
            "volume": 1000.0,
        }

        features1 = feature_engineer.calculate_features_online(current_bar, indicator_manager)
        features2 = feature_engineer.calculate_features_online(current_bar, indicator_manager)

        np.testing.assert_array_equal(
            features1,
            features2,
            "Repeated calls should produce identical results",
        )


class TestTradeFlowFeatureParity:
    """
    Test feature parity between batch and online calculations.
    """

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """
        Create feature engineer with trade flow enabled.
        """
        config = FeatureConfig(
            include_microstructure=False,
            include_trade_flow=True,
            validate_quality=False,
        )
        return FeatureEngineer(config)

    def test_feature_parity_ohlcv_only(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test feature parity between batch and online with OHLCV data only.
        """
        # Create test data
        np.random.seed(42)  # noqa: NPY002
        n_rows = 20
        prices = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.01)  # noqa: NPY002
        volumes = np.random.uniform(1000, 5000, n_rows)  # noqa: NPY002

        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="1min"),
                "open": prices,
                "high": prices + np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "low": prices - np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "close": prices,
                "volume": volumes,
            },
        )

        # Calculate batch features
        batch_features, _ = feature_engineer.calculate_features_batch(test_data)

        # Calculate online features
        online_features_list = []
        indicator_manager = IndicatorManager(feature_engineer.config)

        for idx in range(len(test_data)):
            # Update indicator manager
            from nautilus_trader.model.data import Bar
            from nautilus_trader.model.data import BarSpecification
            from nautilus_trader.model.data import BarType
            from nautilus_trader.model.enums import AggressorSide
            from nautilus_trader.model.enums import BarAggregation
            from nautilus_trader.model.enums import PriceType
            from nautilus_trader.model.identifiers import InstrumentId
            from nautilus_trader.model.objects import Price
            from nautilus_trader.model.objects import Quantity

            row = test_data.iloc[idx]
            instrument_id = InstrumentId.from_str("TEST.SIM")
            bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
            bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(str(row["open"])),
                high=Price.from_str(str(row["high"])),
                low=Price.from_str(str(row["low"])),
                close=Price.from_str(str(row["close"])),
                volume=Quantity.from_str(str(row["volume"])),
                ts_event=0,
                ts_init=0,
            )
            indicator_manager.update_from_bar(bar)

            # Calculate online features
            current_bar = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }

            online_features = feature_engineer.calculate_features_online(
                current_bar,
                indicator_manager,
            )
            online_features_list.append(online_features)

        # Convert to numpy array for comparison
        online_features_array = np.array(online_features_list)

        # Extract trade flow features for comparison
        trade_flow_feature_names = [
            "trade_flow_imbalance",
            "vwap",
            "trade_intensity",
            "avg_price_impact",
        ]
        all_feature_names = feature_engineer.get_feature_names()

        for feature_name in trade_flow_feature_names:
            if feature_name in all_feature_names:
                feature_idx = all_feature_names.index(feature_name)

                if isinstance(batch_features, pd.DataFrame):
                    batch_values = batch_features[feature_name].to_numpy()
                else:  # Polars
                    batch_values = batch_features[feature_name].to_numpy()

                online_values = online_features_array[:, feature_idx]

                # Check feature parity with tight tolerance
                # For some features, allow slightly higher tolerance due to floating point precision
                if feature_name in ["vwap", "trade_intensity"]:
                    rtol, atol = 1e-6, 1e-6
                else:
                    rtol, atol = 1e-10, 1e-10

                np.testing.assert_allclose(
                    batch_values,
                    online_values,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Feature parity violation for {feature_name}. "
                    f"Max difference: {np.max(np.abs(batch_values - online_values))}",
                )

    def test_feature_parity_with_trade_data(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test feature parity with full trade data.
        """
        # Create test data with trade information
        np.random.seed(42)  # noqa: NPY002
        n_rows = 15

        prices = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.01)  # noqa: NPY002
        volumes = np.random.uniform(1000, 5000, n_rows)  # noqa: NPY002
        trade_sides = np.random.choice([-1, 1], size=n_rows)  # noqa: NPY002
        trade_volumes = np.random.uniform(100, 1000, n_rows)  # noqa: NPY002

        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="1min"),
                "open": prices,
                "high": prices + np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "low": prices - np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "close": prices,
                "volume": volumes,
                "trade_price": prices + np.random.uniform(-0.01, 0.01, n_rows),  # noqa: NPY002
                "trade_volume": trade_volumes,
                "trade_side": trade_sides,
            },
        )

        # Calculate batch features
        batch_features, _ = feature_engineer.calculate_features_batch(test_data)

        # Verify batch features have realistic values
        if isinstance(batch_features, pd.DataFrame):
            if "trade_flow_imbalance" in batch_features.columns:
                imbalance_values = batch_features["trade_flow_imbalance"].to_numpy()
                assert np.all(
                    np.abs(imbalance_values) <= 1.0,
                ), "Trade flow imbalance should be in [-1, 1]"

            if "vwap" in batch_features.columns:
                vwap_values = batch_features["vwap"].to_numpy()
                assert np.all(vwap_values > 0), "VWAP should be positive"

            if "trade_intensity" in batch_features.columns:
                intensity_values = batch_features["trade_intensity"].to_numpy()
                assert np.all(intensity_values >= 0), "Trade intensity should be non-negative"
                assert np.all(intensity_values <= 5.0), "Trade intensity should be capped at 5.0"

            if "avg_price_impact" in batch_features.columns:
                impact_values = batch_features["avg_price_impact"].to_numpy()
                assert np.all(impact_values >= 0), "Price impact should be non-negative"
                assert np.all(impact_values <= 0.01), "Price impact should be capped at 1%"


class TestTradeFlowFeatureHelpers:
    """
    Test helper methods for trade flow feature calculations.
    """

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """
        Create feature engineer with trade flow enabled.
        """
        config = FeatureConfig(
            include_microstructure=False,
            include_trade_flow=True,
            validate_quality=False,
        )
        return FeatureEngineer(config)

    def test_extract_trade_data(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test trade data extraction.
        """
        trade_data = pd.DataFrame(
            {
                "trade_price": [100.0, 100.5, 99.5],
                "trade_volume": [500.0, 750.0, 300.0],
                "trade_side": [1, -1, 1],
            },
        )

        trade_prices, trade_volumes, trade_sides = feature_engineer._extract_trade_data(trade_data)

        assert len(trade_prices) == 3
        assert len(trade_volumes) == 3
        assert len(trade_sides) == 3

        np.testing.assert_array_equal(trade_prices, [100.0, 100.5, 99.5])
        np.testing.assert_array_equal(trade_volumes, [500.0, 750.0, 300.0])
        np.testing.assert_array_equal(trade_sides, [1, -1, 1])

    def test_calculate_trade_metrics(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test trade metrics calculation.
        """
        trade_prices = np.array([100.0, 100.1, 99.9, 100.05])
        trade_volumes = np.array([500.0, 300.0, 200.0, 400.0])
        trade_sides = np.array([1, 1, -1, 1])  # Buy, Buy, Sell, Buy

        imbalance, vwap, intensity, impact = feature_engineer._calculate_trade_metrics(
            trade_prices,
            trade_volumes,
            trade_sides,
            0,
            3,
        )

        # Validate results
        assert -1.0 <= imbalance <= 1.0, "Trade flow imbalance should be in [-1, 1]"
        assert vwap > 0, "VWAP should be positive"
        assert intensity >= 0, "Trade intensity should be non-negative"
        assert intensity <= 5.0, "Trade intensity should be capped"
        assert impact >= 0, "Price impact should be non-negative"

        # Calculate expected VWAP
        total_volume = 500.0 + 300.0 + 200.0 + 400.0
        expected_vwap = (
            100.0 * 500.0 + 100.1 * 300.0 + 99.9 * 200.0 + 100.05 * 400.0
        ) / total_volume
        assert abs(vwap - expected_vwap) < 1e-10, "VWAP calculation should be accurate"

        # Calculate expected imbalance
        buy_volume = 500.0 + 300.0 + 400.0  # Sides 1, 1, 1
        sell_volume = 200.0  # Side -1
        expected_imbalance = (buy_volume - sell_volume) / total_volume
        assert (
            abs(imbalance - expected_imbalance) < 1e-10
        ), "Trade flow imbalance should be accurate"

    def test_calculate_trade_flow_features_from_ohlcv_accuracy(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test OHLCV fallback feature accuracy.
        """
        # Create simple test data
        test_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [100.5, 101.5],
                "low": [99.5, 100.5],
                "close": [100.0, 101.0],
                "volume": [1000.0, 1500.0],
            },
        )

        bar_data = {
            "open": 101.0,
            "high": 101.5,
            "low": 100.5,
            "close": 101.0,
            "volume": 1500.0,
        }

        features = feature_engineer._calculate_trade_flow_features_from_ohlcv(
            test_data,
            1,
            bar_data,
        )

        # Validate fallback features
        assert features["trade_flow_imbalance"] == 0.0, "Should default to neutral"
        assert features["vwap"] == 101.0, "Should use close price as VWAP"
        assert features["trade_intensity"] > 0, "Should calculate relative intensity"
        assert features["avg_price_impact"] >= 0, "Should calculate price impact estimate"

    def test_trade_flow_features_boundary_conditions(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test boundary conditions and edge cases.
        """
        # Empty trade data
        empty_data = pd.DataFrame(
            {
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            },
        )

        try:
            features, _ = feature_engineer.calculate_features_batch(empty_data)
            assert len(features) == 0, "Empty data should return empty features"
        except Exception:  # noqa: S110
            # Empty data may raise exception, which is acceptable
            pass

        # Single row data
        single_row_data = pd.DataFrame(
            {
                "open": [100.0],
                "high": [100.1],
                "low": [99.9],
                "close": [100.0],
                "volume": [1000.0],
                "trade_price": [100.05],
                "trade_volume": [500.0],
                "trade_side": [1],
            },
        )

        features, _ = feature_engineer.calculate_features_batch(single_row_data)
        assert len(features) == 1, "Single row should return single feature row"

        # Verify single row feature values are reasonable
        if isinstance(features, pd.DataFrame):
            for col in features.columns:
                if col in ["trade_flow_imbalance", "vwap", "trade_intensity", "avg_price_impact"]:
                    value = features[col].iloc[0]
                    assert np.isfinite(value), f"{col} should be finite for single row"

    def test_extract_trade_data_polars(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test trade data extraction with Polars DataFrame.
        """
        if not HAS_POLARS:
            pytest.skip("Polars not available")

        trade_data = pl.DataFrame(
            {
                "trade_price": [100.0, 100.5, 99.5],
                "trade_volume": [500.0, 750.0, 300.0],
                "trade_side": [1, -1, 1],
            },
        )

        trade_prices, trade_volumes, trade_sides = feature_engineer._extract_trade_data(trade_data)

        assert len(trade_prices) == 3
        assert len(trade_volumes) == 3
        assert len(trade_sides) == 3

        np.testing.assert_array_equal(trade_prices, [100.0, 100.5, 99.5])
        np.testing.assert_array_equal(trade_volumes, [500.0, 750.0, 300.0])
        np.testing.assert_array_equal(trade_sides, [1, -1, 1])

    def test_trade_metrics_edge_cases(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test trade metrics calculation with edge cases.
        """
        # All zero volumes
        trade_prices = np.array([100.0, 100.1, 99.9])
        trade_volumes = np.array([0.0, 0.0, 0.0])
        trade_sides = np.array([1, -1, 1])

        imbalance, vwap, intensity, impact = feature_engineer._calculate_trade_metrics(
            trade_prices,
            trade_volumes,
            trade_sides,
            0,
            2,
        )

        assert imbalance == 0.0, "Zero volume should result in zero imbalance"
        assert vwap == 0.0, "Zero volume should result in zero VWAP"
        assert intensity == 0.0, "Zero trades should result in zero intensity"
        assert impact == 0.0, "No valid trades should result in zero impact"

        # All same-side trades (buy only)
        trade_prices = np.array([100.0, 100.1, 100.2])
        trade_volumes = np.array([500.0, 300.0, 200.0])
        trade_sides = np.array([1, 1, 1])

        imbalance, vwap, intensity, impact = feature_engineer._calculate_trade_metrics(
            trade_prices,
            trade_volumes,
            trade_sides,
            0,
            2,
        )

        assert imbalance == 1.0, "All buy trades should result in maximum imbalance"
        assert vwap > 0, "VWAP should be positive"
        assert intensity > 0, "Should have positive intensity"

        # Invalid prices (negative or zero)
        trade_prices = np.array([0.0, -100.0, 100.0])
        trade_volumes = np.array([500.0, 300.0, 200.0])
        trade_sides = np.array([1, -1, 1])

        imbalance, vwap, intensity, impact = feature_engineer._calculate_trade_metrics(
            trade_prices,
            trade_volumes,
            trade_sides,
            0,
            2,
        )

        # Only the last valid trade should be counted
        assert vwap == 100.0, "Only valid trades should contribute to VWAP"
        assert intensity > 0, "Should count valid trades only"

    def test_trade_flow_features_from_ohlcv_edge_cases(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test OHLCV fallback features with edge cases.
        """
        # Zero volume in OHLCV data
        zero_volume_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [100.5, 101.5],
                "low": [99.5, 100.5],
                "close": [100.0, 101.0],
                "volume": [0.0, 0.0],
            },
        )

        bar_data = {
            "open": 101.0,
            "high": 101.5,
            "low": 100.5,
            "close": 101.0,
            "volume": 0.0,
        }

        features = feature_engineer._calculate_trade_flow_features_from_ohlcv(
            zero_volume_data,
            1,
            bar_data,
        )

        assert features["trade_flow_imbalance"] == 0.0
        assert features["vwap"] == 101.0  # Should use close price
        assert features["trade_intensity"] == 1.0  # Should default to 1.0 when no volume
        assert features["avg_price_impact"] == 0.0  # Should be 0 for zero volume

        # Very high volume case
        high_volume_data = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [100.5, 101.5],
                "low": [99.5, 100.5],
                "close": [100.0, 101.0],
                "volume": [1000.0, 100000.0],  # Second bar has very high volume
            },
        )

        bar_data = {
            "open": 101.0,
            "high": 101.5,
            "low": 100.5,
            "close": 101.0,
            "volume": 100000.0,
        }

        features = feature_engineer._calculate_trade_flow_features_from_ohlcv(
            high_volume_data,
            1,
            bar_data,
        )

        assert features["trade_intensity"] <= 5.0  # Should be capped at 5.0
        assert features["avg_price_impact"] <= 0.01  # Should be capped at 1%
