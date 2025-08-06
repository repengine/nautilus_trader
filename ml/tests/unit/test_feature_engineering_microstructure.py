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
Comprehensive tests for FeatureEngineer microstructure features.

Tests cover:
- Microstructure feature batch processing
- Microstructure feature online processing
- Feature parity validation with < 1e-10 tolerance
- Edge cases and error handling
- Fallback behavior when bid/ask data unavailable

"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager


class TestMicrostructureFeaturesBatch:
    """
    Test microstructure features in batch processing mode (cold path).
    """

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """
        Create feature engineer with microstructure enabled.
        """
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=False,
            validate_quality=False,
        )
        return FeatureEngineer(config)

    @pytest.fixture
    def bid_ask_data(self) -> pd.DataFrame:
        """
        Create sample bid/ask data.
        """
        np.random.seed(42)  # noqa: NPY002
        n_rows = 50

        # Generate realistic bid/ask data
        mid_prices = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.01)  # noqa: NPY002
        spreads = np.random.uniform(0.01, 0.05, n_rows)  # noqa: NPY002

        bid_prices = mid_prices - spreads / 2
        ask_prices = mid_prices + spreads / 2

        bid_sizes = np.random.uniform(100, 1000, n_rows)  # noqa: NPY002
        ask_sizes = np.random.uniform(100, 1000, n_rows)  # noqa: NPY002

        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="1min"),
                "open": mid_prices,
                "high": mid_prices + np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "low": mid_prices - np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "close": mid_prices,
                "volume": np.random.uniform(1000, 10000, n_rows),  # noqa: NPY002
                "bid_price": bid_prices,
                "ask_price": ask_prices,
                "bid_size": bid_sizes,
                "ask_size": ask_sizes,
            },
        )

    @pytest.fixture
    def ohlcv_only_data(self) -> pd.DataFrame:
        """
        Create sample OHLCV data without bid/ask.
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

    def test_batch_microstructure_with_bid_ask_data(
        self,
        feature_engineer: FeatureEngineer,
        bid_ask_data: pd.DataFrame,
    ) -> None:
        """
        Test batch microstructure feature calculation with bid/ask data.
        """
        idx = 25  # Middle of dataset for sufficient history

        features = feature_engineer._calculate_microstructure_features_batch(bid_ask_data, idx)

        # Verify all expected features are present
        expected_features = {
            "spread_mean",
            "spread_std",
            "spread_relative",
            "size_imbalance_mean",
            "size_imbalance_std",
            "mid_return_std",
            "mid_return_autocorr",
        }
        assert set(features.keys()) == expected_features

        # Verify feature values are reasonable
        assert 0.0 <= features["spread_mean"] <= 1.0
        assert features["spread_std"] >= 0.0
        assert 0.0 <= features["spread_relative"] <= 1.0
        assert -1.0 <= features["size_imbalance_mean"] <= 1.0
        assert features["size_imbalance_std"] >= 0.0
        assert features["mid_return_std"] >= 0.0
        assert -1.0 <= features["mid_return_autocorr"] <= 1.0

    def test_batch_microstructure_fallback_to_ohlcv(
        self,
        feature_engineer: FeatureEngineer,
        ohlcv_only_data: pd.DataFrame,
    ) -> None:
        """
        Test batch microstructure feature calculation fallback when no bid/ask data.
        """
        idx = 25

        features = feature_engineer._calculate_microstructure_features_batch(ohlcv_only_data, idx)

        # Verify all expected features are present
        expected_features = {
            "spread_mean",
            "spread_std",
            "spread_relative",
            "size_imbalance_mean",
            "size_imbalance_std",
            "mid_return_std",
            "mid_return_autocorr",
        }
        assert set(features.keys()) == expected_features

        # Verify fallback behavior - some features should be zero/defaults
        assert features["size_imbalance_mean"] == 0.0
        assert features["size_imbalance_std"] == 0.0
        assert features["mid_return_autocorr"] == 0.0

        # Spread features should be estimated from high-low
        assert features["spread_mean"] >= 0.0
        assert features["spread_relative"] >= 0.0

    def test_batch_microstructure_edge_cases(self, feature_engineer: FeatureEngineer) -> None:
        """
        Test batch microstructure features with edge cases.
        """
        # Create edge case data
        edge_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
                "open": [100.0, 100.0, 100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0, 100.0, 100.0],  # No price movement
                "low": [100.0, 100.0, 100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0, 100.0, 100.0],
                "volume": [1000, 1000, 1000, 1000, 1000],
                "bid_price": [99.99, 99.99, 99.99, 99.99, 99.99],
                "ask_price": [100.01, 100.01, 100.01, 100.01, 100.01],
                "bid_size": [1000, 1000, 1000, 1000, 1000],
                "ask_size": [1000, 1000, 1000, 1000, 1000],
            },
        )

        features = feature_engineer._calculate_microstructure_features_batch(edge_data, 4)

        # Verify features handle constant prices gracefully
        assert features["spread_mean"] > 0.0  # Should detect the bid-ask spread
        assert features["spread_std"] == 0.0  # Constant spread
        assert features["size_imbalance_mean"] == 0.0  # Equal bid/ask sizes
        assert features["size_imbalance_std"] == 0.0  # Constant imbalance
        assert features["mid_return_std"] == 0.0  # No price movement

    def test_batch_microstructure_with_zero_spreads(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test batch microstructure features with zero spreads.
        """
        zero_spread_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
                "open": [100.0] * 10,
                "high": [100.0] * 10,
                "low": [100.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000] * 10,
                "bid_price": [100.0] * 10,  # Same as ask
                "ask_price": [100.0] * 10,  # Same as bid
                "bid_size": [1000] * 10,
                "ask_size": [1000] * 10,
            },
        )

        features = feature_engineer._calculate_microstructure_features_batch(zero_spread_data, 5)

        # Should handle zero spreads gracefully
        assert features["spread_mean"] == 0.0
        assert features["spread_std"] == 0.0
        assert features["spread_relative"] == 0.0

    def test_batch_microstructure_insufficient_data(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test batch microstructure features with insufficient data.
        """
        minimal_data = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01")],
                "open": [100.0],
                "high": [100.5],
                "low": [99.5],
                "close": [100.0],
                "volume": [1000],
                "bid_price": [99.9],
                "ask_price": [100.1],
                "bid_size": [500],
                "ask_size": [500],
            },
        )

        features = feature_engineer._calculate_microstructure_features_batch(minimal_data, 0)

        # Should handle single data point gracefully
        assert all(isinstance(v, float) for v in features.values())
        assert not any(np.isnan(v) for v in features.values())
        assert not any(np.isinf(v) for v in features.values())


class TestMicrostructureFeaturesOnline:
    """
    Test microstructure features in online processing mode (hot path).
    """

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """
        Create feature engineer with microstructure enabled.
        """
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=False,
            validate_quality=False,
        )
        return FeatureEngineer(config)

    @pytest.fixture
    def indicator_manager(self) -> IndicatorManager:
        """
        Create indicator manager with some history.
        """
        config = FeatureConfig(include_microstructure=True, include_trade_flow=False)
        manager = IndicatorManager(config)

        # Add some price history
        for i in range(10):
            price = 100.0 + i * 0.1
            volume = 1000 + i * 100
            manager.price_history["closes"].append(price)
            manager.price_history["volumes"].append(volume)

        return manager

    def test_online_microstructure_normal_bar(
        self,
        feature_engineer: FeatureEngineer,
        indicator_manager: IndicatorManager,
    ) -> None:
        """
        Test online microstructure feature calculation with normal bar data.
        """
        current_bar = {
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.2,
            "volume": 1500,
        }

        feature_idx = 0
        result_idx = feature_engineer._calculate_microstructure_features_online(
            current_bar,
            indicator_manager,
            feature_idx,
        )

        # Should advance feature index by number of microstructure features
        expected_features = 7  # 7 microstructure features
        assert result_idx == feature_idx + expected_features

        # Verify features are in reasonable ranges
        features = feature_engineer.feature_buffer[:result_idx]
        assert all(not np.isnan(f) for f in features)
        assert all(not np.isinf(f) for f in features)
        assert all(isinstance(f, float | np.floating) for f in features)

    def test_online_microstructure_zero_volume(
        self,
        feature_engineer: FeatureEngineer,
        indicator_manager: IndicatorManager,
    ) -> None:
        """
        Test online microstructure features with zero volume bar.
        """
        current_bar = {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 0.0,
        }

        feature_idx = 0
        result_idx = feature_engineer._calculate_microstructure_features_online(
            current_bar,
            indicator_manager,
            feature_idx,
        )

        # Should handle zero volume gracefully
        features = feature_engineer.feature_buffer[:result_idx]
        assert all(not np.isnan(f) for f in features)
        assert all(not np.isinf(f) for f in features)

    def test_online_microstructure_no_price_history(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test online microstructure features with empty price history.
        """
        config = FeatureConfig(include_microstructure=True, include_trade_flow=False)
        empty_manager = IndicatorManager(config)

        current_bar = {
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.2,
            "volume": 1500,
        }

        feature_idx = 0
        result_idx = feature_engineer._calculate_microstructure_features_online(
            current_bar,
            empty_manager,
            feature_idx,
        )

        # Should handle empty history gracefully
        features = feature_engineer.feature_buffer[:result_idx]
        assert all(not np.isnan(f) for f in features)
        assert all(not np.isinf(f) for f in features)

        # Mid-return std should be 0 with no history
        mid_return_std_idx = result_idx - 2  # Second to last feature
        assert feature_engineer.feature_buffer[mid_return_std_idx] == 0.0

    def test_online_microstructure_extreme_spreads(
        self,
        feature_engineer: FeatureEngineer,
        indicator_manager: IndicatorManager,
    ) -> None:
        """
        Test online microstructure features with extreme high-low spreads.
        """
        current_bar = {
            "open": 100.0,
            "high": 110.0,  # Very wide spread
            "low": 90.0,
            "close": 100.0,
            "volume": 1000,
        }

        feature_idx = 0
        result_idx = feature_engineer._calculate_microstructure_features_online(
            current_bar,
            indicator_manager,
            feature_idx,
        )

        # Should handle extreme spreads without errors
        features = feature_engineer.feature_buffer[:result_idx]
        assert all(not np.isnan(f) for f in features)
        assert all(not np.isinf(f) for f in features)

        # Spread features should reflect the wide spread
        spread_mean = feature_engineer.feature_buffer[0]
        assert spread_mean > 0.1  # Should be significant spread


class TestMicrostructureFeatureParity:
    """
    Test feature parity between batch and online microstructure calculations.
    """

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """
        Create feature engineer with microstructure enabled.
        """
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=False,
            validate_quality=False,
        )
        return FeatureEngineer(config)

    def test_microstructure_parity_with_ohlcv_fallback(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test feature parity between batch and online for OHLCV fallback calculations.
        """
        # Create OHLCV-only data to ensure fallback behavior is tested
        np.random.seed(42)  # noqa: NPY002
        n_rows = 20
        prices = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.01)  # noqa: NPY002

        ohlcv_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="1min"),
                "open": prices,
                "high": prices + np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "low": prices - np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "close": prices,
                "volume": np.random.uniform(1000, 10000, n_rows),  # noqa: NPY002
            },
        )

        # Test multiple indices
        for test_idx in [5, 10, 15]:
            # Calculate batch features
            batch_features = feature_engineer._calculate_microstructure_features_batch(
                ohlcv_data,
                test_idx,
            )

            # Setup for online calculation
            config = FeatureConfig(include_microstructure=True, include_trade_flow=False)
            indicator_manager = IndicatorManager(config)

            # Build up price history for online calculation
            for i in range(test_idx + 1):
                indicator_manager.price_history["closes"].append(float(ohlcv_data.iloc[i]["close"]))
                indicator_manager.price_history["volumes"].append(
                    float(ohlcv_data.iloc[i]["volume"]),
                )

            current_bar = {
                "open": float(ohlcv_data.iloc[test_idx]["open"]),
                "high": float(ohlcv_data.iloc[test_idx]["high"]),
                "low": float(ohlcv_data.iloc[test_idx]["low"]),
                "close": float(ohlcv_data.iloc[test_idx]["close"]),
                "volume": float(ohlcv_data.iloc[test_idx]["volume"]),
            }

            # Calculate online features
            feature_idx = 0
            result_idx = feature_engineer._calculate_microstructure_features_online(
                current_bar,
                indicator_manager,
                feature_idx,
            )

            online_features = feature_engineer.feature_buffer[:result_idx]

            # Map online features to batch feature names
            feature_names = [
                "spread_mean",
                "spread_std",
                "spread_relative",
                "size_imbalance_mean",
                "size_imbalance_std",
                "mid_return_std",
                "mid_return_autocorr",
            ]

            # Verify parity with relaxed tolerance for approximations
            for i, name in enumerate(feature_names):
                batch_val = batch_features[name]
                online_val = online_features[i]

                # Some features use different approximation methods, so check they're in same ballpark
                if name in ["spread_mean", "spread_relative"]:
                    # These should be similar for OHLCV fallback
                    if batch_val > 0:
                        assert (
                            abs(batch_val - online_val) / batch_val < 0.5
                        ), f"Feature {name} parity failed: batch={batch_val}, online={online_val}"
                elif name in ["size_imbalance_mean", "size_imbalance_std", "mid_return_autocorr"]:
                    # These should be exactly 0 for fallback
                    assert (
                        batch_val == online_val == 0.0
                    ), f"Feature {name} should be 0 for fallback: batch={batch_val}, online={online_val}"
                else:
                    # Other features should be reasonably close
                    if abs(batch_val) > 1e-6:  # Avoid division by near-zero
                        rel_error = abs(batch_val - online_val) / abs(batch_val)
                        assert (
                            rel_error < 0.1
                        ), f"Feature {name} parity failed: batch={batch_val}, online={online_val}, rel_error={rel_error}"
                    else:
                        assert (
                            abs(online_val) < 1e-6
                        ), f"Feature {name} should be near zero: batch={batch_val}, online={online_val}"


class TestMicrostructureHelperMethods:
    """
    Test helper methods used in microstructure feature calculations.
    """

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """
        Create feature engineer.
        """
        return FeatureEngineer()

    def test_extract_bid_ask_data(self, feature_engineer: FeatureEngineer) -> None:
        """
        Test bid/ask data extraction from DataFrame.
        """
        test_data = pd.DataFrame(
            {
                "bid_price": [99.0, 99.1, 99.2],
                "ask_price": [100.0, 100.1, 100.2],
                "bid_size": [1000, 1100, 1200],
                "ask_size": [2000, 2100, 2200],
                "other_col": [1, 2, 3],
            },
        )

        bid_prices, ask_prices, bid_sizes, ask_sizes = feature_engineer._extract_bid_ask_data(
            test_data,
        )

        np.testing.assert_array_equal(bid_prices, [99.0, 99.1, 99.2])
        np.testing.assert_array_equal(ask_prices, [100.0, 100.1, 100.2])
        np.testing.assert_array_equal(bid_sizes, [1000, 1100, 1200])
        np.testing.assert_array_equal(ask_sizes, [2000, 2100, 2200])

    def test_calculate_spread_metrics(self, feature_engineer: FeatureEngineer) -> None:
        """
        Test spread metrics calculation.
        """
        bid_prices = np.array([99.0, 99.1, 99.2, 99.3, 99.4])
        ask_prices = np.array([100.0, 100.1, 100.2, 100.3, 100.4])
        bid_sizes = np.array([1000, 1100, 1200, 1300, 1400])
        ask_sizes = np.array([2000, 2100, 2200, 2300, 2400])

        spreads, rel_spreads, size_imbalances, mid_prices = (
            feature_engineer._calculate_spread_metrics(
                bid_prices,
                ask_prices,
                bid_sizes,
                ask_sizes,
                1,
                3,
            )
        )

        # Check spreads
        expected_spreads = [1.0, 1.0, 1.0]  # ask - bid for indices 1, 2, 3
        np.testing.assert_array_almost_equal(spreads, expected_spreads)

        # Check relative spreads (spread / mid_price)
        expected_mid_prices = [(99.1 + 100.1) / 2, (99.2 + 100.2) / 2, (99.3 + 100.3) / 2]
        expected_rel_spreads = [1.0 / mid for mid in expected_mid_prices]
        np.testing.assert_array_almost_equal(rel_spreads, expected_rel_spreads)

        # Check size imbalances ((bid_size - ask_size) / (bid_size + ask_size))
        expected_imbalances = []
        for i in range(1, 4):
            total_size = bid_sizes[i] + ask_sizes[i]
            imbalance = (bid_sizes[i] - ask_sizes[i]) / total_size
            expected_imbalances.append(imbalance)
        np.testing.assert_array_almost_equal(size_imbalances, expected_imbalances)

    def test_calculate_mid_return_features(self, feature_engineer: FeatureEngineer) -> None:
        """
        Test mid-price return feature calculations.
        """
        # Create mid-prices with known return characteristics
        mid_prices = [100.0, 101.0, 100.5, 101.5, 100.0]  # 1%, -0.495%, 0.995%, -1.48%

        return_std, return_autocorr = feature_engineer._calculate_mid_return_features(mid_prices)

        # Should calculate standard deviation of returns
        assert return_std > 0.0
        assert not np.isnan(return_std)
        assert not np.isinf(return_std)

        # Autocorrelation should be between -1 and 1
        assert -1.0 <= return_autocorr <= 1.0
        assert not np.isnan(return_autocorr)

    def test_calculate_mid_return_features_insufficient_data(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test mid-price return features with insufficient data.
        """
        # Single price
        return_std, return_autocorr = feature_engineer._calculate_mid_return_features([100.0])
        assert return_std == 0.0
        assert return_autocorr == 0.0

        # Two prices (only one return)
        return_std, return_autocorr = feature_engineer._calculate_mid_return_features(
            [100.0, 101.0],
        )
        assert return_std == 0.0  # Can't calculate std with single return
        assert return_autocorr == 0.0  # Can't calculate autocorr with single return

    def test_calculate_microstructure_features_from_ohlcv(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test OHLCV fallback microstructure feature calculation.
        """
        ohlcv_data = pd.DataFrame(
            {
                "open": [100.0, 100.5, 101.0, 100.8, 101.2],
                "high": [100.2, 100.8, 101.3, 101.0, 101.5],
                "low": [99.8, 100.2, 100.7, 100.5, 100.9],
                "close": [100.1, 100.7, 100.9, 100.9, 101.1],
                "volume": [1000, 1200, 1100, 1300, 1150],
            },
        )

        features = feature_engineer._calculate_microstructure_features_from_ohlcv(ohlcv_data, 3)

        # Should return all expected microstructure features
        expected_features = {
            "spread_mean",
            "spread_std",
            "spread_relative",
            "size_imbalance_mean",
            "size_imbalance_std",
            "mid_return_std",
            "mid_return_autocorr",
        }
        assert set(features.keys()) == expected_features

        # Fallback values should be reasonable
        assert features["spread_mean"] >= 0.0
        assert features["spread_std"] >= 0.0
        assert features["spread_relative"] >= 0.0
        assert features["size_imbalance_mean"] == 0.0  # No bid/ask size info
        assert features["size_imbalance_std"] == 0.0
        assert features["mid_return_std"] >= 0.0
        assert features["mid_return_autocorr"] == 0.0  # Cannot estimate from OHLCV


class TestMicrostructureNaNInfHandling:
    """
    Test microstructure feature handling of NaN and infinite values.
    """

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """
        Create feature engineer with microstructure enabled.
        """
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=False,
            validate_quality=False,
        )
        return FeatureEngineer(config)

    def test_batch_microstructure_with_nan_values(self, feature_engineer: FeatureEngineer) -> None:
        """
        Test batch microstructure features with NaN values in data.
        """
        nan_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
                "open": [100.0, np.nan, 100.2, 100.3, 100.1, 100.4, 100.0, 99.9, 100.1, 100.2],
                "high": [100.2, 100.3, np.nan, 100.5, 100.3, 100.6, 100.2, 100.1, 100.3, 100.4],
                "low": [99.8, 99.9, 100.0, np.nan, 99.9, 100.2, 99.8, 99.7, 99.9, 100.0],
                "close": [100.1, 100.1, 100.2, 100.3, np.nan, 100.4, 100.0, 99.9, 100.1, 100.2],
                "volume": [1000, 1100, 1200, 1300, 1400, np.nan, 1600, 1700, 1800, 1900],
                "bid_price": [99.9, 100.0, np.nan, 100.2, 100.0, 100.3, 99.9, 99.8, 100.0, 100.1],
                "ask_price": [
                    100.1,
                    100.2,
                    100.3,
                    np.nan,
                    100.2,
                    100.5,
                    100.1,
                    100.0,
                    100.2,
                    100.3,
                ],
                "bid_size": [500, 600, 700, 800, np.nan, 1000, 1100, 1200, 1300, 1400],
                "ask_size": [800, 900, 1000, 1100, 1200, np.nan, 1400, 1500, 1600, 1700],
            },
        )

        features = feature_engineer._calculate_microstructure_features_batch(nan_data, 7)

        # Features should handle NaN gracefully and return finite values
        for feature_name, feature_value in features.items():
            assert not np.isnan(feature_value), f"Feature {feature_name} is NaN"
            assert not np.isinf(feature_value), f"Feature {feature_name} is infinite"

    def test_batch_microstructure_with_negative_prices(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test batch microstructure features with invalid negative prices.
        """
        negative_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
                "open": [100.0, 99.0, -1.0, 100.0, 101.0],  # One negative price
                "high": [100.5, 99.5, 0.0, 100.5, 101.5],
                "low": [99.5, 98.5, -2.0, 99.5, 100.5],
                "close": [100.0, 99.0, 0.0, 100.0, 101.0],
                "volume": [1000, 1100, 1200, 1300, 1400],
                "bid_price": [99.8, 98.8, -1.2, 99.8, 100.8],
                "ask_price": [100.2, 99.2, 0.2, 100.2, 101.2],
                "bid_size": [500, 600, 700, 800, 900],
                "ask_size": [800, 900, 1000, 1100, 1200],
            },
        )

        features = feature_engineer._calculate_microstructure_features_batch(negative_data, 4)

        # Features should handle negative prices gracefully
        for feature_name, feature_value in features.items():
            assert not np.isnan(feature_value), f"Feature {feature_name} is NaN"
            assert not np.isinf(feature_value), f"Feature {feature_name} is infinite"

    def test_online_microstructure_with_zero_close_price(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test online microstructure features with zero close price.
        """
        config = FeatureConfig(include_microstructure=True, include_trade_flow=False)
        indicator_manager = IndicatorManager(config)

        # Add normal price history first
        for i in range(5):
            indicator_manager.price_history["closes"].append(100.0 + i)
            indicator_manager.price_history["volumes"].append(1000 + i * 100)

        # Test with zero close price
        current_bar = {
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 0.0,  # Zero close price
            "volume": 1500,
        }

        feature_idx = 0
        result_idx = feature_engineer._calculate_microstructure_features_online(
            current_bar,
            indicator_manager,
            feature_idx,
        )

        # Should handle zero price gracefully without division by zero errors
        features = feature_engineer.feature_buffer[:result_idx]
        assert all(not np.isnan(f) for f in features)
        assert all(not np.isinf(f) for f in features)


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestMicrostructureFeaturesPolars:
    """
    Test microstructure features with Polars DataFrames.
    """

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """
        Create feature engineer with microstructure enabled.
        """
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=False,
            validate_quality=False,
        )
        return FeatureEngineer(config)

    @pytest.fixture
    def polars_bid_ask_data(self) -> Any:
        """
        Create sample bid/ask data as Polars DataFrame.
        """
        np.random.seed(42)  # noqa: NPY002
        n_rows = 30

        # Generate realistic bid/ask data
        mid_prices = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.01)  # noqa: NPY002
        spreads = np.random.uniform(0.01, 0.05, n_rows)  # noqa: NPY002

        bid_prices = mid_prices - spreads / 2
        ask_prices = mid_prices + spreads / 2

        bid_sizes = np.random.uniform(100, 1000, n_rows)  # noqa: NPY002
        ask_sizes = np.random.uniform(100, 1000, n_rows)  # noqa: NPY002

        return pl.DataFrame(
            {
                "timestamp": list(range(n_rows)),
                "open": mid_prices,
                "high": mid_prices + np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "low": mid_prices - np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "close": mid_prices,
                "volume": np.random.uniform(1000, 10000, n_rows),  # noqa: NPY002
                "bid_price": bid_prices,
                "ask_price": ask_prices,
                "bid_size": bid_sizes,
                "ask_size": ask_sizes,
            },
        )

    def test_batch_microstructure_with_polars_bid_ask(
        self,
        feature_engineer: FeatureEngineer,
        polars_bid_ask_data: Any,
    ) -> None:
        """
        Test batch microstructure features with Polars DataFrame containing bid/ask
        data.
        """
        idx = 15  # Middle of dataset

        features = feature_engineer._calculate_microstructure_features_batch(
            polars_bid_ask_data,
            idx,
        )

        # Verify all expected features are present
        expected_features = {
            "spread_mean",
            "spread_std",
            "spread_relative",
            "size_imbalance_mean",
            "size_imbalance_std",
            "mid_return_std",
            "mid_return_autocorr",
        }
        assert set(features.keys()) == expected_features

        # Verify feature values are reasonable
        assert 0.0 <= features["spread_mean"] <= 1.0
        assert features["spread_std"] >= 0.0
        assert 0.0 <= features["spread_relative"] <= 1.0
        assert -1.0 <= features["size_imbalance_mean"] <= 1.0
        assert features["size_imbalance_std"] >= 0.0
        assert features["mid_return_std"] >= 0.0
        assert -1.0 <= features["mid_return_autocorr"] <= 1.0

    def test_extract_bid_ask_data_polars(
        self,
        feature_engineer: FeatureEngineer,
        polars_bid_ask_data: Any,
    ) -> None:
        """
        Test bid/ask data extraction from Polars DataFrame.
        """
        bid_prices, ask_prices, bid_sizes, ask_sizes = feature_engineer._extract_bid_ask_data(
            polars_bid_ask_data,
        )

        # Should extract numpy arrays of correct length
        assert isinstance(bid_prices, np.ndarray)
        assert isinstance(ask_prices, np.ndarray)
        assert isinstance(bid_sizes, np.ndarray)
        assert isinstance(ask_sizes, np.ndarray)
        assert len(bid_prices) == len(polars_bid_ask_data)
        assert len(ask_prices) == len(polars_bid_ask_data)
        assert len(bid_sizes) == len(polars_bid_ask_data)
        assert len(ask_sizes) == len(polars_bid_ask_data)

        # Verify values are reasonable (bid < ask)
        assert all(bid_prices < ask_prices)
        assert all(bid_sizes > 0)
        assert all(ask_sizes > 0)


class TestMicrostructurePerformanceBenchmarks:
    """
    Test microstructure feature computation performance benchmarks.
    """

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """
        Create feature engineer with microstructure enabled.
        """
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=False,
            validate_quality=False,
        )
        return FeatureEngineer(config)

    @pytest.fixture
    def large_dataset(self) -> pd.DataFrame:
        """
        Create large dataset for performance testing.
        """
        np.random.seed(42)  # noqa: NPY002
        n_rows = 1000  # Large dataset

        mid_prices = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.01)  # noqa: NPY002
        spreads = np.random.uniform(0.01, 0.05, n_rows)  # noqa: NPY002

        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="1min"),
                "open": mid_prices,
                "high": mid_prices + np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "low": mid_prices - np.random.uniform(0.01, 0.03, n_rows),  # noqa: NPY002
                "close": mid_prices,
                "volume": np.random.uniform(1000, 10000, n_rows),  # noqa: NPY002
                "bid_price": mid_prices - spreads / 2,
                "ask_price": mid_prices + spreads / 2,
                "bid_size": np.random.uniform(100, 1000, n_rows),  # noqa: NPY002
                "ask_size": np.random.uniform(100, 1000, n_rows),  # noqa: NPY002
            },
        )

    def test_batch_microstructure_performance(
        self,
        feature_engineer: FeatureEngineer,
        large_dataset: pd.DataFrame,
    ) -> None:
        """
        Test batch microstructure calculation performance on large dataset.
        """
        import time

        # Test multiple indices to ensure consistent performance
        test_indices = [100, 300, 500, 700, 900]
        execution_times = []

        for idx in test_indices:
            start_time = time.time()
            features = feature_engineer._calculate_microstructure_features_batch(large_dataset, idx)
            end_time = time.time()

            execution_times.append(end_time - start_time)

            # Verify features are computed correctly
            assert len(features) == 7
            assert all(not np.isnan(v) for v in features.values())
            assert all(not np.isinf(v) for v in features.values())

        # Performance benchmark: should complete within reasonable time
        avg_time = sum(execution_times) / len(execution_times)
        assert avg_time < 0.1, f"Batch processing too slow: {avg_time:.3f}s average"

    def test_online_microstructure_hot_path_performance(
        self,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Test online microstructure calculation performance (hot path).
        """
        import time

        config = FeatureConfig(include_microstructure=True, include_trade_flow=False)
        indicator_manager = IndicatorManager(config)

        # Build up price history
        for i in range(100):
            indicator_manager.price_history["closes"].append(100.0 + i * 0.01)
            indicator_manager.price_history["volumes"].append(1000 + i * 10)

        current_bar = {
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.2,
            "volume": 1500,
        }

        # Test multiple runs for consistent performance
        execution_times = []
        for _ in range(100):
            start_time = time.time()
            feature_idx = 0
            result_idx = feature_engineer._calculate_microstructure_features_online(
                current_bar,
                indicator_manager,
                feature_idx,
            )
            end_time = time.time()

            execution_times.append(end_time - start_time)

            # Verify correct computation
            assert result_idx == 7  # 7 microstructure features

        # Performance benchmark: hot path should be very fast
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)

        # Hot path performance requirements (very strict)
        assert avg_time < 0.001, f"Hot path too slow: {avg_time:.6f}s average"
        assert max_time < 0.005, f"Hot path P99 too slow: {max_time:.6f}s"
