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
Additional tests to reach 90% coverage for feature engineering module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager


class TestOnlineFeatureCalculation:
    """
    Additional tests for online feature calculation paths.
    """

    def test_calculate_return_features_online(self) -> None:
        """
        Test return feature calculation in online mode.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Manually set up feature buffer
        fe.feature_buffer.fill(0.0)

        # Test with some price history
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        current_close = 106.0

        # Calculate returns
        feature_idx = fe._calculate_return_features(current_close, closes, 0)

        # Check calculations
        # The function looks at the last N+1 elements for N-period return
        # For 1-period return: closes[-2] = 104, so (106 - 104) / 104
        assert fe.feature_buffer[0] == pytest.approx((106 - 104) / 104, rel=1e-6)

        # For 5-period return: closes[-6] = 100, so (106 - 100) / 100
        assert fe.feature_buffer[1] == pytest.approx((106 - 100) / 100, rel=1e-6)

        # Check feature index advanced correctly
        # It advances through both return_periods and momentum_periods
        assert feature_idx == len(config.return_periods) + len(config.momentum_periods)

    def test_calculate_volatility_features_online(self) -> None:
        """
        Test volatility feature calculation in online mode.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create price history with known pattern
        # Generate 25 prices with small random returns
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 25)
        closes = [100.0]
        for r in returns:
            closes.append(closes[-1] * (1 + r))

        # Calculate volatility
        feature_idx = fe._calculate_volatility_features(closes, 0)

        # Should have calculated 2 volatility features
        assert feature_idx == 2

        # Both volatilities should be non-zero
        assert fe.feature_buffer[0] > 0  # vol_5
        assert fe.feature_buffer[1] > 0  # vol_20

    def test_calculate_indicator_features_online(self) -> None:
        """
        Test indicator feature calculation in online mode.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)
        mgr = IndicatorManager(config)

        # Set up some indicator values
        indicator_values = {
            "volume_sma_5": 1000000.0,
            "volume_sma_10": 1100000.0,
            "volume_sma_20": 1050000.0,
            "rsi": 0.2,  # Normalized RSI
            "bb_upper": 102.0,
            "bb_lower": 98.0,
            "bb_middle": 100.0,
            "atr": 1.5,
            "ema_fast": 99.5,
            "ema_slow": 99.0,
            "macd_line": 0.025,
            "macd_signal": 0.02,
            "macd_diff": 0.005,
        }

        # Add some price history
        mgr.price_history["highs"] = [99.0] * 20 + [102.0]
        mgr.price_history["lows"] = [97.0] * 20 + [98.0]

        current_bar = {
            "close": 100.0,
            "volume": 1200000.0,
            "high": 101.0,
            "low": 99.0,
        }

        # Calculate features
        fe._calculate_indicator_features(
            current_bar["close"],
            current_bar["volume"],
            current_bar,
            indicator_values,
            mgr,
            0,
        )

        # Check some calculations
        # Volume ratio for 5-period
        expected_vol_ratio = 1200000.0 / 1000000.0
        assert fe.feature_buffer[0] == expected_vol_ratio

        # RSI features
        assert fe.feature_buffer[3] == 0.2  # Normalized RSI
        assert fe.feature_buffer[4] == 0.0  # Not overbought (raw RSI = 60)
        assert fe.feature_buffer[5] == 0.0  # Not oversold

    def test_online_features_with_scaler(self) -> None:
        """
        Test online feature calculation with pre-fitted scaler.
        """
        pytest.skip("Skipping sklearn-dependent test")


class TestEdgeCasesAdditional:
    """
    Additional edge case tests.
    """

    def test_price_position_without_history(self) -> None:
        """
        Test price position calculation without sufficient history.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create minimal data
        df = pd.DataFrame(
            {
                "close": [100.0] * 5,  # Less than 20 required
                "volume": [1000000.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "open": [100.0] * 5,
            },
        )

        features_df, _ = fe.calculate_features_batch(df)

        # Price position should default to 0.5
        assert features_df["price_position_20"].iloc[-1] == 0.5

    def test_hl_spread_calculation(self) -> None:
        """
        Test high-low spread calculation.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create data with known spread
        df = pd.DataFrame(
            {
                "close": [100.0],
                "volume": [1000000.0],
                "high": [105.0],  # 5% above close
                "low": [95.0],  # 5% below close
                "open": [100.0],
            },
        )

        features_df, _ = fe.calculate_features_batch(df)

        # HL spread = (high - low) / close = (105 - 95) / 100 = 0.1
        assert features_df["hl_spread"].iloc[0] == pytest.approx(0.1, rel=1e-6)

    def test_ema_cross_calculation(self) -> None:
        """
        Test EMA cross feature calculation.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create trending data to ensure EMAs diverge
        prices = list(range(100, 150))  # Uptrend

        df = pd.DataFrame(
            {
                "close": prices,
                "volume": [1000000] * len(prices),
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "open": prices,
            },
        )

        features_df, _ = fe.calculate_features_batch(df)

        # In an uptrend, fast EMA should be above slow EMA
        # ema_cross = (ema_fast - ema_slow) / ema_slow
        last_row = features_df.iloc[-1]
        assert last_row["ema_cross"] > 0  # Fast above slow in uptrend

    def test_features_with_zero_close_price(self) -> None:
        """
        Test feature calculation when close price is zero.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create data with zero close (edge case)
        df = pd.DataFrame(
            {
                "close": [100.0, 50.0, 0.0],  # Price goes to zero
                "volume": [1000000] * 3,
                "high": [101.0, 51.0, 1.0],
                "low": [99.0, 49.0, 0.0],
                "open": [100.0, 50.0, 1.0],
            },
        )

        features_df, _ = fe.calculate_features_batch(df)

        # Should handle zero price gracefully
        assert len(features_df) == 3
        # Many features will be 0 or default values due to division by zero

    def test_batch_features_all_same_price(self) -> None:
        """
        Test batch calculation with constant prices.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # All same price
        df = pd.DataFrame(
            {
                "close": [100.0] * 50,
                "volume": [1000000] * 50,
                "high": [100.0] * 50,
                "low": [100.0] * 50,
                "open": [100.0] * 50,
            },
        )

        features_df, _ = fe.calculate_features_batch(df)

        # All returns should be 0
        last_row = features_df.iloc[-1]
        assert last_row["return_1"] == 0.0
        assert last_row["return_5"] == 0.0
        assert last_row["momentum_5"] == 0.0

        # Volatility should be 0
        assert last_row["volatility_5"] == 0.0
        assert last_row["volatility_20"] == 0.0

        # HL spread should be 0
        assert last_row["hl_spread"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
