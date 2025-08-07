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
Edge case feature parity validation tests.

This module tests perfect parity between batch and online computation under challenging
conditions including missing data, market gaps, extreme values, numerical edge cases,
and initialization scenarios.

"""

import numpy as np
import pytest

from ml._imports import HAS_POLARS
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from ml.tests.unit.feature_parity.utils import ParityTestUtils
from ml.tests.unit.feature_parity.utils import PerformanceProfiler
from ml.tests.unit.feature_parity.utils import TestDataGenerators
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class TestEdgeCaseFeatureParity:
    """
    Test suite for validating feature parity under edge case conditions.

    Ensures robust handling of missing data, extreme values, numerical instability, and
    other challenging scenarios while maintaining perfect parity.

    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        self.data_generator = TestDataGenerators(seed=42)
        self.profiler = PerformanceProfiler()

        # Standard configuration for edge case testing
        self.config = FeatureConfig(
            return_periods=[1, 5, 10],
            momentum_periods=[5, 10],
            volume_ma_periods=[5, 10],
            rsi_period=14,
            bb_period=20,
            include_microstructure=True,
            include_trade_flow=True,
        )

    def _create_bars_from_dataframe(self, df) -> list[Bar]:
        """
        Create Bar objects from DataFrame, handling edge cases.
        """
        instrument_id = InstrumentId.from_str("TEST.EDGE_CASES")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        bars = []

        # Extract data arrays with NaN handling
        if HAS_POLARS and hasattr(df, "to_numpy"):
            opens = df["open"].to_numpy()
            highs = df["high"].to_numpy()
            lows = df["low"].to_numpy()
            closes = df["close"].to_numpy()
            volumes = df["volume"].to_numpy()
        else:
            opens = df["open"].to_numpy()
            highs = df["high"].to_numpy()
            lows = df["low"].to_numpy()
            closes = df["close"].to_numpy()
            volumes = df["volume"].to_numpy()

        # Replace any NaN values with reasonable defaults
        opens = np.nan_to_num(opens, nan=100.0, posinf=1000.0, neginf=0.1)
        highs = np.nan_to_num(highs, nan=100.0, posinf=1000.0, neginf=0.1)
        lows = np.nan_to_num(lows, nan=100.0, posinf=1000.0, neginf=0.1)
        closes = np.nan_to_num(closes, nan=100.0, posinf=1000.0, neginf=0.1)
        volumes = np.nan_to_num(volumes, nan=1000.0, posinf=100000.0, neginf=1.0)

        # Ensure OHLC relationships are valid
        for i in range(len(df)):
            open_price = max(opens[i], 0.001)  # Minimum positive price
            close_price = max(closes[i], 0.001)
            high_price = max(highs[i], max(open_price, close_price))
            low_price = min(lows[i], min(open_price, close_price))
            low_price = max(low_price, 0.001)  # Ensure positive
            volume = max(volumes[i], 1.0)  # Minimum volume

            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{open_price:.6f}"),
                high=Price.from_str(f"{high_price:.6f}"),
                low=Price.from_str(f"{low_price:.6f}"),
                close=Price.from_str(f"{close_price:.6f}"),
                volume=Quantity.from_str(f"{volume:.6f}"),
                ts_event=i * 1_000_000_000,
                ts_init=i * 1_000_000_000,
            )
            bars.append(bar)

        return bars

    def test_extremely_small_values_parity(self) -> None:
        """
        Test feature parity with extremely small price and volume values.
        """
        # Generate data with very small values
        df = self.data_generator.generate_normal_ohlcv(n_bars=80, base_price=0.001, volatility=0.01)

        # Make volumes very small too
        if HAS_POLARS and hasattr(df, "to_numpy"):
            volumes = df["volume"].to_numpy()
        else:
            volumes = df["volume"].to_numpy()

        small_volumes = volumes * 0.001  # Very small volumes
        small_volumes = np.maximum(small_volumes, 1e-6)  # Ensure not zero

        if HAS_POLARS and hasattr(df, "with_columns"):
            df = df.with_columns(
                [
                    df["volume"].replace(small_volumes),
                ],
            )
        else:
            df.data["volume"] = small_volumes

        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation
        online_features = []
        for bar in bars:
            indicator_manager.update_from_bar(bar)
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
            online_features.append(features)

        # Compare with slightly relaxed tolerance for numerical precision
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
            tolerance=1e-8,
        )

    def test_extremely_large_values_parity(self) -> None:
        """
        Test feature parity with extremely large price and volume values.
        """
        # Generate data with very large values
        df = self.data_generator.generate_normal_ohlcv(
            n_bars=80,
            base_price=1000000.0,
            volatility=0.02,
        )

        # Make volumes very large too
        if HAS_POLARS and hasattr(df, "to_numpy"):
            volumes = df["volume"].to_numpy()
        else:
            volumes = df["volume"].to_numpy()

        large_volumes = volumes * 1000000.0  # Very large volumes

        if HAS_POLARS and hasattr(df, "with_columns"):
            df = df.with_columns(
                [
                    df["volume"].replace(large_volumes),
                ],
            )
        else:
            df.data["volume"] = large_volumes

        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation
        online_features = []
        for bar in bars:
            indicator_manager.update_from_bar(bar)
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
            online_features.append(features)

        # Validate parity
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

        # Ensure no overflow or infinity values
        online_array = np.array(online_features)
        assert not np.any(np.isinf(online_array)), "Infinite values detected with large inputs"
        assert not np.any(np.isnan(online_array)), "NaN values detected with large inputs"

    def test_zero_values_handling_parity(self) -> None:
        """
        Test feature parity with zero values in prices and volumes.
        """
        # Generate normal data first
        df = self.data_generator.generate_normal_ohlcv(n_bars=100)

        # Introduce some zero values (but handle them properly)
        if HAS_POLARS and hasattr(df, "to_numpy"):
            volumes = df["volume"].to_numpy()
            closes = df["close"].to_numpy()
        else:
            volumes = df["volume"].to_numpy()
            closes = df["close"].to_numpy()

        # Set some volumes to very small values (not exactly zero to avoid division issues)
        zero_volume_indices = np.random.choice(len(df), size=len(df) // 10, replace=False)
        volumes[zero_volume_indices] = 1e-6  # Very small but not zero

        # Set some price spreads to be very small
        opens = closes * (1 + np.random.normal(0, 1e-8, len(df)))  # Very tight spreads
        highs = np.maximum(opens, closes) + np.random.exponential(closes * 1e-6)
        lows = np.minimum(opens, closes) - np.random.exponential(closes * 1e-6)
        lows = np.maximum(lows, closes * 0.9999)  # Ensure positive and reasonable

        if HAS_POLARS and hasattr(df, "with_columns"):
            df = df.with_columns(
                [
                    df["volume"].replace(volumes),
                    df["open"].replace(opens),
                    df["high"].replace(highs),
                    df["low"].replace(lows),
                ],
            )
        else:
            df.data["volume"] = volumes
            df.data["open"] = opens
            df.data["high"] = highs
            df.data["low"] = lows

        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation
        online_features = []
        for bar in bars:
            indicator_manager.update_from_bar(bar)
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
            online_features.append(features)

        # Validate parity
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

    def test_constant_price_data_parity(self) -> None:
        """
        Test feature parity with constant (no movement) price data.
        """
        # Create data with constant prices
        constant_price = 100.0
        n_bars = 100

        data = {
            "open": np.full(n_bars, constant_price),
            "high": np.full(n_bars, constant_price * 1.0001),  # Tiny spread
            "low": np.full(n_bars, constant_price * 0.9999),
            "close": np.full(n_bars, constant_price),
            "volume": np.random.uniform(1000, 2000, n_bars),  # Varying volume
        }

        if HAS_POLARS:
            df = (
                HAS_POLARS.DataFrame(data)
                if hasattr(HAS_POLARS, "DataFrame")
                else self.data_generator.MockPolarsModule.DataFrame(data)
            )
        else:
            from ml.tests.unit.test_fixtures import MockPolarsModule

            df = MockPolarsModule.DataFrame(data)

        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation
        online_features = []
        for bar in bars:
            indicator_manager.update_from_bar(bar)
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
            online_features.append(features)

        # Validate parity
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

        # Verify that return features are near zero
        online_array = np.array(online_features)
        return_features = [i for i, name in enumerate(feature_names) if "return" in name]

        for ret_idx in return_features:
            return_values = online_array[:, ret_idx]
            # After warmup period, returns should be very small for constant prices
            if len(return_values) > 20:
                assert np.all(
                    np.abs(return_values[20:]) < 0.001,
                ), "Returns should be near zero for constant prices"

    def test_single_bar_data_parity(self) -> None:
        """
        Test feature parity with minimal single bar data.
        """
        # Create single bar data
        data = {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
        }

        if HAS_POLARS:
            df = (
                HAS_POLARS.DataFrame(data)
                if hasattr(HAS_POLARS, "DataFrame")
                else self.data_generator.MockPolarsModule.DataFrame(data)
            )
        else:
            from ml.tests.unit.test_fixtures import MockPolarsModule

            df = MockPolarsModule.DataFrame(data)

        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation
        online_features = []
        for bar in bars:
            indicator_manager.update_from_bar(bar)
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
            online_features.append(features)

        # Validate parity
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

        # Verify that indicators handle single bar gracefully
        online_array = np.array(online_features)
        assert online_array.shape == (
            1,
            len(feature_names),
        ), "Single bar should produce one feature vector"
        assert not np.any(np.isnan(online_array)), "Single bar should not produce NaN features"

    def test_extreme_volatility_parity(self) -> None:
        """
        Test feature parity with extreme volatility conditions.
        """
        # Generate data with extreme price swings
        df = self.data_generator.generate_volatile_data(
            n_bars=100,
            volatility=0.20,
        )  # 20% volatility

        # Add some extreme outliers
        if HAS_POLARS and hasattr(df, "to_numpy"):
            closes = df["close"].to_numpy()
        else:
            closes = df["close"].to_numpy()

        # Add some extreme price jumps
        outlier_indices = np.random.choice(len(df), size=5, replace=False)
        outlier_multipliers = np.random.choice([-0.5, 2.0], size=len(outlier_indices))

        for i, mult in zip(outlier_indices, outlier_multipliers):
            if i > 0:
                closes[i] = closes[i - 1] * (1 + mult)

        # Regenerate OHLC to be consistent with new closes
        opens = np.roll(closes, 1)
        opens[0] = closes[0]

        # Generate extreme high/low spreads
        extreme_spreads = closes * np.random.exponential(0.05, len(df))  # 5% average spread
        highs = np.maximum(opens, closes) + extreme_spreads * 0.7
        lows = np.minimum(opens, closes) - extreme_spreads * 0.3
        lows = np.maximum(lows, closes * 0.1)  # Ensure positive

        if HAS_POLARS and hasattr(df, "with_columns"):
            df = df.with_columns(
                [
                    df["open"].replace(opens),
                    df["high"].replace(highs),
                    df["low"].replace(lows),
                    df["close"].replace(closes),
                ],
            )
        else:
            df.data["open"] = opens
            df.data["high"] = highs
            df.data["low"] = lows
            df.data["close"] = closes

        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation
        online_features = []
        for bar in bars:
            indicator_manager.update_from_bar(bar)
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
            online_features.append(features)

        # Validate parity with slightly relaxed tolerance for extreme conditions
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
            tolerance=1e-9,
        )

    def test_numerical_precision_edge_cases(self) -> None:
        """
        Test feature parity with numbers near machine precision limits.
        """
        # Generate data with values near floating point precision limits
        n_bars = 80

        # Prices that differ by very small amounts
        base_price = 1.0
        tiny_increments = np.random.normal(0, 1e-12, n_bars)  # Near machine epsilon
        closes = base_price + np.cumsum(tiny_increments)

        # Ensure all prices stay positive and reasonable
        closes = np.abs(closes) + 0.1

        opens = np.roll(closes, 1)
        opens[0] = base_price

        # Very small spreads
        tiny_spreads = np.random.exponential(1e-8, n_bars)
        highs = np.maximum(opens, closes) + tiny_spreads
        lows = np.minimum(opens, closes) - tiny_spreads * 0.5
        lows = np.maximum(lows, np.minimum(opens, closes) * 0.99999)

        volumes = np.random.uniform(1000, 2000, n_bars)

        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }

        if HAS_POLARS:
            df = (
                HAS_POLARS.DataFrame(data)
                if hasattr(HAS_POLARS, "DataFrame")
                else self.data_generator.MockPolarsModule.DataFrame(data)
            )
        else:
            from ml.tests.unit.test_fixtures import MockPolarsModule

            df = MockPolarsModule.DataFrame(data)

        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation
        online_features = []
        for bar in bars:
            indicator_manager.update_from_bar(bar)
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
            online_features.append(features)

        # Validate parity with appropriate tolerance for precision limits
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
            tolerance=1e-8,
        )

    def test_initialization_with_insufficient_data(self) -> None:
        """
        Test feature parity during initialization with insufficient data for indicators.
        """
        # Create very short data series
        short_configs = [
            FeatureConfig(rsi_period=50, bb_period=50),  # Requires more data than available
            FeatureConfig(ema_slow=40, macd_signal=30),
            FeatureConfig(return_periods=[20, 50]),
        ]

        for config in short_configs:
            # Generate less data than required by indicators
            df = self.data_generator.generate_normal_ohlcv(n_bars=20)
            bars = self._create_bars_from_dataframe(df)

            feature_engineer = FeatureEngineer(config=config)
            indicator_manager = IndicatorManager(config=config)

            # Batch computation
            batch_features, _ = feature_engineer.calculate_features_batch(df)
            feature_names = config.get_feature_names()

            # Online computation
            online_features = []
            for bar in bars:
                indicator_manager.update_from_bar(bar)
                bar_dict = {
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
                features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
                online_features.append(features)

            # Validate parity even with insufficient data
            ParityTestUtils.compare_feature_vectors(
                batch_features,
                online_features,
                feature_names,
            )

    @pytest.mark.parametrize(
        "stress_scenario",
        [
            "rapid_price_changes",
            "volume_spikes",
            "alternating_extremes",
            "gradual_drift",
        ],
    )
    def test_stress_scenarios_parity(self, stress_scenario: str) -> None:
        """
        Test feature parity under various stress scenarios.
        """
        n_bars = 100

        if stress_scenario == "rapid_price_changes":
            # Prices that change dramatically every bar
            base_price = 100.0
            price_multipliers = np.random.choice([0.5, 1.5, 0.8, 1.2], n_bars)
            closes = [base_price]
            for mult in price_multipliers[1:]:
                closes.append(closes[-1] * mult)
            closes = np.array(closes)

        elif stress_scenario == "volume_spikes":
            # Normal prices but extreme volume variations
            df_base = self.data_generator.generate_normal_ohlcv(n_bars)
            closes = (
                df_base["close"].to_numpy()
                if hasattr(df_base, "to_numpy")
                else df_base["close"].to_numpy()
            )

        elif stress_scenario == "alternating_extremes":
            # Alternating between very high and very low values
            closes = np.array([100.0 if i % 2 == 0 else 200.0 for i in range(n_bars)])

        elif stress_scenario == "gradual_drift":
            # Slow continuous drift with occasional corrections
            closes = np.zeros(n_bars)
            closes[0] = 100.0
            drift = 0.001  # 0.1% per bar
            for i in range(1, n_bars):
                correction = -0.05 if i % 20 == 0 else 0  # 5% correction every 20 bars
                closes[i] = closes[i - 1] * (1 + drift + correction)

        # Generate consistent OHLC data
        opens = np.roll(closes, 1)
        opens[0] = closes[0]

        spreads = np.abs(closes) * np.random.exponential(0.01, n_bars)
        highs = np.maximum(opens, closes) + spreads * 0.6
        lows = np.minimum(opens, closes) - spreads * 0.4
        lows = np.maximum(lows, np.minimum(opens, closes) * 0.95)  # Ensure reasonable bounds

        if stress_scenario == "volume_spikes":
            # Extreme volume variations
            volumes = np.random.choice([100, 10000, 100000], n_bars, p=[0.7, 0.2, 0.1])
        else:
            volumes = np.random.uniform(1000, 5000, n_bars)

        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }

        if HAS_POLARS:
            df = (
                HAS_POLARS.DataFrame(data)
                if hasattr(HAS_POLARS, "DataFrame")
                else self.data_generator.MockPolarsModule.DataFrame(data)
            )
        else:
            from ml.tests.unit.test_fixtures import MockPolarsModule

            df = MockPolarsModule.DataFrame(data)

        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation
        online_features = []
        for bar in bars:
            indicator_manager.update_from_bar(bar)
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
            online_features.append(features)

        # Validate parity
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

    def test_performance_under_stress_conditions(self) -> None:
        """
        Test that performance requirements are met even under stress conditions.
        """
        # Generate challenging data
        df = self.data_generator.generate_volatile_data(n_bars=200, volatility=0.15)
        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Prepare bar dictionaries
        bar_dicts = []
        for bar in bars:
            indicator_manager.update_from_bar(bar)
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            bar_dicts.append(bar_dict)

        # Reset for performance test
        indicator_manager.reset()

        # Profile performance under stress
        performance_metrics = self.profiler.profile_feature_computation(
            feature_engineer,
            indicator_manager,
            bar_dicts,
            "Edge Case Stress Test",
        )

        # Validate that performance is still acceptable under stress
        self.profiler.validate_latency_requirements(
            performance_metrics,
            max_p99_latency=6.0,
        )  # Slightly relaxed

        # Additional stress-specific checks
        assert performance_metrics["mean_latency_ms"] < 4.0, "Mean latency too high under stress"
        assert performance_metrics["max_latency_ms"] < 10.0, "Maximum latency too high under stress"

    def test_robustness_across_all_edge_cases(self) -> None:
        """
        Test overall robustness by combining multiple edge cases.
        """
        # Create a dataset that combines multiple challenging conditions
        df = self.data_generator.generate_normal_ohlcv(n_bars=150)

        if HAS_POLARS and hasattr(df, "to_numpy"):
            opens = df["open"].to_numpy()
            highs = df["high"].to_numpy()
            lows = df["low"].to_numpy()
            closes = df["close"].to_numpy()
            volumes = df["volume"].to_numpy()
        else:
            opens = df["open"].to_numpy()
            highs = df["high"].to_numpy()
            lows = df["low"].to_numpy()
            closes = df["close"].to_numpy()
            volumes = df["volume"].to_numpy()

        # Apply multiple stress conditions

        # 1. Some very small values
        small_indices = np.random.choice(len(df), size=10, replace=False)
        closes[small_indices] *= 0.001

        # 2. Some very large values
        large_indices = np.random.choice(len(df), size=10, replace=False)
        closes[large_indices] *= 10000

        # 3. Some near-zero volumes
        zero_vol_indices = np.random.choice(len(df), size=15, replace=False)
        volumes[zero_vol_indices] = 1e-6

        # 4. Some extreme price gaps
        gap_indices = np.random.choice(len(df)[1:], size=8, replace=False) + 1
        for idx in gap_indices:
            closes[idx] = closes[idx - 1] * np.random.choice([0.7, 1.4])

        # 5. Regenerate OHLC consistently
        for i in range(len(df)):
            if i > 0:
                opens[i] = closes[i - 1] * (1 + np.random.normal(0, 0.001))

            # Ensure valid OHLC relationships
            close_val = closes[i]
            open_val = opens[i]

            spread = max(abs(close_val), abs(open_val)) * np.random.exponential(0.01)
            highs[i] = max(open_val, close_val) + spread * 0.7
            lows[i] = min(open_val, close_val) - spread * 0.3
            lows[i] = max(lows[i], min(open_val, close_val) * 0.99)

        # Update DataFrame
        if HAS_POLARS and hasattr(df, "with_columns"):
            df = df.with_columns(
                [
                    df["open"].replace(opens),
                    df["high"].replace(highs),
                    df["low"].replace(lows),
                    df["close"].replace(closes),
                    df["volume"].replace(volumes),
                ],
            )
        else:
            df.data["open"] = opens
            df.data["high"] = highs
            df.data["low"] = lows
            df.data["close"] = closes
            df.data["volume"] = volumes

        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation
        online_features = []
        for bar in bars:
            indicator_manager.update_from_bar(bar)
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
            online_features.append(features)

        # Validate parity with appropriate tolerance for complex edge cases
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
            tolerance=1e-8,
        )

        # Ensure no invalid values in results
        online_array = np.array(online_features)
        batch_array = (
            batch_features.to_numpy()
            if hasattr(batch_features, "to_numpy")
            else batch_features.to_numpy()
        )

        assert not np.any(np.isnan(online_array)), "Online computation produced NaN values"
        assert not np.any(np.isinf(online_array)), "Online computation produced infinite values"
        assert not np.any(np.isnan(batch_array)), "Batch computation produced NaN values"
        assert not np.any(np.isinf(batch_array)), "Batch computation produced infinite values"
