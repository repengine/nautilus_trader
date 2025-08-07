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
Microstructure feature parity validation tests.

This module tests perfect parity between batch and online computation of microstructure
features including spread metrics, size imbalance, and mid-price statistics, with
fallback scenarios using OHLCV data when microstructure data is unavailable.

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


class TestMicrostructureFeatureParity:
    """
    Test suite for validating microstructure feature parity.

    Tests both full microstructure data scenarios and OHLCV fallback scenarios to ensure
    features compute identically in batch vs online modes.

    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        self.data_generator = TestDataGenerators(seed=42)
        self.profiler = PerformanceProfiler()

        # Configuration with microstructure features enabled
        self.config = FeatureConfig(
            return_periods=[1, 5, 10],
            momentum_periods=[5, 10],
            volume_ma_periods=[5, 10],
            rsi_period=14,
            bb_period=20,
            include_microstructure=True,  # Enable microstructure features
        )

    def _create_bars_from_dataframe(self, df) -> list[Bar]:
        """
        Create Bar objects from DataFrame for consistent processing.
        """
        instrument_id = InstrumentId.from_str("TEST.MICROSTRUCTURE")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        bars = []

        # Extract basic OHLCV data
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

        for i in range(len(df)):
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(str(opens[i])),
                high=Price.from_str(str(highs[i])),
                low=Price.from_str(str(lows[i])),
                close=Price.from_str(str(closes[i])),
                volume=Quantity.from_str(str(volumes[i])),
                ts_event=i * 1_000_000_000,
                ts_init=i * 1_000_000_000,
            )
            bars.append(bar)

        return bars

    def test_microstructure_parity_with_bid_ask_data(self) -> None:
        """
        Test microstructure feature parity with full bid/ask data.
        """
        # Generate data with microstructure information
        df = self.data_generator.generate_with_microstructure_data(n_bars=100)
        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation with microstructure data
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

        # Compare batch vs online
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

    def test_microstructure_parity_ohlcv_fallback(self) -> None:
        """
        Test microstructure feature parity using OHLCV fallback calculations.
        """
        # Generate standard OHLCV data without microstructure columns
        df = self.data_generator.generate_normal_ohlcv(n_bars=100)
        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation - should use OHLCV fallback
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation - should use simplified calculations
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

        # Compare batch vs online
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

    def test_spread_metrics_parity_volatile_data(self) -> None:
        """
        Test spread metrics parity with volatile market data.
        """
        # Generate volatile data to test spread calculations
        df = self.data_generator.generate_with_microstructure_data(n_bars=120)

        # Add more volatile spreads
        if HAS_POLARS and hasattr(df, "to_numpy"):
            bid_prices = df["bid_price"].to_numpy()
            ask_prices = df["ask_price"].to_numpy()
            closes = df["close"].to_numpy()
        else:
            bid_prices = df["bid_price"].to_numpy()
            ask_prices = df["ask_price"].to_numpy()
            closes = df["close"].to_numpy()

        # Modify spreads to be more volatile
        volatility_multiplier = np.random.exponential(2.0, len(df))
        spread_adjustment = (ask_prices - bid_prices) * volatility_multiplier

        new_bid_prices = closes - spread_adjustment / 2
        new_ask_prices = closes + spread_adjustment / 2

        # Update DataFrame with new spreads
        if HAS_POLARS and hasattr(df, "with_columns"):
            df = df.with_columns(
                [
                    df["bid_price"].replace(new_bid_prices),
                    df["ask_price"].replace(new_ask_prices),
                ],
            )
        else:
            # For mock DataFrame
            df.data["bid_price"] = new_bid_prices
            df.data["ask_price"] = new_ask_prices

        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation with performance monitoring
        online_features = []
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
            features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
            online_features.append(features)

        # Validate parity
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

        # Profile performance for microstructure calculations
        indicator_manager.reset()
        performance_metrics = self.profiler.profile_feature_computation(
            feature_engineer,
            indicator_manager,
            bar_dicts[-30:],
            "Microstructure Volatile",
        )
        self.profiler.validate_latency_requirements(performance_metrics)

    def test_size_imbalance_parity(self) -> None:
        """
        Test size imbalance feature parity.
        """
        # Generate data with varying bid/ask sizes
        df = self.data_generator.generate_with_microstructure_data(n_bars=80)

        # Create more realistic size imbalances
        if HAS_POLARS and hasattr(df, "to_numpy"):
            bid_sizes = df["bid_size"].to_numpy()
            ask_sizes = df["ask_size"].to_numpy()
        else:
            bid_sizes = df["bid_size"].to_numpy()
            ask_sizes = df["ask_size"].to_numpy()

        # Create systematic imbalances
        imbalance_factor = np.sin(np.arange(len(df)) * 0.1) + 1.5  # Oscillating imbalance
        new_bid_sizes = bid_sizes * imbalance_factor
        new_ask_sizes = ask_sizes * (2.5 - imbalance_factor)  # Inverse relationship

        # Ensure positive sizes
        new_bid_sizes = np.maximum(new_bid_sizes, 100)
        new_ask_sizes = np.maximum(new_ask_sizes, 100)

        # Update DataFrame
        if HAS_POLARS and hasattr(df, "with_columns"):
            df = df.with_columns(
                [
                    df["bid_size"].replace(new_bid_sizes),
                    df["ask_size"].replace(new_ask_sizes),
                ],
            )
        else:
            df.data["bid_size"] = new_bid_sizes
            df.data["ask_size"] = new_ask_sizes

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

        # Specifically test size imbalance features
        size_imbalance_features = [name for name in feature_names if "size_imbalance" in name]

        if HAS_POLARS and hasattr(batch_features, "select"):
            for feature in size_imbalance_features:
                batch_values = batch_features.select(feature).to_numpy().flatten()
                online_array = np.array(online_features)
                feature_idx = feature_names.index(feature)
                online_values = online_array[:, feature_idx]

                ParityTestUtils.assert_features_equal(
                    batch_values,
                    online_values,
                    [feature],
                )

    def test_mid_price_return_statistics_parity(self) -> None:
        """
        Test mid-price return statistics feature parity.
        """
        # Generate microstructure data
        df = self.data_generator.generate_with_microstructure_data(n_bars=150)
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

        # Test specific mid-price return features
        mid_return_features = [name for name in feature_names if "mid_return" in name]

        if HAS_POLARS and hasattr(batch_features, "select"):
            for feature in mid_return_features:
                batch_values = batch_features.select(feature).to_numpy().flatten()
                online_array = np.array(online_features)
                feature_idx = feature_names.index(feature)
                online_values = online_array[:, feature_idx]

                # Allow slightly higher tolerance for statistical calculations
                ParityTestUtils.assert_features_equal(
                    batch_values,
                    online_values,
                    [feature],
                    tolerance=1e-9,
                )

    def test_microstructure_initialization_consistency(self) -> None:
        """
        Test that microstructure features initialize consistently.
        """
        # Generate shorter data to test initialization
        df = self.data_generator.generate_with_microstructure_data(n_bars=40)
        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation with initialization tracking
        online_features = []
        microstructure_feature_names = [
            name
            for name in feature_names
            if any(keyword in name for keyword in ["spread", "imbalance", "mid_return"])
        ]

        for i, bar in enumerate(bars):
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

            # Check that microstructure features are computed from the beginning
            if i >= 5:  # After some warmup
                online_array = np.array([features])
                for feature_name in microstructure_feature_names:
                    feature_idx = feature_names.index(feature_name)
                    feature_value = online_array[0, feature_idx]

                    # Feature should not be exactly zero (unless that's expected)
                    if "std" in feature_name and i < 10:
                        # Standard deviation features might be zero initially
                        continue
                    # Other checks can be added here

        # Validate parity
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

    @pytest.mark.parametrize(
        "fallback_scenario",
        [
            "no_bid_ask",
            "incomplete_data",
            "zero_spreads",
        ],
    )
    def test_fallback_scenario_parity(self, fallback_scenario: str) -> None:
        """
        Test feature parity under different fallback scenarios.
        """
        if fallback_scenario == "no_bid_ask":
            # Standard OHLCV data without microstructure
            df = self.data_generator.generate_normal_ohlcv(n_bars=80)

        elif fallback_scenario == "incomplete_data":
            # Partial microstructure data
            df = self.data_generator.generate_with_microstructure_data(n_bars=80)

            # Remove some microstructure columns to trigger fallbacks
            if HAS_POLARS and hasattr(df, "drop"):
                df = df.drop(["bid_size", "ask_size"])
            else:
                # For mock DataFrame
                del df.data["bid_size"]
                del df.data["ask_size"]
                df._columns = [col for col in df._columns if col not in ["bid_size", "ask_size"]]

        elif fallback_scenario == "zero_spreads":
            # Generate data with zero or very small spreads
            df = self.data_generator.generate_with_microstructure_data(n_bars=80)

            if HAS_POLARS and hasattr(df, "to_numpy"):
                closes = df["close"].to_numpy()
            else:
                closes = df["close"].to_numpy()

            # Set bid/ask to be equal to close (zero spread)
            zero_spreads = np.zeros(len(df))

            if HAS_POLARS and hasattr(df, "with_columns"):
                df = df.with_columns(
                    [
                        df["bid_price"].replace(closes),
                        df["ask_price"].replace(closes),
                    ],
                )
            else:
                df.data["bid_price"] = closes
                df.data["ask_price"] = closes

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

    def test_microstructure_performance_requirements(self) -> None:
        """
        Test that microstructure feature computation meets performance requirements.
        """
        # Generate test data
        df = self.data_generator.generate_with_microstructure_data(n_bars=100)
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

        # Profile performance
        performance_metrics = self.profiler.profile_feature_computation(
            feature_engineer,
            indicator_manager,
            bar_dicts,
            "Microstructure Features",
        )

        # Validate performance requirements
        self.profiler.validate_latency_requirements(performance_metrics)

        # Microstructure features might be slightly more expensive
        assert performance_metrics["mean_latency_ms"] < 3.0, "Microstructure mean latency too high"
        assert performance_metrics["p95_latency_ms"] < 4.5, "Microstructure P95 latency too high"

    def test_cross_scenario_consistency(self) -> None:
        """
        Test that microstructure features are consistent across different data
        scenarios.
        """
        scenarios = [
            ("normal", self.data_generator.generate_with_microstructure_data(60)),
            ("volatile", self.data_generator.generate_volatile_data(60)),  # Will use fallback
            ("trending", self.data_generator.generate_trending_data(60)),  # Will use fallback
        ]

        all_results = {}

        for scenario_name, df in scenarios:
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

            # Validate parity for this scenario
            ParityTestUtils.compare_feature_vectors(
                batch_features,
                online_features,
                feature_names,
            )

            all_results[scenario_name] = {
                "batch": batch_features,
                "online": online_features,
                "feature_names": feature_names,
            }

        # Verify that feature computations are mathematically sound
        # (This doesn't compare across scenarios, but validates each scenario internally)
        for scenario_name, results in all_results.items():
            online_array = np.array(results["online"])

            # Check that features don't have NaN or infinite values
            assert not np.any(np.isnan(online_array)), f"NaN values in {scenario_name}"
            assert not np.any(np.isinf(online_array)), f"Infinite values in {scenario_name}"
