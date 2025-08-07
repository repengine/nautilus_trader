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
Trade flow feature parity validation tests.

This module tests perfect parity between batch and online computation of trade flow
features including trade imbalance, VWAP, trade intensity, and price impact metrics,
with fallback scenarios using OHLCV data when trade-level data is unavailable.

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


class TestTradeFlowFeatureParity:
    """
    Test suite for validating trade flow feature parity.

    Tests both full trade-level data scenarios and OHLCV fallback scenarios to ensure
    features compute identically in batch vs online modes.

    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        self.data_generator = TestDataGenerators(seed=42)
        self.profiler = PerformanceProfiler()

        # Configuration with trade flow features enabled
        self.config = FeatureConfig(
            return_periods=[1, 5, 10],
            momentum_periods=[5, 10],
            volume_ma_periods=[5, 10],
            rsi_period=14,
            bb_period=20,
            include_trade_flow=True,  # Enable trade flow features
        )

    def _create_bars_from_dataframe(self, df) -> list[Bar]:
        """
        Create Bar objects from DataFrame for consistent processing.
        """
        instrument_id = InstrumentId.from_str("TEST.TRADEFLOW")
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

    def test_trade_flow_parity_with_trade_data(self) -> None:
        """
        Test trade flow feature parity with full trade-level data.
        """
        # Generate data with trade information
        df = self.data_generator.generate_with_trade_data(n_bars=100)
        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation with trade data
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

    def test_trade_flow_parity_ohlcv_fallback(self) -> None:
        """
        Test trade flow feature parity using OHLCV fallback calculations.
        """
        # Generate standard OHLCV data without trade columns
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

    def test_trade_imbalance_parity_directional_data(self) -> None:
        """
        Test trade imbalance feature parity with directional trade data.
        """
        # Generate data with strong directional bias
        df = self.data_generator.generate_with_trade_data(n_bars=120)

        # Modify trade sides to create stronger directional imbalances
        if HAS_POLARS and hasattr(df, "to_numpy"):
            trade_sides = df["trade_side"].to_numpy()
            trade_volumes = df["trade_volume"].to_numpy()
        else:
            trade_sides = df["trade_side"].to_numpy()
            trade_volumes = df["trade_volume"].to_numpy()

        # Create periods of strong buying and selling
        n_bars = len(df)
        imbalance_pattern = np.concatenate(
            [
                np.ones(n_bars // 3),  # Strong buying
                -np.ones(n_bars // 3),  # Strong selling
                np.sin(np.arange(n_bars - 2 * (n_bars // 3)) * 0.2),  # Oscillating
            ],
        )

        # Apply the pattern
        new_trade_sides = np.sign(imbalance_pattern)
        new_trade_sides[new_trade_sides == 0] = 1  # No neutral trades

        # Update DataFrame
        if HAS_POLARS and hasattr(df, "with_columns"):
            df = df.with_columns(
                [
                    df["trade_side"].replace(new_trade_sides),
                ],
            )
        else:
            df.data["trade_side"] = new_trade_sides

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

        # Profile performance for trade flow calculations
        indicator_manager.reset()
        performance_metrics = self.profiler.profile_feature_computation(
            feature_engineer,
            indicator_manager,
            bar_dicts[-30:],
            "Trade Flow Imbalanced",
        )
        self.profiler.validate_latency_requirements(performance_metrics)

    def test_vwap_calculation_parity(self) -> None:
        """
        Test VWAP calculation feature parity.
        """
        # Generate trade data with varying volumes and prices
        df = self.data_generator.generate_with_trade_data(n_bars=100)

        # Create more realistic VWAP scenarios by adjusting trade prices and volumes
        if HAS_POLARS and hasattr(df, "to_numpy"):
            trade_prices = df["trade_price"].to_numpy()
            trade_volumes = df["trade_volume"].to_numpy()
            closes = df["close"].to_numpy()
        else:
            trade_prices = df["trade_price"].to_numpy()
            trade_volumes = df["trade_volume"].to_numpy()
            closes = df["close"].to_numpy()

        # Create price variations around close for more realistic VWAP
        price_variations = np.random.normal(0, 0.001, len(df))  # 0.1% price variation
        new_trade_prices = closes * (1 + price_variations)

        # Create volume clusters (some bars with much higher volumes)
        volume_multipliers = np.random.choice([1, 2, 5], len(df), p=[0.7, 0.2, 0.1])
        new_trade_volumes = trade_volumes * volume_multipliers

        # Update DataFrame
        if HAS_POLARS and hasattr(df, "with_columns"):
            df = df.with_columns(
                [
                    df["trade_price"].replace(new_trade_prices),
                    df["trade_volume"].replace(new_trade_volumes),
                ],
            )
        else:
            df.data["trade_price"] = new_trade_prices
            df.data["trade_volume"] = new_trade_volumes

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

        # Specifically test VWAP feature
        if "vwap" in feature_names:
            if HAS_POLARS and hasattr(batch_features, "select"):
                batch_vwap = batch_features.select("vwap").to_numpy().flatten()
                online_array = np.array(online_features)
                vwap_idx = feature_names.index("vwap")
                online_vwap = online_array[:, vwap_idx]

                ParityTestUtils.assert_features_equal(
                    batch_vwap,
                    online_vwap,
                    ["vwap"],
                )

    def test_trade_intensity_parity(self) -> None:
        """
        Test trade intensity feature parity.
        """
        # Generate data with varying trade frequencies
        df = self.data_generator.generate_with_trade_data(n_bars=80)
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

        # Test specific trade intensity feature
        if "trade_intensity" in feature_names:
            if HAS_POLARS and hasattr(batch_features, "select"):
                batch_intensity = batch_features.select("trade_intensity").to_numpy().flatten()
                online_array = np.array(online_features)
                intensity_idx = feature_names.index("trade_intensity")
                online_intensity = online_array[:, intensity_idx]

                ParityTestUtils.assert_features_equal(
                    batch_intensity,
                    online_intensity,
                    ["trade_intensity"],
                )

                # Ensure trade intensity values are reasonable
                assert np.all(batch_intensity >= 0), "Trade intensity should be non-negative"
                assert np.all(batch_intensity <= 10), "Trade intensity should be bounded"

    def test_price_impact_parity_volatile_data(self) -> None:
        """
        Test price impact feature parity with volatile market data.
        """
        # Generate volatile trade data
        df = self.data_generator.generate_with_trade_data(n_bars=100)

        # Make prices more volatile to test price impact calculations
        if HAS_POLARS and hasattr(df, "to_numpy"):
            trade_prices = df["trade_price"].to_numpy()
        else:
            trade_prices = df["trade_price"].to_numpy()

        # Add larger price movements between trades
        price_impacts = np.random.normal(0, 0.005, len(df))  # 0.5% average impact
        cumulative_impacts = np.cumsum(price_impacts)

        base_price = trade_prices[0]
        new_trade_prices = base_price * (1 + cumulative_impacts)

        # Update DataFrame
        if HAS_POLARS and hasattr(df, "with_columns"):
            df = df.with_columns(
                [
                    df["trade_price"].replace(new_trade_prices),
                ],
            )
        else:
            df.data["trade_price"] = new_trade_prices

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

    @pytest.mark.parametrize(
        "fallback_scenario",
        [
            "no_trade_data",
            "incomplete_trade_data",
            "zero_volumes",
        ],
    )
    def test_fallback_scenario_parity(self, fallback_scenario: str) -> None:
        """
        Test feature parity under different fallback scenarios.
        """
        if fallback_scenario == "no_trade_data":
            # Standard OHLCV data without trade-level information
            df = self.data_generator.generate_normal_ohlcv(n_bars=80)

        elif fallback_scenario == "incomplete_trade_data":
            # Partial trade data
            df = self.data_generator.generate_with_trade_data(n_bars=80)

            # Remove some trade columns to trigger fallbacks
            if HAS_POLARS and hasattr(df, "drop"):
                df = df.drop(["trade_side"])
            else:
                # For mock DataFrame
                del df.data["trade_side"]
                df._columns = [col for col in df._columns if col != "trade_side"]

        elif fallback_scenario == "zero_volumes":
            # Generate data with zero or very small trade volumes
            df = self.data_generator.generate_with_trade_data(n_bars=80)

            # Set some trade volumes to zero
            if HAS_POLARS and hasattr(df, "to_numpy"):
                trade_volumes = df["trade_volume"].to_numpy()
            else:
                trade_volumes = df["trade_volume"].to_numpy()

            # Set 20% of trade volumes to zero
            zero_indices = np.random.choice(len(df), size=len(df) // 5, replace=False)
            trade_volumes[zero_indices] = 0

            if HAS_POLARS and hasattr(df, "with_columns"):
                df = df.with_columns(
                    [
                        df["trade_volume"].replace(trade_volumes),
                    ],
                )
            else:
                df.data["trade_volume"] = trade_volumes

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

    def test_trade_flow_initialization_consistency(self) -> None:
        """
        Test that trade flow features initialize consistently.
        """
        # Generate data for initialization testing
        df = self.data_generator.generate_with_trade_data(n_bars=50)
        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation with initialization tracking
        online_features = []
        trade_flow_feature_names = [
            name
            for name in feature_names
            if any(
                keyword in name
                for keyword in ["trade_flow", "vwap", "trade_intensity", "price_impact"]
            )
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

            # Check that trade flow features are reasonable from the beginning
            if i >= 5:  # After some warmup
                online_array = np.array([features])
                for feature_name in trade_flow_feature_names:
                    feature_idx = feature_names.index(feature_name)
                    feature_value = online_array[0, feature_idx]

                    # Basic sanity checks
                    if feature_name == "trade_intensity":
                        assert (
                            feature_value >= 0
                        ), f"Trade intensity should be non-negative at bar {i}"
                    elif feature_name == "vwap":
                        assert feature_value > 0, f"VWAP should be positive at bar {i}"

        # Validate parity
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

    def test_trade_flow_performance_requirements(self) -> None:
        """
        Test that trade flow feature computation meets performance requirements.
        """
        # Generate test data
        df = self.data_generator.generate_with_trade_data(n_bars=100)
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
            "Trade Flow Features",
        )

        # Validate performance requirements
        self.profiler.validate_latency_requirements(performance_metrics)

        # Trade flow features should still meet hot path requirements
        assert performance_metrics["mean_latency_ms"] < 3.0, "Trade flow mean latency too high"
        assert performance_metrics["p95_latency_ms"] < 4.5, "Trade flow P95 latency too high"

    def test_cross_data_type_consistency(self) -> None:
        """
        Test trade flow feature consistency across data types.
        """
        # Test with both complete trade data and OHLCV fallback
        complete_data = self.data_generator.generate_with_trade_data(n_bars=60)
        fallback_data = self.data_generator.generate_normal_ohlcv(n_bars=60)

        test_datasets = [
            ("complete", complete_data),
            ("fallback", fallback_data),
        ]

        all_results = {}

        for data_type, df in test_datasets:
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

            # Validate parity for this data type
            ParityTestUtils.compare_feature_vectors(
                batch_features,
                online_features,
                feature_names,
            )

            all_results[data_type] = {
                "batch": batch_features,
                "online": online_features,
                "feature_names": feature_names,
            }

        # Verify that fallback calculations produce reasonable results
        fallback_online = np.array(all_results["fallback"]["online"])

        # Check that fallback features are not all zeros (they should have reasonable defaults)
        trade_flow_indices = [
            i
            for i, name in enumerate(feature_names)
            if any(
                keyword in name
                for keyword in ["trade_flow", "vwap", "trade_intensity", "price_impact"]
            )
        ]

        for idx in trade_flow_indices:
            feature_values = fallback_online[:, idx]
            feature_name = feature_names[idx]

            if feature_name == "trade_flow_imbalance":
                # Should be near zero (neutral) for fallback
                assert np.all(
                    np.abs(feature_values) < 0.1,
                ), f"Fallback {feature_name} should be near neutral"
            elif feature_name == "vwap":
                # Should be close to close prices
                closes = (
                    fallback_data["close"].to_numpy()
                    if hasattr(fallback_data, "to_numpy")
                    else fallback_data["close"].to_numpy()
                )
                assert np.all(feature_values > 0), f"Fallback {feature_name} should be positive"
            elif feature_name == "trade_intensity":
                # Should be reasonable values (not zero, not too high)
                assert np.all(feature_values > 0), f"Fallback {feature_name} should be positive"
                assert np.all(feature_values < 10), f"Fallback {feature_name} should be bounded"
