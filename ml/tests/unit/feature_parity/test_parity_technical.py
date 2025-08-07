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
Technical indicator feature parity validation tests.

This module tests perfect parity between batch and online computation of technical
indicator features including SMA, EMA, RSI, Bollinger Bands, MACD, and ATR.

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


class TestTechnicalIndicatorParity:
    """
    Test suite for validating technical indicator feature parity.

    Ensures that batch and online computation of technical indicators produce identical
    results within < 1e-10 tolerance.

    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        self.data_generator = TestDataGenerators(seed=42)
        self.profiler = PerformanceProfiler()

        # Standard configuration for testing
        self.config = FeatureConfig(
            return_periods=[1, 5, 10],
            momentum_periods=[5, 10],
            volume_ma_periods=[5, 10, 20],
            rsi_period=14,
            bb_period=20,
            bb_std=2.0,
            atr_period=14,
            ema_fast=12,
            ema_slow=26,
            macd_signal=9,
        )

    def _create_bars_from_dataframe(self, df) -> list[Bar]:
        """
        Create Bar objects from DataFrame for consistent processing.

        Parameters
        ----------
        df : DataFrame
            OHLCV data.

        Returns
        -------
        list[Bar]
            List of Bar objects.

        """
        # Create dummy bar type
        instrument_id = InstrumentId.from_str("TEST.PARITY")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        bars = []

        # Extract data arrays
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
                ts_event=i * 1_000_000_000,  # 1 second intervals
                ts_init=i * 1_000_000_000,
            )
            bars.append(bar)

        return bars

    def test_sma_feature_parity_normal_data(self) -> None:
        """
        Test SMA feature parity with normal market data.
        """
        # Generate test data
        df = self.data_generator.generate_normal_ohlcv(n_bars=100)
        bars = self._create_bars_from_dataframe(df)

        # Initialize feature engineer and indicator manager
        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Compute batch features
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Compute online features
        online_features = []
        for bar in bars:
            # Update indicators
            indicator_manager.update_from_bar(bar)

            # Create bar dict for online calculation
            bar_dict = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }

            # Compute features
            features = feature_engineer.calculate_features_online(bar_dict, indicator_manager)
            online_features.append(features)

        # Compare batch vs online
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

    def test_ema_feature_parity_trending_data(self) -> None:
        """
        Test EMA feature parity with trending market data.
        """
        # Generate trending data to test EMA responsiveness
        df = self.data_generator.generate_trending_data(n_bars=150, trend_strength=0.002)
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

    def test_rsi_feature_parity_volatile_data(self) -> None:
        """
        Test RSI feature parity with volatile market data.
        """
        # Generate volatile data to test RSI behavior
        df = self.data_generator.generate_volatile_data(n_bars=120, volatility=0.04)
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

        # Profile performance
        indicator_manager.reset()  # Reset for performance test
        performance_metrics = self.profiler.profile_feature_computation(
            feature_engineer,
            indicator_manager,
            bar_dicts[-50:],
            "RSI Volatile Data",
        )
        self.profiler.validate_latency_requirements(performance_metrics)

    def test_bollinger_bands_feature_parity(self) -> None:
        """
        Test Bollinger Bands feature parity across different market conditions.
        """
        # Test with multiple data scenarios
        test_scenarios = [
            ("normal", self.data_generator.generate_normal_ohlcv(80)),
            ("trending", self.data_generator.generate_trending_data(80, trend_strength=0.0015)),
            ("volatile", self.data_generator.generate_volatile_data(80, volatility=0.035)),
        ]

        for scenario_name, df in test_scenarios:
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

    def test_macd_feature_parity_with_signals(self) -> None:
        """
        Test MACD feature parity including signal line computations.
        """
        # Generate data with sufficient length for MACD initialization
        df = self.data_generator.generate_normal_ohlcv(n_bars=200)
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

        # Specifically verify MACD components
        if HAS_POLARS and hasattr(batch_features, "select"):
            batch_macd_line = batch_features.select("macd_line").to_numpy().flatten()
            batch_macd_signal = batch_features.select("macd_signal").to_numpy().flatten()
            batch_macd_diff = batch_features.select("macd_diff").to_numpy().flatten()
        else:
            batch_macd_line = batch_features["macd_line"].to_numpy()
            batch_macd_signal = batch_features["macd_signal"].to_numpy()
            batch_macd_diff = batch_features["macd_diff"].to_numpy()

        online_array = np.array(online_features)
        macd_line_idx = feature_names.index("macd_line")
        macd_signal_idx = feature_names.index("macd_signal")
        macd_diff_idx = feature_names.index("macd_diff")

        online_macd_line = online_array[:, macd_line_idx]
        online_macd_signal = online_array[:, macd_signal_idx]
        online_macd_diff = online_array[:, macd_diff_idx]

        # Individual component validation
        ParityTestUtils.assert_features_equal(batch_macd_line, online_macd_line)
        ParityTestUtils.assert_features_equal(batch_macd_signal, online_macd_signal)
        ParityTestUtils.assert_features_equal(batch_macd_diff, online_macd_diff)

    def test_atr_feature_parity_with_gaps(self) -> None:
        """
        Test ATR feature parity with gapped market data.
        """
        # Generate data with price gaps to test ATR calculation
        df = self.data_generator.generate_gapped_data(
            n_bars=150,
            gap_probability=0.08,
            gap_size_range=(0.015, 0.04),
        )
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

    def test_volume_indicator_parity(self) -> None:
        """
        Test volume-based indicator feature parity.
        """
        # Generate data with varying volume patterns
        df = self.data_generator.generate_normal_ohlcv(n_bars=100)
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

        # Specifically test volume ratio features
        volume_features = [name for name in feature_names if "volume_ratio" in name]

        if HAS_POLARS and hasattr(batch_features, "select"):
            for vol_feature in volume_features:
                batch_vol = batch_features.select(vol_feature).to_numpy().flatten()
                online_array = np.array(online_features)
                vol_idx = feature_names.index(vol_feature)
                online_vol = online_array[:, vol_idx]

                ParityTestUtils.assert_features_equal(
                    batch_vol,
                    online_vol,
                    [vol_feature],
                )

    def test_cross_validation_consistency(self) -> None:
        """
        Test that feature computations are consistent across multiple runs.
        """
        # Use fixed seed to ensure reproducibility
        df = self.data_generator.generate_normal_ohlcv(n_bars=80)
        bars = self._create_bars_from_dataframe(df)

        # Run computation multiple times
        n_runs = 3
        all_online_features = []

        for run in range(n_runs):
            feature_engineer = FeatureEngineer(config=self.config)
            indicator_manager = IndicatorManager(config=self.config)

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

            all_online_features.append(np.array(online_features))

        # Verify all runs produce identical results
        for i in range(1, n_runs):
            ParityTestUtils.assert_features_equal(
                all_online_features[0],
                all_online_features[i],
                tolerance=1e-15,  # Even stricter for identical runs
            )

    def test_indicator_initialization_consistency(self) -> None:
        """
        Test that indicators initialize consistently in batch vs online modes.
        """
        # Generate shorter data to test initialization phase
        df = self.data_generator.generate_normal_ohlcv(n_bars=50)
        bars = self._create_bars_from_dataframe(df)

        feature_engineer = FeatureEngineer(config=self.config)
        indicator_manager = IndicatorManager(config=self.config)

        # Batch computation
        batch_features, _ = feature_engineer.calculate_features_batch(df)
        feature_names = self.config.get_feature_names()

        # Online computation
        online_features = []
        initialization_states = []

        for i, bar in enumerate(bars):
            indicator_manager.update_from_bar(bar)

            # Track when indicators become initialized
            all_initialized = indicator_manager.all_initialized()
            initialization_states.append((i, all_initialized))

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

        # Verify initialization happened at reasonable point
        # (not too early, not too late)
        init_point = None
        for i, (bar_idx, is_init) in enumerate(initialization_states):
            if is_init:
                init_point = bar_idx
                break

        assert init_point is not None, "Indicators never initialized"
        assert init_point >= 20, f"Indicators initialized too early at bar {init_point}"
        assert init_point <= 40, f"Indicators initialized too late at bar {init_point}"

    @pytest.mark.parametrize(
        "config_variant",
        [
            "minimal",
            "standard",
            "comprehensive",
        ],
    )
    def test_different_configurations_parity(self, config_variant: str) -> None:
        """
        Test feature parity with different indicator configurations.
        """
        # Define configuration variants
        configs = {
            "minimal": FeatureConfig(
                return_periods=[1, 5],
                momentum_periods=[5],
                volume_ma_periods=[5, 10],
                rsi_period=14,
                bb_period=20,
            ),
            "standard": self.config,
            "comprehensive": FeatureConfig(
                return_periods=[1, 2, 5, 10, 20],
                momentum_periods=[3, 5, 10, 15],
                volume_ma_periods=[5, 10, 20, 50],
                rsi_period=21,
                bb_period=25,
                bb_std=2.5,
                atr_period=20,
                ema_fast=10,
                ema_slow=30,
                macd_signal=12,
            ),
        }

        config = configs[config_variant]
        df = self.data_generator.generate_normal_ohlcv(n_bars=120)
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

        # Validate parity
        ParityTestUtils.compare_feature_vectors(
            batch_features,
            online_features,
            feature_names,
        )

    def test_performance_regression_monitoring(self) -> None:
        """
        Test performance regression monitoring for online computation.
        """
        # Generate test data
        df = self.data_generator.generate_normal_ohlcv(n_bars=100)
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
            "Technical Indicators",
        )

        # Validate performance requirements
        self.profiler.validate_latency_requirements(performance_metrics)

        # Additional performance checks
        assert performance_metrics["mean_latency_ms"] < 2.0, "Mean latency too high"
        assert performance_metrics["p95_latency_ms"] < 4.0, "P95 latency too high"

        # Get overall performance summary
        summary = self.profiler.get_performance_summary()
        assert summary["total_measurements"] > 0
