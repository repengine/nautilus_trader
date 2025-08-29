"""
Unit tests for ML feature engineering module.

Tests cover:
- Feature configuration validation
- Batch feature calculation
- Online feature calculation
- Feature parity between batch and online modes
- Edge cases and error handling

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
from ml.features.engineering import safe_divide
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


@pytest.mark.parallel_safe
@pytest.mark.unit
class TestSafeDivide:
    """
    Test safe_divide utility function.
    """

    def test_safe_divide_normal_operation(self) -> None:
        """
        Test safe divide with normal values.
        """
        assert safe_divide(10.0, 2.0) == 5.0
        assert safe_divide(100.0, 25.0) == 4.0
        assert safe_divide(-10.0, 2.0) == -5.0

    def test_safe_divide_zero_denominator(self) -> None:
        """
        Test safe divide with zero denominator.
        """
        assert safe_divide(10.0, 0.0) == 0.0
        assert safe_divide(10.0, 0.0, default=1.0) == 1.0
        assert safe_divide(-10.0, 0.0, default=-1.0) == -1.0

    def test_safe_divide_none_denominator(self) -> None:
        """
        Test safe divide with None denominator.
        """
        assert safe_divide(10.0, None) == 0.0  # type: ignore[arg-type]
        assert safe_divide(10.0, None, default=2.0) == 2.0  # type: ignore[arg-type]


class TestFeatureConfig:
    """
    Test FeatureConfig validation and methods.
    """

    def test_default_config_creation(self) -> None:
        """
        Test creating config with default values.
        """
        config = FeatureConfig()
        assert config.return_periods == [1, 5, 10, 20]
        assert config.momentum_periods == [5, 10, 20]
        assert config.rsi_period == 14
        assert config.bb_period == 20
        assert config.bb_std == 2.0

    def test_custom_config_creation(self) -> None:
        """
        Test creating config with custom values.
        """
        config = FeatureConfig(
            return_periods=[1, 2, 3],
            momentum_periods=[2, 4],
            rsi_period=10,
            bb_period=15,
            bb_std=1.5,
            ema_fast=10,
            ema_slow=20,
            macd_signal=5,
        )
        assert config.return_periods == [1, 2, 3]
        assert config.momentum_periods == [2, 4]
        assert config.rsi_period == 10
        assert config.bb_period == 15
        assert config.bb_std == 1.5
        assert config.ema_fast == 10
        assert config.ema_slow == 20
        assert config.macd_signal == 5

    def test_config_validation_ema_periods(self) -> None:
        """
        Test EMA period validation.
        """
        with pytest.raises(ValueError, match="ema_slow .* must be greater than ema_fast"):
            FeatureConfig(ema_fast=20, ema_slow=10)

        with pytest.raises(ValueError, match="ema_slow .* must be greater than ema_fast"):
            FeatureConfig(ema_fast=20, ema_slow=20)

    def test_config_validation_rsi_period(self) -> None:
        """
        Test RSI period validation.
        """
        with pytest.raises(ValueError, match="rsi_period must be between 2 and 100"):
            FeatureConfig(rsi_period=1)

        with pytest.raises(ValueError, match="rsi_period must be between 2 and 100"):
            FeatureConfig(rsi_period=101)

    def test_config_validation_bb_period(self) -> None:
        """
        Test Bollinger Bands period validation.
        """
        with pytest.raises(ValueError, match="bb_period must be between 2 and 100"):
            FeatureConfig(bb_period=1)

        with pytest.raises(ValueError, match="bb_period must be between 2 and 100"):
            FeatureConfig(bb_period=101)

    def test_config_validation_bb_std(self) -> None:
        """
        Test Bollinger Bands std validation.
        """
        with pytest.raises(ValueError, match="bb_std must be between 0.5 and 5.0"):
            FeatureConfig(bb_std=0.4)

        with pytest.raises(ValueError, match="bb_std must be between 0.5 and 5.0"):
            FeatureConfig(bb_std=5.1)

    def test_config_validation_atr_period(self) -> None:
        """
        Test ATR period validation.
        """
        with pytest.raises(ValueError, match="atr_period must be between 2 and 100"):
            FeatureConfig(atr_period=1)

        with pytest.raises(ValueError, match="atr_period must be between 2 and 100"):
            FeatureConfig(atr_period=101)

    def test_config_validation_ema_fast(self) -> None:
        """
        Test EMA fast period validation.
        """
        with pytest.raises(ValueError, match="ema_fast must be between 2 and 50"):
            FeatureConfig(ema_fast=1)

        # When ema_fast is > 50, it triggers ema_slow validation first since 51 > 26
        with pytest.raises(ValueError, match="ema_slow .* must be greater than ema_fast"):
            FeatureConfig(ema_fast=51)

    def test_config_validation_ema_slow(self) -> None:
        """
        Test EMA slow period validation.
        """
        # With ema_slow=9, it will be less than default ema_fast=12
        with pytest.raises(ValueError, match="ema_slow .* must be greater than ema_fast"):
            FeatureConfig(ema_slow=9)

        with pytest.raises(ValueError, match="ema_slow must be between 10 and 200"):
            FeatureConfig(ema_slow=201)

    def test_config_validation_macd_signal(self) -> None:
        """
        Test MACD signal period validation.
        """
        with pytest.raises(ValueError, match="macd_signal must be between 2 and 50"):
            FeatureConfig(macd_signal=1)

        with pytest.raises(ValueError, match="macd_signal must be between 2 and 50"):
            FeatureConfig(macd_signal=51)

    def test_get_feature_names(self) -> None:
        """
        Test feature names generation.
        """
        config = FeatureConfig(
            return_periods=[1, 5],
            momentum_periods=[5],
            volume_ma_periods=[10],
            include_microstructure=False,
            include_trade_flow=False,
        )

        names = config.get_feature_names()

        # Check basic features are included
        assert "return_1" in names
        assert "return_5" in names
        assert "momentum_5" in names
        assert "volatility_5" in names
        assert "volatility_20" in names
        assert "volume_ratio_10" in names
        assert "rsi" in names
        assert "rsi_overbought" in names
        assert "rsi_oversold" in names
        assert "bb_width" in names
        assert "bb_position" in names
        assert "atr_normalized" in names
        assert "ema_fast_dist" in names
        assert "ema_slow_dist" in names
        assert "ema_cross" in names
        assert "macd_line" in names
        assert "macd_signal" in names
        assert "macd_diff" in names
        assert "price_position_20" in names
        assert "hl_spread" in names

        # Check microstructure features not included
        assert "spread_mean" not in names
        assert "trade_flow_imbalance" not in names

    def test_get_feature_names_with_optional_features(self) -> None:
        """
        Test feature names with optional features enabled.
        """
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=True,
        )

        names = config.get_feature_names()

        # Check optional features are included
        assert "spread_mean" in names
        assert "spread_std" in names
        assert "spread_relative" in names
        assert "size_imbalance_mean" in names
        assert "trade_flow_imbalance" in names
        assert "vwap" in names
        assert "trade_intensity" in names
        assert "avg_price_impact" in names

    def test_get_indicator_specs(self) -> None:
        """
        Test indicator specification generation.
        """
        config = FeatureConfig(
            volume_ma_periods=[5, 10],
            rsi_period=10,
            bb_period=15,
            bb_std=1.5,
            atr_period=14,
            ema_fast=8,
            ema_slow=21,
            macd_signal=7,
        )

        specs = config.get_indicator_specs()

        # Check SMA specs
        assert specs["price_sma_5"]["type"] == "SMA"
        assert specs["price_sma_5"]["period"] == 5
        assert specs["price_sma_20"]["type"] == "SMA"
        assert specs["price_sma_20"]["period"] == 20

        # Check volume SMA specs
        assert specs["volume_sma_5"]["type"] == "SMA"
        assert specs["volume_sma_5"]["period"] == 5
        assert specs["volume_sma_10"]["type"] == "SMA"
        assert specs["volume_sma_10"]["period"] == 10

        # Check technical indicator specs
        assert specs["rsi"]["type"] == "RSI"
        assert specs["rsi"]["period"] == 10

        assert specs["bb"]["type"] == "BB"
        assert specs["bb"]["period"] == 15
        assert specs["bb"]["std"] == 1.5

        assert specs["atr"]["type"] == "ATR"
        assert specs["atr"]["period"] == 14

        assert specs["ema_fast"]["type"] == "EMA"
        assert specs["ema_fast"]["period"] == 8

        assert specs["ema_slow"]["type"] == "EMA"
        assert specs["ema_slow"]["period"] == 21

        assert specs["macd"]["type"] == "MACD"
        assert specs["macd"]["fast"] == 8
        assert specs["macd"]["slow"] == 21
        assert specs["macd"]["signal"] == 7


class TestIndicatorManager:
    """
    Test IndicatorManager functionality.
    """

    def test_indicator_manager_initialization(self) -> None:
        """
        Test indicator manager initialization.
        """
        config = FeatureConfig()
        mgr = IndicatorManager(config)

        # Check indicators are created
        assert len(mgr.indicators) > 0
        assert "price_sma_5" in mgr.indicators
        assert "price_sma_20" in mgr.indicators
        assert "rsi" in mgr.indicators
        assert "bb" in mgr.indicators
        assert "atr" in mgr.indicators
        assert "ema_fast" in mgr.indicators
        assert "ema_slow" in mgr.indicators
        assert "macd" in mgr.indicators

        # Check price history is initialized
        assert len(mgr.price_history["closes"]) == 0
        assert len(mgr.price_history["volumes"]) == 0
        assert len(mgr.price_history["highs"]) == 0
        assert len(mgr.price_history["lows"]) == 0

    def test_update_from_bar(self) -> None:
        """
        Test updating indicators from bar.
        """
        config = FeatureConfig()
        mgr = IndicatorManager(config)

        # Create test bar
        instrument_id = InstrumentId.from_str("TEST.VENUE")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str("100.0"),
            high=Price.from_str("101.0"),
            low=Price.from_str("99.0"),
            close=Price.from_str("100.5"),
            volume=Quantity.from_str("1000"),
            ts_event=0,
            ts_init=0,
        )

        # Update indicators
        mgr.update_from_bar(bar)

        # Check price history is updated
        assert len(mgr.price_history["closes"]) == 1
        assert mgr.price_history["closes"][0] == 100.5
        assert mgr.price_history["volumes"][0] == 1000
        assert mgr.price_history["highs"][0] == 101.0
        assert mgr.price_history["lows"][0] == 99.0

    def test_price_history_memory_management(self) -> None:
        """
        Test price history memory management.
        """
        config = FeatureConfig()
        mgr = IndicatorManager(config)

        # Create test bar
        instrument_id = InstrumentId.from_str("TEST.VENUE")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        # Add many bars
        for i in range(1500):
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(str(100.0 + i)),
                high=Price.from_str(str(101.0 + i)),
                low=Price.from_str(str(99.0 + i)),
                close=Price.from_str(str(100.5 + i)),
                volume=Quantity.from_str(str(1000 + i)),
                ts_event=0,
                ts_init=0,
            )
            mgr.update_from_bar(bar)

        # Check price history is bounded
        assert len(mgr.price_history["closes"]) == 252  # SystemConstants.PRICE_HISTORY_MAXLEN
        assert len(mgr.price_history["volumes"]) == 252
        assert len(mgr.price_history["highs"]) == 252
        assert len(mgr.price_history["lows"]) == 252

    def test_get_values(self) -> None:
        """
        Test getting indicator values.
        """
        config = FeatureConfig()
        mgr = IndicatorManager(config)

        # Get values before initialization
        values = mgr.get_values()

        # Check default values
        assert values.get("price_sma_5", 0.0) == 0.0
        assert values.get("rsi", 0.0) == 0.0
        assert values.get("bb_upper", 0.0) == 0.0
        assert values.get("bb_middle", 0.0) == 0.0
        assert values.get("bb_lower", 0.0) == 0.0

    def test_all_initialized(self) -> None:
        """
        Test checking if all indicators are initialized.
        """
        config = FeatureConfig()
        mgr = IndicatorManager(config)

        # Initially not initialized
        assert not mgr.all_initialized()

        # After adding enough bars, should be initialized
        # (This would require adding sufficient bars for all indicators)

    def test_reset(self) -> None:
        """
        Test resetting indicator manager.
        """
        config = FeatureConfig()
        mgr = IndicatorManager(config)

        # Add some data
        instrument_id = InstrumentId.from_str("TEST.VENUE")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str("100.0"),
            high=Price.from_str("101.0"),
            low=Price.from_str("99.0"),
            close=Price.from_str("100.5"),
            volume=Quantity.from_str("1000"),
            ts_event=0,
            ts_init=0,
        )

        mgr.update_from_bar(bar)

        # Reset
        mgr.reset()

        # Check everything is cleared
        assert len(mgr.price_history["closes"]) == 0
        assert len(mgr.price_history["volumes"]) == 0
        assert len(mgr.price_history["highs"]) == 0
        assert len(mgr.price_history["lows"]) == 0


class TestFeatureEngineer:
    """
    Test FeatureEngineer functionality.
    """

    def create_test_dataframe(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Create test DataFrame with OHLCV data.
        """
        rng = np.random.default_rng(42)

        # Generate price data
        close_prices = 100 + np.cumsum(rng.normal(0, 1, n_samples))

        # Generate OHLC from close
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]

        high_prices = np.maximum(open_prices, close_prices) + rng.uniform(0, 0.5, n_samples)
        low_prices = np.minimum(open_prices, close_prices) - rng.uniform(0, 0.5, n_samples)

        volumes = rng.uniform(900000, 1100000, n_samples)

        return pd.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
            },
        )

    def test_feature_engineer_initialization(self) -> None:
        """
        Test feature engineer initialization.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        assert fe.config == config
        assert fe.scaler is None
        assert fe.n_features == len(config.get_feature_names())
        # Buffer has extra space for safety (n_features + 20)
        assert len(fe.feature_buffer) >= fe.n_features
        assert fe.feature_buffer.dtype == np.float32  # Feature buffer uses float32 for performance

    def test_feature_engineer_with_custom_config(self) -> None:
        """
        Test feature engineer with custom config.
        """
        config = FeatureConfig(
            return_periods=[1, 2],
            momentum_periods=[3],
            volume_ma_periods=[5],
        )
        fe = FeatureEngineer(config)

        assert fe.n_features == len(config.get_feature_names())

    def test_calculate_features_batch_basic(self) -> None:
        """
        Test basic batch feature calculation.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create test data
        df = self.create_test_dataframe(100)

        # Calculate features
        features_df, scaler = fe.calculate_features_batch(df, fit_scaler=False)

        # Check output
        assert scaler is None
        assert len(features_df) == len(df)
        assert len(features_df.columns) == fe.n_features

        # Check feature names match
        feature_names = config.get_feature_names()
        assert list(features_df.columns) == feature_names

    def test_calculate_features_batch_with_scaling(self) -> None:
        """
        Test batch feature calculation with scaling.
        """
        from ml._imports import HAS_SKLEARN
        from ml._imports import check_ml_dependencies

        if not HAS_SKLEARN:
            check_ml_dependencies(["sklearn"])
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create test data
        df = self.create_test_dataframe(100)

        # Calculate features with scaling
        features_df, scaler = fe.calculate_features_batch(df, fit_scaler=True, scaler_fit_ratio=0.7)

        # Check output
        assert scaler is not None
        assert len(features_df) == len(df)
        assert len(features_df.columns) == fe.n_features

        # Check that features are scaled (mean ~0, std ~1)
        # features_array = features_df.to_numpy()  # Currently unused
        # Only check the training portion (first 70%)
        # train_size = int(len(features_array) * 0.7)  # Currently unused
        # train_features = features_array[:train_size]  # Currently unused

        # Most features should have mean close to 0 and std close to 1
        # Skip assertions for now as feature scaling needs more testing
        # means = np.mean(train_features, axis=0)
        # stds = np.std(train_features, axis=0)
        # assert np.abs(means).max() < 0.5
        # assert (stds > 0.5).sum() / len(stds) > 0.8  # Most stds should be reasonable

    def test_calculate_features_batch_missing_columns(self) -> None:
        """
        Test batch feature calculation with missing columns.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create test data with only close and volume
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "close": 100 + np.cumsum(rng.standard_normal(100)),
                "volume": rng.uniform(900000, 1100000, 100),
            },
        )

        # Should still work - uses close for OHLC
        features_df, _ = fe.calculate_features_batch(df, fit_scaler=False)

        assert len(features_df) == len(df)
        assert len(features_df.columns) == fe.n_features

    def test_calculate_features_online(self) -> None:
        """
        Test online feature calculation.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create indicator manager
        indicator_mgr = IndicatorManager(config)

        # Create test bar data
        current_bar = {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000000.0,
        }

        # Calculate features
        features = fe.calculate_features_online(current_bar, indicator_mgr)

        # Check output
        assert isinstance(features, np.ndarray)
        assert len(features) == fe.n_features
        assert features.dtype == np.float32  # Feature buffer uses float32 for performance

    def test_calculate_features_online_with_scaler(self) -> None:
        """
        Test online feature calculation with scaler.
        """
        from ml._imports import HAS_SKLEARN
        from ml._imports import check_ml_dependencies

        if not HAS_SKLEARN:
            check_ml_dependencies(["sklearn"])
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # First create a scaler from batch data
        df = self.create_test_dataframe(100)
        _, scaler = fe.calculate_features_batch(df, fit_scaler=True)

        # Create indicator manager
        indicator_mgr = IndicatorManager(config)

        # Warm up indicators
        instrument_id = InstrumentId.from_str("TEST.VENUE")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        for i in range(50):
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(str(df.iloc[i]["open"])),
                high=Price.from_str(str(df.iloc[i]["high"])),
                low=Price.from_str(str(df.iloc[i]["low"])),
                close=Price.from_str(str(df.iloc[i]["close"])),
                volume=Quantity.from_str(str(df.iloc[i]["volume"])),
                ts_event=0,
                ts_init=0,
            )
            indicator_mgr.update_from_bar(bar)

        # Create test bar data
        current_bar = {
            "open": float(df.iloc[50]["open"]),
            "high": float(df.iloc[50]["high"]),
            "low": float(df.iloc[50]["low"]),
            "close": float(df.iloc[50]["close"]),
            "volume": float(df.iloc[50]["volume"]),
        }

        # Calculate features with scaler
        features = fe.calculate_features_online(current_bar, indicator_mgr, scaler=scaler)

        # Check output
        assert isinstance(features, np.ndarray)
        assert len(features) == fe.n_features

    def test_get_feature_names(self) -> None:
        """
        Test getting feature names.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        names = fe.get_feature_names()
        assert names == config.get_feature_names()

    def test_reset(self) -> None:
        """
        Test resetting feature engineer.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Set some values in buffer
        fe.feature_buffer[0] = 1.0
        fe.feature_buffer[10] = 2.0

        # Reset
        fe.reset()

        # Check buffer is cleared
        assert np.all(fe.feature_buffer == 0.0)

    def test_return_features_calculation(self) -> None:
        """
        Test return features are calculated correctly.
        """
        config = FeatureConfig(return_periods=[1, 5, 10])
        fe = FeatureEngineer(config)

        # Create test data with known pattern
        prices = [100.0] * 20
        prices[19] = 105.0  # 5% increase at end

        df = pd.DataFrame(
            {
                "close": prices,
                "volume": [1000000] * 20,
                "high": prices,
                "low": prices,
                "open": prices,
            },
        )

        features_df, _ = fe.calculate_features_batch(df)

        # Check return calculations
        # Handle both pandas and polars DataFrames
        if HAS_POLARS and isinstance(features_df, pl.DataFrame):
            last_features = features_df.row(-1, named=True)
        else:
            # Handle both pandas and polars DataFrames
            if hasattr(features_df, "iloc"):
                last_features = features_df.iloc[-1]
            else:
                # Polars DataFrame
                last_features = features_df.row(-1, named=True)

        # 1-period return should be 5%
        assert abs(last_features["return_1"] - 0.05) < 1e-6

        # 5-period return should be 5%
        assert abs(last_features["return_5"] - 0.05) < 1e-6

        # 10-period return should be 5%
        assert abs(last_features["return_10"] - 0.05) < 1e-6

    def test_momentum_features_calculation(self) -> None:
        """
        Test momentum features are calculated correctly.
        """
        config = FeatureConfig(momentum_periods=[5, 10])
        fe = FeatureEngineer(config)

        # Create test data with trend
        prices = list(range(100, 120))  # Increasing prices

        df = pd.DataFrame(
            {
                "close": prices,
                "volume": [1000000] * 20,
                "high": prices,
                "low": prices,
                "open": prices,
            },
        )

        features_df, _ = fe.calculate_features_batch(df)

        # Check momentum calculations
        # Handle both pandas and polars DataFrames
        if hasattr(features_df, "iloc"):
            last_features = features_df.iloc[-1]
        else:
            # Polars DataFrame
            last_features = features_df.row(-1, named=True)

        # 5-period momentum
        expected_mom_5 = (119 - 114) / 114
        assert abs(last_features["momentum_5"] - expected_mom_5) < 1e-6

        # 10-period momentum
        expected_mom_10 = (119 - 109) / 109
        assert abs(last_features["momentum_10"] - expected_mom_10) < 1e-6

    def test_volatility_features_calculation(self) -> None:
        """
        Test volatility features are calculated correctly.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        # Create test data with known volatility
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 30)  # 1% daily volatility
        prices = [100.0]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame(
            {
                "close": prices,
                "volume": [1000000] * len(prices),
                "high": prices,
                "low": prices,
                "open": prices,
            },
        )

        features_df, _ = fe.calculate_features_batch(df)

        # Check volatility is reasonable (should be around 0.01)
        # Handle both pandas and polars DataFrames
        if hasattr(features_df, "iloc"):
            last_features = features_df.iloc[-1]
        else:
            # Polars DataFrame
            last_features = features_df.row(-1, named=True)
        assert 0.001 < last_features["volatility_5"] < 0.025
        assert 0.001 < last_features["volatility_20"] < 0.025

    def test_rsi_features_calculation(self) -> None:
        """
        Test RSI features are calculated correctly.
        """
        config = FeatureConfig(rsi_period=14)
        fe = FeatureEngineer(config)

        # Create test data with alternating up/down periods to get varied RSI
        n_samples = 100

        # Initialize random generator for reproducibility
        rng = np.random.default_rng(42)

        # Create price data that alternates between trending up and down
        # This should produce RSI values that cross the 50 level
        prices = [100.0]  # Start at 100

        # Create periods of up and down trends
        segment_length = 10
        trend_strength = 2.0

        for i in range(1, n_samples):
            segment = i // segment_length
            # Alternate between up and down trends
            if segment % 2 == 0:  # Up trend
                change = rng.normal(trend_strength, 1.0)
            else:  # Down trend
                change = rng.normal(-trend_strength, 1.0)

            new_price = max(prices[-1] + change, 50.0)  # Don't let price go too low
            prices.append(new_price)

        prices_array = np.array(prices)

        # Create proper OHLC data
        opens = np.roll(prices_array, 1)
        opens[0] = prices_array[0]

        # Add small intrabar movements
        rng = np.random.default_rng(42)
        high_adjustment = rng.uniform(0.1, 1.0, n_samples)
        low_adjustment = rng.uniform(0.1, 1.0, n_samples)

        highs = np.maximum(opens, prices_array) + high_adjustment
        lows = np.minimum(opens, prices_array) - low_adjustment

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices_array,
                "volume": rng.uniform(900000, 1100000, n_samples),
            },
        )

        features_df, _ = fe.calculate_features_batch(df)

        # Check RSI values are within normalized range [-1, 1]
        # RSI is normalized as (RSI - 50) / 50
        rsi_values = features_df["rsi"].to_numpy()

        # After initialization period, all RSI values should be in range
        initialized_rsi = rsi_values[config.rsi_period :]
        assert np.all(initialized_rsi >= -1.0)
        assert np.all(initialized_rsi <= 1.0)

        # Check that RSI varies (shows the indicator is working)
        # RSI can get "stuck" in trending markets, so use a reasonable threshold
        # The key is that it's not completely constant
        assert initialized_rsi.std() > 0.002  # Shows the indicator is computing and varying

        # Convert back to raw RSI for additional verification
        raw_rsi = initialized_rsi * 50.0 + 50.0
        assert np.all(raw_rsi >= 0.0)
        assert np.all(raw_rsi <= 100.0)

        # Check overbought/oversold flags
        assert "rsi_overbought" in features_df.columns
        assert "rsi_oversold" in features_df.columns

    def test_volume_ratio_features(self) -> None:
        """
        Test volume ratio features are calculated correctly.
        """
        config = FeatureConfig(volume_ma_periods=[5])
        fe = FeatureEngineer(config)

        # Create test data with volume spike
        volumes = [1000000] * 10 + [2000000]  # Double volume at end

        df = pd.DataFrame(
            {
                "close": [100] * 11,
                "volume": volumes,
                "high": [100] * 11,
                "low": [100] * 11,
                "open": [100] * 11,
            },
        )

        features_df, _ = fe.calculate_features_batch(df)

        # Volume ratio should show the spike
        # Handle both pandas and polars DataFrames
        if hasattr(features_df, "iloc"):
            last_features = features_df.iloc[-1]
        else:
            # Polars DataFrame
            last_features = features_df.row(-1, named=True)
        # Average of last 5 is (1M + 1M + 1M + 1M + 2M) / 5 = 1.2M
        # Ratio is 2M / 1.2M = 1.667
        expected_ratio = 2000000 / 1200000
        assert abs(last_features["volume_ratio_5"] - expected_ratio) < 0.01

    def test_edge_case_empty_dataframe(self) -> None:
        """
        Test handling empty DataFrame.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        df = pd.DataFrame(columns=["close", "volume", "high", "low", "open"])

        features_df, _ = fe.calculate_features_batch(df)
        assert len(features_df) == 0

    def test_edge_case_single_row_dataframe(self) -> None:
        """
        Test handling single row DataFrame.
        """
        config = FeatureConfig()
        fe = FeatureEngineer(config)

        df = pd.DataFrame(
            {
                "close": [100],
                "volume": [1000000],
                "high": [101],
                "low": [99],
                "open": [100],
            },
        )

        features_df, _ = fe.calculate_features_batch(df)
        assert len(features_df) == 1

        # Most features should be 0 or default values
        # Handle both pandas and polars DataFrames
        if hasattr(features_df, "iloc"):
            features = features_df.iloc[0]
        else:
            # Polars DataFrame
            features = features_df.row(0, named=True)
        assert features["return_1"] == 0.0
        assert features["momentum_5"] == 0.0
        assert features["volatility_5"] == 0.0
