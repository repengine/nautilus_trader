"""
Integration tests for FeatureCalculator component (Phase 2.1.4).

CRITICAL TEST: Batch/Online Parity - ML models depend on this!
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.common.feature_calculator import FeatureCalculator
from ml.features.engineering import FeatureConfig, IndicatorManager


# ==================== Fixtures ====================


@pytest.fixture
def feature_config():
    """Standard FeatureConfig for integration tests."""
    return FeatureConfig(
        return_periods=[1, 2, 5],
        momentum_periods=[1, 3],
        volume_ma_periods=[10, 20],
        ema_fast=12,
        ema_slow=26,
        rsi_period=14,
        bb_period=20,
        bb_std=2.0,
        atr_period=14,
        enable_returns=True,
        enable_momentum=True,
        enable_volatility=True,
        enable_technical=True,
        include_microstructure=False,
        include_trade_flow=False,
    )


@pytest.fixture
def realistic_market_data_100_bars():
    """Generate 100 bars of realistic market data."""
    np.random.seed(123)

    # Simulate realistic price walk with trend + noise
    n_bars = 100
    base_price = 100.0
    trend = 0.001  # Slight upward trend
    volatility = 0.005

    prices = [base_price]
    for i in range(1, n_bars):
        change = np.random.normal(trend, volatility)
        prices.append(prices[-1] * (1 + change))

    close_prices = np.array(prices)
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.002, n_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.002, n_bars)))
    open_prices = close_prices + np.random.normal(0, 0.001, n_bars) * close_prices
    volumes = np.random.lognormal(mean=13.8, sigma=0.3, size=n_bars)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n_bars, freq="1min"),
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        }
    )


# ==================== Integration Tests ====================


@pytest.mark.integration
class TestFeatureCalculatorIntegration:
    """Integration test suite for FeatureCalculator."""

    def test_feature_calculator_with_realistic_market_data(
        self, feature_config, realistic_market_data_100_bars
    ):
        """End-to-end test with realistic OHLCV data."""
        calculator = FeatureCalculator(config=feature_config)

        features_df, _ = calculator.calculate_features(
            realistic_market_data_100_bars, mode="batch"
        )

        # Verify output shape
        assert len(features_df) == 100, "Output should have 100 rows"
        assert len(features_df.columns) == calculator.n_features, "Column count should match n_features"

        # Verify no NaN after warmup period (first 20 bars may be zero-padded)
        assert not features_df.iloc[20:].isna().any().any(), "Features contain NaN after warmup"

        # Verify no Inf
        assert not np.isinf(features_df.to_numpy()).any(), "Features contain Inf"

        # Verify feature value sanity checks
        # Returns should be small (realistic price changes)
        return_cols = [col for col in features_df.columns if "return" in col]
        if return_cols:
            assert (
                np.abs(features_df[return_cols].iloc[20:]).max().max() < 1.0
            ), "Returns too large (> 100%)"

        # Volatility should be >= 0
        volatility_cols = [col for col in features_df.columns if "volatility" in col]
        if volatility_cols:
            assert (features_df[volatility_cols] >= 0).all().all(), "Volatility should be non-negative"

        # Volume ratios should be > 0
        volume_ratio_cols = [col for col in features_df.columns if "volume_ratio" in col]
        if volume_ratio_cols:
            assert (
                features_df[volume_ratio_cols].iloc[20:] > 0
            ).all().all(), "Volume ratios should be positive"

    def test_feature_calculator_batch_online_parity(
        self, feature_config, realistic_market_data_100_bars
    ):
        """
        CRITICAL TEST: Verify batch and online modes produce IDENTICAL results.

        ML model performance depends on exact parity between training (batch) and
        inference (online) features. Any divergence causes train/test mismatch.
        """
        calculator = FeatureCalculator(config=feature_config)

        # Batch mode: process all bars at once
        batch_features_df, _ = calculator.calculate_features(
            realistic_market_data_100_bars, mode="batch"
        )

        # Online mode: process bars one by one (simulating real-time inference)
        online_features_list = []
        indicator_mgr = IndicatorManager(feature_config)

        for idx, row in realistic_market_data_100_bars.iterrows():
            # Update indicator manager with current bar
            indicator_mgr.update_from_values(
                close=row["close"],
                high=row["high"],
                low=row["low"],
                volume=row["volume"],
            )

            # Calculate features for current bar
            bar_dict = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }

            features = calculator.calculate_features(
                bar_dict, mode="online", indicator_manager=indicator_mgr
            )

            # Copy to avoid reusing the calculator's internal buffer across iterations
            online_features_list.append(features.copy())

        # Convert online results to numpy array
        online_features_array = np.array(online_features_list)

        # CRITICAL ASSERTION: Batch and online MUST be numerically identical
        np.testing.assert_allclose(
            batch_features_df.to_numpy(),
            online_features_array,
            rtol=1e-10,
            atol=1e-12,
            err_msg="BATCH/ONLINE PARITY VIOLATED - ML model will have train/inference mismatch!",
        )

    def test_feature_calculator_with_minimal_config(self, realistic_market_data_100_bars):
        """Test calculator handles minimal config (most features disabled)."""
        minimal_config = FeatureConfig(
            return_periods=[],
            momentum_periods=[],
            volume_ma_periods=[10],
            enable_returns=False,
            enable_momentum=False,
            enable_volatility=False,
            enable_technical=False,
        )

        calculator = FeatureCalculator(config=minimal_config)

        features_df, _ = calculator.calculate_features(
            realistic_market_data_100_bars, mode="batch"
        )

        # Should only have volume ratio features
        assert len(features_df.columns) == 1, "Should only have 1 volume ratio feature"
        assert "volume_ratio_10" in features_df.columns, "Should have volume_ratio_10"

        # All values should be valid floats
        assert not features_df.isna().any().any(), "Features contain NaN"
        assert not np.isinf(features_df.to_numpy()).any(), "Features contain Inf"


# ==================== Summary ====================

"""
Integration Test Coverage:
- Realistic market data integration: 1 test
- Batch/online parity (CRITICAL): 1 test
- Minimal config handling: 1 test

Total: 3 integration tests

The batch/online parity test is CRITICAL for ML model correctness and must pass
with strict numerical tolerance (rtol=1e-10).
"""
