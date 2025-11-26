"""
Property tests for FeatureCalculator component (Phase 2.1.4).

Uses Hypothesis to verify invariants across wide input space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from ml.features.common.feature_calculator import FeatureCalculator
from ml.features.engineering import FeatureConfig, IndicatorManager


# ==================== Module-Level Config (avoid function-scoped fixture with Hypothesis) ====================


def get_feature_config():
    """Create FeatureConfig for property tests (not a fixture to avoid Hypothesis conflicts)."""
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


# ==================== Hypothesis Strategies ====================


# Valid price range (avoid extreme values that cause overflow)
price_strategy = st.floats(
    min_value=1.0,
    max_value=10000.0,
    allow_nan=False,
    allow_infinity=False,
)

# Valid volume range
volume_strategy = st.floats(
    min_value=100.0,
    max_value=1e12,
    allow_nan=False,
    allow_infinity=False,
)

# Bar count for sequences
bar_count_strategy = st.integers(min_value=50, max_value=150)


# ==================== Property Tests ====================


@pytest.mark.property
class TestFeatureCalculatorProperties:
    """Property test suite for FeatureCalculator invariants."""

    @settings(max_examples=50, deadline=None)
    @given(
        bar_count=bar_count_strategy,
        close_base=st.floats(min_value=10.0, max_value=500.0, allow_nan=False),
    )
    def test_features_no_nan_invariant(self, bar_count, close_base):
        """
        Property: For any valid bar data with sufficient history → output has no NaN.
        """
        # Generate bar sequence
        np.random.seed(hash((bar_count, close_base)) % 2**32)

        close_prices = close_base + np.cumsum(np.random.randn(bar_count) * 0.01 * close_base)
        high_prices = close_prices + np.abs(np.random.randn(bar_count) * 0.005 * close_base)
        low_prices = close_prices - np.abs(np.random.randn(bar_count) * 0.005 * close_base)
        volumes = np.random.uniform(100.0, 1e9, bar_count)

        df = pd.DataFrame(
            {
                "close": close_prices,
                "high": high_prices,
                "low": low_prices,
                "volume": volumes,
            }
        )

        feature_config = get_feature_config()
        calculator = FeatureCalculator(config=feature_config)

        # Calculate features
        features, _ = calculator.calculate_features(df, mode="batch")

        # Property: No NaN after warmup period (first 20 bars)
        assert not features.iloc[20:].isna().any().any(), "Features contain NaN after warmup"

    @settings(max_examples=50, deadline=None)
    @given(
        bar_count=bar_count_strategy,
        close_base=st.floats(min_value=10.0, max_value=500.0, allow_nan=False),
    )
    def test_features_no_inf_invariant(self, bar_count, close_base):
        """
        Property: For any valid bar data → output has no Inf.
        """
        np.random.seed(hash((bar_count, close_base)) % 2**32)

        close_prices = close_base + np.cumsum(np.random.randn(bar_count) * 0.01 * close_base)
        high_prices = close_prices + np.abs(np.random.randn(bar_count) * 0.005 * close_base)
        low_prices = close_prices - np.abs(np.random.randn(bar_count) * 0.005 * close_base)
        volumes = np.random.uniform(100.0, 1e9, bar_count)

        df = pd.DataFrame(
            {
                "close": close_prices,
                "high": high_prices,
                "low": low_prices,
                "volume": volumes,
            }
        )

        feature_config = get_feature_config()
        calculator = FeatureCalculator(config=feature_config)

        features, _ = calculator.calculate_features(df, mode="batch")

        # Property: No Inf anywhere
        assert not np.isinf(features.to_numpy()).any(), "Features contain Inf"

    @settings(max_examples=30, deadline=None)
    @given(bar_count=bar_count_strategy)
    def test_features_shape_invariant(self, bar_count):
        """
        Property: Output shape matches input shape and feature count.
        """
        np.random.seed(bar_count)

        df = pd.DataFrame(
            {
                "close": 100.0 + np.cumsum(np.random.randn(bar_count) * 0.5),
                "high": 101.0 + np.cumsum(np.random.randn(bar_count) * 0.5),
                "low": 99.0 + np.cumsum(np.random.randn(bar_count) * 0.5),
                "volume": np.random.uniform(900000, 1100000, bar_count),
            }
        )

        feature_config = get_feature_config()
        calculator = FeatureCalculator(config=feature_config)

        features, _ = calculator.calculate_features(df, mode="batch")

        # Property: Output row count matches input
        assert len(features) == bar_count, "Output row count should match input"

        # Property: Output column count matches n_features
        assert (
            len(features.columns) == calculator.n_features
        ), "Output column count should match n_features"

    @settings(max_examples=20, deadline=None)
    @given(
        bar_count=st.integers(min_value=50, max_value=100),
        close_base=st.floats(min_value=10.0, max_value=500.0, allow_nan=False),
    )
    def test_batch_online_parity_property(self, bar_count, close_base):
        """
        Property: For any bar sequence → batch and online modes produce identical results.
        """
        np.random.seed(hash((bar_count, close_base)) % 2**32)

        close_prices = close_base + np.cumsum(np.random.randn(bar_count) * 0.01 * close_base)
        high_prices = close_prices + np.abs(np.random.randn(bar_count) * 0.005 * close_base)
        low_prices = close_prices - np.abs(np.random.randn(bar_count) * 0.005 * close_base)
        volumes = np.random.uniform(100.0, 1e9, bar_count)

        df = pd.DataFrame(
            {
                "close": close_prices,
                "high": high_prices,
                "low": low_prices,
                "volume": volumes,
            }
        )

        feature_config = get_feature_config()
        calculator = FeatureCalculator(config=feature_config)

        # Batch mode
        batch_features, _ = calculator.calculate_features(df, mode="batch")

        # Online mode (row by row)
        online_features = []
        indicator_mgr = IndicatorManager(feature_config)

        for _, row in df.iterrows():
            indicator_mgr.update_from_values(
                close=row["close"],
                high=row["high"],
                low=row["low"],
                volume=row["volume"],
            )

            bar_dict = {
                "close": row["close"],
                "high": row["high"],
                "low": row["low"],
                "volume": row["volume"],
            }

            features = calculator.calculate_features(
                bar_dict, mode="online", indicator_manager=indicator_mgr
            )
            online_features.append(features)

        online_features_array = np.array(online_features)

        # Property: MUST be identical (batch == online)
        np.testing.assert_allclose(
            batch_features.to_numpy(),
            online_features_array,
            rtol=1e-10,
            atol=1e-12,
            err_msg="BATCH/ONLINE PARITY VIOLATED - ML model will have train/inference mismatch!",
        )

    @settings(max_examples=30, deadline=None)
    @given(bar_count=st.integers(min_value=50, max_value=150))
    def test_features_bounded_invariant(self, bar_count):
        """
        Property: Feature values stay within reasonable bounds.
        """
        np.random.seed(bar_count)

        close_prices = 100.0 + np.cumsum(np.random.randn(bar_count) * 0.5)
        high_prices = close_prices + np.abs(np.random.randn(bar_count) * 0.3)
        low_prices = close_prices - np.abs(np.random.randn(bar_count) * 0.3)
        volumes = np.random.uniform(100.0, 1e9, bar_count)

        df = pd.DataFrame(
            {
                "close": close_prices,
                "high": high_prices,
                "low": low_prices,
                "volume": volumes,
            }
        )

        feature_config = get_feature_config()
        calculator = FeatureCalculator(config=feature_config)

        features, _ = calculator.calculate_features(df, mode="batch")

        # Property: Volatility >= 0
        volatility_cols = [col for col in features.columns if "volatility" in col]
        if volatility_cols:
            assert (features[volatility_cols] >= 0).all().all(), "Volatility should be non-negative"

        # Property: Volume ratios > 0 (after warmup)
        volume_ratio_cols = [col for col in features.columns if "volume_ratio" in col]
        if volume_ratio_cols:
            # Allow zeros during warmup, but should be positive after
            assert (
                features[volume_ratio_cols].iloc[20:] > 0
            ).all().all(), "Volume ratios should be positive after warmup"

    @settings(max_examples=20, deadline=None)
    @given(n_calls=st.integers(min_value=10, max_value=50))
    def test_feature_buffer_no_corruption_property(self, n_calls):
        """
        Property: After calculation, feature_buffer has no leftover NaN/Inf from previous calls.
        """
        feature_config = get_feature_config()
        calculator = FeatureCalculator(config=feature_config)

        # Create indicator manager with history
        indicator_mgr = IndicatorManager(feature_config)
        for i in range(50):
            indicator_mgr.update_from_values(
                close=100.0 + i * 0.1,
                high=101.0 + i * 0.1,
                low=99.0 + i * 0.1,
                volume=1000000.0 + i * 1000,
            )

        # Make consecutive calls
        np.random.seed(n_calls)
        for i in range(n_calls):
            bar_dict = {
                "close": 100.0 + i * 0.1 + np.random.randn() * 0.05,
                "high": 101.0 + i * 0.1 + np.random.randn() * 0.05,
                "low": 99.0 + i * 0.1 + np.random.randn() * 0.05,
                "volume": 1000000.0 + np.random.uniform(-10000, 10000),
            }

            features = calculator.calculate_features(
                bar_dict, mode="online", indicator_manager=indicator_mgr
            )

            # Property: No NaN, no Inf in buffer
            assert not np.isnan(features).any(), f"Call {i}: features contain NaN"
            assert not np.isinf(features).any(), f"Call {i}: features contain Inf"


# ==================== Summary ====================

"""
Property Test Coverage:
- No NaN invariant: 1 test (50 examples)
- No Inf invariant: 1 test (50 examples)
- Shape invariant: 1 test (30 examples)
- Batch/online parity property: 1 test (20 examples) - CRITICAL
- Bounded values property: 1 test (30 examples)
- Buffer integrity property: 1 test (20 examples)

Total: 6 property tests

This comprehensive property test suite validates invariants across wide input spaces
using Hypothesis-generated examples, ensuring robustness and ML model correctness.
"""
