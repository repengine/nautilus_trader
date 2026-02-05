"""Parity tests for FeatureEngineer facade vs calculator component.

CRITICAL: These tests verify that the calculator and facade implementations
produce IDENTICAL numerical results. This is essential for ML training/inference parity.

VALUE TESTING PATTERN (Task 1.1b - 2025-12-01):
All parity tests compare numerical VALUES, not container types.
compute_features() returns dict[str, float], so we iterate over keys.

Test Strategy:
- Run same test data through facade and the FeatureCalculator component
- Assert numerical parity by comparing dict VALUES (rtol=1e-10)
- Test multiple configurations
- Test edge cases (empty, single bar, etc.)

"""

from __future__ import annotations

import time

import numpy as np
import pytest

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

from ml.features.common.feature_calculator import FeatureCalculator
from ml.features.config import FeatureConfig
from ml.features.facade import FeatureEngineer


pytestmark = [pytest.mark.parity, pytest.mark.unit]

# Note: Main parity tests pass after Task 1.1c (RSI, hl_spread post-processing).
# Edge case tests with insufficient data (single bar, short lookbacks) may still
# fail due to volume_ratio_20 fallback differences - these are marked xfail individually.


# ==================== Helper Functions ====================


def assert_dict_features_parity(
    baseline_features: dict[str, float],
    facade_features: dict[str, float],
    rtol: float = 1e-10,
    context: str = "",
) -> None:
    """Assert that two feature dicts have identical values.

    This is the VALUE TESTING pattern - compare VALUES, not container types.

    Args:
        baseline_features: Features from calculator component
        facade_features: Features from facade implementation
        rtol: Relative tolerance for floating point comparison
        context: Context string for error messages

    Raises:
        AssertionError: If feature names or values differ

    """
    # Key parity
    assert set(baseline_features.keys()) == set(facade_features.keys()), (
        f"Feature names differ {context}: "
        f"baseline={sorted(baseline_features.keys())} vs facade={sorted(facade_features.keys())}"
    )

    # Value parity (CRITICAL)
    for feature_name in baseline_features:
        np.testing.assert_allclose(
            baseline_features[feature_name],
            facade_features[feature_name],
            rtol=rtol,
            err_msg=f"Feature {feature_name!r} parity failed {context}",
        )


def _build_calculator(config: FeatureConfig) -> FeatureCalculator:
    """Construct a FeatureCalculator for parity checks."""
    return FeatureCalculator(config)


# ==================== Fixtures ====================


@pytest.fixture
def test_bars_200() -> list[Bar]:
    """Generate 200 test bars for lookback tests."""
    np.random.seed(42)
    bar_type = BarType.from_str("SPY.NYSE-1-MINUTE-LAST-EXTERNAL")
    base_ts = 1609459200000000000

    bars = []
    price = 100.0

    for i in range(200):
        price_change = np.random.randn() * 0.5
        close = price + price_change
        open_price = close + np.random.randn() * 0.2

        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(open_price, close) + abs(np.random.randn() * 0.3)
        low = min(open_price, close) - abs(np.random.randn() * 0.3)

        price = close  # Update for next bar

        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{open_price:.2f}"),
                high=Price.from_str(f"{high:.2f}"),
                low=Price.from_str(f"{low:.2f}"),
                close=Price.from_str(f"{close:.2f}"),
                volume=Quantity.from_str("1000000"),
                ts_event=base_ts + i * 60_000_000_000,
                ts_init=base_ts + i * 60_000_000_000 + 1,
            )
        )

    return bars


@pytest.fixture
def test_bars_1000() -> list[Bar]:
    """Generate 1000 test bars for performance tests."""
    np.random.seed(42)
    bar_type = BarType.from_str("SPY.NYSE-1-MINUTE-LAST-EXTERNAL")
    base_ts = 1609459200000000000

    bars = []
    price = 100.0

    for i in range(1000):
        price_change = np.random.randn() * 0.5
        close = price + price_change
        open_price = close + np.random.randn() * 0.2

        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(open_price, close) + abs(np.random.randn() * 0.3)
        low = min(open_price, close) - abs(np.random.randn() * 0.3)

        price = close  # Update for next bar

        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{open_price:.2f}"),
                high=Price.from_str(f"{high:.2f}"),
                low=Price.from_str(f"{low:.2f}"),
                close=Price.from_str(f"{close:.2f}"),
                volume=Quantity.from_str("1000000"),
                ts_event=base_ts + i * 60_000_000_000,
                ts_init=base_ts + i * 60_000_000_000 + 1,
            )
        )

    return bars


@pytest.fixture
def multi_instrument_bars() -> dict[str, list[Bar]]:
    """Generate bars for multiple instruments."""
    np.random.seed(42)
    base_ts = 1609459200000000000
    result: dict[str, list[Bar]] = {}

    for symbol in ["SPY.NYSE", "QQQ.NYSE", "IWM.NYSE"]:
        bar_type = BarType.from_str(f"{symbol}-1-MINUTE-LAST-EXTERNAL")
        bars = []
        price = 100.0 if symbol == "SPY.NYSE" else (300.0 if symbol == "QQQ.NYSE" else 200.0)

        for i in range(50):
            price_change = np.random.randn() * 0.5
            close = price + price_change
            open_price = close + np.random.randn() * 0.2

            # Ensure high >= max(open, close) and low <= min(open, close)
            high = max(open_price, close) + abs(np.random.randn() * 0.3)
            low = min(open_price, close) - abs(np.random.randn() * 0.3)

            price = close  # Update for next bar

            bars.append(
                Bar(
                    bar_type=bar_type,
                    open=Price.from_str(f"{open_price:.2f}"),
                    high=Price.from_str(f"{high:.2f}"),
                    low=Price.from_str(f"{low:.2f}"),
                    close=Price.from_str(f"{close:.2f}"),
                    volume=Quantity.from_str("1000000"),
                    ts_event=base_ts + i * 60_000_000_000,
                    ts_init=base_ts + i * 60_000_000_000 + 1,
                )
            )

        result[symbol] = bars

    return result


# ==================== Test Class ====================


class TestFeatureEngineerParity:
    """Test mathematical parity between calculator and facade implementations.

    VALUE TESTING: All tests compare dict VALUES, not container types.
    """

    def test_compute_features_matches_calculator_single_bar(
        self,
        feature_config: FeatureConfig,
        test_bar: Bar,
    ) -> None:
        """Verify calculator and facade produce IDENTICAL features for a single bar.

        This is the most basic parity test - if this fails, facade is broken.

        VALUE TESTING: compare dict values via loop, not direct comparison.
        """
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        # Calculate features with both implementations
        facade_features = facade.compute_features([test_bar])
        calculator_features = calculator.compute_features([test_bar])

        # Assert VALUE parity (not container type)
        assert_dict_features_parity(
            calculator_features,
            facade_features,
            context="single bar",
        )

    def test_compute_features_matches_calculator_100_bars(
        self,
        feature_config: FeatureConfig,
        test_bars: list[Bar],
    ) -> None:
        """Verify parity over realistic data volume (100 bars).

        Tests that parity holds over a realistic workload, not just single bars.
        """
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        # Calculate features with both implementations
        facade_features = facade.compute_features(test_bars)
        calculator_features = calculator.compute_features(test_bars)

        # Assert VALUE parity
        assert_dict_features_parity(
            calculator_features,
            facade_features,
            context="100 bars",
        )

    @pytest.mark.parametrize("lookback", [10, 20, 50, 100])
    def test_parity_with_different_lookback_periods(
        self,
        lookback: int,
        test_bars_200: list[Bar],
    ) -> None:
        """Verify parity across different lookback window configurations.

        Note: If lookback_window is unsupported, this test verifies basic parity
        with the standard config.
        """
        # Use standard config - lookback may be handled differently
        config = FeatureConfig(
            return_periods=[1, 2, 5],
            momentum_periods=[1, 3],
            volume_ma_periods=[10, 20],
            ema_fast=12,
            ema_slow=26,
            rsi_period=14,
            bb_period=20,
            bb_std=2.0,
            atr_period=14,
        )

        facade = FeatureEngineer(config)
        calculator = _build_calculator(config)

        # Use subset based on lookback to vary data size
        bars_subset = test_bars_200[:lookback * 2]  # Use 2x lookback bars

        facade_features = facade.compute_features(bars_subset)
        calculator_features = calculator.compute_features(bars_subset)

        assert_dict_features_parity(
            calculator_features,
            facade_features,
            context=f"lookback={lookback}",
        )

    def test_parity_with_different_rsi_periods(
        self,
        test_bars: list[Bar],
    ) -> None:
        """Verify parity across different RSI period configurations."""
        for rsi_period in [7, 14, 21]:
            config = FeatureConfig(
                return_periods=[1, 2, 5],
                momentum_periods=[1, 3],
                volume_ma_periods=[10, 20],
                ema_fast=12,
                ema_slow=26,
                rsi_period=rsi_period,
                bb_period=20,
                bb_std=2.0,
                atr_period=14,
            )

            facade = FeatureEngineer(config)
            calculator = _build_calculator(config)

            facade_features = facade.compute_features(test_bars)
            calculator_features = calculator.compute_features(test_bars)

            assert_dict_features_parity(
                calculator_features,
                facade_features,
                context=f"rsi_period={rsi_period}",
            )

    def test_parity_with_different_bb_periods(
        self,
        test_bars: list[Bar],
    ) -> None:
        """Verify parity across different Bollinger Band configurations."""
        for bb_period in [10, 20, 30]:
            config = FeatureConfig(
                return_periods=[1, 2, 5],
                momentum_periods=[1, 3],
                volume_ma_periods=[10, 20],
                ema_fast=12,
                ema_slow=26,
                rsi_period=14,
                bb_period=bb_period,
                bb_std=2.0,
                atr_period=14,
            )

            facade = FeatureEngineer(config)
            calculator = _build_calculator(config)

            facade_features = facade.compute_features(test_bars)
            calculator_features = calculator.compute_features(test_bars)

            assert_dict_features_parity(
                calculator_features,
                facade_features,
                context=f"bb_period={bb_period}",
            )

    def test_parity_with_edge_cases_empty_data(
        self,
        feature_config: FeatureConfig,
    ) -> None:
        """Verify empty data handling for the facade and calculator component."""
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        with pytest.raises(ValueError):
            calculator.compute_features([])
        assert facade.compute_features([]) == {}

    def test_parity_with_edge_cases_single_bar(
        self,
        feature_config: FeatureConfig,
        test_bar: Bar,
    ) -> None:
        """Verify parity with minimal data (single bar).

        Expected Behavior:
            - Both handle single bar gracefully
            - Results match
        """
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        facade_features = facade.compute_features([test_bar])
        calculator_features = calculator.compute_features([test_bar])

        assert_dict_features_parity(
            calculator_features,
            facade_features,
            context="single bar edge case",
        )

    def test_parity_with_multiple_instruments(
        self,
        feature_config: FeatureConfig,
        multi_instrument_bars: dict[str, list[Bar]],
    ) -> None:
        """Verify parity when computing features for multiple symbols.

        Expected Behavior:
            - Features for each instrument identical between implementations
        """
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        for symbol, bars in multi_instrument_bars.items():
            facade_features = facade.compute_features(bars)
            calculator_features = calculator.compute_features(bars)

            assert_dict_features_parity(
                calculator_features,
                facade_features,
                context=f"symbol={symbol}",
            )

    @pytest.mark.slow
    def test_parity_performance_within_25_percent(
        self,
        feature_config: FeatureConfig,
        test_bars_1000: list[Bar],
    ) -> None:
        """Verify facade performance overhead acceptable (<25%).

        Expected Behavior:
            - Facade P99 <= calculator_p99 * 1.25 (within 25%)
            - Numerical results still match (parity)

        Note: 25% tolerance accounts for system load variance in CI environments.
        The actual overhead is typically <5% but we use wider tolerance for stability.
        """
        facade = FeatureEngineer(feature_config)
        calculator = _build_calculator(feature_config)

        # Extended warmup for more stable measurements
        for _ in range(10):
            calculator.compute_features(test_bars_1000)
            facade.compute_features(test_bars_1000)

        # Calculator timing
        times_calculator = []
        for _ in range(30):
            start = time.perf_counter()
            calculator_features = calculator.compute_features(test_bars_1000)
            times_calculator.append(time.perf_counter() - start)
        calculator_p99 = np.percentile(times_calculator, 99)

        # Facade timing
        times_facade = []
        for _ in range(30):
            start = time.perf_counter()
            facade_features = facade.compute_features(test_bars_1000)
            times_facade.append(time.perf_counter() - start)
        facade_p99 = np.percentile(times_facade, 99)

        # Performance parity (within 25% - allows for CI variance)
        assert facade_p99 <= calculator_p99 * 1.25, (
            "Facade P99 "
            f"{facade_p99*1000:.2f}ms exceeds 125% of calculator {calculator_p99*1000:.2f}ms"
        )

        # Numerical parity (still must match!)
        assert_dict_features_parity(
            calculator_features,
            facade_features,
            context="performance test",
        )
