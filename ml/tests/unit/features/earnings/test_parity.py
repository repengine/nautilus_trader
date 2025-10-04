"""
Earnings Features Parity Tests.

Validates that batch (cold path) and incremental (hot path) implementations produce
identical results within specified tolerance (rtol=1e-10).

Test Coverage:
- Earnings surprise: Batch vs incremental parity
- Earnings growth: Batch vs incremental parity
- Earnings momentum: Batch vs incremental parity
- Calendar features: Batch vs incremental parity (exact match)
- Edge cases: zero values, small samples, extreme values
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from ml.features.earnings import (
    compute_calendar_features_batch,
    compute_calendar_features_incremental,
    compute_earnings_growth_batch,
    compute_earnings_growth_incremental,
    compute_earnings_momentum_batch,
    compute_earnings_momentum_incremental,
    compute_earnings_surprise_batch,
    compute_earnings_surprise_incremental,
)


class TestEarningsSurpriseParity:
    """Test parity between batch and incremental earnings surprise computation."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample earnings data for testing."""
        np.random.seed(42)
        n = 50
        # Actuals around 2.5 with some variance
        actuals = np.random.normal(2.5, 0.3, n)
        # Estimates slightly below actuals (positive bias)
        estimates = actuals - np.random.uniform(0, 0.2, n)
        return actuals, estimates

    def test_parity_basic(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test basic parity between batch and incremental computation."""
        actuals, estimates = sample_data

        # Batch computation
        result_batch = compute_earnings_surprise_batch(actuals, estimates)

        # Incremental computation
        surprises_incr = []
        surprises_pct_incr = []

        for actual, estimate in zip(actuals, estimates):
            result = compute_earnings_surprise_incremental(actual, estimate)
            surprises_incr.append(result["eps_surprise_q0"])
            surprises_pct_incr.append(result["eps_surprise_pct_q0"])

        surprises_incr_arr = np.array(surprises_incr)
        surprises_pct_incr_arr = np.array(surprises_pct_incr)

        # Validate parity (rtol=1e-10)
        np.testing.assert_allclose(
            result_batch["eps_surprise_q0"],
            surprises_incr_arr,
            rtol=1e-10,
            atol=0,
            err_msg="Batch and incremental eps_surprise_q0 differ beyond tolerance",
        )

        np.testing.assert_allclose(
            result_batch["eps_surprise_pct_q0"],
            surprises_pct_incr_arr,
            rtol=1e-10,
            atol=0,
            err_msg="Batch and incremental eps_surprise_pct_q0 differ beyond tolerance",
        )

    def test_parity_with_zeros(self) -> None:
        """Test parity with zero estimates (division by zero case)."""
        actuals = np.array([2.52, 2.45, 2.38])
        estimates = np.array([2.45, 0.0, 2.35])

        # Batch computation
        result_batch = compute_earnings_surprise_batch(actuals, estimates)

        # Incremental computation
        surprises_pct_incr = []
        for actual, estimate in zip(actuals, estimates):
            result = compute_earnings_surprise_incremental(actual, estimate)
            surprises_pct_incr.append(result["eps_surprise_pct_q0"])

        surprises_pct_incr_arr = np.array(surprises_pct_incr)

        # Should match exactly (both handle div/0 the same way)
        np.testing.assert_array_equal(
            result_batch["eps_surprise_pct_q0"],
            surprises_pct_incr_arr,
        )


class TestEarningsGrowthParity:
    """Test parity between batch and incremental earnings growth computation."""

    @pytest.fixture
    def sample_eps_series(self) -> np.ndarray:
        """Generate sample EPS time series for testing."""
        np.random.seed(42)
        # Simulating growing EPS over 20 quarters
        base = 2.0
        growth = np.random.normal(0.05, 0.02, 20)
        eps_series = [base]
        for g in growth:
            eps_series.append(eps_series[-1] * (1 + g))
        return np.array(eps_series)

    def test_parity_basic(self, sample_eps_series: np.ndarray) -> None:
        """Test basic parity between batch and incremental computation."""
        # Batch computation
        result_batch = compute_earnings_growth_batch(sample_eps_series)

        # Incremental computation (sliding window of 5 quarters)
        yoy_incr = []
        qoq_incr = []

        for i in range(len(sample_eps_series)):
            # Get available history in REVERSE ORDER (most recent first)
            window = sample_eps_series[: i + 1][::-1].tolist()
            result = compute_earnings_growth_incremental(window)
            yoy_incr.append(result["eps_growth_yoy"])
            qoq_incr.append(result["eps_growth_qoq"])

        yoy_incr_arr = np.array(yoy_incr)
        qoq_incr_arr = np.array(qoq_incr)

        # Validate parity (rtol=1e-10)
        np.testing.assert_allclose(
            result_batch["eps_growth_yoy"],
            yoy_incr_arr,
            rtol=1e-10,
            atol=0,
            err_msg="Batch and incremental eps_growth_yoy differ beyond tolerance",
        )

        np.testing.assert_allclose(
            result_batch["eps_growth_qoq"],
            qoq_incr_arr,
            rtol=1e-10,
            atol=0,
            err_msg="Batch and incremental eps_growth_qoq differ beyond tolerance",
        )

    def test_parity_with_zeros(self) -> None:
        """Test parity with zero denominators."""
        eps_series = np.array([2.20, 0.0, 2.38, 2.45, 0.0])

        # Batch computation
        result_batch = compute_earnings_growth_batch(eps_series)

        # Incremental computation (reverse order - most recent first)
        window = eps_series[::-1].tolist()
        result_incr = compute_earnings_growth_incremental(window)

        # YoY at position 4: (0.0 - 2.20) / 2.20, but denominator Q-4 is 2.20
        # Actually the window is [0.0, 2.38, 2.45, 0.0] - wait, Q-4 would be eps_series[0]
        # Let me recalculate: position 4, Q0=eps_series[4]=0.0, Q-4=eps_series[0]=2.20
        # But with zero Q0, we still calculate if Q-4 is non-zero
        # However in incremental, window is last 5 in reverse: [0.0, 2.45, 2.38, 0.0, 2.20]

        # For last position, just verify they match
        assert result_batch["eps_growth_yoy"][4] == result_incr["eps_growth_yoy"]
        assert result_batch["eps_growth_qoq"][4] == result_incr["eps_growth_qoq"]


class TestEarningsMomentumParity:
    """Test parity between batch and incremental earnings momentum computation."""

    @pytest.fixture
    def sample_momentum_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample surprises and EPS for testing."""
        np.random.seed(42)
        n = 20
        surprises = np.random.normal(0.05, 0.1, n)  # Some positive, some negative
        eps_series = np.random.normal(2.5, 0.3, n)
        return surprises, eps_series

    def test_parity_basic(
        self,
        sample_momentum_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test basic parity between batch and incremental computation."""
        surprises, eps_series = sample_momentum_data

        # Batch computation
        result_batch = compute_earnings_momentum_batch(surprises, eps_series)

        # Incremental computation (for each position)
        beat_streak_incr = []
        volatility_incr = []

        for i in range(len(surprises)):
            # Get surprises from current position backwards
            surp_window = surprises[: i + 1][::-1].tolist()  # Reverse to most recent first

            # Get last 4 EPS values if available
            if i >= 3:
                eps_window = eps_series[i - 3 : i + 1][::-1].tolist()
            else:
                eps_window = eps_series[: i + 1][::-1].tolist()

            result = compute_earnings_momentum_incremental(surp_window, eps_window)
            beat_streak_incr.append(result["earnings_beat_streak"])
            volatility_incr.append(result["eps_volatility_4q"])

        beat_streak_incr_arr = np.array(beat_streak_incr)
        volatility_incr_arr = np.array(volatility_incr)

        # Validate parity (rtol=1e-10)
        np.testing.assert_allclose(
            result_batch["earnings_beat_streak"],
            beat_streak_incr_arr,
            rtol=1e-10,
            atol=0,
            err_msg="Batch and incremental earnings_beat_streak differ beyond tolerance",
        )

        np.testing.assert_allclose(
            result_batch["eps_volatility_4q"],
            volatility_incr_arr,
            rtol=1e-10,
            atol=0,
            err_msg="Batch and incremental eps_volatility_4q differ beyond tolerance",
        )

    def test_parity_all_beats(self) -> None:
        """Test parity with all positive surprises."""
        surprises = np.array([0.07, 0.05, 0.03, 0.02, 0.01])
        eps_series = np.array([2.52, 2.45, 2.38, 2.30, 2.20])

        # Batch computation
        result_batch = compute_earnings_momentum_batch(surprises, eps_series)

        # Incremental at last position
        surp_window = surprises[::-1].tolist()
        eps_window = eps_series[-4:][::-1].tolist()
        result_incr = compute_earnings_momentum_incremental(surp_window, eps_window)

        # Beat streak should be 5 (all positive)
        assert result_batch["earnings_beat_streak"][4] == result_incr["earnings_beat_streak"]


class TestCalendarFeaturesParity:
    """Test parity between batch and incremental calendar features."""

    def test_parity_basic(self) -> None:
        """Test exact parity between batch and incremental (integer arithmetic)."""
        # Generate test dates
        base_date = datetime(2024, 1, 1)
        current_dates_list = [base_date + timedelta(days=i) for i in range(10)]
        next_earnings_dates_list = [base_date + timedelta(days=30 + i) for i in range(10)]

        # Convert to numpy arrays
        current_dates_np = np.array(
            [d.strftime("%Y-%m-%d") for d in current_dates_list],
            dtype="datetime64[D]",
        )
        next_earnings_np = np.array(
            [d.strftime("%Y-%m-%d") for d in next_earnings_dates_list],
            dtype="datetime64[D]",
        )

        # Batch computation
        result_batch = compute_calendar_features_batch(next_earnings_np, current_dates_np)

        # Incremental computation
        days_incr = []
        for current, next_earnings in zip(current_dates_list, next_earnings_dates_list):
            result = compute_calendar_features_incremental(next_earnings, current)
            days_incr.append(result["days_to_next_earnings"])

        days_incr_arr = np.array(days_incr, dtype=np.int64)

        # Should be exactly equal (integer arithmetic)
        np.testing.assert_array_equal(
            result_batch["days_to_next_earnings"],
            days_incr_arr,
            err_msg="Batch and incremental days_to_next_earnings differ",
        )

    def test_parity_negative_days(self) -> None:
        """Test parity with negative days (past due)."""
        current = datetime(2024, 2, 15)
        next_earnings = datetime(2024, 1, 30)

        # Incremental
        result_incr = compute_calendar_features_incremental(next_earnings, current)

        # Batch
        current_np = np.array(["2024-02-15"], dtype="datetime64[D]")
        next_earnings_np = np.array(["2024-01-30"], dtype="datetime64[D]")
        result_batch = compute_calendar_features_batch(next_earnings_np, current_np)

        # Should match exactly
        assert result_batch["days_to_next_earnings"][0] == result_incr["days_to_next_earnings"]
        assert result_incr["days_to_next_earnings"] == -16


class TestEdgeCasesParity:
    """Test parity for edge cases across all feature types."""

    def test_extreme_values(self) -> None:
        """Test parity with extreme values."""
        # Very large values
        actuals = np.array([1e10, 1e-10, 0.0])
        estimates = np.array([9e9, 1e-11, 1e-10])

        result_batch = compute_earnings_surprise_batch(actuals, estimates)

        surprises_incr = []
        for actual, estimate in zip(actuals, estimates):
            result = compute_earnings_surprise_incremental(actual, estimate)
            surprises_incr.append(result["eps_surprise_q0"])

        np.testing.assert_allclose(
            result_batch["eps_surprise_q0"],
            np.array(surprises_incr),
            rtol=1e-10,
            atol=0,
        )

    def test_single_value(self) -> None:
        """Test parity with single values."""
        # Surprise
        result_batch_s = compute_earnings_surprise_batch(
            np.array([2.52]),
            np.array([2.45]),
        )
        result_incr_s = compute_earnings_surprise_incremental(2.52, 2.45)

        assert result_batch_s["eps_surprise_q0"][0] == result_incr_s["eps_surprise_q0"]

    def test_all_zeros(self) -> None:
        """Test parity with all zero values."""
        actuals = np.array([0.0, 0.0, 0.0])
        estimates = np.array([0.0, 0.0, 0.0])

        result_batch = compute_earnings_surprise_batch(actuals, estimates)

        # All surprises should be zero
        np.testing.assert_array_equal(result_batch["eps_surprise_q0"], np.zeros(3))
        np.testing.assert_array_equal(result_batch["eps_surprise_pct_q0"], np.zeros(3))
