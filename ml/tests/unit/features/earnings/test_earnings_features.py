"""
Earnings Features Tests.

Tests earnings surprise, growth, momentum, and calendar feature calculations.
Covers happy paths, edge cases, and error conditions.

Test Coverage:
- Earnings surprise: Basic calculations, division by zero, negative values
- Earnings growth: YoY/QoQ calculations, insufficient history, zero denominators
- Earnings momentum: Beat streaks, volatility, edge cases
- Calendar features: Date arithmetic, negative days (past due)
"""

from __future__ import annotations

from datetime import datetime

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


class TestEarningsSurprise:
    """Test earnings surprise calculations."""

    def test_surprise_basic(self) -> None:
        """Test basic earnings surprise calculation."""
        actual = 2.52
        estimate = 2.45

        result = compute_earnings_surprise_incremental(actual, estimate)

        assert result["eps_surprise_q0"] == pytest.approx(0.07, abs=1e-10)
        assert result["eps_surprise_pct_q0"] == pytest.approx(2.857142857142857, rel=1e-10)

    def test_surprise_negative(self) -> None:
        """Test earnings miss (negative surprise)."""
        actual = 2.30
        estimate = 2.45

        result = compute_earnings_surprise_incremental(actual, estimate)

        assert result["eps_surprise_q0"] == pytest.approx(-0.15, abs=1e-10)
        assert result["eps_surprise_pct_q0"] == pytest.approx(-6.122448979591837, rel=1e-10)

    def test_surprise_zero_estimate(self) -> None:
        """Test division by zero protection when estimate is zero."""
        actual = 2.52
        estimate = 0.0

        result = compute_earnings_surprise_incremental(actual, estimate)

        assert result["eps_surprise_q0"] == pytest.approx(2.52, abs=1e-10)
        assert result["eps_surprise_pct_q0"] == 0.0  # Protected from div/0

    def test_surprise_near_zero_estimate(self) -> None:
        """Test handling of very small estimates (< 1e-12)."""
        actual = 2.52
        estimate = 1e-13

        result = compute_earnings_surprise_incremental(actual, estimate)

        assert result["eps_surprise_pct_q0"] == 0.0  # Below threshold, returns 0

    def test_surprise_batch(self) -> None:
        """Test batch earnings surprise calculation."""
        actuals = np.array([2.52, 2.45, 2.38, 2.30])
        estimates = np.array([2.45, 2.40, 2.35, 2.28])

        result = compute_earnings_surprise_batch(actuals, estimates)

        expected_surprise = np.array([0.07, 0.05, 0.03, 0.02])
        np.testing.assert_allclose(
            result["eps_surprise_q0"],
            expected_surprise,
            rtol=1e-10,
        )

    def test_surprise_batch_shape_mismatch(self) -> None:
        """Test error handling for shape mismatch."""
        actuals = np.array([2.52, 2.45])
        estimates = np.array([2.45, 2.40, 2.35])

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_earnings_surprise_batch(actuals, estimates)

    def test_surprise_batch_empty(self) -> None:
        """Test batch computation with empty arrays."""
        actuals = np.array([])
        estimates = np.array([])

        result = compute_earnings_surprise_batch(actuals, estimates)

        assert len(result["eps_surprise_q0"]) == 0
        assert len(result["eps_surprise_pct_q0"]) == 0


class TestEarningsGrowth:
    """Test earnings growth calculations."""

    def test_growth_yoy_basic(self) -> None:
        """Test basic YoY growth calculation."""
        eps_history = [2.52, 2.45, 2.38, 2.30, 2.20]  # Q0, Q-1, Q-2, Q-3, Q-4

        result = compute_earnings_growth_incremental(eps_history)

        expected_yoy = ((2.52 - 2.20) / 2.20) * 100  # 14.545%
        assert result["eps_growth_yoy"] == pytest.approx(expected_yoy, rel=1e-10)

    def test_growth_qoq_basic(self) -> None:
        """Test basic QoQ growth calculation."""
        eps_history = [2.52, 2.45, 2.38, 2.30, 2.20]

        result = compute_earnings_growth_incremental(eps_history)

        expected_qoq = ((2.52 - 2.45) / 2.45) * 100  # 2.857%
        assert result["eps_growth_qoq"] == pytest.approx(expected_qoq, rel=1e-10)

    def test_growth_insufficient_history(self) -> None:
        """Test handling of insufficient history for YoY (< 5 quarters)."""
        eps_history = [2.52, 2.45, 2.38]  # Only 3 quarters

        result = compute_earnings_growth_incremental(eps_history)

        # YoY requires 5 quarters - should be 0
        assert result["eps_growth_yoy"] == 0.0
        # QoQ should work with 2+ quarters
        expected_qoq = ((2.52 - 2.45) / 2.45) * 100
        assert result["eps_growth_qoq"] == pytest.approx(expected_qoq, rel=1e-10)

    def test_growth_zero_denominator(self) -> None:
        """Test division by zero protection."""
        eps_history = [2.52, 0.0, 2.38, 2.30, 0.0]  # Zero at Q-1 and Q-4

        result = compute_earnings_growth_incremental(eps_history)

        assert result["eps_growth_yoy"] == 0.0  # Protected from div/0
        assert result["eps_growth_qoq"] == 0.0  # Protected from div/0

    def test_growth_batch(self) -> None:
        """Test batch growth calculation."""
        eps_series = np.array([2.20, 2.30, 2.38, 2.45, 2.52])

        result = compute_earnings_growth_batch(eps_series)

        # YoY should be 0 for first 4 values, then calculated
        assert result["eps_growth_yoy"][0] == 0.0
        assert result["eps_growth_yoy"][3] == 0.0
        expected_yoy = ((2.52 - 2.20) / 2.20) * 100
        assert result["eps_growth_yoy"][4] == pytest.approx(expected_yoy, rel=1e-10)

        # QoQ should be 0 for first value, then calculated
        assert result["eps_growth_qoq"][0] == 0.0
        expected_qoq_4 = ((2.52 - 2.45) / 2.45) * 100
        assert result["eps_growth_qoq"][4] == pytest.approx(expected_qoq_4, rel=1e-10)

    def test_growth_batch_empty(self) -> None:
        """Test batch computation with empty array."""
        eps_series = np.array([])

        result = compute_earnings_growth_batch(eps_series)

        assert len(result["eps_growth_yoy"]) == 0
        assert len(result["eps_growth_qoq"]) == 0


class TestEarningsMomentum:
    """Test earnings momentum calculations."""

    def test_momentum_beat_streak(self) -> None:
        """Test consecutive beat streak counting."""
        surprises = [0.07, 0.05, 0.03, -0.02, 0.01]  # Last 5 quarters
        eps_history = [2.52, 2.45, 2.38, 2.30]

        result = compute_earnings_momentum_incremental(surprises, eps_history)

        assert result["earnings_beat_streak"] == 3.0  # First 3 are positive

    def test_momentum_no_beats(self) -> None:
        """Test beat streak when first surprise is negative."""
        surprises = [-0.02, 0.05, 0.03]
        eps_history = [2.52, 2.45, 2.38, 2.30]

        result = compute_earnings_momentum_incremental(surprises, eps_history)

        assert result["earnings_beat_streak"] == 0.0

    def test_momentum_all_beats(self) -> None:
        """Test beat streak when all surprises are positive."""
        surprises = [0.07, 0.05, 0.03, 0.02, 0.01]
        eps_history = [2.52, 2.45, 2.38, 2.30]

        result = compute_earnings_momentum_incremental(surprises, eps_history)

        assert result["earnings_beat_streak"] == 5.0

    def test_momentum_volatility(self) -> None:
        """Test EPS volatility calculation (coefficient of variation)."""
        surprises = [0.07, 0.05, 0.03, -0.02]
        eps_history = [2.52, 2.45, 2.38, 2.30]

        result = compute_earnings_momentum_incremental(surprises, eps_history)

        # Calculate expected volatility
        eps_array = np.array(eps_history, dtype=np.float64)
        expected_std = np.std(eps_array, ddof=1)
        expected_mean = np.mean(eps_array)
        expected_volatility = expected_std / expected_mean

        assert result["eps_volatility_4q"] == pytest.approx(expected_volatility, rel=1e-10)

    def test_momentum_insufficient_history(self) -> None:
        """Test volatility with insufficient history (< 4 quarters)."""
        surprises = [0.07, 0.05]
        eps_history = [2.52, 2.45]

        result = compute_earnings_momentum_incremental(surprises, eps_history)

        assert result["eps_volatility_4q"] == 0.0

    def test_momentum_zero_mean(self) -> None:
        """Test volatility when mean EPS is near zero."""
        surprises = [0.07, 0.05, 0.03, -0.02]
        eps_history = [1e-13, -1e-13, 1e-14, -1e-14]  # Near zero

        result = compute_earnings_momentum_incremental(surprises, eps_history)

        assert result["eps_volatility_4q"] == 0.0  # Protected from div/0

    def test_momentum_batch(self) -> None:
        """Test batch momentum calculation."""
        surprises = np.array([0.07, 0.05, 0.03, -0.02, 0.01])
        eps_series = np.array([2.52, 2.45, 2.38, 2.30, 2.20])

        result = compute_earnings_momentum_batch(surprises, eps_series)

        # Check beat streak at last position
        # From position 4 backwards: 0.01 > 0 (count 1), -0.02 <= 0 (stop)
        assert result["earnings_beat_streak"][4] == 1.0

        # Check volatility at last position (has 4 quarters of history)
        window = eps_series[1:5]  # Last 4 quarters from position 4
        expected_std = np.std(window, ddof=1)
        expected_mean = np.mean(window)
        expected_volatility = expected_std / expected_mean
        assert result["eps_volatility_4q"][4] == pytest.approx(expected_volatility, rel=1e-10)

    def test_momentum_batch_shape_mismatch(self) -> None:
        """Test error handling for shape mismatch."""
        surprises = np.array([0.07, 0.05])
        eps_series = np.array([2.52, 2.45, 2.38])

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_earnings_momentum_batch(surprises, eps_series)


class TestCalendarFeatures:
    """Test earnings calendar calculations."""

    def test_calendar_basic(self) -> None:
        """Test basic days to earnings calculation."""
        next_earnings = datetime(2024, 1, 30)
        current = datetime(2024, 1, 1)

        result = compute_calendar_features_incremental(next_earnings, current)

        assert result["days_to_next_earnings"] == 29

    def test_calendar_negative_days(self) -> None:
        """Test negative days (earnings date has passed)."""
        next_earnings = datetime(2024, 1, 1)
        current = datetime(2024, 1, 30)

        result = compute_calendar_features_incremental(next_earnings, current)

        assert result["days_to_next_earnings"] == -29

    def test_calendar_same_day(self) -> None:
        """Test same day (earnings today)."""
        date = datetime(2024, 1, 15)

        result = compute_calendar_features_incremental(date, date)

        assert result["days_to_next_earnings"] == 0

    def test_calendar_batch(self) -> None:
        """Test batch calendar calculation."""
        next_earnings = np.array(
            ["2024-01-30", "2024-02-15"],
            dtype="datetime64[D]",
        )
        current = np.array(
            ["2024-01-01", "2024-01-20"],
            dtype="datetime64[D]",
        )

        result = compute_calendar_features_batch(next_earnings, current)

        expected = np.array([29, 26], dtype=np.int64)
        np.testing.assert_array_equal(result["days_to_next_earnings"], expected)

    def test_calendar_batch_shape_mismatch(self) -> None:
        """Test error handling for shape mismatch."""
        next_earnings = np.array(["2024-01-30"], dtype="datetime64[D]")
        current = np.array(
            ["2024-01-01", "2024-01-15"],
            dtype="datetime64[D]",
        )

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_calendar_features_batch(next_earnings, current)
