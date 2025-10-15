"""Unit tests for rolling beta estimation and stability analysis."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import math
import numpy as np
import polars as pl
import pytest

from playground.risk_model.rolling_beta import compute_beta_stability_analysis
from playground.risk_model.rolling_beta import compute_rolling_betas


class TestRollingBetas:
    """Test suite for rolling beta estimation."""

    @pytest.fixture
    def stable_beta_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create data with constant betas over time."""
        n = 500
        np.random.seed(42)

        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

        # Factors
        factor_duration = np.random.randn(n)
        factor_credit = np.random.randn(n)
        factor_liquidity = np.random.randn(n)

        # Sector returns with constant betas: 0.5, 0.3, -0.2
        sector_returns = (
            0.01  # alpha
            + 0.5 * factor_duration
            + 0.3 * factor_credit
            + -0.2 * factor_liquidity
            + 0.02 * np.random.randn(n)  # noise
        )

        sector_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["TEST"] * n,
                "return": sector_returns,
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": factor_duration,
                "factor_credit": factor_credit,
                "factor_liquidity": factor_liquidity,
            }
        )

        return sector_df, factor_df

    @pytest.fixture
    def time_varying_beta_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create data with time-varying betas (structural break at midpoint)."""
        n = 500
        np.random.seed(123)

        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

        # Factors
        factor_duration = np.random.randn(n)
        factor_credit = np.random.randn(n)
        factor_liquidity = np.random.randn(n)

        # Sector returns with structural break at n/2
        sector_returns = np.zeros(n)
        for i in range(n):
            if i < n // 2:
                # First half: betas = [0.5, 0.3, -0.2]
                sector_returns[i] = (
                    0.01
                    + 0.5 * factor_duration[i]
                    + 0.3 * factor_credit[i]
                    + -0.2 * factor_liquidity[i]
                    + 0.02 * np.random.randn()
                )
            else:
                # Second half: betas = [0.8, 0.1, -0.4]
                sector_returns[i] = (
                    0.01
                    + 0.8 * factor_duration[i]
                    + 0.1 * factor_credit[i]
                    + -0.4 * factor_liquidity[i]
                    + 0.02 * np.random.randn()
                )

        sector_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["TEST"] * n,
                "return": sector_returns,
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": factor_duration,
                "factor_credit": factor_credit,
                "factor_liquidity": factor_liquidity,
            }
        )

        return sector_df, factor_df

    @pytest.fixture
    def multiple_sectors_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create data with multiple sectors."""
        n = 300
        np.random.seed(456)

        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

        # Factors (shared across sectors)
        factor_duration = np.random.randn(n)
        factor_credit = np.random.randn(n)
        factor_liquidity = np.random.randn(n)

        # Sector 1: high duration beta
        sector1_returns = 0.01 + 0.8 * factor_duration + 0.2 * factor_credit + 0.02 * np.random.randn(n)

        # Sector 2: high credit beta
        sector2_returns = 0.01 + 0.2 * factor_duration + 0.7 * factor_credit + 0.02 * np.random.randn(n)

        sector_df = pl.DataFrame(
            {
                "timestamp": timestamps * 2,
                "symbol": ["SECTOR1"] * n + ["SECTOR2"] * n,
                "return": list(sector1_returns) + list(sector2_returns),
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": factor_duration,
                "factor_credit": factor_credit,
                "factor_liquidity": factor_liquidity,
            }
        )

        return sector_df, factor_df

    def test_stable_betas(self, stable_beta_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test rolling betas with constant true betas."""
        sector_df, factor_df = stable_beta_data

        results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
        )

        assert "TEST" in results
        result = results["TEST"]

        # Should have multiple windows
        assert result.n_windows > 0
        assert len(result.beta_duration) == result.n_windows
        assert len(result.beta_credit) == result.n_windows
        assert len(result.beta_liquidity) == result.n_windows
        assert len(result.alpha) == result.n_windows
        assert len(result.r_squared) == result.n_windows
        assert len(result.timestamps) == result.n_windows

        # Betas should be close to true values (0.5, 0.3, -0.2)
        mean_beta_dur = np.mean(result.beta_duration)
        mean_beta_cred = np.mean(result.beta_credit)
        mean_beta_liq = np.mean(result.beta_liquidity)

        assert abs(mean_beta_dur - 0.5) < 0.1
        assert abs(mean_beta_cred - 0.3) < 0.1
        assert abs(mean_beta_liq - (-0.2)) < 0.1

        # Betas should be stable (low std)
        std_beta_dur = np.std(result.beta_duration)
        assert std_beta_dur < 0.15

    def test_time_varying_betas(
        self, time_varying_beta_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Test rolling betas with structural break."""
        sector_df, factor_df = time_varying_beta_data

        results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
        )

        assert "TEST" in results
        result = results["TEST"]

        # Should detect changing betas
        assert result.n_windows > 0

        # Duration beta should show variation
        std_beta_dur = np.std(result.beta_duration)
        assert std_beta_dur > 0.05  # Should vary more than stable case

    def test_window_size_validation(
        self, stable_beta_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Test validation of window_days parameter."""
        sector_df, factor_df = stable_beta_data

        # Window size smaller than min_observations should fail
        with pytest.raises(ValueError, match=r"window_days.*must be >= min_observations"):
            compute_rolling_betas(
                sector_df,
                factor_df,
                factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
                window_days=50,
                min_observations=100,
            )

    def test_minimum_observations_requirement(
        self, stable_beta_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Test minimum observations requirement."""
        sector_df, factor_df = stable_beta_data

        # Use only first 100 observations
        small_sector = sector_df.head(100)
        small_factor = factor_df.head(100)

        # Should work with window_days=50, min_observations=40
        results = compute_rolling_betas(
            small_sector,
            small_factor,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=50,
            min_observations=40,
        )

        assert "TEST" in results
        assert results["TEST"].n_windows > 0

    def test_factor_column_validation(
        self, stable_beta_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Ensure factor column validation rejects incomplete sets."""
        sector_df, factor_df = stable_beta_data

        with pytest.raises(ValueError, match="factor_columns must contain exactly"):
            compute_rolling_betas(
                sector_df,
                factor_df,
                factor_columns=["factor_duration", "factor_credit"],
            )

    def test_empty_data_handling(self) -> None:
        """Test handling of empty data."""
        empty_sector = pl.DataFrame(
            {
                "timestamp": [],
                "symbol": [],
                "return": [],
            }
        )

        empty_factor = pl.DataFrame(
            {
                "timestamp": [],
                "factor_duration": [],
                "factor_credit": [],
                "factor_liquidity": [],
            }
        )

        with pytest.raises(ValueError, match="sector_returns cannot be empty"):
            compute_rolling_betas(
                empty_sector,
                empty_factor,
                factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            )

    def test_multiple_sectors(
        self, multiple_sectors_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Test rolling betas with multiple sectors."""
        sector_df, factor_df = multiple_sectors_data

        results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=150,
            min_observations=100,
        )

        # Should have both sectors
        assert "SECTOR1" in results
        assert "SECTOR2" in results

        # SECTOR1 should have higher duration beta
        assert np.mean(results["SECTOR1"].beta_duration) > np.mean(
            results["SECTOR2"].beta_duration
        )

        # SECTOR2 should have higher credit beta
        assert np.mean(results["SECTOR2"].beta_credit) > np.mean(results["SECTOR1"].beta_credit)

    def test_insufficient_data_sector_skipped(
        self, stable_beta_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Test that sectors with insufficient data are skipped."""
        sector_df, factor_df = stable_beta_data

        # Add a sector with very few observations
        small_sector_data = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)] * 10,
                "symbol": ["SMALL"] * 10,
                "return": [0.01] * 10,
            }
        )

        combined_sector = pl.concat([sector_df, small_sector_data])

        results = compute_rolling_betas(
            combined_sector,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
            min_observations=100,
        )

        # Should only have TEST sector, not SMALL
        assert "TEST" in results
        assert "SMALL" not in results

    def test_r_squared_bounds(self, stable_beta_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test that R² values are in valid range [0, 1]."""
        sector_df, factor_df = stable_beta_data

        results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
        )

        result = results["TEST"]

        # All R² values should be between 0 and 1
        for r2 in result.r_squared:
            assert 0.0 <= r2 <= 1.0

    def test_window_end_dates_monotonic(
        self, stable_beta_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Test that window end dates are monotonically increasing."""
        sector_df, factor_df = stable_beta_data

        results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
        )

        result = results["TEST"]

        # Timestamps should be strictly increasing
        for i in range(len(result.timestamps) - 1):
            assert result.timestamps[i] < result.timestamps[i + 1]


class TestStabilityAnalysis:
    """Test suite for beta stability analysis."""

    @pytest.fixture
    def stable_data_with_split(self) -> tuple[pl.DataFrame, pl.DataFrame, datetime]:
        """Create stable beta data with train/test split."""
        n = 500
        np.random.seed(42)

        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

        # Factors
        factor_duration = np.random.randn(n)
        factor_credit = np.random.randn(n)
        factor_liquidity = np.random.randn(n)

        # Sector returns with constant betas
        sector_returns = (
            0.01 + 0.5 * factor_duration + 0.3 * factor_credit + -0.2 * factor_liquidity + 0.02 * np.random.randn(n)
        )

        sector_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["TEST"] * n,
                "return": sector_returns,
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": factor_duration,
                "factor_credit": factor_credit,
                "factor_liquidity": factor_liquidity,
            }
        )

        # Split at 80% mark
        test_start = timestamps[400]

        return sector_df, factor_df, test_start

    def test_stable_betas_recommendation(
        self, stable_data_with_split: tuple[pl.DataFrame, pl.DataFrame, datetime]
    ) -> None:
        """Test stability analysis with stable betas (should recommend stable)."""
        sector_df, factor_df, test_start = stable_data_with_split

        # Compute rolling betas
        rolling_results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
        )

        # Compute stability analysis
        stability = compute_beta_stability_analysis(
            sector_df,
            factor_df,
            rolling_results,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            test_period_start=test_start,
        )

        assert "TEST" in stability
        analysis = stability["TEST"]

        # Should recommend stable for stable betas
        assert analysis.recommended_approach == "stable"
        assert analysis.beta_duration_cv < 0.5
        assert analysis.beta_credit_cv < 0.5
        assert analysis.beta_liquidity_cv < 0.5

    def test_forecast_accuracy_metrics(
        self, stable_data_with_split: tuple[pl.DataFrame, pl.DataFrame, datetime]
    ) -> None:
        """Test that forecast accuracy metrics are computed."""
        sector_df, factor_df, test_start = stable_data_with_split

        rolling_results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
        )

        stability = compute_beta_stability_analysis(
            sector_df,
            factor_df,
            rolling_results,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            test_period_start=test_start,
        )

        analysis = stability["TEST"]

        # Both forecast R² should be computed
        assert isinstance(analysis.stable_forecast_r2, float)
        assert isinstance(analysis.rolling_forecast_r2, float)

        # Should be reasonable values (not necessarily positive for OOS)
        assert -1.0 <= analysis.stable_forecast_r2 <= 1.0
        assert -1.0 <= analysis.rolling_forecast_r2 <= 1.0

    def test_coefficient_of_variation_calculation(
        self, stable_data_with_split: tuple[pl.DataFrame, pl.DataFrame, datetime]
    ) -> None:
        """Test coefficient of variation calculation."""
        sector_df, factor_df, test_start = stable_data_with_split

        rolling_results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
        )

        stability = compute_beta_stability_analysis(
            sector_df,
            factor_df,
            rolling_results,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            test_period_start=test_start,
        )

        analysis = stability["TEST"]

        # CV = std / mean
        expected_cv_dur = analysis.rolling_beta_std_duration / abs(
            analysis.rolling_beta_mean_duration
        )
        assert abs(analysis.beta_duration_cv - expected_cv_dur) < 1e-6

    def test_single_window_has_finite_metrics(self) -> None:
        """Single window inputs should not produce NaN coefficients or CVs."""
        n = 126
        np.random.seed(123)

        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
        factor_duration = np.random.randn(n)
        factor_credit = np.random.randn(n)
        factor_liquidity = np.random.randn(n)

        sector_returns = (
            0.01
            + 0.6 * factor_duration
            + 0.4 * factor_credit
            - 0.3 * factor_liquidity
            + 0.01 * np.random.randn(n)
        )

        sector_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["TEST"] * n,
                "return": sector_returns,
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": factor_duration,
                "factor_credit": factor_credit,
                "factor_liquidity": factor_liquidity,
            }
        )

        rolling_results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=n,
            min_observations=n,
        )

        stability = compute_beta_stability_analysis(
            sector_df,
            factor_df,
            rolling_results,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            test_period_start=timestamps[90],
        )

        analysis = stability["TEST"]

        assert math.isfinite(analysis.beta_duration_cv)
        assert math.isfinite(analysis.beta_credit_cv)
        assert math.isfinite(analysis.beta_liquidity_cv)
        assert math.isfinite(analysis.rolling_beta_std_duration)
        assert math.isfinite(analysis.rolling_beta_std_credit)
        assert math.isfinite(analysis.rolling_beta_std_liquidity)

    def test_empty_rolling_results_error(
        self, stable_data_with_split: tuple[pl.DataFrame, pl.DataFrame, datetime]
    ) -> None:
        """Test error when rolling_results is empty."""
        sector_df, factor_df, test_start = stable_data_with_split

        with pytest.raises(ValueError, match="rolling_results cannot be empty"):
            compute_beta_stability_analysis(
                sector_df,
                factor_df,
                {},  # Empty rolling results
                factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
                test_period_start=test_start,
            )

    def test_no_test_data_error(
        self, stable_data_with_split: tuple[pl.DataFrame, pl.DataFrame, datetime]
    ) -> None:
        """Test error when test period has no data."""
        sector_df, factor_df, _test_start = stable_data_with_split

        rolling_results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
        )

        # Set test start to future date
        future_date = datetime(2025, 1, 1, tzinfo=UTC)

        with pytest.raises(ValueError, match="No test data"):
            compute_beta_stability_analysis(
                sector_df,
                factor_df,
                rolling_results,
                factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
                test_period_start=future_date,
            )


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_single_window(self) -> None:
        """Test with data size equal to window size (single window)."""
        n = 126
        np.random.seed(42)

        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

        factor_duration = np.random.randn(n)
        factor_credit = np.random.randn(n)
        factor_liquidity = np.random.randn(n)

        sector_returns = 0.01 + 0.5 * factor_duration + 0.3 * factor_credit + 0.02 * np.random.randn(n)

        sector_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["TEST"] * n,
                "return": sector_returns,
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": factor_duration,
                "factor_credit": factor_credit,
                "factor_liquidity": factor_liquidity,
            }
        )

        results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
        )

        # Should have exactly 1 window
        assert results["TEST"].n_windows == 1

    def test_non_overlapping_windows(self) -> None:
        """Test with window size = total size (non-overlapping)."""
        n = 252
        np.random.seed(42)

        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

        factor_duration = np.random.randn(n)
        factor_credit = np.random.randn(n)
        factor_liquidity = np.random.randn(n)

        sector_returns = 0.01 + 0.5 * factor_duration + 0.02 * np.random.randn(n)

        sector_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["TEST"] * n,
                "return": sector_returns,
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": factor_duration,
                "factor_credit": factor_credit,
                "factor_liquidity": factor_liquidity,
            }
        )

        results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=252,
        )

        # Should have exactly 1 window (full sample)
        assert results["TEST"].n_windows == 1

    def test_missing_values_in_window(self) -> None:
        """Test handling of missing values in windows."""
        n = 300
        np.random.seed(42)

        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

        factor_duration = np.random.randn(n)
        factor_credit = np.random.randn(n)
        factor_liquidity = np.random.randn(n)

        sector_returns = 0.01 + 0.5 * factor_duration + 0.02 * np.random.randn(n)

        # Introduce some NaN values
        sector_returns_list = list(sector_returns)
        sector_returns_list[100] = None
        sector_returns_list[150] = None

        sector_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["TEST"] * n,
                "return": sector_returns_list,
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": factor_duration,
                "factor_credit": factor_credit,
                "factor_liquidity": factor_liquidity,
            }
        )

        # Should handle missing values gracefully
        results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
        )

        # Should still produce results (missing values dropped during join)
        assert "TEST" in results

    def test_zero_variance_in_window(self) -> None:
        """Test handling of zero variance factor in a window."""
        n = 200
        np.random.seed(42)

        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

        factor_duration = np.random.randn(n)
        # Make credit factor constant in first window
        factor_credit = np.concatenate([np.ones(100), np.random.randn(100)])
        factor_liquidity = np.random.randn(n)

        sector_returns = 0.01 + 0.5 * factor_duration + 0.02 * np.random.randn(n)

        sector_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["TEST"] * n,
                "return": sector_returns,
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": factor_duration,
                "factor_credit": factor_credit,
                "factor_liquidity": factor_liquidity,
            }
        )

        # Should handle zero variance gracefully (statsmodels handles this)
        results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
            min_observations=100,
        )

        # Should still have some valid windows
        assert "TEST" in results

    def test_extreme_outliers_in_window(self) -> None:
        """Test handling of extreme outliers."""
        n = 300
        np.random.seed(42)

        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

        factor_duration = np.random.randn(n)
        factor_credit = np.random.randn(n)
        factor_liquidity = np.random.randn(n)

        sector_returns = 0.01 + 0.5 * factor_duration + 0.02 * np.random.randn(n)

        # Add extreme outliers
        sector_returns_list = list(sector_returns)
        sector_returns_list[50] = 10.0  # Extreme positive
        sector_returns_list[100] = -10.0  # Extreme negative

        sector_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["TEST"] * n,
                "return": sector_returns_list,
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": factor_duration,
                "factor_credit": factor_credit,
                "factor_liquidity": factor_liquidity,
            }
        )

        # Should handle outliers (OLS is sensitive but won't crash)
        results = compute_rolling_betas(
            sector_df,
            factor_df,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            window_days=126,
        )

        assert "TEST" in results
        assert results["TEST"].n_windows > 0

    def test_missing_columns_error(self) -> None:
        """Test error when required columns are missing."""
        sector_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)],
                "symbol": ["TEST"],
                # Missing 'return' column
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)],
                "factor_duration": [0.1],
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            compute_rolling_betas(
                sector_df,
                factor_df,
                factor_columns=["factor_duration"],
            )

    def test_empty_factor_columns_error(self) -> None:
        """Test error when factor_columns is empty."""
        sector_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)],
                "symbol": ["TEST"],
                "return": [0.01],
            }
        )

        factor_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)],
            }
        )

        with pytest.raises(ValueError, match="factor_columns cannot be empty"):
            compute_rolling_betas(
                sector_df,
                factor_df,
                factor_columns=[],
            )
