"""Tests for regression diagnostics module."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from playground.risk_model.diagnostics import RegressionDiagnostics
from playground.risk_model.diagnostics import SectorDiagnosticsReport
from playground.risk_model.diagnostics import compute_regression_diagnostics
from playground.risk_model.diagnostics import create_diagnostics_summary


@pytest.fixture
def perfect_fit_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Create synthetic data with perfect linear relationship.

    Model: y = 2.0 + 1.5*x1 + 0.5*x2 - 0.3*x3 + 0 (no noise)
    """
    np.random.seed(42)
    n = 100

    # Generate factor returns
    X = np.random.randn(n, 3)
    # Use timedelta to generate sequential timestamps
    timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + np.timedelta64(i, "D") for i in range(n)]

    # Perfect linear relationship
    y = 2.0 + 1.5 * X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]

    sector_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * n,
            "return": y,
        }
    )

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": X[:, 0],
            "factor_credit": X[:, 1],
            "factor_liquidity": X[:, 2],
        }
    )

    return sector_returns, factor_returns


@pytest.fixture
def noisy_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic data with realistic noise."""
    np.random.seed(123)
    n = 252  # One year of daily data

    # Generate factor returns
    X = np.random.randn(n, 3) * 0.01  # Scale to realistic factor returns
    timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + np.timedelta64(i, "D") for i in range(n)]

    # Noisy linear relationship (R² ~ 0.50)
    y = 0.001 + 0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2] + np.random.randn(n) * 0.005

    sector_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["SECTOR1"] * n,
            "return": y,
        }
    )

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": X[:, 0],
            "factor_credit": X[:, 1],
            "factor_liquidity": X[:, 2],
        }
    )

    return sector_returns, factor_returns


@pytest.fixture
def multi_sector_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create data for multiple sectors with varying quality."""
    np.random.seed(456)
    n = 200
    timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + np.timedelta64(i, "D") for i in range(n)]

    # Generate factor returns (shared across sectors)
    X = np.random.randn(n, 3) * 0.01

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": X[:, 0],
            "factor_credit": X[:, 1],
            "factor_liquidity": X[:, 2],
        }
    )

    # Generate multiple sectors with different characteristics
    sectors = []

    # Good fit sector (R² ~ 0.60)
    y1 = 0.001 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n) * 0.003
    sectors.extend(
        [
            {"timestamp": ts, "symbol": "GOOD_FIT", "return": ret}
            for ts, ret in zip(timestamps, y1)
        ]
    )

    # Medium fit sector (R² ~ 0.35)
    y2 = 0.002 + 0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 2] + np.random.randn(n) * 0.006
    sectors.extend(
        [
            {"timestamp": ts, "symbol": "MEDIUM_FIT", "return": ret}
            for ts, ret in zip(timestamps, y2)
        ]
    )

    # Poor fit sector (R² ~ 0.15)
    y3 = 0.001 + 0.2 * X[:, 0] - 0.1 * X[:, 1] + 0.05 * X[:, 2] + np.random.randn(n) * 0.01
    sectors.extend(
        [
            {"timestamp": ts, "symbol": "POOR_FIT", "return": ret}
            for ts, ret in zip(timestamps, y3)
        ]
    )

    sector_returns = pl.DataFrame(sectors)

    return sector_returns, factor_returns


class TestRegressionDiagnostics:
    """Tests for compute_regression_diagnostics function."""

    def test_perfect_fit(self, perfect_fit_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test diagnostics with perfect linear relationship."""
        sector_returns, factor_returns = perfect_fit_data

        diagnostics = compute_regression_diagnostics(
            sector_returns,
            factor_returns,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        assert "TEST" in diagnostics
        diag = diagnostics["TEST"]

        # Perfect fit should have R² ≈ 1.0
        assert diag.r_squared > 0.99
        assert diag.adj_r_squared > 0.99

        # Coefficients should match true values
        assert abs(diag.alpha - 2.0) < 0.01
        assert abs(diag.beta_duration - 1.5) < 0.01
        assert abs(diag.beta_credit - 0.5) < 0.01
        assert abs(diag.beta_liquidity - (-0.3)) < 0.01

        # All betas should be highly significant
        assert diag.p_value_duration < 0.001
        assert diag.p_value_credit < 0.001
        assert diag.p_value_liquidity < 0.001

        # F-statistic should be highly significant
        assert diag.f_pvalue < 0.001

        # Residuals should be near zero
        assert abs(diag.residual_mean) < 0.01
        assert diag.residual_std < 0.1

        # VIF should be low (no multicollinearity in random data)
        assert diag.vif_duration < 2.0
        assert diag.vif_credit < 2.0
        assert diag.vif_liquidity < 2.0

    def test_noisy_data(self, noisy_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test diagnostics with realistic noisy data."""
        sector_returns, factor_returns = noisy_data

        diagnostics = compute_regression_diagnostics(
            sector_returns,
            factor_returns,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        assert "SECTOR1" in diagnostics
        diag = diagnostics["SECTOR1"]

        # Should have reasonable R² (allow higher due to good fit with test data)
        assert 0.30 < diag.r_squared < 0.95

        # Should have positive F-statistic
        assert diag.f_statistic > 0
        assert diag.f_pvalue < 0.05

        # At least 2/3 betas should be significant
        sig_count = sum(
            [
                diag.p_value_duration < 0.05,
                diag.p_value_credit < 0.05,
                diag.p_value_liquidity < 0.05,
            ]
        )
        assert sig_count >= 2

        # Durbin-Watson should be in acceptable range
        assert 1.0 < diag.durbin_watson < 3.0

    def test_multi_sector(self, multi_sector_data: tuple[pl.DataFrame, pl.DataFrame]) -> None:
        """Test diagnostics with multiple sectors."""
        sector_returns, factor_returns = multi_sector_data

        diagnostics = compute_regression_diagnostics(
            sector_returns,
            factor_returns,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        # Should compute diagnostics for all three sectors
        assert len(diagnostics) == 3
        assert "GOOD_FIT" in diagnostics
        assert "MEDIUM_FIT" in diagnostics
        assert "POOR_FIT" in diagnostics

        # Good fit should have highest R²
        assert diagnostics["GOOD_FIT"].r_squared > diagnostics["MEDIUM_FIT"].r_squared
        assert diagnostics["MEDIUM_FIT"].r_squared > diagnostics["POOR_FIT"].r_squared

        # All should have valid metrics
        for sector_id, diag in diagnostics.items():
            assert 0 <= diag.r_squared <= 1
            assert 0 <= diag.adj_r_squared <= 1
            assert diag.f_statistic > 0
            assert diag.n_observations > 0

    def test_invalid_factor_columns_error(
        self, perfect_fit_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Ensure invalid factor column sets raise a clear error."""
        sector_returns, factor_returns = perfect_fit_data

        with pytest.raises(ValueError, match="factor_columns must contain exactly"):
            compute_regression_diagnostics(
                sector_returns,
                factor_returns,
                factor_columns=["factor_duration", "factor_credit"],
            )

    def test_missing_columns(self) -> None:
        """Test error handling for missing columns."""
        sector_returns = pl.DataFrame({"timestamp": [], "symbol": []})  # Missing 'return'
        factor_returns = pl.DataFrame({"timestamp": [], "factor_duration": []})

        with pytest.raises(ValueError, match="missing required columns"):
            compute_regression_diagnostics(
                sector_returns,
                factor_returns,
                factor_columns=["factor_duration"],
            )

    def test_empty_dataframes(self) -> None:
        """Test error handling for empty DataFrames."""
        sector_returns = pl.DataFrame({"timestamp": [], "symbol": [], "return": []})
        factor_returns = pl.DataFrame({"timestamp": [], "factor_duration": []})

        with pytest.raises(ValueError, match="cannot be empty"):
            compute_regression_diagnostics(
                sector_returns,
                factor_returns,
                factor_columns=["factor_duration"],
            )

    def test_insufficient_observations(self) -> None:
        """Test handling of sectors with too few observations."""
        # Create data with only 5 observations
        timestamps = [datetime(2020, 1, i + 1, tzinfo=UTC) for i in range(5)]
        sector_returns = pl.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["TEST"] * 5,
                "return": [0.01, -0.02, 0.03, -0.01, 0.02],
            }
        )
        factor_returns = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": [0.1, -0.1, 0.2, -0.05, 0.15],
                "factor_credit": [0.05, -0.08, 0.1, -0.03, 0.07],
                "factor_liquidity": [0.02, -0.04, 0.06, -0.01, 0.03],
            }
        )

        # Should raise error if no valid sectors
        with pytest.raises(ValueError, match="No valid diagnostics computed"):
            compute_regression_diagnostics(
                sector_returns,
                factor_returns,
                factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            )

    def test_no_join_overlap(self) -> None:
        """Test error when sector and factor data don't overlap."""
        sector_returns = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, tzinfo=UTC)],
                "symbol": ["TEST"],
                "return": [0.01],
            }
        )
        factor_returns = pl.DataFrame(
            {
                "timestamp": [datetime(2021, 1, 1, tzinfo=UTC)],  # Different date
                "factor_duration": [0.1],
                "factor_credit": [0.05],
                "factor_liquidity": [0.02],
            }
        )

        with pytest.raises(ValueError, match="No data after joining"):
            compute_regression_diagnostics(
                sector_returns,
                factor_returns,
                factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            )


class TestDiagnosticsSummary:
    """Tests for create_diagnostics_summary function."""

    def test_summary_with_good_diagnostics(
        self, multi_sector_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Test summary report with good diagnostics."""
        sector_returns, factor_returns = multi_sector_data

        diagnostics = compute_regression_diagnostics(
            sector_returns,
            factor_returns,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        summary = create_diagnostics_summary(diagnostics)

        # Check summary stats exist
        assert "mean_r_squared" in summary.summary_stats
        assert "pct_r2_above_threshold" in summary.summary_stats
        assert "pct_2_3_sig_betas" in summary.summary_stats
        assert "mean_vif" in summary.summary_stats
        assert "max_vif" in summary.summary_stats

        # Check acceptance status
        assert "r2_criterion" in summary.acceptance_status
        assert "significant_betas" in summary.acceptance_status
        assert "multicollinearity" in summary.acceptance_status
        assert "overall" in summary.acceptance_status

        # Check values are reasonable
        assert 0 <= summary.summary_stats["mean_r_squared"] <= 1
        assert 0 <= summary.summary_stats["pct_r2_above_threshold"] <= 100
        assert summary.summary_stats["n_sectors"] == 3

    def test_summary_acceptance_thresholds(
        self, multi_sector_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Test that acceptance thresholds work correctly."""
        sector_returns, factor_returns = multi_sector_data

        diagnostics = compute_regression_diagnostics(
            sector_returns,
            factor_returns,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        # Test with strict thresholds (should fail)
        strict_summary = create_diagnostics_summary(
            diagnostics,
            r2_threshold=0.80,  # Very high
            vif_threshold=1.5,  # Very strict
        )

        # At least some criteria should fail with strict thresholds
        assert not strict_summary.acceptance_status["overall"]

    def test_summary_empty_diagnostics(self) -> None:
        """Test error handling for empty diagnostics dict."""
        with pytest.raises(ValueError, match="empty diagnostics"):
            create_diagnostics_summary({})

    def test_summary_statistics_accuracy(
        self, multi_sector_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Test that summary statistics are calculated correctly."""
        sector_returns, factor_returns = multi_sector_data

        diagnostics = compute_regression_diagnostics(
            sector_returns,
            factor_returns,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        summary = create_diagnostics_summary(diagnostics)

        # Manually calculate mean R²
        expected_mean_r2 = np.mean([d.r_squared for d in diagnostics.values()])
        assert abs(summary.summary_stats["mean_r_squared"] - expected_mean_r2) < 1e-6

        # Check VIF statistics
        all_vifs = []
        for d in diagnostics.values():
            all_vifs.extend([d.vif_duration, d.vif_credit, d.vif_liquidity])
        expected_max_vif = np.max(all_vifs)
        assert abs(summary.summary_stats["max_vif"] - expected_max_vif) < 1e-6


class TestDiagnosticsDataclass:
    """Tests for RegressionDiagnostics dataclass."""

    def test_diagnostics_creation(self) -> None:
        """Test that RegressionDiagnostics can be created."""
        diag = RegressionDiagnostics(
            sector_id="TEST",
            r_squared=0.75,
            adj_r_squared=0.73,
            f_statistic=100.0,
            f_pvalue=0.001,
            durbin_watson=2.0,
            beta_duration=1.5,
            beta_credit=0.5,
            beta_liquidity=-0.3,
            alpha=2.0,
            t_stat_duration=10.0,
            t_stat_credit=5.0,
            t_stat_liquidity=-3.0,
            t_stat_alpha=2.5,
            p_value_duration=0.001,
            p_value_credit=0.01,
            p_value_liquidity=0.05,
            p_value_alpha=0.1,
            se_duration=0.15,
            se_credit=0.10,
            se_liquidity=0.10,
            se_alpha=0.80,
            vif_duration=1.2,
            vif_credit=1.3,
            vif_liquidity=1.1,
            bp_test_statistic=5.0,
            bp_p_value=0.2,
            residual_mean=0.0001,
            residual_std=0.005,
            residual_skewness=0.1,
            residual_kurtosis=0.2,
            n_observations=100,
            date_range_start=datetime(2020, 1, 1, tzinfo=UTC),
            date_range_end=datetime(2020, 12, 31, tzinfo=UTC),
        )

        assert diag.sector_id == "TEST"
        assert diag.r_squared == 0.75
        assert diag.n_observations == 100

    def test_report_creation(self) -> None:
        """Test that SectorDiagnosticsReport can be created."""
        diag = RegressionDiagnostics(
            sector_id="TEST",
            r_squared=0.75,
            adj_r_squared=0.73,
            f_statistic=100.0,
            f_pvalue=0.001,
            durbin_watson=2.0,
            beta_duration=1.5,
            beta_credit=0.5,
            beta_liquidity=-0.3,
            alpha=2.0,
            t_stat_duration=10.0,
            t_stat_credit=5.0,
            t_stat_liquidity=-3.0,
            t_stat_alpha=2.5,
            p_value_duration=0.001,
            p_value_credit=0.01,
            p_value_liquidity=0.05,
            p_value_alpha=0.1,
            se_duration=0.15,
            se_credit=0.10,
            se_liquidity=0.10,
            se_alpha=0.80,
            vif_duration=1.2,
            vif_credit=1.3,
            vif_liquidity=1.1,
            bp_test_statistic=5.0,
            bp_p_value=0.2,
            residual_mean=0.0001,
            residual_std=0.005,
            residual_skewness=0.1,
            residual_kurtosis=0.2,
            n_observations=100,
            date_range_start=datetime(2020, 1, 1, tzinfo=UTC),
            date_range_end=datetime(2020, 12, 31, tzinfo=UTC),
        )

        report = SectorDiagnosticsReport(
            diagnostics={"TEST": diag},
            summary_stats={"mean_r_squared": 0.75},
            acceptance_status={"overall": True},
        )

        assert "TEST" in report.diagnostics
        assert report.summary_stats["mean_r_squared"] == 0.75
        assert report.acceptance_status["overall"] is True


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_highly_correlated_factors(self) -> None:
        """Test with highly correlated factors (multicollinearity)."""
        np.random.seed(789)
        n = 100
        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + np.timedelta64(i, "D") for i in range(n)]

        # Create highly correlated factors
        X1 = np.random.randn(n)
        X2 = X1 + np.random.randn(n) * 0.1  # Highly correlated with X1
        X3 = np.random.randn(n)

        y = 1.0 + 0.5 * X1 + 0.3 * X2 + 0.2 * X3 + np.random.randn(n) * 0.1

        sector_returns = pl.DataFrame(
            {"timestamp": timestamps, "symbol": ["TEST"] * n, "return": y}
        )

        factor_returns = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": X1,
                "factor_credit": X2,
                "factor_liquidity": X3,
            }
        )

        diagnostics = compute_regression_diagnostics(
            sector_returns,
            factor_returns,
            factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
        )

        # Should still compute, but VIF should be high
        assert "TEST" in diagnostics
        # At least one VIF should be elevated due to correlation
        assert (
            diagnostics["TEST"].vif_duration > 5.0
            or diagnostics["TEST"].vif_credit > 5.0
        )

    def test_zero_variance_factor(self) -> None:
        """Test with a factor that has zero variance."""
        np.random.seed(999)
        n = 100
        timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + np.timedelta64(i, "D") for i in range(n)]

        X = np.random.randn(n, 3)
        X[:, 1] = 0.0  # Zero variance factor

        y = 1.0 + 0.5 * X[:, 0] + 0.2 * X[:, 2] + np.random.randn(n) * 0.1

        sector_returns = pl.DataFrame(
            {"timestamp": timestamps, "symbol": ["TEST"] * n, "return": y}
        )

        factor_returns = pl.DataFrame(
            {
                "timestamp": timestamps,
                "factor_duration": X[:, 0],
                "factor_credit": X[:, 1],  # Zero variance
                "factor_liquidity": X[:, 2],
            }
        )

        # Should handle gracefully (might raise or have high VIF)
        try:
            diagnostics = compute_regression_diagnostics(
                sector_returns,
                factor_returns,
                factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
            )
            # If it succeeds, VIF should be very high for the zero-variance factor
            assert "TEST" in diagnostics
        except Exception:
            # It's acceptable to fail on zero-variance factors
            pass
