"""
Tests for structural break detection in factor betas.

This module validates the Chow test implementation for detecting structural
breaks in factor regression betas across major market regime changes.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.dataset import SectorDataset
from playground.risk_model.structural_break_tests import ChowTestResult
from playground.risk_model.structural_break_tests import StructuralBreakSummary
from playground.risk_model.structural_break_tests import compute_chow_test
from playground.risk_model.structural_break_tests import compute_structural_break_analysis


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_dataset_no_break() -> SectorDataset:
    """
    Create synthetic dataset with stable betas (no structural break).

    Factor model: R = 0.0001 + 0.5*Duration + 0.3*Credit + 0.2*Liquidity + ε
    where ε ~ N(0, 0.01)

    Timeline: 100 days (50 pre, 50 post)
    """
    np.random.seed(42)
    n = 100
    timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

    # Generate factors
    duration = np.random.randn(n) * 0.01
    credit = np.random.randn(n) * 0.01
    liquidity = np.random.randn(n) * 0.01

    # Generate sector returns with stable betas
    epsilon = np.random.randn(n) * 0.01
    sector_returns_vals = 0.0001 + 0.5 * duration + 0.3 * credit + 0.2 * liquidity + epsilon

    sector_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["XLK"] * n,
            "return": sector_returns_vals,
        }
    )

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": duration,
            "factor_credit": credit,
            "factor_liquidity": liquidity,
        }
    )

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=n,
        factor_expected_days=n,
        sector_coverage={"XLK": 1.0},
        factor_coverage={"factor_duration": 1.0, "factor_credit": 1.0, "factor_liquidity": 1.0},
    )

    return SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )


@pytest.fixture
def synthetic_dataset_with_break() -> SectorDataset:
    """
    Create synthetic dataset with structural break at t=50.

    Pre-break: R = 0.0001 + 0.5*Duration + 0.3*Credit + 0.2*Liquidity + ε
    Post-break: R = 0.0001 + 1.0*Duration + 0.8*Credit + 0.5*Liquidity + ε

    Timeline: 100 days (50 pre, 50 post)
    """
    np.random.seed(42)
    n_pre = 50
    n_post = 50
    n = n_pre + n_post

    timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

    # Generate factors
    duration = np.random.randn(n) * 0.01
    credit = np.random.randn(n) * 0.01
    liquidity = np.random.randn(n) * 0.01

    # Generate sector returns with structural break
    epsilon = np.random.randn(n) * 0.01
    sector_returns_vals = np.zeros(n)

    # Pre-break period (0:50)
    sector_returns_vals[:n_pre] = (
        0.0001
        + 0.5 * duration[:n_pre]
        + 0.3 * credit[:n_pre]
        + 0.2 * liquidity[:n_pre]
        + epsilon[:n_pre]
    )

    # Post-break period (50:100) - significantly different betas
    sector_returns_vals[n_pre:] = (
        0.0001
        + 1.0 * duration[n_pre:]
        + 0.8 * credit[n_pre:]
        + 0.5 * liquidity[n_pre:]
        + epsilon[n_pre:]
    )

    sector_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["XLK"] * n,
            "return": sector_returns_vals,
        }
    )

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": duration,
            "factor_credit": credit,
            "factor_liquidity": liquidity,
        }
    )

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=n,
        factor_expected_days=n,
        sector_coverage={"XLK": 1.0},
        factor_coverage={"factor_duration": 1.0, "factor_credit": 1.0, "factor_liquidity": 1.0},
    )

    return SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )


@pytest.fixture
def multi_sector_dataset() -> SectorDataset:
    """
    Create dataset with multiple sectors for comprehensive testing.

    Sectors: XLK, XLF, XLU
    Timeline: 200 days
    """
    np.random.seed(123)
    n = 200
    timestamps = [datetime(2010, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

    # Generate factors
    duration = np.random.randn(n) * 0.01
    credit = np.random.randn(n) * 0.01
    liquidity = np.random.randn(n) * 0.01

    sector_data = []
    for sector in ["XLK", "XLF", "XLU"]:
        epsilon = np.random.randn(n) * 0.01
        # Different betas for different sectors
        if sector == "XLK":
            returns = 0.0001 + 0.5 * duration + 0.3 * credit + 0.2 * liquidity + epsilon
        elif sector == "XLF":
            returns = 0.0001 + 0.4 * duration + 0.6 * credit + 0.3 * liquidity + epsilon
        else:  # XLU
            returns = 0.0001 + 0.2 * duration + 0.1 * credit + 0.4 * liquidity + epsilon

        for i in range(n):
            sector_data.append(
                {
                    "timestamp": timestamps[i],
                    "symbol": sector,
                    "return": returns[i],
                }
            )

    sector_returns = pl.DataFrame(sector_data)

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": duration,
            "factor_credit": credit,
            "factor_liquidity": liquidity,
        }
    )

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=n,
        factor_expected_days=n,
        sector_coverage={"XLK": 1.0, "XLF": 1.0, "XLU": 1.0},
        factor_coverage={"factor_duration": 1.0, "factor_credit": 1.0, "factor_liquidity": 1.0},
    )

    return SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )


# ============================================================================
# Core Functionality Tests
# ============================================================================


def test_chow_test_synthetic_no_break(synthetic_dataset_no_break: SectorDataset) -> None:
    """
    Test Chow test on synthetic data with no structural break.

    Expected: Low F-statistic, p > 0.05, no break detected.
    """
    break_date = datetime(2020, 2, 20, tzinfo=UTC)  # Middle of data

    result = compute_chow_test(
        synthetic_dataset_no_break,
        "XLK",
        break_date,
    )

    # Assertions
    assert isinstance(result, ChowTestResult)
    assert result.sector_id == "XLK"
    assert result.break_date == break_date
    assert result.f_statistic >= 0.0
    assert 0.0 <= result.p_value <= 1.0
    assert result.p_value > 0.05, "Should not detect break when none exists"
    assert result.structural_break_detected is False
    assert result.pre_break_n > 0
    assert result.post_break_n > 0
    assert result.degrees_of_freedom_numerator == 4
    assert result.degrees_of_freedom_denominator > 0


def test_chow_test_synthetic_with_break(synthetic_dataset_with_break: SectorDataset) -> None:
    """
    Test Chow test on synthetic data with known structural break.

    Expected: High F-statistic, p < 0.05, break detected.
    """
    break_date = datetime(2020, 2, 20, tzinfo=UTC)  # Break point

    result = compute_chow_test(
        synthetic_dataset_with_break,
        "XLK",
        break_date,
    )

    # Assertions
    assert isinstance(result, ChowTestResult)
    assert result.sector_id == "XLK"
    assert result.structural_break_detected is True, "Should detect structural break"
    assert result.p_value < 0.05, f"p-value {result.p_value} should be < 0.05"
    assert result.f_statistic > result.critical_value_5pct

    # Check beta changes are significant
    assert abs(result.beta_change_magnitude["duration"]) > 10.0  # >10% change
    assert abs(result.beta_change_magnitude["credit"]) > 10.0
    assert abs(result.beta_change_magnitude["liquidity"]) > 10.0


def test_beta_change_magnitude_calculation(synthetic_dataset_with_break: SectorDataset) -> None:
    """
    Test beta change magnitude calculation.

    Expected: Percentage changes reflect true beta shifts.
    """
    break_date = datetime(2020, 2, 20, tzinfo=UTC)

    result = compute_chow_test(
        synthetic_dataset_with_break,
        "XLK",
        break_date,
    )

    # Pre-break betas should be close to [0.5, 0.3, 0.2]
    assert 0.3 < result.pre_break_betas["duration"] < 0.7
    assert 0.1 < result.pre_break_betas["credit"] < 0.5
    assert 0.0 < result.pre_break_betas["liquidity"] < 0.4

    # Post-break betas should be close to [1.0, 0.8, 0.5]
    assert 0.75 < result.post_break_betas["duration"] < 1.25
    assert 0.55 < result.post_break_betas["credit"] < 1.05
    assert 0.25 < result.post_break_betas["liquidity"] < 0.75

    # Magnitude changes should be positive (increase)
    assert result.beta_change_magnitude["duration"] > 0
    assert result.beta_change_magnitude["credit"] > 0
    assert result.beta_change_magnitude["liquidity"] > 0


def test_chow_test_invalid_sector_id(synthetic_dataset_no_break: SectorDataset) -> None:
    """Test error handling for invalid sector ID."""
    break_date = datetime(2020, 2, 20, tzinfo=UTC)

    with pytest.raises(ValueError, match="not found in dataset"):
        compute_chow_test(
            synthetic_dataset_no_break,
            "INVALID",
            break_date,
        )


def test_chow_test_break_date_before_data(synthetic_dataset_no_break: SectorDataset) -> None:
    """Test error handling for break date before data range."""
    break_date = datetime(2019, 1, 1, tzinfo=UTC)  # Before data starts

    with pytest.raises(ValueError, match="Insufficient pre-break observations"):
        compute_chow_test(
            synthetic_dataset_no_break,
            "XLK",
            break_date,
        )


def test_chow_test_break_date_after_data(synthetic_dataset_no_break: SectorDataset) -> None:
    """Test error handling for break date after data range."""
    break_date = datetime(2025, 1, 1, tzinfo=UTC)  # After data ends

    with pytest.raises(ValueError, match="Insufficient post-break observations"):
        compute_chow_test(
            synthetic_dataset_no_break,
            "XLK",
            break_date,
        )


def test_chow_test_insufficient_pre_break_data(synthetic_dataset_no_break: SectorDataset) -> None:
    """Test error handling for insufficient pre-break observations."""
    # Break date too early (only 5 days of pre-break data)
    break_date = datetime(2020, 1, 6, tzinfo=UTC)

    with pytest.raises(ValueError, match="Insufficient pre-break observations"):
        compute_chow_test(
            synthetic_dataset_no_break,
            "XLK",
            break_date,
            min_observations_per_period=20,
        )


def test_chow_test_insufficient_post_break_data(synthetic_dataset_no_break: SectorDataset) -> None:
    """Test error handling for insufficient post-break observations."""
    # Break date too late (only 5 days of post-break data)
    break_date = datetime(2020, 4, 5, tzinfo=UTC)

    with pytest.raises(ValueError, match="Insufficient post-break observations"):
        compute_chow_test(
            synthetic_dataset_no_break,
            "XLK",
            break_date,
            min_observations_per_period=20,
        )


def test_chow_test_empty_dataset() -> None:
    """Test error handling for empty dataset."""
    empty_sector = pl.DataFrame(
        {
            "timestamp": [],
            "symbol": [],
            "return": [],
        }
    ).cast({"timestamp": pl.Datetime, "symbol": pl.String, "return": pl.Float64})

    empty_factors = pl.DataFrame(
        {
            "timestamp": [],
            "factor_duration": [],
            "factor_credit": [],
            "factor_liquidity": [],
        }
    ).cast(
        {
            "timestamp": pl.Datetime,
            "factor_duration": pl.Float64,
            "factor_credit": pl.Float64,
            "factor_liquidity": pl.Float64,
        }
    )

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=0,
        factor_expected_days=0,
        sector_coverage={},
        factor_coverage={},
    )

    dataset = SectorDataset(
        sector_returns=empty_sector,
        factor_returns=empty_factors,
        coverage=coverage,
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        compute_chow_test(
            dataset,
            "XLK",
            datetime(2020, 1, 1, tzinfo=UTC),
        )


def test_chow_test_missing_factor_column(synthetic_dataset_no_break: SectorDataset) -> None:
    """Test error handling for missing factor column."""
    break_date = datetime(2020, 2, 20, tzinfo=UTC)

    with pytest.raises(ValueError, match=r"not found in dataset\.factor_returns"):
        compute_chow_test(
            synthetic_dataset_no_break,
            "XLK",
            break_date,
            factor_columns=("factor_duration", "factor_nonexistent", "factor_liquidity"),
        )


def test_chow_test_invalid_factor_column_set(synthetic_dataset_no_break: SectorDataset) -> None:
    """Factor sets that exclude 3D model columns should raise an error."""
    break_date = datetime(2020, 2, 20, tzinfo=UTC)

    with pytest.raises(ValueError, match="factor_columns must contain exactly"):
        compute_chow_test(
            synthetic_dataset_no_break,
            "XLK",
            break_date,
            factor_columns=("factor_duration", "factor_credit"),
        )


# ============================================================================
# Structural Break Analysis Tests
# ============================================================================


def test_structural_break_analysis_multi_sector(multi_sector_dataset: SectorDataset) -> None:
    """
    Test structural break analysis across multiple sectors and dates.
    """
    sectors = ["XLK", "XLF", "XLU"]
    break_dates = [
        datetime(2010, 3, 15, tzinfo=UTC),  # ~75 days pre, ~125 post
        datetime(2010, 5, 1, tzinfo=UTC),  # ~120 days pre, ~80 post
    ]

    summary = compute_structural_break_analysis(
        multi_sector_dataset,
        sectors,
        break_dates,
    )

    # Assertions
    assert isinstance(summary, StructuralBreakSummary)
    assert summary.n_sectors == 3
    assert summary.n_break_dates == 2
    assert summary.n_total_tests == 6  # 3 sectors * 2 dates
    assert len(summary.test_results) == 6
    assert 0.0 <= summary.break_detection_rate <= 1.0
    assert len(summary.breaks_by_date) <= 2
    assert len(summary.breaks_by_sector) <= 3


def test_structural_break_analysis_all_break_dates() -> None:
    """
    Test analysis with all major crisis dates.
    """
    np.random.seed(456)
    n = 4852  # ~13.3 years of daily data to cover 2007-2020
    start_date = datetime(2007, 1, 1, tzinfo=UTC)
    timestamps = [start_date + timedelta(days=i) for i in range(n)]

    # Generate factors
    duration = np.random.randn(n) * 0.01
    credit = np.random.randn(n) * 0.01
    liquidity = np.random.randn(n) * 0.01

    # Generate sector returns
    epsilon = np.random.randn(n) * 0.01
    returns = 0.0001 + 0.5 * duration + 0.3 * credit + 0.2 * liquidity + epsilon

    sector_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["XLK"] * n,
            "return": returns,
        }
    )

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": duration,
            "factor_credit": credit,
            "factor_liquidity": liquidity,
        }
    )

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=n,
        factor_expected_days=n,
        sector_coverage={"XLK": 1.0},
        factor_coverage={"factor_duration": 1.0, "factor_credit": 1.0, "factor_liquidity": 1.0},
    )

    dataset = SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )

    # Test major crisis dates
    break_dates = [
        datetime(2008, 9, 15, tzinfo=UTC),  # Lehman collapse
        datetime(2020, 3, 15, tzinfo=UTC),  # COVID crash
    ]

    summary = compute_structural_break_analysis(
        dataset,
        ["XLK"],
        break_dates,
    )

    assert summary.n_total_tests == 2
    assert len(summary.test_results) == 2
    assert all(isinstance(r, ChowTestResult) for r in summary.test_results)


def test_structural_break_summary_most_unstable_identification(
    multi_sector_dataset: SectorDataset,
) -> None:
    """
    Test that summary correctly identifies most unstable sectors/dates.
    """
    # Create dataset with known instability pattern
    sectors = ["XLK", "XLF", "XLU"]
    break_dates = [
        datetime(2010, 3, 15, tzinfo=UTC),
        datetime(2010, 5, 1, tzinfo=UTC),
    ]

    summary = compute_structural_break_analysis(
        multi_sector_dataset,
        sectors,
        break_dates,
    )

    # Most unstable sectors/dates should be sorted by break count
    if summary.most_unstable_sectors:
        for i in range(len(summary.most_unstable_sectors) - 1):
            sector_i = summary.most_unstable_sectors[i]
            sector_j = summary.most_unstable_sectors[i + 1]
            assert summary.breaks_by_sector[sector_i] >= summary.breaks_by_sector[sector_j]

    if summary.most_unstable_dates:
        for i in range(len(summary.most_unstable_dates) - 1):
            date_i = summary.most_unstable_dates[i]
            date_j = summary.most_unstable_dates[i + 1]
            assert summary.breaks_by_date[date_i] >= summary.breaks_by_date[date_j]


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


def test_chow_test_zero_variance_factors() -> None:
    """
    Test handling of zero variance factors (all zeros).

    Should still run but may have low R² and unstable estimates.
    """
    n = 100
    timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

    # All factors are zero
    sector_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["XLK"] * n,
            "return": np.random.randn(n) * 0.01,
        }
    )

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": [0.0] * n,
            "factor_credit": [0.0] * n,
            "factor_liquidity": [0.0] * n,
        }
    )

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=n,
        factor_expected_days=n,
        sector_coverage={"XLK": 1.0},
        factor_coverage={"factor_duration": 1.0, "factor_credit": 1.0, "factor_liquidity": 1.0},
    )

    dataset = SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )

    break_date = datetime(2020, 2, 20, tzinfo=UTC)

    # Should complete without error (statsmodels handles singular matrices)
    result = compute_chow_test(dataset, "XLK", break_date)

    assert isinstance(result, ChowTestResult)
    # R² should be very low (factors don't explain returns)
    assert result.pooled_r_squared < 0.1


def test_chow_test_single_observation_periods_error() -> None:
    """
    Test that single observation per period raises error.
    """
    n = 10
    timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

    sector_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["XLK"] * n,
            "return": np.random.randn(n) * 0.01,
        }
    )

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": np.random.randn(n) * 0.01,
            "factor_credit": np.random.randn(n) * 0.01,
            "factor_liquidity": np.random.randn(n) * 0.01,
        }
    )

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=n,
        factor_expected_days=n,
        sector_coverage={"XLK": 1.0},
        factor_coverage={"factor_duration": 1.0, "factor_credit": 1.0, "factor_liquidity": 1.0},
    )

    dataset = SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )

    break_date = datetime(2020, 1, 6, tzinfo=UTC)

    with pytest.raises(ValueError, match="Insufficient"):
        compute_chow_test(dataset, "XLK", break_date, min_observations_per_period=10)


def test_chow_test_r_squared_values(synthetic_dataset_with_break: SectorDataset) -> None:
    """
    Test that R² values are computed correctly for all periods.
    """
    break_date = datetime(2020, 2, 20, tzinfo=UTC)

    result = compute_chow_test(
        synthetic_dataset_with_break,
        "XLK",
        break_date,
    )

    # All R² values should be in [0, 1]
    assert 0.0 <= result.pre_break_r_squared <= 1.0
    assert 0.0 <= result.post_break_r_squared <= 1.0
    assert 0.0 <= result.pooled_r_squared <= 1.0

    # For synthetic data with factor loadings, R² should be positive
    assert result.pre_break_r_squared > 0.2  # Some explained variance
    assert result.post_break_r_squared > 0.2


def test_chow_test_degrees_of_freedom(synthetic_dataset_no_break: SectorDataset) -> None:
    """
    Test degrees of freedom calculation.

    For 3 factors + intercept: k = 4
    df_numerator = k = 4
    df_denominator = n1 + n2 - 2k
    """
    break_date = datetime(2020, 2, 20, tzinfo=UTC)

    result = compute_chow_test(
        synthetic_dataset_no_break,
        "XLK",
        break_date,
    )

    k = 4  # 3 factors + intercept
    expected_df_num = k
    expected_df_denom = result.pre_break_n + result.post_break_n - 2 * k

    assert result.degrees_of_freedom_numerator == expected_df_num
    assert result.degrees_of_freedom_denominator == expected_df_denom


def test_structural_break_analysis_empty_sector_list(
    multi_sector_dataset: SectorDataset,
) -> None:
    """
    Test that empty sector list raises appropriate error.
    """
    break_dates = [datetime(2010, 3, 15, tzinfo=UTC)]

    with pytest.raises(ValueError, match="No valid Chow test results"):
        compute_structural_break_analysis(
            multi_sector_dataset,
            [],  # Empty sector list
            break_dates,
        )


def test_structural_break_analysis_partial_failures(multi_sector_dataset: SectorDataset) -> None:
    """
    Test that analysis continues when some tests fail due to insufficient data.
    """
    sectors = ["XLK", "XLF", "INVALID"]  # Include invalid sector
    break_dates = [datetime(2010, 3, 15, tzinfo=UTC)]

    # Should complete for valid sectors and log warnings for invalid
    summary = compute_structural_break_analysis(
        multi_sector_dataset,
        sectors,
        break_dates,
    )

    # Should have results for XLK and XLF only
    assert summary.n_total_tests == 2
    assert summary.n_sectors == 2
    assert all(r.sector_id in ["XLK", "XLF"] for r in summary.test_results)


def test_chow_test_custom_min_observations(synthetic_dataset_no_break: SectorDataset) -> None:
    """
    Test custom minimum observations parameter.
    """
    break_date = datetime(2020, 2, 20, tzinfo=UTC)

    # Should work with lower threshold
    result = compute_chow_test(
        synthetic_dataset_no_break,
        "XLK",
        break_date,
        min_observations_per_period=10,
    )

    assert isinstance(result, ChowTestResult)
    assert result.pre_break_n >= 10
    assert result.post_break_n >= 10


def test_beta_change_magnitude_zero_pre_beta() -> None:
    """
    Test beta change magnitude when pre-break beta is near zero.

    Should handle division by zero gracefully.
    """
    np.random.seed(789)
    n = 100
    timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]

    duration = np.random.randn(n) * 0.01
    credit = np.random.randn(n) * 0.01
    liquidity = np.random.randn(n) * 0.01

    epsilon = np.random.randn(n) * 0.01

    # Pre-break: duration beta = 0 (no exposure)
    # Post-break: duration beta = 0.5
    sector_returns_vals = np.zeros(n)
    sector_returns_vals[:50] = 0.0001 + 0.0 * duration[:50] + 0.3 * credit[:50] + epsilon[:50]
    sector_returns_vals[50:] = 0.0001 + 0.5 * duration[50:] + 0.3 * credit[50:] + epsilon[50:]

    sector_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["XLK"] * n,
            "return": sector_returns_vals,
        }
    )

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": duration,
            "factor_credit": credit,
            "factor_liquidity": liquidity,
        }
    )

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=n,
        factor_expected_days=n,
        sector_coverage={"XLK": 1.0},
        factor_coverage={"factor_duration": 1.0, "factor_credit": 1.0, "factor_liquidity": 1.0},
    )

    dataset = SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )

    break_date = datetime(2020, 2, 20, tzinfo=UTC)
    result = compute_chow_test(dataset, "XLK", break_date)

    # Should handle gracefully (returns a finite float or inf)
    assert isinstance(result.beta_change_magnitude["duration"], float)
    # When pre-break is near zero and post-break is non-zero, can return large value or inf
    # Just verify it's a valid float (not NaN)
    assert not np.isnan(result.beta_change_magnitude["duration"])
