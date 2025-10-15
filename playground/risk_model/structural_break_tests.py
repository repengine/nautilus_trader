"""
Structural break testing for 3D risk model factor betas.

This module implements the Chow test to detect structural breaks in factor
regression betas, validating whether factor exposures remain stable across
major market regime changes (e.g., 2008 Financial Crisis, 2020 COVID Crash,
2022 Rate Hiking Cycle).

Key capabilities:
- Chow test for structural breaks in factor betas
- F-statistic computation and hypothesis testing
- Pre/post break beta estimation and comparison
- Regime change validation for major market events
- Integration with SectorDataset for seamless testing

Performance: Cold path only (training/validation, not real-time inference)
Target: <1 second per sector-breakpoint test
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog
from scipy import stats


if TYPE_CHECKING:
    import statsmodels.api as sm
else:
    sm = None

from playground.risk_model.dataset import SectorDataset


# Lazy import of statsmodels to reduce startup time
_SM_MODULE: object | None = None


def _get_sm() -> object:
    """Lazy import of statsmodels to reduce startup time."""
    global _SM_MODULE
    if _SM_MODULE is None:
        import statsmodels.api

        _SM_MODULE = statsmodels.api
    return _SM_MODULE


LOGGER = structlog.get_logger(__name__)

EXPECTED_FACTOR_COLUMNS: tuple[str, str, str] = (
    "factor_duration",
    "factor_credit",
    "factor_liquidity",
)

# ============================================================================
# Data Structures
# ============================================================================


@dataclass(slots=True)
class ChowTestResult:
    """
    Result from Chow test for structural break in factor regression.

    The Chow test tests the null hypothesis that there is no structural break
    at the specified break date. Rejection (p < 0.05) indicates that factor
    betas differ significantly between pre-break and post-break periods.

    Attributes
    ----------
    sector_id : str
        Sector ETF ticker (e.g., "XLK", "XLU").
    break_date : datetime
        Date at which structural break is tested.
    f_statistic : float
        F-statistic for the Chow test.
    p_value : float
        P-value for the F-statistic.
    critical_value_5pct : float
        Critical value at 5% significance level.
    structural_break_detected : bool
        True if p < 0.05 (reject null of no break).
    pre_break_betas : dict[str, float]
        Factor betas estimated on pre-break period (duration, credit, liquidity).
    post_break_betas : dict[str, float]
        Factor betas estimated on post-break period.
    beta_change_magnitude : dict[str, float]
        Percentage change in betas from pre to post ((post-pre)/pre * 100).
    pre_break_n : int
        Number of observations in pre-break period.
    post_break_n : int
        Number of observations in post-break period.
    pre_break_r_squared : float
        R² for pre-break regression.
    post_break_r_squared : float
        R² for post-break regression.
    pooled_r_squared : float
        R² for pooled (full-sample) regression.
    degrees_of_freedom_numerator : int
        Degrees of freedom for F-test numerator (k).
    degrees_of_freedom_denominator : int
        Degrees of freedom for F-test denominator (n1 + n2 - 2k).
    """

    sector_id: str
    break_date: datetime
    f_statistic: float
    p_value: float
    critical_value_5pct: float
    structural_break_detected: bool
    pre_break_betas: dict[str, float]
    post_break_betas: dict[str, float]
    beta_change_magnitude: dict[str, float]
    pre_break_n: int
    post_break_n: int
    pre_break_r_squared: float
    post_break_r_squared: float
    pooled_r_squared: float
    degrees_of_freedom_numerator: int
    degrees_of_freedom_denominator: int


@dataclass(slots=True)
class StructuralBreakSummary:
    """
    Summary of structural break analysis across multiple sectors and dates.

    Attributes
    ----------
    test_results : list[ChowTestResult]
        Individual Chow test results for each sector-date combination.
    n_sectors : int
        Number of sectors tested.
    n_break_dates : int
        Number of break dates tested.
    n_total_tests : int
        Total number of tests performed (n_sectors * n_break_dates).
    n_breaks_detected : int
        Number of tests rejecting null (structural break detected).
    break_detection_rate : float
        Proportion of tests detecting breaks (0-1).
    breaks_by_date : dict[datetime, int]
        Count of sectors showing breaks at each date.
    breaks_by_sector : dict[str, int]
        Count of break dates showing breaks for each sector.
    most_unstable_sectors : list[str]
        Sectors with most breaks (sorted descending).
    most_unstable_dates : list[datetime]
        Dates with most breaks (sorted descending).
    """

    test_results: list[ChowTestResult]
    n_sectors: int
    n_break_dates: int
    n_total_tests: int
    n_breaks_detected: int
    break_detection_rate: float
    breaks_by_date: dict[datetime, int]
    breaks_by_sector: dict[str, int]
    most_unstable_sectors: list[str]
    most_unstable_dates: list[datetime]


# ============================================================================
# Core Chow Test Implementation
# ============================================================================


def compute_chow_test(
    dataset: SectorDataset,
    sector_id: str,
    break_date: datetime,
    *,
    factor_columns: Sequence[str] = ("factor_duration", "factor_credit", "factor_liquidity"),
    min_observations_per_period: int = 20,
) -> ChowTestResult:
    """
    Perform Chow test for structural break in factor regression.

    Tests the null hypothesis that factor betas are equal in pre-break and
    post-break periods. The test is implemented as an F-test comparing:
    - Unrestricted model: Separate regressions for pre/post periods
    - Restricted model: Pooled regression across full sample

    The F-statistic is computed as:
        F = ((RSS_pooled - (RSS_pre + RSS_post)) / k) / ((RSS_pre + RSS_post) / (n1 + n2 - 2k))

    where:
        k = number of parameters (3 factors + intercept = 4)
        n1 = observations in pre-break period
        n2 = observations in post-break period
        RSS = residual sum of squares

    Parameters
    ----------
    dataset : SectorDataset
        Dataset containing aligned sector and factor returns.
    sector_id : str
        Sector ETF ticker to test (must exist in dataset.sector_returns).
    break_date : datetime
        Date at which to test for structural break.
    factor_columns : Sequence[str]
        Factor column names in dataset.factor_returns.
        Default: ("factor_duration", "factor_credit", "factor_liquidity").
    min_observations_per_period : int
        Minimum observations required in each period (pre and post).
        Default: 20.

    Returns
    -------
    ChowTestResult
        Chow test results including F-statistic, p-value, and beta estimates.

    Raises
    ------
    ValueError
        If sector_id not in dataset, break_date outside data range, or
        insufficient observations in either period.

    Notes
    -----
    - Break date should align with known regime changes (2008-09-15, 2020-03-15, etc.)
    - Test requires at least min_observations_per_period in both periods
    - Uses OLS regression for beta estimation
    - F-statistic follows F-distribution with (k, n1+n2-2k) degrees of freedom
    - Critical values computed from scipy.stats.f.ppf at 5% significance level

    Examples
    --------
    >>> from playground.risk_model.dataset import SectorDataset
    >>> from datetime import datetime
    >>> # Assuming dataset is loaded
    >>> result = compute_chow_test(
    ...     dataset,
    ...     "XLK",
    ...     datetime(2020, 3, 15),
    ... )
    >>> print(f"F-stat: {result.f_statistic:.2f}, p-value: {result.p_value:.4f}")
    >>> print(f"Break detected: {result.structural_break_detected}")
    """
    LOGGER.info(
        "Computing Chow test",
        sector=sector_id,
        break_date=break_date.isoformat(),
        n_factors=len(factor_columns),
    )

    # Validate inputs
    _validate_chow_test_inputs(
        dataset,
        sector_id,
        break_date,
        factor_columns,
        min_observations_per_period,
    )

    # Extract sector data and join with factors
    sector_data = dataset.sector_returns.filter(pl.col("symbol") == sector_id)
    joined = sector_data.join(dataset.factor_returns, on="timestamp", how="inner").sort("timestamp")

    if joined.is_empty():
        msg = f"No data after joining sector {sector_id} with factors"
        raise ValueError(msg)

    # Split into pre and post periods
    pre_break = joined.filter(pl.col("timestamp") < break_date)
    post_break = joined.filter(pl.col("timestamp") >= break_date)

    # Validate sufficient observations
    if pre_break.height < min_observations_per_period:
        msg = (
            f"Insufficient pre-break observations for {sector_id} at {break_date.isoformat()}: "
            f"{pre_break.height} < {min_observations_per_period}"
        )
        raise ValueError(msg)

    if post_break.height < min_observations_per_period:
        msg = (
            f"Insufficient post-break observations for {sector_id} at {break_date.isoformat()}: "
            f"{post_break.height} < {min_observations_per_period}"
        )
        raise ValueError(msg)

    # Run regressions
    sm_module = _get_sm()

    # Pre-break regression
    y_pre = pre_break["return"].to_numpy()
    X_pre = pre_break.select(list(factor_columns)).to_numpy()
    X_pre_const = sm_module.add_constant(X_pre)  # type: ignore[attr-defined]
    model_pre = sm_module.OLS(y_pre, X_pre_const).fit()  # type: ignore[attr-defined]

    # Post-break regression
    y_post = post_break["return"].to_numpy()
    X_post = post_break.select(list(factor_columns)).to_numpy()
    X_post_const = sm_module.add_constant(X_post)  # type: ignore[attr-defined]
    model_post = sm_module.OLS(y_post, X_post_const).fit()  # type: ignore[attr-defined]

    # Pooled regression
    y_pooled = joined["return"].to_numpy()
    X_pooled = joined.select(list(factor_columns)).to_numpy()
    X_pooled_const = sm_module.add_constant(X_pooled)  # type: ignore[attr-defined]
    model_pooled = sm_module.OLS(y_pooled, X_pooled_const).fit()  # type: ignore[attr-defined]

    # Compute RSS (residual sum of squares)
    rss_pre = float(np.sum(model_pre.resid**2))
    rss_post = float(np.sum(model_post.resid**2))
    rss_pooled = float(np.sum(model_pooled.resid**2))

    # Compute F-statistic
    n1 = pre_break.height
    n2 = post_break.height
    k = len(factor_columns) + 1  # number of parameters (factors + intercept)

    numerator = (rss_pooled - (rss_pre + rss_post)) / k
    denominator = (rss_pre + rss_post) / (n1 + n2 - 2 * k)

    f_statistic = float(numerator / denominator) if denominator > 1e-10 else 0.0

    # Compute p-value and critical value
    df_numerator = k
    df_denominator = n1 + n2 - 2 * k

    p_value = float(1 - stats.f.cdf(f_statistic, df_numerator, df_denominator))
    critical_value_5pct = float(stats.f.ppf(0.95, df_numerator, df_denominator))

    # Structural break detected if p < 0.05
    structural_break_detected = p_value < 0.05

    # Extract betas
    pre_params = model_pre.params
    post_params = model_post.params
    pre_break_betas = {
        "duration": float(pre_params[1]),
        "credit": float(pre_params[2]),
        "liquidity": float(pre_params[3]),
    }

    post_break_betas = {
        "duration": float(post_params[1]),
        "credit": float(post_params[2]),
        "liquidity": float(post_params[3]),
    }

    # Compute beta change magnitude (percentage)
    beta_change_magnitude = {}
    for factor in ["duration", "credit", "liquidity"]:
        pre_val = pre_break_betas[factor]
        post_val = post_break_betas[factor]
        if abs(pre_val) > 1e-10:
            pct_change = ((post_val - pre_val) / abs(pre_val)) * 100.0
        else:
            pct_change = 0.0 if abs(post_val) < 1e-10 else float("inf")
        beta_change_magnitude[factor] = float(pct_change)

    LOGGER.info(
        "Chow test completed",
        sector=sector_id,
        break_date=break_date.isoformat(),
        f_statistic=f"{f_statistic:.2f}",
        p_value=f"{p_value:.4f}",
        break_detected=structural_break_detected,
    )

    return ChowTestResult(
        sector_id=sector_id,
        break_date=break_date,
        f_statistic=f_statistic,
        p_value=p_value,
        critical_value_5pct=critical_value_5pct,
        structural_break_detected=structural_break_detected,
        pre_break_betas=pre_break_betas,
        post_break_betas=post_break_betas,
        beta_change_magnitude=beta_change_magnitude,
        pre_break_n=n1,
        post_break_n=n2,
        pre_break_r_squared=float(model_pre.rsquared),
        post_break_r_squared=float(model_post.rsquared),
        pooled_r_squared=float(model_pooled.rsquared),
        degrees_of_freedom_numerator=df_numerator,
        degrees_of_freedom_denominator=df_denominator,
    )


def compute_structural_break_analysis(
    dataset: SectorDataset,
    sector_ids: Sequence[str],
    break_dates: Sequence[datetime],
    *,
    factor_columns: Sequence[str] = ("factor_duration", "factor_credit", "factor_liquidity"),
    min_observations_per_period: int = 20,
) -> StructuralBreakSummary:
    """
    Perform Chow test across multiple sectors and break dates.

    This function orchestrates structural break testing across all combinations
    of sectors and dates, producing a comprehensive summary of beta stability
    across major market regime changes.

    Parameters
    ----------
    dataset : SectorDataset
        Dataset containing aligned sector and factor returns.
    sector_ids : Sequence[str]
        Sector ETF tickers to test (e.g., ["XLK", "XLU", "XLF"]).
    break_dates : Sequence[datetime]
        Dates at which to test for structural breaks.
        Common choices: 2008-09-15, 2020-03-15, 2022-03-01.
    factor_columns : Sequence[str]
        Factor column names in dataset.factor_returns.
        Default: ("factor_duration", "factor_credit", "factor_liquidity").
    min_observations_per_period : int
        Minimum observations required in each period.
        Default: 20.

    Returns
    -------
    StructuralBreakSummary
        Summary statistics across all tests, including:
        - Individual test results
        - Break detection rates
        - Most unstable sectors/dates

    Notes
    -----
    - Total tests = len(sector_ids) * len(break_dates)
    - Failed tests (insufficient data) are logged and skipped
    - Summary identifies sectors/dates with highest instability

    Examples
    --------
    >>> from datetime import datetime
    >>> summary = compute_structural_break_analysis(
    ...     dataset,
    ...     ["XLK", "XLF", "XLU"],
    ...     [datetime(2008, 9, 15), datetime(2020, 3, 15)],
    ... )
    >>> print(f"Break detection rate: {summary.break_detection_rate:.1%}")
    >>> print(f"Most unstable sector: {summary.most_unstable_sectors[0]}")
    """
    LOGGER.info(
        "Starting structural break analysis",
        n_sectors=len(sector_ids),
        n_dates=len(break_dates),
        n_total_tests=len(sector_ids) * len(break_dates),
    )

    test_results: list[ChowTestResult] = []

    for sector_id in sector_ids:
        for break_date in break_dates:
            try:
                result = compute_chow_test(
                    dataset,
                    sector_id,
                    break_date,
                    factor_columns=factor_columns,
                    min_observations_per_period=min_observations_per_period,
                )
                test_results.append(result)
            except ValueError as e:
                LOGGER.warning(
                    "Skipping Chow test due to insufficient data",
                    sector=sector_id,
                    break_date=break_date.isoformat(),
                    error=str(e),
                )
                continue
            except Exception:
                LOGGER.exception(
                    "Failed to compute Chow test",
                    sector=sector_id,
                    break_date=break_date.isoformat(),
                )
                continue

    if not test_results:
        msg = "No valid Chow test results computed"
        raise ValueError(msg)

    # Compute summary statistics
    n_total_tests = len(test_results)
    n_breaks_detected = sum(1 for r in test_results if r.structural_break_detected)
    break_detection_rate = n_breaks_detected / n_total_tests if n_total_tests > 0 else 0.0

    # Count breaks by date
    breaks_by_date: dict[datetime, int] = {}
    for result in test_results:
        if result.structural_break_detected:
            breaks_by_date[result.break_date] = breaks_by_date.get(result.break_date, 0) + 1

    # Count breaks by sector
    breaks_by_sector: dict[str, int] = {}
    for result in test_results:
        if result.structural_break_detected:
            breaks_by_sector[result.sector_id] = breaks_by_sector.get(result.sector_id, 0) + 1

    # Identify most unstable sectors and dates
    most_unstable_sectors = sorted(
        breaks_by_sector.keys(),
        key=lambda s: breaks_by_sector[s],
        reverse=True,
    )
    most_unstable_dates = sorted(
        breaks_by_date.keys(),
        key=lambda d: breaks_by_date[d],
        reverse=True,
    )

    summary = StructuralBreakSummary(
        test_results=test_results,
        n_sectors=len({r.sector_id for r in test_results}),
        n_break_dates=len({r.break_date for r in test_results}),
        n_total_tests=n_total_tests,
        n_breaks_detected=n_breaks_detected,
        break_detection_rate=break_detection_rate,
        breaks_by_date=breaks_by_date,
        breaks_by_sector=breaks_by_sector,
        most_unstable_sectors=most_unstable_sectors,
        most_unstable_dates=most_unstable_dates,
    )

    LOGGER.info(
        "Structural break analysis completed",
        n_tests=n_total_tests,
        n_breaks=n_breaks_detected,
        detection_rate=f"{break_detection_rate:.1%}",
    )

    return summary


# ============================================================================
# Validation Helpers
# ============================================================================


def _validate_chow_test_inputs(
    dataset: SectorDataset,
    sector_id: str,
    break_date: datetime,
    factor_columns: Sequence[str],
    min_observations_per_period: int,
) -> None:
    """Validate inputs for Chow test."""
    # Validate dataset
    if dataset.sector_returns.is_empty():
        msg = "dataset.sector_returns cannot be empty"
        raise ValueError(msg)

    if dataset.factor_returns.is_empty():
        msg = "dataset.factor_returns cannot be empty"
        raise ValueError(msg)

    # Validate sector_id exists
    available_sectors = dataset.sector_returns["symbol"].unique().to_list()
    if sector_id not in available_sectors:
        msg = f"sector_id '{sector_id}' not found in dataset. Available: {sorted(available_sectors)}"
        raise ValueError(msg)

    # Validate factor columns
    if not factor_columns:
        msg = "factor_columns cannot be empty"
        raise ValueError(msg)

    missing_factors = set(factor_columns) - set(dataset.factor_returns.columns)
    if missing_factors:
        msg = f"factor_columns {sorted(missing_factors)} not found in dataset.factor_returns"
        raise ValueError(msg)
    _ensure_expected_factor_columns(factor_columns)

    # Validate break_date is within data range
    sector_data = dataset.sector_returns.filter(pl.col("symbol") == sector_id)

    if sector_data.is_empty():
        msg = f"No data found for sector {sector_id}"
        raise ValueError(msg)

    # Try to get min/max dates - handle different polars versions and dtypes gracefully
    try:
        # Attempt to use polars methods if timestamp is proper datetime type
        min_date_df = sector_data.select(pl.col("timestamp").min())
        max_date_df = sector_data.select(pl.col("timestamp").max())

        # Try to extract scalar values
        min_date = min_date_df.item(0, 0) if min_date_df.height > 0 else None
        max_date = max_date_df.item(0, 0) if max_date_df.height > 0 else None

        if min_date is not None and max_date is not None:
            if break_date <= min_date:
                msg = (
                    f"break_date ({break_date.isoformat()}) must be after data start "
                    f"({min_date.isoformat() if hasattr(min_date, 'isoformat') else str(min_date)})"
                )
                raise ValueError(msg)

            if break_date >= max_date:
                msg = (
                    f"break_date ({break_date.isoformat()}) must be before data end "
                    f"({max_date.isoformat() if hasattr(max_date, 'isoformat') else str(max_date)})"
                )
                raise ValueError(msg)
    except Exception:
        # If we can't validate date range (dtype=object timestamps), skip validation
        # The actual Chow test will fail later with better error if dates are truly invalid
        pass

    # Validate min_observations_per_period
    if min_observations_per_period < 1:
        msg = f"min_observations_per_period must be >= 1, got {min_observations_per_period}"
        raise ValueError(msg)


__all__ = [
    "ChowTestResult",
    "StructuralBreakSummary",
    "compute_chow_test",
    "compute_structural_break_analysis",
]


def _ensure_expected_factor_columns(factor_columns: Sequence[str]) -> None:
    """Ensure factor_columns matches the required order for the 3D model."""
    if tuple(factor_columns) != EXPECTED_FACTOR_COLUMNS:
        expected = ", ".join(EXPECTED_FACTOR_COLUMNS)
        msg = (
            "factor_columns must contain exactly the 3D risk model factors "
            f"in the order [{expected}], received {list(factor_columns)}"
        )
        raise ValueError(msg)
