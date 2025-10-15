"""
Regression diagnostics for the 3D factor risk model.

This module provides comprehensive diagnostic metrics to validate the explanatory
power and statistical significance of the 3-factor model (Duration, Credit, Liquidity)
applied to sector ETF returns.

Key capabilities:
- OLS regression diagnostics (R², F-stat, t-stats, p-values)
- Multicollinearity detection (VIF)
- Heteroskedasticity testing (Breusch-Pagan)
- Autocorrelation detection (Durbin-Watson)
- Residual analysis (normality, skewness, kurtosis)
- Acceptance criteria validation

Performance: Cold path only (training/validation, not real-time inference)
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
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


if TYPE_CHECKING:
    import statsmodels.api as sm
else:
    sm = None

# Import statsmodels only when needed to avoid heavy import
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


@dataclass(slots=True)
class RegressionDiagnostics:
    """
    Diagnostic metrics for a single sector's factor regression.

    Attributes
    ----------
    sector_id : str
        Sector ETF ticker (e.g., "XLK", "XLU").
    r_squared : float
        Coefficient of determination (0-1).
    adj_r_squared : float
        Adjusted R² accounting for number of predictors.
    f_statistic : float
        Overall model significance test statistic.
    f_pvalue : float
        P-value for F-statistic.
    durbin_watson : float
        Autocorrelation test statistic (1.5-2.5 ideal).
    beta_duration : float
        Coefficient for duration factor.
    beta_credit : float
        Coefficient for credit factor.
    beta_liquidity : float
        Coefficient for liquidity factor.
    alpha : float
        Regression intercept.
    t_stat_duration : float
        T-statistic for duration beta.
    t_stat_credit : float
        T-statistic for credit beta.
    t_stat_liquidity : float
        T-statistic for liquidity beta.
    t_stat_alpha : float
        T-statistic for intercept.
    p_value_duration : float
        P-value for duration beta.
    p_value_credit : float
        P-value for credit beta.
    p_value_liquidity : float
        P-value for liquidity beta.
    p_value_alpha : float
        P-value for intercept.
    se_duration : float
        Standard error for duration beta.
    se_credit : float
        Standard error for credit beta.
    se_liquidity : float
        Standard error for liquidity beta.
    se_alpha : float
        Standard error for intercept.
    vif_duration : float
        Variance Inflation Factor for duration (>5 indicates multicollinearity).
    vif_credit : float
        VIF for credit.
    vif_liquidity : float
        VIF for liquidity.
    bp_test_statistic : float
        Breusch-Pagan test statistic for heteroskedasticity.
    bp_p_value : float
        P-value for BP test (p < 0.05 indicates heteroskedasticity).
    residual_mean : float
        Mean of residuals (should be ~0).
    residual_std : float
        Standard deviation of residuals.
    residual_skewness : float
        Skewness of residuals (0 = symmetric).
    residual_kurtosis : float
        Excess kurtosis of residuals (0 = normal).
    n_observations : int
        Number of observations used in regression.
    date_range_start : datetime
        Start date of data.
    date_range_end : datetime
        End date of data.
    """

    sector_id: str
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    durbin_watson: float

    # Beta coefficients
    beta_duration: float
    beta_credit: float
    beta_liquidity: float
    alpha: float

    # T-statistics
    t_stat_duration: float
    t_stat_credit: float
    t_stat_liquidity: float
    t_stat_alpha: float

    # P-values
    p_value_duration: float
    p_value_credit: float
    p_value_liquidity: float
    p_value_alpha: float

    # Standard errors
    se_duration: float
    se_credit: float
    se_liquidity: float
    se_alpha: float

    # Multicollinearity (VIF)
    vif_duration: float
    vif_credit: float
    vif_liquidity: float

    # Heteroskedasticity test
    bp_test_statistic: float
    bp_p_value: float

    # Residual diagnostics
    residual_mean: float
    residual_std: float
    residual_skewness: float
    residual_kurtosis: float

    # Sample info
    n_observations: int
    date_range_start: datetime
    date_range_end: datetime


@dataclass(slots=True)
class SectorDiagnosticsReport:
    """
    Aggregate diagnostics across all sectors.

    Attributes
    ----------
    diagnostics : dict[str, RegressionDiagnostics]
        Diagnostics keyed by sector ID.
    summary_stats : dict[str, float]
        Aggregate statistics (mean R², % significant betas, etc.).
    acceptance_status : dict[str, bool]
        Pass/fail status for acceptance criteria.
    """

    diagnostics: dict[str, RegressionDiagnostics]
    summary_stats: dict[str, float]
    acceptance_status: dict[str, bool]


def compute_regression_diagnostics(
    sector_returns: pl.DataFrame,
    factor_returns: pl.DataFrame,
    *,
    factor_columns: Sequence[str],
) -> dict[str, RegressionDiagnostics]:
    """
    Compute full regression diagnostics for each sector.

    For each sector, runs OLS regression:
        R_sector = α + β_dur*ΔDuration + β_cred*ΔCredit + β_liq*ΔLiquidity + ε

    And computes:
    - Goodness of fit: R², adjusted R², F-statistic
    - Coefficient significance: t-statistics, p-values, standard errors
    - Multicollinearity: VIF for each factor
    - Heteroskedasticity: Breusch-Pagan test
    - Autocorrelation: Durbin-Watson statistic
    - Residual properties: mean, std, skewness, kurtosis

    Parameters
    ----------
    sector_returns : pl.DataFrame
        Sector returns with columns: timestamp, symbol, return.
    factor_returns : pl.DataFrame
        Factor returns with columns: timestamp, factor_duration, factor_credit, factor_liquidity.
    factor_columns : Sequence[str]
        Factor column names (e.g., ["factor_duration", "factor_credit", "factor_liquidity"]).

    Returns
    -------
    dict[str, RegressionDiagnostics]
        Diagnostics keyed by sector ID.

    Raises
    ------
    ValueError
        If required columns are missing or if no valid sectors found.

    Notes
    -----
    - Minimum 10 observations required per sector for reliable diagnostics
    - VIF > 5 indicates potential multicollinearity issues
    - Durbin-Watson values outside [1.5, 2.5] indicate autocorrelation
    - BP p-value < 0.05 indicates heteroskedasticity
    """
    LOGGER.info(
        "Computing regression diagnostics",
        n_sectors=sector_returns["symbol"].n_unique(),
        n_observations=len(sector_returns),
        factors=list(factor_columns),
    )

    # Validate inputs
    _validate_diagnostics_inputs(sector_returns, factor_returns, factor_columns)

    # Join sector and factor data
    joined = sector_returns.join(factor_returns, on="timestamp", how="inner").sort("timestamp")

    if joined.is_empty():
        msg = "No data after joining sector and factor returns"
        raise ValueError(msg)

    diagnostics: dict[str, RegressionDiagnostics] = {}
    sm = _get_sm()

    for sector in joined["symbol"].unique().to_list():
        sector_data = joined.filter(pl.col("symbol") == sector)

        if sector_data.height < 10:
            LOGGER.warning(
                "Insufficient observations for sector",
                sector=sector,
                n_obs=sector_data.height,
            )
            continue

        try:
            sector_diagnostics = _compute_sector_diagnostics(
            sector_data,
            sector,
            factor_columns,
            sm,
        )
            diagnostics[sector] = sector_diagnostics

            LOGGER.info(
                "Computed diagnostics for sector",
                sector=sector,
                r_squared=f"{sector_diagnostics.r_squared:.4f}",
                n_significant_betas=sum(
                    [
                        sector_diagnostics.p_value_duration < 0.05,
                        sector_diagnostics.p_value_credit < 0.05,
                        sector_diagnostics.p_value_liquidity < 0.05,
                    ]
                ),
            )
        except Exception:
            LOGGER.exception("Failed to compute diagnostics for sector", sector=sector)
            continue

    if not diagnostics:
        msg = "No valid diagnostics computed for any sector"
        raise ValueError(msg)

    LOGGER.info(
        "Completed regression diagnostics",
        n_sectors=len(diagnostics),
        mean_r_squared=np.mean([d.r_squared for d in diagnostics.values()]),
    )

    return diagnostics


def _validate_diagnostics_inputs(
    sector_returns: pl.DataFrame,
    factor_returns: pl.DataFrame,
    factor_columns: Sequence[str],
) -> None:
    """Validate inputs for regression diagnostics computation."""
    # Check sector returns columns
    required_sector_cols = {"timestamp", "symbol", "return"}
    missing = required_sector_cols - set(sector_returns.columns)
    if missing:
        msg = f"sector_returns missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    # Check factor returns columns
    required_factor_cols = {"timestamp"} | set(factor_columns)
    missing_factors = required_factor_cols - set(factor_returns.columns)
    if missing_factors:
        msg = f"factor_returns missing required columns: {sorted(missing_factors)}"
        raise ValueError(msg)

    # Check for empty DataFrames
    if sector_returns.is_empty():
        msg = "sector_returns cannot be empty"
        raise ValueError(msg)

    if factor_returns.is_empty():
        msg = "factor_returns cannot be empty"
        raise ValueError(msg)

    # Validate factor columns
    if not factor_columns:
        msg = "factor_columns cannot be empty"
        raise ValueError(msg)

    _ensure_expected_factor_columns(factor_columns)


def _compute_sector_diagnostics(
    sector_data: pl.DataFrame,
    sector_id: str,
    factor_columns: Sequence[str],
    sm_module: object,
) -> RegressionDiagnostics:
    """Compute diagnostics for a single sector."""
    # Prepare regression data
    y = sector_data["return"].to_numpy()
    X = sector_data.select(factor_columns).to_numpy()
    X_with_const = sm_module.add_constant(X)  # type: ignore[attr-defined]

    # Run OLS regression
    model = sm_module.OLS(y, X_with_const).fit()  # type: ignore[attr-defined]

    # Extract basic metrics
    r_squared = float(model.rsquared)
    adj_r_squared = float(model.rsquared_adj)
    f_statistic = float(model.fvalue)
    f_pvalue = float(model.f_pvalue)

    # Extract coefficients (intercept + 3 factors)
    params = model.params
    alpha = float(params[0])
    beta_duration = float(params[1])
    beta_credit = float(params[2])
    beta_liquidity = float(params[3])

    # T-statistics
    t_stats = model.tvalues
    t_stat_alpha = float(t_stats[0])
    t_stat_duration = float(t_stats[1])
    t_stat_credit = float(t_stats[2])
    t_stat_liquidity = float(t_stats[3])

    # P-values
    p_values = model.pvalues
    p_value_alpha = float(p_values[0])
    p_value_duration = float(p_values[1])
    p_value_credit = float(p_values[2])
    p_value_liquidity = float(p_values[3])

    # Standard errors
    se = model.bse
    se_alpha = float(se[0])
    se_duration = float(se[1])
    se_credit = float(se[2])
    se_liquidity = float(se[3])

    # VIF (multicollinearity) - skip constant column
    vif_duration = float(variance_inflation_factor(X_with_const, 1))
    vif_credit = float(variance_inflation_factor(X_with_const, 2))
    vif_liquidity = float(variance_inflation_factor(X_with_const, 3))

    # Breusch-Pagan test (heteroskedasticity)
    bp_test = het_breuschpagan(model.resid, X_with_const)
    bp_test_statistic = float(bp_test[0])
    bp_p_value = float(bp_test[1])

    # Durbin-Watson (autocorrelation)
    dw = float(durbin_watson(model.resid))

    # Residual diagnostics
    residuals = model.resid
    residual_mean = float(np.mean(residuals))
    residual_std = float(np.std(residuals))
    residual_skewness = float(stats.skew(residuals))
    residual_kurtosis = float(stats.kurtosis(residuals))

    # Date range
    timestamps = sector_data["timestamp"].to_list()
    date_range_start = min(timestamps)
    date_range_end = max(timestamps)

    return RegressionDiagnostics(
        sector_id=sector_id,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        f_statistic=f_statistic,
        f_pvalue=f_pvalue,
        durbin_watson=dw,
        beta_duration=beta_duration,
        beta_credit=beta_credit,
        beta_liquidity=beta_liquidity,
        alpha=alpha,
        t_stat_duration=t_stat_duration,
        t_stat_credit=t_stat_credit,
        t_stat_liquidity=t_stat_liquidity,
        t_stat_alpha=t_stat_alpha,
        p_value_duration=p_value_duration,
        p_value_credit=p_value_credit,
        p_value_liquidity=p_value_liquidity,
        p_value_alpha=p_value_alpha,
        se_duration=se_duration,
        se_credit=se_credit,
        se_liquidity=se_liquidity,
        se_alpha=se_alpha,
        vif_duration=vif_duration,
        vif_credit=vif_credit,
        vif_liquidity=vif_liquidity,
        bp_test_statistic=bp_test_statistic,
        bp_p_value=bp_p_value,
        residual_mean=residual_mean,
        residual_std=residual_std,
        residual_skewness=residual_skewness,
        residual_kurtosis=residual_kurtosis,
        n_observations=len(residuals),
        date_range_start=date_range_start,
        date_range_end=date_range_end,
    )


def create_diagnostics_summary(
    diagnostics: dict[str, RegressionDiagnostics],
    *,
    r2_threshold: float = 0.30,
    p_value_threshold: float = 0.05,
    vif_threshold: float = 5.0,
    dw_lower: float = 1.5,
    dw_upper: float = 2.5,
) -> SectorDiagnosticsReport:
    """
    Create summary report across all sectors.

    Parameters
    ----------
    diagnostics : dict[str, RegressionDiagnostics]
        Diagnostics for each sector.
    r2_threshold : float
        Minimum R² threshold for acceptance (default: 0.30).
    p_value_threshold : float
        Maximum p-value for beta significance (default: 0.05).
    vif_threshold : float
        Maximum VIF for multicollinearity detection (default: 5.0).
    dw_lower : float
        Lower bound for acceptable Durbin-Watson statistic (default: 1.5).
    dw_upper : float
        Upper bound for acceptable Durbin-Watson statistic (default: 2.5).

    Returns
    -------
    SectorDiagnosticsReport
        Summary report with aggregate statistics and acceptance status.

    Notes
    -----
    Acceptance criteria:
    - At least 70% of sectors have R² > r2_threshold
    - At least 70% of sectors have 2/3 significant betas
    - All factors have VIF < vif_threshold
    - At least 70% of sectors have DW in [dw_lower, dw_upper]
    """
    LOGGER.info(
        "Creating diagnostics summary",
        n_sectors=len(diagnostics),
        r2_threshold=r2_threshold,
        p_value_threshold=p_value_threshold,
    )

    if not diagnostics:
        msg = "Cannot create summary from empty diagnostics"
        raise ValueError(msg)

    # R² statistics
    r2_values = [d.r_squared for d in diagnostics.values()]
    mean_r2 = float(np.mean(r2_values))
    median_r2 = float(np.median(r2_values))
    std_r2 = float(np.std(r2_values))
    min_r2 = float(np.min(r2_values))
    max_r2 = float(np.max(r2_values))
    pct_above_r2_threshold = float(sum(r2 > r2_threshold for r2 in r2_values) / len(r2_values) * 100)

    # Significant betas (2/3 rule)
    sectors_with_2_3_sig_betas = 0
    for diag in diagnostics.values():
        sig_count = sum(
            [
                diag.p_value_duration < p_value_threshold,
                diag.p_value_credit < p_value_threshold,
                diag.p_value_liquidity < p_value_threshold,
            ]
        )
        if sig_count >= 2:
            sectors_with_2_3_sig_betas += 1

    pct_2_3_sig_betas = float(sectors_with_2_3_sig_betas / len(diagnostics) * 100)

    # VIF statistics
    all_vifs: list[float] = []
    for diag in diagnostics.values():
        all_vifs.extend([diag.vif_duration, diag.vif_credit, diag.vif_liquidity])

    mean_vif = float(np.mean(all_vifs))
    max_vif = float(np.max(all_vifs))

    # Durbin-Watson statistics
    dw_values = [d.durbin_watson for d in diagnostics.values()]
    dw_in_range = sum(dw_lower <= dw <= dw_upper for dw in dw_values)
    pct_dw_acceptable = float(dw_in_range / len(dw_values) * 100)
    mean_dw = float(np.mean(dw_values))

    # F-statistic
    f_pvalues = [d.f_pvalue for d in diagnostics.values()]
    pct_significant_f = float(sum(p < 0.05 for p in f_pvalues) / len(f_pvalues) * 100)

    summary_stats = {
        "mean_r_squared": mean_r2,
        "median_r_squared": median_r2,
        "std_r_squared": std_r2,
        "min_r_squared": min_r2,
        "max_r_squared": max_r2,
        "pct_r2_above_threshold": pct_above_r2_threshold,
        "pct_2_3_sig_betas": pct_2_3_sig_betas,
        "mean_vif": mean_vif,
        "max_vif": max_vif,
        "mean_durbin_watson": mean_dw,
        "pct_dw_acceptable": pct_dw_acceptable,
        "pct_significant_f": pct_significant_f,
        "n_sectors": float(len(diagnostics)),
    }

    # Acceptance status
    acceptance_status = {
        "r2_criterion": pct_above_r2_threshold >= 70.0,
        "significant_betas": pct_2_3_sig_betas >= 70.0,
        "multicollinearity": max_vif < vif_threshold,
        "autocorrelation": pct_dw_acceptable >= 70.0,
    }

    overall_pass = all(acceptance_status.values())
    acceptance_status["overall"] = overall_pass

    LOGGER.info(
        "Diagnostics summary created",
        mean_r2=f"{mean_r2:.4f}",
        pct_above_threshold=f"{pct_above_r2_threshold:.1f}%",
        pct_sig_betas=f"{pct_2_3_sig_betas:.1f}%",
        max_vif=f"{max_vif:.2f}",
        overall_pass=overall_pass,
    )

    return SectorDiagnosticsReport(
        diagnostics=diagnostics,
        summary_stats=summary_stats,
        acceptance_status=acceptance_status,
    )


__all__ = [
    "RegressionDiagnostics",
    "SectorDiagnosticsReport",
    "compute_regression_diagnostics",
    "create_diagnostics_summary",
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
