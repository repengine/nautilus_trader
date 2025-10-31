"""
Factor attribution analysis for portfolio returns.

This module provides comprehensive factor attribution analysis to decompose
portfolio returns into contributions from systematic factors and alpha.

Key Features:
- Decomposes returns into factor contributions using OLS regression
- Tests statistical significance of alpha (skill-based returns)
- Validates attribution identity (sum = total return)
- Supports rolling windows and strategy comparison
- Generates detailed attribution reports

Mathematical Framework:
    R_portfolio,t = α + β_dur*R_dur,t + β_cred*R_cred,t + β_liq*R_liq,t + ε_t

Where:
    - α (alpha) = skill-based return (intercept)
    - β_i = exposure to factor i (regression coefficient)
    - R_i,t = return of factor i at time t
    - ε_t = residual (unexplained return)

The attribution satisfies the identity:
    R_portfolio,t = α + Σ(β_i * R_i,t) + ε_t

Performance Targets (Cold Path):
- Full attribution calculation: < 5 seconds for 10 years of monthly data
- Statistical tests: < 1 second
- Report generation: < 2 seconds

Hot/Cold Path Separation:
- This is a cold-path module (attribution is offline analysis)
- No real-time constraints, optimized for correctness over speed

Formula References:
- OLS Regression: Y = Xβ + ε, β = (X'X)^(-1)X'Y
- T-statistic: t = β / SE(β), where SE(β) = sqrt(σ²(X'X)^(-1))
- P-value: p = 2 * (1 - CDF(|t|, df=n-k))
- Alpha significance: |t| > 2.0 (approximately p < 0.05)

Integration Notes:
- Compatible with BacktestResult from engine.py
- Factor data from SectorDataset (dataset.py)
- Follows Phase 3.3 requirements from 3D_Risk_Model_Roadmap.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import polars as pl
import structlog
from scipy import stats


if TYPE_CHECKING:
    pass

# Import BacktestResult for runtime use
from playground.backtest.engine import BacktestResult


class RegressionResult(TypedDict):
    """Type definition for OLS regression results."""

    alpha: float
    beta_dur: float
    beta_cred: float
    beta_liq: float
    alpha_se: float
    residuals: np.ndarray


LOGGER = structlog.get_logger(__name__)


# ===== Constants =====

TRADING_DAYS_PER_YEAR = 252
TRADING_MONTHS_PER_YEAR = 12
EPSILON = 1e-10  # Numerical precision tolerance
ALPHA_SIGNIFICANCE_THRESHOLD = 2.0  # T-statistic threshold for significance
BASIS_POINT_TOLERANCE = 1e-4  # 1 basis point deviation tolerance


# ===== Data Classes =====


@dataclass(slots=True, frozen=True)
class FactorContribution:
    """
    Factor contribution to portfolio returns for a single period.

    This dataclass captures the decomposition of portfolio returns into
    factor-based contributions for a specific time period.

    Attributes
    ----------
    date : datetime
        Period date (timezone-aware UTC)
    duration_return : float
        Duration factor return for this period
    credit_return : float
        Credit factor return for this period
    liquidity_return : float
        Liquidity factor return for this period
    duration_beta : float
        Portfolio exposure to duration factor
    credit_beta : float
        Portfolio exposure to credit factor
    liquidity_beta : float
        Portfolio exposure to liquidity factor
    duration_contribution : float
        Duration contribution (beta × factor_return)
    credit_contribution : float
        Credit contribution (beta × factor_return)
    liquidity_contribution : float
        Liquidity contribution (beta × factor_return)
    alpha : float
        Skill-based return (intercept)
    residual : float
        Unexplained return (ε)
    portfolio_return : float
        Actual portfolio return for this period

    Notes
    -----
    The attribution identity is validated in __post_init__:
        portfolio_return = alpha + Σ(contributions) + residual

    This ensures mathematical correctness with numerical tolerance of 1e-6.
    """

    date: datetime

    # Factor returns (independent variables)
    duration_return: float
    credit_return: float
    liquidity_return: float

    # Portfolio factor exposures (betas)
    duration_beta: float
    credit_beta: float
    liquidity_beta: float

    # Contributions (beta × factor_return)
    duration_contribution: float
    credit_contribution: float
    liquidity_contribution: float

    # Alpha and residual
    alpha: float  # Intercept (skill-based return)
    residual: float  # Unexplained return (ε)

    # Actual portfolio return
    portfolio_return: float

    def __post_init__(self) -> None:
        """Validate that attribution sums to total return."""
        total_attribution = (
            self.alpha
            + self.duration_contribution
            + self.credit_contribution
            + self.liquidity_contribution
            + self.residual
        )

        # Allow small numerical error (1e-6)
        if not np.isclose(total_attribution, self.portfolio_return, atol=1e-6):
            msg = (
                f"Attribution doesn't sum to portfolio return: "
                f"{total_attribution:.6f} != {self.portfolio_return:.6f}"
            )
            raise ValueError(msg)


@dataclass(slots=True)
class AttributionResult:
    """
    Complete factor attribution analysis for a backtest.

    This dataclass contains the full attribution analysis including time series
    of contributions, aggregate statistics, and statistical significance tests.

    Attributes
    ----------
    strategy_name : str
        Name of the strategy being analyzed
    start_date : datetime
        Start date of analysis period (timezone-aware UTC)
    end_date : datetime
        End date of analysis period (timezone-aware UTC)
    contributions : list[FactorContribution]
        Time series of monthly factor contributions
    average_alpha : float
        Average alpha per period (annualized)
    alpha_t_stat : float
        T-statistic for alpha significance test
    alpha_p_value : float
        P-value for alpha significance test (two-tailed)
    avg_duration_contribution : float
        Average duration contribution (annualized %)
    avg_credit_contribution : float
        Average credit contribution (annualized %)
    avg_liquidity_contribution : float
        Average liquidity contribution (annualized %)
    avg_residual : float
        Average residual (annualized %)
    total_return : float
        Cumulative portfolio return
    total_alpha : float
        Cumulative alpha contribution
    total_duration : float
        Cumulative duration contribution
    total_credit : float
        Cumulative credit contribution
    total_liquidity : float
        Cumulative liquidity contribution
    total_residual : float
        Cumulative residual

    Notes
    -----
    Statistical significance is determined by:
        |t| > 2.0 → statistically significant at ~5% level
        |t| <= 2.0 → not statistically significant

    All annualized contributions are scaled by periods per year.
    """

    strategy_name: str
    start_date: datetime
    end_date: datetime

    # Time series of monthly contributions
    contributions: list[FactorContribution]

    # Aggregate statistics
    average_alpha: float
    alpha_t_stat: float  # Statistical significance
    alpha_p_value: float

    # Average factor contributions (annualized %)
    avg_duration_contribution: float
    avg_credit_contribution: float
    avg_liquidity_contribution: float
    avg_residual: float

    # Decomposition of total return
    total_return: float  # Cumulative portfolio return
    total_alpha: float  # Cumulative alpha contribution
    total_duration: float  # Cumulative duration contribution
    total_credit: float  # Cumulative credit contribution
    total_liquidity: float  # Cumulative liquidity contribution
    total_residual: float  # Cumulative residual

    def __post_init__(self) -> None:
        """Ensure aggregate contributions reconcile to total return within 1 bp."""
        cumulative = (
            self.total_alpha
            + self.total_duration
            + self.total_credit
            + self.total_liquidity
            + self.total_residual
        )
        deviation = abs(cumulative - self.total_return)
        if deviation > BASIS_POINT_TOLERANCE:
            msg = (
                "Attribution totals deviate from portfolio return by "
                f"{deviation:.6f}, exceeding 1 bp tolerance."
            )
            raise ValueError(msg)

    @property
    def is_alpha_significant(self) -> bool:
        """
        Check if alpha is statistically significant.

        Returns
        -------
        bool
            True if |t-stat| > 2.0 (approximately p < 0.05)
        """
        return abs(self.alpha_t_stat) > ALPHA_SIGNIFICANCE_THRESHOLD

    def summary_table(self) -> pl.DataFrame:
        """
        Generate summary table of factor contributions.

        Returns
        -------
        pl.DataFrame
            Summary table with columns:
            - component: Name of contribution component
            - average_contribution: Average contribution (annualized %)
            - total_contribution: Cumulative contribution
            - pct_of_total: Percentage of total return
        """
        if abs(self.total_return) < EPSILON:
            # Avoid division by zero
            pct_alpha = 0.0
            pct_duration = 0.0
            pct_credit = 0.0
            pct_liquidity = 0.0
            pct_residual = 0.0
        else:
            pct_alpha = (self.total_alpha / self.total_return) * 100
            pct_duration = (self.total_duration / self.total_return) * 100
            pct_credit = (self.total_credit / self.total_return) * 100
            pct_liquidity = (self.total_liquidity / self.total_return) * 100
            pct_residual = (self.total_residual / self.total_return) * 100

        return pl.DataFrame({
            "component": [
                "Alpha",
                "Duration Factor",
                "Credit Factor",
                "Liquidity Factor",
                "Residual",
            ],
            "average_contribution": [
                self.average_alpha,
                self.avg_duration_contribution,
                self.avg_credit_contribution,
                self.avg_liquidity_contribution,
                self.avg_residual,
            ],
            "total_contribution": [
                self.total_alpha,
                self.total_duration,
                self.total_credit,
                self.total_liquidity,
                self.total_residual,
            ],
            "pct_of_total": [
                pct_alpha,
                pct_duration,
                pct_credit,
                pct_liquidity,
                pct_residual,
            ],
        })

    def to_dict(self) -> dict[str, float | str]:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        dict[str, float | str]
            Dictionary with all attribution metrics
        """
        return {
            "strategy_name": self.strategy_name,
            "average_alpha": self.average_alpha,
            "alpha_t_stat": self.alpha_t_stat,
            "alpha_p_value": self.alpha_p_value,
            "is_alpha_significant": float(self.is_alpha_significant),
            "avg_duration_contribution": self.avg_duration_contribution,
            "avg_credit_contribution": self.avg_credit_contribution,
            "avg_liquidity_contribution": self.avg_liquidity_contribution,
            "avg_residual": self.avg_residual,
            "total_return": self.total_return,
            "total_alpha": self.total_alpha,
            "total_duration": self.total_duration,
            "total_credit": self.total_credit,
            "total_liquidity": self.total_liquidity,
            "total_residual": self.total_residual,
        }


# ===== Core Attribution Functions =====


def calculate_factor_attribution(
    backtest_result: BacktestResult,
    factor_returns: pl.DataFrame,
    portfolio_betas: pl.DataFrame | None = None,
    frequency: str = "monthly",
) -> AttributionResult:
    """
    Decompose portfolio returns into factor contributions.

    This function performs a factor attribution analysis by regressing
    portfolio returns on factor returns:

        R_portfolio,t = α + β_dur*R_dur,t + β_cred*R_cred,t + β_liq*R_liq,t + ε_t

    Where:
    - α (alpha) = skill-based return (intercept)
    - β_i = exposure to factor i
    - R_i,t = return of factor i at time t
    - ε_t = residual (unexplained return)

    Parameters
    ----------
    backtest_result : BacktestResult
        Backtest result with daily returns
    factor_returns : pl.DataFrame
        Factor returns with columns: timestamp, duration_return, credit_return, liquidity_return
    portfolio_betas : pl.DataFrame | None
        Time-varying portfolio betas (if None, estimate via regression)
        Columns: timestamp, duration_beta, credit_beta, liquidity_beta
    frequency : str
        Aggregation frequency ("daily", "weekly", "monthly")

    Returns
    -------
    AttributionResult
        Complete attribution analysis

    Raises
    ------
    ValueError
        If factor_returns is missing required columns or frequency is invalid

    Notes
    -----
    The attribution satisfies the identity:
        R_portfolio = α + Σ(β_i * R_i) + ε

    Statistical significance of alpha is tested using t-statistics:
        t = α / SE(α)

    where SE(α) is the standard error of the alpha estimate.

    Examples
    --------
    >>> attribution = calculate_factor_attribution(result, factor_data)
    >>> print(f"Alpha: {attribution.average_alpha:.2%}")
    >>> print(f"Significant: {attribution.is_alpha_significant}")
    """
    # Validate inputs
    required_factor_columns = {"timestamp", "duration_return", "credit_return", "liquidity_return"}
    if not required_factor_columns.issubset(factor_returns.columns):
        missing = required_factor_columns - set(factor_returns.columns)
        msg = f"Factor returns missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    if frequency not in {"daily", "weekly", "monthly"}:
        msg = f"Invalid frequency: {frequency}, must be 'daily', 'weekly', or 'monthly'"
        raise ValueError(msg)

    LOGGER.debug(
        "Calculating factor attribution",
        strategy=backtest_result.strategy_name,
        frequency=frequency,
        num_days=len(backtest_result.returns),
    )

    # Create portfolio returns DataFrame
    portfolio_df = pl.DataFrame({
        "timestamp": backtest_result.dates[1:],  # Skip first date (no return)
        "portfolio_return": backtest_result.returns,
    })

    # Aggregate to desired frequency
    portfolio_agg = _aggregate_returns(portfolio_df, frequency)
    factor_agg = _aggregate_factor_returns(factor_returns, frequency)

    # Merge portfolio and factor returns
    merged = portfolio_agg.join(factor_agg, on="timestamp", how="inner")

    if merged.is_empty():
        msg = "No overlapping timestamps between portfolio and factor returns"
        raise ValueError(msg)

    # Extract arrays for regression
    y = merged["portfolio_return"].to_numpy()
    X_dur = merged["duration_return"].to_numpy()
    X_cred = merged["credit_return"].to_numpy()
    X_liq = merged["liquidity_return"].to_numpy()

    # Perform OLS regression
    regression_result = _perform_ols_regression(
        y=y,
        X_dur=X_dur,
        X_cred=X_cred,
        X_liq=X_liq,
    )

    # Extract regression results (ensure float types)
    alpha = float(regression_result["alpha"])
    beta_dur = float(regression_result["beta_dur"])
    beta_cred = float(regression_result["beta_cred"])
    beta_liq = float(regression_result["beta_liq"])
    alpha_se = float(regression_result["alpha_se"])
    residuals_array = regression_result["residuals"]
    if not isinstance(residuals_array, np.ndarray):
        msg = "Expected residuals to be numpy array"
        raise TypeError(msg)

    # Calculate t-statistic and p-value
    if alpha_se > EPSILON:
        alpha_t_stat = alpha / alpha_se
        df = len(y) - 4  # Degrees of freedom: n - k (k=4: intercept + 3 betas)
        alpha_p_value = 2 * (1 - stats.t.cdf(abs(alpha_t_stat), df=df))
    else:
        alpha_t_stat = 0.0
        alpha_p_value = 1.0

    # Build contributions for each period
    contributions: list[FactorContribution] = []
    for i, row in enumerate(merged.iter_rows(named=True)):
        timestamp = row["timestamp"]
        port_ret = row["portfolio_return"]
        dur_ret = row["duration_return"]
        cred_ret = row["credit_return"]
        liq_ret = row["liquidity_return"]

        # Calculate contributions
        dur_contrib = beta_dur * dur_ret
        cred_contrib = beta_cred * cred_ret
        liq_contrib = beta_liq * liq_ret
        residual = float(residuals_array[i])

        contribution = FactorContribution(
            date=timestamp,
            duration_return=dur_ret,
            credit_return=cred_ret,
            liquidity_return=liq_ret,
            duration_beta=beta_dur,
            credit_beta=beta_cred,
            liquidity_beta=beta_liq,
            duration_contribution=dur_contrib,
            credit_contribution=cred_contrib,
            liquidity_contribution=liq_contrib,
            alpha=alpha,
            residual=residual,
            portfolio_return=port_ret,
        )
        contributions.append(contribution)

    # Calculate aggregate statistics
    n_periods = len(contributions)
    periods_per_year = _get_periods_per_year(frequency)

    # Average contributions (annualized)
    avg_alpha = alpha * periods_per_year
    avg_duration = float(np.mean([c.duration_contribution for c in contributions])) * periods_per_year
    avg_credit = float(np.mean([c.credit_contribution for c in contributions])) * periods_per_year
    avg_liquidity = float(np.mean([c.liquidity_contribution for c in contributions])) * periods_per_year
    avg_residual = float(np.mean([c.residual for c in contributions])) * periods_per_year

    # Total contributions (cumulative)
    total_return = backtest_result.total_return
    total_alpha = alpha * n_periods
    total_duration = sum(c.duration_contribution for c in contributions)
    total_credit = sum(c.credit_contribution for c in contributions)
    total_liquidity = sum(c.liquidity_contribution for c in contributions)
    total_residual = sum(c.residual for c in contributions)
    cumulative_total = total_alpha + total_duration + total_credit + total_liquidity + total_residual
    difference = total_return - cumulative_total
    if abs(difference) > BASIS_POINT_TOLERANCE:
        LOGGER.warning(
            "Attribution totals adjusted to match portfolio return",
            strategy=backtest_result.strategy_name,
            difference=difference,
        )
    total_residual += difference

    result = AttributionResult(
        strategy_name=backtest_result.strategy_name,
        start_date=backtest_result.start_date,
        end_date=backtest_result.end_date,
        contributions=contributions,
        average_alpha=avg_alpha,
        alpha_t_stat=alpha_t_stat,
        alpha_p_value=alpha_p_value,
        avg_duration_contribution=avg_duration,
        avg_credit_contribution=avg_credit,
        avg_liquidity_contribution=avg_liquidity,
        avg_residual=avg_residual,
        total_return=total_return,
        total_alpha=total_alpha,
        total_duration=total_duration,
        total_credit=total_credit,
        total_liquidity=total_liquidity,
        total_residual=total_residual,
    )

    LOGGER.info(
        "Attribution analysis complete",
        strategy=backtest_result.strategy_name,
        alpha_annualized=f"{avg_alpha:.2%}",
        alpha_t_stat=f"{alpha_t_stat:.2f}",
        is_significant=result.is_alpha_significant,
    )

    return result


def perform_rolling_attribution(
    backtest_result: BacktestResult,
    factor_returns: pl.DataFrame,
    window_months: int = 12,
) -> list[AttributionResult]:
    """
    Perform rolling factor attribution analysis.

    Useful for understanding how factor contributions change over time.

    Parameters
    ----------
    backtest_result : BacktestResult
        Full backtest result
    factor_returns : pl.DataFrame
        Factor returns over full period
    window_months : int
        Rolling window size in months (default 12)

    Returns
    -------
    list[AttributionResult]
        Attribution results for each rolling window

    Raises
    ------
    ValueError
        If window_months is less than 6 (minimum for statistical reliability)
    """
    if window_months < 6:
        msg = f"Window size must be at least 6 months for statistical reliability, got {window_months}"
        raise ValueError(msg)

    LOGGER.debug(
        "Performing rolling attribution",
        strategy=backtest_result.strategy_name,
        window_months=window_months,
    )

    # Create monthly aggregated data
    portfolio_df = pl.DataFrame({
        "timestamp": backtest_result.dates[1:],
        "portfolio_return": backtest_result.returns,
    })
    monthly_returns = _aggregate_returns(portfolio_df, "monthly")

    # Generate rolling windows
    rolling_results: list[AttributionResult] = []
    n_months = len(monthly_returns)

    for i in range(window_months, n_months + 1):
        window_start_idx = i - window_months
        window_end_idx = i

        # Extract window data
        window_dates = monthly_returns["timestamp"][window_start_idx:window_end_idx].to_list()
        window_returns = monthly_returns["portfolio_return"][window_start_idx:window_end_idx].to_list()

        # Create window backtest result
        window_result = BacktestResult(
            strategy_name=f"{backtest_result.strategy_name} (Rolling {window_months}m)",
            start_date=window_dates[0],
            end_date=window_dates[-1],
            dates=[window_dates[0]] + window_dates,  # Add initial date
            portfolio_values=[1.0] + [1.0],  # Placeholder
            returns=window_returns,
            positions=pl.DataFrame(),  # Not needed for attribution
            total_return=float(np.prod(1.0 + np.array(window_returns)) - 1.0),
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            total_transaction_costs=0.0,
            turnover_rate=0.0,
            num_rebalances=0,
        )

        # Calculate attribution for window
        window_attribution = calculate_factor_attribution(
            window_result,
            factor_returns,
            frequency="monthly",
        )
        rolling_results.append(window_attribution)

    LOGGER.info(
        "Rolling attribution complete",
        strategy=backtest_result.strategy_name,
        num_windows=len(rolling_results),
    )

    return rolling_results


def compare_attribution_across_strategies(
    attribution_results: dict[str, AttributionResult],
) -> pl.DataFrame:
    """
    Compare factor attribution across multiple strategies.

    Parameters
    ----------
    attribution_results : dict[str, AttributionResult]
        Mapping of strategy_name -> attribution result

    Returns
    -------
    pl.DataFrame
        Comparison table with columns:
        - strategy_name
        - average_alpha
        - alpha_t_stat
        - is_significant
        - duration_contribution
        - credit_contribution
        - liquidity_contribution
        - residual_contribution
    """
    comparison_data = []

    for strategy_name, attribution in attribution_results.items():
        comparison_data.append({
            "strategy_name": strategy_name,
            "average_alpha": attribution.average_alpha,
            "alpha_t_stat": attribution.alpha_t_stat,
            "alpha_p_value": attribution.alpha_p_value,
            "is_significant": attribution.is_alpha_significant,
            "duration_contribution": attribution.avg_duration_contribution,
            "credit_contribution": attribution.avg_credit_contribution,
            "liquidity_contribution": attribution.avg_liquidity_contribution,
            "residual_contribution": attribution.avg_residual,
            "total_return": attribution.total_return,
        })

    return pl.DataFrame(comparison_data).sort("average_alpha", descending=True)


def decompose_performance_by_factor(
    attribution: AttributionResult,
) -> dict[str, float]:
    """
    Calculate percentage of total return explained by each component.

    Parameters
    ----------
    attribution : AttributionResult
        Attribution analysis result

    Returns
    -------
    dict[str, float]
        Percentage of total return from each source:
        {
            "alpha": 15.2,  # 15.2% of return from alpha
            "duration": 45.3,  # 45.3% from duration factor
            "credit": 30.1,  # 30.1% from credit factor
            "liquidity": 5.4,  # 5.4% from liquidity factor
            "residual": 4.0,  # 4.0% from residual
        }
    """
    if abs(attribution.total_return) < EPSILON:
        # No return to decompose
        return {
            "alpha": 0.0,
            "duration": 0.0,
            "credit": 0.0,
            "liquidity": 0.0,
            "residual": 0.0,
        }

    return {
        "alpha": (attribution.total_alpha / attribution.total_return) * 100,
        "duration": (attribution.total_duration / attribution.total_return) * 100,
        "credit": (attribution.total_credit / attribution.total_return) * 100,
        "liquidity": (attribution.total_liquidity / attribution.total_return) * 100,
        "residual": (attribution.total_residual / attribution.total_return) * 100,
    }


def generate_attribution_report(
    attribution: AttributionResult,
    output_path: Path,
) -> None:
    """
    Generate markdown report with factor attribution analysis.

    Parameters
    ----------
    attribution : AttributionResult
        Attribution analysis
    output_path : Path
        Path to save markdown report
    """
    # Generate summary table
    summary = attribution.summary_table()

    # Generate decomposition
    decomposition = decompose_performance_by_factor(attribution)

    # Build markdown report
    report_lines = [
        f"# Factor Attribution Analysis: {attribution.strategy_name}",
        "",
        "## Executive Summary",
        "",
        f"- **Alpha (Annualized)**: {attribution.average_alpha:.2%} (t-stat: {attribution.alpha_t_stat:.2f}, p-value: {attribution.alpha_p_value:.3f})",
        f"- **Statistical Significance**: {'✅ Significant' if attribution.is_alpha_significant else '❌ Not Significant'}",
        "- **Total Return Decomposition**:",
        f"  - Alpha: {decomposition['alpha']:.1f}%",
        f"  - Duration Factor: {decomposition['duration']:.1f}%",
        f"  - Credit Factor: {decomposition['credit']:.1f}%",
        f"  - Liquidity Factor: {decomposition['liquidity']:.1f}%",
        f"  - Residual: {decomposition['residual']:.1f}%",
        "",
        "## Factor Contribution Summary",
        "",
        "| Component | Average (Annualized) | Total | % of Total Return |",
        "|-----------|---------------------|-------|-------------------|",
    ]

    for row in summary.iter_rows(named=True):
        report_lines.append(
            f"| {row['component']} | {row['average_contribution']:.2%} | "
            f"{row['total_contribution']:.2%} | {row['pct_of_total']:.1f}% |"
        )

    report_lines.extend([
        "",
        "## Statistical Analysis",
        "",
        "### Alpha Significance Test",
        "",
        "- **Null Hypothesis**: alpha = 0 (no skill-based return)",
        f"- **T-statistic**: {attribution.alpha_t_stat:.2f}",
        f"- **P-value**: {attribution.alpha_p_value:.3f}",
        f"- **Conclusion**: {'Reject' if attribution.is_alpha_significant else 'Fail to reject'} null hypothesis at 5% significance level",
        "",
        "## Interpretation",
        "",
        f"The strategy generated {attribution.average_alpha:.2%} annualized alpha, which is "
        f"{'statistically significant' if attribution.is_alpha_significant else 'not statistically significant'}.",
        "",
        f"- Duration factor contributed {decomposition['duration']:.1f}% of total return",
        f"- Credit factor contributed {decomposition['credit']:.1f}% of total return",
        f"- Liquidity factor contributed {decomposition['liquidity']:.1f}% of total return",
        f"- Residual is {abs(decomposition['residual']):.1f}%, suggesting the 3-factor model "
        f"{'explains returns well' if abs(decomposition['residual']) < 10 else 'leaves significant unexplained variation'}",
        "",
    ])

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")

    LOGGER.info(
        "Attribution report generated",
        strategy=attribution.strategy_name,
        output_path=str(output_path),
    )


# ===== Helper Functions =====


def _aggregate_returns(
    portfolio_df: pl.DataFrame,
    frequency: str,
) -> pl.DataFrame:
    """
    Aggregate portfolio returns to specified frequency.

    Parameters
    ----------
    portfolio_df : pl.DataFrame
        DataFrame with columns: timestamp, portfolio_return
    frequency : str
        Aggregation frequency ("daily", "weekly", "monthly")

    Returns
    -------
    pl.DataFrame
        Aggregated returns with columns: timestamp, portfolio_return
    """
    if frequency == "daily":
        return portfolio_df

    if frequency == "weekly":
        # Group by week and compound returns
        return (
            portfolio_df
            .with_columns(
                pl.col("timestamp").dt.truncate("1w").alias("week"),
            )
            .group_by("week")
            .agg(
                pl.col("timestamp").first().alias("timestamp"),
                ((pl.col("portfolio_return") + 1.0).product() - 1.0).alias("portfolio_return"),
            )
            .sort("timestamp")
        )

    if frequency == "monthly":
        # Group by month and compound returns
        return (
            portfolio_df
            .with_columns(
                pl.col("timestamp").dt.truncate("1mo").alias("month"),
            )
            .group_by("month")
            .agg(
                pl.col("timestamp").first().alias("timestamp"),
                ((pl.col("portfolio_return") + 1.0).product() - 1.0).alias("portfolio_return"),
            )
            .sort("timestamp")
        )

    msg = f"Invalid frequency: {frequency}"
    raise ValueError(msg)


def _aggregate_factor_returns(
    factor_df: pl.DataFrame,
    frequency: str,
) -> pl.DataFrame:
    """
    Aggregate factor returns to specified frequency.

    Parameters
    ----------
    factor_df : pl.DataFrame
        DataFrame with columns: timestamp, duration_return, credit_return, liquidity_return
    frequency : str
        Aggregation frequency ("daily", "weekly", "monthly")

    Returns
    -------
    pl.DataFrame
        Aggregated factor returns
    """
    if frequency == "daily":
        return factor_df

    # Determine grouping column
    if frequency == "weekly":
        group_col = pl.col("timestamp").dt.truncate("1w").alias("week")
        group_name = "week"
    elif frequency == "monthly":
        group_col = pl.col("timestamp").dt.truncate("1mo").alias("month")
        group_name = "month"
    else:
        msg = f"Invalid frequency: {frequency}"
        raise ValueError(msg)

    # Aggregate by compounding returns
    return (
        factor_df
        .with_columns(group_col)
        .group_by(group_name)
        .agg(
            pl.col("timestamp").first().alias("timestamp"),
            ((pl.col("duration_return") + 1.0).product() - 1.0).alias("duration_return"),
            ((pl.col("credit_return") + 1.0).product() - 1.0).alias("credit_return"),
            ((pl.col("liquidity_return") + 1.0).product() - 1.0).alias("liquidity_return"),
        )
        .sort("timestamp")
    )


def _perform_ols_regression(
    y: np.ndarray,
    X_dur: np.ndarray,
    X_cred: np.ndarray,
    X_liq: np.ndarray,
) -> RegressionResult:
    """
    Perform OLS regression: y = α + β_dur*X_dur + β_cred*X_cred + β_liq*X_liq + ε.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (portfolio returns)
    X_dur : np.ndarray
        Duration factor returns
    X_cred : np.ndarray
        Credit factor returns
    X_liq : np.ndarray
        Liquidity factor returns

    Returns
    -------
    RegressionResult
        Regression results:
        - alpha: Intercept
        - beta_dur: Duration coefficient
        - beta_cred: Credit coefficient
        - beta_liq: Liquidity coefficient
        - alpha_se: Standard error of alpha
        - residuals: Residual array

    Notes
    -----
    Uses numpy linear algebra to solve: β = (X'X)^(-1)X'Y
    """
    # Build design matrix [1, X_dur, X_cred, X_liq]
    n = len(y)
    X = np.column_stack([
        np.ones(n),
        X_dur,
        X_cred,
        X_liq,
    ])

    # Solve normal equations: (X'X)β = X'y
    XtX = X.T @ X
    Xty = X.T @ y

    # Check for singularity and use regularization if needed
    det = np.linalg.det(XtX)
    if abs(det) < EPSILON:
        # Matrix is singular or near-singular, use pseudo-inverse
        # This handles perfect multicollinearity gracefully
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ Xty
    else:
        # Normal case: solve directly
        beta = np.linalg.solve(XtX, Xty)

    # Extract coefficients
    alpha = float(beta[0])
    beta_dur = float(beta[1])
    beta_cred = float(beta[2])
    beta_liq = float(beta[3])

    # Calculate residuals
    y_pred = X @ beta
    residuals = y - y_pred

    # Calculate standard errors
    # SE(β) = sqrt(σ²(X'X)^(-1))
    # σ² = RSS / (n - k)
    rss = np.sum(residuals**2)
    k = X.shape[1]  # Number of parameters (4)
    df = n - k

    if df > 0:
        sigma_squared = rss / df
        # Use pseudo-inverse if matrix was singular
        if abs(det) < EPSILON:
            cov_matrix = sigma_squared * np.linalg.pinv(XtX)
        else:
            cov_matrix = sigma_squared * np.linalg.inv(XtX)
        alpha_se = float(np.sqrt(cov_matrix[0, 0]))
    else:
        alpha_se = 0.0

    return {
        "alpha": alpha,
        "beta_dur": beta_dur,
        "beta_cred": beta_cred,
        "beta_liq": beta_liq,
        "alpha_se": alpha_se,
        "residuals": residuals,
    }


def _get_periods_per_year(frequency: str) -> int:
    """
    Get number of periods per year for a given frequency.

    Parameters
    ----------
    frequency : str
        Aggregation frequency ("daily", "weekly", "monthly")

    Returns
    -------
    int
        Number of periods per year
    """
    if frequency == "daily":
        return TRADING_DAYS_PER_YEAR
    if frequency == "weekly":
        return 52
    if frequency == "monthly":
        return TRADING_MONTHS_PER_YEAR
    msg = f"Invalid frequency: {frequency}"
    raise ValueError(msg)


# ===== Public API =====

__all__ = [
    "AttributionResult",
    "FactorContribution",
    "calculate_factor_attribution",
    "compare_attribution_across_strategies",
    "decompose_performance_by_factor",
    "generate_attribution_report",
    "perform_rolling_attribution",
]
