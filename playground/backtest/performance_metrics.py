"""
Performance metric calculations for backtesting results.

This module provides comprehensive performance metrics for evaluating portfolio
strategies in the 3D Factor Risk Model backtesting framework. All metrics follow
academic and industry-standard formulas to ensure comparability with external
benchmarks.

Key Features:
- Return metrics (annualized, cumulative, monthly distribution)
- Risk metrics (volatility, drawdown, VaR, CVaR)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar, Information Ratio)
- Trade metrics (turnover, transaction costs)
- Train/test period separation
- Mathematically rigorous implementations

Performance Targets (Cold Path):
- Metric calculation: < 1 second for full backtest
- Vectorized NumPy operations for efficiency
- No performance-critical constraints (offline analysis)

Hot/Cold Path Separation:
- This is a cold-path module (performance analysis is offline)
- No real-time constraints, optimized for correctness over speed

Formula References:
- Sharpe Ratio: (mean_return - risk_free) / std_return * sqrt(252)
- Sortino Ratio: (mean_return - risk_free) / downside_std * sqrt(252)
- Calmar Ratio: annualized_return / abs(max_drawdown)
- Information Ratio: mean(excess_return) / std(excess_return) * sqrt(252)
- VaR(α): α-quantile of return distribution
- CVaR(α): mean of returns below VaR threshold

Integration Notes:
- Compatible with BacktestResult from engine.py
- All dates are timezone-aware (UTC)
- Metrics validated against hand-calculated examples
- Follows Phase 3.2.2 requirements from 3D_Risk_Model_Roadmap.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import structlog


if TYPE_CHECKING:
    from playground.backtest.engine import BacktestResult


LOGGER = structlog.get_logger(__name__)


# ===== Constants =====

TRADING_DAYS_PER_YEAR = 252
TRADING_MONTHS_PER_YEAR = 12
EPSILON = 1e-10  # Small value to prevent division by zero


# ===== Performance Metrics Dataclass =====


@dataclass(slots=True)
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a backtest.

    This dataclass contains all key performance indicators for evaluating
    a portfolio strategy, organized into return, risk, risk-adjusted, and
    trade metric categories.

    Attributes
    ----------
    annualized_return : float
        Geometric mean return annualized (compounded)
    cumulative_return : float
        Total return over the full period
    monthly_return_mean : float
        Arithmetic mean of monthly returns
    monthly_return_std : float
        Standard deviation of monthly returns
    annualized_volatility : float
        Standard deviation of daily returns, annualized
    maximum_drawdown : float
        Largest peak-to-trough decline (negative value)
    var_95 : float
        Value at Risk at 95% confidence (5th percentile)
    var_99 : float
        Value at Risk at 99% confidence (1st percentile)
    cvar_95 : float
        Conditional VaR (Expected Shortfall) at 95%
    cvar_99 : float
        Conditional VaR (Expected Shortfall) at 99%
    sharpe_ratio : float
        Risk-adjusted return (excess return / volatility)
    sortino_ratio : float
        Downside risk-adjusted return (excess return / downside deviation)
    calmar_ratio : float
        Return-to-drawdown ratio (annualized return / abs(max drawdown))
    information_ratio : float | None
        Excess return vs benchmark per unit of tracking error (None if no benchmark)
    turnover_rate : float
        Average monthly portfolio turnover (sum of absolute weight changes / 2)
    transaction_costs_total : float
        Total transaction costs in dollars
    transaction_costs_pct : float
        Transaction costs as percentage of total returns
    num_rebalances : int
        Number of rebalancing events
    start_date : datetime
        Start date of the period (timezone-aware UTC)
    end_date : datetime
        End date of the period (timezone-aware UTC)
    total_days : int
        Total calendar days in the period

    Notes
    -----
    - All ratios use risk-free rate specified in calculation
    - Negative Sharpe/Sortino/Calmar indicate underperformance vs risk-free
    - VaR and CVaR are negative for losses (standard convention)
    - Turnover computed as half of sum of absolute weight changes
    """

    # Return metrics
    annualized_return: float
    cumulative_return: float
    monthly_return_mean: float
    monthly_return_std: float

    # Risk metrics
    annualized_volatility: float
    maximum_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float | None

    # Trade metrics
    turnover_rate: float
    transaction_costs_total: float
    transaction_costs_pct: float
    num_rebalances: int

    # Period metadata
    start_date: datetime
    end_date: datetime
    total_days: int


# ===== Core Metric Calculation Functions =====


def calculate_performance_metrics(
    result: BacktestResult,
    benchmark_result: BacktestResult | None = None,
    risk_free_rate: float = 0.02,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from backtest result.

    This function computes all key performance indicators including return,
    risk, risk-adjusted, and trade metrics using industry-standard formulas.

    Parameters
    ----------
    result : BacktestResult
        Backtest result to analyze (from FactorBacktester.run_backtest)
    benchmark_result : BacktestResult | None
        Optional benchmark for information ratio calculation
        If provided, computes tracking error and information ratio vs benchmark
    risk_free_rate : float, default 0.02
        Annual risk-free rate for Sharpe/Sortino calculations (default 2%)

    Returns
    -------
    PerformanceMetrics
        Comprehensive performance metrics for the strategy

    Raises
    ------
    ValueError
        If result.returns is empty or risk_free_rate is negative

    Examples
    --------
    >>> metrics = calculate_performance_metrics(result, risk_free_rate=0.02)
    >>> print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    Sharpe Ratio: 0.85
    >>> print(f"Max Drawdown: {metrics.maximum_drawdown:.2%}")
    Max Drawdown: -15.32%

    Notes
    -----
    Algorithm:
    1. Extract daily returns from BacktestResult
    2. Compute return metrics (annualized, cumulative, monthly)
    3. Compute risk metrics (volatility, drawdown, VaR, CVaR)
    4. Compute risk-adjusted ratios (Sharpe, Sortino, Calmar)
    5. Compute information ratio if benchmark provided
    6. Extract trade metrics from BacktestResult
    7. Package into PerformanceMetrics dataclass

    All metrics use vectorized NumPy operations for efficiency.
    """
    if risk_free_rate < 0:
        msg = f"Risk-free rate must be non-negative, got {risk_free_rate}"
        raise ValueError(msg)

    if not result.returns:
        msg = "Cannot calculate metrics for empty returns"
        raise ValueError(msg)

    LOGGER.debug(
        "Calculating performance metrics",
        strategy=result.strategy_name,
        num_returns=len(result.returns),
        start_date=result.start_date.isoformat(),
        end_date=result.end_date.isoformat(),
    )

    # Convert to numpy array for vectorized operations
    returns_arr = np.array(result.returns, dtype=float)

    # === Return Metrics ===
    cumulative_return = _calculate_cumulative_return(returns_arr)
    annualized_return = _calculate_annualized_return(returns_arr, len(returns_arr))
    monthly_return_mean, monthly_return_std = _calculate_monthly_statistics(result.dates, returns_arr)

    # === Risk Metrics ===
    annualized_volatility = _calculate_annualized_volatility(returns_arr)
    maximum_drawdown = _calculate_maximum_drawdown(returns_arr)
    var_95, cvar_95 = _calculate_var_cvar(returns_arr, confidence_level=0.95)
    var_99, cvar_99 = _calculate_var_cvar(returns_arr, confidence_level=0.99)

    # === Risk-Adjusted Metrics ===
    sharpe_ratio = _calculate_sharpe_ratio(returns_arr, risk_free_rate)
    sortino_ratio = _calculate_sortino_ratio(returns_arr, risk_free_rate)
    calmar_ratio = _calculate_calmar_ratio(annualized_return, maximum_drawdown)

    # === Information Ratio (vs Benchmark) ===
    information_ratio = None
    if benchmark_result is not None:
        information_ratio = _calculate_information_ratio(result, benchmark_result)

    # === Trade Metrics ===
    # Extract from BacktestResult (already computed in engine)
    transaction_costs_total = result.total_transaction_costs
    num_rebalances = result.num_rebalances
    turnover_rate = result.turnover_rate

    # Transaction costs as percentage of returns
    if abs(cumulative_return) > EPSILON:
        transaction_costs_pct = transaction_costs_total / (result.portfolio_values[0] * abs(cumulative_return))
    else:
        transaction_costs_pct = 0.0

    # === Period Metadata ===
    total_days = (result.end_date - result.start_date).days + 1

    metrics = PerformanceMetrics(
        annualized_return=annualized_return,
        cumulative_return=cumulative_return,
        monthly_return_mean=monthly_return_mean,
        monthly_return_std=monthly_return_std,
        annualized_volatility=annualized_volatility,
        maximum_drawdown=maximum_drawdown,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        information_ratio=information_ratio,
        turnover_rate=turnover_rate,
        transaction_costs_total=transaction_costs_total,
        transaction_costs_pct=transaction_costs_pct,
        num_rebalances=num_rebalances,
        start_date=result.start_date,
        end_date=result.end_date,
        total_days=total_days,
    )

    LOGGER.info(
        "Performance metrics calculated",
        strategy=result.strategy_name,
        sharpe_ratio=f"{sharpe_ratio:.2f}",
        annualized_return=f"{annualized_return:.2%}",
        max_drawdown=f"{maximum_drawdown:.2%}",
    )

    return metrics


# ===== Return Metrics =====


def _calculate_cumulative_return(returns: np.ndarray) -> float:
    """
    Calculate cumulative return over the period.

    Formula: R_total = prod(1 + r_i) - 1

    Parameters
    ----------
    returns : np.ndarray
        Daily returns array

    Returns
    -------
    float
        Cumulative return (geometric)
    """
    if len(returns) == 0:
        return 0.0

    cumulative = np.prod(1.0 + returns) - 1.0
    return float(cumulative)


def _calculate_annualized_return(returns: np.ndarray, num_days: int) -> float:
    """
    Calculate annualized return using geometric compounding.

    Formula: R_annual = (1 + R_total)^(252/N) - 1

    Parameters
    ----------
    returns : np.ndarray
        Daily returns array
    num_days : int
        Number of trading days in the period

    Returns
    -------
    float
        Annualized return
    """
    if len(returns) == 0 or num_days <= 0:
        return 0.0

    cumulative = _calculate_cumulative_return(returns)
    n_years = num_days / TRADING_DAYS_PER_YEAR

    if n_years <= 0:
        return 0.0

    annualized = (1.0 + cumulative) ** (1.0 / n_years) - 1.0
    return float(annualized)


def _calculate_monthly_statistics(dates: list[datetime], returns: np.ndarray) -> tuple[float, float]:
    """
    Calculate monthly return statistics.

    Aggregates daily returns into monthly returns and computes mean and std.

    Parameters
    ----------
    dates : list[datetime]
        Trading dates corresponding to returns
    returns : np.ndarray
        Daily returns array

    Returns
    -------
    tuple[float, float]
        (monthly_return_mean, monthly_return_std)

    Notes
    -----
    Monthly returns are computed as geometric compounding within each month.
    """
    if len(returns) == 0 or len(dates) <= 1:
        return 0.0, 0.0

    # Group returns by month
    monthly_returns: dict[tuple[int, int], list[float]] = {}
    for date, ret in zip(dates[1:], returns):  # Skip first date (no return)
        month_key = (date.year, date.month)
        if month_key not in monthly_returns:
            monthly_returns[month_key] = []
        monthly_returns[month_key].append(ret)

    if not monthly_returns:
        return 0.0, 0.0

    # Compute geometric monthly returns
    monthly_returns_list = [
        float(np.prod(np.array(daily_rets) + 1.0) - 1.0)
        for daily_rets in monthly_returns.values()
    ]

    if len(monthly_returns_list) < 2:
        return float(monthly_returns_list[0]) if monthly_returns_list else 0.0, 0.0

    mean = float(np.mean(monthly_returns_list))
    std = float(np.std(monthly_returns_list, ddof=1))

    return mean, std


# ===== Risk Metrics =====


def _calculate_annualized_volatility(returns: np.ndarray) -> float:
    """
    Calculate annualized volatility (standard deviation of returns).

    Formula: σ_annual = σ_daily × sqrt(252)

    Parameters
    ----------
    returns : np.ndarray
        Daily returns array

    Returns
    -------
    float
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0

    daily_std = float(np.std(returns, ddof=1))

    # Handle zero volatility (or near-zero from floating point precision)
    if daily_std < EPSILON:
        return 0.0

    annualized_vol = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    return float(annualized_vol)


def _calculate_maximum_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown (largest peak-to-trough decline).

    Formula: MDD = min((V_t - V_peak) / V_peak) for all t

    Parameters
    ----------
    returns : np.ndarray
        Daily returns array

    Returns
    -------
    float
        Maximum drawdown (negative value, e.g., -0.15 for 15% drawdown)

    Notes
    -----
    Drawdown is computed on cumulative returns to find the worst peak-to-trough.
    """
    if len(returns) == 0:
        return 0.0

    # Compute cumulative portfolio value (starting at 1.0)
    cumulative = np.cumprod(1.0 + returns)

    # Track running maximum
    running_max = np.maximum.accumulate(cumulative)

    # Compute drawdown at each point
    drawdown = (cumulative - running_max) / running_max

    # Maximum drawdown is the most negative value
    max_drawdown = float(np.min(drawdown))

    return max_drawdown


def _calculate_var_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR).

    VaR(α): The α-quantile of the return distribution (e.g., 5th percentile for 95% confidence)
    CVaR(α): The expected value of returns below VaR (Expected Shortfall)

    Parameters
    ----------
    returns : np.ndarray
        Daily returns array
    confidence_level : float, default 0.95
        Confidence level (e.g., 0.95 for 95% VaR)

    Returns
    -------
    tuple[float, float]
        (var, cvar) - both negative for losses

    Raises
    ------
    ValueError
        If insufficient data (< 100 observations required)

    Notes
    -----
    Minimum 100 observations recommended for stable VaR/CVaR estimation.
    For small samples, estimates may be unreliable.
    """
    min_observations = 100

    if len(returns) < min_observations:
        LOGGER.warning(
            "Insufficient data for VaR/CVaR estimation, returning zeros",
            observations=len(returns),
            required=min_observations,
        )
        return 0.0, 0.0

    # VaR is the (1 - confidence_level) quantile
    # E.g., 95% VaR is 5th percentile
    var_percentile = (1.0 - confidence_level) * 100
    var = float(np.percentile(returns, var_percentile))

    # CVaR is the mean of all returns below VaR
    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        # Should not happen if VaR computed correctly, but handle gracefully
        cvar = var
    else:
        cvar = float(np.mean(tail_returns))

    return var, cvar


# ===== Risk-Adjusted Metrics =====


def _calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float) -> float:
    """
    Calculate Sharpe ratio (excess return per unit of volatility).

    Formula: Sharpe = (mean_return - rf_daily) / std_return × sqrt(252)

    Parameters
    ----------
    returns : np.ndarray
        Daily returns array
    risk_free_rate : float
        Annual risk-free rate

    Returns
    -------
    float
        Annualized Sharpe ratio

    Notes
    -----
    - Risk-free rate is converted to daily equivalent
    - Returns are assumed to be daily
    - Result is annualized using sqrt(252) factor
    """
    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free rate to daily
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR

    # Compute excess returns
    excess_returns = returns - rf_daily

    # Mean and std of excess returns
    mean_excess = float(np.mean(excess_returns))
    std_excess = float(np.std(excess_returns, ddof=1))

    if std_excess < EPSILON:
        return 0.0

    # Annualized Sharpe ratio
    sharpe = (mean_excess / std_excess) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return float(sharpe)


def _calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float) -> float:
    """
    Calculate Sortino ratio (excess return per unit of downside risk).

    Formula: Sortino = (mean_return - rf_daily) / downside_std × sqrt(252)

    Parameters
    ----------
    returns : np.ndarray
        Daily returns array
    risk_free_rate : float
        Annual risk-free rate

    Returns
    -------
    float
        Annualized Sortino ratio

    Notes
    -----
    - Only considers downside volatility (returns below risk-free rate)
    - More appropriate than Sharpe for non-symmetric return distributions
    - Penalizes only downside deviations, not upside volatility
    """
    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free rate to daily
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR

    # Compute excess returns
    excess_returns = returns - rf_daily
    mean_excess = float(np.mean(excess_returns))

    # Downside deviation (only negative excess returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) < 2:
        # No downside volatility - return large positive number or zero
        return float(mean_excess * np.sqrt(TRADING_DAYS_PER_YEAR)) if mean_excess > 0 else 0.0

    downside_std = float(np.std(downside_returns, ddof=1))

    if downside_std < EPSILON:
        return 0.0

    # Annualized Sortino ratio
    sortino = (mean_excess / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return float(sortino)


def _calculate_calmar_ratio(annualized_return: float, maximum_drawdown: float) -> float:
    """
    Calculate Calmar ratio (return-to-drawdown ratio).

    Formula: Calmar = annualized_return / abs(maximum_drawdown)

    Parameters
    ----------
    annualized_return : float
        Annualized return
    maximum_drawdown : float
        Maximum drawdown (negative value)

    Returns
    -------
    float
        Calmar ratio

    Notes
    -----
    - Higher is better (more return per unit of drawdown)
    - Undefined if maximum drawdown is zero (returns zero)
    - Commonly used for trend-following and alternative strategies
    """
    if abs(maximum_drawdown) < EPSILON:
        # No drawdown - return zero (convention)
        return 0.0

    calmar = annualized_return / abs(maximum_drawdown)
    return float(calmar)


def _calculate_information_ratio(
    result: BacktestResult,
    benchmark_result: BacktestResult,
) -> float | None:
    """
    Calculate Information Ratio (excess return vs benchmark per unit of tracking error).

    Formula: IR = mean(excess_return) / std(excess_return) × sqrt(252)

    Parameters
    ----------
    result : BacktestResult
        Strategy backtest result
    benchmark_result : BacktestResult
        Benchmark backtest result

    Returns
    -------
    float | None
        Annualized information ratio, or None if calculation not possible

    Notes
    -----
    - Measures consistency of outperformance vs benchmark
    - IR > 0.5 is considered good, > 1.0 is excellent
    - Requires aligned dates between strategy and benchmark
    """
    try:
        # Align returns on common dates
        strategy_returns = np.array(result.returns, dtype=float)
        benchmark_returns = np.array(benchmark_result.returns, dtype=float)

        # Check if lengths match
        if len(strategy_returns) != len(benchmark_returns):
            LOGGER.warning(
                "Strategy and benchmark have different lengths, cannot compute IR",
                strategy_len=len(strategy_returns),
                benchmark_len=len(benchmark_returns),
            )
            return None

        # Compute excess returns
        excess_returns = strategy_returns - benchmark_returns

        if len(excess_returns) < 2:
            return None

        # Mean and std of excess returns (tracking error)
        mean_excess = float(np.mean(excess_returns))
        tracking_error = float(np.std(excess_returns, ddof=1))

        if tracking_error < EPSILON:
            return None

        # Annualized information ratio
        information_ratio = (mean_excess / tracking_error) * np.sqrt(TRADING_DAYS_PER_YEAR)

        return float(information_ratio)

    except Exception:
        LOGGER.exception("Failed to calculate information ratio")
        return None


# ===== Public API =====

__all__ = [
    "PerformanceMetrics",
    "calculate_performance_metrics",
]
