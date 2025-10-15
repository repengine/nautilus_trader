"""
Comprehensive tests for performance metrics calculations.

This test suite validates all performance metrics against hand-calculated
examples and known benchmarks to ensure mathematical correctness.

Test Coverage:
- Return metrics (cumulative, annualized, monthly statistics)
- Risk metrics (volatility, drawdown, VaR, CVaR)
- Risk-adjusted ratios (Sharpe, Sortino, Calmar, Information)
- Edge cases (zero volatility, negative returns, empty data)
- Integration with BacktestResult

All tests use deterministic data to enable exact validation.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from playground.backtest.engine import BacktestResult
from playground.backtest.performance_metrics import PerformanceMetrics
from playground.backtest.performance_metrics import calculate_performance_metrics


# ===== Fixtures =====


@pytest.fixture
def sample_backtest_result() -> BacktestResult:
    """
    Create a sample backtest result with known characteristics.

    Returns:
    - 252 trading days (1 year)
    - Daily returns: constant 0.1% (25.2% annual geometric)
    - Sharpe ratio should be positive
    - No drawdown
    """
    num_days = 252
    daily_return = 0.001  # 0.1% per day

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(num_days + 1)]

    returns = [daily_return] * num_days
    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    # Mock positions DataFrame
    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "SPY": [0.6, 0.6],
        "AGG": [0.4, 0.4],
    })

    result = BacktestResult(
        strategy_name="Test Strategy",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=np.prod(1.0 + np.array(returns)) - 1.0,
        annualized_return=0.0,  # Will be recalculated
        annualized_volatility=0.0,  # Will be recalculated
        sharpe_ratio=0.0,  # Will be recalculated
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=1000.0,
        turnover_rate=0.05,
        num_rebalances=12,
    )

    return result


@pytest.fixture
def volatile_backtest_result() -> BacktestResult:
    """
    Create a backtest result with high volatility and drawdowns.

    Returns:
    - 252 trading days
    - Alternating +2% / -1% returns (creates volatility and drawdown)
    """
    num_days = 252
    returns = [0.02 if i % 2 == 0 else -0.01 for i in range(num_days)]

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(num_days + 1)]

    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "XLK": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="Volatile Strategy",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=5000.0,
        turnover_rate=0.10,
        num_rebalances=24,
    )

    return result


# ===== Return Metrics Tests =====


def test_cumulative_return_calculation(sample_backtest_result: BacktestResult) -> None:
    """Test cumulative return calculation with known constant returns."""
    metrics = calculate_performance_metrics(sample_backtest_result, risk_free_rate=0.0)

    # Expected: (1.001)^252 - 1 ≈ 0.2872 (28.72%)
    expected_cumulative = (1.001 ** 252) - 1.0

    assert abs(metrics.cumulative_return - expected_cumulative) < 1e-4, (
        f"Expected cumulative return {expected_cumulative:.4f}, "
        f"got {metrics.cumulative_return:.4f}"
    )


def test_annualized_return_calculation(sample_backtest_result: BacktestResult) -> None:
    """Test annualized return matches daily compounding."""
    metrics = calculate_performance_metrics(sample_backtest_result, risk_free_rate=0.0)

    # Expected: (1.001)^252 - 1 ≈ 0.2872
    expected_annual = (1.001 ** 252) - 1.0

    assert abs(metrics.annualized_return - expected_annual) < 1e-4, (
        f"Expected annualized return {expected_annual:.4f}, "
        f"got {metrics.annualized_return:.4f}"
    )


def test_monthly_statistics_calculation() -> None:
    """Test monthly return statistics with known distribution."""
    # Create 24 months of data (2 years)
    # Each month has 21 trading days
    # Month 1: +1% daily, Month 2: -0.5% daily, alternating
    returns = []
    dates = []
    start_date = datetime(2022, 1, 1, tzinfo=UTC)

    for month in range(24):
        month_return = 0.01 if month % 2 == 0 else -0.005
        for day in range(21):
            returns.append(month_return)
            dates.append(start_date + timedelta(days=len(dates)))

    dates = [start_date] + dates  # Add initial date

    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "TEST": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="Monthly Test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    metrics = calculate_performance_metrics(result, risk_free_rate=0.0)

    # Monthly returns should alternate between ~23.3% and ~-9.9%
    # Mean should be positive, std should be substantial
    assert metrics.monthly_return_mean > 0.0
    assert metrics.monthly_return_std > 0.0


# ===== Risk Metrics Tests =====


def test_annualized_volatility_calculation() -> None:
    """Test volatility calculation with known standard deviation."""
    # Create returns with known std dev
    # 252 days, std = 0.01 (1% daily)
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 252).tolist()

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(253)]

    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "TEST": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="Vol Test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    metrics = calculate_performance_metrics(result, risk_free_rate=0.0)

    # Expected: sample std ≈ 0.01, annualized ≈ 0.01 * sqrt(252) ≈ 0.1587
    daily_std = np.std(returns, ddof=1)
    expected_annual_vol = daily_std * np.sqrt(252)

    assert abs(metrics.annualized_volatility - expected_annual_vol) < 1e-6


def test_maximum_drawdown_calculation() -> None:
    """Test maximum drawdown with known drawdown scenario."""
    # Create returns with specific drawdown pattern
    # Need at least 100 observations for VaR/CVaR
    # Start with growth phase, then drawdown, then recovery
    returns = []

    # Growth phase: +2% for 30 days (100 -> 181.1)
    returns.extend([0.02] * 30)

    # Peak reached, now create 25% drawdown
    # From peak, need to lose 25%, so multiply by 0.75
    # Need sequence that compounds to 0.75
    # Use -2% for 14 days: (0.98)^14 ≈ 0.75
    returns.extend([-0.02] * 14)

    # Add some flat/recovery period to get to 100+ observations
    returns.extend([0.001] * 60)  # Small positive returns

    # Total: 30 + 14 + 60 = 104 observations (enough for VaR/CVaR)

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(len(returns) + 1)]

    portfolio_values = [100.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "TEST": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="Drawdown Test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    metrics = calculate_performance_metrics(result, risk_free_rate=0.0)

    # Maximum drawdown should be approximately -25%
    # Calculate actual peak and trough
    peak_value = portfolio_values[30]  # After growth phase
    trough_value = portfolio_values[44]  # After drawdown phase
    expected_drawdown = (trough_value - peak_value) / peak_value

    assert abs(metrics.maximum_drawdown - expected_drawdown) < 0.01, (
        f"Expected drawdown ~{expected_drawdown:.4f}, "
        f"got {metrics.maximum_drawdown:.4f}"
    )


def test_var_cvar_calculation() -> None:
    """Test VaR and CVaR calculations with known distribution."""
    # Create 1000 returns with known distribution
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 1000).tolist()

    start_date = datetime(2020, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(len(returns) + 1)]

    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "TEST": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="VaR Test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    metrics = calculate_performance_metrics(result, risk_free_rate=0.0)

    # VaR 95% should be approximately 5th percentile
    expected_var_95 = np.percentile(returns, 5)

    # CVaR should be mean of tail (returns <= VaR)
    tail_returns = [r for r in returns if r <= metrics.var_95]
    expected_cvar_95 = np.mean(tail_returns)

    assert abs(metrics.var_95 - expected_var_95) < 1e-6
    assert abs(metrics.cvar_95 - expected_cvar_95) < 1e-6

    # CVaR should be more negative than VaR (deeper tail)
    assert metrics.cvar_95 < metrics.var_95


def test_var_insufficient_data_returns_zero() -> None:
    """VaR/CVaR fallback to zero when the sample is too small."""
    returns = [0.001] * 50

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(len(returns) + 1)]

    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "TEST": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="Insufficient Data",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    metrics = calculate_performance_metrics(result, risk_free_rate=0.0)
    assert metrics.var_95 == 0.0
    assert metrics.cvar_95 == 0.0
    assert metrics.var_99 == 0.0
    assert metrics.cvar_99 == 0.0


# ===== Risk-Adjusted Metrics Tests =====


def test_sharpe_ratio_calculation() -> None:
    """Test Sharpe ratio with known mean and volatility."""
    # Create returns: mean = 0.001 (0.1% daily), std = 0.01 (1%)
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 252).tolist()

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(len(returns) + 1)]

    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "TEST": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="Sharpe Test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    metrics = calculate_performance_metrics(result, risk_free_rate=0.02)

    # Expected Sharpe: (mean - rf/252) / std * sqrt(252)
    rf_daily = 0.02 / 252
    mean_excess = np.mean(returns) - rf_daily
    std_excess = np.std(returns, ddof=1)
    expected_sharpe = (mean_excess / std_excess) * np.sqrt(252)

    assert abs(metrics.sharpe_ratio - expected_sharpe) < 1e-6


def test_sortino_ratio_calculation() -> None:
    """Test Sortino ratio with asymmetric returns."""
    # Create asymmetric returns with mix of positive and negative
    # Need enough negative returns relative to risk-free rate
    np.random.seed(42)
    # Generate 252 returns with positive mean but some downside
    returns_positive = np.random.normal(0.003, 0.01, 176).tolist()  # ~70% positive days
    returns_negative = np.random.normal(-0.002, 0.01, 76).tolist()   # ~30% negative days
    returns = returns_positive + returns_negative

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(len(returns) + 1)]

    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "TEST": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="Sortino Test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    metrics = calculate_performance_metrics(result, risk_free_rate=0.0)

    # Sortino should be different from Sharpe (only penalizes downside)
    assert metrics.sortino_ratio != metrics.sharpe_ratio

    # With mixed returns, Sortino should be positive (positive mean)
    assert metrics.sortino_ratio > 0.0
    assert metrics.sharpe_ratio > 0.0

    # Sortino typically higher than Sharpe for positively skewed returns
    # (since it ignores upside volatility)
    assert metrics.sortino_ratio >= metrics.sharpe_ratio * 0.5  # At least half


def test_calmar_ratio_calculation() -> None:
    """Test Calmar ratio with known return and drawdown."""
    # Create 10% annualized return with 20% max drawdown
    # Calmar should be 10% / 20% = 0.5
    returns = [0.0004] * 252  # ~10% annual

    # Insert a drawdown sequence
    returns[100:110] = [-0.02] * 10  # Creates drawdown

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(len(returns) + 1)]

    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "TEST": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="Calmar Test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    metrics = calculate_performance_metrics(result, risk_free_rate=0.0)

    # Calmar = annualized_return / abs(max_drawdown)
    expected_calmar = metrics.annualized_return / abs(metrics.maximum_drawdown)

    assert abs(metrics.calmar_ratio - expected_calmar) < 1e-6


def test_information_ratio_vs_benchmark() -> None:
    """Test information ratio calculation vs benchmark."""
    # Strategy returns: mean 0.002, std 0.01
    # Benchmark returns: mean 0.001, std 0.008
    np.random.seed(42)
    strategy_returns = np.random.normal(0.002, 0.01, 252).tolist()
    benchmark_returns = np.random.normal(0.001, 0.008, 252).tolist()

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(253)]

    def create_result(name: str, returns: list[float]) -> BacktestResult:
        portfolio_values = [1_000_000.0]
        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))

        positions = pl.DataFrame({
            "timestamp": [dates[0], dates[-1]],
            "TEST": [1.0, 1.0],
        })

        return BacktestResult(
            strategy_name=name,
            start_date=start_date,
            end_date=dates[-1],
            dates=dates,
            portfolio_values=portfolio_values,
            returns=returns,
            positions=positions,
            total_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            total_transaction_costs=0.0,
            turnover_rate=0.0,
            num_rebalances=0,
        )

    strategy_result = create_result("Strategy", strategy_returns)
    benchmark_result = create_result("Benchmark", benchmark_returns)

    metrics = calculate_performance_metrics(
        strategy_result,
        benchmark_result=benchmark_result,
        risk_free_rate=0.0,
    )

    # Information ratio should be computed
    assert metrics.information_ratio is not None

    # IR = mean(excess) / std(excess) * sqrt(252)
    excess_returns = np.array(strategy_returns) - np.array(benchmark_returns)
    expected_ir = (np.mean(excess_returns) / np.std(excess_returns, ddof=1)) * np.sqrt(252)

    assert abs(metrics.information_ratio - expected_ir) < 1e-6


# ===== Edge Cases Tests =====


def test_zero_volatility_returns() -> None:
    """Test metrics with zero volatility (constant returns)."""
    returns = [0.001] * 252  # Constant returns

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(len(returns) + 1)]

    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "TEST": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="Zero Vol",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    metrics = calculate_performance_metrics(result, risk_free_rate=0.0)

    # Volatility should be exactly zero
    assert metrics.annualized_volatility == 0.0

    # Sharpe/Sortino should be zero (can't divide by zero vol)
    assert metrics.sharpe_ratio == 0.0


def test_all_negative_returns() -> None:
    """Test metrics with all negative returns."""
    # Use varying negative returns to create non-zero volatility
    np.random.seed(42)
    returns = np.random.normal(-0.002, 0.005, 252).tolist()  # Mean -0.2%, std 0.5%

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(len(returns) + 1)]

    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "TEST": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="All Negative",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    metrics = calculate_performance_metrics(result, risk_free_rate=0.02)

    # Cumulative and annualized returns should be negative
    assert metrics.cumulative_return < 0.0
    assert metrics.annualized_return < 0.0

    # Sharpe should be negative (returns < risk-free)
    assert metrics.sharpe_ratio < 0.0, f"Expected negative Sharpe, got {metrics.sharpe_ratio}"

    # Volatility should be non-zero
    assert metrics.annualized_volatility > 0.0

    # Drawdown should be negative
    assert metrics.maximum_drawdown < 0.0


def test_empty_returns_raises_error() -> None:
    """Test that empty returns raises ValueError."""
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date]

    positions = pl.DataFrame({
        "timestamp": [dates[0]],
        "TEST": [1.0],
    })

    result = BacktestResult(
        strategy_name="Empty",
        start_date=start_date,
        end_date=start_date,
        dates=dates,
        portfolio_values=[1_000_000.0],
        returns=[],  # Empty!
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    with pytest.raises(ValueError, match="Cannot calculate metrics for empty returns"):
        calculate_performance_metrics(result, risk_free_rate=0.0)


def test_negative_risk_free_rate_raises_error() -> None:
    """Test that negative risk-free rate raises ValueError."""
    returns = [0.001] * 100

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(len(returns) + 1)]

    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "TEST": [1.0, 1.0],
    })

    result = BacktestResult(
        strategy_name="Test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=0,
    )

    with pytest.raises(ValueError, match="Risk-free rate must be non-negative"):
        calculate_performance_metrics(result, risk_free_rate=-0.02)


def test_performance_metrics_dataclass_attributes(sample_backtest_result: BacktestResult) -> None:
    """Test that all PerformanceMetrics attributes are populated."""
    metrics = calculate_performance_metrics(sample_backtest_result, risk_free_rate=0.02)

    # Check all attributes are present and have expected types
    assert isinstance(metrics, PerformanceMetrics)
    assert isinstance(metrics.annualized_return, float)
    assert isinstance(metrics.cumulative_return, float)
    assert isinstance(metrics.monthly_return_mean, float)
    assert isinstance(metrics.monthly_return_std, float)
    assert isinstance(metrics.annualized_volatility, float)
    assert isinstance(metrics.maximum_drawdown, float)
    assert isinstance(metrics.var_95, float)
    assert isinstance(metrics.var_99, float)
    assert isinstance(metrics.cvar_95, float)
    assert isinstance(metrics.cvar_99, float)
    assert isinstance(metrics.sharpe_ratio, float)
    assert isinstance(metrics.sortino_ratio, float)
    assert isinstance(metrics.calmar_ratio, float)
    assert metrics.information_ratio is None  # No benchmark
    assert isinstance(metrics.turnover_rate, float)
    assert isinstance(metrics.transaction_costs_total, float)
    assert isinstance(metrics.transaction_costs_pct, float)
    assert isinstance(metrics.num_rebalances, int)
    assert isinstance(metrics.start_date, datetime)
    assert isinstance(metrics.end_date, datetime)
    assert isinstance(metrics.total_days, int)


# ===== Total: 18 Tests =====
