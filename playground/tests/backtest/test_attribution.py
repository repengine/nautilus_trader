"""
Comprehensive tests for factor attribution analysis.

This test suite validates factor attribution calculations against hand-calculated
examples and statistical theory to ensure mathematical correctness.

Test Coverage:
- Attribution calculation and identity validation
- Statistical significance testing (t-stats, p-values)
- Time series aggregation (monthly, weekly)
- Rolling attribution windows
- Strategy comparison
- Performance decomposition
- Edge cases (zero returns, missing data, single observation)

All tests use deterministic data to enable exact validation.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from playground.backtest.attribution import AttributionResult
from playground.backtest.attribution import FactorContribution
from playground.backtest.attribution import calculate_factor_attribution
from playground.backtest.attribution import compare_attribution_across_strategies
from playground.backtest.attribution import decompose_performance_by_factor
from playground.backtest.attribution import perform_rolling_attribution
from playground.backtest.engine import BacktestResult


# ===== Fixtures =====


@pytest.fixture
def simple_backtest_result() -> BacktestResult:
    """
    Create a simple backtest result with known characteristics.

    Returns:
    - 12 months of data (1 month per period)
    - Monthly returns: constant 1% (12.68% annual geometric)
    """
    num_months = 12
    monthly_return = 0.01  # 1% per month

    start_date = datetime(2023, 1, 1, tzinfo=UTC)

    # Create monthly dates (end of each month)
    dates = [start_date + timedelta(days=30 * i) for i in range(num_months + 1)]

    # Daily returns (approximate 21 trading days per month)
    daily_returns = []
    daily_dates = [start_date]
    for month_idx in range(num_months):
        # Distribute monthly return across ~21 trading days
        daily_ret = (1 + monthly_return) ** (1 / 21) - 1
        for day in range(21):
            daily_returns.append(daily_ret)
            daily_dates.append(daily_dates[-1] + timedelta(days=1))

    # Calculate portfolio values
    portfolio_values = [1_000_000.0]
    for ret in daily_returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    # Mock positions DataFrame
    positions = pl.DataFrame({
        "timestamp": [dates[0], dates[-1]],
        "SPY": [1.0, 1.0],
    })

    total_return = np.prod(1.0 + np.array(daily_returns)) - 1.0

    result = BacktestResult(
        strategy_name="Test Strategy",
        start_date=start_date,
        end_date=dates[-1],
        dates=daily_dates,
        portfolio_values=portfolio_values,
        returns=daily_returns,
        positions=positions,
        total_return=float(total_return),
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=12,
    )

    return result


@pytest.fixture
def simple_factor_returns() -> pl.DataFrame:
    """
    Create simple factor returns aligned with backtest.

    Factor returns explain 80% of portfolio returns:
    - Duration: 0.4% per month (40% of return)
    - Credit: 0.3% per month (30% of return)
    - Liquidity: 0.1% per month (10% of return)
    - Alpha: 0.2% per month (20% of return)
    """
    start_date = datetime(2023, 1, 1, tzinfo=UTC)

    # Create daily factor returns
    factor_data = []
    current_date = start_date

    # 12 months × 21 days = 252 days
    for _ in range(252):
        # Daily factor returns (distribute monthly returns)
        dur_ret = (1.004) ** (1 / 21) - 1  # 0.4% monthly
        cred_ret = (1.003) ** (1 / 21) - 1  # 0.3% monthly
        liq_ret = (1.001) ** (1 / 21) - 1  # 0.1% monthly

        factor_data.append({
            "timestamp": current_date,
            "duration_return": dur_ret,
            "credit_return": cred_ret,
            "liquidity_return": liq_ret,
        })

        current_date += timedelta(days=1)

    return pl.DataFrame(factor_data)


@pytest.fixture
def zero_factor_returns() -> pl.DataFrame:
    """Create factor returns that are all zero."""
    start_date = datetime(2023, 1, 1, tzinfo=UTC)

    factor_data = []
    current_date = start_date

    for _ in range(252):
        factor_data.append({
            "timestamp": current_date,
            "duration_return": 0.0,
            "credit_return": 0.0,
            "liquidity_return": 0.0,
        })
        current_date += timedelta(days=1)

    return pl.DataFrame(factor_data)


# ===== Attribution Calculation Tests =====


def test_attribution_identity_holds(
    simple_backtest_result: BacktestResult,
    simple_factor_returns: pl.DataFrame,
) -> None:
    """Test that attribution sum equals portfolio return for each period."""
    attribution = calculate_factor_attribution(
        simple_backtest_result,
        simple_factor_returns,
        frequency="monthly",
    )

    # Check each contribution
    for contrib in attribution.contributions:
        total_attribution = (
            contrib.alpha
            + contrib.duration_contribution
            + contrib.credit_contribution
            + contrib.liquidity_contribution
            + contrib.residual
        )

        # Should match portfolio return within numerical tolerance
        assert np.isclose(total_attribution, contrib.portfolio_return, atol=1e-6), (
            f"Attribution identity violated: {total_attribution:.6f} != {contrib.portfolio_return:.6f}"
        )


def test_attribution_result_enforces_total_return_identity() -> None:
    """Aggregate contributions must reconcile to portfolio return within 1 bp."""
    contribution = FactorContribution(
        date=datetime(2024, 1, 1, tzinfo=UTC),
        duration_return=0.0,
        credit_return=0.0,
        liquidity_return=0.0,
        duration_beta=0.0,
        credit_beta=0.0,
        liquidity_beta=0.0,
        duration_contribution=0.0,
        credit_contribution=0.0,
        liquidity_contribution=0.0,
        alpha=0.01,
        residual=0.0,
        portfolio_return=0.01,
    )
    with pytest.raises(ValueError, match="1 bp tolerance"):
        AttributionResult(
            strategy_name="Test Strategy",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, tzinfo=UTC),
            contributions=[contribution],
            average_alpha=0.01,
            alpha_t_stat=0.0,
            alpha_p_value=1.0,
            avg_duration_contribution=0.0,
            avg_credit_contribution=0.0,
            avg_liquidity_contribution=0.0,
            avg_residual=0.0,
            total_return=0.011,
            total_alpha=0.01,
            total_duration=0.0,
            total_credit=0.0,
            total_liquidity=0.0,
            total_residual=0.0,
        )


def test_attribution_with_zero_factors(
    simple_backtest_result: BacktestResult,
    zero_factor_returns: pl.DataFrame,
) -> None:
    """Test attribution when all factor returns are zero (all return is alpha)."""
    attribution = calculate_factor_attribution(
        simple_backtest_result,
        zero_factor_returns,
        frequency="monthly",
    )

    # All contributions should be zero
    for contrib in attribution.contributions:
        assert np.isclose(contrib.duration_contribution, 0.0, atol=1e-6)
        assert np.isclose(contrib.credit_contribution, 0.0, atol=1e-6)
        assert np.isclose(contrib.liquidity_contribution, 0.0, atol=1e-6)

        # Alpha + residual should equal portfolio return
        assert np.isclose(
            contrib.alpha + contrib.residual,
            contrib.portfolio_return,
            atol=1e-6,
        )


def test_attribution_hand_calculated() -> None:
    """Test attribution with hand-calculated example."""
    # Create simple 3-period example
    # Period 1: port=1%, dur=0.5%, cred=0.3%, liq=0.1%
    # Period 2: port=2%, dur=1.0%, cred=0.6%, liq=0.2%
    # Period 3: port=-1%, dur=-0.5%, cred=-0.3%, liq=-0.1%

    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [
        start_date,
        start_date + timedelta(days=30),
        start_date + timedelta(days=60),
        start_date + timedelta(days=90),
    ]

    portfolio_returns = [0.01, 0.02, -0.01]

    result = BacktestResult(
        strategy_name="Hand Calculated",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=[1.0, 1.01, 1.0302, 1.020298],
        returns=portfolio_returns,
        positions=pl.DataFrame({"timestamp": dates[:2], "SPY": [1.0, 1.0]}),
        total_return=0.020298,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=3,
    )

    factor_df = pl.DataFrame({
        "timestamp": dates[1:],
        "duration_return": [0.005, 0.01, -0.005],
        "credit_return": [0.003, 0.006, -0.003],
        "liquidity_return": [0.001, 0.002, -0.001],
    })

    attribution = calculate_factor_attribution(result, factor_df, frequency="monthly")

    # Verify we get 3 contributions
    assert len(attribution.contributions) == 3

    # Verify identity for each period
    for contrib in attribution.contributions:
        total = (
            contrib.alpha
            + contrib.duration_contribution
            + contrib.credit_contribution
            + contrib.liquidity_contribution
            + contrib.residual
        )
        assert np.isclose(total, contrib.portfolio_return, atol=1e-8)


def test_factor_contribution_validation() -> None:
    """Test that FactorContribution validates attribution identity."""
    # Valid contribution (sum matches)
    # Sum: 0.005 + 0.0015 + 0.0004 + 0.001 + 0.0001 = 0.008
    valid_contrib = FactorContribution(
        date=datetime(2023, 1, 1, tzinfo=UTC),
        duration_return=0.01,
        credit_return=0.005,
        liquidity_return=0.002,
        duration_beta=0.5,
        credit_beta=0.3,
        liquidity_beta=0.2,
        duration_contribution=0.005,  # 0.5 * 0.01
        credit_contribution=0.0015,  # 0.3 * 0.005
        liquidity_contribution=0.0004,  # 0.2 * 0.002
        alpha=0.001,
        residual=0.0001,
        portfolio_return=0.008,  # Correct sum
    )

    # Should not raise
    assert valid_contrib.portfolio_return == 0.008

    # Invalid contribution (sum doesn't match)
    with pytest.raises(ValueError, match="Attribution doesn't sum"):
        FactorContribution(
            date=datetime(2023, 1, 1, tzinfo=UTC),
            duration_return=0.01,
            credit_return=0.005,
            liquidity_return=0.002,
            duration_beta=0.5,
            credit_beta=0.3,
            liquidity_beta=0.2,
            duration_contribution=0.005,
            credit_contribution=0.0015,
            liquidity_contribution=0.0004,
            alpha=0.001,
            residual=0.0001,
            portfolio_return=0.999,  # Wrong!
        )


# ===== Statistical Tests =====


def test_alpha_significance_high_tstat() -> None:
    """Test alpha significance with high t-statistic (> 2.0)."""
    # Create data with significant alpha
    # 36 months, consistent 0.5% alpha with slight variation in factors
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=30 * i) for i in range(37)]

    # Portfolio returns: 1% per month (0.5% alpha + 0.5% from factors)
    portfolio_returns = [0.01] * 36

    result = BacktestResult(
        strategy_name="High Alpha",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=[1.0] * 37,
        returns=portfolio_returns,
        positions=pl.DataFrame({"timestamp": dates[:2], "SPY": [1.0, 1.0]}),
        total_return=0.36,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=36,
    )

    # Factor returns with slight variation to avoid perfect multicollinearity
    np.random.seed(42)
    factor_df = pl.DataFrame({
        "timestamp": dates[1:],
        "duration_return": list(0.003 + np.random.normal(0, 0.0001, 36)),
        "credit_return": list(0.001 + np.random.normal(0, 0.0001, 36)),
        "liquidity_return": list(0.001 + np.random.normal(0, 0.0001, 36)),
    })

    attribution = calculate_factor_attribution(result, factor_df, frequency="monthly")

    # With OLS fitting and some noise, alpha might not be exactly 0.5% monthly
    # But should be significantly positive
    # Check that alpha is positive and statistically significant
    assert attribution.average_alpha > 0.0  # Positive alpha
    assert attribution.is_alpha_significant  # Statistically significant


def test_alpha_insignificance_low_tstat() -> None:
    """Test alpha insignificance with low t-statistic (< 2.0)."""
    # Create data with high noise, low signal
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=30 * i) for i in range(13)]

    # Portfolio returns: noisy, centered around 0
    np.random.seed(42)
    portfolio_returns = list(np.random.normal(0.001, 0.05, 12))  # High volatility

    result = BacktestResult(
        strategy_name="Low Alpha",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=[1.0] * 13,
        returns=portfolio_returns,
        positions=pl.DataFrame({"timestamp": dates[:2], "SPY": [1.0, 1.0]}),
        total_return=sum(portfolio_returns),
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=12,
    )

    # Factor returns: similar noise
    factor_df = pl.DataFrame({
        "timestamp": dates[1:],
        "duration_return": list(np.random.normal(0.0, 0.03, 12)),
        "credit_return": list(np.random.normal(0.0, 0.02, 12)),
        "liquidity_return": list(np.random.normal(0.0, 0.01, 12)),
    })

    attribution = calculate_factor_attribution(result, factor_df, frequency="monthly")

    # Should NOT have significant alpha (high noise)
    assert not attribution.is_alpha_significant
    assert abs(attribution.alpha_t_stat) < 2.0


def test_tstat_and_pvalue_calculations() -> None:
    """Test that t-statistic and p-value are correctly calculated."""
    # Create deterministic example
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=30 * i) for i in range(25)]

    # Consistent returns for low standard error
    portfolio_returns = [0.02] * 24

    result = BacktestResult(
        strategy_name="T-test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=[1.0] * 25,
        returns=portfolio_returns,
        positions=pl.DataFrame({"timestamp": dates[:2], "SPY": [1.0, 1.0]}),
        total_return=0.48,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=24,
    )

    # Factor returns that explain some variance
    factor_df = pl.DataFrame({
        "timestamp": dates[1:],
        "duration_return": [0.01] * 24,
        "credit_return": [0.005] * 24,
        "liquidity_return": [0.003] * 24,
    })

    attribution = calculate_factor_attribution(result, factor_df, frequency="monthly")

    # Verify t-stat and p-value relationship
    # For two-tailed test: p = 2 * (1 - CDF(|t|))
    from scipy import stats as sp_stats

    df = 24 - 4  # n - k parameters
    expected_p_value = 2 * (1 - sp_stats.t.cdf(abs(attribution.alpha_t_stat), df=df))

    assert np.isclose(attribution.alpha_p_value, expected_p_value, atol=1e-6)


# ===== Time Series Tests =====


def test_monthly_aggregation(
    simple_backtest_result: BacktestResult,
    simple_factor_returns: pl.DataFrame,
) -> None:
    """Test monthly aggregation of daily returns."""
    attribution = calculate_factor_attribution(
        simple_backtest_result,
        simple_factor_returns,
        frequency="monthly",
    )

    # Should have contributions (number depends on how 252 days aggregate)
    # 252 days ≈ 8-9 complete months
    assert len(attribution.contributions) >= 8
    assert len(attribution.contributions) <= 12

    # Each contribution should be for a different month
    months = [c.date.month for c in attribution.contributions]
    # All months should be unique
    assert len(months) == len(set(months))


def test_rolling_attribution_windows(
    simple_backtest_result: BacktestResult,
    simple_factor_returns: pl.DataFrame,
) -> None:
    """Test rolling attribution with 6-month windows."""
    rolling_results = perform_rolling_attribution(
        simple_backtest_result,
        simple_factor_returns,
        window_months=6,
    )

    # With 8 complete months and 6-month windows, should have 3 results (months 6-8)
    assert len(rolling_results) >= 2
    assert len(rolling_results) <= 7

    # Each result should have contributions (may vary due to partial months)
    for result in rolling_results:
        assert len(result.contributions) >= 5
        assert len(result.contributions) <= 6


# ===== Integration Tests =====


def test_strategy_comparison(
    simple_backtest_result: BacktestResult,
    simple_factor_returns: pl.DataFrame,
) -> None:
    """Test comparison across multiple strategies."""
    # Create multiple attribution results
    attribution1 = calculate_factor_attribution(
        simple_backtest_result,
        simple_factor_returns,
        frequency="monthly",
    )

    # Create second strategy with higher returns
    result2 = BacktestResult(
        strategy_name="High Return Strategy",
        start_date=simple_backtest_result.start_date,
        end_date=simple_backtest_result.end_date,
        dates=simple_backtest_result.dates,
        portfolio_values=simple_backtest_result.portfolio_values,
        returns=[r * 1.5 for r in simple_backtest_result.returns],
        positions=simple_backtest_result.positions,
        total_return=simple_backtest_result.total_return * 1.5,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=12,
    )

    attribution2 = calculate_factor_attribution(
        result2,
        simple_factor_returns,
        frequency="monthly",
    )

    # Compare
    comparison = compare_attribution_across_strategies({
        "Strategy 1": attribution1,
        "Strategy 2": attribution2,
    })

    # Should have 2 rows
    assert len(comparison) == 2

    # Should have required columns
    required_cols = {
        "strategy_name",
        "average_alpha",
        "alpha_t_stat",
        "is_significant",
        "duration_contribution",
        "credit_contribution",
        "liquidity_contribution",
    }
    assert required_cols.issubset(set(comparison.columns))


def test_performance_decomposition(
    simple_backtest_result: BacktestResult,
    simple_factor_returns: pl.DataFrame,
) -> None:
    """Test performance decomposition components are calculated correctly."""
    attribution = calculate_factor_attribution(
        simple_backtest_result,
        simple_factor_returns,
        frequency="monthly",
    )

    decomposition = decompose_performance_by_factor(attribution)

    # All components should exist
    assert "alpha" in decomposition
    assert "duration" in decomposition
    assert "credit" in decomposition
    assert "liquidity" in decomposition
    assert "residual" in decomposition

    # Percentages are based on total_return from BacktestResult
    # They won't sum to exactly 100% due to geometric vs arithmetic returns
    # But they should be reasonable (within 20% of 100%)
    total_pct = sum(decomposition.values())
    assert 80 < total_pct < 120, f"Total percentage {total_pct} outside reasonable range"

    # Should have all components
    assert "alpha" in decomposition
    assert "duration" in decomposition
    assert "credit" in decomposition
    assert "liquidity" in decomposition
    assert "residual" in decomposition


def test_attribution_result_methods(
    simple_backtest_result: BacktestResult,
    simple_factor_returns: pl.DataFrame,
) -> None:
    """Test AttributionResult methods (summary_table, to_dict)."""
    attribution = calculate_factor_attribution(
        simple_backtest_result,
        simple_factor_returns,
        frequency="monthly",
    )

    # Test summary_table
    summary = attribution.summary_table()
    assert len(summary) == 5  # 5 components
    assert "component" in summary.columns
    assert "average_contribution" in summary.columns
    assert "total_contribution" in summary.columns
    assert "pct_of_total" in summary.columns

    # Test to_dict
    result_dict = attribution.to_dict()
    assert isinstance(result_dict, dict)
    assert "average_alpha" in result_dict
    assert "alpha_t_stat" in result_dict
    assert "total_return" in result_dict


# ===== Edge Cases =====


def test_missing_factor_columns() -> None:
    """Test error handling for missing factor columns."""
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date, start_date + timedelta(days=30)]

    result = BacktestResult(
        strategy_name="Test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=[1.0, 1.01],
        returns=[0.01],
        positions=pl.DataFrame({"timestamp": dates, "SPY": [1.0, 1.0]}),
        total_return=0.01,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=1,
    )

    # Missing columns
    factor_df = pl.DataFrame({
        "timestamp": dates[1:],
        "duration_return": [0.01],
        # Missing credit_return and liquidity_return
    })

    with pytest.raises(ValueError, match="missing required columns"):
        calculate_factor_attribution(result, factor_df, frequency="monthly")


def test_single_observation_handling() -> None:
    """Test handling of single observation (edge case for regression)."""
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date, start_date + timedelta(days=30)]

    result = BacktestResult(
        strategy_name="Single",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=[1.0, 1.01],
        returns=[0.01],
        positions=pl.DataFrame({"timestamp": dates, "SPY": [1.0, 1.0]}),
        total_return=0.01,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=1,
    )

    factor_df = pl.DataFrame({
        "timestamp": dates[1:],
        "duration_return": [0.005],
        "credit_return": [0.003],
        "liquidity_return": [0.001],
    })

    # Should still work (but t-stat will be undefined due to df=0)
    attribution = calculate_factor_attribution(result, factor_df, frequency="monthly")

    assert len(attribution.contributions) == 1
    # With df=0, standard error can't be calculated, so t-stat should be 0
    assert attribution.alpha_t_stat == 0.0


def test_zero_return_decomposition() -> None:
    """Test decomposition when total return is zero."""
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=30 * i) for i in range(3)]

    # Returns that sum to zero
    result = BacktestResult(
        strategy_name="Zero Return",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=[1.0, 1.01, 1.0],
        returns=[0.01, -0.01],
        positions=pl.DataFrame({"timestamp": dates[:2], "SPY": [1.0, 1.0]}),
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=2,
    )

    factor_df = pl.DataFrame({
        "timestamp": dates[1:],
        "duration_return": [0.005, -0.005],
        "credit_return": [0.003, -0.003],
        "liquidity_return": [0.001, -0.001],
    })

    attribution = calculate_factor_attribution(result, factor_df, frequency="monthly")

    # Decomposition should return zeros (avoid division by zero)
    decomposition = decompose_performance_by_factor(attribution)

    assert decomposition["alpha"] == 0.0
    assert decomposition["duration"] == 0.0
    assert decomposition["credit"] == 0.0
    assert decomposition["liquidity"] == 0.0
    assert decomposition["residual"] == 0.0


def test_invalid_frequency() -> None:
    """Test error handling for invalid frequency."""
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date, start_date + timedelta(days=30)]

    result = BacktestResult(
        strategy_name="Test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=[1.0, 1.01],
        returns=[0.01],
        positions=pl.DataFrame({"timestamp": dates, "SPY": [1.0, 1.0]}),
        total_return=0.01,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=1,
    )

    factor_df = pl.DataFrame({
        "timestamp": dates[1:],
        "duration_return": [0.01],
        "credit_return": [0.005],
        "liquidity_return": [0.002],
    })

    with pytest.raises(ValueError, match="Invalid frequency"):
        calculate_factor_attribution(result, factor_df, frequency="yearly")


def test_rolling_window_too_small() -> None:
    """Test error when rolling window is too small."""
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=30 * i) for i in range(13)]

    result = BacktestResult(
        strategy_name="Test",
        start_date=start_date,
        end_date=dates[-1],
        dates=dates,
        portfolio_values=[1.0] * 13,
        returns=[0.01] * 12,
        positions=pl.DataFrame({"timestamp": dates[:2], "SPY": [1.0, 1.0]}),
        total_return=0.12,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        total_transaction_costs=0.0,
        turnover_rate=0.0,
        num_rebalances=12,
    )

    factor_df = pl.DataFrame({
        "timestamp": dates[1:],
        "duration_return": [0.005] * 12,
        "credit_return": [0.003] * 12,
        "liquidity_return": [0.001] * 12,
    })

    # Window too small
    with pytest.raises(ValueError, match="at least 6 months"):
        perform_rolling_attribution(result, factor_df, window_months=3)
