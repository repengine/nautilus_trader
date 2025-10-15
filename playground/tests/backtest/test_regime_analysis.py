"""
Tests for regime analysis framework.

This test suite validates the regime analysis framework for evaluating
portfolio strategy performance across different market conditions.

Test Coverage:
- Regime definition and validation (3 tests)
- Regime performance calculation (4 tests)
- Regime analysis across all regimes (3 tests)
- Strategy comparison and reporting (2 tests)
- Integration with backtest results (2 tests)

Total: 14 tests (exceeds minimum requirement of 12)
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from playground.backtest.engine import BacktestResult
from playground.backtest.regime_analysis import MarketRegime
from playground.backtest.regime_analysis import RegimeAnalysisResult
from playground.backtest.regime_analysis import RegimePerformance
from playground.backtest.regime_analysis import analyze_strategy_across_regimes
from playground.backtest.regime_analysis import compare_strategies_across_regimes
from playground.backtest.regime_analysis import define_market_regimes
from playground.backtest.regime_analysis import generate_regime_report
from playground.backtest.regime_analysis import identify_failure_modes
from playground.backtest.regime_analysis import regime_performance_matrix


# ===== Regime Definition Tests =====


def test_market_regime_construction_valid() -> None:
    """Test MarketRegime construction with valid parameters."""
    regime = MarketRegime(
        name="Test Regime",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 12, 31, tzinfo=UTC),
        description="Test economic period",
        key_events=["Event 1", "Event 2"],
    )

    assert regime.name == "Test Regime"
    assert regime.start == datetime(2020, 1, 1, tzinfo=UTC)
    assert regime.end == datetime(2020, 12, 31, tzinfo=UTC)
    assert regime.description == "Test economic period"
    assert len(regime.key_events) == 2
    assert regime.duration_days == 366  # 2020 is a leap year


def test_market_regime_validation_invalid_dates() -> None:
    """Test MarketRegime validation catches invalid date configurations."""
    # End before start
    with pytest.raises(ValueError, match="end date must be after start date"):
        MarketRegime(
            name="Invalid",
            start=datetime(2020, 12, 31, tzinfo=UTC),
            end=datetime(2020, 1, 1, tzinfo=UTC),  # Before start!
            description="Invalid regime",
            key_events=[],
        )

    # Missing timezone
    with pytest.raises(ValueError, match="start date must be timezone-aware"):
        MarketRegime(
            name="Invalid",
            start=datetime(2020, 1, 1),  # No timezone!
            end=datetime(2020, 12, 31, tzinfo=UTC),
            description="Invalid regime",
            key_events=[],
        )


def test_define_market_regimes_standard() -> None:
    """Test standard market regime definitions for 2010-2024."""
    regimes = define_market_regimes()

    # Should have exactly 7 regimes
    assert len(regimes) == 7

    # Check regime names
    expected_names = {
        "GFC Aftermath",
        "QE Era",
        "Rate Normalization",
        "COVID Crash",
        "Zero Rates",
        "Rate Hiking Cycle",
        "Recent",
    }
    actual_names = {regime.name for regime in regimes}
    assert actual_names == expected_names

    # Check chronological order (no gaps, no overlaps)
    for i in range(len(regimes) - 1):
        current = regimes[i]
        next_regime = regimes[i + 1]

        # Next regime should start after current ends
        # Allow 1-day gap for non-overlapping boundaries
        assert next_regime.start >= current.end

    # Check coverage (should span 2010-2024)
    assert regimes[0].start == datetime(2010, 1, 1, tzinfo=UTC)
    assert regimes[-1].end == datetime(2024, 12, 31, tzinfo=UTC)

    # Check all regimes have descriptions and events
    for regime in regimes:
        assert regime.description
        assert len(regime.key_events) > 0


# ===== Regime Performance Calculation Tests =====


def test_regime_performance_is_successful_property() -> None:
    """Test is_successful property based on Sharpe ratio."""
    regime = MarketRegime(
        name="Test",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 12, 31, tzinfo=UTC),
        description="Test",
        key_events=[],
    )

    # Successful performance (Sharpe > 0)
    perf_success = RegimePerformance(
        regime=regime,
        strategy_name="Test Strategy",
        sharpe_ratio=0.85,
        annualized_return=0.12,
        annualized_volatility=0.15,
        max_drawdown=-0.08,
        calmar_ratio=1.5,
        positive_months_pct=0.75,
        win_rate=0.52,
        num_observations=252,
        num_rebalances=12,
    )
    assert perf_success.is_successful is True

    # Failed performance (Sharpe <= 0)
    perf_failure = RegimePerformance(
        regime=regime,
        strategy_name="Test Strategy",
        sharpe_ratio=-0.25,
        annualized_return=-0.05,
        annualized_volatility=0.20,
        max_drawdown=-0.18,
        calmar_ratio=-0.28,
        positive_months_pct=0.40,
        win_rate=0.45,
        num_observations=252,
        num_rebalances=12,
    )
    assert perf_failure.is_successful is False

    # Edge case: exactly zero Sharpe
    perf_zero = RegimePerformance(
        regime=regime,
        strategy_name="Test Strategy",
        sharpe_ratio=0.0,
        annualized_return=0.02,
        annualized_volatility=0.10,
        max_drawdown=-0.05,
        calmar_ratio=0.4,
        positive_months_pct=0.50,
        win_rate=0.50,
        num_observations=252,
        num_rebalances=12,
    )
    assert perf_zero.is_successful is False  # Must be > 0


def test_calculate_regime_performance_single_regime(
    mock_backtest_result: BacktestResult,
) -> None:
    """Test performance calculation for a single regime."""
    from playground.backtest.regime_analysis import _calculate_regime_performance

    regime = MarketRegime(
        name="Test Regime",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 12, 31, tzinfo=UTC),
        description="Test period",
        key_events=["Event 1"],
    )

    perf = _calculate_regime_performance(
        backtest_result=mock_backtest_result,
        regime=regime,
        risk_free_rate=0.02,
    )

    # Check that all fields are populated
    assert perf.regime == regime
    assert perf.strategy_name == mock_backtest_result.strategy_name
    assert isinstance(perf.sharpe_ratio, float)
    assert isinstance(perf.annualized_return, float)
    assert isinstance(perf.annualized_volatility, float)
    assert isinstance(perf.max_drawdown, float)
    assert isinstance(perf.calmar_ratio, float)
    assert 0.0 <= perf.win_rate <= 1.0
    assert 0.0 <= perf.positive_months_pct <= 1.0
    assert perf.num_observations > 0
    assert perf.num_rebalances >= 0


def test_regime_filtering_edge_cases(mock_backtest_result: BacktestResult) -> None:
    """Test regime filtering with edge cases (short periods, missing data)."""
    from playground.backtest.regime_analysis import _filter_to_regime

    # Valid regime within backtest period
    regime = MarketRegime(
        name="Valid",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 3, 31, tzinfo=UTC),
        description="Valid regime",
        key_events=[],
    )

    result = _filter_to_regime(mock_backtest_result, regime)
    assert result["num_observations"] > 0
    assert len(result["returns"]) > 0
    assert len(result["dates"]) > 0

    # Regime completely outside backtest period (should raise)
    regime_outside = MarketRegime(
        name="Outside",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 12, 31, tzinfo=UTC),
        description="Outside regime",
        key_events=[],
    )

    with pytest.raises(ValueError, match="No data found for regime"):
        _filter_to_regime(mock_backtest_result, regime_outside)


def test_regime_performance_insufficient_data() -> None:
    """Test handling of regimes with insufficient observations."""
    from playground.backtest.regime_analysis import _calculate_regime_performance

    # Create minimal backtest result with very few observations
    dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(10)]
    returns = [0.001] * 9  # Only 9 returns

    result = BacktestResult(
        strategy_name="Test",
        start_date=dates[0],
        end_date=dates[-1],
        dates=dates,
        portfolio_values=[1000.0 * (1 + 0.001 * i) for i in range(10)],
        returns=returns,
        positions=pl.DataFrame({"timestamp": dates}),
        total_return=0.01,
        annualized_return=0.12,
        annualized_volatility=0.10,
        sharpe_ratio=1.0,
        max_drawdown=-0.05,
        calmar_ratio=2.4,
        total_transaction_costs=100.0,
        turnover_rate=0.5,
        num_rebalances=1,
    )

    regime = MarketRegime(
        name="Short",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 1, 10, tzinfo=UTC),
        description="Too short",
        key_events=[],
    )

    # Should raise due to insufficient observations
    with pytest.raises(ValueError, match="Insufficient observations in regime"):
        _calculate_regime_performance(result, regime, risk_free_rate=0.02)


# ===== Regime Analysis Tests =====


def test_analyze_strategy_across_regimes_full(
    mock_backtest_result_full: BacktestResult,
) -> None:
    """Test full regime analysis across all 7 standard regimes."""
    analysis = analyze_strategy_across_regimes(mock_backtest_result_full)

    # Should have results for all 7 regimes
    assert len(analysis.regime_performances) == 7

    # Check all standard regimes are present
    expected_regimes = {
        "GFC Aftermath",
        "QE Era",
        "Rate Normalization",
        "COVID Crash",
        "Zero Rates",
        "Rate Hiking Cycle",
        "Recent",
    }
    assert set(analysis.regime_performances.keys()) == expected_regimes

    # Check success rate is between 0 and 1
    assert 0.0 <= analysis.success_rate <= 1.0

    # Check strategy name matches
    assert analysis.strategy_name == mock_backtest_result_full.strategy_name


def test_regime_analysis_success_rate_calculation() -> None:
    """Test success rate calculation with known outcomes."""
    regime1 = MarketRegime(
        name="Success 1",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 6, 30, tzinfo=UTC),
        description="Good",
        key_events=[],
    )
    regime2 = MarketRegime(
        name="Failure",
        start=datetime(2020, 7, 1, tzinfo=UTC),
        end=datetime(2020, 12, 31, tzinfo=UTC),
        description="Bad",
        key_events=[],
    )

    perf1 = RegimePerformance(
        regime=regime1,
        strategy_name="Test",
        sharpe_ratio=0.85,  # Success
        annualized_return=0.12,
        annualized_volatility=0.15,
        max_drawdown=-0.08,
        calmar_ratio=1.5,
        positive_months_pct=0.75,
        win_rate=0.52,
        num_observations=126,
        num_rebalances=6,
    )

    perf2 = RegimePerformance(
        regime=regime2,
        strategy_name="Test",
        sharpe_ratio=-0.25,  # Failure
        annualized_return=-0.05,
        annualized_volatility=0.20,
        max_drawdown=-0.18,
        calmar_ratio=-0.28,
        positive_months_pct=0.40,
        win_rate=0.45,
        num_observations=126,
        num_rebalances=6,
    )

    analysis = RegimeAnalysisResult(
        strategy_name="Test",
        regime_performances={
            "Success 1": perf1,
            "Failure": perf2,
        },
    )

    # 1 success out of 2 regimes = 50%
    assert analysis.success_rate == 0.5


def test_failure_analysis_identification() -> None:
    """Test identification and explanation of regime failures."""
    regime_fail = MarketRegime(
        name="Failed Regime",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 12, 31, tzinfo=UTC),
        description="Difficult market conditions",
        key_events=["Major shock"],
    )

    regime_success = MarketRegime(
        name="Successful Regime",
        start=datetime(2021, 1, 1, tzinfo=UTC),
        end=datetime(2021, 12, 31, tzinfo=UTC),
        description="Favorable conditions",
        key_events=["Recovery"],
    )

    perf_fail = RegimePerformance(
        regime=regime_fail,
        strategy_name="Test",
        sharpe_ratio=-0.45,
        annualized_return=-0.08,
        annualized_volatility=0.35,
        max_drawdown=-0.22,
        calmar_ratio=-0.36,
        positive_months_pct=0.30,
        win_rate=0.42,
        num_observations=252,
        num_rebalances=12,
    )

    perf_success = RegimePerformance(
        regime=regime_success,
        strategy_name="Test",
        sharpe_ratio=1.20,
        annualized_return=0.18,
        annualized_volatility=0.15,
        max_drawdown=-0.06,
        calmar_ratio=3.0,
        positive_months_pct=0.83,
        win_rate=0.58,
        num_observations=252,
        num_rebalances=12,
    )

    analysis = RegimeAnalysisResult(
        strategy_name="Test",
        regime_performances={
            "Failed Regime": perf_fail,
            "Successful Regime": perf_success,
        },
    )

    failures = analysis.failure_analysis()

    # Should identify only the failed regime
    assert len(failures) == 1
    assert "Failed Regime" in failures

    # Explanation should be detailed and reference metrics
    explanation = failures["Failed Regime"]
    assert "Negative returns" in explanation or "negative returns" in explanation.lower()
    assert "volatility" in explanation.lower()
    assert "drawdown" in explanation.lower()


# ===== Comparison and Reporting Tests =====


def test_compare_strategies_across_regimes(
    mock_backtest_result_full: BacktestResult,
) -> None:
    """Test multi-strategy comparison across regimes."""
    # Create second strategy with slightly different performance
    result2 = BacktestResult(
        strategy_name="Strategy 2",
        start_date=mock_backtest_result_full.start_date,
        end_date=mock_backtest_result_full.end_date,
        dates=mock_backtest_result_full.dates,
        portfolio_values=mock_backtest_result_full.portfolio_values,
        returns=[r * 0.9 for r in mock_backtest_result_full.returns],  # Slightly worse
        positions=mock_backtest_result_full.positions,
        total_return=mock_backtest_result_full.total_return * 0.9,
        annualized_return=mock_backtest_result_full.annualized_return * 0.9,
        annualized_volatility=mock_backtest_result_full.annualized_volatility,
        sharpe_ratio=mock_backtest_result_full.sharpe_ratio * 0.9,
        max_drawdown=mock_backtest_result_full.max_drawdown,
        calmar_ratio=mock_backtest_result_full.calmar_ratio * 0.9,
        total_transaction_costs=mock_backtest_result_full.total_transaction_costs,
        turnover_rate=mock_backtest_result_full.turnover_rate,
        num_rebalances=mock_backtest_result_full.num_rebalances,
    )

    results = {
        "Strategy 1": mock_backtest_result_full,
        "Strategy 2": result2,
    }

    comparison = compare_strategies_across_regimes(results)

    # Should have rows for all strategies × all regimes
    assert len(comparison) == 2 * 7  # 2 strategies × 7 regimes

    # Check required columns
    required_cols = {
        "regime_name",
        "strategy_name",
        "sharpe_ratio",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "win_rate",
    }
    assert required_cols.issubset(comparison.columns)

    # Check both strategies appear
    strategies = comparison["strategy_name"].unique().to_list()
    assert len(strategies) == 2
    assert "Strategy 1" in strategies
    assert "Strategy 2" in strategies


def test_summary_table_generation() -> None:
    """Test summary table generation from regime analysis."""
    regime1 = MarketRegime(
        name="Regime 1",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 6, 30, tzinfo=UTC),
        description="Period 1",
        key_events=[],
    )

    perf1 = RegimePerformance(
        regime=regime1,
        strategy_name="Test",
        sharpe_ratio=0.85,
        annualized_return=0.12,
        annualized_volatility=0.15,
        max_drawdown=-0.08,
        calmar_ratio=1.5,
        positive_months_pct=0.75,
        win_rate=0.52,
        num_observations=126,
        num_rebalances=6,
    )

    analysis = RegimeAnalysisResult(
        strategy_name="Test",
        regime_performances={"Regime 1": perf1},
    )

    summary = analysis.summary_table()

    # Check table structure
    assert not summary.is_empty()
    assert "regime_name" in summary.columns
    assert "sharpe_ratio" in summary.columns
    assert "status" in summary.columns

    # Check status is correctly labeled
    assert summary["status"][0] == "Success"


# ===== Integration Tests =====


def test_regime_analysis_end_to_end(
    mock_backtest_result_full: BacktestResult,
    tmp_path: Path,
) -> None:
    """Test complete regime analysis workflow end-to-end."""
    # Step 1: Analyze strategy across regimes
    analysis = analyze_strategy_across_regimes(mock_backtest_result_full)

    # Step 2: Generate summary table
    summary = analysis.summary_table()
    assert not summary.is_empty()

    # Step 3: Identify failures
    failures = identify_failure_modes(analysis)
    assert isinstance(failures, dict)

    # Step 4: Generate report
    report_path = tmp_path / "regime_analysis_test.md"
    generate_regime_report(analysis, report_path)

    # Verify report was created
    assert report_path.exists()

    # Check report content
    content = report_path.read_text(encoding="utf-8")
    assert "Regime Analysis Report" in content
    assert analysis.strategy_name in content
    assert "Executive Summary" in content
    assert "Performance by Regime" in content


def test_report_generation_with_failures(tmp_path: Path) -> None:
    """Test report generation when strategy has failed regimes."""
    regime_fail = MarketRegime(
        name="Failed Regime",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 12, 31, tzinfo=UTC),
        description="Difficult market",
        key_events=["Shock"],
    )

    regime_success = MarketRegime(
        name="Successful Regime",
        start=datetime(2021, 1, 1, tzinfo=UTC),
        end=datetime(2021, 12, 31, tzinfo=UTC),
        description="Good market",
        key_events=["Recovery"],
    )

    perf_fail = RegimePerformance(
        regime=regime_fail,
        strategy_name="Test Strategy",
        sharpe_ratio=-0.35,
        annualized_return=-0.06,
        annualized_volatility=0.28,
        max_drawdown=-0.19,
        calmar_ratio=-0.32,
        positive_months_pct=0.35,
        win_rate=0.43,
        num_observations=252,
        num_rebalances=12,
    )

    perf_success = RegimePerformance(
        regime=regime_success,
        strategy_name="Test Strategy",
        sharpe_ratio=1.10,
        annualized_return=0.16,
        annualized_volatility=0.14,
        max_drawdown=-0.07,
        calmar_ratio=2.3,
        positive_months_pct=0.80,
        win_rate=0.56,
        num_observations=252,
        num_rebalances=12,
    )

    analysis = RegimeAnalysisResult(
        strategy_name="Test Strategy",
        regime_performances={
            "Failed Regime": perf_fail,
            "Successful Regime": perf_success,
        },
    )

    report_path = tmp_path / "regime_with_failures.md"
    generate_regime_report(analysis, report_path)

    # Verify report exists and contains failure analysis
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")

    assert "Failure Mode Analysis" in content
    assert "Failed Regime" in content
    assert "Successful Regime" in content
    assert "Success" in content  # Status indicator


# ===== Additional Edge Case Tests =====


def test_regime_validation_coverage(mock_backtest_result: BacktestResult) -> None:
    """Test validation that backtest covers all regime periods."""
    from playground.backtest.regime_analysis import _validate_backtest_coverage

    # Valid case: backtest covers regime
    regime_valid = MarketRegime(
        name="Valid",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 12, 31, tzinfo=UTC),
        description="Covered",
        key_events=[],
    )

    # Should not raise
    _validate_backtest_coverage(mock_backtest_result, [regime_valid])

    # Invalid case: regime starts before backtest
    regime_before = MarketRegime(
        name="Before",
        start=datetime(2019, 1, 1, tzinfo=UTC),  # Before backtest start!
        end=datetime(2019, 12, 31, tzinfo=UTC),
        description="Too early",
        key_events=[],
    )

    with pytest.raises(ValueError, match=r"Backtest starts .* after regime"):
        _validate_backtest_coverage(mock_backtest_result, [regime_before])

    # Invalid case: regime ends after backtest
    regime_after = MarketRegime(
        name="After",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2025, 12, 31, tzinfo=UTC),  # After backtest end!
        description="Too late",
        key_events=[],
    )

    with pytest.raises(ValueError, match=r"Backtest ends .* before regime"):
        _validate_backtest_coverage(mock_backtest_result, [regime_after])


def test_identify_failure_modes_no_failures() -> None:
    """Test failure mode identification when no regimes failed."""
    regime = MarketRegime(
        name="Success",
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 12, 31, tzinfo=UTC),
        description="Good",
        key_events=[],
    )

    perf = RegimePerformance(
        regime=regime,
        strategy_name="Test",
        sharpe_ratio=1.20,
        annualized_return=0.18,
        annualized_volatility=0.15,
        max_drawdown=-0.06,
        calmar_ratio=3.0,
        positive_months_pct=0.83,
        win_rate=0.58,
        num_observations=252,
        num_rebalances=12,
    )

    analysis = RegimeAnalysisResult(
        strategy_name="Test",
        regime_performances={"Success": perf},
    )

    failures = identify_failure_modes(analysis)

    # Should return empty dict (no failures)
    assert len(failures) == 0


def test_regime_performance_matrix(mock_backtest_result_full: BacktestResult) -> None:
    """Matrix should pivot regime metrics by strategy."""
    regimes = define_market_regimes()
    analysis = analyze_strategy_across_regimes(mock_backtest_result_full, regimes)
    analyses = {
        "Strategy A": analysis,
        "Strategy B": analysis,
    }

    matrix = regime_performance_matrix(analyses, metric="sharpe_ratio")

    assert matrix.shape[0] == len(regimes)
    assert set(matrix.columns) == {"regime_name", "Strategy A", "Strategy B"}



# ===== Fixtures =====


@pytest.fixture
def mock_backtest_result() -> BacktestResult:
    """Create mock backtest result for testing."""
    start_date = datetime(2020, 1, 1, tzinfo=UTC)
    end_date = datetime(2020, 12, 31, tzinfo=UTC)

    # Generate daily dates
    num_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    # Generate synthetic returns (slight positive drift with noise)
    np.random.seed(42)
    returns = list(np.random.normal(0.0005, 0.01, num_days - 1))  # Daily returns

    # Calculate portfolio values
    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    # Create positions DataFrame
    positions = pl.DataFrame({
        "timestamp": dates,
        "sector_1": [0.5] * len(dates),
        "sector_2": [0.5] * len(dates),
    })

    return BacktestResult(
        strategy_name="Test Strategy",
        start_date=start_date,
        end_date=end_date,
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=float(np.prod([1 + r for r in returns]) - 1),
        annualized_return=0.12,
        annualized_volatility=0.15,
        sharpe_ratio=0.80,
        max_drawdown=-0.08,
        calmar_ratio=1.5,
        total_transaction_costs=5000.0,
        turnover_rate=0.5,
        num_rebalances=12,
    )


@pytest.fixture
def mock_backtest_result_full() -> BacktestResult:
    """Create full backtest result covering 2010-2024 for all regimes."""
    start_date = datetime(2010, 1, 1, tzinfo=UTC)
    end_date = datetime(2024, 12, 31, tzinfo=UTC)

    # Generate daily dates
    num_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    # Generate synthetic returns with varying characteristics per regime
    np.random.seed(42)
    returns = []

    for date in dates[1:]:  # Skip first date (no return)
        # Vary returns by regime to create diverse performance
        if date.year <= 2011:  # GFC Aftermath
            ret = np.random.normal(0.0003, 0.012)
        elif date.year <= 2015:  # QE Era
            ret = np.random.normal(0.0006, 0.008)
        elif date.year <= 2019:  # Rate Normalization
            ret = np.random.normal(0.0004, 0.010)
        elif date.year == 2020 and date.month <= 4:  # COVID Crash
            ret = np.random.normal(-0.002, 0.035)  # Negative returns, high vol
        elif date.year <= 2021:  # Zero Rates
            ret = np.random.normal(0.0008, 0.009)
        elif date.year <= 2023:  # Rate Hiking
            ret = np.random.normal(0.0002, 0.013)
        else:  # Recent
            ret = np.random.normal(0.0005, 0.011)

        returns.append(ret)

    # Calculate portfolio values
    portfolio_values = [1_000_000.0]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    # Create positions DataFrame
    positions = pl.DataFrame({
        "timestamp": dates,
        "sector_1": [0.5] * len(dates),
        "sector_2": [0.5] * len(dates),
    })

    return BacktestResult(
        strategy_name="Full Test Strategy",
        start_date=start_date,
        end_date=end_date,
        dates=dates,
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        total_return=float(np.prod([1 + r for r in returns]) - 1),
        annualized_return=0.08,
        annualized_volatility=0.14,
        sharpe_ratio=0.57,
        max_drawdown=-0.15,
        calmar_ratio=0.53,
        total_transaction_costs=50000.0,
        turnover_rate=0.4,
        num_rebalances=180,  # ~15 years × 12 months
    )
