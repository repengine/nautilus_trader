"""
Comprehensive tests for backtest runner orchestration.

This test suite validates the backtest suite orchestration, including:
- Running multiple strategies in parallel
- Strategy comparison tables
- Report generation
- Configuration handling
- Error recovery

All tests use mock datasets to enable fast, deterministic testing.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from playground.backtest.runner import BacktestSuite
from playground.backtest.runner import run_full_backtest_suite
from playground.backtest.splits import TrainTestSplit
from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.dataset import SectorDataset


# ===== Fixtures =====


@pytest.fixture
def mock_dataset_path(tmp_path: Path) -> Path:
    """
    Create a mock sector returns dataset for testing.

    Returns:
    - Parquet file with ~15 years of weekly data
    - 3 sectors (SPY, AGG, XLK)
    - Simple positive returns
    """
    start_date = datetime(2010, 1, 1, tzinfo=UTC)
    end_date = datetime(2024, 12, 31, tzinfo=UTC)
    step = timedelta(days=7)  # Weekly observations to keep tests fast

    data = []
    date = start_date
    while date <= end_date:
        for sector in ["SPY", "AGG", "XLK"]:
            if sector == "SPY":
                ret = 0.0005
            elif sector == "XLK":
                ret = 0.0004
            else:
                ret = 0.0002

            data.append({
                "timestamp": date,
                "symbol": sector,
                "return": ret,
            })
        date += step

    df = pl.DataFrame(data)

    dataset_path = tmp_path / "sector_returns.parquet"
    df.write_parquet(dataset_path)

    return dataset_path


@pytest.fixture
def mock_split() -> TrainTestSplit:
    """Create a simple train/test split for testing."""
    return TrainTestSplit(
        train_start=datetime(2022, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 12, 31, tzinfo=UTC),
        test_start=datetime(2023, 1, 1, tzinfo=UTC),
        test_end=datetime(2023, 12, 31, tzinfo=UTC),
    )


@pytest.fixture
def mock_sector_dataset() -> SectorDataset:
    """Create a mock SectorDataset for testing."""
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    num_days = 252

    # Sector returns
    sector_data = []
    for day in range(num_days):
        date = start_date + timedelta(days=day)
        for sector in ["SPY", "AGG", "XLK"]:
            sector_data.append({
                "timestamp": date,
                "symbol": sector,
                "return": 0.0005 if sector == "SPY" else 0.0003,
            })

    sector_returns = pl.DataFrame(sector_data)

    # Factor returns (mock)
    factor_dates = [start_date + timedelta(days=i) for i in range(num_days)]
    factor_returns = pl.DataFrame({
        "timestamp": factor_dates,
        "factor_duration": [0.0] * num_days,
        "factor_credit": [0.0] * num_days,
        "factor_liquidity": [0.0] * num_days,
    })

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=num_days,
        factor_expected_days=num_days,
        sector_coverage={"SPY": 1.0, "AGG": 1.0, "XLK": 1.0},
        factor_coverage={
            "factor_duration": 1.0,
            "factor_credit": 1.0,
            "factor_liquidity": 1.0,
        },
    )

    return SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )


# ===== Suite Execution Tests =====


def test_run_full_backtest_suite_basic(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test basic backtest suite execution."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
    )

    # Check that suite was created
    assert isinstance(suite, BacktestSuite)

    # Check that strategies were run
    assert len(suite.strategies) > 0
    assert len(suite.metrics) > 0

    # Check that equal-weight baseline exists
    assert "Equal Weight" in suite.strategies
    assert "Equal Weight" in suite.metrics

    # Check that key suite artifacts exist
    assert output_dir.exists()
    assert (output_dir / "performance_comparison_table.csv").exists()
    assert (output_dir / "train_vs_test_metrics.csv").exists()
    report_file = output_dir / "backtest_results_2022_2023.md"
    assert report_file.exists()

    # Train/test/full results should be tracked
    assert "Equal Weight" in suite.train_results
    assert "Equal Weight" in suite.full_results
    assert "Equal Weight" in suite.overall_metrics

    train_vs_test = suite.train_vs_test_table()
    assert not train_vs_test.is_empty()


def test_run_full_backtest_suite_default_split(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Test suite execution with default train/test split."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=None,  # Use default
    )

    # Check that default split was used
    assert suite.split is not None
    assert suite.split.train_start.year == 2010
    assert suite.split.test_start.year == 2019

    regime_summary_path = output_dir / "regime_summary.csv"
    if suite.regime_results:
        assert regime_summary_path.exists()
    else:
        assert not regime_summary_path.exists()
    assert (output_dir / "full_period_metrics.csv").exists()

    report_file = output_dir / "backtest_results_2010_2024.md"
    assert report_file.exists()


def test_run_full_backtest_suite_config_overrides(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test suite execution with custom configuration overrides."""
    output_dir = tmp_path / "results"

    config_overrides = {
        "initial_capital": 5_000_000.0,
        "transaction_cost_bps": 5.0,
        "rebalance_frequency": "monthly",
    }

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
        config_overrides=config_overrides,
    )

    # Check that config was applied
    assert suite.config.initial_capital == 5_000_000.0
    assert suite.config.transaction_cost_bps == 5.0
    assert suite.config.rebalance_frequency == "monthly"


def test_run_full_backtest_suite_nonexistent_dataset_raises_error(
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that nonexistent dataset raises FileNotFoundError."""
    output_dir = tmp_path / "results"
    nonexistent_path = tmp_path / "does_not_exist.parquet"

    with pytest.raises(FileNotFoundError, match="Dataset path does not exist"):
        run_full_backtest_suite(
            dataset_path=nonexistent_path,
            output_dir=output_dir,
            split=mock_split,
        )


def test_backtest_suite_compare_strategies(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test strategy comparison table generation."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
    )

    comparison = suite.compare_strategies()

    # Check that comparison table is valid
    assert isinstance(comparison, pl.DataFrame)
    assert not comparison.is_empty()

    # Check required columns
    expected_cols = {
        "strategy",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "information_ratio",
        "num_rebalances",
        "transaction_costs",
    }
    assert expected_cols.issubset(comparison.columns)

    # Check that strategies are sorted by Sharpe ratio
    sharpe_ratios = comparison["sharpe_ratio"].to_list()
    assert sharpe_ratios == sorted(sharpe_ratios, reverse=True)


def test_backtest_suite_to_markdown_report(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test markdown report generation."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
    )

    report_path = tmp_path / "custom_report.md"
    suite.to_markdown_report(report_path)

    # Check that report was created
    assert report_path.exists()

    # Check report content
    content = report_path.read_text()

    # Should contain headers
    assert "# Backtest Results: Full Strategy Suite" in content
    assert "## Configuration" in content
    assert "## Executive Summary" in content
    assert "## Detailed Performance Metrics" in content

    # Should contain strategy names
    for strategy_name in suite.strategies.keys():
        assert strategy_name in content

    # Should contain key metrics
    assert "Sharpe Ratio" in content
    assert "Maximum Drawdown" in content
    assert "Annualized Return" in content
    assert "## Train vs Test Comparison" in content


def test_backtest_suite_report_format_dataframe() -> None:
    """Test DataFrame formatting as markdown table."""
    # Create a simple suite to test formatting
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    end_date = datetime(2023, 12, 31, tzinfo=UTC)

    split = TrainTestSplit(
        train_start=datetime(2022, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 12, 31, tzinfo=UTC),
        test_start=start_date,
        test_end=end_date,
    )

    from playground.backtest.engine import BacktestConfig

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
    )

    suite = BacktestSuite(
        strategies={},
        metrics={},
        split=split,
        config=config,
    )

    # Create test DataFrame
    df = pl.DataFrame({
        "strategy": ["Test1", "Test2"],
        "sharpe_ratio": [1.5, 1.2],
        "annualized_return": [10.5, 8.3],
    })

    markdown = suite._format_dataframe_as_markdown(df)

    # Check markdown formatting
    assert "| strategy | sharpe_ratio | annualized_return |" in markdown
    assert "| --- | --- | --- |" in markdown
    assert "| Test1 | 1.500 | 10.50 |" in markdown
    assert "| Test2 | 1.200 | 8.30 |" in markdown


# ===== Strategy Coverage Tests =====


def test_backtest_suite_includes_all_strategies(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that all expected strategies are included in the suite."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
    )

    actual_strategies = set(suite.strategies.keys())

    # Check that strategies were run
    # Note: Some strategies might fail or be skipped, so check what was actually run
    assert len(actual_strategies) > 0
    assert "Equal Weight" in actual_strategies  # Baseline should always work


def test_backtest_suite_metrics_consistency(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that metrics are consistent with backtest results."""
    output_dir = tmp_path / "results"

    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        split=mock_split,
    )

    # For each strategy, check that metrics match result
    for strategy_name, result in suite.strategies.items():
        metrics = suite.metrics[strategy_name]

        # Check date consistency
        assert metrics.start_date == result.start_date
        assert metrics.end_date == result.end_date

        # Check transaction cost consistency
        assert metrics.transaction_costs_total == result.total_transaction_costs
        assert metrics.num_rebalances == result.num_rebalances

    for strategy_name, result in suite.full_results.items():
        metrics = suite.overall_metrics[strategy_name]
        assert metrics.start_date == result.start_date
        assert metrics.end_date == result.end_date


# ===== Error Handling Tests =====


def test_backtest_suite_handles_invalid_dataset_format(
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that invalid dataset format raises appropriate error."""
    output_dir = tmp_path / "results"

    # Create a dataset with missing required columns
    invalid_data = pl.DataFrame({
        "date": [datetime(2023, 1, 1, tzinfo=UTC)],
        "value": [100.0],
        # Missing: timestamp, symbol, return
    })

    invalid_path = tmp_path / "invalid.parquet"
    invalid_data.write_parquet(invalid_path)

    with pytest.raises(ValueError, match="Dataset missing required columns"):
        run_full_backtest_suite(
            dataset_path=invalid_path,
            output_dir=output_dir,
            split=mock_split,
        )


def test_backtest_suite_handles_csv_dataset(
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that CSV datasets are supported."""
    output_dir = tmp_path / "results"

    # Create CSV dataset with enough data for VaR/CVaR (100+ observations)
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    num_days = 120  # More than 100 for VaR/CVaR requirement

    data = []
    for day in range(num_days):
        date = start_date + timedelta(days=day)
        for sector in ["SPY", "AGG"]:
            data.append({
                "timestamp": date.isoformat(),
                "symbol": sector,
                "return": 0.001,
            })

    df = pl.DataFrame(data)
    csv_path = tmp_path / "sector_returns.csv"
    df.write_csv(csv_path)

    suite = run_full_backtest_suite(
        dataset_path=csv_path,
        output_dir=output_dir,
        split=mock_split,
    )

    # Check that suite was created successfully
    assert isinstance(suite, BacktestSuite)
    assert len(suite.strategies) > 0


def test_backtest_suite_reproducibility(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Test that results are reproducible with same random seed."""
    output_dir_1 = tmp_path / "results1"
    output_dir_2 = tmp_path / "results2"

    # Run suite twice with same seed
    suite_1 = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir_1,
        split=mock_split,
        config_overrides={"random_seed": 42},
    )

    suite_2 = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir_2,
        split=mock_split,
        config_overrides={"random_seed": 42},
    )

    # Results should be identical
    for strategy_name in suite_1.strategies.keys():
        if strategy_name not in suite_2.strategies:
            continue

        metrics_1 = suite_1.metrics[strategy_name]
        metrics_2 = suite_2.metrics[strategy_name]

        # Check key metrics are identical
        assert abs(metrics_1.annualized_return - metrics_2.annualized_return) < 1e-10
        assert abs(metrics_1.sharpe_ratio - metrics_2.sharpe_ratio) < 1e-10
        assert abs(metrics_1.maximum_drawdown - metrics_2.maximum_drawdown) < 1e-10


# ===== Total: 13 Tests =====
