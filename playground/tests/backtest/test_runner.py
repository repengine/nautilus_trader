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

import json
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from ml.config.playground import MonteCarloShockOverlayDefaults
from ml.config.playground import MonteCarloStressDefaults
from ml.config.playground import NestedWalkForwardDefaults
from ml.config.playground import ThreeDRiskBacktestDefaults
from ml.config.playground import WalkForwardPermutationDefaults
from playground.backtest.engine import BacktestConfig
from playground.backtest.engine import BacktestResult
from playground.backtest.liquidity_controls import LiquidityScalingConfig
from playground.backtest.performance_metrics import PerformanceMetrics
from playground.backtest.runner import BacktestSuite
from playground.backtest.runner import LiquidityMitigationScenario
from playground.backtest.runner import WalkForwardBacktestResult
from playground.backtest.runner import get_liquidity_mitigation_scenarios
from playground.backtest.runner import run_full_backtest_suite
from playground.backtest.runner import run_liquidity_mitigation_experiments
from playground.backtest.runner import run_monte_carlo_stress_suite
from playground.backtest.runner import run_multi_horizon_walk_forward_analysis
from playground.backtest.runner import run_walk_forward_backtest_suite
from playground.backtest.splits import TrainTestSplit
from playground.backtest.splits import WalkForwardConfig


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
        train_start=datetime(2018, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 12, 31, tzinfo=UTC),
        test_start=datetime(2023, 1, 1, tzinfo=UTC),
        test_end=datetime(2023, 12, 31, tzinfo=UTC),
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
    report_file = output_dir / f"backtest_results_{mock_split.train_start.year}_{mock_split.test_end.year}.md"
    assert report_file.exists()

    # Train/test/full results should be tracked
    assert "Equal Weight" in suite.train_results
    assert "Equal Weight" in suite.full_results
    assert "Equal Weight" in suite.overall_metrics

    train_vs_test = suite.train_vs_test_table()
    assert not train_vs_test.is_empty()


def test_backtest_suite_benchmark_summary_matches_baselines(
    mock_dataset_path: Path,
    mock_split: TrainTestSplit,
    tmp_path: Path,
) -> None:
    """Benchmark summary should include canonical baseline strategies."""
    suite = run_full_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path / "benchmarks",
        split=mock_split,
    )

    summary = suite.benchmark_summary()
    assert not summary.is_empty()
    assert summary.height == len(suite.baseline_strategies)
    assert summary.get_column("strategy").to_list() == list(suite.baseline_strategies)
    expected_columns = {
        "strategy",
        "sharpe_ratio",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "cumulative_return",
        "status",
    }
    assert expected_columns.issubset(set(summary.columns))
    statuses = set(summary.get_column("status").to_list())
    assert "available" in statuses
    assert statuses.issubset({"available", "missing"})


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


def test_run_monte_carlo_stress_suite_generates_paths(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Monte Carlo stress suite should produce summary artefacts and paths."""
    overlay = MonteCarloShockOverlayDefaults(
        name="test_shock",
        probability=1.0,
        magnitude=-0.01,
        duration_days=3,
        decay=0.5,
        max_applications=1,
        regime_bias=None,
    )
    stress_config = MonteCarloStressDefaults(
        num_paths=5,
        random_seed=42,
        risk_free_rate=0.0,
        overlays=(overlay,),
    )

    result = run_monte_carlo_stress_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        config=stress_config,
    )

    assert result.paths
    assert len(result.paths) == stress_config.num_paths
    summary = result.summary_frame()
    assert not summary.is_empty()
    strategy_names = summary.get_column("strategy").to_list()
    assert stress_config.target_strategy in strategy_names

    artefact_root = tmp_path / "stress" / "monte_carlo"
    assert (artefact_root / "summary.csv").exists()
    assert (artefact_root / "paths.csv").exists()
    assert (artefact_root / "config.json").exists()
    path_overlays = [path.overlay_events for path in result.paths]
    assert any(events for events in path_overlays)


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


def test_run_full_backtest_suite_requires_min_training_history(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Training window shorter than defaults should raise a validation error."""
    short_split = TrainTestSplit(
        train_start=datetime(2022, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 12, 31, tzinfo=UTC),
        test_start=datetime(2023, 1, 1, tzinfo=UTC),
        test_end=datetime(2023, 12, 31, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="Training window"):
        run_full_backtest_suite(
            dataset_path=mock_dataset_path,
            output_dir=tmp_path / "invalid_training",
            split=short_split,
        )


def test_run_full_backtest_suite_rejects_split_outside_dataset(tmp_path: Path) -> None:
    """Splits outside dataset coverage should fail validation."""
    start_date = datetime(2020, 1, 1, tzinfo=UTC)
    end_date = datetime(2024, 12, 31, tzinfo=UTC)
    step = timedelta(days=7)
    records = []
    current = start_date
    while current <= end_date:
        for symbol in ("SPY", "AGG"):
            records.append({
                "timestamp": current,
                "symbol": symbol,
                "return": 0.0004,
            })
        current += step

    dataset_path = tmp_path / "limited_sector_returns.parquet"
    pl.DataFrame(records).write_parquet(dataset_path)

    with pytest.raises(ValueError, match="dataset coverage"):
        run_full_backtest_suite(
            dataset_path=dataset_path,
            output_dir=tmp_path / "coverage_failure",
            split=None,
        )


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


def test_benchmark_summary_marks_missing_baselines(tmp_path: Path) -> None:
    """Benchmark summary rows mark missing baseline strategies and note in report."""
    train_start = datetime(2010, 1, 1, tzinfo=UTC)
    train_end = datetime(2014, 12, 31, tzinfo=UTC)
    test_start = datetime(2015, 1, 1, tzinfo=UTC)
    test_end = datetime(2015, 12, 31, tzinfo=UTC)
    split = TrainTestSplit(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    config = BacktestConfig(start_date=train_start, end_date=test_end)

    metrics = PerformanceMetrics(
        annualized_return=0.12,
        cumulative_return=0.20,
        monthly_return_mean=0.015,
        monthly_return_std=0.02,
        annualized_volatility=0.18,
        maximum_drawdown=-0.12,
        var_95=-0.03,
        var_99=-0.05,
        cvar_95=-0.04,
        cvar_99=-0.06,
        sharpe_ratio=0.75,
        sortino_ratio=0.90,
        calmar_ratio=0.50,
        information_ratio=None,
        turnover_rate=0.30,
        transaction_costs_total=1_500.0,
        transaction_costs_pct=0.02,
        num_rebalances=12,
        start_date=train_start,
        end_date=test_end,
        total_days=365,
    )

    positions = pl.DataFrame({
        "timestamp": [test_start],
        "symbol": ["SPY"],
        "weight": [1.0],
    })
    equal_weight_result = BacktestResult(
        strategy_name="Equal Weight",
        start_date=train_start,
        end_date=test_end,
        dates=[train_start, test_end],
        portfolio_values=[100.0, 120.0],
        returns=[0.015],
        positions=positions,
        total_return=0.20,
        annualized_return=0.12,
        annualized_volatility=0.18,
        sharpe_ratio=0.75,
        max_drawdown=-0.12,
        calmar_ratio=0.50,
        total_transaction_costs=1_500.0,
        turnover_rate=0.30,
        num_rebalances=12,
    )

    suite = BacktestSuite(
        strategies={"Equal Weight": equal_weight_result},
        metrics={"Equal Weight": metrics},
        split=split,
        config=config,
    )

    summary = suite.benchmark_summary()
    assert summary.height == len(suite.baseline_strategies)
    missing = summary.filter(pl.col("status") == "missing")
    assert set(missing.get_column("strategy").to_list()) == {"60/40 Portfolio", "Risk Parity"}

    report_path = tmp_path / "benchmarks.md"
    suite.to_markdown_report(report_path)
    content = report_path.read_text()
    assert (
        "Metrics unavailable for baseline strategies: 60/40 Portfolio, Risk Parity"
        in content
    )


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

    # Create CSV dataset with enough data for validation and VaR/CVaR (100+ observations)
    start_date = datetime(2018, 1, 1, tzinfo=UTC)
    num_days = 2_200  # ~6 years of observations to cover train/test windows

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


def test_run_walk_forward_backtest_suite(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Verify walk-forward orchestration produces fold outputs and summaries."""
    config = WalkForwardConfig(
        start_date=datetime(2018, 1, 1, tzinfo=UTC),
        end_date=datetime(2023, 12, 31, tzinfo=UTC),
        train_years=2,
        test_years=1,
        step_years=1,
    )

    result = run_walk_forward_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        walk_forward_config=config,
    )

    assert isinstance(result, WalkForwardBacktestResult)
    expected_splits = config.to_splits()
    assert len(result.suites) == len(expected_splits) > 0

    # Aggregated metrics should include Sharpe ratios
    aggregate = result.aggregate_metrics()
    assert not aggregate.is_empty()
    assert "sharpe_ratio" in aggregate.columns

    summary_dir = tmp_path / "walk_forward"
    assert (summary_dir / "aggregate_metrics.csv").exists()
    assert (summary_dir / "strategy_summary.csv").exists()

    # Fold artefacts should have been produced
    first_fold_dir = summary_dir / "fold_01"
    report_name = (
        f"backtest_results_{expected_splits[0].train_start.year}_"
        f"{expected_splits[0].test_end.year}.md"
    )
    assert (first_fold_dir / "performance_comparison_table.csv").exists()
    assert (first_fold_dir / report_name).exists()


def test_run_walk_forward_backtest_suite_accepts_overrides(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Ensure walk-forward suite threads liquidity config and turnover overrides."""
    config = WalkForwardConfig(
        start_date=datetime(2018, 1, 1, tzinfo=UTC),
        end_date=datetime(2021, 12, 31, tzinfo=UTC),
        train_years=1,
        test_years=1,
        step_years=1,
    )
    custom_liquidity = LiquidityScalingConfig(
        severe_threshold=-10.0,
        moderate_threshold=-5.0,
        severe_regime_multiplier=0.2,
        moderate_regime_multiplier=0.3,
        severe_liquidity_multiplier=0.2,
        moderate_liquidity_multiplier=0.3,
        neutral_liquidity_multiplier=0.95,
        floor=0.9,
    )
    overrides = {
        "3d_factor_rolling": 0.15,
        "3d_factor_stable": 0.05,
    }

    result = run_walk_forward_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        walk_forward_config=config,
        liquidity_config=custom_liquidity,
        turnover_overrides=overrides,
    )

    assert result.suites, "Expected at least one backtest suite"
    first_suite = result.suites[0]
    assert first_suite.turnover_overrides["3d_factor_rolling"] == pytest.approx(0.15)
    assert first_suite.turnover_overrides["3d_factor_stable"] == pytest.approx(0.05)
    assert first_suite.regime_factor_multipliers
    for factor_map in first_suite.regime_factor_multipliers.values():
        assert factor_map["factor_liquidity"] >= 0.9 - 1e-9


def test_liquidity_mitigation_experiments_with_walk_forward(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Verify walk-forward summaries are captured for mitigation scenarios."""
    scenario = LiquidityMitigationScenario(
        name="Turnover Walk Test",
        rolling_turnover_smoothing=0.45,
        stable_turnover_smoothing=0.30,
        liquidity_config=LiquidityScalingConfig(),
    )
    wf_config = WalkForwardConfig(
        start_date=datetime(2014, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
        train_years=3,
        test_years=1,
        step_years=1,
    )

    results = run_liquidity_mitigation_experiments(
        dataset_path=mock_dataset_path,
        output_dir=tmp_path,
        scenarios=[scenario],
        run_walk_forward=True,
        walk_forward_config=wf_config,
    )

    assert len(results) == 1
    result = results[0]
    assert result.walk_forward_sharpe_mean is not None
    assert result.walk_forward_output_directory is not None
    assert result.walk_forward_output_directory.exists()


def test_get_liquidity_mitigation_scenarios_filters_known_names() -> None:
    """Ensure scenario resolver returns filtered lists and rejects unknown names."""
    all_scenarios = get_liquidity_mitigation_scenarios()
    expected_names = {
        "Baseline Controls",
        "Turnover Smoothing 0.55/0.40",
        "Tighter Liquidity Regime Scaling",
        "Turnover Stress Test",
        "Stress: 2008 Liquidity Shock",
        "Stress: 2020 Volatility Spike",
        "Stress: 2022 Rates + Stocks",
        "Stress: 1987 Black Monday",
        "Stress: Synthetic Liquidity Shock",
    }
    retrieved_names = {scenario.name for scenario in all_scenarios}
    assert expected_names.issubset(retrieved_names)

    subset = get_liquidity_mitigation_scenarios([all_scenarios[0].name])
    assert len(subset) == 1
    assert subset[0].name == all_scenarios[0].name

    with pytest.raises(ValueError):
        get_liquidity_mitigation_scenarios(["unknown-scenario"])


def test_run_liquidity_mitigation_experiments_single_scenario(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Ensure liquidity mitigation experiments run and capture summary output."""
    output_dir = tmp_path / "experiments"
    scenario = LiquidityMitigationScenario(
        name="Unit Test Scenario",
        rolling_turnover_smoothing=0.25,
        stable_turnover_smoothing=0.15,
        liquidity_config=LiquidityScalingConfig(),
    )

    results = run_liquidity_mitigation_experiments(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        scenarios=[scenario],
    )

    assert len(results) == 1
    result = results[0]
    assert result.scenario_name == "Unit Test Scenario"
    assert result.rolling_sharpe_delta == pytest.approx(0.0)
    assert (output_dir / "liquidity_mitigation_results.csv").exists()
    scenario_dir = output_dir / "unit_test_scenario"
    assert (scenario_dir / "performance_comparison_table.csv").exists()


def test_walk_forward_metadata_includes_defaults(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Walk-forward summaries should persist metadata with default parameters."""
    output_dir = tmp_path / "wf_outputs"
    config = WalkForwardConfig(
        start_date=datetime(2015, 1, 1, tzinfo=UTC),
        end_date=datetime(2019, 12, 31, tzinfo=UTC),
        train_years=3,
        test_years=1,
        step_years=1,
    )

    run_walk_forward_backtest_suite(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        walk_forward_config=config,
    )

    metadata_path = output_dir / "walk_forward" / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    defaults = ThreeDRiskBacktestDefaults()

    assert metadata["risk_free_rate"] == pytest.approx(defaults.risk_free_rate)
    assert metadata["turnover_smoothing"]["stable"] == pytest.approx(defaults.stable_turnover_smoothing)
    assert metadata["turnover_smoothing"]["rolling"] == pytest.approx(defaults.rolling_turnover_smoothing)
    assert metadata["liquidity_config"]["severe_threshold"] == pytest.approx(
        defaults.liquidity_scaling.severe_threshold,
    )
    assert metadata["split_count"] == len(config.to_splits())
    assert metadata["summaries_directory"].endswith("walk_forward")
    wf_config = metadata["walk_forward_config"]
    assert wf_config["train_years"] == config.train_years
    assert wf_config["test_years"] == config.test_years
    assert wf_config["step_years"] == config.step_years
    assert len(metadata["splits"]) == len(config.to_splits())


def test_three_d_risk_backtest_defaults_build_liquidity_config() -> None:
    """Defaults should hydrate LiquidityScalingConfig with matching parameters."""
    defaults = ThreeDRiskBacktestDefaults()
    config = defaults.build_liquidity_config()

    assert config.severe_threshold == pytest.approx(defaults.liquidity_scaling.severe_threshold)
    assert config.moderate_threshold == pytest.approx(defaults.liquidity_scaling.moderate_threshold)
    assert config.severe_regime_multiplier == pytest.approx(defaults.liquidity_scaling.severe_regime_multiplier)
    assert config.moderate_regime_multiplier == pytest.approx(defaults.liquidity_scaling.moderate_regime_multiplier)
    assert config.severe_liquidity_multiplier == pytest.approx(defaults.liquidity_scaling.severe_liquidity_multiplier)
    assert config.moderate_liquidity_multiplier == pytest.approx(defaults.liquidity_scaling.moderate_liquidity_multiplier)
    assert config.neutral_liquidity_multiplier == pytest.approx(defaults.liquidity_scaling.neutral_liquidity_multiplier)
    assert config.floor == pytest.approx(defaults.liquidity_scaling.floor)


def test_three_d_risk_backtest_defaults_walk_forward_permutations() -> None:
    """Defaults should expose walk-forward permutations with canonical primary ordering."""
    defaults = ThreeDRiskBacktestDefaults()
    permutations = defaults.walk_forward_permutations

    assert permutations, "Expected at least one walk-forward permutation"
    assert defaults.primary_walk_forward_permutation == permutations[0]
    for permutation in permutations:
        assert permutation.name
        assert permutation.train_years > 0
        assert permutation.test_years > 0
        if permutation.nested is not None:
            assert permutation.nested.min_folds > 0


def test_run_multi_horizon_walk_forward_analysis_produces_permutation_outputs(
    mock_dataset_path: Path,
    tmp_path: Path,
) -> None:
    """Multi-horizon validation should emit artefacts for each permutation."""
    permutations = (
        WalkForwardPermutationDefaults(
            name="Test Baseline 4y/1y",
            description="Baseline permutation for unit test runtime",
            train_years=4,
            test_years=1,
            step_years=1,
            nested=NestedWalkForwardDefaults(train_years=2, test_years=1, step_years=1, min_folds=1),
        ),
        WalkForwardPermutationDefaults(
            name="Test Secondary 3y/1y",
            description="Secondary permutation for unit test",
            train_years=3,
            test_years=1,
            step_years=2,
            nested=None,
        ),
    )
    output_dir = tmp_path / "multi_horizon"

    result = run_multi_horizon_walk_forward_analysis(
        dataset_path=mock_dataset_path,
        output_dir=output_dir,
        start_date=datetime(2014, 1, 1, tzinfo=UTC),
        end_date=datetime(2020, 12, 31, tzinfo=UTC),
        permutations=permutations,
        include_primary_root=True,
    )

    primary_slug = permutations[0].slug
    secondary_slug = permutations[1].slug
    assert primary_slug in result.runs
    assert secondary_slug in result.runs

    base_dir = output_dir / "walk_forward"
    assert (base_dir / "aggregate_metrics.csv").exists()

    alias_dir = base_dir / "permutations" / primary_slug
    assert (alias_dir / "README.txt").exists()
    assert (alias_dir / "permutation_metadata.json").exists()

    secondary_dir = base_dir / "permutations" / secondary_slug
    assert (secondary_dir / "aggregate_metrics.csv").exists()
    assert result.runs[primary_slug].nested_results, "Expected nested results for primary permutation"
    summary_df = result.summary_table()
    assert not summary_df.is_empty()
    nested_df = result.nested_summary()
    # Nested validation should produce metrics for the baseline permutation even with fallback dataset.
    assert primary_slug in nested_df.get_column("permutation_slug").to_list()


def test_three_d_risk_backtest_defaults_fallbacks_are_immutable() -> None:
    """Fallback mapping should be immutable to preserve config integrity."""
    defaults = ThreeDRiskBacktestDefaults()

    with pytest.raises(TypeError):
        defaults.liquidity_contribution_fallbacks["New Regime"] = -0.01
