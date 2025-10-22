"""
Tests for parameter sensitivity analysis module.

This test suite validates the sensitivity analysis framework for backtesting
strategies, including single parameter sensitivity, grid search optimization,
and stability analysis.

Test Coverage:
- Sensitivity calculation (3 tests)
- Grid search (3 tests)
- Stability analysis (2 tests)
- Integration tests (2 tests)
- Edge cases (2 tests)
- Reporting (1 test)

Total: 13 tests (exceeds minimum requirement of 12)
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.config.playground import ThreeDRiskBacktestDefaults
from playground.backtest.sensitivity import COMPREHENSIVE_GRID
from playground.backtest.sensitivity import STANDARD_GRIDS
from playground.backtest.sensitivity import ParameterConfig
from playground.backtest.sensitivity import analyze_parameter_stability
from playground.backtest.sensitivity import compare_strategies_sensitivity
from playground.backtest.sensitivity import generate_sensitivity_report
from playground.backtest.sensitivity import run_grid_search
from playground.backtest.sensitivity import run_parameter_sensitivity
from playground.backtest.splits import TrainTestSplit


# ===== Fixtures =====


@pytest.fixture
def mock_dataset_path(tmp_path: Path) -> Path:
    """Create a mock sector dataset for testing."""
    # Generate 2 years of daily data
    start_date = datetime(2022, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(500)]

    # Create data for 3 sectors with different characteristics
    sectors = ["XLK", "XLV", "XLU"]
    rows = []

    np.random.seed(42)
    for date in dates:
        for sector in sectors:
            # Generate returns with different volatilities
            if sector == "XLK":
                ret = np.random.normal(0.0005, 0.015)  # High vol tech
            elif sector == "XLV":
                ret = np.random.normal(0.0004, 0.010)  # Medium vol healthcare
            else:
                ret = np.random.normal(0.0003, 0.008)  # Low vol utilities

            rows.append({
                "timestamp": date,
                "symbol": sector,
                "return": ret,
            })

    df = pl.DataFrame(rows)

    # Save to parquet
    dataset_path = tmp_path / "test_sectors.parquet"
    df.write_parquet(dataset_path)

    return dataset_path


@pytest.fixture
def test_split() -> TrainTestSplit:
    """Create a test train/test split."""
    return TrainTestSplit(
        train_start=datetime(2022, 1, 1, tzinfo=UTC),
        train_end=datetime(2022, 12, 31, tzinfo=UTC),
        test_start=datetime(2023, 1, 1, tzinfo=UTC),
        test_end=datetime(2023, 6, 30, tzinfo=UTC),
    )


# ===== Sensitivity Calculation Tests (3 tests) =====


def test_single_parameter_sensitivity(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test single parameter sensitivity analysis."""
    result = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="transaction_cost_bps",
        parameter_values=[5.0, 10.0, 20.0],
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    # Validate structure
    assert result.parameter_name == "transaction_cost_bps"
    assert len(result.parameter_values) == 3
    assert len(result.sharpe_ratios) == 3
    assert len(result.calmar_ratios) == 3
    assert len(result.max_drawdowns) == 3
    assert len(result.annualized_returns) == 3

    # Validate optimal value is one of the tested values
    assert result.optimal_value in result.parameter_values

    # Validate Sharpe ratios decrease with higher transaction costs
    # (higher costs should reduce performance)
    assert result.sharpe_ratios[0] >= result.sharpe_ratios[-1]


def test_sensitivity_metrics_calculation(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test sensitivity metric calculation (range, std, is_sensitive)."""
    result = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="transaction_cost_bps",
        parameter_values=[0.0, 10.0, 20.0],
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    # Sharpe range should be non-negative
    assert result.sharpe_range >= 0.0

    # Sharpe std should be non-negative
    assert result.sharpe_std >= 0.0

    # is_sensitive is a boolean
    assert isinstance(result.is_sensitive, bool)

    # Sharpe range should equal max - min
    expected_range = max(result.sharpe_ratios) - min(result.sharpe_ratios)
    assert abs(result.sharpe_range - expected_range) < 1e-10


def test_optimal_value_identification(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test that optimal value corresponds to highest Sharpe ratio."""
    result = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="rebalance_frequency",
        parameter_values=["weekly", "monthly"],
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    # Find index of max Sharpe
    max_sharpe_idx = result.sharpe_ratios.index(max(result.sharpe_ratios))
    expected_optimal = result.parameter_values[max_sharpe_idx]

    # Verify optimal value matches
    assert result.optimal_value == expected_optimal
    assert result.optimal_sharpe == max(result.sharpe_ratios)

    # Verify optimal_rank is 0 (best)
    assert result.optimal_rank == 0


def test_parameter_sensitivity_custom_risk_free(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Custom risk-free rates should change computed Sharpe ratios."""
    defaults = ThreeDRiskBacktestDefaults()
    baseline = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="transaction_cost_bps",
        parameter_values=[5.0, 10.0],
        dataset_path=mock_dataset_path,
        split=test_split,
    )
    custom_rate = defaults.risk_free_rate + 0.01
    adjusted = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="transaction_cost_bps",
        parameter_values=[5.0, 10.0],
        dataset_path=mock_dataset_path,
        split=test_split,
        risk_free_rate=custom_rate,
    )

    assert not np.allclose(baseline.sharpe_ratios, adjusted.sharpe_ratios)


# ===== Grid Search Tests (3 tests) =====


def test_small_grid_search(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test grid search with small grid (2x2 = 4 combinations)."""
    grid = {
        "transaction_cost_bps": [5.0, 10.0],
        "rebalance_frequency": ["weekly", "monthly"],
    }

    result = run_grid_search(
        strategy_name="equal_weight",
        parameter_grid=grid,
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    # Should have 4 rows (2x2 combinations)
    assert len(result.results_table) == 4

    # Should have best config
    assert "transaction_cost_bps" in result.best_config
    assert "rebalance_frequency" in result.best_config

    # Best config values should be from the grid
    assert result.best_config["transaction_cost_bps"] in grid["transaction_cost_bps"]
    assert result.best_config["rebalance_frequency"] in grid["rebalance_frequency"]

    # Should have sensitivity summary for each parameter
    assert "transaction_cost_bps" in result.sensitivity_summary
    assert "rebalance_frequency" in result.sensitivity_summary


def test_grid_search_custom_risk_free(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Custom risk-free rates should affect grid-search Sharpe scores."""
    defaults = ThreeDRiskBacktestDefaults()
    grid = {
        "transaction_cost_bps": [5.0, 10.0],
        "rebalance_frequency": ["weekly", "monthly"],
    }

    baseline = run_grid_search(
        strategy_name="equal_weight",
        parameter_grid=grid,
        dataset_path=mock_dataset_path,
        split=test_split,
        risk_free_rate=defaults.risk_free_rate,
    )
    adjusted = run_grid_search(
        strategy_name="equal_weight",
        parameter_grid=grid,
        dataset_path=mock_dataset_path,
        split=test_split,
        risk_free_rate=defaults.risk_free_rate + 0.01,
    )

    assert not np.isclose(baseline.best_sharpe, adjusted.best_sharpe)


def test_grid_search_three_parameters(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test grid search with 3 parameters."""
    grid = {
        "transaction_cost_bps": [5.0, 10.0],
        "rebalance_frequency": ["weekly", "monthly"],
        "initial_capital": [500_000.0, 1_000_000.0],
    }

    result = run_grid_search(
        strategy_name="equal_weight",
        parameter_grid=grid,
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    # Should have 8 rows (2x2x2 combinations)
    assert len(result.results_table) == 8

    # All parameters should be in results table
    assert "transaction_cost_bps" in result.results_table.columns
    assert "rebalance_frequency" in result.results_table.columns
    assert "initial_capital" in result.results_table.columns

    # All performance metrics should be present
    assert "sharpe_ratio" in result.results_table.columns
    assert "calmar_ratio" in result.results_table.columns
    assert "annualized_return" in result.results_table.columns


def test_optimal_configuration_identification(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test that optimal configuration is correctly identified."""
    grid = {
        "transaction_cost_bps": [5.0, 15.0],
        "rebalance_frequency": ["weekly", "monthly"],
    }

    result = run_grid_search(
        strategy_name="equal_weight",
        parameter_grid=grid,
        dataset_path=mock_dataset_path,
        split=test_split,
        optimization_metric="sharpe_ratio",
    )

    # Find best row manually
    best_row = result.results_table.sort("sharpe_ratio", descending=True).head(1)

    # Verify best_config matches the row with highest Sharpe
    assert result.best_config["transaction_cost_bps"] == best_row["transaction_cost_bps"][0]
    assert result.best_config["rebalance_frequency"] == best_row["rebalance_frequency"][0]
    assert result.best_sharpe == best_row["sharpe_ratio"][0]


# ===== Stability Analysis Tests (2 tests) =====


def test_stable_parameter_identification(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test identification of stable parameters."""
    # Create sensitivity results with one stable and one unstable parameter
    result1 = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="initial_capital",  # Should be stable (doesn't affect ratios)
        parameter_values=[500_000.0, 1_000_000.0, 2_000_000.0],
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    result2 = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="transaction_cost_bps",  # May be unstable
        parameter_values=[0.0, 10.0, 30.0],
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    sensitivity_results = {
        "initial_capital": result1,
        "transaction_cost_bps": result2,
    }

    stability = analyze_parameter_stability(sensitivity_results, stability_threshold=0.10)

    # Should have stability assessment for both parameters
    assert "initial_capital" in stability
    assert "transaction_cost_bps" in stability

    # Both should be boolean
    assert isinstance(stability["initial_capital"], bool)
    assert isinstance(stability["transaction_cost_bps"], bool)


def test_unstable_parameter_detection(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test detection of unstable (sensitive) parameters."""
    # Create a parameter with wide range of values (should be unstable)
    result = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="transaction_cost_bps",
        parameter_values=[0.0, 50.0],  # Wide range should cause instability
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    # Manually check if it's marked as sensitive
    # (Sharpe range should be > 0.10 threshold)
    if result.sharpe_range > 0.10:
        assert result.is_sensitive is True


# ===== Integration Tests (2 tests) =====


def test_integration_with_backtest_engine(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test integration with real backtest engine."""
    # Run a complete sensitivity analysis
    result = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="transaction_cost_bps",
        parameter_values=[5.0, 10.0],
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    # Should produce valid results
    assert len(result.sharpe_ratios) == 2
    assert all(isinstance(sr, float) for sr in result.sharpe_ratios)
    assert all(isinstance(cr, float) for cr in result.calmar_ratios)

    # Summary table should be valid
    df = result.summary_table()
    assert len(df) == 2
    assert "parameter_value" in df.columns
    assert "sharpe_ratio" in df.columns


def test_strategy_comparison(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test comparison of sensitivity across strategies."""
    # Create sensitivity results for two different strategies
    result1 = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="transaction_cost_bps",
        parameter_values=[5.0, 10.0],
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    # For now, use equal_weight again (other strategies would be added later)
    result2 = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="transaction_cost_bps",
        parameter_values=[5.0, 10.0],
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    strategy_sensitivity = {
        "strategy1": {"transaction_cost_bps": result1},
        "strategy2": {"transaction_cost_bps": result2},
    }

    comparison = compare_strategies_sensitivity(strategy_sensitivity)

    # Should have 2 rows (one per strategy)
    assert len(comparison) == 2
    assert "strategy_name" in comparison.columns
    assert "parameter_name" in comparison.columns
    assert "sharpe_range" in comparison.columns
    assert "is_sensitive" in comparison.columns


# ===== Edge Cases (2 tests) =====


def test_single_parameter_value(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test with single parameter value (no variation)."""
    result = run_parameter_sensitivity(
        strategy_name="equal_weight",
        parameter_name="transaction_cost_bps",
        parameter_values=[10.0],  # Only one value
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    # Should still work
    assert len(result.sharpe_ratios) == 1
    assert result.optimal_value == 10.0

    # Sensitivity metrics should be zero (no variation)
    assert result.sharpe_range == 0.0
    assert result.sharpe_std == 0.0
    assert result.is_sensitive is False


def test_large_grid_no_errors(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test that large grid completes without errors."""
    # Create a moderately large grid
    grid = {
        "transaction_cost_bps": [5.0, 10.0, 15.0],
        "rebalance_frequency": ["weekly", "monthly"],
    }

    # Should complete without errors (6 combinations)
    result = run_grid_search(
        strategy_name="equal_weight",
        parameter_grid=grid,
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    assert len(result.results_table) == 6
    assert result.best_config is not None


# ===== Reporting Tests (1 test) =====


def test_report_generation(mock_dataset_path: Path, test_split: TrainTestSplit, tmp_path: Path) -> None:
    """Test markdown report generation."""
    # Run a grid search
    grid = {
        "transaction_cost_bps": [5.0, 10.0],
        "rebalance_frequency": ["weekly", "monthly"],
    }

    result = run_grid_search(
        strategy_name="equal_weight",
        parameter_grid=grid,
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    # Generate report
    report_path = tmp_path / "test_sensitivity_report.md"
    generate_sensitivity_report(
        grid_result=result,
        output_path=report_path,
        strategy_name="Test Strategy",
    )

    # Verify report was created
    assert report_path.exists()

    # Read and validate content
    content = report_path.read_text()
    assert "# Parameter Sensitivity Analysis: Test Strategy" in content
    assert "Executive Summary" in content
    assert "Optimal Configuration" in content
    assert "Parameter Sensitivity Results" in content
    assert "Grid Search Results" in content
    assert "Recommendations" in content


# ===== Additional Validation Tests =====


def test_parameter_config_validation() -> None:
    """Test ParameterConfig validation."""
    # Valid config
    config = ParameterConfig(
        parameter_name="transaction_cost_bps",
        parameter_value=10.0,
        description="Transaction cost in basis points",
    )
    assert config.parameter_name == "transaction_cost_bps"

    # Invalid config (empty name)
    with pytest.raises(ValueError, match="parameter_name cannot be empty"):
        ParameterConfig(
            parameter_name="",
            parameter_value=10.0,
            description="Test",
        )


def test_grid_search_top_k_configs(mock_dataset_path: Path, test_split: TrainTestSplit) -> None:
    """Test get_top_k_configs method."""
    grid = {
        "transaction_cost_bps": [5.0, 10.0, 15.0],
        "rebalance_frequency": ["weekly", "monthly"],
    }

    result = run_grid_search(
        strategy_name="equal_weight",
        parameter_grid=grid,
        dataset_path=mock_dataset_path,
        split=test_split,
    )

    # Get top 3 configs
    top_3 = result.get_top_k_configs(k=3, metric="sharpe_ratio")
    assert len(top_3) == 3

    # Should be sorted by Sharpe ratio (descending)
    sharpe_values = top_3["sharpe_ratio"].to_list()
    assert sharpe_values == sorted(sharpe_values, reverse=True)


def test_standard_grids_defined() -> None:
    """Test that standard parameter grids are properly defined."""
    # Transaction costs grid
    assert "transaction_costs" in STANDARD_GRIDS
    assert "transaction_cost_bps" in STANDARD_GRIDS["transaction_costs"]
    assert len(STANDARD_GRIDS["transaction_costs"]["transaction_cost_bps"]) == 5

    # Rebalancing grid
    assert "rebalancing" in STANDARD_GRIDS
    assert "rebalance_frequency" in STANDARD_GRIDS["rebalancing"]
    assert "monthly" in STANDARD_GRIDS["rebalancing"]["rebalance_frequency"]
    assert "weekly" in STANDARD_GRIDS["rebalancing"]["rebalance_frequency"]

    # Comprehensive grid
    assert "transaction_cost_bps" in COMPREHENSIVE_GRID
    assert "rebalance_frequency" in COMPREHENSIVE_GRID


def test_invalid_dataset_path() -> None:
    """Test error handling for invalid dataset path."""
    with pytest.raises(FileNotFoundError):
        run_parameter_sensitivity(
            strategy_name="equal_weight",
            parameter_name="transaction_cost_bps",
            parameter_values=[10.0],
            dataset_path=Path("/nonexistent/path.parquet"),
        )


def test_empty_parameter_values(mock_dataset_path: Path) -> None:
    """Test error handling for empty parameter values."""
    with pytest.raises(ValueError, match="parameter_values cannot be empty"):
        run_parameter_sensitivity(
            strategy_name="equal_weight",
            parameter_name="transaction_cost_bps",
            parameter_values=[],  # Empty list
            dataset_path=mock_dataset_path,
        )
