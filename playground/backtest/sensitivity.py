"""
Parameter sensitivity analysis for backtesting strategies.

This module provides a comprehensive framework for testing the sensitivity of
backtest results to key parameter choices, identifying robust parameter ranges,
and performing grid search optimization.

Key Features:
- Single parameter sensitivity analysis
- Multi-parameter grid search optimization
- Parameter stability assessment
- Strategy comparison across parameter ranges
- Markdown report generation
- Reproducible with fixed random seed

Performance Targets (Cold Path):
- Single parameter test (5 values): < 30 seconds
- Grid search (50 combinations): < 5 minutes
- Report generation: < 10 seconds

Hot/Cold Path Separation:
- This is a cold-path module (offline analysis)
- No real-time constraints, optimized for correctness

Integration Notes:
- Compatible with BacktestEngine and BacktestRunner
- Uses standard parameter grids for common parameters
- Follows Phase 4.1.1 requirements from 3D_Risk_Model_Roadmap.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import structlog

from ml.config.playground import ThreeDRiskBacktestDefaults
from playground.backtest.engine import BacktestConfig
from playground.backtest.engine import FactorBacktester
from playground.backtest.performance_metrics import calculate_performance_metrics
from playground.backtest.splits import TrainTestSplit
from playground.backtest.splits import define_train_test_split


if TYPE_CHECKING:
    from playground.backtest.engine import BacktestResult
    from playground.risk_model.dataset import SectorDataset


LOGGER = structlog.get_logger(__name__)


# ===== Constants =====

PLAYGROUND_DEFAULTS = ThreeDRiskBacktestDefaults()

# Standard parameter grids for common parameters
STANDARD_GRIDS = {
    "transaction_costs": {
        "transaction_cost_bps": [0.0, 5.0, 10.0, 15.0, 20.0],
    },
    "rebalancing": {
        "rebalance_frequency": ["daily", "weekly", "monthly"],
    },
    "rolling_window": {
        "rolling_window_days": [63, 126, 252, 504],  # 3mo, 6mo, 1yr, 2yr
    },
}

# Comprehensive grid for full analysis
COMPREHENSIVE_GRID = {
    "transaction_cost_bps": [5.0, 10.0, 20.0],
    "rebalance_frequency": ["weekly", "monthly"],
}


# ===== Sensitivity Analysis Dataclasses =====


@dataclass(slots=True, frozen=True)
class ParameterConfig:
    """Single parameter configuration for sensitivity testing."""

    parameter_name: str
    parameter_value: Any
    description: str  # Human-readable description

    def __post_init__(self) -> None:
        """Validate parameter configuration."""
        if not self.parameter_name:
            msg = "parameter_name cannot be empty"
            raise ValueError(msg)


@dataclass(slots=True)
class SensitivityResult:
    """Results from sensitivity analysis for a single parameter."""

    parameter_name: str
    parameter_values: list[Any]  # Tested values
    sharpe_ratios: list[float]  # Sharpe for each value
    calmar_ratios: list[float]  # Calmar for each value
    max_drawdowns: list[float]  # Max DD for each value
    annualized_returns: list[float]  # Returns for each value

    # Optimal parameter (by Sharpe ratio)
    optimal_value: Any
    optimal_sharpe: float

    # Sensitivity metrics
    sharpe_range: float  # Max - Min Sharpe
    sharpe_std: float  # Standard deviation of Sharpe across values
    is_sensitive: bool  # True if Sharpe range > threshold

    def summary_table(self) -> pl.DataFrame:
        """
        Generate summary table of parameter sensitivity.

        Returns
        -------
        pl.DataFrame
            Table with parameter values and corresponding metrics

        Examples
        --------
        >>> result = SensitivityResult(...)
        >>> df = result.summary_table()
        >>> print(df)
        """
        return pl.DataFrame({
            "parameter_value": self.parameter_values,
            "sharpe_ratio": self.sharpe_ratios,
            "calmar_ratio": self.calmar_ratios,
            "max_drawdown": self.max_drawdowns,
            "annualized_return": self.annualized_returns,
        })

    @property
    def optimal_rank(self) -> int:
        """
        Rank of optimal value (0 = best).

        Returns
        -------
        int
            Rank of optimal value in sorted list by Sharpe ratio
        """
        sorted_sharpe = sorted(enumerate(self.sharpe_ratios), key=lambda x: x[1], reverse=True)
        optimal_idx = self.parameter_values.index(self.optimal_value)
        for rank, (idx, _) in enumerate(sorted_sharpe):
            if idx == optimal_idx:
                return rank
        return len(self.parameter_values)


@dataclass(slots=True)
class GridSearchResult:
    """Results from full grid search across multiple parameters."""

    parameter_grid: dict[str, list[Any]]  # Parameter name -> values tested
    results_table: pl.DataFrame  # All combinations with metrics

    # Best configuration
    best_config: dict[str, Any]
    best_sharpe: float
    best_metrics: dict[str, float]  # All performance metrics

    # Sensitivity summary
    sensitivity_summary: dict[str, SensitivityResult]

    def get_top_k_configs(self, k: int = 10, metric: str = "sharpe_ratio") -> pl.DataFrame:
        """
        Get top K parameter configurations by specified metric.

        Parameters
        ----------
        k : int, default 10
            Number of top configurations to return
        metric : str, default "sharpe_ratio"
            Metric to sort by

        Returns
        -------
        pl.DataFrame
            Top K configurations sorted by metric (descending)

        Examples
        --------
        >>> grid_result = GridSearchResult(...)
        >>> top_10 = grid_result.get_top_k_configs(k=10)
        >>> print(top_10)
        """
        return (
            self.results_table
            .sort(metric, descending=True)
            .head(k)
        )

    def interaction_heatmap_data(
        self,
        param1: str,
        param2: str,
        metric: str = "sharpe_ratio",
    ) -> pl.DataFrame:
        """
        Get heatmap data for interaction between two parameters.

        Returns pivot table with param1 as rows, param2 as columns.

        Parameters
        ----------
        param1 : str
            First parameter name
        param2 : str
            Second parameter name
        metric : str, default "sharpe_ratio"
            Metric to display in heatmap

        Returns
        -------
        pl.DataFrame
            Pivot table for heatmap visualization

        Examples
        --------
        >>> heatmap_data = grid_result.interaction_heatmap_data(
        ...     "transaction_cost_bps",
        ...     "rebalance_frequency",
        ...     metric="sharpe_ratio"
        ... )
        """
        if param1 not in self.results_table.columns or param2 not in self.results_table.columns:
            msg = f"Parameters {param1} or {param2} not in results table"
            raise ValueError(msg)

        # Create pivot table using polars pivot_table API
        pivot = self.results_table.pivot(
            on=param2,
            index=param1,
            values=metric,
        )

        return pivot


# ===== Core Sensitivity Analysis Functions =====


def run_parameter_sensitivity(
    strategy_name: str,
    parameter_name: str,
    parameter_values: list[Any],
    dataset_path: Path,
    base_config: dict[str, Any] | None = None,
    split: TrainTestSplit | None = None,
    *,
    risk_free_rate: float | None = None,
) -> SensitivityResult:
    """
    Test sensitivity to a single parameter.

    Runs backtest for each parameter value while holding others constant.

    Parameters
    ----------
    strategy_name : str
        Strategy to test (e.g., "risk_parity", "equal_weight")
    parameter_name : str
        Parameter to vary (e.g., "transaction_cost_bps", "rebalance_frequency")
    parameter_values : list[Any]
        Values to test for the parameter
    dataset_path : Path
        Path to sector dataset
    base_config : dict | None
        Base configuration (other parameters held constant)
    split : TrainTestSplit | None
        Train/test split (defaults to standard 2010-2018/2019-2024)
    risk_free_rate : float | None, default None
        Annual risk-free rate used when computing Sharpe/Sortino metrics. Defaults to
        ``ThreeDRiskBacktestDefaults().risk_free_rate``.

    Returns
    -------
    SensitivityResult
        Sensitivity analysis for the parameter

    Raises
    ------
    ValueError
        If parameter_values is empty or dataset not found
    FileNotFoundError
        If dataset_path does not exist

    Examples
    --------
    >>> result = run_parameter_sensitivity(
    ...     strategy_name="risk_parity",
    ...     parameter_name="transaction_cost_bps",
    ...     parameter_values=[5.0, 10.0, 15.0, 20.0],
    ...     dataset_path=Path("data/sectors.parquet"),
    ... )
    >>> print(f"Optimal: {result.optimal_value} bps")
    >>> print(f"Sensitive: {result.is_sensitive}")
    """
    if not dataset_path.exists():
        msg = f"Dataset path does not exist: {dataset_path}"
        raise FileNotFoundError(msg)

    if not parameter_values:
        msg = "parameter_values cannot be empty"
        raise ValueError(msg)

    LOGGER.info(
        "Starting parameter sensitivity analysis",
        strategy=strategy_name,
        parameter=parameter_name,
        num_values=len(parameter_values),
    )

    # Load dataset
    dataset = _load_dataset(dataset_path)

    # Define split
    if split is None:
        split = define_train_test_split()

    # Base configuration
    if base_config is None:
        base_config = {}

    resolved_risk_free_rate = (
        risk_free_rate if risk_free_rate is not None else PLAYGROUND_DEFAULTS.risk_free_rate
    )

    # Run backtests for each parameter value
    sharpe_ratios: list[float] = []
    calmar_ratios: list[float] = []
    max_drawdowns: list[float] = []
    annualized_returns: list[float] = []

    for value in parameter_values:
        LOGGER.debug(
            "Testing parameter value",
            parameter=parameter_name,
            value=value,
        )

        # Create config with this parameter value
        config = base_config.copy()
        config[parameter_name] = value

        # Run backtest
        result = _run_backtest(
            dataset=dataset,
            strategy_name=strategy_name,
            config=config,
            split=split,
        )

        # Calculate metrics
        metrics = calculate_performance_metrics(result, risk_free_rate=resolved_risk_free_rate)

        sharpe_ratios.append(metrics.sharpe_ratio)
        calmar_ratios.append(metrics.calmar_ratio)
        max_drawdowns.append(metrics.maximum_drawdown)
        annualized_returns.append(metrics.annualized_return)

    # Find optimal value (by Sharpe ratio)
    optimal_idx = int(np.argmax(sharpe_ratios))
    optimal_value = parameter_values[optimal_idx]
    optimal_sharpe = sharpe_ratios[optimal_idx]

    # Compute sensitivity metrics
    sharpe_range = float(np.max(sharpe_ratios) - np.min(sharpe_ratios))
    sharpe_std = float(np.std(sharpe_ratios, ddof=1)) if len(sharpe_ratios) > 1 else 0.0

    # Sensitivity threshold (Sharpe range > 0.10 indicates sensitivity)
    is_sensitive = sharpe_range > 0.10

    sensitivity_result = SensitivityResult(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        sharpe_ratios=sharpe_ratios,
        calmar_ratios=calmar_ratios,
        max_drawdowns=max_drawdowns,
        annualized_returns=annualized_returns,
        optimal_value=optimal_value,
        optimal_sharpe=optimal_sharpe,
        sharpe_range=sharpe_range,
        sharpe_std=sharpe_std,
        is_sensitive=is_sensitive,
    )

    LOGGER.info(
        "Parameter sensitivity analysis completed",
        parameter=parameter_name,
        optimal_value=optimal_value,
        optimal_sharpe=f"{optimal_sharpe:.3f}",
        sharpe_range=f"{sharpe_range:.3f}",
        is_sensitive=is_sensitive,
    )

    return sensitivity_result


def run_grid_search(
    strategy_name: str,
    parameter_grid: dict[str, list[Any]],
    dataset_path: Path,
    split: TrainTestSplit | None = None,
    optimization_metric: str = "sharpe_ratio",
    *,
    risk_free_rate: float | None = None,
) -> GridSearchResult:
    """
    Perform exhaustive grid search across multiple parameters.

    Tests all combinations of parameter values to find optimal configuration.

    Parameters
    ----------
    strategy_name : str
        Strategy to optimize
    parameter_grid : dict[str, list[Any]]
        Grid of parameters to search
        Example: {
            "transaction_cost_bps": [5, 10, 20],
            "rebalance_frequency": ["monthly", "quarterly"],
        }
    dataset_path : Path
        Path to sector dataset
    split : TrainTestSplit | None
        Train/test split
    optimization_metric : str, default "sharpe_ratio"
        Metric to optimize
        Options: "sharpe_ratio", "calmar_ratio", "annualized_return"
    risk_free_rate : float | None, default None
        Annual risk-free rate used for metric calculations. Defaults to
        ``ThreeDRiskBacktestDefaults().risk_free_rate``.

    Returns
    -------
    GridSearchResult
        Complete grid search results with optimal configuration

    Raises
    ------
    ValueError
        If parameter_grid is empty or dataset not found
    FileNotFoundError
        If dataset_path does not exist

    Notes
    -----
    For large grids (>100 combinations), consider using random search
    or Bayesian optimization instead.

    Examples
    --------
    >>> grid = {
    ...     "transaction_cost_bps": [5.0, 10.0, 20.0],
    ...     "rebalance_frequency": ["monthly", "quarterly"],
    ... }
    >>> result = run_grid_search("risk_parity", grid, dataset_path)
    >>> print(result.best_config)
    {'transaction_cost_bps': 10.0, 'rebalance_frequency': 'monthly'}
    """
    if not dataset_path.exists():
        msg = f"Dataset path does not exist: {dataset_path}"
        raise FileNotFoundError(msg)

    if not parameter_grid:
        msg = "parameter_grid cannot be empty"
        raise ValueError(msg)

    # Calculate total number of combinations
    num_combinations = 1
    for values in parameter_grid.values():
        num_combinations *= len(values)

    LOGGER.info(
        "Starting grid search",
        strategy=strategy_name,
        num_parameters=len(parameter_grid),
        num_combinations=num_combinations,
    )

    # Load dataset
    dataset = _load_dataset(dataset_path)

    # Define split
    if split is None:
        split = define_train_test_split()

    # Generate all parameter combinations
    combinations = _generate_parameter_combinations(parameter_grid)

    resolved_risk_free_rate = (
        risk_free_rate if risk_free_rate is not None else PLAYGROUND_DEFAULTS.risk_free_rate
    )

    # Run backtests for all combinations
    results_rows: list[dict[str, Any]] = []

    for i, param_config in enumerate(combinations):
        LOGGER.debug(
            "Testing configuration",
            config_num=f"{i+1}/{num_combinations}",
            config=param_config,
        )

        # Run backtest
        result = _run_backtest(
            dataset=dataset,
            strategy_name=strategy_name,
            config=param_config,
            split=split,
        )

        # Calculate metrics
        metrics = calculate_performance_metrics(result, risk_free_rate=resolved_risk_free_rate)

        # Build row with parameters and metrics
        row = param_config.copy()
        row.update({
            "sharpe_ratio": metrics.sharpe_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "max_drawdown": metrics.maximum_drawdown,
            "annualized_return": metrics.annualized_return,
            "annualized_volatility": metrics.annualized_volatility,
            "total_return": metrics.cumulative_return,
            "num_rebalances": metrics.num_rebalances,
            "transaction_costs": metrics.transaction_costs_total,
        })
        results_rows.append(row)

    # Create results table
    results_table = pl.DataFrame(results_rows)

    # Find best configuration
    best_row = results_table.sort(optimization_metric, descending=True).head(1)
    best_config = {
        param: best_row[param][0]
        for param in parameter_grid.keys()
    }
    best_sharpe = float(best_row["sharpe_ratio"][0])
    best_metrics = {
        "sharpe_ratio": float(best_row["sharpe_ratio"][0]),
        "calmar_ratio": float(best_row["calmar_ratio"][0]),
        "max_drawdown": float(best_row["max_drawdown"][0]),
        "annualized_return": float(best_row["annualized_return"][0]),
        "annualized_volatility": float(best_row["annualized_volatility"][0]),
    }

    # Compute single-parameter sensitivity for each parameter
    sensitivity_summary: dict[str, SensitivityResult] = {}

    for param_name, param_values in parameter_grid.items():
        # Extract results for this parameter (averaging over other parameters)
        param_sensitivities = _compute_marginal_sensitivity(
            results_table=results_table,
            parameter_name=param_name,
            parameter_values=param_values,
        )
        sensitivity_summary[param_name] = param_sensitivities

    grid_result = GridSearchResult(
        parameter_grid=parameter_grid,
        results_table=results_table,
        best_config=best_config,
        best_sharpe=best_sharpe,
        best_metrics=best_metrics,
        sensitivity_summary=sensitivity_summary,
    )

    LOGGER.info(
        "Grid search completed",
        best_config=best_config,
        best_sharpe=f"{best_sharpe:.3f}",
        num_combinations_tested=num_combinations,
    )

    return grid_result


def analyze_parameter_stability(
    sensitivity_results: dict[str, SensitivityResult],
    stability_threshold: float = 0.10,
) -> dict[str, bool]:
    """
    Analyze which parameters are stable (low sensitivity).

    A parameter is considered stable if:
    - Sharpe ratio range < stability_threshold
    - Sharpe std / mean Sharpe < 0.20

    Parameters
    ----------
    sensitivity_results : dict[str, SensitivityResult]
        Results for each parameter tested
    stability_threshold : float, default 0.10
        Maximum allowed Sharpe range for stability

    Returns
    -------
    dict[str, bool]
        Mapping of parameter_name -> is_stable

    Examples
    --------
    >>> stability = analyze_parameter_stability(
    ...     sensitivity_results={"transaction_cost_bps": result1, ...},
    ...     stability_threshold=0.10,
    ... )
    >>> print(stability)
    {'transaction_cost_bps': False, 'rebalance_frequency': True}
    """
    stability_map: dict[str, bool] = {}

    for param_name, result in sensitivity_results.items():
        # Check Sharpe range
        is_stable_range = result.sharpe_range < stability_threshold

        # Check coefficient of variation
        mean_sharpe = float(np.mean(result.sharpe_ratios))
        if abs(mean_sharpe) > 1e-10:
            cv = result.sharpe_std / abs(mean_sharpe)
            is_stable_cv = cv < 0.20
        else:
            is_stable_cv = True  # If mean is zero, consider stable

        # Parameter is stable if both conditions met
        is_stable = is_stable_range and is_stable_cv

        stability_map[param_name] = is_stable

        LOGGER.debug(
            "Parameter stability analysis",
            parameter=param_name,
            sharpe_range=f"{result.sharpe_range:.3f}",
            cv=f"{cv:.3f}" if abs(mean_sharpe) > 1e-10 else "N/A",
            is_stable=is_stable,
        )

    return stability_map


def compare_strategies_sensitivity(
    strategy_sensitivity: dict[str, dict[str, SensitivityResult]],
) -> pl.DataFrame:
    """
    Compare parameter sensitivity across multiple strategies.

    Parameters
    ----------
    strategy_sensitivity : dict[str, dict[str, SensitivityResult]]
        Nested dict: strategy_name -> parameter_name -> SensitivityResult

    Returns
    -------
    pl.DataFrame
        Comparison table with columns:
        - strategy_name
        - parameter_name
        - sharpe_range (sensitivity measure)
        - is_sensitive
        - optimal_value

    Examples
    --------
    >>> comparison = compare_strategies_sensitivity({
    ...     "risk_parity": {"transaction_cost_bps": result1, ...},
    ...     "min_variance": {"transaction_cost_bps": result2, ...},
    ... })
    >>> print(comparison)
    """
    rows: list[dict[str, Any]] = []

    for strategy_name, param_results in strategy_sensitivity.items():
        for param_name, result in param_results.items():
            rows.append({
                "strategy_name": strategy_name,
                "parameter_name": param_name,
                "sharpe_range": result.sharpe_range,
                "is_sensitive": result.is_sensitive,
                "optimal_value": result.optimal_value,
                "optimal_sharpe": result.optimal_sharpe,
            })

    return pl.DataFrame(rows)


def generate_sensitivity_report(
    grid_result: GridSearchResult,
    output_path: Path,
    strategy_name: str = "Unknown Strategy",
) -> None:
    """
    Generate markdown report with sensitivity analysis.

    Parameters
    ----------
    grid_result : GridSearchResult
        Grid search results
    output_path : Path
        Path to save markdown report
    strategy_name : str, default "Unknown Strategy"
        Name of strategy for report title

    Raises
    ------
    IOError
        If unable to write report file

    Examples
    --------
    >>> generate_sensitivity_report(
    ...     grid_result=result,
    ...     output_path=Path("reports/sensitivity_analysis.md"),
    ...     strategy_name="Risk Parity",
    ... )
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write(f"# Parameter Sensitivity Analysis: {strategy_name}\n\n")
        f.write(f"**Generated:** {datetime.now(UTC).isoformat().replace('+00:00', 'Z')}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("**Optimal Configuration:**\n\n")
        for param, value in grid_result.best_config.items():
            f.write(f"- **{param}**: {value}\n")
        f.write("\n")

        f.write("**Optimal Performance:**\n\n")
        for metric, value in grid_result.best_metrics.items():
            if "ratio" in metric:
                f.write(f"- **{metric}**: {value:.3f}\n")
            else:
                f.write(f"- **{metric}**: {value:.2%}\n")
        f.write("\n")

        # Stability Assessment
        stability = analyze_parameter_stability(grid_result.sensitivity_summary)
        stable_params = [p for p, is_stable in stability.items() if is_stable]
        sensitive_params = [p for p, is_stable in stability.items() if not is_stable]

        f.write("**Stability Assessment:**\n\n")
        f.write(f"- Stable parameters: {', '.join(stable_params) if stable_params else 'None'}\n")
        f.write(f"- Sensitive parameters: {', '.join(sensitive_params) if sensitive_params else 'None'}\n\n")

        # Parameter Sensitivity Results
        f.write("## Parameter Sensitivity Results\n\n")

        for param_name, result in grid_result.sensitivity_summary.items():
            f.write(f"### {param_name}\n\n")

            # Summary table
            summary_df = result.summary_table()
            f.write(_format_dataframe_as_markdown(summary_df))
            f.write("\n\n")

            # Analysis
            f.write("**Analysis:**\n\n")
            f.write(f"- Sharpe range: {result.sharpe_range:.3f}\n")
            f.write(f"- Sharpe std: {result.sharpe_std:.3f}\n")
            f.write(f"- Sensitive: {'Yes' if result.is_sensitive else 'No'}\n")
            f.write(f"- Optimal value: {result.optimal_value}\n")
            f.write(f"- Optimal Sharpe: {result.optimal_sharpe:.3f}\n\n")

        # Grid Search Results
        f.write("## Grid Search Results\n\n")
        f.write("### Top 10 Configurations\n\n")
        top_10 = grid_result.get_top_k_configs(k=10)
        f.write(_format_dataframe_as_markdown(top_10))
        f.write("\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("1. **Production Configuration**: Use optimal parameters from grid search\n")
        if sensitive_params:
            f.write(f"2. **Conservative Alternative**: Consider robust values for sensitive parameters: {', '.join(sensitive_params)}\n")
        f.write("3. **Monitoring**: Track parameter performance in live trading\n")
        f.write("4. **Reoptimization**: Review parameters quarterly based on regime changes\n")

    LOGGER.info("Sensitivity report generated", output_path=str(output_path))


# ===== Helper Functions =====


def _load_dataset(dataset_path: Path) -> SectorDataset:
    """Load sector dataset from Parquet or CSV file."""
    from playground.risk_model.dataset import CoverageSummary
    from playground.risk_model.dataset import SectorDataset

    if dataset_path.suffix == ".parquet":
        df = pl.read_parquet(dataset_path)
    elif dataset_path.suffix == ".csv":
        df = pl.read_csv(dataset_path)
    else:
        msg = f"Unsupported file format: {dataset_path.suffix}"
        raise ValueError(msg)

    # Validate required columns
    required_cols = {"timestamp", "symbol", "return"}
    if not required_cols.issubset(df.columns):
        msg = f"Dataset missing required columns: {required_cols - set(df.columns)}"
        raise ValueError(msg)

    # Convert timestamp to datetime if needed
    if df["timestamp"].dtype != pl.Datetime:
        try:
            df = df.with_columns(
                pl.col("timestamp").str.to_datetime(time_zone="UTC")
            )
        except Exception:
            df = df.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z")
            )

    # Create mock factor returns (all zeros for now)
    unique_timestamps = df.select("timestamp").unique().sort("timestamp")
    factor_returns = unique_timestamps.with_columns([
        pl.lit(0.0).alias("factor_duration"),
        pl.lit(0.0).alias("factor_credit"),
        pl.lit(0.0).alias("factor_liquidity"),
    ])

    # Create coverage summary
    sectors = df["symbol"].unique().to_list()
    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=len(unique_timestamps),
        factor_expected_days=len(unique_timestamps),
        sector_coverage=dict.fromkeys(sectors, 1.0),
        factor_coverage={
            "factor_duration": 1.0,
            "factor_credit": 1.0,
            "factor_liquidity": 1.0,
        },
    )

    return SectorDataset(
        sector_returns=df,
        factor_returns=factor_returns,
        coverage=coverage,
    )


def _run_backtest(
    dataset: SectorDataset,
    strategy_name: str,
    config: dict[str, Any],
    split: TrainTestSplit,
) -> BacktestResult:
    """Run a single backtest with given configuration."""
    # Create BacktestConfig
    backtest_config = BacktestConfig(
        start_date=split.test_start,
        end_date=split.test_end,
        initial_capital=config.get("initial_capital", 1_000_000.0),
        rebalance_frequency=config.get("rebalance_frequency", "monthly"),
        transaction_cost_bps=config.get("transaction_cost_bps", 10.0),
        slippage_bps=config.get("slippage_bps", 0.0),
        position_limits=config.get("position_limits", None),
        rebalance_threshold=config.get("rebalance_threshold", 0.05),
        random_seed=config.get("random_seed", 42),
    )

    # Run backtest
    backtester = FactorBacktester(backtest_config)
    result = backtester.run_backtest(
        dataset=dataset,
        strategy=strategy_name,
        strategy_params=config.get("strategy_params", {}),
    )

    return result


def _generate_parameter_combinations(
    parameter_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Generate all combinations of parameter values."""
    import itertools

    keys = list(parameter_grid.keys())
    values = [parameter_grid[k] for k in keys]

    combinations: list[dict[str, Any]] = []
    for value_combination in itertools.product(*values):
        combinations.append(dict(zip(keys, value_combination)))

    return combinations


def _compute_marginal_sensitivity(
    results_table: pl.DataFrame,
    parameter_name: str,
    parameter_values: list[Any],
) -> SensitivityResult:
    """Compute marginal sensitivity for a single parameter."""
    # Group by parameter value and compute mean metrics
    sharpe_ratios: list[float] = []
    calmar_ratios: list[float] = []
    max_drawdowns: list[float] = []
    annualized_returns: list[float] = []

    for value in parameter_values:
        subset = results_table.filter(pl.col(parameter_name) == value)
        sharpe_mean = subset["sharpe_ratio"].mean()
        calmar_mean = subset["calmar_ratio"].mean()
        maxdd_mean = subset["max_drawdown"].mean()
        ret_mean = subset["annualized_return"].mean()

        sharpe_ratios.append(float(sharpe_mean) if isinstance(sharpe_mean, Real) else 0.0)
        calmar_ratios.append(float(calmar_mean) if isinstance(calmar_mean, Real) else 0.0)
        max_drawdowns.append(float(maxdd_mean) if isinstance(maxdd_mean, Real) else 0.0)
        annualized_returns.append(float(ret_mean) if isinstance(ret_mean, Real) else 0.0)

    # Find optimal
    optimal_idx = int(np.argmax(sharpe_ratios))
    optimal_value = parameter_values[optimal_idx]
    optimal_sharpe = sharpe_ratios[optimal_idx]

    # Sensitivity metrics
    sharpe_range = float(np.max(sharpe_ratios) - np.min(sharpe_ratios))
    sharpe_std = float(np.std(sharpe_ratios, ddof=1)) if len(sharpe_ratios) > 1 else 0.0
    is_sensitive = sharpe_range > 0.10

    return SensitivityResult(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        sharpe_ratios=sharpe_ratios,
        calmar_ratios=calmar_ratios,
        max_drawdowns=max_drawdowns,
        annualized_returns=annualized_returns,
        optimal_value=optimal_value,
        optimal_sharpe=optimal_sharpe,
        sharpe_range=sharpe_range,
        sharpe_std=sharpe_std,
        is_sensitive=is_sensitive,
    )


def _format_dataframe_as_markdown(df: pl.DataFrame) -> str:
    """Format a Polars DataFrame as a markdown table."""
    if df.is_empty():
        return "*No data*"

    # Get column names
    columns = df.columns

    # Header
    lines = []
    header = "| " + " | ".join(columns) + " |"
    lines.append(header)

    # Separator
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.append(separator)

    # Rows
    for row in df.iter_rows(named=True):
        formatted_values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                # Format floats with appropriate precision
                if col in {"sharpe_ratio", "calmar_ratio"}:
                    formatted_values.append(f"{value:.3f}")
                elif col in {"max_drawdown", "annualized_return"}:
                    formatted_values.append(f"{value:.2%}")
                else:
                    formatted_values.append(f"{value:.4f}")
            elif isinstance(value, int):
                formatted_values.append(f"{value:,}")
            else:
                formatted_values.append(str(value))

        row_str = "| " + " | ".join(formatted_values) + " |"
        lines.append(row_str)

    return "\n".join(lines)


# ===== Public API =====

__all__ = [
    "COMPREHENSIVE_GRID",
    "STANDARD_GRIDS",
    "GridSearchResult",
    "ParameterConfig",
    "SensitivityResult",
    "analyze_parameter_stability",
    "compare_strategies_sensitivity",
    "generate_sensitivity_report",
    "run_grid_search",
    "run_parameter_sensitivity",
]
