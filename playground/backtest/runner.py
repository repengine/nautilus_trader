"""
Backtest suite orchestrator for running and comparing multiple strategies.

This module orchestrates comprehensive backtest execution across all portfolio
strategies, calculates performance metrics, and generates comparison reports
for the 3D Factor Risk Model project.

Key Features:
- Runs all benchmark strategies (equal-weight, 60/40, risk parity, min variance)
- Placeholder for 3D Factor Model strategies (stable/rolling betas)
- Performance metric calculation for train/test periods
- Strategy comparison tables and reports
- Markdown report generation
- Reproducible with fixed random seed

Performance Targets (Cold Path):
- Full suite execution: < 5 minutes for 2010-2024 data
- Report generation: < 10 seconds
- No performance-critical constraints (offline analysis)

Hot/Cold Path Separation:
- This is a cold-path module (backtesting is offline analysis)
- No real-time constraints, optimized for correctness over repeatability

Integration Notes:
- Compatible with FactorBacktester from engine.py
- Uses all benchmark strategies from benchmarks.py
- Follows Phase 3.2.2 requirements from 3D_Risk_Model_Roadmap.md
- Outputs ready for Phase 3.2.3 (regime analysis)

Examples
--------
Run full backtest suite:

>>> split = define_train_test_split()
>>> suite = run_full_backtest_suite(
...     dataset_path=Path("data/sector_returns.parquet"),
...     output_dir=Path("reports/"),
...     split=split,
... )
>>> print(suite.compare_strategies())
>>> suite.to_markdown_report(Path("reports/backtest_results_2010_2024.md"))
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, TextIO, cast

import numpy as np
import polars as pl
import structlog

from playground.backtest.benchmarks import MinimumVarianceStrategy
from playground.backtest.benchmarks import RiskParityStrategy
from playground.backtest.benchmarks import SixtyFortyStrategy
from playground.backtest.engine import BacktestConfig
from playground.backtest.engine import FactorBacktester
from playground.backtest.liquidity_controls import LiquidityScalingConfig
from playground.backtest.liquidity_controls import build_regime_scaling_maps
from playground.backtest.liquidity_controls import load_liquidity_contributions_from_csv
from playground.backtest.performance_metrics import PerformanceMetrics
from playground.backtest.performance_metrics import calculate_performance_metrics
from playground.backtest.regime_analysis import RegimeAnalysisResult
from playground.backtest.regime_analysis import analyze_strategy_across_regimes
from playground.backtest.regime_analysis import compare_strategies_across_regimes
from playground.backtest.regime_analysis import define_market_regimes
from playground.backtest.splits import TrainTestSplit
from playground.backtest.splits import define_train_test_split


if TYPE_CHECKING:
    from playground.backtest.engine import BacktestResult
    from playground.backtest.regime_analysis import MarketRegime
    from playground.risk_model.dataset import SectorDataset


LOGGER = structlog.get_logger(__name__)


LIQUIDITY_ATTRIBUTION_DIR = Path("playground/reports/backtesting/attribution/regime")
LIQUIDITY_STRATEGY_SLUG = "3d_factor_rolling_betas"
LIQUIDITY_FALLBACK_CONTRIBUTIONS: dict[str, float] = {
    "Rate Hiking Cycle": -0.0204,
}


# ===== Backtest Suite Dataclass =====


@dataclass(slots=True)
class BacktestSuite:
    """
    Results from a full backtest suite across all strategies.

    This dataclass stores results, metrics, and metadata for a complete
    backtest run covering multiple strategies and train/test periods.

    Attributes
    ----------
    strategies : dict[str, BacktestResult]
        Mapping of strategy name to backtest result
    metrics : dict[str, PerformanceMetrics]
        Mapping of strategy name to performance metrics
    split : TrainTestSplit
        Train/test split configuration used
    config : BacktestConfig
        Backtest configuration parameters
    train_results : dict[str, BacktestResult]
        Training period backtest results by strategy
    train_metrics : dict[str, PerformanceMetrics]
        Training period performance metrics by strategy
    overall_metrics : dict[str, PerformanceMetrics]
        Full-period performance metrics by strategy (train + test)
    full_results : dict[str, BacktestResult]
        Full-period backtest results (train + test)
    regime_results : dict[str, RegimeAnalysisResult]
        Regime analysis results by strategy

    Methods
    -------
    compare_strategies() -> pl.DataFrame
        Generate comparison table of all strategies
    to_markdown_report(output_path: Path) -> None
        Generate markdown report with results

    Examples
    --------
    >>> suite = run_full_backtest_suite(dataset_path, output_dir)
    >>> comparison = suite.compare_strategies()
    >>> print(comparison)
    >>> suite.to_markdown_report(Path("reports/results.md"))
    """

    strategies: dict[str, BacktestResult]
    metrics: dict[str, PerformanceMetrics]
    split: TrainTestSplit
    config: BacktestConfig
    train_results: dict[str, BacktestResult] = field(default_factory=dict)
    train_metrics: dict[str, PerformanceMetrics] = field(default_factory=dict)
    overall_metrics: dict[str, PerformanceMetrics] = field(default_factory=dict)
    full_results: dict[str, BacktestResult] = field(default_factory=dict)
    regime_results: dict[str, RegimeAnalysisResult] = field(default_factory=dict)
    attribution: dict[str, FactorAttribution] = field(default_factory=dict)
    regime_attribution: dict[str, list[FactorAttribution]] = field(default_factory=dict)
    regime_scaling_map: dict[str, float] = field(default_factory=dict)
    regime_factor_multipliers: dict[str, dict[str, float]] = field(default_factory=dict)
    liquidity_contributions: dict[str, float] = field(default_factory=dict)

    def compare_strategies(self) -> pl.DataFrame:
        """
        Generate comparison table of all strategies.

        Returns
        -------
        pl.DataFrame
            Comparison table with columns:
            - strategy: Strategy name
            - annualized_return: Annualized return (%)
            - annualized_volatility: Annualized volatility (%)
            - sharpe_ratio: Sharpe ratio
            - sortino_ratio: Sortino ratio
            - calmar_ratio: Calmar ratio
            - max_drawdown: Maximum drawdown (%)
            - information_ratio: Information ratio vs benchmark (if available)
            - num_rebalances: Number of rebalances
            - transaction_costs: Total transaction costs ($)

        Notes
        -----
        Strategies are sorted by Sharpe ratio (descending).
        """
        rows = []

        for strategy_name, metrics in self.metrics.items():
            row = {
                "strategy": strategy_name,
                "annualized_return": metrics.annualized_return * 100,
                "annualized_volatility": metrics.annualized_volatility * 100,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "max_drawdown": metrics.maximum_drawdown * 100,
                "information_ratio": metrics.information_ratio if metrics.information_ratio is not None else float("nan"),
                "num_rebalances": metrics.num_rebalances,
                "transaction_costs": metrics.transaction_costs_total,
            }
            rows.append(row)

        df = pl.DataFrame(rows)

        # Sort by Sharpe ratio descending
        df = df.sort("sharpe_ratio", descending=True)

        return df

    def train_vs_test_table(self) -> pl.DataFrame:
        """
        Generate comparison table for train vs test metrics.

        Returns
        -------
        pl.DataFrame
            Table with columns:
            - strategy
            - period ("train", "test")
            - annualized_return
            - annualized_volatility
            - sharpe_ratio
            - calmar_ratio
            - max_drawdown
            - cumulative_return
            - num_rebalances
        """
        rows: list[dict[str, object]] = []
        strategy_names = sorted(set(self.metrics) | set(self.train_metrics))

        for strategy in strategy_names:
            if strategy in self.train_metrics:
                metrics = self.train_metrics[strategy]
                rows.append({
                    "strategy": strategy,
                    "period": "train",
                    "annualized_return": metrics.annualized_return,
                    "annualized_volatility": metrics.annualized_volatility,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "calmar_ratio": metrics.calmar_ratio,
                    "max_drawdown": metrics.maximum_drawdown,
                    "cumulative_return": metrics.cumulative_return,
                    "num_rebalances": metrics.num_rebalances,
                })

            if strategy in self.metrics:
                metrics = self.metrics[strategy]
                rows.append({
                    "strategy": strategy,
                    "period": "test",
                    "annualized_return": metrics.annualized_return,
                    "annualized_volatility": metrics.annualized_volatility,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "calmar_ratio": metrics.calmar_ratio,
                    "max_drawdown": metrics.maximum_drawdown,
                    "cumulative_return": metrics.cumulative_return,
                    "num_rebalances": metrics.num_rebalances,
                })

        return pl.DataFrame(rows) if rows else pl.DataFrame(schema={
            "strategy": pl.Utf8,
            "period": pl.Utf8,
            "annualized_return": pl.Float64,
            "annualized_volatility": pl.Float64,
            "sharpe_ratio": pl.Float64,
            "calmar_ratio": pl.Float64,
            "max_drawdown": pl.Float64,
            "cumulative_return": pl.Float64,
            "num_rebalances": pl.Int64,
        })

    def regime_summary(self) -> pl.DataFrame:
        """
        Generate regime performance summary across strategies.

        Returns
        -------
        pl.DataFrame
            Table with columns:
            - strategy
            - regime_name
            - sharpe_ratio
            - annualized_return
            - annualized_volatility
            - max_drawdown
            - calmar_ratio
            - win_rate
            - num_observations
            - status
        """
        rows: list[dict[str, object]] = []

        for strategy, analysis in self.regime_results.items():
            for regime_name, perf in analysis.regime_performances.items():
                rows.append({
                    "strategy": strategy,
                    "regime_name": regime_name,
                    "sharpe_ratio": perf.sharpe_ratio,
                    "annualized_return": perf.annualized_return,
                    "annualized_volatility": perf.annualized_volatility,
                    "max_drawdown": perf.max_drawdown,
                    "calmar_ratio": perf.calmar_ratio,
                    "win_rate": perf.win_rate,
                    "num_observations": perf.num_observations,
                    "status": "Success" if perf.is_successful else "Failed",
                })

        if not rows:
            return pl.DataFrame(schema={
                "strategy": pl.Utf8,
                "regime_name": pl.Utf8,
                "sharpe_ratio": pl.Float64,
                "annualized_return": pl.Float64,
                "annualized_volatility": pl.Float64,
                "max_drawdown": pl.Float64,
                "calmar_ratio": pl.Float64,
                "win_rate": pl.Float64,
                "num_observations": pl.Int64,
                "status": pl.Utf8,
            })

        return pl.DataFrame(rows).sort(["regime_name", "sharpe_ratio"], descending=[False, True])

    def to_markdown_report(self, output_path: Path) -> None:
        """
        Generate markdown report with backtest results.

        The report includes:
        - Executive summary
        - Strategy comparison table
        - Detailed metrics for each strategy
        - Train vs test period comparison
        - Configuration parameters

        Parameters
        ----------
        output_path : Path
            Path to output markdown file

        Raises
        ------
        IOError
            If unable to write report file

        Examples
        --------
        >>> suite.to_markdown_report(Path("reports/backtest_results.md"))
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # Header
            f.write("# Backtest Results: Full Strategy Suite\n\n")
            f.write(f"**Generated:** {datetime.now(UTC).isoformat()}\n\n")

            # Configuration
            f.write("## Configuration\n\n")
            f.write(f"- **Training Period:** {self.split.train_start.date()} to {self.split.train_end.date()}\n")
            f.write(f"- **Testing Period:** {self.split.test_start.date()} to {self.split.test_end.date()}\n")
            f.write(f"- **Initial Capital:** ${self.config.initial_capital:,.0f}\n")
            f.write(f"- **Rebalance Frequency:** {self.config.rebalance_frequency}\n")
        f.write(f"- **Transaction Cost:** {self.config.transaction_cost_bps} bps\n")
        f.write(f"- **Random Seed:** {self.config.random_seed}\n\n")

        if self.liquidity_contributions:
            f.write("### Liquidity Controls Snapshot\n\n")
            sorted_contributions = sorted(
                self.liquidity_contributions.items(),
                key=lambda item: item[1],
            )
            highlight_count = min(3, len(sorted_contributions))
            for regime_name, contribution in sorted_contributions[:highlight_count]:
                regime_multiplier = self.regime_scaling_map.get(regime_name, 1.0)
                factor_overrides = self.regime_factor_multipliers.get(regime_name, {})
                liquidity_multiplier = factor_overrides.get("factor_liquidity", 1.0)
                f.write(
                    f"- {regime_name}: contribution {contribution:.2%}, "
                    f"regime multiplier {regime_multiplier:.2f}, "
                    f"liquidity multiplier {liquidity_multiplier:.2f}\n"
                )
            f.write("\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        comparison_df = self.compare_strategies()
        f.write(self._format_dataframe_as_markdown(comparison_df))
        f.write("\n\n")
        top_row = comparison_df.sort("sharpe_ratio", descending=True).to_dicts()[0]
        f.write(
            f"*Top Sharpe (test period):* {top_row['strategy']} "
            f"({top_row['sharpe_ratio']:.3f} Sharpe, "
            f"{top_row['annualized_return']:.2f}% annualized return).\n\n"
        )
        # Detailed Metrics
        f.write("## Detailed Performance Metrics\n\n")
        for strategy_name in sorted(self.strategies.keys()):
            f.write(f"### {strategy_name}\n\n")

            train_metrics = self.train_metrics.get(strategy_name)
            test_metrics = self.metrics.get(strategy_name)
            full_metrics = self.overall_metrics.get(strategy_name)
            if train_metrics is not None:
                f.write("#### Train Period\n\n")
                self._write_metrics_block(f, train_metrics)
                f.write("\n")

            if test_metrics is not None:
                f.write("#### Test Period\n\n")
                self._write_metrics_block(f, test_metrics)
                f.write("\n")

            if full_metrics is not None:
                f.write("#### Full Period\n\n")
                self._write_metrics_block(f, full_metrics)
                f.write("\n")

            f.write("---\n\n")

        train_test_df = self.train_vs_test_table()
        if not train_test_df.is_empty():
            f.write("## Train vs Test Comparison\n\n")
            f.write(self._format_dataframe_as_markdown(train_test_df))
            f.write("\n\n")

        regime_df = self.regime_summary()
        if not regime_df.is_empty():
            f.write("## Regime Analysis Summary\n\n")
            f.write(self._format_dataframe_as_markdown(regime_df))
            f.write("\n\n")

        if self.regime_results:
            f.write("## Regime Observations\n\n")
            for strategy_name, analysis in sorted(self.regime_results.items()):
                failed = [
                    regime_name
                    for regime_name, performance in analysis.regime_performances.items()
                    if not performance.is_successful
                ]
                if failed:
                    failed_list = ", ".join(failed)
                    f.write(f"- {strategy_name}: underperformance in {failed_list} regimes.\n")
                else:
                    f.write(f"- {strategy_name}: met targets in all regimes.\n")
            f.write("\n")

            factor_rows = comparison_df.filter(pl.col("strategy").str.contains("3D Factor"))
            if not factor_rows.is_empty():
                f.write("## Additional Insights\n\n")
                mean_cost_series = factor_rows["transaction_costs"]
                mean_cost_value = mean_cost_series.mean()
                mean_cost = float(cast(float, mean_cost_value)) if mean_cost_value is not None else 0.0
                f.write(
                    "- 3D Factor strategies incur average transaction costs "
                    f"of ${mean_cost:,.0f}; monitor turnover budgeting despite recent reductions.\n"
                )
                if self.regime_results:
                    hiking_failures = [
                        name for name, analysis in self.regime_results.items()
                        if "Rate Hiking Cycle" in [
                            regime_name
                            for regime_name, perf in analysis.regime_performances.items()
                            if not perf.is_successful
                        ]
                    ]
                    if hiking_failures:
                        joined = ", ".join(hiking_failures)
                        f.write(
                            f"- Rate Hiking Cycle remains a weak spot for {joined}; "
                            "consider regime-aware factor scaling in future iterations.\n"
                        )
                f.write("\n")

            if self.liquidity_contributions:
                f.write("## Liquidity Regime Controls\n\n")
                for regime_name in sorted(self.liquidity_contributions):
                    contribution = self.liquidity_contributions[regime_name]
                    regime_multiplier = self.regime_scaling_map.get(regime_name, 1.0)
                    factor_overrides = self.regime_factor_multipliers.get(regime_name, {})
                    liquidity_multiplier = factor_overrides.get("factor_liquidity", 1.0)
                    override_items = [
                        f"{factor} {multiplier:.2f}"
                        for factor, multiplier in sorted(factor_overrides.items())
                    ]
                    if "factor_liquidity" not in factor_overrides:
                        override_items.append("factor_liquidity 1.00")
                    parts = [
                        f"liquidity contribution {contribution:.2%}",
                        f"regime multiplier {regime_multiplier:.2f}",
                        ", ".join(override_items),
                    ]
                    f.write(f"- {regime_name}: " + ", ".join(parts) + "\n")
                f.write("\n")

            if self.attribution:
                f.write("## Factor Attribution (Test Period)\n\n")
                for strategy_name in sorted(self.attribution):
                    attribution = self.attribution[strategy_name]
                    f.write(f"### {attribution.strategy_name}\n\n")
                    f.write(f"- Alpha (daily): {attribution.alpha:.6f}\n")
                    f.write(f"- Alpha (annualized): {attribution.alpha_annualized:.2%}\n")
                    f.write(f"- R²: {attribution.r_squared:.3f}\n")
                    if attribution.betas:
                        f.write("- Betas:\n")
                        for factor, beta in attribution.betas.items():
                            f.write(f"  - {factor}: {beta:.4f}\n")
                    if attribution.factor_contributions:
                        f.write("- Annualized Factor Contributions:\n")
                        for factor, contribution in attribution.factor_contributions.items():
                            f.write(f"  - {factor}: {contribution:.2%}\n")
                    export_name = re.sub(r"[^a-z0-9]+", "_", attribution.strategy_name.lower()).strip("_") + "_attribution.csv"
                    f.write(f"- CSV Export: attribution/{export_name}\n")
                    regime_attrs = self.regime_attribution.get(strategy_name, [])
                    if regime_attrs:
                        f.write("- Regime Highlights:\n")
                        for regime_attr in sorted(regime_attrs, key=lambda item: item.regime_name or ""):
                            top_factor = None
                            if regime_attr.factor_contributions:
                                top_factor = max(
                                    regime_attr.factor_contributions.items(),
                                    key=lambda pair: abs(pair[1]),
                                )
                            highlight = (
                                f"alpha {regime_attr.alpha_annualized:.2%}"
                            )
                            if top_factor is not None:
                                highlight += f", top factor {top_factor[0]} {top_factor[1]:.2%}"
                            f.write(
                                f"  - {regime_attr.regime_name or 'N/A'}: {highlight}\n"
                            )
                        regime_export = re.sub(r"[^a-z0-9]+", "_", attribution.strategy_name.lower()).strip("_") + "_regime_attribution.csv"
                        f.write(f"- Regime CSV Export: attribution/regime/{regime_export}\n")
                    f.write("\n")

            # Footer
            f.write("## Notes\n\n")
            f.write("- All returns are geometric (compounded)\n")
            f.write("- Volatility and ratios are annualized (252 trading days)\n")
            f.write("- VaR and CVaR are computed on daily returns\n")
            f.write("- Information ratio computed vs equal-weight benchmark (if applicable)\n")
            f.write("- Transaction costs deducted from portfolio value at each rebalance\n")

        LOGGER.info("Markdown report generated", output_path=str(output_path))

    def _format_dataframe_as_markdown(self, df: pl.DataFrame) -> str:
        """
        Format a Polars DataFrame as a markdown table.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame to format

        Returns
        -------
        str
            Markdown table string
        """
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
                    # Format floats with 2 decimal places
                    if col in {"sharpe_ratio", "sortino_ratio", "calmar_ratio", "information_ratio"}:
                        formatted_values.append(f"{value:.3f}")
                    else:
                        formatted_values.append(f"{value:.2f}")
                elif isinstance(value, int):
                    formatted_values.append(f"{value:,}")
                else:
                    formatted_values.append(str(value))

            row_str = "| " + " | ".join(formatted_values) + " |"
            lines.append(row_str)

        return "\n".join(lines)

    def _write_metrics_block(self, file_obj: TextIO, metrics: PerformanceMetrics) -> None:
        """Write a metrics block to the markdown report."""
        file_obj.write("**Return Metrics:**\n\n")
        file_obj.write(f"- Cumulative Return: {metrics.cumulative_return:.2%}\n")
        file_obj.write(f"- Annualized Return: {metrics.annualized_return:.2%}\n")
        file_obj.write(f"- Monthly Return (Mean): {metrics.monthly_return_mean:.2%}\n")
        file_obj.write(f"- Monthly Return (Std): {metrics.monthly_return_std:.2%}\n\n")

        file_obj.write("**Risk Metrics:**\n\n")
        file_obj.write(f"- Annualized Volatility: {metrics.annualized_volatility:.2%}\n")
        file_obj.write(f"- Maximum Drawdown: {metrics.maximum_drawdown:.2%}\n")
        file_obj.write(f"- VaR (95%): {metrics.var_95:.2%}\n")
        file_obj.write(f"- VaR (99%): {metrics.var_99:.2%}\n")
        file_obj.write(f"- CVaR (95%): {metrics.cvar_95:.2%}\n")
        file_obj.write(f"- CVaR (99%): {metrics.cvar_99:.2%}\n\n")

        file_obj.write("**Risk-Adjusted Metrics:**\n\n")
        file_obj.write(f"- Sharpe Ratio: {metrics.sharpe_ratio:.3f}\n")
        file_obj.write(f"- Sortino Ratio: {metrics.sortino_ratio:.3f}\n")
        file_obj.write(f"- Calmar Ratio: {metrics.calmar_ratio:.3f}\n")
        if metrics.information_ratio is not None:
            file_obj.write(f"- Information Ratio: {metrics.information_ratio:.3f}\n")
        file_obj.write("\n")

        file_obj.write("**Trade Metrics:**\n\n")
        file_obj.write(f"- Number of Rebalances: {metrics.num_rebalances}\n")
        file_obj.write(f"- Average Monthly Turnover: {metrics.turnover_rate:.2%}\n")
        file_obj.write(f"- Total Transaction Costs: ${metrics.transaction_costs_total:,.2f}\n")
        file_obj.write(f"- Transaction Costs (% of Returns): {metrics.transaction_costs_pct:.2%}\n")


# ===== Main Orchestrator Function =====


def run_full_backtest_suite(
    dataset_path: Path,
    output_dir: Path,
    split: TrainTestSplit | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> BacktestSuite:
    """
    Run complete backtest suite for all strategies.

    This function orchestrates the full backtesting pipeline:
    1. Load sector dataset
    2. Define train/test split
    3. Run backtests for all strategies
    4. Calculate performance metrics
    5. Save results to output directory

    Parameters
    ----------
    dataset_path : Path
        Path to sector dataset (Parquet or CSV)
        Expected columns: timestamp, symbol, return
    output_dir : Path
        Directory to save results and reports
        Will be created if it doesn't exist
    split : TrainTestSplit | None
        Train/test split configuration
        If None, uses default 2010-2018/2019-2024 split
    config_overrides : dict[str, Any] | None
        Overrides for BacktestConfig parameters
        Keys: initial_capital, transaction_cost_bps, rebalance_frequency, etc.

    Returns
    -------
    BacktestSuite
        Complete suite results with all strategies

    Raises
    ------
    FileNotFoundError
        If dataset_path does not exist
    ValueError
        If dataset is invalid or empty

    Examples
    --------
    Basic usage:

    >>> suite = run_full_backtest_suite(
    ...     dataset_path=Path("data/sector_returns.parquet"),
    ...     output_dir=Path("reports/"),
    ... )
    >>> print(suite.compare_strategies())

    Custom split:

    >>> split = TrainTestSplit(
    ...     train_start=datetime(2012, 1, 1, tzinfo=UTC),
    ...     train_end=datetime(2018, 12, 31, tzinfo=UTC),
    ...     test_start=datetime(2019, 1, 1, tzinfo=UTC),
    ...     test_end=datetime(2023, 12, 31, tzinfo=UTC),
    ... )
    >>> suite = run_full_backtest_suite(
    ...     dataset_path=Path("data/sector_returns.parquet"),
    ...     output_dir=Path("reports/"),
    ...     split=split,
    ... )

    Notes
    -----
    Strategies executed:
    1. Equal Weight (baseline)
    2. 60/40 Portfolio (classic benchmark)
    3. Risk Parity (equal risk contribution)
    4. Minimum Variance (optimal low-risk)
    5. 3D Factor Model (stable betas)
    6. 3D Factor Model (rolling betas)

    Acceptance criteria validation:
    - All metrics calculated for train and test periods
    - Regime summaries generated for all strategies
    - Results reproducible with fixed random seed
    """
    if not dataset_path.exists():
        msg = f"Dataset path does not exist: {dataset_path}"
        raise FileNotFoundError(msg)

    LOGGER.info(
        "Starting full backtest suite",
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
    )

    # === Load Dataset ===
    dataset = _load_sector_dataset(dataset_path)

    # === Define Split ===
    if split is None:
        split = define_train_test_split()

    LOGGER.info(
        "Using train/test split",
        train_period=f"{split.train_start.date()} to {split.train_end.date()}",
        test_period=f"{split.test_start.date()} to {split.test_end.date()}",
    )


    # === Create Backtest Config ===
    train_config = _create_backtest_config(split, config_overrides, period="train")
    test_config = _create_backtest_config(split, config_overrides, period="test")
    full_config = _create_backtest_config(split, config_overrides, period="full")
    regimes = define_market_regimes()
    regime_resolver = _build_regime_resolver(regimes)
    liquidity_config = LiquidityScalingConfig()
    rolling_liquidity = load_liquidity_contributions_from_csv(
        LIQUIDITY_ATTRIBUTION_DIR,
        LIQUIDITY_STRATEGY_SLUG,
    )
    if not rolling_liquidity:
        rolling_liquidity = LIQUIDITY_FALLBACK_CONTRIBUTIONS
    regime_scaling_map, liquidity_controls = build_regime_scaling_maps(
        rolling_liquidity,
        config=liquidity_config,
    )

    # === Run Backtests Across Periods ===
    strategies_to_run: list[tuple[str, str, dict[str, Any]]] = [
        ("Equal Weight", "equal_weight", {}),
        ("60/40 Portfolio", "sixty_forty", {}),
        ("Risk Parity", "risk_parity", {}),
        ("Minimum Variance", "minimum_variance", {}),
        ("3D Factor (Stable Betas)", "3d_factor_stable", {
            "max_weight": 0.30,
            "min_observations": 180,
            "blend_to_equal": 0.20,
            "turnover_smoothing": 0.30,
        }),
        ("3D Factor (Rolling Betas)", "3d_factor_rolling", {
            "rolling_window": 252,
            "max_weight": 0.30,
            "min_observations": 180,
            "blend_to_equal": 0.20,
            "turnover_smoothing": 0.40,
            "dynamic_factor_scaling": True,
            "regime_scaling": True,
            "regime_resolver": regime_resolver,
            "regime_scaling_map": regime_scaling_map,
            "regime_factor_multipliers": liquidity_controls,
            "regime_scaling_floor": liquidity_config.floor,
        }),
    ]

    train_results: dict[str, BacktestResult] = {}
    test_results: dict[str, BacktestResult] = {}
    full_results: dict[str, BacktestResult] = {}

    for strategy_display_name, strategy_key, strategy_params in strategies_to_run:
        LOGGER.info("Running backtests", strategy=strategy_display_name)
        params = dict(strategy_params)
        for period_label, config in (
            ("train", train_config),
            ("test", test_config),
            ("full", full_config),
        ):
            try:
                result = _run_single_backtest(
                    dataset=dataset,
                    config=config,
                    strategy_key=strategy_key,
                    strategy_params=params,
                )
            except Exception:
                LOGGER.exception(
                    "Backtest failed",
                    strategy=strategy_display_name,
                    period=period_label,
                )
                continue

            if period_label == "train":
                train_results[strategy_display_name] = result
            elif period_label == "test":
                test_results[strategy_display_name] = result
            else:
                full_results[strategy_display_name] = result

    # === Compute Metrics ===
    risk_free_rate = 0.02
    train_metrics: dict[str, PerformanceMetrics] = {}
    test_metrics: dict[str, PerformanceMetrics] = {}
    overall_metrics: dict[str, PerformanceMetrics] = {}

    def _compute_metrics(
        results: dict[str, BacktestResult],
        benchmark_key: str,
        target: dict[str, PerformanceMetrics],
    ) -> None:
        benchmark_result = results.get(benchmark_key)
        for strategy_name, result in results.items():
            try:
                benchmark = benchmark_result if strategy_name != benchmark_key else None
                target[strategy_name] = calculate_performance_metrics(
                    result=result,
                    benchmark_result=benchmark,
                    risk_free_rate=risk_free_rate,
                )
            except Exception:
                LOGGER.exception(
                    "Failed to compute metrics",
                    strategy=strategy_name,
                    period_label=f"{result.start_date.date()}→{result.end_date.date()}",
                )

    _compute_metrics(train_results, "Equal Weight", train_metrics)
    _compute_metrics(test_results, "Equal Weight", test_metrics)
    _compute_metrics(full_results, "Equal Weight", overall_metrics)

    regime_results: dict[str, RegimeAnalysisResult] = {}
    for strategy_name, result in full_results.items():
        if result.start_date > regimes[0].start or result.end_date < regimes[-1].end:
            LOGGER.info("Skipping regime analysis due to insufficient coverage", strategy=strategy_name)
            continue
        try:
            regime_results[strategy_name] = analyze_strategy_across_regimes(
                result,
                regimes,
                risk_free_rate=risk_free_rate,
            )
        except Exception:
            LOGGER.exception("Regime analysis failed", strategy=strategy_name)

    attribution_results: dict[str, FactorAttribution] = {}
    regime_attribution_results: dict[str, list[FactorAttribution]] = {}
    try:
        for strategy_name, result in test_results.items():
            attribution = _calculate_factor_attribution(result, dataset.factor_returns)
            if attribution is not None:
                attribution_results[strategy_name] = attribution
        for strategy_name, analysis in regime_results.items():
            full_result = full_results.get(strategy_name)
            if full_result is None:
                continue
            for regime_perf in analysis.regime_performances.values():
                regime_attr = _calculate_factor_attribution(
                    full_result,
                    dataset.factor_returns,
                    start=regime_perf.regime.start,
                    end=regime_perf.regime.end,
                    regime_name=regime_perf.regime.name,
                )
                if regime_attr is not None:
                    regime_attribution_results.setdefault(strategy_name, []).append(regime_attr)
    except Exception:
        LOGGER.exception("Factor attribution calculation failed")

    # === Assemble Suite and Persist Outputs ===
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = BacktestSuite(
        strategies=test_results,
        metrics=test_metrics,
        split=split,
        config=test_config,
        train_results=train_results,
        train_metrics=train_metrics,
        overall_metrics=overall_metrics,
        full_results=full_results,
        regime_results=regime_results,
        attribution=attribution_results,
        regime_attribution=regime_attribution_results,
        regime_scaling_map=regime_scaling_map,
        regime_factor_multipliers=liquidity_controls,
        liquidity_contributions=rolling_liquidity,
    )

    comparison_df = suite.compare_strategies()
    comparison_path = output_dir / "performance_comparison_table.csv"
    comparison_df.write_csv(comparison_path)
    LOGGER.info("Comparison table saved", path=str(comparison_path))

    train_vs_test_df = suite.train_vs_test_table()
    if not train_vs_test_df.is_empty():
        train_vs_test_path = output_dir / "train_vs_test_metrics.csv"
        train_vs_test_df.write_csv(train_vs_test_path)
        LOGGER.info("Train vs test metrics saved", path=str(train_vs_test_path))

    regime_summary_df = suite.regime_summary()
    if not regime_summary_df.is_empty():
        regime_summary_path = output_dir / "regime_summary.csv"
        regime_summary_df.write_csv(regime_summary_path)
        LOGGER.info("Regime summary saved", path=str(regime_summary_path))

    if suite.attribution or suite.regime_attribution:
        attribution_dir = output_dir / "attribution"
        attribution_dir.mkdir(parents=True, exist_ok=True)

        for strategy_name, attribution in suite.attribution.items():
            factors = sorted(set(attribution.betas) | set(attribution.factor_contributions))
            safe_name = re.sub(r"[^a-z0-9]+", "_", strategy_name.lower()).strip("_")
            file_name = f"{safe_name}_attribution.csv"
            attribution_path = attribution_dir / file_name
            with attribution_path.open("w", encoding="utf-8", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["factor", "beta", "annualized_contribution"])
                for factor in factors:
                    writer.writerow([
                        factor,
                        attribution.betas.get(factor, float("nan")),
                        attribution.factor_contributions.get(factor, float("nan")),
                    ])
                writer.writerow(["alpha", attribution.alpha, attribution.alpha_annualized])
            LOGGER.info("Factor attribution saved", path=str(attribution_path), strategy=strategy_name)

        if suite.regime_attribution:
            regime_dir = attribution_dir / "regime"
            regime_dir.mkdir(parents=True, exist_ok=True)
            for strategy_name, regime_attrs in suite.regime_attribution.items():
                safe_name = re.sub(r"[^a-z0-9]+", "_", strategy_name.lower()).strip("_")
                file_name = f"{safe_name}_regime_attribution.csv"
                regime_path = regime_dir / file_name
                with regime_path.open("w", encoding="utf-8", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["regime", "factor", "beta", "annualized_contribution", "alpha", "alpha_annualized"])
                    for regime_attr in regime_attrs:
                        factors = sorted(set(regime_attr.betas) | set(regime_attr.factor_contributions))
                        for factor in factors:
                            writer.writerow([
                                regime_attr.regime_name or "",
                                factor,
                                regime_attr.betas.get(factor, float("nan")),
                                regime_attr.factor_contributions.get(factor, float("nan")),
                                regime_attr.alpha,
                                regime_attr.alpha_annualized,
                            ])
                        writer.writerow([
                            regime_attr.regime_name or "",
                            "alpha",
                            float("nan"),
                            float("nan"),
                            regime_attr.alpha,
                            regime_attr.alpha_annualized,
                        ])
                LOGGER.info("Regime factor attribution saved", path=str(regime_path), strategy=strategy_name)

    full_coverage = bool(full_results) and all(
        result.start_date <= regimes[0].start and result.end_date >= regimes[-1].end
        for result in full_results.values()
    )
    if full_coverage:
        try:
            regime_comparison_df = compare_strategies_across_regimes(full_results, regimes)
        except Exception:
            LOGGER.exception("Unable to generate regime comparison table")
        else:
            regime_comparison_path = output_dir / "regime_comparison.csv"
            regime_comparison_df.write_csv(regime_comparison_path)
            LOGGER.info("Regime comparison saved", path=str(regime_comparison_path))

    if overall_metrics:
        overall_rows = [
            {
                "strategy": name,
                "annualized_return": metrics.annualized_return,
                "annualized_volatility": metrics.annualized_volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "max_drawdown": metrics.maximum_drawdown,
                "cumulative_return": metrics.cumulative_return,
            }
            for name, metrics in overall_metrics.items()
        ]
        overall_df = pl.DataFrame(overall_rows)
        overall_path = output_dir / "full_period_metrics.csv"
        overall_df.write_csv(overall_path)
        LOGGER.info("Full-period metrics saved", path=str(overall_path))

    report_filename = f"backtest_results_{split.train_start.year}_{split.test_end.year}.md"
    report_path = output_dir / report_filename
    suite.to_markdown_report(report_path)

    LOGGER.info(
        "Backtest suite completed",
        num_strategies=len(test_results),
        output_dir=str(output_dir),
    )

    return suite


# ===== Helper Functions =====



def _load_sector_dataset(dataset_path: Path) -> SectorDataset:
    """Load sector returns, factor returns, and coverage metadata."""
    from playground.risk_model.dataset import CoverageSummary
    from playground.risk_model.dataset import SectorDataset

    def _read_frame(path: Path) -> pl.DataFrame:
        if path.suffix == ".parquet":
            return pl.read_parquet(path)
        if path.suffix == ".csv":
            return pl.read_csv(path)
        msg = f"Unsupported file format: {path.suffix}"
        raise ValueError(msg)

    def _ensure_timestamp(frame: pl.DataFrame, column: str = "timestamp") -> pl.DataFrame:
        dtype = frame.schema.get(column)
        if dtype is None:
            msg = f"DataFrame missing column '{column}'"
            raise ValueError(msg)
        if dtype == pl.Utf8:
            return frame.with_columns(pl.col(column).str.to_datetime(time_zone="UTC"))
        if hasattr(dtype, "time_zone"):
            time_zone = getattr(dtype, "time_zone", None)
            if time_zone is None:
                return frame.with_columns(pl.col(column).dt.replace_time_zone("UTC"))
            return frame
        return frame.with_columns(pl.col(column).cast(pl.Datetime(time_zone="UTC")))

    if dataset_path.is_dir():
        sector_path = dataset_path / "sector_returns.parquet"
        factor_path = dataset_path / "factor_returns.parquet"
        coverage_path = dataset_path / "coverage_summary.json"
    else:
        sector_path = dataset_path
        factor_path = dataset_path.parent / "factor_returns.parquet"
        coverage_path = dataset_path.parent / "coverage_summary.json"

    if not sector_path.exists():
        raise FileNotFoundError(f"Sector returns file not found: {sector_path}")

    sector_df = _read_frame(sector_path)

    required_cols = {"timestamp", "symbol", "return"}
    if not required_cols.issubset(sector_df.columns):
        missing = required_cols - set(sector_df.columns)
        msg = f"Dataset missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    sector_df = _ensure_timestamp(sector_df, column="timestamp").sort("timestamp")

    if factor_path.exists():
        factor_df = _read_frame(factor_path)
        if "timestamp" not in factor_df.columns:
            msg = "Factor returns file missing 'timestamp' column"
            raise ValueError(msg)
        factor_df = _ensure_timestamp(factor_df, column="timestamp").sort("timestamp")
        factor_columns = [col for col in factor_df.columns if col != "timestamp"]
    else:
        LOGGER.warning(
            "Factor returns file not found; using zero placeholders",
            path=str(factor_path),
        )
        factor_columns = ["factor_duration", "factor_credit", "factor_liquidity"]
        unique_timestamps = sector_df.select("timestamp").unique().sort("timestamp")
        factor_df = unique_timestamps.with_columns([
            pl.lit(0.0).alias(name)
            for name in factor_columns
        ])

    if coverage_path.exists():
        coverage_data = json.loads(coverage_path.read_text(encoding="utf-8"))
        sector_coverage = {
            key: float(value)
            for key, value in coverage_data.get("sector_coverage", {}).items()
        }
        factor_coverage = {
            key: float(value)
            for key, value in coverage_data.get("factor_coverage", {}).items()
        }
        coverage = CoverageSummary(
            calendar_name=coverage_data.get("calendar_name", "XNYS"),
            sector_expected_days=int(coverage_data.get("sector_expected_days", len(sector_df))),
            factor_expected_days=int(coverage_data.get("factor_expected_days", len(factor_df))),
            sector_coverage=sector_coverage,
            factor_coverage=factor_coverage,
            composite_coverage=coverage_data.get("composite_coverage", {}),
        )
    else:
        sectors = sector_df["symbol"].unique().to_list()
        sector_expected = int(sector_df.get_column("timestamp").n_unique())
        factor_expected = int(factor_df.get_column("timestamp").n_unique())
        coverage = CoverageSummary(
            calendar_name="XNYS",
            sector_expected_days=sector_expected,
            factor_expected_days=factor_expected,
            sector_coverage=dict.fromkeys(sectors, 1.0),
            factor_coverage=dict.fromkeys(factor_columns, 1.0),
        )

    dataset = SectorDataset(
        sector_returns=sector_df,
        factor_returns=factor_df,
        coverage=coverage,
    )

    LOGGER.info(
        "Dataset loaded",
        sector_path=str(sector_path),
        factor_path=str(factor_path) if factor_path.exists() else "placeholder",
        num_sectors=len(sector_df["symbol"].unique()),
        num_factor_columns=len([col for col in factor_df.columns if col != "timestamp"]),
        date_range=f"{sector_df['timestamp'].min()!s} → {sector_df['timestamp'].max()!s}",
    )

    return dataset



def _create_backtest_config(
    split: TrainTestSplit,
    overrides: dict[str, Any] | None,
    *,
    period: str = "test",
) -> BacktestConfig:
    r"""
    Create BacktestConfig with defaults and overrides.

    Parameters
    ----------
    split : TrainTestSplit
        Train/test split defining date range
    overrides : dict[str, Any] | None
        Configuration overrides
    period : str
        Which period to cover: ``\"train\"``, ``\"test\"``, or ``\"full\"``

    Returns
    -------
    BacktestConfig
        Backtest configuration
    """
    config_params: dict[str, Any] = {
        "initial_capital": 1_000_000.0,
        "rebalance_frequency": "monthly",
        "transaction_cost_bps": 10.0,
        "slippage_bps": 0.0,
        "position_limits": None,
        "rebalance_threshold": 0.05,
        "random_seed": 42,
    }

    if overrides is not None:
        config_params.update(overrides)

    period_normalized = period.lower()
    if period_normalized == "train":
        config_params["start_date"] = split.train_start
        config_params["end_date"] = split.train_end
    elif period_normalized == "test":
        config_params["start_date"] = split.test_start
        config_params["end_date"] = split.test_end
    elif period_normalized == "full":
        config_params["start_date"] = split.train_start
        config_params["end_date"] = split.test_end
    else:
        msg = f"Unknown period '{period}' for backtest config"
        raise ValueError(msg)

    return BacktestConfig(**config_params)


def _run_single_backtest(
    dataset: SectorDataset,
    config: BacktestConfig,
    strategy_key: str,
    strategy_params: dict[str, Any],
) -> BacktestResult:
    """
    Run backtest for a single strategy.

    Parameters
    ----------
    dataset : SectorDataset
        Sector dataset with returns and factors
    config : BacktestConfig
        Backtest configuration
    strategy_key : str
        Strategy identifier (for FactorBacktester)
    strategy_params : dict[str, Any]
        Strategy-specific parameters

    Returns
    -------
    BacktestResult
        Backtest result

    Notes
    -----
    Strategy keys map to compute_weights implementations:
    - "equal_weight": Equal-weight allocation
    - "sixty_forty": 60/40 portfolio (via benchmarks.py)
    - "risk_parity": Risk parity (via benchmarks.py)
    - "minimum_variance": Minimum variance (via benchmarks.py)
    - "3d_factor_stable": 3D Factor with stable betas
    - "3d_factor_rolling": 3D Factor with rolling betas
    """
    backtester = FactorBacktester(config)

    # Handle benchmark strategies that need custom implementation
    if strategy_key == "sixty_forty":
        return _run_benchmark_strategy(
            backtester=backtester,
            dataset=dataset,
            strategy_name="60/40 Portfolio",
            strategy_class=SixtyFortyStrategy,
            strategy_params=strategy_params,
        )
    elif strategy_key == "risk_parity":
        return _run_benchmark_strategy(
            backtester=backtester,
            dataset=dataset,
            strategy_name="Risk Parity",
            strategy_class=RiskParityStrategy,
            strategy_params=strategy_params,
        )
    elif strategy_key == "minimum_variance":
        return _run_benchmark_strategy(
            backtester=backtester,
            dataset=dataset,
            strategy_name="Minimum Variance",
            strategy_class=MinimumVarianceStrategy,
            strategy_params=strategy_params,
        )
    else:
        # Use FactorBacktester's built-in strategy dispatch
        return backtester.run_backtest(
            dataset=dataset,
            strategy=strategy_key,
            strategy_params=strategy_params,
        )


def _calculate_factor_attribution(
    result: BacktestResult,
    factor_returns: pl.DataFrame,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    regime_name: str | None = None,
) -> FactorAttribution | None:
    """Compute linear factor attribution for a backtest result or sub-window."""
    if not result.returns:
        return None

    window_start = start or result.start_date
    window_end = end or result.end_date

    filtered_factors = factor_returns.filter(
        (pl.col("timestamp") >= window_start)
        & (pl.col("timestamp") <= window_end)
    ).sort("timestamp")

    factor_columns = [col for col in filtered_factors.columns if col != "timestamp"]
    if not factor_columns or filtered_factors.is_empty():
        return None

    filtered_dates: list[datetime] = []
    filtered_returns: list[float] = []
    for dt, ret in zip(result.dates[1:], result.returns):
        if window_start <= dt <= window_end:
            filtered_dates.append(dt)
            filtered_returns.append(ret)

    if len(filtered_returns) < len(factor_columns) + 1:
        return None

    strategy_frame = pl.DataFrame({
        "timestamp": filtered_dates,
        "strategy_return": filtered_returns,
    })

    joined = strategy_frame.join(filtered_factors, on="timestamp", how="inner").drop_nulls()
    if joined.height < len(factor_columns) + 1:
        return None

    y = joined.get_column("strategy_return").to_numpy()
    X = np.column_stack([joined.get_column(col).to_numpy() for col in factor_columns])
    X = np.column_stack([X, np.ones(len(y))])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        LOGGER.warning("Attribution regression failed (singular matrix)", strategy=result.strategy_name)
        return None

    betas = {
        factor_columns[i]: float(coeffs[i])
        for i in range(len(factor_columns))
    }
    alpha = float(coeffs[-1])
    alpha_annualized = alpha * 252.0

    factor_contributions: dict[str, float] = {}
    for factor in factor_columns:
        mean_value = joined.get_column(factor).mean()
        mean_return = float(cast(float, mean_value)) if mean_value is not None else 0.0
        factor_contributions[factor] = betas[factor] * mean_return * 252.0

    y_hat = X @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return FactorAttribution(
        strategy_name=result.strategy_name,
        regime_name=regime_name,
        betas=betas,
        factor_contributions=factor_contributions,
        alpha=alpha,
        alpha_annualized=alpha_annualized,
        r_squared=r_squared,
    )


def _run_benchmark_strategy(
    backtester: FactorBacktester,
    dataset: SectorDataset,
    strategy_name: str,
    strategy_class: type,
    strategy_params: dict[str, Any],
) -> BacktestResult:
    """
    Run backtest using a benchmark strategy from benchmarks.py.

    This helper creates a custom strategy instance and integrates it
    with the backtester by overriding weight computation.

    Parameters
    ----------
    backtester : FactorBacktester
        Backtester instance
    dataset : SectorDataset
        Sector dataset
    strategy_name : str
        Display name for the strategy
    strategy_class : type
        Strategy class (e.g., SixtyFortyStrategy)
    strategy_params : dict[str, Any]
        Parameters for strategy initialization

    Returns
    -------
    BacktestResult
        Backtest result
    """
    # Create strategy instance
    benchmark_strategy = strategy_class(**strategy_params)

    # Patch backtester to use custom strategy
    original_compute_weights = backtester._compute_target_weights

    def custom_compute_weights(
        self: FactorBacktester,
        strategy: str,
        date: datetime,
        dataset: SectorDataset,
        sectors: list[str],
        params: dict[str, object],
    ) -> dict[str, float]:
        _ = (self, strategy, sectors, params)
        weights: dict[str, float] = benchmark_strategy.compute_weights(date, dataset)
        return weights

    backtester._compute_target_weights = MethodType(custom_compute_weights, backtester)  # type: ignore[method-assign]

    try:
        result = backtester.run_backtest(
            dataset=dataset,
            strategy="custom",  # Dummy key
            strategy_params={},
        )
        # Override strategy name in result
        result.strategy_name = strategy_name
        return result
    finally:
        # Restore original method
        backtester._compute_target_weights = original_compute_weights  # type: ignore[method-assign]


def _build_regime_resolver(
    regimes: Sequence[MarketRegime],
) -> Callable[[datetime], str | None]:
    """Return a callable that resolves the active regime for a given timestamp."""
    def resolver(date: datetime) -> str | None:
        comparison_date = date.astimezone(UTC) if date.tzinfo is not None else date.replace(tzinfo=UTC)
        for regime in regimes:
            if regime.start <= comparison_date <= regime.end:
                return regime.name
        return None

    return resolver


# ===== Public API =====

__all__ = [
    "BacktestSuite",
    "run_full_backtest_suite",
]
@dataclass(slots=True)
class FactorAttribution:
    """Summary of factor attribution for a strategy."""

    strategy_name: str
    betas: dict[str, float]
    factor_contributions: dict[str, float]
    alpha: float
    alpha_annualized: float
    r_squared: float
    regime_name: str | None = None
