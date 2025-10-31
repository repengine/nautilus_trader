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
import itertools
import json
import math
import re
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, TextIO, TypedDict, cast

import numpy as np
import polars as pl
import structlog

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.playground import DiagnosticsDefaults
from ml.config.playground import MonitoringExportDefaults
from ml.config.playground import MonteCarloShockOverlayDefaults
from ml.config.playground import MonteCarloStressDefaults
from ml.config.playground import NestedWalkForwardDefaults
from ml.config.playground import ParameterHeatmapSpecDefaults
from ml.config.playground import ParameterHeatmapSuiteDefaults
from ml.config.playground import ParameterSensitivitySpecDefaults
from ml.config.playground import ParameterSensitivitySuiteDefaults
from ml.config.playground import ParameterValue
from ml.config.playground import ProxyDatasetSpecDefaults
from ml.config.playground import ProxyDatasetSuiteDefaults
from ml.config.playground import ThreeDRiskBacktestDefaults
from ml.config.playground import VintageWindowDefaults
from ml.config.playground import WalkForwardPermutationDefaults
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
from playground.backtest.splits import WalkForwardConfig
from playground.backtest.splits import define_train_test_split
from playground.backtest.splits import validate_no_lookahead
from playground.backtest.splits import validate_sufficient_training_data
from playground.scripts.export_phase3_visuals import export_phase3_visuals


if TYPE_CHECKING:
    from playground.backtest.engine import BacktestResult
    from playground.backtest.regime_analysis import MarketRegime
    from playground.risk_model.dataset import SectorDataset


LOGGER = structlog.get_logger(__name__)


_COMPARISON_SCHEMA = {
    "strategy": pl.Utf8,
    "annualized_return": pl.Float64,
    "annualized_volatility": pl.Float64,
    "sharpe_ratio": pl.Float64,
    "sortino_ratio": pl.Float64,
    "calmar_ratio": pl.Float64,
    "max_drawdown": pl.Float64,
    "information_ratio": pl.Float64,
    "num_rebalances": pl.Float64,
    "transaction_costs": pl.Float64,
    "transaction_costs_pct": pl.Float64,
    "turnover_rate": pl.Float64,
}


LIQUIDITY_ATTRIBUTION_DIR = Path("playground/reports/backtesting/attribution/regime")
LIQUIDITY_STRATEGY_SLUG = "3d_factor_rolling_betas"
BACKTEST_DEFAULTS = ThreeDRiskBacktestDefaults()
TRADING_DAY_RATIO = 252 / 365.25
THREE_D_ROLLING_STRATEGY = "3D Factor (Rolling Betas)"
PHASE3_TARGET_SHARPE = 0.50
MONTE_CARLO_SHARPE_HIST = get_histogram(
    "phase3_monte_carlo_sharpe",
    "Sharpe ratio distribution from Phase 3 Monte Carlo stress sweeps",
    labelnames=("strategy",),
    buckets=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
)
MONTE_CARLO_DRAWDOWN_HIST = get_histogram(
    "phase3_monte_carlo_drawdown",
    "Max drawdown distribution from Phase 3 Monte Carlo stress sweeps",
    labelnames=("strategy",),
    buckets=(-0.8, -0.6, -0.4, -0.3, -0.2, -0.1, -0.05, 0.0),
)
MONTE_CARLO_OVERLAY_COUNTER = get_counter(
    "phase3_monte_carlo_overlay_activations_total",
    "Count of Monte Carlo overlay activations during Phase 3 stress sweeps.",
    labelnames=("strategy", "overlay", "category"),
)
HEATMAP_METRIC_HIST = get_histogram(
    "phase3_parameter_heatmap_metric",
    "Performance metric distribution captured during Phase 3 parameter heatmaps",
    labelnames=("strategy", "spec"),
    buckets=(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
)
SENSITIVITY_METRIC_HIST = get_histogram(
    "phase4_parameter_sensitivity_metric",
    "Performance metric distribution captured during Phase 4 parameter sensitivity sweeps",
    labelnames=("strategy", "spec", "metric"),
    buckets=(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
)
SENSITIVITY_DELTA_COUNTER = get_counter(
    "phase4_sensitivity_sharpe_delta_breach_total",
    "Count of Phase 4 sensitivity specs exceeding Sharpe delta tolerance.",
    labelnames=("spec",),
)
DIAGNOSTIC_TAIL_HIST = get_histogram(
    "phase3_tail_risk_metric",
    "Tail risk metrics observed during Phase 3 diagnostics",
    labelnames=("strategy", "quantile"),
    buckets=(-0.10, -0.08, -0.06, -0.04, -0.02, -0.01, 0.0),
)
PROXY_STATUS_COUNTER = get_counter(
    "phase3_proxy_dataset_status_total",
    "Count of proxy dataset validation outcomes grouped by status.",
    labelnames=("status", "dataset"),
)
VINTAGE_STATUS_COUNTER = get_counter(
    "phase3_vintage_simulation_status_total",
    "Count of vintage simulation outcomes grouped by status.",
    labelnames=("status", "window"),
)
MONITORING_EXPORT_COUNTER = get_counter(
    "phase3_monitoring_snapshot_total",
    "Number of monitoring snapshots generated during Phase 3 runs.",
    labelnames=("status",),
)


class _ProxyMetadataEntry(TypedDict):
    status: str
    allow_missing: bool
    tags: list[str]
    message: str | None


class _VintageWindowMetadataEntry(TypedDict):
    slug: str
    label: str
    status: str
    fold_count: int
    min_folds: int
    message: str | None


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
    turnover_overrides : dict[str, float]
        Turnover smoothing overrides applied per strategy key (e.g., "3d_factor_rolling")
    baseline_strategies : tuple[str, ...]
        Strategies highlighted in benchmark summaries (default Equal Weight, 60/40, Risk Parity)

    Methods
    -------
    compare_strategies() -> pl.DataFrame
        Generate comparison table of all strategies
    to_markdown_report(output_path: Path) -> None
        Generate markdown report with results
    benchmark_summary() -> pl.DataFrame
        Generate condensed comparison for canonical benchmark strategies

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
    turnover_overrides: dict[str, float] = field(default_factory=dict)
    baseline_strategies: tuple[str, ...] = BACKTEST_DEFAULTS.baseline_strategies

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
                "num_rebalances": float(metrics.num_rebalances),
                "transaction_costs": metrics.transaction_costs_total,
                "transaction_costs_pct": metrics.transaction_costs_pct,
                "turnover_rate": metrics.turnover_rate,
            }
            rows.append(row)

        if not rows:
            return pl.DataFrame(schema=_COMPARISON_SCHEMA)

        df = pl.DataFrame(rows).select(list(_COMPARISON_SCHEMA.keys()))

        # Sort by Sharpe ratio descending
        df = df.sort("sharpe_ratio", descending=True)

        return df

    def benchmark_summary(self) -> pl.DataFrame:
        """
        Generate a compact summary for canonical benchmark strategies.

        Returns
        -------
        pl.DataFrame
            Table containing Sharpe, return, volatility, drawdown metrics, and an
            availability status for strategies listed in ``baseline_strategies``.

        Examples
        --------
        >>> suite = run_full_backtest_suite(dataset_path, output_dir)
        >>> suite.benchmark_summary()
        shape: (3, 6)
        ┌──────────────┬────────────┬─────────────┬────────────┬─────────────┬───────────────┐
        │ strategy     ┆ sharpe_ra… ┆ annualized… ┆ annualize… ┆ max_drawdo… ┆ cumulative_r… │
        │ ---          ┆ ---        ┆ ---         ┆ ---        ┆ ---         ┆ ---           │
        │ str          ┆ f64        ┆ f64         ┆ f64        ┆ f64         ┆ f64           │
        ╞══════════════╪════════════╪═════════════╪════════════╪═════════════╪═══════════════╡
        │ Equal Weight ┆ 0.55       ┆ 9.87        ┆ 14.11      ┆ -19.23      ┆ 98.10        │
        └──────────────┴────────────┴─────────────┴────────────┴─────────────┴───────────────┘
        """
        rows: list[dict[str, float | str]] = []
        seen: set[str] = set()

        for strategy in self.baseline_strategies:
            if strategy in seen:
                continue
            metrics = self.metrics.get(strategy)
            if metrics is None:
                rows.append({
                    "strategy": strategy,
                    "sharpe_ratio": float("nan"),
                    "annualized_return": float("nan"),
                    "annualized_volatility": float("nan"),
                    "max_drawdown": float("nan"),
                    "cumulative_return": float("nan"),
                    "status": "missing",
                })
            else:
                rows.append({
                    "strategy": strategy,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "annualized_return": metrics.annualized_return * 100,
                    "annualized_volatility": metrics.annualized_volatility * 100,
                    "max_drawdown": metrics.maximum_drawdown * 100,
                    "cumulative_return": metrics.cumulative_return * 100,
                    "status": "available",
                })
            seen.add(strategy)

        if not rows:
            return pl.DataFrame(schema={
                "strategy": pl.Utf8,
                "sharpe_ratio": pl.Float64,
                "annualized_return": pl.Float64,
                "annualized_volatility": pl.Float64,
                "max_drawdown": pl.Float64,
                "cumulative_return": pl.Float64,
                "status": pl.Utf8,
            })

        return pl.DataFrame(rows)

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

            benchmark_df = self.benchmark_summary()
            if not benchmark_df.is_empty():
                f.write("## Benchmark Snapshot\n\n")
                missing_baselines = benchmark_df.filter(pl.col("status") == "missing")
                if not missing_baselines.is_empty():
                    missing_names = ", ".join(missing_baselines.get_column("strategy").to_list())
                    f.write(
                        f"> Note: Metrics unavailable for baseline strategies: {missing_names}.\n\n"
                    )
                f.write(self._format_dataframe_as_markdown(benchmark_df))
                f.write("\n\n")

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

                visuals_dir = output_path.parent / "visuals"
                visual_entries = [
                    ("Rolling vs Benchmark Sharpe", "rolling_vs_benchmark_sharpe.png"),
                    ("Regime Factor Contributions", "regime_contributions.png"),
                    ("Liquidity Stress Panel", "liquidity_stress_panel.png"),
                    ("Sharpe vs Transaction Costs", "sharpe_vs_tc.png"),
                    ("Attribution Waterfall", "attribution_waterfall.png"),
                ]
                available_visuals = [
                    (title, filename)
                    for title, filename in visual_entries
                    if (visuals_dir / filename).exists()
                ]
                if available_visuals:
                    f.write("## Visual Highlights\n\n")
                    for title, filename in available_visuals:
                        f.write(f"![{title}](visuals/{filename})\n\n")

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


@dataclass(slots=True)
class WalkForwardBacktestResult:
    """Encapsulate walk-forward suites and aggregated diagnostics."""

    splits: list[TrainTestSplit]
    suites: list[BacktestSuite]
    _aggregate_cache: pl.DataFrame | None = field(default=None, init=False, repr=False)
    _summary_cache: pl.DataFrame | None = field(default=None, init=False, repr=False)

    def aggregate_metrics(self) -> pl.DataFrame:
        """
        Stack per-fold strategy metrics (test period) into a single frame.
        """
        if self._aggregate_cache is None:
            frames: list[pl.DataFrame] = []
            for index, (suite, split) in enumerate(zip(self.suites, self.splits), start=1):
                comparison = suite.compare_strategies()
                if comparison.is_empty():
                    continue
                comparison = comparison.with_columns(
                    pl.lit(index).alias("fold"),
                    pl.lit(split.test_start.date().isoformat()).alias("test_start"),
                    pl.lit(split.test_end.date().isoformat()).alias("test_end"),
                )
                comparison = comparison.join(
                    pl.DataFrame({
                        "strategy": list(suite.baseline_strategies),
                        "baseline": [True] * len(suite.baseline_strategies),
                    }),
                    on="strategy",
                    how="left",
                ).with_columns(
                    pl.col("baseline").fill_null(False),
                )
                frames.append(comparison)
            self._aggregate_cache = (
                pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
            )
        return self._aggregate_cache.clone()

    def summarize_metrics(self) -> pl.DataFrame:
        """
        Compute strategy-level summary statistics across walk-forward folds.
        """
        if self._summary_cache is None:
            aggregated = self.aggregate_metrics()
            if aggregated.is_empty():
                self._summary_cache = pl.DataFrame()
            else:
                self._summary_cache = aggregated.group_by("strategy").agg(
                    [
                        pl.col("fold").count().alias("num_folds"),
                        pl.col("sharpe_ratio").mean().alias("sharpe_ratio_mean"),
                        pl.col("sharpe_ratio").std().alias("sharpe_ratio_std"),
                        pl.col("sharpe_ratio").min().alias("sharpe_ratio_min"),
                        pl.col("sharpe_ratio").max().alias("sharpe_ratio_max"),
                        pl.col("annualized_return").mean().alias("annualized_return_mean"),
                        pl.col("annualized_volatility").mean().alias("annualized_volatility_mean"),
                    ]
                ).sort("strategy")
        return self._summary_cache.clone()

    def write_summaries(
        self,
        directory: Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """
        Persist aggregated fold metrics and summary tables to CSV.
        """
        directory.mkdir(parents=True, exist_ok=True)
        aggregate = self.aggregate_metrics()
        if not aggregate.is_empty():
            aggregate.write_csv(directory / "aggregate_metrics.csv")
        summary = self.summarize_metrics()
        if not summary.is_empty():
            summary.write_csv(directory / "strategy_summary.csv")
        if metadata is not None:
            metadata_path = directory / "metadata.json"
            metadata_json = json.dumps(metadata, indent=2, sort_keys=True)
            metadata_path.write_text(metadata_json, encoding="utf-8")


@dataclass(slots=True)
class NestedCrossValidationResult:
    """
    Encapsulate a nested walk-forward sweep executed within an outer fold.

    Attributes
    ----------
    spec : NestedWalkForwardDefaults
        Nested configuration metadata.
    outer_fold_index : int
        Index (1-based) of the outer fold associated with this nested sweep.
    outer_split : TrainTestSplit
        Outer fold split describing the training/testing periods.
    config : WalkForwardConfig
        Materialised configuration used to generate nested splits.
    result : WalkForwardBacktestResult
        Aggregated nested walk-forward results.
    output_directory : Path
        Directory containing persisted artefacts for this nested sweep.
    """

    spec: NestedWalkForwardDefaults
    outer_fold_index: int
    outer_split: TrainTestSplit
    config: WalkForwardConfig
    result: WalkForwardBacktestResult
    output_directory: Path

    def summarize_metrics(self) -> pl.DataFrame:
        """
        Return nested summary metrics with outer fold annotations.
        """
        summary = self.result.summarize_metrics()
        if summary.is_empty():
            return summary
        return summary.with_columns(
            pl.lit(self.outer_fold_index).alias("outer_fold"),
            pl.lit(self.config.train_years).alias("inner_train_years"),
            pl.lit(self.config.test_years).alias("inner_test_years"),
            pl.lit(self.config.step_years).alias("inner_step_years"),
            pl.lit(self.outer_split.train_start.date().isoformat()).alias("outer_train_start"),
            pl.lit(self.outer_split.train_end.date().isoformat()).alias("outer_train_end"),
        )

    def aggregate_metrics(self) -> pl.DataFrame:
        """
        Return concatenated per-fold metrics annotated with outer fold index.
        """
        aggregate = self.result.aggregate_metrics()
        if aggregate.is_empty():
            return aggregate
        return aggregate.with_columns(
            pl.lit(self.outer_fold_index).alias("outer_fold"),
            pl.lit(self.config.train_years).alias("inner_train_years"),
            pl.lit(self.config.test_years).alias("inner_test_years"),
            pl.lit(self.config.step_years).alias("inner_step_years"),
        )


@dataclass(slots=True)
class WalkForwardPermutationRun:
    """
    Container for a single walk-forward permutation execution.

    Attributes
    ----------
    spec : WalkForwardPermutationDefaults
        Permutation metadata.
    config : WalkForwardConfig
        Outer walk-forward configuration used for the run.
    outer_result : WalkForwardBacktestResult
        Aggregated outer walk-forward results.
    summary_directory : Path
        Directory containing outer walk-forward artefacts.
    nested_results : list[NestedCrossValidationResult]
        Optional nested sweeps executed within each outer fold.
    """

    spec: WalkForwardPermutationDefaults
    config: WalkForwardConfig
    outer_result: WalkForwardBacktestResult
    summary_directory: Path
    nested_results: list[NestedCrossValidationResult] = field(default_factory=list)

    def outer_summary(self) -> pl.DataFrame:
        """
        Return strategy-level summary metrics for the outer sweep.
        """
        summary = self.outer_result.summarize_metrics()
        if summary.is_empty():
            return summary
        return summary.with_columns(
            pl.lit(self.spec.slug).alias("permutation_slug"),
            pl.lit(self.spec.name).alias("permutation_name"),
            pl.lit(self.spec.train_years).alias("train_years"),
            pl.lit(self.spec.test_years).alias("test_years"),
            pl.lit(self.spec.step_years).alias("step_years"),
        )

    def nested_summary(self) -> pl.DataFrame:
        """
        Return concatenated nested summaries across outer folds.
        """
        frames: list[pl.DataFrame] = []
        for nested in self.nested_results:
            summary = nested.summarize_metrics()
            if summary.is_empty():
                continue
            summary = summary.with_columns(
                pl.lit(self.spec.slug).alias("permutation_slug"),
                pl.lit(self.spec.name).alias("permutation_name"),
            )
            frames.append(summary)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed")

    def nested_rollup(self) -> pl.DataFrame:
        """
        Aggregate nested summaries by strategy and outer fold.
        """
        summary = self.nested_summary()
        if summary.is_empty():
            return summary
        return summary.group_by(
            [
                "permutation_slug",
                "permutation_name",
                "strategy",
                "outer_fold",
            ]
        ).agg(
            [
                pl.col("sharpe_ratio").mean().alias("sharpe_ratio_mean"),
                pl.col("sharpe_ratio").std().alias("sharpe_ratio_std"),
                pl.col("annualized_return").mean().alias("annualized_return_mean"),
                pl.col("annualized_volatility").mean().alias("annualized_volatility_mean"),
            ]
        ).sort(["permutation_slug", "strategy", "outer_fold"])


@dataclass(slots=True)
class MultiHorizonWalkForwardResult:
    """
    Summary of all executed walk-forward permutations.

    Attributes
    ----------
    base_directory : Path
        Base output directory supplied to the runner.
    runs : dict[str, WalkForwardPermutationRun]
        Mapping of permutation slug to run artefacts/results.
    """

    base_directory: Path
    runs: dict[str, WalkForwardPermutationRun]

    def summary_table(self) -> pl.DataFrame:
        """
        Return concatenated outer summaries across permutations.
        """
        frames: list[pl.DataFrame] = []
        for run in self.runs.values():
            summary = run.outer_summary()
            if summary.is_empty():
                continue
            frames.append(summary)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed").sort(
            ["permutation_slug", "strategy"]
        )

    def nested_summary(self) -> pl.DataFrame:
        """
        Concatenate nested summaries across all permutations.
        """
        frames: list[pl.DataFrame] = []
        for run in self.runs.values():
            summary = run.nested_summary()
            if summary.is_empty():
                continue
            frames.append(summary)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed").sort(
            ["permutation_slug", "outer_fold", "strategy"]
        )

    def nested_rollup(self) -> pl.DataFrame:
        """
        Aggregate nested summaries across permutations.
        """
        frames: list[pl.DataFrame] = []
        for run in self.runs.values():
            summary = run.nested_rollup()
            if summary.is_empty():
                continue
            frames.append(summary)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed").sort(
            ["permutation_slug", "outer_fold", "strategy"]
        )

    def write_summary(self) -> None:
        """
        Persist aggregated permutation summaries to CSV artefacts.
        """
        base = self.base_directory / "walk_forward"
        base.mkdir(parents=True, exist_ok=True)
        outer_summary = self.summary_table()
        if not outer_summary.is_empty():
            outer_summary.write_csv(base / "permutation_summary.csv")
        nested_summary = self.nested_summary()
        if not nested_summary.is_empty():
            nested_summary.write_csv(base / "nested_summary.csv")
        nested_rollup = self.nested_rollup()
        if not nested_rollup.is_empty():
            nested_rollup.write_csv(base / "nested_rollup.csv")


# ===== Monte Carlo Stress Testing =====


@dataclass(slots=True)
class MonteCarloOverlayActivation:
    """
    Structured metadata describing a triggered Monte Carlo overlay.

    Attributes
    ----------
    name : str
        Overlay identifier.
    category : str
        Overlay category (rates, growth, liquidity, etc.).
    start_index : int
        Zero-based index within the synthetic path where the overlay activates.
    duration : int
        Number of days the overlay remains active (bounded by path length).
    regime : str | None
        Regime label active when the overlay triggers, if determinable.
    magnitude : float
        Initial additive shock applied on the first activation day.
    decay : float
        Decay multiplier applied for subsequent days.
    total_impact : float
        Aggregate additive impact applied across all active days.
    tags : tuple[str, ...]
        Optional structured tags for reporting/alert routing.
    """

    name: str
    category: str
    start_index: int
    duration: int
    regime: str | None
    magnitude: float
    decay: float
    total_impact: float
    tags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """
        Convert the activation metadata into a serialisable dictionary.
        """
        return {
            "name": self.name,
            "category": self.category,
            "start_index": self.start_index,
            "duration": self.duration,
            "regime": self.regime,
            "magnitude": self.magnitude,
            "decay": self.decay,
            "total_impact": self.total_impact,
            "tags": list(self.tags),
        }


@dataclass(slots=True)
class _OverlayCategoryAccumulator:
    """Mutable accumulator for Monte Carlo overlay category summaries."""

    activation_count: int = 0
    total_impact: float = 0.0
    overlay_names: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)


@dataclass(slots=True)
class MonteCarloStressPathResult:
    """
    Result of a single Monte Carlo stress simulation path.

    Attributes
    ----------
    simulation_id : int
        Sequential identifier for the simulation path (1-based).
    sharpe_ratio : float
        Annualised Sharpe ratio for the stressed return path.
    annualized_return : float
        Annualised total return for the path.
    annualized_volatility : float
        Annualised volatility derived from daily returns.
    max_drawdown : float
        Worst peak-to-trough drawdown observed during the path.
    var_alpha : float
        Value-at-Risk at the configured confidence level.
    cvar_alpha : float
        Conditional Value-at-Risk (Expected Shortfall) at the confidence level.
    terminal_value : float
        Terminal wealth assuming an initial value of 1.0.
    positive_terminal : bool
        Indicator that terminal value exceeded the starting value.
    regime_sequence : tuple[str, ...]
        Sequence of regime names used to construct the synthetic path.
    overlay_events : tuple[MonteCarloOverlayActivation, ...]
        Overlay activations applied during the simulation.
    path_length : int
        Number of return observations in the synthetic path.
    """

    simulation_id: int
    sharpe_ratio: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    var_alpha: float
    cvar_alpha: float
    terminal_value: float
    positive_terminal: bool
    regime_sequence: tuple[str, ...]
    overlay_events: tuple[MonteCarloOverlayActivation, ...]
    path_length: int

    @property
    def overlay_total_impact(self) -> float:
        """Return aggregate impact from all overlays applied in the path."""
        if not self.overlay_events:
            return 0.0
        return float(sum(event.total_impact for event in self.overlay_events))

    def regime_signature(self) -> str:
        """
        Return a hyphenated signature of the regime sequence.
        """
        return "-".join(self.regime_sequence)

    def as_dict(self) -> dict[str, object]:
        """
        Convert the path result into a serialisable dictionary.
        """
        return {
            "simulation_id": self.simulation_id,
            "sharpe_ratio": self.sharpe_ratio,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "max_drawdown": self.max_drawdown,
            "var_alpha": self.var_alpha,
            "cvar_alpha": self.cvar_alpha,
            "terminal_value": self.terminal_value,
            "positive_terminal": self.positive_terminal,
            "regime_sequence": list(self.regime_sequence),
            "overlay_events": [event.to_dict() for event in self.overlay_events],
            "path_length": self.path_length,
            "regime_signature": self.regime_signature(),
            "overlay_count": len(self.overlay_events),
            "overlay_names": [event.name for event in self.overlay_events],
            "overlay_categories": sorted({event.category for event in self.overlay_events}),
            "overlay_total_impact": self.overlay_total_impact,
        }


@dataclass(slots=True)
class MonteCarloStressSuiteResult:
    """
    Aggregated results from the Monte Carlo stress sweep.

    Attributes
    ----------
    base_directory : Path
        Output directory provided to the runner.
    stress_config : MonteCarloStressDefaults
        Configuration used to drive the stress sweep.
    target_strategy : str
        Name of the stressed strategy.
    baseline_metrics : PerformanceMetrics
        Baseline metrics for the unstressed strategy.
    paths : tuple[MonteCarloStressPathResult, ...]
        Collection of per-path results.
    """

    base_directory: Path
    stress_config: MonteCarloStressDefaults
    target_strategy: str
    baseline_metrics: PerformanceMetrics
    paths: tuple[MonteCarloStressPathResult, ...]

    def to_frame(self) -> pl.DataFrame:
        """
        Convert path results into a Polars DataFrame.
        """
        if not self.paths:
            return pl.DataFrame()
        return pl.DataFrame([path.as_dict() for path in self.paths])

    def summary_frame(self) -> pl.DataFrame:
        """
        Compute distributional summary statistics across simulations.
        """
        frame = self.to_frame()
        if frame.is_empty():
            return frame
        quantiles = list(self.stress_config.report_quantiles)
        summary_columns = [
            pl.lit(self.target_strategy).alias("strategy"),
            pl.lit(self.stress_config.num_paths).alias("num_paths"),
            pl.col("sharpe_ratio").mean().alias("sharpe_mean"),
            pl.col("sharpe_ratio").std().alias("sharpe_std"),
            pl.col("annualized_return").mean().alias("return_mean"),
            pl.col("annualized_volatility").mean().alias("volatility_mean"),
            pl.col("max_drawdown").mean().alias("max_drawdown_mean"),
            pl.col("positive_terminal").mean().alias("positive_terminal_rate"),
            pl.col("terminal_value").mean().alias("terminal_value_mean"),
            pl.col("overlay_count").mean().alias("overlay_count_mean"),
        ]
        existing_aliases = {
            "strategy",
            "num_paths",
            "sharpe_mean",
            "sharpe_std",
            "return_mean",
            "volatility_mean",
            "max_drawdown_mean",
            "positive_terminal_rate",
            "terminal_value_mean",
            "overlay_count_mean",
        }
        for metric in self.stress_config.report_metrics:
            if metric not in frame.columns:
                continue
            alias = f"{metric}_mean"
            if alias in existing_aliases:
                continue
            summary_columns.append(pl.col(metric).mean().alias(alias))
            existing_aliases.add(alias)
        for q in quantiles:
            label = f"sharpe_p{int(q * 100):02d}"
            summary_columns.append(pl.col("sharpe_ratio").quantile(q).alias(label))
        return frame.select(summary_columns)

    def overlay_summary_frame(self) -> pl.DataFrame:
        """
        Summarise overlay activation counts and aggregate impacts.
        """
        rows: list[dict[str, object]] = []
        for path in self.paths:
            for event in path.overlay_events:
                rows.append({
                    "name": event.name,
                    "category": event.category,
                    "regime": event.regime,
                    "duration": event.duration,
                    "total_impact": event.total_impact,
                    "tags": "|".join(sorted(event.tags)),
                })
        if not rows:
            return pl.DataFrame()
        frame = pl.DataFrame(rows)
        aggregated = frame.group_by("name", "category").agg(
            [
                pl.len().alias("activation_count"),
                pl.mean("duration").alias("mean_duration"),
                pl.max("duration").alias("max_duration"),
                pl.sum("total_impact").alias("total_impact"),
                pl.first("tags").alias("tags"),
            ],
        )
        return aggregated.sort(["category", "name"])

    def overlay_category_summary_frame(self) -> pl.DataFrame:
        """
        Summarise overlay activation statistics aggregated by category.
        """
        category_rows: dict[str, _OverlayCategoryAccumulator] = {}
        for path in self.paths:
            for event in path.overlay_events:
                accumulator = category_rows.setdefault(event.category, _OverlayCategoryAccumulator())
                accumulator.activation_count += 1
                accumulator.total_impact += event.total_impact
                accumulator.overlay_names.add(event.name)
                accumulator.tags.update(event.tags)
        if not category_rows:
            return pl.DataFrame()
        records: list[dict[str, object]] = []
        for category, accumulator in sorted(category_rows.items()):
            records.append({
                "category": category,
                "activation_count": accumulator.activation_count,
                "total_impact": accumulator.total_impact,
                "overlay_names": "|".join(sorted(accumulator.overlay_names)),
                "tags": "|".join(sorted(accumulator.tags)),
            })
        return pl.DataFrame(records)

    def overlay_category_summary(self) -> list[dict[str, object]]:
        """
        Convert overlay category summary to a list of dictionaries.

        Returns
        -------
        list[dict[str, object]]
            Overlay category activation statistics for telemetry exports.
        """
        frame = self.overlay_category_summary_frame()
        if frame.is_empty():
            return []
        return frame.sort(["category"]).to_dicts()

    def baseline_metrics_dict(self) -> dict[str, object]:
        """
        Serialise baseline performance metrics for downstream dashboards.
        """
        payload = {
            "strategy": self.target_strategy,
            "annualized_return": self.baseline_metrics.annualized_return,
            "annualized_volatility": self.baseline_metrics.annualized_volatility,
            "cumulative_return": self.baseline_metrics.cumulative_return,
            "maximum_drawdown": self.baseline_metrics.maximum_drawdown,
            "sharpe_ratio": self.baseline_metrics.sharpe_ratio,
            "sortino_ratio": self.baseline_metrics.sortino_ratio,
            "calmar_ratio": self.baseline_metrics.calmar_ratio,
            "var_95": self.baseline_metrics.var_95,
            "var_99": self.baseline_metrics.var_99,
            "cvar_95": self.baseline_metrics.cvar_95,
            "cvar_99": self.baseline_metrics.cvar_99,
            "turnover_rate": self.baseline_metrics.turnover_rate,
            "transaction_costs_total": self.baseline_metrics.transaction_costs_total,
            "transaction_costs_pct": self.baseline_metrics.transaction_costs_pct,
            "num_rebalances": self.baseline_metrics.num_rebalances,
            "total_days": self.baseline_metrics.total_days,
        }
        if self.baseline_metrics.start_date is not None:
            payload["start_date"] = self.baseline_metrics.start_date.isoformat()
        if self.baseline_metrics.end_date is not None:
            payload["end_date"] = self.baseline_metrics.end_date.isoformat()
        return payload

    def write_summary(self) -> None:
        """
        Persist stress sweep artefacts to disk.
        """
        directory = self.base_directory / "stress" / "monte_carlo"
        directory.mkdir(parents=True, exist_ok=True)
        paths_frame = self.to_frame()
        if not paths_frame.is_empty():
            def _serialize_overlay_events(value: object) -> str:
                candidate = value
                if hasattr(candidate, "to_list"):
                    candidate = candidate.to_list()
                if isinstance(candidate, list):
                    serializable: list[object] = []
                    for item in candidate:
                        if hasattr(item, "as_dict"):
                            serializable.append(item.as_dict())
                        elif hasattr(item, "to_dict"):
                            serializable.append(item.to_dict())
                        else:
                            serializable.append(item)
                    return json.dumps(serializable, sort_keys=True)
                return json.dumps(candidate, sort_keys=True)

            sanitized = paths_frame.with_columns(
                [
                    pl.col("regime_sequence")
                    .list.join("|")
                    .cast(pl.Utf8)
                    .alias("regime_sequence"),
                    pl.col("overlay_events")
                    .map_elements(_serialize_overlay_events, return_dtype=pl.Utf8)
                    .alias("overlay_events"),
                    pl.col("overlay_names")
                    .list.join("|")
                    .cast(pl.Utf8)
                    .alias("overlay_names"),
                    pl.col("overlay_categories")
                    .list.join("|")
                    .cast(pl.Utf8)
                    .alias("overlay_categories"),
                ],
            )
            sanitized.write_csv(directory / "paths.csv")
        summary = self.summary_frame()
        if not summary.is_empty():
            summary.write_csv(directory / "summary.csv")
        overlay_summary = self.overlay_summary_frame()
        if not overlay_summary.is_empty():
            overlay_summary.write_csv(directory / "overlay_summary.csv")
        overlay_category_summary = self.overlay_category_summary_frame()
        if not overlay_category_summary.is_empty():
            overlay_category_summary.write_csv(directory / "overlay_category_summary.csv")
        baseline_metrics_payload = self.baseline_metrics_dict()
        pl.DataFrame([baseline_metrics_payload]).write_csv(directory / "baseline_metrics.csv")
        baseline_payload = asdict(self.baseline_metrics)
        for field_name in ("start_date", "end_date"):
            value = baseline_payload.get(field_name)
            if isinstance(value, datetime):
                baseline_payload[field_name] = value.isoformat()
        config_payload = {
            "stress_config": self.stress_config.to_dict(),
            "target_strategy": self.target_strategy,
            "baseline_metrics": baseline_payload,
            "overlay_summary_path": "overlay_summary.csv"
            if not overlay_summary.is_empty()
            else None,
            "overlay_category_summary_path": "overlay_category_summary.csv"
            if not overlay_category_summary.is_empty()
            else None,
            "baseline_metrics_path": "baseline_metrics.csv",
        }
        (directory / "config.json").write_text(
            json.dumps(config_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def run_monte_carlo_stress_suite(
    dataset_path: Path,
    output_dir: Path,
    *,
    config: MonteCarloStressDefaults | None = None,
    split: TrainTestSplit | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> MonteCarloStressSuiteResult:
    """
    Execute Monte Carlo stress sweeps by randomising regime blocks and applying overlays.

    Parameters
    ----------
    dataset_path : Path
        Location of the sector dataset (directory or standalone Parquet file).
    output_dir : Path
        Directory for stress artefacts. Baseline outputs are written under a ``baseline`` subdirectory.
    config : MonteCarloStressDefaults | None
        Optional stress configuration overriding ``ThreeDRiskBacktestDefaults.monte_carlo_stress``.
    split : TrainTestSplit | None
        Optional explicit train/test split. Defaults to ``define_train_test_split()``.
    config_overrides : Mapping[str, Any] | None
        Backtest configuration overrides applied when generating the baseline suite.

    Returns
    -------
    MonteCarloStressSuiteResult
        Aggregated Monte Carlo stress sweep results.
    """
    defaults = BACKTEST_DEFAULTS
    stress_config = config or defaults.monte_carlo_stress
    target_strategy = stress_config.target_strategy

    dataset = _load_sector_dataset(dataset_path)
    resolved_split = split or define_train_test_split()
    _validate_split_against_dataset(dataset, resolved_split, defaults)

    baseline_dir = output_dir / "stress" / "monte_carlo" / "baseline"
    suite = _execute_backtest_suite(
        dataset=dataset,
        output_dir=baseline_dir,
        split=resolved_split,
        config_overrides=dict(config_overrides or {}),
        liquidity_config=defaults.build_liquidity_config(),
        turnover_overrides={
            "3d_factor_stable": defaults.stable_turnover_smoothing,
            "3d_factor_rolling": defaults.rolling_turnover_smoothing,
        },
    )

    baseline_metrics = suite.metrics.get(target_strategy)
    baseline_result = suite.full_results.get(target_strategy)
    if baseline_metrics is None or baseline_result is None:
        msg = f"Target strategy '{target_strategy}' not found in baseline suite"
        raise ValueError(msg)

    regimes = define_market_regimes()
    returns_frame = _build_regime_return_frame(baseline_result, regimes)
    regime_blocks = _prepare_regime_blocks(returns_frame)
    if not regime_blocks:
        msg = "No regime blocks available for Monte Carlo stress sweep"
        raise ValueError(msg)

    LOGGER.info(
        "Executing Monte Carlo stress sweep",
        strategy=target_strategy,
        num_paths=stress_config.num_paths,
        overlays=[overlay.name for overlay in stress_config.overlays],
        overlay_categories=sorted({overlay.category for overlay in stress_config.overlays}),
    )

    rng = np.random.default_rng(stress_config.random_seed)
    available_regimes = list(regime_blocks.keys())
    results: list[MonteCarloStressPathResult] = []

    for index in range(1, stress_config.num_paths + 1):
        if stress_config.sample_with_replacement:
            sampled_indices = rng.integers(0, len(available_regimes), size=len(available_regimes))
            regime_sequence = tuple(available_regimes[i] for i in sampled_indices)
        else:
            shuffled = available_regimes.copy()
            rng.shuffle(shuffled)
            regime_sequence = tuple(shuffled)

        block_lengths: list[int] = []
        path_returns: list[np.ndarray] = []
        for regime_name in regime_sequence:
            block = regime_blocks[regime_name]
            block_lengths.append(block.size)
            path_returns.append(block)

        if not path_returns:
            continue

        concatenated = np.concatenate(path_returns).astype(np.float64, copy=True)
        overlay_events = _apply_monte_carlo_overlays(
            rng=rng,
            returns=concatenated,
            regime_sequence=regime_sequence,
            block_lengths=tuple(block_lengths),
            overlays=stress_config.overlays,
        )
        for event in overlay_events:
            MONTE_CARLO_OVERLAY_COUNTER.labels(
                strategy=target_strategy,
                overlay=event.name,
                category=event.category,
            ).inc()

        path_result = _compute_monte_carlo_metrics(
            simulation_id=index,
            returns=concatenated,
            stress_config=stress_config,
            regime_sequence=regime_sequence,
            overlay_events=tuple(overlay_events),
        )
        results.append(path_result)
        MONTE_CARLO_SHARPE_HIST.labels(strategy=target_strategy).observe(path_result.sharpe_ratio)
        MONTE_CARLO_DRAWDOWN_HIST.labels(strategy=target_strategy).observe(path_result.max_drawdown)

    suite_result = MonteCarloStressSuiteResult(
        base_directory=output_dir,
        stress_config=stress_config,
        target_strategy=target_strategy,
        baseline_metrics=baseline_metrics,
        paths=tuple(results),
    )
    suite_result.write_summary()

    LOGGER.info(
        "Monte Carlo stress sweep completed",
        strategy=target_strategy,
        num_paths=len(results),
        positive_terminal=sum(1 for path in results if path.positive_terminal),
        overlay_events=sum(len(path.overlay_events) for path in results),
    )
    return suite_result


# ===== Parameter Heatmap Analysis =====


@dataclass(slots=True)
class ParameterHeatmapRunResult:
    """
    Result of executing a single parameter heatmap specification.

    Attributes
    ----------
    spec : ParameterHeatmapSpecDefaults
        Configuration defining the parameter sweep.
    results_frame : pl.DataFrame
        Tall table containing evaluated parameter combinations and metric outputs.
    pivot_frame : pl.DataFrame
        Heatmap-ready pivot table for the primary parameter pair.
    output_directory : Path
        Directory containing persisted artefacts for the run.
    metadata : dict[str, object]
        Serialised metadata summarising optimal configuration and diagnostics.
    """

    spec: ParameterHeatmapSpecDefaults
    results_frame: pl.DataFrame
    pivot_frame: pl.DataFrame
    output_directory: Path
    metadata: dict[str, object]

    def write_summary(self) -> None:
        """Persist heatmap artefacts to disk."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if not self.results_frame.is_empty():
            self.results_frame.write_csv(self.output_directory / "results.csv")
        if not self.pivot_frame.is_empty():
            self.pivot_frame.write_csv(self.output_directory / "heatmap.csv")
        metadata_path = self.output_directory / "metadata.json"
        metadata_path.write_text(json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(slots=True)
class ParameterHeatmapSuiteResult:
    """
    Aggregated results for all parameter heatmap specifications.
    """

    base_directory: Path
    output_dirname: str
    runs: tuple[ParameterHeatmapRunResult, ...]

    def summary_frame(self) -> pl.DataFrame:
        """Return summary dataframe capturing best metric per specification."""
        if not self.runs:
            return pl.DataFrame()
        rows: list[dict[str, object]] = []
        for run in self.runs:
            metadata = run.metadata
            best_config_json = json.dumps(metadata.get("best_config", {}), sort_keys=True)
            rows.append({
                "spec": run.spec.name,
                "slug": run.spec.slug,
                "strategy": metadata.get("target_strategy"),
                "metric": metadata.get("metric_name"),
                "metric_value": metadata.get("best_metric"),
                "row_parameter": run.spec.parameters[0],
                "column_parameter": run.spec.parameters[1],
                "evaluated_combinations": metadata.get("evaluated_combinations"),
                "best_config": best_config_json,
            })
        return pl.DataFrame(rows)

    def write_summary(self) -> None:
        """Persist suite-level summary table."""
        summary_frame = self.summary_frame()
        if summary_frame.is_empty():
            return
        root = self.base_directory / self.output_dirname
        root.mkdir(parents=True, exist_ok=True)
        summary_frame.write_csv(root / "summary.csv")


@dataclass(slots=True)
class ParameterSensitivityRunResult:
    """
    Result of executing a parameter sensitivity specification.
    """

    spec: ParameterSensitivitySpecDefaults
    results_frame: pl.DataFrame
    output_directory: Path
    metadata: Mapping[str, object]

    def write_summary(self) -> None:
        """Persist sensitivity artefacts to disk."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if not self.results_frame.is_empty():
            self.results_frame.write_csv(self.output_directory / "results.csv")
        metadata_path = self.output_directory / "metadata.json"
        metadata_path.write_text(json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(slots=True)
class ParameterSensitivitySuiteResult:
    """
    Aggregated results for parameter sensitivity sweeps.
    """

    base_directory: Path
    output_dirname: str
    runs: tuple[ParameterSensitivityRunResult, ...]

    def summary_frame(self) -> pl.DataFrame:
        """Return summary dataframe capturing best metric per specification."""
        if not self.runs:
            return pl.DataFrame()
        rows: list[dict[str, object]] = []
        for run in self.runs:
            metadata = run.metadata
            best_config_json = json.dumps(metadata.get("best_config", {}), sort_keys=True)
            rows.append({
                "spec": run.spec.name,
                "slug": run.spec.slug,
                "strategy": metadata.get("target_strategy"),
                "metric": metadata.get("metric_name"),
                "metric_value": metadata.get("best_metric"),
                 "metric_spread": metadata.get("metric_spread"),
                 "metric_spread_tolerance": metadata.get("metric_spread_tolerance"),
                 "metric_spread_ok": metadata.get("metric_spread_ok"),
                "evaluated_combinations": metadata.get("evaluated_combinations"),
                "best_config": best_config_json,
            })
        return pl.DataFrame(rows)

    def write_summary(self) -> None:
        """Persist suite-level summary table."""
        summary_frame = self.summary_frame()
        if summary_frame.is_empty():
            return
        root = self.base_directory / self.output_dirname
        root.mkdir(parents=True, exist_ok=True)
        summary_frame.write_csv(root / "summary.csv")


def _apply_heatmap_override(
    key: str,
    value: ParameterValue,
    *,
    config_overrides: dict[str, Any],
    turnover_overrides: dict[str, float],
    strategy_overrides: dict[str, dict[str, Any]],
    liquidity_overrides: dict[str, float],
    applied_values: dict[str, ParameterValue],
) -> None:
    """
    Partition a heatmap override key into the appropriate override mapping.
    """
    applied_values[key] = value
    if key.startswith("config."):
        config_overrides[key.split(".", 1)[1]] = value
    elif key.startswith("turnover_overrides."):
        turnover_overrides[key.split(".", 1)[1]] = float(value)
    elif key.startswith("strategy_params."):
        parts = key.split(".", 2)
        if len(parts) != 3:
            msg = f"strategy_params override '{key}' must include strategy and parameter"
            raise ValueError(msg)
        strategy_slug = parts[1]
        param_name = parts[2]
        overrides = strategy_overrides.setdefault(strategy_slug, {})
        overrides[param_name] = value
    elif key.startswith("liquidity_scaling."):
        liquidity_key = key.split(".", 1)[1]
        liquidity_overrides[liquidity_key] = float(value)
    else:
        config_overrides[key] = value


def _build_liquidity_config_from_overrides(
    *,
    base_defaults: ThreeDRiskBacktestDefaults,
    overrides: Mapping[str, float],
) -> LiquidityScalingConfig:
    """
    Construct a LiquidityScalingConfig applying override values.
    """
    if not overrides:
        return base_defaults.build_liquidity_config()
    base_kwargs = base_defaults.liquidity_scaling.to_kwargs()
    for key, value in overrides.items():
        base_kwargs[key] = float(value)
    return LiquidityScalingConfig(**base_kwargs)


def run_parameter_heatmap_suite(
    dataset_path: Path,
    output_dir: Path,
    *,
    specs: Sequence[ParameterHeatmapSpecDefaults] | None = None,
    split: TrainTestSplit | None = None,
    spec_slugs: Sequence[str] | None = None,
) -> ParameterHeatmapSuiteResult:
    """
    Execute configured parameter heatmap sweeps and persist the resulting artefacts.

    Parameters
    ----------
    dataset_path : Path
        Location of the sector dataset (directory or standalone Parquet file).
    output_dir : Path
        Directory for persisted artefacts.
    specs : Sequence[ParameterHeatmapSpecDefaults] | None, optional
        Explicit specification list overriding defaults.
    split : TrainTestSplit | None, optional
        Custom train/test split; defaults to canonical split when omitted.
    spec_slugs : Sequence[str] | None, optional
        When provided, restrict execution to heatmap specs whose slugs match the
        supplied values. Slugs are matched case-sensitively after trimming.
    """
    defaults = BACKTEST_DEFAULTS
    suite_defaults: ParameterHeatmapSuiteDefaults = defaults.parameter_heatmaps
    resolved_specs = tuple(specs or suite_defaults.specs)
    if not resolved_specs:
        raise ValueError("No parameter heatmap specifications provided")
    if spec_slugs:
        normalized_slugs: list[str] = []
        seen_slugs: set[str] = set()
        for slug in spec_slugs:
            trimmed = slug.strip()
            if not trimmed or trimmed in seen_slugs:
                continue
            normalized_slugs.append(trimmed)
            seen_slugs.add(trimmed)
        if normalized_slugs:
            slug_to_spec = {spec.slug: spec for spec in resolved_specs}
            filtered_specs: list[ParameterHeatmapSpecDefaults] = []
            missing: list[str] = []
            for slug in normalized_slugs:
                candidate = slug_to_spec.get(slug)
                if candidate is None:
                    missing.append(slug)
                else:
                    filtered_specs.append(candidate)
            if missing:
                missing_joined = ", ".join(sorted(missing))
                msg = f"No parameter heatmap specifications matched requested slugs: {missing_joined}"
                raise ValueError(msg)
            resolved_specs = tuple(filtered_specs)

    dataset = _load_sector_dataset(dataset_path)
    resolved_split = split or define_train_test_split()
    _validate_split_against_dataset(dataset, resolved_split, defaults)

    runs: list[ParameterHeatmapRunResult] = []
    for spec in resolved_specs:
        run_base_dir = output_dir / suite_defaults.output_dirname / spec.slug
        combination_rows: list[dict[str, object]] = []
        best_metric_value = float("-inf")
        best_configuration: dict[str, ParameterValue] | None = None
        pivot_frame = pl.DataFrame()

        grid_keys = tuple(spec.grid.keys())
        grid_values = [spec.grid[key] for key in grid_keys]
        combinations = list(itertools.product(*grid_values))
        if not combinations:
            LOGGER.warning("Heatmap spec has no parameter combinations", spec=spec.name)
            run = ParameterHeatmapRunResult(
                spec=spec,
                results_frame=pl.DataFrame(),
                pivot_frame=pl.DataFrame(),
                output_directory=run_base_dir,
                metadata={
                    "target_strategy": spec.target_strategy,
                    "metric_name": spec.metric,
                    "best_metric": None,
                    "best_config": {},
                    "comment": "No parameter combinations evaluated.",
                },
            )
            run.write_summary()
            runs.append(run)
            continue

        for index, combo in enumerate(combinations, start=1):
            config_overrides: dict[str, Any] = {}
            turnover_overrides: dict[str, float] = {}
            strategy_override_map: dict[str, dict[str, Any]] = {}
            liquidity_override_map: dict[str, float] = {}
            applied_values: dict[str, ParameterValue] = {}

            for base_key, base_value in spec.base_overrides.items():
                _apply_heatmap_override(
                    base_key,
                    base_value,
                    config_overrides=config_overrides,
                    turnover_overrides=turnover_overrides,
                    strategy_overrides=strategy_override_map,
                    liquidity_overrides=liquidity_override_map,
                    applied_values=applied_values,
                )

            for key, value in zip(grid_keys, combo):
                _apply_heatmap_override(
                    key,
                    value,
                    config_overrides=config_overrides,
                    turnover_overrides=turnover_overrides,
                    strategy_overrides=strategy_override_map,
                    liquidity_overrides=liquidity_override_map,
                    applied_values=applied_values,
                )

            liquidity_config = _build_liquidity_config_from_overrides(
                base_defaults=defaults,
                overrides=liquidity_override_map,
            )

            combination_dir = run_base_dir / "suites" / f"combination_{index:02d}"
            suite = _execute_backtest_suite(
                dataset=dataset,
                output_dir=combination_dir,
                split=resolved_split,
                config_overrides=config_overrides or None,
                liquidity_config=liquidity_config,
                turnover_overrides=turnover_overrides or None,
                strategy_overrides=strategy_override_map or None,
            )
            metrics = suite.metrics.get(spec.target_strategy)
            if metrics is None:
                LOGGER.warning(
                    "Target strategy metrics missing in heatmap sweep",
                    strategy=spec.target_strategy,
                    spec=spec.name,
                )
                continue
            if not hasattr(metrics, spec.metric):
                msg = (
                    f"PerformanceMetrics lacks attribute '{spec.metric}' "
                    f"for heatmap specification '{spec.name}'"
                )
                raise AttributeError(msg)
            metric_value = float(getattr(metrics, spec.metric))
            HEATMAP_METRIC_HIST.labels(strategy=spec.target_strategy, spec=spec.slug).observe(metric_value)
            row_payload: dict[str, object] = {
                "combination_index": index,
                "strategy": spec.target_strategy,
                "metric": metric_value,
            }
            for key in grid_keys:
                row_payload[key] = applied_values.get(key)
            row_payload.update({
                key: applied_values.get(key)
                for key in spec.base_overrides.keys()
                if key not in row_payload
            })
            combination_rows.append(row_payload)
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_configuration = dict(sorted(applied_values.items()))

        if combination_rows:
            results_frame = pl.DataFrame(combination_rows)
            row_param, column_param = spec.parameters
            pivot_columns = [row_param, column_param, "metric"]
            for column in pivot_columns:
                if column not in results_frame.columns:
                    results_frame = results_frame.with_columns(pl.lit(None).alias(column))
            pivot_ready = results_frame.with_columns(
                pl.col(row_param).cast(pl.Utf8),
                pl.col(column_param).cast(pl.Utf8),
            )
            try:
                pivot_frame = pivot_ready.pivot(
                    on=column_param,
                    index=row_param,
                    values="metric",
                )
            except Exception:
                LOGGER.exception(
                    "Failed to pivot heatmap results",
                    spec=spec.name,
                )
                pivot_frame = pl.DataFrame()
        else:
            results_frame = pl.DataFrame()

        metadata: dict[str, object] = {
            "target_strategy": spec.target_strategy,
            "metric_name": spec.metric,
            "best_metric": best_metric_value if combination_rows else None,
            "evaluated_combinations": len(combination_rows),
        }
        metadata["best_config"] = cast(object, best_configuration or {})
        run_result = ParameterHeatmapRunResult(
            spec=spec,
            results_frame=results_frame,
            pivot_frame=pivot_frame,
            output_directory=run_base_dir,
            metadata=metadata,
        )
        run_result.write_summary()
        runs.append(run_result)

    suite_result = ParameterHeatmapSuiteResult(
        base_directory=output_dir,
        output_dirname=suite_defaults.output_dirname,
        runs=tuple(runs),
    )
    suite_result.write_summary()
    return suite_result


# ===== Extended Diagnostics =====


def run_parameter_sensitivity_suite(
    dataset_path: Path,
    output_dir: Path,
    *,
    specs: Sequence[ParameterSensitivitySpecDefaults] | None = None,
    split: TrainTestSplit | None = None,
    spec_slugs: Sequence[str] | None = None,
) -> ParameterSensitivitySuiteResult:
    """
    Execute configured parameter sensitivity sweeps and persist artefacts.

    Parameters
    ----------
    dataset_path : Path
        Location of the sector dataset (directory or standalone Parquet file).
    output_dir : Path
        Directory for persisted artefacts.
    specs : Sequence[ParameterSensitivitySpecDefaults] | None, optional
        Explicit specification list overriding defaults.
    split : TrainTestSplit | None, optional
        Custom train/test split; defaults to canonical split when omitted.
    spec_slugs : Sequence[str] | None, optional
        When provided, restrict execution to specifications whose slugs match
        the supplied values. Slugs are matched case-sensitively after trimming.
    """
    defaults = BACKTEST_DEFAULTS
    suite_defaults: ParameterSensitivitySuiteDefaults = defaults.parameter_sensitivity
    resolved_specs = tuple(specs or suite_defaults.specs)
    if not resolved_specs:
        raise ValueError("No parameter sensitivity specifications provided")
    if spec_slugs:
        normalized_slugs: list[str] = []
        seen_slugs: set[str] = set()
        for slug in spec_slugs:
            trimmed = slug.strip()
            if not trimmed or trimmed in seen_slugs:
                continue
            normalized_slugs.append(trimmed)
            seen_slugs.add(trimmed)
        if normalized_slugs:
            slug_to_spec = {spec.slug: spec for spec in resolved_specs}
            filtered_specs: list[ParameterSensitivitySpecDefaults] = []
            missing: list[str] = []
            for slug in normalized_slugs:
                candidate = slug_to_spec.get(slug)
                if candidate is None:
                    missing.append(slug)
                else:
                    filtered_specs.append(candidate)
            if missing:
                missing_joined = ", ".join(sorted(missing))
                msg = f"No parameter sensitivity specifications matched requested slugs: {missing_joined}"
                raise ValueError(msg)
            resolved_specs = tuple(filtered_specs)

    dataset = _load_sector_dataset(dataset_path)
    resolved_split = split or define_train_test_split()
    _validate_split_against_dataset(dataset, resolved_split, defaults)

    runs: list[ParameterSensitivityRunResult] = []
    for spec in resolved_specs:
        run_base_dir = output_dir / suite_defaults.output_dirname / spec.slug
        combination_rows: list[dict[str, object]] = []
        best_metric_value = float("-inf")
        best_configuration: dict[str, ParameterValue] | None = None
        metric_values: list[float] = []

        grid_keys = tuple(spec.parameter_grid.keys())
        grid_values = [spec.parameter_grid[key] for key in grid_keys]
        combinations = list(itertools.product(*grid_values))
        if not combinations:
            LOGGER.warning("Sensitivity spec has no parameter combinations", spec=spec.name)
            run = ParameterSensitivityRunResult(
                spec=spec,
                results_frame=pl.DataFrame(),
                output_directory=run_base_dir,
                metadata={
                    "target_strategy": spec.target_strategy,
                    "metric_name": spec.metric,
                    "best_metric": None,
                    "best_config": {},
                    "evaluated_combinations": 0,
                    "comment": "No parameter combinations evaluated.",
                },
            )
            run.write_summary()
            runs.append(run)
            continue

        for index, combo in enumerate(combinations, start=1):
            config_overrides: dict[str, Any] = {}
            turnover_overrides: dict[str, float] = {}
            strategy_override_map: dict[str, dict[str, Any]] = {}
            liquidity_override_map: dict[str, float] = {}
            applied_values: dict[str, ParameterValue] = {}

            for base_key, base_value in spec.base_overrides.items():
                _apply_heatmap_override(
                    base_key,
                    base_value,
                    config_overrides=config_overrides,
                    turnover_overrides=turnover_overrides,
                    strategy_overrides=strategy_override_map,
                    liquidity_overrides=liquidity_override_map,
                    applied_values=applied_values,
                )

            for key, value in zip(grid_keys, combo):
                _apply_heatmap_override(
                    key,
                    value,
                    config_overrides=config_overrides,
                    turnover_overrides=turnover_overrides,
                    strategy_overrides=strategy_override_map,
                    liquidity_overrides=liquidity_override_map,
                    applied_values=applied_values,
                )

            liquidity_config = _build_liquidity_config_from_overrides(
                base_defaults=defaults,
                overrides=liquidity_override_map,
            )

            combination_dir = run_base_dir / "suites" / f"combination_{index:02d}"
            suite = _execute_backtest_suite(
                dataset=dataset,
                output_dir=combination_dir,
                split=resolved_split,
                config_overrides=config_overrides or None,
                liquidity_config=liquidity_config,
                turnover_overrides=turnover_overrides or None,
                strategy_overrides=strategy_override_map or None,
            )
            metrics = suite.metrics.get(spec.target_strategy)
            if metrics is None:
                LOGGER.warning(
                    "Target strategy metrics missing in sensitivity sweep",
                    strategy=spec.target_strategy,
                    spec=spec.name,
                )
                continue
            if not hasattr(metrics, spec.metric):
                msg = (
                    f"PerformanceMetrics lacks attribute '{spec.metric}' "
                    f"for sensitivity specification '{spec.name}'"
                )
                raise AttributeError(msg)
            metric_value = float(getattr(metrics, spec.metric))
            SENSITIVITY_METRIC_HIST.labels(
                strategy=spec.target_strategy,
                spec=spec.slug,
                metric=spec.metric,
            ).observe(metric_value)
            row_payload: dict[str, object] = {
                "combination_index": index,
                "strategy": spec.target_strategy,
                "metric_name": spec.metric,
                "metric_value": metric_value,
                "annualized_return": metrics.annualized_return,
                "annualized_volatility": metrics.annualized_volatility,
                "calmar_ratio": metrics.calmar_ratio,
                "max_drawdown": metrics.maximum_drawdown,
                "sharpe_ratio": metrics.sharpe_ratio,
            }
            metric_values.append(metric_value)
            for key in grid_keys:
                row_payload[key] = applied_values.get(key)
            for key in spec.base_overrides:
                if key not in row_payload:
                    row_payload[key] = applied_values.get(key)

            combination_rows.append(row_payload)
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_configuration = dict(sorted(applied_values.items()))

        if combination_rows:
            results_frame = pl.DataFrame(combination_rows)
        else:
            results_frame = pl.DataFrame()

        metadata: dict[str, object] = {
            "target_strategy": spec.target_strategy,
            "metric_name": spec.metric,
            "best_metric": best_metric_value if combination_rows else None,
            "evaluated_combinations": len(combination_rows),
            "best_config": cast(object, best_configuration or {}),
        }
        metric_spread: float | None = None
        if metric_values:
            metric_spread = max(metric_values) - min(metric_values)
        tolerance: float | None = None
        spread_ok = True
        if metric_spread is not None and spec.metric == "sharpe_ratio":
            tolerance = suite_defaults.sharpe_delta_tolerance
            spread_ok = metric_spread <= tolerance
            if not spread_ok:
                LOGGER.warning(
                    "phase4_sensitivity_sharpe_delta_violation",
                    spec=spec.slug,
                    spread=metric_spread,
                    tolerance=tolerance,
                )
                SENSITIVITY_DELTA_COUNTER.labels(spec=spec.slug).inc()
        metadata["metric_spread"] = metric_spread
        metadata["metric_spread_tolerance"] = tolerance
        metadata["metric_spread_ok"] = spread_ok
        run_result = ParameterSensitivityRunResult(
            spec=spec,
            results_frame=results_frame,
            output_directory=run_base_dir,
            metadata=metadata,
        )
        run_result.write_summary()
        runs.append(run_result)

    suite_result = ParameterSensitivitySuiteResult(
        base_directory=output_dir,
        output_dirname=suite_defaults.output_dirname,
        runs=tuple(runs),
    )
    suite_result.write_summary()
    return suite_result


@dataclass(slots=True)
class DiagnosticsResult:
    """
    Aggregated diagnostic outputs derived from a backtest suite.

    Attributes
    ----------
    tail_metrics : pl.DataFrame
        Distribution of tail risk metrics by strategy and quantile.
    turnover_distribution : pl.DataFrame
        Histogram of turnover observations bucketed by configured bins.
    benchmark_deltas : pl.DataFrame
        Performance deltas versus alternative benchmarks.
    output_directory : Path
        Directory housing persisted artefacts.
    config : DiagnosticsDefaults
        Configuration used for the diagnostic calculations.
    """

    tail_metrics: pl.DataFrame
    turnover_distribution: pl.DataFrame
    benchmark_deltas: pl.DataFrame
    output_directory: Path
    config: DiagnosticsDefaults

    def write_summary(self) -> None:
        """Persist diagnostics artefacts."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if not self.tail_metrics.is_empty():
            self.tail_metrics.write_csv(self.output_directory / "tail_metrics.csv")
        if not self.turnover_distribution.is_empty():
            self.turnover_distribution.write_csv(self.output_directory / "turnover_distribution.csv")
        if not self.benchmark_deltas.is_empty():
            self.benchmark_deltas.write_csv(self.output_directory / "benchmark_deltas.csv")
        config_payload = {
            "tail_quantiles": list(self.config.tail_quantiles),
            "turnover_bins": list(self.config.turnover_bins),
            "alternative_benchmarks": list(self.config.alternative_benchmarks),
            "turnover_window_days": self.config.turnover_window_days,
            "benchmark_delta_metrics": list(self.config.benchmark_delta_metrics),
        }
        (self.output_directory / "config.json").write_text(
            json.dumps(config_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _compute_turnover_series(result: BacktestResult) -> np.ndarray:
    """Derive turnover estimates from backtest position weights."""
    if result.positions.is_empty():
        return np.array([], dtype=np.float64)
    frame = result.positions.sort("timestamp")
    weight_columns = [column for column in frame.columns if column != "timestamp"]
    if not weight_columns:
        return np.array([], dtype=np.float64)
    try:
        turnover_frame = frame.select(
            pl.sum_horizontal([pl.col(col).diff().abs() for col in weight_columns]).alias("gross_turnover")
        )
    except Exception:
        LOGGER.exception("Failed to derive turnover series", strategy=result.strategy_name)
        return np.array([], dtype=np.float64)
    turnover_values = turnover_frame.get_column("gross_turnover").to_numpy()
    if turnover_values.size <= 1:
        return np.array([], dtype=np.float64)
    return np.asarray(turnover_values[1:] / 2.0, dtype=np.float64)


def run_extended_diagnostics(
    suite: BacktestSuite,
    output_dir: Path,
    *,
    config: DiagnosticsDefaults | None = None,
) -> DiagnosticsResult:
    """
    Execute extended diagnostics covering tail risk, turnover distribution, and benchmark deltas.
    """
    defaults = BACKTEST_DEFAULTS
    diagnostics_config = config or defaults.diagnostics

    tail_rows: list[dict[str, object]] = []
    for strategy_name, backtest_result in suite.full_results.items():
        returns = np.asarray(backtest_result.returns, dtype=np.float64)
        if returns.size == 0:
            continue
        for quantile in diagnostics_config.tail_quantiles:
            value_at_risk = float(np.quantile(returns, quantile))
            mask = returns <= value_at_risk + 1e-12
            if np.any(mask):
                conditional_var = float(np.mean(returns[mask]))
            else:
                conditional_var = value_at_risk
            quantile_label = f"{int(quantile * 100)}%"
            DIAGNOSTIC_TAIL_HIST.labels(strategy=strategy_name, quantile=quantile_label).observe(value_at_risk)
            tail_rows.append({
                "strategy": strategy_name,
                "quantile": quantile,
                "value_at_risk": value_at_risk,
                "conditional_value_at_risk": conditional_var,
                "num_observations": int(returns.size),
            })
    tail_frame = pl.DataFrame(tail_rows) if tail_rows else pl.DataFrame()

    turnover_rows: list[dict[str, object]] = []
    bin_edges = [0.0, *diagnostics_config.turnover_bins, float("inf")]
    bin_labels: list[str] = []
    for index in range(len(bin_edges) - 1):
        lower = bin_edges[index]
        upper = bin_edges[index + 1]
        if math.isinf(upper):
            label = f">={lower:.2f}"
        else:
            label = f"{lower:.2f}-{upper:.2f}"
        bin_labels.append(label)

    for strategy_name, backtest_result in suite.full_results.items():
        turnover_series = _compute_turnover_series(backtest_result)
        if turnover_series.size == 0:
            continue
        counts, _ = np.histogram(turnover_series, bins=bin_edges)
        total = int(np.sum(counts))
        mean_turnover = float(np.mean(turnover_series))
        p95_turnover = float(np.quantile(turnover_series, 0.95))
        window = min(diagnostics_config.turnover_window_days, turnover_series.size)
        if window > 0:
            rolling_mean = float(np.mean(turnover_series[-window:]))
            rolling_max = float(np.max(turnover_series[-window:]))
        else:
            rolling_mean = mean_turnover
            rolling_max = mean_turnover
        for label, count in zip(bin_labels, counts):
            turnover_rows.append({
                "strategy": strategy_name,
                "bucket": label,
                "count": int(count),
                "proportion": float(count / total) if total > 0 else 0.0,
                "mean_turnover": mean_turnover,
                "p95_turnover": p95_turnover,
                "rolling_window_days": diagnostics_config.turnover_window_days,
                "rolling_mean_turnover": rolling_mean,
                "rolling_max_turnover": rolling_max,
            })
    turnover_frame = pl.DataFrame(turnover_rows) if turnover_rows else pl.DataFrame()

    benchmark_rows: list[dict[str, object]] = []
    for benchmark_name in diagnostics_config.alternative_benchmarks:
        baseline = suite.metrics.get(benchmark_name)
        if baseline is None:
            continue
        for strategy_name, metrics in suite.metrics.items():
            row: dict[str, object] = {
                "benchmark": benchmark_name,
                "strategy": strategy_name,
            }
            for metric_name in diagnostics_config.benchmark_delta_metrics:
                strategy_value = getattr(metrics, metric_name, None)
                baseline_value = getattr(baseline, metric_name, None)
                if strategy_value is None or baseline_value is None:
                    continue
                alias = f"{metric_name}_delta"
                row[alias] = strategy_value - baseline_value
            if len(row) > 2:
                benchmark_rows.append(row)
    benchmark_frame = pl.DataFrame(benchmark_rows) if benchmark_rows else pl.DataFrame()

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_result = DiagnosticsResult(
        tail_metrics=tail_frame,
        turnover_distribution=turnover_frame,
        benchmark_deltas=benchmark_frame,
        output_directory=diagnostics_dir,
        config=diagnostics_config,
    )
    diagnostics_result.write_summary()
    return diagnostics_result


# ===== Proxy Dataset Validation =====


@dataclass(slots=True)
class ProxyDatasetRunResult:
    """
    Result of validating a single proxy dataset specification.
    """

    spec: ProxyDatasetSpecDefaults
    status: str
    message: str | None
    output_directory: Path
    summary_frame: pl.DataFrame

    def write_summary(self) -> None:
        """Persist proxy dataset summary artefacts."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if not self.summary_frame.is_empty():
            self.summary_frame.write_csv(self.output_directory / "benchmark_summary.csv")
        metadata = {
            "name": self.spec.name,
            "status": self.status,
            "message": self.message,
            "relative_path": self.spec.relative_path,
            "allow_missing": self.spec.allow_missing,
            "min_train_years": self.spec.min_train_years,
            "min_test_years": self.spec.min_test_years,
            "tags": list(self.spec.tags),
        }
        (self.output_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )


@dataclass(slots=True)
class ProxyDatasetSuiteResult:
    """
    Aggregated results for all proxy dataset validations.
    """

    base_directory: Path
    runs: tuple[ProxyDatasetRunResult, ...]

    def summary_frame(self) -> pl.DataFrame:
        """Return consolidated status summary."""
        if not self.runs:
            return pl.DataFrame()
        rows = [{
            "name": run.spec.name,
            "slug": run.spec.slug,
            "status": run.status,
            "message": run.message,
            "allow_missing": run.spec.allow_missing,
            "tags": "|".join(run.spec.tags),
        } for run in self.runs]
        return pl.DataFrame(rows)

    def write_summary(self) -> None:
        """Persist suite-level summary."""
        summary = self.summary_frame()
        if summary.is_empty():
            return
        directory = self.base_directory / "proxy_datasets"
        directory.mkdir(parents=True, exist_ok=True)
        summary.write_csv(directory / "summary.csv")


def run_proxy_dataset_validation(
    dataset_path: Path,
    output_dir: Path,
    *,
    suite_defaults: ProxyDatasetSuiteDefaults | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> ProxyDatasetSuiteResult:
    """
    Execute proxy dataset validations using configured specifications.
    """
    defaults = BACKTEST_DEFAULTS
    resolved_defaults = suite_defaults or defaults.proxy_datasets
    runs: list[ProxyDatasetRunResult] = []
    base_directory = output_dir / "proxy_datasets"

    for spec in resolved_defaults.specs:
        proxy_path = Path(spec.relative_path)
        if not proxy_path.is_absolute():
            proxy_path = Path.cwd() / proxy_path
        run_dir = base_directory / spec.slug
        if not proxy_path.exists():
            status = "missing_allowed" if spec.allow_missing else "missing"
            error_message = f"Dataset not found at {proxy_path}"
            PROXY_STATUS_COUNTER.labels(status=status, dataset=spec.slug).inc()
            run = ProxyDatasetRunResult(
                spec=spec,
                status=status,
                message=error_message if not spec.allow_missing else f"{error_message} (allowed)",
                output_directory=run_dir,
                summary_frame=pl.DataFrame(),
            )
            run.write_summary()
            runs.append(run)
            continue

        message: str | None
        try:
            suite = run_full_backtest_suite(
                dataset_path=proxy_path,
                output_dir=run_dir / "suite",
                split=None,
                config_overrides=dict(config_overrides or {}),
            )
            summary = suite.benchmark_summary()
            status = "success"
            message = None
        except Exception as exc:
            LOGGER.exception(
                "Proxy dataset validation failed",
                dataset=str(proxy_path),
                spec=spec.name,
            )
            status = "error"
            message = str(exc)
            summary = pl.DataFrame()

        PROXY_STATUS_COUNTER.labels(status=status, dataset=spec.slug).inc()
        run = ProxyDatasetRunResult(
            spec=spec,
            status=status,
            message=message,
            output_directory=run_dir,
            summary_frame=summary,
        )
        run.write_summary()
        runs.append(run)

    suite_result = ProxyDatasetSuiteResult(
        base_directory=output_dir,
        runs=tuple(runs),
    )
    suite_result.write_summary()
    return suite_result


# ===== Vintage Simulation Suite =====


@dataclass(slots=True)
class VintageSimulationRunResult:
    """
    Result of executing a vintage walk-forward simulation.
    """

    window: VintageWindowDefaults
    summary: pl.DataFrame
    output_directory: Path
    fold_count: int
    status: str
    message: str | None

    def write_summary(self) -> None:
        """Persist vintage simulation summary."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if not self.summary.is_empty():
            self.summary.write_csv(self.output_directory / "summary.csv")
        payload = self.window.to_dict()
        payload["fold_count"] = self.fold_count
        payload["status"] = self.status
        if self.message is not None:
            payload["message"] = self.message
        (self.output_directory / "metadata.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


@dataclass(slots=True)
class VintageSimulationSuiteResult:
    """
    Aggregated results for vintage simulation windows.
    """

    base_directory: Path
    runs: tuple[VintageSimulationRunResult, ...]

    def summary_frame(self) -> pl.DataFrame:
        """Return summary of fold counts per vintage specification."""
        if not self.runs:
            return pl.DataFrame()
        rows = [{
            "label": run.window.label,
            "slug": run.window.slug,
            "fold_count": run.fold_count,
            "status": run.status,
            "message": run.message,
            "min_folds": run.window.min_folds,
        } for run in self.runs]
        return pl.DataFrame(rows)

    def write_summary(self) -> None:
        """Persist suite summary."""
        summary = self.summary_frame()
        if summary.is_empty():
            return
        directory = self.base_directory / "vintage"
        directory.mkdir(parents=True, exist_ok=True)
        summary.write_csv(directory / "summary.csv")


def run_vintage_simulation_suite(
    dataset_path: Path,
    output_dir: Path,
    *,
    windows: Sequence[VintageWindowDefaults] | None = None,
) -> VintageSimulationSuiteResult:
    """
    Execute sequential vintage simulations across configured windows.
    """
    defaults = BACKTEST_DEFAULTS
    resolved_windows = tuple(windows or defaults.proxy_datasets.vintage_windows)
    if not resolved_windows:
        raise ValueError("No vintage window specifications provided")

    dataset = _load_sector_dataset(dataset_path)
    bounds = dataset.sector_returns.select(
        pl.col("timestamp").min().alias("min_ts"),
        pl.col("timestamp").max().alias("max_ts"),
    ).row(0)
    data_start = bounds[0]
    data_end = bounds[1]
    if isinstance(data_start, datetime) and data_start.tzinfo is None:
        data_start = data_start.replace(tzinfo=UTC)
    if isinstance(data_end, datetime) and data_end.tzinfo is None:
        data_end = data_end.replace(tzinfo=UTC)

    runs: list[VintageSimulationRunResult] = []
    vintage_root = output_dir / "vintage"

    for window in resolved_windows:
        walk_forward_config = WalkForwardConfig(
            start_date=data_start,
            end_date=data_end,
            train_years=window.train_years,
            test_years=window.test_years,
            step_years=window.step_years,
        )
        window_dir = vintage_root / window.slug
        status = "success"
        message: str | None = None
        summary = pl.DataFrame()
        fold_count = 0
        try:
            result = run_walk_forward_backtest_suite(
                dataset_path=dataset_path,
                output_dir=window_dir,
                walk_forward_config=walk_forward_config,
            )
            summary = result.summarize_metrics()
            fold_count = len(result.splits)
        except Exception as exc:
            LOGGER.exception(
                "Vintage simulation execution failed",
                window=window.label,
            )
            status = "error"
            message = str(exc)
        else:
            if fold_count < window.min_folds:
                status = "insufficient_folds"
                message = (
                    f"Produced {fold_count} folds but require >= {window.min_folds}"
                )
                LOGGER.warning(
                    "Vintage simulation produced fewer folds than expected",
                    label=window.label,
                    folds=fold_count,
                    minimum=window.min_folds,
                )
        run_result = VintageSimulationRunResult(
            window=window,
            summary=summary,
            output_directory=window_dir,
            fold_count=fold_count,
            status=status,
            message=message,
        )
        run_result.write_summary()
        VINTAGE_STATUS_COUNTER.labels(status=status, window=window.slug).inc()
        runs.append(run_result)

    suite_result = VintageSimulationSuiteResult(
        base_directory=output_dir,
        runs=tuple(runs),
    )
    suite_result.write_summary()
    return suite_result


# ===== Monitoring Snapshot =====


@dataclass(slots=True)
class MonitoringSnapshotResult:
    """
    Metadata snapshot exported for monitoring dashboards.
    """

    path: Path
    payload: dict[str, object]


def export_phase3_monitoring_snapshot(
    output_dir: Path,
    *,
    walk_forward: MultiHorizonWalkForwardResult | None = None,
    monte_carlo: MonteCarloStressSuiteResult | None = None,
    heatmaps: ParameterHeatmapSuiteResult | None = None,
    diagnostics: DiagnosticsResult | None = None,
    proxy_datasets: ProxyDatasetSuiteResult | None = None,
    vintage: VintageSimulationSuiteResult | None = None,
    defaults: MonitoringExportDefaults | None = None,
) -> MonitoringSnapshotResult:
    """
    Generate consolidated monitoring snapshot referencing generated artefacts.
    """
    resolved_defaults = defaults or BACKTEST_DEFAULTS.monitoring
    snapshot_dir = output_dir
    sections: dict[str, dict[str, object]] = {}
    diagnostics_summary: dict[str, object] | None = None
    proxy_metadata: dict[str, _ProxyMetadataEntry] | None = None
    vintage_windows_metadata: list[_VintageWindowMetadataEntry] | None = None
    overlay_category_stats: list[dict[str, object]] | None = None
    baseline_metrics_payload: dict[str, object] | None = None
    heatmap_metadata: list[dict[str, object]] | None = None
    sensitivity_metadata: list[dict[str, object]] | None = None
    data_quality_summary: dict[str, object] | None = None
    outlier_summary: dict[str, object] | None = None

    def _rel_path(path: Path) -> str | None:
        return str(path) if path.exists() else None

    def _summarize_diagnostics(result: DiagnosticsResult) -> dict[str, object]:
        def _coerce_float(value: object, default: float = 0.0) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return default
            return default

        def _coerce_int(value: object, default: int = 0) -> int:
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return default
            return default

        tail_summary: dict[str, dict[str, float]] = {}
        if not result.tail_metrics.is_empty():
            for row in result.tail_metrics.to_dicts():
                strategy = str(row.get("strategy", ""))
                if not strategy:
                    continue
                quantile_raw = row.get("quantile")
                if quantile_raw is None:
                    continue
                try:
                    quantile = float(quantile_raw)
                except (TypeError, ValueError):
                    continue
                quantile_label = f"p{round(quantile * 100):02d}"
                bucket = tail_summary.setdefault(strategy, {})
                value_at_risk = row.get("value_at_risk")
                if value_at_risk is not None:
                    bucket[f"var_{quantile_label}"] = _coerce_float(value_at_risk)
                conditional_value = row.get("conditional_value_at_risk")
                if conditional_value is not None:
                    bucket[f"cvar_{quantile_label}"] = _coerce_float(conditional_value)

        turnover_summary: dict[str, dict[str, float | int]] = {}
        if not result.turnover_distribution.is_empty():
            for row in result.turnover_distribution.to_dicts():
                strategy = str(row.get("strategy", ""))
                if not strategy or strategy in turnover_summary:
                    continue
                turnover_summary[strategy] = {
                    "mean_turnover": _coerce_float(row.get("mean_turnover", 0.0)),
                    "p95_turnover": _coerce_float(row.get("p95_turnover", 0.0)),
                    "rolling_window_days": _coerce_int(row.get("rolling_window_days", 0)),
                    "rolling_mean_turnover": _coerce_float(row.get("rolling_mean_turnover", 0.0)),
                    "rolling_max_turnover": _coerce_float(row.get("rolling_max_turnover", 0.0)),
                }

        benchmark_summary: dict[str, dict[str, dict[str, float]]] = {}
        if not result.benchmark_deltas.is_empty():
            for row in result.benchmark_deltas.to_dicts():
                benchmark = str(row.get("benchmark", ""))
                strategy = str(row.get("strategy", ""))
                if not benchmark or not strategy:
                    continue
                metrics: dict[str, float] = {}
                for key, value in row.items():
                    if key in {"benchmark", "strategy"} or value is None:
                        continue
                    metrics[key] = _coerce_float(value)
                if metrics:
                    strategy_metrics = benchmark_summary.setdefault(benchmark, {})
                    strategy_metrics[strategy] = metrics

        return {
            "tail": tail_summary,
            "turnover": turnover_summary,
            "benchmark_deltas": benchmark_summary,
            "config": {
                "tail_quantiles": list(result.config.tail_quantiles),
                "turnover_window_days": result.config.turnover_window_days,
                "benchmark_delta_metrics": list(result.config.benchmark_delta_metrics),
            },
        }

    if "walk_forward" in resolved_defaults.include_sections and walk_forward is not None:
        walk_forward_root = walk_forward.base_directory / "walk_forward"
        sections["walk_forward"] = {
            "summary_path": _rel_path(walk_forward_root / "permutation_summary.csv"),
            "nested_summary_path": _rel_path(walk_forward_root / "nested_summary.csv"),
            "nested_rollup_path": _rel_path(walk_forward_root / "nested_rollup.csv"),
        }

    if "monte_carlo" in resolved_defaults.include_sections and monte_carlo is not None:
        stress_root = monte_carlo.base_directory / "stress" / "monte_carlo"
        overlay_category_stats = monte_carlo.overlay_category_summary()
        baseline_metrics_payload = monte_carlo.baseline_metrics_dict()
        sections["monte_carlo"] = {
            "summary_path": _rel_path(stress_root / "summary.csv"),
            "paths_path": _rel_path(stress_root / "paths.csv"),
            "overlay_summary_path": _rel_path(stress_root / "overlay_summary.csv"),
            "overlay_category_summary_path": _rel_path(stress_root / "overlay_category_summary.csv"),
            "baseline_metrics_path": _rel_path(stress_root / "baseline_metrics.csv"),
            "config_path": _rel_path(stress_root / "config.json"),
        }

    if "parameter_heatmaps" in resolved_defaults.include_sections and heatmaps is not None:
        heatmap_root = heatmaps.base_directory / heatmaps.output_dirname
        sections["parameter_heatmaps"] = {
            "summary_path": _rel_path(heatmap_root / "summary.csv"),
            "config_specs": [run.spec.slug for run in heatmaps.runs],
        }
        heatmap_metadata = [
            {
                "slug": run.spec.slug,
                "metric": run.metadata.get("metric_name"),
                "best_metric": run.metadata.get("best_metric"),
                "evaluated_combinations": run.metadata.get("evaluated_combinations"),
                "best_config": run.metadata.get("best_config"),
            }
            for run in heatmaps.runs
        ]

    if "phase4_sensitivity" in resolved_defaults.include_sections:
        sensitivity_defaults = BACKTEST_DEFAULTS.parameter_sensitivity
        sensitivity_root = snapshot_dir / sensitivity_defaults.output_dirname
        summary_path = _rel_path(sensitivity_root / "summary.csv")
        pdf_candidates = (
            sensitivity_root / "sensitivity_analysis.pdf",
            snapshot_dir / "sensitivity_analysis.pdf",
            snapshot_dir.parent / "sensitivity_analysis.pdf",
        )
        pdf_path: str | None = None
        for candidate in pdf_candidates:
            candidate_path = _rel_path(candidate)
            if candidate_path is not None:
                pdf_path = candidate_path
                break
        sections["phase4_sensitivity"] = {
            "summary_path": summary_path,
            "report_path": pdf_path,
        }
        if summary_path is not None:
            summary_file = Path(summary_path)
            try:
                summary_frame = pl.read_csv(summary_file)
            except Exception:
                LOGGER.warning("Failed to load sensitivity summary for monitoring snapshot", path=summary_path, exc_info=True)
            else:
                sensitivity_metadata = [
                    {
                        "spec": str(row.get("spec", "")),
                        "slug": str(row.get("slug", "")),
                        "metric": row.get("metric"),
                        "metric_value": row.get("metric_value"),
                        "evaluated_combinations": row.get("evaluated_combinations"),
                        "best_config": row.get("best_config"),
                    }
                    for row in summary_frame.to_dicts()
                ]

    if "phase4_data_quality" in resolved_defaults.include_sections:
        data_quality_root = snapshot_dir / "data_quality"
        audit_path = _rel_path(data_quality_root / "missing_data_audit.json")
        sections["phase4_data_quality"] = {
            "audit_path": audit_path,
        }
        if audit_path is not None:
            audit_file = Path(audit_path)
            try:
                data_quality_summary = json.loads(audit_file.read_text(encoding="utf-8"))
            except Exception:
                LOGGER.warning("Failed to parse data quality audit for monitoring snapshot", path=audit_path, exc_info=True)

    if "phase4_outliers" in resolved_defaults.include_sections:
        outlier_root = snapshot_dir / "outliers"
        report_path = _rel_path(outlier_root / "factor_outlier_report.json")
        sections["phase4_outliers"] = {
            "report_path": report_path,
        }
        if report_path is not None:
            report_file = Path(report_path)
            try:
                outlier_summary = json.loads(report_file.read_text(encoding="utf-8"))
            except Exception:
                LOGGER.warning("Failed to parse outlier report for monitoring snapshot", path=report_path, exc_info=True)

    if "benchmarks" in resolved_defaults.include_sections:
        benchmark_root = snapshot_dir / "benchmarks"
        latest_run: Path | None = None
        if benchmark_root.exists():
            run_dirs = [path for path in benchmark_root.iterdir() if path.is_dir()]
            if run_dirs:
                latest_run = max(run_dirs, key=lambda candidate: candidate.stat().st_mtime)
        if latest_run is not None:
            sections["benchmarks"] = {
                "root": _rel_path(benchmark_root),
                "latest_slug": latest_run.name,
                "summary_path": _rel_path(latest_run / "benchmark_summary.csv"),
                "baseline_metrics_path": _rel_path(latest_run / "baseline_metrics.csv"),
                "comparison_path": _rel_path(latest_run / "performance_comparison_table.csv"),
                "audit_path": _rel_path(latest_run / "benchmark_audit.csv"),
                "metadata_path": _rel_path(latest_run / "metadata.json"),
            }

    if "extended_diagnostics" in resolved_defaults.include_sections and diagnostics is not None:
        diagnostics_section: dict[str, object] = {
            "tail_metrics_path": _rel_path(diagnostics.output_directory / "tail_metrics.csv"),
            "turnover_distribution_path": _rel_path(diagnostics.output_directory / "turnover_distribution.csv"),
            "benchmark_deltas_path": _rel_path(diagnostics.output_directory / "benchmark_deltas.csv"),
            "config_path": _rel_path(diagnostics.output_directory / "config.json"),
        }
        diagnostics_summary = _summarize_diagnostics(diagnostics)
        diagnostics_section["summary"] = diagnostics_summary
        sections["extended_diagnostics"] = diagnostics_section

    if "proxy_datasets" in resolved_defaults.include_sections and proxy_datasets is not None:
        proxy_root = proxy_datasets.base_directory / "proxy_datasets"
        proxy_section: dict[str, object] = {
            "summary_path": _rel_path(proxy_root / "summary.csv"),
            "datasets": [run.spec.slug for run in proxy_datasets.runs],
        }
        proxy_metadata = {
            run.spec.slug: _ProxyMetadataEntry(
                status=run.status,
                allow_missing=run.spec.allow_missing,
                tags=list(run.spec.tags),
                message=run.message,
            )
            for run in proxy_datasets.runs
        }
        proxy_section["dataset_status"] = {slug: meta["status"] for slug, meta in proxy_metadata.items()}
        proxy_section["allow_missing"] = {slug: meta["allow_missing"] for slug, meta in proxy_metadata.items()}
        sections["proxy_datasets"] = proxy_section

    if "vintage_simulations" in resolved_defaults.include_sections and vintage is not None:
        vintage_root = vintage.base_directory / "vintage"
        vintage_windows_metadata = [
            _VintageWindowMetadataEntry(
                slug=run.window.slug,
                label=run.window.label,
                status=run.status,
                fold_count=run.fold_count,
                min_folds=run.window.min_folds,
                message=run.message,
            )
            for run in vintage.runs
        ]
        sections["vintage_simulations"] = {
            "summary_path": _rel_path(vintage_root / "summary.csv"),
            "windows": vintage_windows_metadata,
        }

    payload: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "sections": cast(object, sections),
        "alert_channels": list(resolved_defaults.alert_channels),
        "dashboard_targets": dict(resolved_defaults.dashboard_targets),
        "alert_rules": dict(resolved_defaults.alert_rules),
        "automation_targets": dict(resolved_defaults.automation_targets),
    }
    if monte_carlo is not None:
        payload["monte_carlo_metadata"] = {
            "overlay_events": int(sum(len(path.overlay_events) for path in monte_carlo.paths)),
            "report_metrics": list(monte_carlo.stress_config.report_metrics),
            "report_quantiles": list(monte_carlo.stress_config.report_quantiles),
            "overlay_category_stats": overlay_category_stats or [],
            "baseline_metrics": baseline_metrics_payload,
        }
    if heatmap_metadata is not None:
        payload["parameter_heatmap_metadata"] = heatmap_metadata
    if diagnostics_summary is not None:
        payload["diagnostics_metadata"] = diagnostics_summary
    if proxy_metadata is not None:
        payload["proxy_dataset_metadata"] = proxy_metadata
    if vintage_windows_metadata is not None:
        payload["vintage_metadata"] = vintage_windows_metadata
    if sensitivity_metadata is not None:
        payload["phase4_sensitivity_metadata"] = sensitivity_metadata
    if data_quality_summary is not None:
        payload["phase4_data_quality"] = data_quality_summary
    if outlier_summary is not None:
        payload["phase4_outlier_summary"] = outlier_summary
    snapshot_path = snapshot_dir / resolved_defaults.filename
    snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    MONITORING_EXPORT_COUNTER.labels(status="success").inc()
    return MonitoringSnapshotResult(path=snapshot_path, payload=payload)


# ===== Liquidity Mitigation Experiments =====


@dataclass(slots=True)
class LiquidityMitigationScenario:
    """
    Configuration for a liquidity mitigation experiment.

    Attributes
    ----------
    name : str
        Human-readable scenario name used in output directories.
    rolling_turnover_smoothing : float
        Turnover smoothing applied to the rolling beta strategy.
    stable_turnover_smoothing : float
        Turnover smoothing applied to the stable beta strategy.
    liquidity_config : LiquidityScalingConfig
        Liquidity scaling configuration to apply when deriving regime controls.
    notes : str | None
        Optional descriptive notes for reporting.
    """

    name: str
    rolling_turnover_smoothing: float
    stable_turnover_smoothing: float
    liquidity_config: LiquidityScalingConfig
    notes: str | None = None

    def __post_init__(self) -> None:
        for attr_name, value in (
            ("rolling_turnover_smoothing", self.rolling_turnover_smoothing),
            ("stable_turnover_smoothing", self.stable_turnover_smoothing),
        ):
            if not (0.0 <= value < 1.0):
                msg = f"{attr_name} must be in [0, 1)"
                raise ValueError(msg)


@dataclass(slots=True)
class LiquidityMitigationResult:
    """
    Summary of a liquidity mitigation experiment run.

    Attributes
    ----------
    scenario_name : str
        Scenario identifier.
    notes : str | None
        Optional scenario notes carried through from the configuration.
    rolling_turnover_smoothing : float
        Applied turnover smoothing for rolling betas.
    stable_turnover_smoothing : float
        Applied turnover smoothing for stable betas.
    severe_threshold : float
        Severe attribution threshold used for liquidity scaling.
    moderate_threshold : float
        Moderate attribution threshold used for liquidity scaling.
    severe_regime_multiplier : float
        Regime multiplier applied under severe drag.
    moderate_regime_multiplier : float
        Regime multiplier applied under moderate drag.
    severe_liquidity_multiplier : float
        Liquidity factor multiplier under severe drag.
    moderate_liquidity_multiplier : float
        Liquidity factor multiplier under moderate drag.
    rolling_sharpe : float
        Test-period Sharpe ratio for the rolling beta strategy.
    rolling_sharpe_delta : float
        Sharpe ratio delta versus the baseline scenario.
    rolling_turnover : float
        Average turnover for the rolling beta strategy.
    rolling_transaction_costs : float
        Transaction costs for the rolling beta strategy (dollars).
    rolling_transaction_costs_delta : float
        Transaction cost delta versus the baseline scenario.
    rolling_rate_hiking_liquidity : float | None
        Liquidity contribution within the Rate Hiking regime (annualised).
    rolling_rate_hiking_liquidity_delta : float | None
        Liquidity contribution delta versus the baseline scenario.
    stable_sharpe : float
        Test-period Sharpe ratio for the stable beta strategy.
    stable_sharpe_delta : float
        Sharpe ratio delta versus the baseline scenario.
    stable_turnover : float
        Average turnover for the stable beta strategy.
    stable_transaction_costs : float
        Transaction costs for the stable beta strategy (dollars).
    stable_transaction_costs_delta : float
        Transaction cost delta versus the baseline scenario.
    output_directory : Path
        Directory containing detailed artefacts for the scenario.
    walk_forward_sharpe_mean : float | None
        Mean Sharpe ratio across walk-forward folds (rolling betas).
    walk_forward_sharpe_std : float | None
        Standard deviation of walk-forward Sharpe ratios (rolling betas).
    walk_forward_sharpe_min : float | None
        Minimum walk-forward Sharpe ratio observed.
    walk_forward_sharpe_max : float | None
        Maximum walk-forward Sharpe ratio observed.
    walk_forward_output_directory : Path | None
        Directory containing walk-forward artefacts when generated.
    """

    scenario_name: str
    notes: str | None
    rolling_turnover_smoothing: float
    stable_turnover_smoothing: float
    severe_threshold: float
    moderate_threshold: float
    severe_regime_multiplier: float
    moderate_regime_multiplier: float
    severe_liquidity_multiplier: float
    moderate_liquidity_multiplier: float
    rolling_sharpe: float
    rolling_sharpe_delta: float
    rolling_turnover: float
    rolling_transaction_costs: float
    rolling_transaction_costs_delta: float
    rolling_rate_hiking_liquidity: float | None
    rolling_rate_hiking_liquidity_delta: float | None
    stable_sharpe: float
    stable_sharpe_delta: float
    stable_turnover: float
    stable_transaction_costs: float
    stable_transaction_costs_delta: float
    output_directory: Path
    walk_forward_sharpe_mean: float | None = None
    walk_forward_sharpe_std: float | None = None
    walk_forward_sharpe_min: float | None = None
    walk_forward_sharpe_max: float | None = None
    walk_forward_output_directory: Path | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        """Convert the result into a serialisable dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "notes": self.notes,
            "rolling_turnover_smoothing": self.rolling_turnover_smoothing,
            "stable_turnover_smoothing": self.stable_turnover_smoothing,
            "severe_threshold": self.severe_threshold,
            "moderate_threshold": self.moderate_threshold,
            "severe_regime_multiplier": self.severe_regime_multiplier,
            "moderate_regime_multiplier": self.moderate_regime_multiplier,
            "severe_liquidity_multiplier": self.severe_liquidity_multiplier,
            "moderate_liquidity_multiplier": self.moderate_liquidity_multiplier,
            "rolling_sharpe": self.rolling_sharpe,
            "rolling_sharpe_delta": self.rolling_sharpe_delta,
            "rolling_turnover": self.rolling_turnover,
            "rolling_transaction_costs": self.rolling_transaction_costs,
            "rolling_transaction_costs_delta": self.rolling_transaction_costs_delta,
            "rolling_rate_hiking_liquidity": self.rolling_rate_hiking_liquidity,
            "rolling_rate_hiking_liquidity_delta": self.rolling_rate_hiking_liquidity_delta,
            "stable_sharpe": self.stable_sharpe,
            "stable_sharpe_delta": self.stable_sharpe_delta,
            "stable_turnover": self.stable_turnover,
            "stable_transaction_costs": self.stable_transaction_costs,
            "stable_transaction_costs_delta": self.stable_transaction_costs_delta,
            "output_directory": str(self.output_directory),
            "walk_forward_sharpe_mean": self.walk_forward_sharpe_mean,
            "walk_forward_sharpe_std": self.walk_forward_sharpe_std,
            "walk_forward_sharpe_min": self.walk_forward_sharpe_min,
            "walk_forward_sharpe_max": self.walk_forward_sharpe_max,
            "walk_forward_output_directory": (
                str(self.walk_forward_output_directory)
                if self.walk_forward_output_directory is not None
                else None
            ),
        }


# ===== Benchmark Export Helpers =====


def _resolve_benchmark_root(output_dir: Path) -> Path:
    for candidate in (output_dir, *output_dir.parents):
        if candidate.name == "backtesting":
            return candidate / "benchmarks"
    return output_dir / "benchmarks"


def _benchmark_run_slug(split: TrainTestSplit) -> str:
    return (
        f"train_{split.train_start.date().isoformat()}_{split.train_end.date().isoformat()}__"
        f"test_{split.test_start.date().isoformat()}_{split.test_end.date().isoformat()}"
    )


def _serialize_metrics(strategy: str, metrics: PerformanceMetrics) -> dict[str, object]:
    return {
        "strategy": strategy,
        "annualized_return": metrics.annualized_return,
        "cumulative_return": metrics.cumulative_return,
        "annualized_volatility": metrics.annualized_volatility,
        "maximum_drawdown": metrics.maximum_drawdown,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "information_ratio": metrics.information_ratio,
        "turnover_rate": metrics.turnover_rate,
        "transaction_costs_total": metrics.transaction_costs_total,
        "transaction_costs_pct": metrics.transaction_costs_pct,
        "num_rebalances": metrics.num_rebalances,
        "var_95": metrics.var_95,
        "var_99": metrics.var_99,
        "cvar_95": metrics.cvar_95,
        "cvar_99": metrics.cvar_99,
        "monthly_return_mean": metrics.monthly_return_mean,
        "monthly_return_std": metrics.monthly_return_std,
        "start_date": metrics.start_date.isoformat(),
        "end_date": metrics.end_date.isoformat(),
        "total_days": metrics.total_days,
    }


def _build_ordered_baseline_comparison(
    strategies: Sequence[str],
    comparison_df: pl.DataFrame,
) -> pl.DataFrame:
    if not strategies:
        return comparison_df
    if not comparison_df.columns:
        return comparison_df
    strategy_rows = {str(row.get("strategy")): row for row in comparison_df.to_dicts()}
    ordered_rows: list[dict[str, object]] = []
    columns = comparison_df.columns
    for strategy in strategies:
        record = strategy_rows.get(strategy)
        if record is None:
            placeholder: dict[str, object] = {}
            for column in columns:
                placeholder[column] = float("nan")
            placeholder["strategy"] = strategy
            ordered_rows.append(placeholder)
        else:
            ordered_rows.append(record)
    return pl.DataFrame(ordered_rows, schema=comparison_df.schema)


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float("nan")
    if value is None:
        return float("nan")
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float("nan")


def _build_benchmark_metric_audit(
    strategies: Sequence[str],
    comparison_df: pl.DataFrame,
    metrics_df: pl.DataFrame,
) -> pl.DataFrame:
    if not strategies or comparison_df.is_empty() or metrics_df.is_empty():
        return pl.DataFrame()
    comparison_rows = {str(row.get("strategy")): row for row in comparison_df.to_dicts()}
    metrics_rows = {str(row.get("strategy")): row for row in metrics_df.to_dicts()}
    metric_pairs = (
        ("sharpe_ratio", "sharpe_ratio"),
        ("sortino_ratio", "sortino_ratio"),
        ("calmar_ratio", "calmar_ratio"),
        ("turnover_rate", "turnover_rate"),
        ("transaction_costs", "transaction_costs_total"),
    )
    audit_rows: list[dict[str, object]] = []
    for strategy in strategies:
        comparison_row = comparison_rows.get(strategy, {})
        metrics_row = metrics_rows.get(strategy, {})
        for comparison_field, metrics_field in metric_pairs:
            comparison_value = _to_float(comparison_row.get(comparison_field))
            baseline_value = _to_float(metrics_row.get(metrics_field))
            delta = (
                comparison_value - baseline_value
                if not math.isnan(comparison_value) and not math.isnan(baseline_value)
                else float("nan")
            )
            audit_rows.append({
                "strategy": strategy,
                "metric": metrics_field,
                "comparison_value": comparison_value,
                "mirror_value": baseline_value,
                "delta": delta,
            })
    return pl.DataFrame(audit_rows)


def _export_benchmark_tables(
    suite: BacktestSuite,
    comparison_df: pl.DataFrame,
    output_dir: Path,
) -> None:
    if not suite.baseline_strategies:
        return
    summary = suite.benchmark_summary()
    if summary.is_empty():
        return

    root = _resolve_benchmark_root(output_dir)
    run_dir = root / _benchmark_run_slug(suite.split)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary.write_csv(run_dir / "benchmark_summary.csv")

    metrics_rows: list[dict[str, object]] = []
    for strategy in suite.baseline_strategies:
        metrics = suite.metrics.get(strategy)
        if metrics is None:
            metrics_rows.append({
                "strategy": strategy,
                "annualized_return": float("nan"),
                "cumulative_return": float("nan"),
                "annualized_volatility": float("nan"),
                "maximum_drawdown": float("nan"),
                "sharpe_ratio": float("nan"),
                "sortino_ratio": float("nan"),
                "calmar_ratio": float("nan"),
                "information_ratio": float("nan"),
                "turnover_rate": float("nan"),
                "transaction_costs_total": float("nan"),
                "transaction_costs_pct": float("nan"),
                "num_rebalances": 0,
                "var_95": float("nan"),
                "var_99": float("nan"),
                "cvar_95": float("nan"),
                "cvar_99": float("nan"),
                "monthly_return_mean": float("nan"),
                "monthly_return_std": float("nan"),
                "start_date": None,
                "end_date": None,
                "total_days": 0,
            })
        else:
            metrics_rows.append(_serialize_metrics(strategy, metrics))
    metrics_frame = pl.DataFrame(metrics_rows)
    metrics_frame.write_csv(run_dir / "baseline_metrics.csv")

    baseline_comparison = comparison_df.filter(
        pl.col("strategy").is_in(list(suite.baseline_strategies)),
    )
    ordered_comparison = _build_ordered_baseline_comparison(
        suite.baseline_strategies,
        baseline_comparison,
    )
    if not ordered_comparison.is_empty():
        ordered_comparison.write_csv(run_dir / "performance_comparison_table.csv")

    metadata: dict[str, object] = {
        "train_start": suite.split.train_start.isoformat(),
        "train_end": suite.split.train_end.isoformat(),
        "test_start": suite.split.test_start.isoformat(),
        "test_end": suite.split.test_end.isoformat(),
        "strategies": list(suite.baseline_strategies),
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    rolling_metrics = suite.metrics.get(THREE_D_ROLLING_STRATEGY)
    metadata.update({
        "phase3_sharpe_strategy": THREE_D_ROLLING_STRATEGY,
        "phase3_sharpe_threshold": PHASE3_TARGET_SHARPE,
        "phase3_sharpe_value": rolling_metrics.sharpe_ratio if rolling_metrics is not None else None,
        "phase3_sharpe_ok": (
            rolling_metrics.sharpe_ratio >= PHASE3_TARGET_SHARPE
            if rolling_metrics is not None
            else False
        ),
    })
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    audit_frame = _build_benchmark_metric_audit(
        suite.baseline_strategies,
        ordered_comparison,
        metrics_frame,
    )
    if not audit_frame.is_empty():
        audit_frame.write_csv(run_dir / "benchmark_audit.csv")


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

    _validate_split_against_dataset(dataset, split, BACKTEST_DEFAULTS)

    return _execute_backtest_suite(
        dataset=dataset,
        output_dir=output_dir,
        split=split,
        config_overrides=config_overrides,
    )


def run_walk_forward_backtest_suite(
    dataset_path: Path,
    output_dir: Path,
    splits: Sequence[TrainTestSplit] | None = None,
    *,
    walk_forward_config: WalkForwardConfig | None = None,
    config_overrides: dict[str, Any] | None = None,
    liquidity_config: LiquidityScalingConfig | None = None,
    turnover_overrides: Mapping[str, float] | None = None,
    summary_directory: Path | None = None,
) -> WalkForwardBacktestResult:
    """
    Execute walk-forward backtesting across sequential train/test windows.

    Parameters
    ----------
    dataset_path : Path
        Path to the sector dataset directory or Parquet file.
    output_dir : Path
        Base directory where fold artefacts will be written.
    splits : Sequence[TrainTestSplit] | None
        Optional precomputed splits. If omitted, generated from ``walk_forward_config``.
    walk_forward_config : WalkForwardConfig | None
        Configuration describing walk-forward windows. Ignored when ``splits`` provided.
    config_overrides : dict[str, Any] | None
        Optional overrides for ``BacktestConfig`` parameters.
    liquidity_config : LiquidityScalingConfig | None
        Liquidity scaling configuration applied to each fold (defaults to ``LiquidityScalingConfig()``).
    turnover_overrides : Mapping[str, float] | None
        Optional turnover smoothing overrides keyed by strategy slug (e.g., ``{"3d_factor_rolling": 0.4}``).
    summary_directory : Path | None
        Optional directory (absolute or relative to ``output_dir``) where walk-forward
        summaries should be written. Defaults to ``output_dir / "walk_forward"``.

    Returns
    -------
    WalkForwardBacktestResult
        Object containing per-fold suites and aggregated summaries.
    """
    if not dataset_path.exists():
        msg = f"Dataset path does not exist: {dataset_path}"
        raise FileNotFoundError(msg)

    dataset = _load_sector_dataset(dataset_path)

    if splits is not None:
        resolved_splits = [split for split in splits]
    else:
        if walk_forward_config is None:
            bounds = dataset.sector_returns.select(
                pl.col("timestamp").min().alias("min_ts"),
                pl.col("timestamp").max().alias("max_ts"),
            ).row(0)
            data_start = bounds[0]
            data_end = bounds[1]

            def _ensure_utc(value: datetime) -> datetime:
                return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

            default_primary = define_train_test_split()
            walk_forward_config = WalkForwardConfig(
                start_date=_ensure_utc(max(_ensure_utc(data_start), default_primary.train_start)),
                end_date=_ensure_utc(min(_ensure_utc(data_end), default_primary.test_end)),
                train_years=5,
                test_years=1,
                step_years=1,
            )
        resolved_splits = walk_forward_config.to_splits()

    if not resolved_splits:
        raise ValueError("Walk-forward configuration produced no splits")

    if summary_directory is None:
        summary_dir = output_dir / "walk_forward"
    elif summary_directory.is_absolute():
        summary_dir = summary_directory
    else:
        summary_dir = output_dir / summary_directory
    defaults = BACKTEST_DEFAULTS
    suites: list[BacktestSuite] = []

    for index, split in enumerate(resolved_splits, start=1):
        _validate_split_against_dataset(
            dataset,
            split,
            defaults,
            min_training_days=1,
            min_testing_days=1,
        )
        LOGGER.info(
            "Running walk-forward fold",
            fold=index,
            train_period=f"{split.train_start.date()}→{split.train_end.date()}",
            test_period=f"{split.test_start.date()}→{split.test_end.date()}",
        )
        fold_dir = summary_dir / f"fold_{index:02d}"
        suite = _execute_backtest_suite(
            dataset=dataset,
            output_dir=fold_dir,
            split=split,
            config_overrides=config_overrides,
            liquidity_config=liquidity_config,
            turnover_overrides=turnover_overrides,
        )
        suites.append(suite)

    result = WalkForwardBacktestResult(splits=resolved_splits, suites=suites)
    liquidity_reference = liquidity_config or defaults.build_liquidity_config()
    turnover_metadata = {
        "stable": float(
            (turnover_overrides or {}).get("3d_factor_stable", defaults.stable_turnover_smoothing)
        ),
        "rolling": float(
            (turnover_overrides or {}).get("3d_factor_rolling", defaults.rolling_turnover_smoothing)
        ),
    }
    liquidity_metadata = {
        "severe_threshold": liquidity_reference.severe_threshold,
        "moderate_threshold": liquidity_reference.moderate_threshold,
        "severe_regime_multiplier": liquidity_reference.severe_regime_multiplier,
        "moderate_regime_multiplier": liquidity_reference.moderate_regime_multiplier,
        "severe_liquidity_multiplier": liquidity_reference.severe_liquidity_multiplier,
        "moderate_liquidity_multiplier": liquidity_reference.moderate_liquidity_multiplier,
        "neutral_liquidity_multiplier": liquidity_reference.neutral_liquidity_multiplier,
        "floor": liquidity_reference.floor,
    }
    metadata = {
        "risk_free_rate": defaults.risk_free_rate,
        "baseline_strategies": list(defaults.baseline_strategies),
        "turnover_smoothing": turnover_metadata,
        "liquidity_config": liquidity_metadata,
        "summaries_directory": str(summary_dir),
        "split_count": len(resolved_splits),
    }
    if turnover_overrides is not None:
        metadata["turnover_overrides"] = dict(turnover_overrides)
    if walk_forward_config is not None:
        metadata["walk_forward_config"] = {
            "train_years": walk_forward_config.train_years,
            "test_years": walk_forward_config.test_years,
            "step_years": walk_forward_config.step_years,
            "start_date": walk_forward_config.start_date.date().isoformat(),
            "end_date": walk_forward_config.end_date.date().isoformat(),
        }
    metadata["splits"] = [
        {
            "train_start": split.train_start.date().isoformat(),
            "train_end": split.train_end.date().isoformat(),
            "test_start": split.test_start.date().isoformat(),
            "test_end": split.test_end.date().isoformat(),
        }
        for split in resolved_splits
    ]
    result.write_summaries(summary_dir, metadata=metadata)

    LOGGER.info(
        "Walk-forward backtest suite completed",
        num_folds=len(resolved_splits),
        output_dir=str(summary_dir),
    )

    return result


def run_walk_forward_permutation(
    dataset_path: Path,
    output_dir: Path,
    *,
    permutation: WalkForwardPermutationDefaults,
    start_date: datetime,
    end_date: datetime,
    config_overrides: dict[str, Any] | None = None,
    liquidity_config: LiquidityScalingConfig | None = None,
    turnover_overrides: Mapping[str, float] | None = None,
    summary_directory: Path | None = None,
) -> WalkForwardPermutationRun:
    """
    Execute a single walk-forward permutation with optional nested sweeps.

    Parameters
    ----------
    dataset_path : Path
        Path to the sector dataset directory or Parquet file.
    output_dir : Path
        Base directory where artefacts will be written.
    permutation : WalkForwardPermutationDefaults
        Permutation metadata describing the outer walk-forward sweep.
    start_date : datetime
        Earliest observation included in the outer sweep.
    end_date : datetime
        Latest observation included in the outer sweep.
    config_overrides : dict[str, Any] | None
        Optional overrides for ``BacktestConfig`` parameters.
    liquidity_config : LiquidityScalingConfig | None
        Liquidity scaling configuration applied to each fold.
    turnover_overrides : Mapping[str, float] | None
        Optional turnover smoothing overrides keyed by strategy slug.
    summary_directory : Path | None
        Directory (absolute or relative to ``output_dir``) for outer sweep artefacts.

    Returns
    -------
    WalkForwardPermutationRun
        Container with outer and nested results plus output directory metadata.
    """
    LOGGER.info(
        "Running walk-forward permutation",
        permutation=permutation.slug,
        train_years=permutation.train_years,
        test_years=permutation.test_years,
        step_years=permutation.step_years,
    )
    config = permutation.to_config(start_date=start_date, end_date=end_date)
    default_summary = Path("walk_forward") / "permutations" / permutation.slug
    summary_path = summary_directory or default_summary
    outer_result = run_walk_forward_backtest_suite(
        dataset_path=dataset_path,
        output_dir=output_dir,
        walk_forward_config=config,
        config_overrides=config_overrides,
        liquidity_config=liquidity_config,
        turnover_overrides=turnover_overrides,
        summary_directory=summary_path,
    )
    summary_output = summary_path if summary_path.is_absolute() else output_dir / summary_path
    nested_results: list[NestedCrossValidationResult] = []

    if permutation.nested is not None:
        nested_spec = permutation.nested
        for index, outer_split in enumerate(outer_result.splits, start=1):
            try:
                nested_config = WalkForwardConfig(
                    start_date=outer_split.train_start,
                    end_date=outer_split.train_end,
                    train_years=nested_spec.train_years,
                    test_years=nested_spec.test_years,
                    step_years=nested_spec.step_years,
                )
            except ValueError as error:
                LOGGER.warning(
                    "Skipping nested walk-forward (invalid configuration)",
                    permutation=permutation.slug,
                    outer_fold=index,
                    error=str(error),
                )
                continue
            try:
                nested_splits = nested_config.to_splits()
            except ValueError as error:
                LOGGER.warning(
                    "Skipping nested walk-forward (unable to materialise splits)",
                    permutation=permutation.slug,
                    outer_fold=index,
                    error=str(error),
                )
                continue
            if len(nested_splits) < nested_spec.min_folds:
                LOGGER.warning(
                    "Skipping nested walk-forward (insufficient folds)",
                    permutation=permutation.slug,
                    outer_fold=index,
                    generated_folds=len(nested_splits),
                    required_folds=nested_spec.min_folds,
                )
                continue
            nested_summary = summary_path / "nested" / f"outer_fold_{index:02d}"
            nested_result = run_walk_forward_backtest_suite(
                dataset_path=dataset_path,
                output_dir=output_dir,
                splits=nested_splits,
                config_overrides=config_overrides,
                liquidity_config=liquidity_config,
                turnover_overrides=turnover_overrides,
                summary_directory=nested_summary,
            )
            nested_output = (
                nested_summary
                if nested_summary.is_absolute()
                else output_dir / nested_summary
            )
            nested_results.append(
                NestedCrossValidationResult(
                    spec=nested_spec,
                    outer_fold_index=index,
                    outer_split=outer_split,
                    config=nested_config,
                    result=nested_result,
                    output_directory=nested_output,
                )
            )

    metadata_path = summary_output / "permutation_metadata.json"
    metadata_payload = permutation.to_dict()
    metadata_payload.update(
        {
            "outer_folds": len(outer_result.splits),
            "nested_runs": len(nested_results),
        }
    )
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")

    LOGGER.info(
        "Completed walk-forward permutation",
        permutation=permutation.slug,
        outer_folds=len(outer_result.splits),
        nested_runs=len(nested_results),
        output=str(summary_output),
    )

    return WalkForwardPermutationRun(
        spec=permutation,
        config=config,
        outer_result=outer_result,
        summary_directory=summary_output,
        nested_results=nested_results,
    )


def run_multi_horizon_walk_forward_analysis(
    dataset_path: Path,
    output_dir: Path,
    *,
    start_date: datetime,
    end_date: datetime,
    permutations: Sequence[WalkForwardPermutationDefaults] | None = None,
    config_overrides: dict[str, Any] | None = None,
    liquidity_config: LiquidityScalingConfig | None = None,
    turnover_overrides: Mapping[str, float] | None = None,
    include_primary_root: bool = True,
) -> MultiHorizonWalkForwardResult:
    """
    Execute multi-horizon walk-forward analysis across configured permutations.

    Parameters
    ----------
    dataset_path : Path
        Path to the sector dataset directory or Parquet file.
    output_dir : Path
        Base directory where artefacts will be written.
    start_date : datetime
        Earliest observation considered across permutations.
    end_date : datetime
        Latest observation considered across permutations.
    permutations : Sequence[WalkForwardPermutationDefaults] | None
        Optional permutations to execute. Defaults to ``ThreeDRiskBacktestDefaults.walk_forward_permutations``.
    config_overrides : dict[str, Any] | None
        Optional overrides for ``BacktestConfig`` parameters.
    liquidity_config : LiquidityScalingConfig | None
        Liquidity scaling configuration applied to each fold.
    turnover_overrides : Mapping[str, float] | None
        Optional turnover smoothing overrides keyed by strategy slug.
    include_primary_root : bool
        When ``True``, the first permutation mirrors artefacts into ``walk_forward`` for
        backwards compatibility with existing reports.

    Returns
    -------
    MultiHorizonWalkForwardResult
        Aggregated results for all executed permutations.
    """
    defaults = BACKTEST_DEFAULTS
    resolved_permutations = tuple(permutations or defaults.walk_forward_permutations)
    if not resolved_permutations:
        raise ValueError("No walk-forward permutations provided")

    runs: dict[str, WalkForwardPermutationRun] = {}
    for index, permutation in enumerate(resolved_permutations):
        if include_primary_root and index == 0:
            summary_dir = Path("walk_forward")
        else:
            summary_dir = Path("walk_forward") / "permutations" / permutation.slug
        run = run_walk_forward_permutation(
            dataset_path=dataset_path,
            output_dir=output_dir,
            permutation=permutation,
            start_date=start_date,
            end_date=end_date,
            config_overrides=config_overrides,
            liquidity_config=liquidity_config,
            turnover_overrides=turnover_overrides,
            summary_directory=summary_dir,
        )
        runs[permutation.slug] = run

        if include_primary_root and index == 0:
            permutations_root = output_dir / Path("walk_forward") / "permutations"
            alias_dir = permutations_root / permutation.slug
            alias_dir.mkdir(parents=True, exist_ok=True)
            readme_path = alias_dir / "README.txt"
            readme_path.write_text(
                (
                    "Outputs for this permutation are stored in the parent "
                    "walk_forward directory to preserve legacy artefact paths.\n"
                    "See ../../aggregate_metrics.csv and associated fold files."
                ),
                encoding="utf-8",
            )
            metadata_source = run.summary_directory / "permutation_metadata.json"
            metadata_target = alias_dir / "permutation_metadata.json"
            if metadata_source.exists():
                metadata_target.write_text(metadata_source.read_text(encoding="utf-8"), encoding="utf-8")

    return MultiHorizonWalkForwardResult(base_directory=output_dir, runs=runs)


# ===== Helper Functions =====


def _build_regime_return_frame(
    result: BacktestResult,
    regimes: Sequence[MarketRegime],
) -> pl.DataFrame:
    """
    Construct a regime-labelled return frame from a backtest result.

    Parameters
    ----------
    result : BacktestResult
        Baseline backtest result for the target strategy.
    regimes : Sequence[MarketRegime]
        Regime definitions used to label returns.

    Returns
    -------
    pl.DataFrame
        DataFrame containing ``timestamp``, ``return``, and ``regime`` columns.
    """
    dates = list(result.dates)
    returns = np.asarray(result.returns, dtype=np.float64)
    if len(dates) == returns.size + 1:
        # BacktestResult stores an initial valuation timestamp with no associated return.
        dates = dates[1:]
    if returns.size != len(dates):
        msg = (
            "Backtest result returns and dates are misaligned "
            f"(returns={returns.size}, dates={len(dates)})"
        )
        raise ValueError(msg)

    sorted_regimes = sorted(regimes, key=lambda regime: regime.start)
    labels: list[str | None] = []
    regime_index = 0
    for date in dates:
        while regime_index < len(sorted_regimes) and date > sorted_regimes[regime_index].end:
            regime_index += 1
        if regime_index < len(sorted_regimes):
            candidate = sorted_regimes[regime_index]
            if candidate.start <= date <= candidate.end:
                labels.append(candidate.name)
                continue
        labels.append(None)

    frame = pl.DataFrame({
        "timestamp": dates,
        "return": returns,
        "regime": labels,
    }).sort("timestamp")
    return frame.filter(pl.col("regime").is_not_null())


def _prepare_regime_blocks(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """
    Transform a regime-labelled frame into numpy blocks per regime.
    """
    if frame.is_empty():
        return {}
    blocks: dict[str, np.ndarray] = {}
    grouped = frame.group_by("regime", maintain_order=True).agg(
        pl.col("return").alias("returns"),
    )
    for row in grouped.iter_rows(named=True):
        regime_name = cast(str, row["regime"])
        returns = np.asarray(row["returns"], dtype=np.float64)
        if returns.size == 0:
            continue
        blocks[regime_name] = returns
    return blocks


def _apply_monte_carlo_overlays(
    rng: np.random.Generator,
    returns: np.ndarray,
    *,
    regime_sequence: tuple[str, ...],
    block_lengths: tuple[int, ...],
    overlays: Sequence[MonteCarloShockOverlayDefaults],
) -> list[MonteCarloOverlayActivation]:
    """
    Apply macro shock overlays in-place to the synthetic return path.
    """
    if returns.size == 0:
        return []
    offsets: list[int] = []
    cursor = 0
    for length in block_lengths:
        offsets.append(cursor)
        cursor += length
    total_length = returns.size
    applied: list[MonteCarloOverlayActivation] = []

    def _resolve_regime(index: int) -> str | None:
        for block_index, start in enumerate(offsets):
            end = start + block_lengths[block_index]
            if start <= index < end:
                return regime_sequence[block_index]
        return None

    for overlay in overlays:
        attempts = 0
        while attempts < overlay.max_applications:
            attempts += 1
            if rng.random() > overlay.probability:
                break
            candidate_ranges: list[tuple[int, int]] = []
            if overlay.regime_bias is not None:
                bias = set(overlay.regime_bias)
                for index, regime_name in enumerate(regime_sequence):
                    if regime_name not in bias:
                        continue
                    start = offsets[index]
                    end = start + block_lengths[index]
                    if start < end:
                        candidate_ranges.append((start, end))
            else:
                candidate_ranges.append((0, total_length))

            if not candidate_ranges:
                break
            start_bound, end_bound = candidate_ranges[rng.integers(0, len(candidate_ranges))]
            if start_bound >= end_bound:
                break
            start_idx = int(rng.integers(start_bound, end_bound))
            if start_idx >= total_length:
                break
            for day_offset in range(overlay.duration_days):
                target_idx = start_idx + day_offset
                if target_idx >= total_length:
                    break
                decay_multiplier = overlay.decay ** day_offset
                returns[target_idx] = returns[target_idx] + (overlay.magnitude * decay_multiplier)
            duration_applied = min(overlay.duration_days, max(0, total_length - start_idx))
            if duration_applied == 0:
                continue
            total_impact = float(sum(
                overlay.magnitude * (overlay.decay ** day_offset)
                for day_offset in range(duration_applied)
            ))
            applied.append(
                MonteCarloOverlayActivation(
                    name=overlay.name,
                    category=overlay.category,
                    start_index=start_idx,
                    duration=duration_applied,
                    regime=_resolve_regime(start_idx),
                    magnitude=overlay.magnitude,
                    decay=overlay.decay,
                    total_impact=total_impact,
                    tags=overlay.tags,
                ),
            )
    return applied


def _compute_monte_carlo_metrics(
    *,
    simulation_id: int,
    returns: np.ndarray,
    stress_config: MonteCarloStressDefaults,
    regime_sequence: tuple[str, ...],
    overlay_events: tuple[MonteCarloOverlayActivation, ...],
) -> MonteCarloStressPathResult:
    """
    Compute summary metrics for a synthetic Monte Carlo path.
    """
    if returns.size == 0:
        msg = "Synthetic Monte Carlo path is empty"
        raise ValueError(msg)

    clipped = np.clip(returns, -0.99, None)
    cumulative = np.cumprod(1.0 + clipped, dtype=np.float64)
    terminal_value = float(cumulative[-1])
    peaks = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / peaks - 1.0
    max_drawdown = float(drawdowns.min()) if drawdowns.size > 0 else 0.0

    rf_daily = (1.0 + stress_config.risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_returns = clipped - rf_daily
    mean_daily = float(np.mean(clipped))
    if clipped.size > 1:
        daily_vol = float(np.std(clipped, ddof=1))
    else:
        daily_vol = 0.0
    annualized_return = float((1.0 + mean_daily) ** 252 - 1.0)
    annualized_volatility = float(daily_vol * math.sqrt(252.0))
    sharpe = 0.0
    if daily_vol > 0.0:
        sharpe = float(np.mean(excess_returns) / daily_vol * math.sqrt(252.0))

    tail_quantile = float(np.quantile(clipped, 1.0 - stress_config.cvar_alpha))
    tail_mask = clipped <= tail_quantile + 1e-12
    if np.any(tail_mask):
        cvar = float(np.mean(clipped[tail_mask]))
    else:
        cvar = tail_quantile

    return MonteCarloStressPathResult(
        simulation_id=simulation_id,
        sharpe_ratio=sharpe,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        max_drawdown=max_drawdown,
        var_alpha=tail_quantile,
        cvar_alpha=cvar,
        terminal_value=terminal_value,
        positive_terminal=terminal_value >= 1.0,
        regime_sequence=regime_sequence,
        overlay_events=overlay_events,
        path_length=int(clipped.size),
    )


def _validate_split_against_dataset(
    dataset: SectorDataset,
    split: TrainTestSplit,
    defaults: ThreeDRiskBacktestDefaults,
    *,
    min_training_days: int | None = None,
    min_testing_days: int | None = None,
) -> None:
    """
    Ensure the supplied split is compatible with the dataset window.

    Parameters
    ----------
    dataset : SectorDataset
        Loaded dataset containing aligned sector and factor returns.
    split : TrainTestSplit
        Train/test split to validate.
    defaults : ThreeDRiskBacktestDefaults
        Configuration thresholds governing validation rules.

    Raises
    ------
    ValueError
        If validation fails (e.g., insufficient history or coverage).
    """
    split.validate_no_overlap()
    if not validate_no_lookahead(split):
        msg = "Train/test split violates no-lookahead constraint (test starts before train ends)"
        raise ValueError(msg)
    training_days_required = min_training_days if min_training_days is not None else defaults.min_training_days
    if not validate_sufficient_training_data(split, min_trading_days=training_days_required):
        msg = (
            "Training window shorter than required: "
            f"{training_days_required} trading days minimum"
        )
        raise ValueError(msg)

    testing_days_required = min_testing_days if min_testing_days is not None else defaults.min_testing_days
    estimated_test_days = int(split.test_days * TRADING_DAY_RATIO)
    if estimated_test_days < testing_days_required:
        msg = (
            "Testing window shorter than required: "
            f"{testing_days_required} trading days minimum"
        )
        raise ValueError(msg)

    sector_timestamps = dataset.sector_returns.get_column("timestamp")
    if sector_timestamps.is_empty():
        raise ValueError("Sector returns dataset is empty")

    dataset_start_raw = sector_timestamps.min()
    dataset_end_raw = sector_timestamps.max()
    if dataset_start_raw is None or dataset_end_raw is None:
        raise ValueError("Unable to resolve dataset coverage window")

    if not isinstance(dataset_start_raw, datetime):
        msg = f"Expected datetime timestamp for dataset start, received {type(dataset_start_raw)!r}"
        raise TypeError(msg)
    if not isinstance(dataset_end_raw, datetime):
        msg = f"Expected datetime timestamp for dataset end, received {type(dataset_end_raw)!r}"
        raise TypeError(msg)

    dataset_start = dataset_start_raw
    dataset_end = dataset_end_raw

    tolerance = timedelta(days=defaults.coverage_tolerance_days)
    coverage_start = dataset_start - tolerance
    coverage_end = dataset_end + tolerance

    if split.train_start < coverage_start:
        msg = (
            "Training window begins before dataset coverage even after "
            f"{defaults.coverage_tolerance_days}-day tolerance "
            f"({dataset_start.isoformat()} → {dataset_end.isoformat()})"
        )
        raise ValueError(msg)

    if split.test_end > coverage_end:
        msg = (
            "Testing window extends beyond dataset coverage even after "
            f"{defaults.coverage_tolerance_days}-day tolerance "
            f"({dataset_start.isoformat()} → {dataset_end.isoformat()})"
        )
        raise ValueError(msg)


def _execute_backtest_suite(
    dataset: SectorDataset,
    output_dir: Path,
    split: TrainTestSplit,
    config_overrides: dict[str, Any] | None,
    *,
    liquidity_config: LiquidityScalingConfig | None = None,
    turnover_overrides: Mapping[str, float] | None = None,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> BacktestSuite:
    """
    Execute the backtest suite using a preloaded dataset and split.

    Parameters
    ----------
    dataset : SectorDataset
        Preloaded dataset containing sector and factor returns.
    output_dir : Path
        Directory where artefacts should be written.
    split : TrainTestSplit
        Train/test split configuration.
    config_overrides : dict[str, Any] | None
        Optional overrides for backtest configuration.
    liquidity_config : LiquidityScalingConfig | None
        Optional liquidity scaling configuration. Defaults to ``ThreeDRiskBacktestDefaults.build_liquidity_config()``.
    turnover_overrides : Mapping[str, float] | None
        Optional turnover smoothing overrides keyed by strategy slug
        (``3d_factor_stable`` or ``3d_factor_rolling``).
    strategy_overrides : Mapping[str, Mapping[str, Any]] | None
        Optional per-strategy overrides applied to strategy parameter dictionaries.
    """
    defaults = BACKTEST_DEFAULTS
    train_config = _create_backtest_config(split, config_overrides, period="train")
    test_config = _create_backtest_config(split, config_overrides, period="test")
    full_config = _create_backtest_config(split, config_overrides, period="full")
    coverage_tolerance = timedelta(days=defaults.coverage_tolerance_days)

    regimes = define_market_regimes()
    regime_resolver = _build_regime_resolver(regimes)
    resolved_liquidity_config = liquidity_config or defaults.build_liquidity_config()
    rolling_liquidity = load_liquidity_contributions_from_csv(
        LIQUIDITY_ATTRIBUTION_DIR,
        LIQUIDITY_STRATEGY_SLUG,
    )
    if not rolling_liquidity:
        rolling_liquidity = dict(defaults.liquidity_contribution_fallbacks)
    regime_scaling_map, liquidity_controls = build_regime_scaling_maps(
        rolling_liquidity,
        config=resolved_liquidity_config,
    )
    turnover_map: dict[str, float] = dict(turnover_overrides) if turnover_overrides is not None else {}
    stable_turnover = float(turnover_map.get("3d_factor_stable", defaults.stable_turnover_smoothing))
    rolling_turnover = float(turnover_map.get("3d_factor_rolling", defaults.rolling_turnover_smoothing))

    strategies_to_run: list[tuple[str, str, dict[str, Any]]] = [
        ("Equal Weight", "equal_weight", {}),
        ("60/40 Portfolio", "sixty_forty", {}),
        ("Risk Parity", "risk_parity", {}),
        ("Minimum Variance", "minimum_variance", {}),
        ("3D Factor (Stable Betas)", "3d_factor_stable", {
            "max_weight": 0.30,
            "min_observations": 180,
            "blend_to_equal": 0.20,
            "turnover_smoothing": stable_turnover,
        }),
        ("3D Factor (Rolling Betas)", "3d_factor_rolling", {
            "rolling_window": 252,
            "max_weight": 0.30,
            "min_observations": 180,
            "blend_to_equal": 0.20,
            "turnover_smoothing": rolling_turnover,
            "dynamic_factor_scaling": True,
            "regime_scaling": True,
            "regime_resolver": regime_resolver,
            "regime_scaling_map": regime_scaling_map,
            "regime_factor_multipliers": liquidity_controls,
            "regime_scaling_floor": resolved_liquidity_config.floor,
        }),
    ]

    train_results: dict[str, BacktestResult] = {}
    test_results: dict[str, BacktestResult] = {}
    full_results: dict[str, BacktestResult] = {}
    resolved_strategy_overrides = {
        key: dict(value)
        for key, value in (strategy_overrides or {}).items()
    }

    for strategy_display_name, strategy_key, strategy_params in strategies_to_run:
        LOGGER.info("Running backtests", strategy=strategy_display_name)
        params = dict(strategy_params)
        if resolved_strategy_overrides:
            override = resolved_strategy_overrides.get(strategy_key)
            if override is None:
                override = resolved_strategy_overrides.get(strategy_display_name)
            if override is not None:
                params.update(override)
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

    risk_free_rate = defaults.risk_free_rate
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
        if not regimes:
            continue
        if (
            result.start_date > regimes[0].start + coverage_tolerance
            or result.end_date < regimes[-1].end - coverage_tolerance
        ):
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
        turnover_overrides=turnover_map,
        baseline_strategies=defaults.baseline_strategies,
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

    _export_benchmark_tables(suite, comparison_df, output_dir)

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

    try:
        export_phase3_visuals(
            data_dir=output_dir,
            output_dir=output_dir / "visuals",
            config=liquidity_config,
        )
    except Exception:
        LOGGER.exception("Failed to export Phase 3 visuals", output_dir=str(output_dir))

    report_filename = f"backtest_results_{split.train_start.year}_{split.test_end.year}.md"
    report_path = output_dir / report_filename
    suite.to_markdown_report(report_path)

    LOGGER.info(
        "Backtest suite completed",
        num_strategies=len(test_results),
        output_dir=str(output_dir),
    )

    return suite


def _slugify_scenario_name(name: str) -> str:
    """Convert a scenario name into a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()
    return slug or "scenario"


def _extract_rate_hiking_liquidity(
    regime_attribution: dict[str, list[FactorAttribution]],
    strategy_name: str,
) -> float | None:
    """Return the liquidity contribution for the Rate Hiking regime."""
    for attribution in regime_attribution.get(strategy_name, []):
        if attribution.regime_name == "Rate Hiking Cycle":
            return attribution.factor_contributions.get("factor_liquidity")
    return None


def get_default_liquidity_mitigation_scenarios() -> list[LiquidityMitigationScenario]:
    """Return the default set of liquidity mitigation scenarios."""
    defaults = BACKTEST_DEFAULTS
    return [
        LiquidityMitigationScenario(
            name="Baseline Controls",
            rolling_turnover_smoothing=defaults.rolling_turnover_smoothing,
            stable_turnover_smoothing=defaults.stable_turnover_smoothing,
            liquidity_config=defaults.build_liquidity_config(),
            notes="Legacy configuration retained for comparison only.",
        ),
        LiquidityMitigationScenario(
            name="Turnover Smoothing 0.55/0.40",
            rolling_turnover_smoothing=0.55,
            stable_turnover_smoothing=0.40,
            liquidity_config=LiquidityScalingConfig(),
            notes="Higher smoothing to reduce turnover-induced drag.",
        ),
        LiquidityMitigationScenario(
            name="Tighter Liquidity Regime Scaling",
            rolling_turnover_smoothing=0.45,
            stable_turnover_smoothing=0.35,
            liquidity_config=LiquidityScalingConfig(
                severe_threshold=-0.015,
                moderate_threshold=-0.0075,
                severe_regime_multiplier=0.80,
                moderate_regime_multiplier=0.90,
                severe_liquidity_multiplier=0.50,
                moderate_liquidity_multiplier=0.65,
                neutral_liquidity_multiplier=1.0,
                floor=0.40,
            ),
            notes="Stricter thresholds to clamp liquidity drag in Rate Hiking cycles.",
        ),
        LiquidityMitigationScenario(
            name="Turnover Stress Test",
            rolling_turnover_smoothing=max(defaults.rolling_turnover_smoothing - 0.15, 0.10),
            stable_turnover_smoothing=max(defaults.stable_turnover_smoothing - 0.10, 0.05),
            liquidity_config=defaults.build_liquidity_config(),
            notes="Lower smoothing to stress transaction cost impact while retaining default liquidity config.",
        ),
        LiquidityMitigationScenario(
            name="Stress: 2008 Liquidity Shock",
            rolling_turnover_smoothing=0.50,
            stable_turnover_smoothing=0.35,
            liquidity_config=LiquidityScalingConfig(
                severe_threshold=-0.030,
                moderate_threshold=-0.015,
                severe_regime_multiplier=0.70,
                moderate_regime_multiplier=0.85,
                severe_liquidity_multiplier=0.45,
                moderate_liquidity_multiplier=0.60,
                neutral_liquidity_multiplier=0.95,
                floor=0.35,
            ),
            notes="Mimics 2008-style liquidity drag with tighter multipliers and higher turnover damping.",
        ),
        LiquidityMitigationScenario(
            name="Stress: 2020 Volatility Spike",
            rolling_turnover_smoothing=0.65,
            stable_turnover_smoothing=0.45,
            liquidity_config=LiquidityScalingConfig(
                severe_threshold=-0.020,
                moderate_threshold=-0.010,
                severe_regime_multiplier=0.80,
                moderate_regime_multiplier=0.92,
                severe_liquidity_multiplier=0.50,
                moderate_liquidity_multiplier=0.70,
                neutral_liquidity_multiplier=0.95,
                floor=0.40,
            ),
            notes="Uses elevated smoothing to reflect 2020 volatility with rapid regime swings.",
        ),
        LiquidityMitigationScenario(
            name="Stress: 2022 Rates + Stocks",
            rolling_turnover_smoothing=0.55,
            stable_turnover_smoothing=0.40,
            liquidity_config=LiquidityScalingConfig(
                severe_threshold=-0.025,
                moderate_threshold=-0.012,
                severe_regime_multiplier=0.78,
                moderate_regime_multiplier=0.90,
                severe_liquidity_multiplier=0.52,
                moderate_liquidity_multiplier=0.68,
                neutral_liquidity_multiplier=0.97,
                floor=0.38,
            ),
            notes="Captures cross-asset drawdowns observed during 2022 rate hikes.",
        ),
        LiquidityMitigationScenario(
            name="Stress: 1987 Black Monday",
            rolling_turnover_smoothing=0.75,
            stable_turnover_smoothing=0.50,
            liquidity_config=LiquidityScalingConfig(
                severe_threshold=-0.035,
                moderate_threshold=-0.018,
                severe_regime_multiplier=0.65,
                moderate_regime_multiplier=0.80,
                severe_liquidity_multiplier=0.40,
                moderate_liquidity_multiplier=0.58,
                neutral_liquidity_multiplier=0.90,
                floor=0.30,
            ),
            notes="Approximates 1987-style crash conditions with aggressive turnover damping and liquidity scaling.",
        ),
        LiquidityMitigationScenario(
            name="Stress: Synthetic Liquidity Shock",
            rolling_turnover_smoothing=defaults.rolling_turnover_smoothing,
            stable_turnover_smoothing=defaults.stable_turnover_smoothing,
            liquidity_config=LiquidityScalingConfig(
                severe_threshold=defaults.liquidity_scaling.severe_threshold * 1.5,
                moderate_threshold=defaults.liquidity_scaling.moderate_threshold * 1.5,
                severe_regime_multiplier=defaults.liquidity_scaling.severe_regime_multiplier * 0.9,
                moderate_regime_multiplier=defaults.liquidity_scaling.moderate_regime_multiplier * 0.95,
                severe_liquidity_multiplier=defaults.liquidity_scaling.severe_liquidity_multiplier * 0.85,
                moderate_liquidity_multiplier=defaults.liquidity_scaling.moderate_liquidity_multiplier * 0.9,
                neutral_liquidity_multiplier=max(defaults.liquidity_scaling.neutral_liquidity_multiplier * 0.95, 0.85),
                floor=max(defaults.liquidity_scaling.floor * 0.9, 0.30),
            ),
            notes="Synthetic shock approximating +/-3 sigma liquidity moves; retains turnover defaults.",
        ),
    ]


def get_liquidity_mitigation_scenarios(
    names: Sequence[str] | None = None,
) -> list[LiquidityMitigationScenario]:
    """Return the default scenarios optionally filtered by name."""
    scenarios = get_default_liquidity_mitigation_scenarios()
    if names is None:
        return scenarios

    lookup = {scenario.name: scenario for scenario in scenarios}
    resolved: list[LiquidityMitigationScenario] = []
    missing: list[str] = []
    for name in names:
        scenario = lookup.get(name)
        if scenario is None:
            missing.append(name)
        else:
            resolved.append(scenario)

    if missing:
        missing_str = ", ".join(sorted(missing))
        available = ", ".join(sorted(lookup.keys()))
        msg = f"Unknown liquidity scenarios: {missing_str}. Available: {available}"
        raise ValueError(msg)

    if not resolved:
        msg = "No liquidity mitigation scenarios resolved from provided names"
        raise ValueError(msg)

    return resolved


def run_liquidity_mitigation_experiments(
    dataset_path: Path,
    output_dir: Path,
    *,
    split: TrainTestSplit | None = None,
    config_overrides: dict[str, Any] | None = None,
    scenarios: Sequence[LiquidityMitigationScenario] | None = None,
    run_walk_forward: bool = False,
    walk_forward_config: WalkForwardConfig | None = None,
) -> list[LiquidityMitigationResult]:
    """
    Execute liquidity mitigation experiments across multiple scenarios.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset directory or sector returns file.
    output_dir : Path
        Directory where experiment artefacts and summary CSV are written.
    split : TrainTestSplit | None
        Optional custom train/test split. Defaults to ``define_train_test_split()``.
    config_overrides : dict[str, Any] | None
        Optional overrides for ``BacktestConfig``.
    scenarios : Sequence[LiquidityMitigationScenario] | None
        Scenarios to evaluate. Defaults to a curated set targeting turnover
        smoothing and regime scaling adjustments.
    run_walk_forward : bool, default False
        When True, run a walk-forward suite per scenario and capture summary metrics.
    walk_forward_config : WalkForwardConfig | None
        Custom walk-forward configuration to use when ``run_walk_forward`` is True.

    Returns
    -------
    list[LiquidityMitigationResult]
        Summary results for each evaluated scenario.
    """
    resolved_scenarios = list(scenarios) if scenarios is not None else get_default_liquidity_mitigation_scenarios()
    if not resolved_scenarios:
        return []

    dataset = _load_sector_dataset(dataset_path)
    resolved_split = split or define_train_test_split()
    _validate_split_against_dataset(dataset, resolved_split, BACKTEST_DEFAULTS)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_results: list[dict[str, object]] = []

    for scenario in resolved_scenarios:
        scenario_slug = _slugify_scenario_name(scenario.name)
        scenario_dir = output_dir / scenario_slug
        turnover_overrides = {
            "3d_factor_rolling": scenario.rolling_turnover_smoothing,
            "3d_factor_stable": scenario.stable_turnover_smoothing,
        }
        suite = _execute_backtest_suite(
            dataset=dataset,
            output_dir=scenario_dir,
            split=resolved_split,
            config_overrides=config_overrides,
            liquidity_config=scenario.liquidity_config,
            turnover_overrides=turnover_overrides,
        )

        rolling_metrics = suite.metrics.get("3D Factor (Rolling Betas)")
        stable_metrics = suite.metrics.get("3D Factor (Stable Betas)")
        if rolling_metrics is None or stable_metrics is None:
            msg = f"Missing factor strategy metrics for scenario '{scenario.name}'"
            raise RuntimeError(msg)

        rate_hiking_liquidity = _extract_rate_hiking_liquidity(
            suite.regime_attribution,
            "3D Factor (Rolling Betas)",
        )

        walk_forward_stats: dict[str, float | None] | None = None
        walk_forward_output_dir: Path | None = None

        if run_walk_forward:
            walk_forward_root = scenario_dir / "walk_forward"
            walk_forward_result = run_walk_forward_backtest_suite(
                dataset_path=dataset_path,
                output_dir=walk_forward_root,
                walk_forward_config=walk_forward_config,
                config_overrides=config_overrides,
                liquidity_config=scenario.liquidity_config,
                turnover_overrides=turnover_overrides,
            )
            summary_frame = walk_forward_result.summarize_metrics()
            rolling_summary = summary_frame.filter(pl.col("strategy") == "3D Factor (Rolling Betas)")
            if not rolling_summary.is_empty():
                row = rolling_summary.row(0, named=True)

                def _extract_metric(column: str) -> float | None:
                    raw_value = row.get(column)
                    if raw_value is None:
                        return None
                    try:
                        numeric = float(raw_value)
                    except (TypeError, ValueError):
                        return None
                    if math.isnan(numeric):
                        return None
                    return numeric

                walk_forward_stats = {
                    "mean": _extract_metric("sharpe_ratio_mean"),
                    "std": _extract_metric("sharpe_ratio_std"),
                    "min": _extract_metric("sharpe_ratio_min"),
                    "max": _extract_metric("sharpe_ratio_max"),
                }
                if all(value is None for value in walk_forward_stats.values()):
                    walk_forward_stats = None
            walk_forward_output_dir = walk_forward_root / "walk_forward"

        raw_results.append({
            "scenario": scenario,
            "rolling_metrics": rolling_metrics,
            "stable_metrics": stable_metrics,
            "rate_hiking_liquidity": rate_hiking_liquidity,
            "output_dir": scenario_dir,
            "walk_forward_stats": walk_forward_stats,
            "walk_forward_output_dir": walk_forward_output_dir,
        })

    baseline = raw_results[0]
    baseline_rolling = cast(PerformanceMetrics, baseline["rolling_metrics"])
    baseline_stable = cast(PerformanceMetrics, baseline["stable_metrics"])
    baseline_rate_liquidity = cast(float | None, baseline.get("rate_hiking_liquidity"))

    results: list[LiquidityMitigationResult] = []

    for entry in raw_results:
        scenario = cast(LiquidityMitigationScenario, entry["scenario"])
        rolling_metrics = cast(PerformanceMetrics, entry["rolling_metrics"])
        stable_metrics = cast(PerformanceMetrics, entry["stable_metrics"])
        rate_hiking_liquidity = cast(float | None, entry["rate_hiking_liquidity"])
        scenario_dir = cast(Path, entry["output_dir"])
        walk_forward_stats_entry = cast(dict[str, float | None] | None, entry.get("walk_forward_stats"))
        walk_forward_output_dir_entry = cast(Path | None, entry.get("walk_forward_output_dir"))

        rolling_sharpe_delta = rolling_metrics.sharpe_ratio - baseline_rolling.sharpe_ratio
        rolling_transaction_costs_delta = (
            rolling_metrics.transaction_costs_total - baseline_rolling.transaction_costs_total
        )
        rolling_rate_liquidity_value: float | None = None
        rolling_liquidity_delta: float | None = None
        if rate_hiking_liquidity is not None and baseline_rate_liquidity is not None:
            rolling_rate_liquidity_value = float(rate_hiking_liquidity)
            rolling_liquidity_delta = rolling_rate_liquidity_value - float(baseline_rate_liquidity)
        elif rate_hiking_liquidity is not None:
            rolling_rate_liquidity_value = float(rate_hiking_liquidity)

        stable_sharpe_delta = stable_metrics.sharpe_ratio - baseline_stable.sharpe_ratio
        stable_transaction_costs_delta = (
            stable_metrics.transaction_costs_total - baseline_stable.transaction_costs_total
        )

        walk_forward_mean: float | None = None
        walk_forward_std: float | None = None
        walk_forward_min: float | None = None
        walk_forward_max: float | None = None
        if walk_forward_stats_entry is not None:
            walk_forward_mean = walk_forward_stats_entry.get("mean")
            walk_forward_std = walk_forward_stats_entry.get("std")
            walk_forward_min = walk_forward_stats_entry.get("min")
            walk_forward_max = walk_forward_stats_entry.get("max")

        result = LiquidityMitigationResult(
            scenario_name=scenario.name,
            notes=scenario.notes,
            rolling_turnover_smoothing=scenario.rolling_turnover_smoothing,
            stable_turnover_smoothing=scenario.stable_turnover_smoothing,
            severe_threshold=scenario.liquidity_config.severe_threshold,
            moderate_threshold=scenario.liquidity_config.moderate_threshold,
            severe_regime_multiplier=scenario.liquidity_config.severe_regime_multiplier,
            moderate_regime_multiplier=scenario.liquidity_config.moderate_regime_multiplier,
            severe_liquidity_multiplier=scenario.liquidity_config.severe_liquidity_multiplier,
            moderate_liquidity_multiplier=scenario.liquidity_config.moderate_liquidity_multiplier,
            rolling_sharpe=rolling_metrics.sharpe_ratio,
            rolling_sharpe_delta=rolling_sharpe_delta,
            rolling_turnover=rolling_metrics.turnover_rate,
            rolling_transaction_costs=rolling_metrics.transaction_costs_total,
            rolling_transaction_costs_delta=rolling_transaction_costs_delta,
            rolling_rate_hiking_liquidity=rolling_rate_liquidity_value,
            rolling_rate_hiking_liquidity_delta=rolling_liquidity_delta,
            stable_sharpe=stable_metrics.sharpe_ratio,
            stable_sharpe_delta=stable_sharpe_delta,
            stable_turnover=stable_metrics.turnover_rate,
            stable_transaction_costs=stable_metrics.transaction_costs_total,
            stable_transaction_costs_delta=stable_transaction_costs_delta,
            output_directory=scenario_dir,
            walk_forward_sharpe_mean=walk_forward_mean,
            walk_forward_sharpe_std=walk_forward_std,
            walk_forward_sharpe_min=walk_forward_min,
            walk_forward_sharpe_max=walk_forward_max,
            walk_forward_output_directory=walk_forward_output_dir_entry,
        )
        results.append(result)

    summary_frame = pl.DataFrame([result.as_dict() for result in results])
    summary_path = output_dir / "liquidity_mitigation_results.csv"
    summary_frame.write_csv(summary_path)
    LOGGER.info("Liquidity mitigation experiment summary saved", path=str(summary_path))

    return results


def _load_sector_dataset(dataset_path: Path) -> SectorDataset:
    """Load sector returns, factor returns, and coverage metadata."""
    try:
        from playground.risk_model.dataset import CoverageSummary
        from playground.risk_model.dataset import SectorDataset
    except (ImportError, IndentationError):
        LOGGER.warning(
            "Falling back to simplified SectorDataset due to import error",
            exc_info=True,
        )

        @dataclass(slots=True, frozen=True)
        class CoverageSummary:  # type: ignore[no-redef]
            calendar_name: str
            sector_expected_days: int
            factor_expected_days: int
            sector_coverage: Mapping[str, float]
            factor_coverage: Mapping[str, float]

        @dataclass(slots=True, frozen=True)
        class SectorDataset:  # type: ignore[no-redef]
            sector_returns: pl.DataFrame
            factor_returns: pl.DataFrame
            coverage: CoverageSummary

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
    ComputeWeightsMethod = Callable[
        [str, datetime, SectorDataset, list[str], dict[str, object]],
        dict[str, float],
    ]
    original_compute_weights = cast(
        ComputeWeightsMethod,
        backtester._compute_target_weights,
    )

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

    patched_compute_weights = cast(
        ComputeWeightsMethod,
        MethodType(custom_compute_weights, backtester),
    )
    setattr(backtester, "_compute_target_weights", patched_compute_weights)

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
        setattr(backtester, "_compute_target_weights", original_compute_weights)


def _build_regime_resolver(
    regimes: Sequence[MarketRegime],
) -> Callable[[datetime], str | None]:
    """Return a callable that resolves the active regime for a given timestamp."""
    def resolver(date: datetime) -> str | None:
        comparison_date = date.astimezone(UTC) if date.tzinfo is not None else date.replace(tzinfo=UTC)
        for regime in regimes:
            if regime.start <= comparison_date <= regime.end:
                return str(regime.name)
        return None

    return resolver


# ===== Public API =====

__all__ = [
    "PHASE3_TARGET_SHARPE",
    "THREE_D_ROLLING_STRATEGY",
    "BacktestSuite",
    "DiagnosticsResult",
    "LiquidityMitigationResult",
    "LiquidityMitigationScenario",
    "MonitoringSnapshotResult",
    "MonteCarloStressSuiteResult",
    "ParameterHeatmapSuiteResult",
    "ProxyDatasetSuiteResult",
    "VintageSimulationSuiteResult",
    "WalkForwardBacktestResult",
    "export_phase3_monitoring_snapshot",
    "get_default_liquidity_mitigation_scenarios",
    "get_liquidity_mitigation_scenarios",
    "run_extended_diagnostics",
    "run_full_backtest_suite",
    "run_liquidity_mitigation_experiments",
    "run_monte_carlo_stress_suite",
    "run_parameter_heatmap_suite",
    "run_proxy_dataset_validation",
    "run_vintage_simulation_suite",
    "run_walk_forward_backtest_suite",
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
