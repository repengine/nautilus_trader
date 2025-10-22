"""
Utility to regenerate Phase 3.3 backtest visuals from saved CSV exports.

The script mirrors the exploratory notebook in
``playground/reports/backtesting/visuals/phase3_attribution.ipynb`` and produces
deterministic PNG artefacts for CI consumption.

Example:
    >>> export_phase3_visuals()
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog


matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config.playground import ThreeDRiskBacktestDefaults  # noqa: E402
from playground.backtest.monitoring import log_walk_forward_metadata  # noqa: E402
from playground.backtest.liquidity_controls import LiquidityScalingConfig  # noqa: E402


DEFAULT_DATA_DIR = Path("playground/reports/backtesting")
PLAYGROUND_DEFAULTS = ThreeDRiskBacktestDefaults()
LOGGER = structlog.get_logger(__name__)


def export_phase3_visuals(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    output_dir: Path | None = None,
    config: LiquidityScalingConfig | None = None,
) -> None:
    """
    Generate Phase 3 attribution visuals from CSV artefacts.

    Args:
        data_dir: Directory containing the exported backtest CSV files.
        output_dir: Destination directory for the generated figures. Defaults to
            ``data_dir / 'visuals'``.
        config: Liquidity scaling configuration used to annotate stress thresholds.
            Defaults to ``ThreeDRiskBacktestDefaults.build_liquidity_config()``.
    """
    visuals_path = output_dir or (data_dir / "visuals")
    visuals_path.mkdir(parents=True, exist_ok=True)

    performance = pl.read_csv(data_dir / "performance_comparison_table.csv")
    train_vs_test = pl.read_csv(data_dir / "train_vs_test_metrics.csv")
    regime_summary = pl.read_csv(data_dir / "regime_summary.csv")
    regime_attr = pl.read_csv(
        data_dir / "attribution" / "regime" / "3d_factor_rolling_betas_regime_attribution.csv",
        schema_overrides={"annualized_contribution": pl.Float64},
        ignore_errors=True,
    )
    rolling_attr = pl.read_csv(data_dir / "attribution" / "3d_factor_rolling_betas_attribution.csv")
    sixty_forty_attr = pl.read_csv(data_dir / "attribution" / "60_40_portfolio_attribution.csv")

    _plot_sharpe_vs_transaction_costs(performance, visuals_path / "sharpe_vs_tc.png")
    _plot_sharpe_benchmark_lines(train_vs_test, visuals_path / "rolling_vs_benchmark_sharpe.png")
    _plot_regime_contribution_bars(regime_attr, visuals_path / "regime_contributions.png")
    resolved_config = config or PLAYGROUND_DEFAULTS.build_liquidity_config()
    _plot_liquidity_stress_panel(
        regime_attr,
        regime_summary,
        resolved_config,
        visuals_path / "liquidity_stress_panel.png",
    )
    _plot_attribution_waterfall(
        performance,
        rolling_attr,
        sixty_forty_attr,
        visuals_path / "attribution_waterfall.png",
    )
    aggregate_path = data_dir / "walk_forward" / "aggregate_metrics.csv"
    if aggregate_path.exists():
        aggregate = pl.read_csv(aggregate_path)
        _plot_walk_forward_sharpe_boxplot(
            aggregate,
            visuals_path / "walk_forward_sharpe_boxplot.png",
        )
    metadata_path = data_dir / "walk_forward" / "metadata.json"
    _write_walk_forward_metadata_summary(data_dir, visuals_path)
    log_walk_forward_metadata(metadata_path)


def _plot_sharpe_vs_transaction_costs(frame: pl.DataFrame, output_path: Path) -> None:
    """Scatter Sharpe ratio against transaction costs for factor vs benchmark."""
    factor_rows = frame.filter(pl.col("strategy").str.contains("3D Factor"))
    benchmark_rows = frame.filter(pl.col("strategy") == "60/40 Portfolio")
    comparison = pl.concat([factor_rows, benchmark_rows], how="vertical")
    subset = comparison.select(["strategy", "sharpe_ratio", "transaction_costs"])
    strategies = subset.get_column("strategy").to_list()
    sharpe = subset.get_column("sharpe_ratio").to_list()
    transaction_costs = subset.get_column("transaction_costs").to_list()

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#1f77b4" if "3D" in name else "#ff7f0e" for name in strategies]
    ax.scatter(transaction_costs, sharpe, c=colours)
    for name, x_val, y_val in zip(strategies, transaction_costs, sharpe):
        ax.annotate(name, (x_val, y_val), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Transaction Costs ($)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe vs Transaction Costs (Test Period)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_sharpe_benchmark_lines(frame: pl.DataFrame, output_path: Path) -> None:
    """Line chart of Sharpe ratios for Rolling/Stable vs 60/40 and Equal Weight."""
    strategies = [
        "3D Factor (Rolling Betas)",
        "3D Factor (Stable Betas)",
        "60/40 Portfolio",
        "Equal Weight",
    ]
    period_order = {"train": 0, "test": 1}
    periods = ["train", "test"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Sharpe Ratio by Period")
    ax.set_xlabel("Period")
    ax.set_ylabel("Sharpe Ratio")

    for strategy in strategies:
        strategy_rows = (
            frame.filter(pl.col("strategy") == strategy)
            .with_columns(pl.col("period").replace(period_order).alias("_order"))
            .sort("_order")
        )
        if strategy_rows.is_empty():
            continue
        sharpe_values = strategy_rows.get_column("sharpe_ratio").to_list()
        x_positions = [period_order[p] for p in strategy_rows.get_column("period").to_list()]
        label = strategy.replace("3D Factor (", "3D ").replace(" Betas)", "")
        ax.plot(x_positions, sharpe_values, marker="o", label=label)

    ax.set_xticks([period_order[p] for p in periods], [p.title() for p in periods])

    rolling = (
        frame.filter(pl.col("strategy") == "3D Factor (Rolling Betas)")
        .with_columns(pl.col("period").replace(period_order).alias("_order"))
        .sort("_order")
    )
    sixty_forty = (
        frame.filter(pl.col("strategy") == "60/40 Portfolio")
        .with_columns(pl.col("period").replace(period_order).alias("_order"))
        .sort("_order")
    )
    if not rolling.is_empty() and rolling.height == sixty_forty.height:
        for period, roll_val, bench_val in zip(
            rolling.get_column("period").to_list(),
            rolling.get_column("sharpe_ratio").to_list(),
            sixty_forty.get_column("sharpe_ratio").to_list(),
        ):
            if roll_val > bench_val:
                x_coord = period_order[period]
                ax.annotate(
                    "Rolling > 60/40",
                    (x_coord, roll_val),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=9,
                )

    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_regime_contribution_bars(frame: pl.DataFrame, output_path: Path) -> None:
    """Stacked bar chart of regime-level factor contributions."""
    if frame.is_empty():
        return

    pivoted = (
        frame.filter(pl.col("factor").is_in(["factor_credit", "factor_duration", "factor_liquidity", "alpha"]))
        .pivot(index="regime", on="factor", values="annualized_contribution", aggregate_function="sum")
        .fill_null(0.0)
        .sort("regime")
    )
    regimes = pivoted.get_column("regime").to_list()
    factors: Sequence[str] = [col for col in pivoted.columns if col != "regime"]
    bottom = np.zeros(len(regimes))

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = {
        "factor_credit": "#1f77b4",
        "factor_duration": "#ff7f0e",
        "factor_liquidity": "#2ca02c",
        "alpha": "#9467bd",
    }

    for factor in factors:
        values = pivoted.get_column(factor).to_numpy() * 100.0
        if factor == "factor_liquidity":
            bar_colours = ["#d62728" if value < 0 else colours[factor] for value in values]
        else:
            fill_colour = colours.get(factor, "#7f7f7f")
            bar_colours = [fill_colour] * len(values)
        ax.bar(regimes, values, bottom=bottom, color=bar_colours, label=_format_factor_label(factor))
        bottom = bottom + values

    ax.set_ylabel("Annualized Contribution (%)")
    ax.set_title("Regime Factor Contributions - 3D Rolling Betas")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_liquidity_stress_panel(
    regime_attr: pl.DataFrame,
    regime_summary: pl.DataFrame,
    config: LiquidityScalingConfig,
    output_path: Path,
) -> None:
    """Scatter liquidity attribution vs regime Sharpe with scaling thresholds."""
    liquidity_rows = regime_attr.filter(pl.col("factor") == "factor_liquidity")
    if liquidity_rows.is_empty():
        return

    liquidity = (
        liquidity_rows.select(["regime", "annualized_contribution"])
        .rename({"regime": "regime_name"})
        .with_columns((pl.col("annualized_contribution") * 100.0).alias("liquidity_pct"))
    )
    rolling_summary = regime_summary.filter(pl.col("strategy") == "3D Factor (Rolling Betas)")
    merged = liquidity.join(rolling_summary, on="regime_name", how="inner")
    if merged.is_empty():
        return

    regimes = merged.get_column("regime_name").to_list()
    liquidity_pct = merged.get_column("liquidity_pct").to_list()
    sharpe = merged.get_column("sharpe_ratio").to_list()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(liquidity_pct, sharpe, color="#1f77b4")
    for regime, x_val, y_val in zip(regimes, liquidity_pct, sharpe):
        ax.annotate(regime, (x_val, y_val), textcoords="offset points", xytext=(5, 5))

    moderate = config.moderate_threshold * 100.0
    severe = config.severe_threshold * 100.0
    ax.axvline(moderate, color="#ff7f0e", linestyle="--", linewidth=1.5, label=f"Moderate ({moderate:.1f}%)")
    ax.axvline(severe, color="#d62728", linestyle="--", linewidth=1.5, label=f"Severe ({severe:.1f}%)")

    ax.set_xlabel("Liquidity Contribution (%)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Liquidity Stress vs Regime Sharpe - 3D Rolling Betas")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_attribution_waterfall(
    performance: pl.DataFrame,
    rolling_attr: pl.DataFrame,
    sixty_forty_attr: pl.DataFrame,
    output_path: Path,
) -> None:
    """Waterfall chart from benchmark to rolling betas return."""
    benchmark_row = performance.filter(pl.col("strategy") == "60/40 Portfolio")
    rolling_row = performance.filter(pl.col("strategy") == "3D Factor (Rolling Betas)")
    if benchmark_row.is_empty() or rolling_row.is_empty():
        return

    benchmark_return = float(benchmark_row.get_column("annualized_return")[0])
    rolling_return = float(rolling_row.get_column("annualized_return")[0])

    deltas = _compute_attribution_deltas(rolling_attr, sixty_forty_attr)
    residual = rolling_return - benchmark_return - sum(value for _, value in deltas)

    labels: list[str] = ["60/40 Portfolio"] + [label for label, _ in deltas]
    values: list[float] = [benchmark_return] + [value for _, value in deltas]
    if abs(residual) > 1e-3:
        labels.append("Residual")
        values.append(residual)
    labels.append("3D Factor (Rolling Betas)")
    values.append(rolling_return)

    bottoms = _compute_waterfall_bottoms(values)

    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (label, value, bottom) in enumerate(zip(labels, values, bottoms)):
        if index == 0 or index == len(labels) - 1:
            color = "#1f77b4" if index == 0 else "#2ca02c"
        else:
            color = "#2ca02c" if value >= 0 else "#d62728"
        ax.bar(index, value, bottom=bottom, color=color)
        ax.text(index, bottom + value / 2, f"{value:+.2f}%", ha="center", va="center", color="white", fontsize=9)

    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Attribution Waterfall - 60/40 to 3D Rolling Betas")
    ax.axhline(0.0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_walk_forward_sharpe_boxplot(
    aggregate: pl.DataFrame,
    output_path: Path,
) -> None:
    """Boxplot summarising walk-forward Sharpe dispersion."""
    if aggregate.is_empty() or "strategy" not in aggregate.columns:
        return

    data: list[list[float]] = []
    labels: list[str] = []
    strategy_series = aggregate.get_column("strategy").unique()
    strategy_series = strategy_series.sort()

    for strategy in strategy_series.to_list():
        term = str(strategy)
        sharpe_values = (
            aggregate.filter(pl.col("strategy") == strategy)
            .select(pl.col("sharpe_ratio"))
            .drop_nulls()
            .get_column("sharpe_ratio")
            .to_list()
        )
        values = [float(value) for value in sharpe_values if value is not None]
        if not values:
            continue
        label = term.replace("3D Factor (", "3D ").replace(" Betas)", "")
        data.append(values)
        labels.append(label)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    box_art = ax.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=True,
    )

    for patch in box_art["boxes"]:
        patch.set(facecolor="#1f77b4", alpha=0.18)
    for median in box_art["medians"]:
        median.set(color="#2ca02c", linewidth=1.5)
    for mean_line in box_art["means"]:
        mean_line.set(color="#d62728", linewidth=1.5)

    ax.set_title("Walk-Forward Sharpe Distribution (Test Folds)")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_walk_forward_metadata_summary(data_dir: Path, output_dir: Path) -> None:
    """Persist a metadata summary derived from walk-forward configuration."""
    metadata_path = data_dir / "walk_forward" / "metadata.json"
    if not metadata_path.exists():
        return
    summary_path = output_dir / "walk_forward_metadata.txt"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        summary_path.write_text("Unable to parse walk-forward metadata.\n", encoding="utf-8")
        return

    lines: list[str] = []
    risk_free = metadata.get("risk_free_rate")
    if isinstance(risk_free, (int, float)):
        lines.append(f"Risk-free rate: {risk_free:.4f}")
    turnover = metadata.get("turnover_smoothing")
    if isinstance(turnover, dict):
        stable = turnover.get("stable")
        rolling = turnover.get("rolling")
        lines.append(f"Turnover smoothing (stable/rolling): {stable} / {rolling}")
    liquidity_config = metadata.get("liquidity_config")
    if isinstance(liquidity_config, dict):
        severe = liquidity_config.get("severe_threshold")
        moderate = liquidity_config.get("moderate_threshold")
        lines.append(
            "Liquidity thresholds (severe/moderate): "
            f"{severe} / {moderate}"
        )
        severe_mult = liquidity_config.get("severe_liquidity_multiplier")
        moderate_mult = liquidity_config.get("moderate_liquidity_multiplier")
        neutral_mult = liquidity_config.get("neutral_liquidity_multiplier")
        lines.append(
            "Liquidity multipliers (severe/moderate/neutral): "
            f"{severe_mult} / {moderate_mult} / {neutral_mult}"
        )
    overrides = metadata.get("turnover_overrides")
    if overrides:
        lines.append(f"Turnover overrides: {overrides}")
    baseline_strategies = metadata.get("baseline_strategies")
    if isinstance(baseline_strategies, list):
        lines.append(f"Baseline strategies: {', '.join(baseline_strategies)}")

    if not lines:
        lines.append("Walk-forward metadata available but no recognised fields to summarise.")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info(
        "Walk-forward metadata summary written",
        summary_path=str(summary_path),
    )


def _compute_attribution_deltas(
    rolling: pl.DataFrame,
    benchmark: pl.DataFrame,
) -> list[tuple[str, float]]:
    relevant = rolling.join(benchmark, on="factor", suffix="_benchmark")
    deltas: list[tuple[str, float]] = []
    for row in relevant.iter_rows(named=True):
        if row["factor"] == "alpha":
            label = "Alpha"
        else:
            label = _format_factor_label(str(row["factor"]))
        delta = float((row["annualized_contribution"] - row["annualized_contribution_benchmark"]) * 100.0)
        deltas.append((label, delta))
    return deltas


def _format_factor_label(name: str) -> str:
    return name.replace("factor_", "").replace("_", " ").title()


def _compute_waterfall_bottoms(values: Sequence[float]) -> list[float]:
    bottoms: list[float] = []
    running_total = 0.0
    for index, value in enumerate(values):
        if index == 0:
            bottoms.append(0.0)
            running_total = value
            continue
        if index == len(values) - 1:
            bottoms.append(0.0)
            continue
        if value >= 0:
            bottoms.append(running_total)
        else:
            bottoms.append(running_total + value)
        running_total += value
    return bottoms


if __name__ == "__main__":
    export_phase3_visuals()
