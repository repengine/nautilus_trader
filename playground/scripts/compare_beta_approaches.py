#!/usr/bin/env python3
"""
Compare stable vs rolling beta approaches for sector risk modeling.

This script loads sector and factor data, computes both stable and rolling betas,
runs stability analysis, and generates a comprehensive comparison report.

Output:
- Console: Rich formatted comparison table
- JSON: Detailed results saved to file
- Markdown: Human-readable report
"""

from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

from playground.risk_model.rolling_beta import compute_beta_stability_analysis
from playground.risk_model.rolling_beta import compute_rolling_betas


def main() -> None:
    """Run beta approach comparison analysis."""
    console = Console()

    console.print("\n[bold cyan]Rolling vs Stable Beta Comparison[/bold cyan]\n")

    # Paths
    data_dir = Path("/home/nate/projects/nautilus_trader/playground/data/sector_dataset")
    sector_returns_path = data_dir / "sector_returns.parquet"
    factor_returns_path = data_dir / "factor_returns.parquet"
    output_dir = Path("/home/nate/projects/nautilus_trader/playground/docs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    console.print("[yellow]Loading data...[/yellow]")

    if not sector_returns_path.exists():
        console.print(f"[red]Error: Sector returns file not found at {sector_returns_path}[/red]")
        return

    if not factor_returns_path.exists():
        console.print(f"[red]Error: Factor returns file not found at {factor_returns_path}[/red]")
        return

    sector_returns = pl.read_parquet(sector_returns_path)
    factor_returns = pl.read_parquet(factor_returns_path)

    console.print(f"  Loaded {len(sector_returns)} sector return observations")
    console.print(f"  Loaded {len(factor_returns)} factor return observations")
    console.print(f"  Sectors: {sector_returns['symbol'].n_unique()}")

    # Define factor columns
    factor_columns = ["factor_duration", "factor_credit", "factor_liquidity"]

    # Compute rolling betas
    console.print("\n[yellow]Computing rolling betas (252-day window)...[/yellow]")
    rolling_results = compute_rolling_betas(
        sector_returns,
        factor_returns,
        factor_columns=factor_columns,
        window_days=252,  # 1 year
        min_observations=126,  # 6 months minimum
    )

    console.print(f"  Computed rolling betas for {len(rolling_results)} sectors")

    # Set test period start (last 20% of data for out-of-sample testing)
    all_timestamps = sector_returns["timestamp"].to_list()
    all_timestamps_sorted = sorted(all_timestamps)
    test_start_idx = int(len(all_timestamps_sorted) * 0.8)
    test_period_start = all_timestamps_sorted[test_start_idx]

    console.print(f"\n[yellow]Train/Test split at: {test_period_start.isoformat()}[/yellow]")

    # Compute stability analysis
    console.print("\n[yellow]Computing beta stability analysis...[/yellow]")
    stability_results = compute_beta_stability_analysis(
        sector_returns,
        factor_returns,
        rolling_results,
        factor_columns=factor_columns,
        test_period_start=test_period_start,
    )

    console.print(f"  Computed stability analysis for {len(stability_results)} sectors")

    # Create comparison table
    table = Table(title="Beta Approach Comparison Results")
    table.add_column("Sector", style="cyan", no_wrap=True)
    table.add_column("β_dur (Stable)", justify="right", style="green")
    table.add_column("β_dur CV", justify="right")
    table.add_column("β_cred CV", justify="right")
    table.add_column("β_liq CV", justify="right")
    table.add_column("Stable R²", justify="right", style="blue")
    table.add_column("Rolling R²", justify="right", style="blue")
    table.add_column("Recommendation", style="bold")

    for sector_id, analysis in sorted(stability_results.items()):
        # Color code recommendation
        rec_style = "green" if analysis.recommended_approach == "stable" else "yellow"

        table.add_row(
            sector_id,
            f"{analysis.stable_beta_duration:.3f}",
            f"{analysis.beta_duration_cv:.2f}",
            f"{analysis.beta_credit_cv:.2f}",
            f"{analysis.beta_liquidity_cv:.2f}",
            f"{analysis.stable_forecast_r2:.3f}",
            f"{analysis.rolling_forecast_r2:.3f}",
            f"[{rec_style}]{analysis.recommended_approach.upper()}[/{rec_style}]",
        )

    console.print("\n")
    console.print(table)

    # Print detailed recommendations
    console.print("\n[bold cyan]Detailed Recommendations[/bold cyan]\n")

    stable_count = sum(
        1 for a in stability_results.values() if a.recommended_approach == "stable"
    )
    rolling_count = len(stability_results) - stable_count

    console.print(f"  [green]Stable approach recommended:[/green] {stable_count} sectors")
    console.print(f"  [yellow]Rolling approach recommended:[/yellow] {rolling_count} sectors")

    for sector_id, analysis in sorted(stability_results.items()):
        color = "green" if analysis.recommended_approach == "stable" else "yellow"
        console.print(f"\n  [{color}]{sector_id}[/{color}]: {analysis.rationale}")

    # Calculate summary statistics
    avg_stable_r2 = sum(a.stable_forecast_r2 for a in stability_results.values()) / len(
        stability_results
    )
    avg_rolling_r2 = sum(a.rolling_forecast_r2 for a in stability_results.values()) / len(
        stability_results
    )
    avg_cv_duration = sum(a.beta_duration_cv for a in stability_results.values()) / len(
        stability_results
    )

    console.print("\n[bold cyan]Summary Statistics[/bold cyan]\n")
    console.print(f"  Average Stable R² (out-of-sample): {avg_stable_r2:.4f}")
    console.print(f"  Average Rolling R² (out-of-sample): {avg_rolling_r2:.4f}")
    console.print(f"  Average Duration Beta CV: {avg_cv_duration:.4f}")

    # Save results to JSON
    json_output = {
        "metadata": {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "test_period_start": test_period_start.isoformat(),
            "n_sectors": len(stability_results),
            "window_days": 252,
            "min_observations": 126,
        },
        "summary": {
            "stable_recommended_count": stable_count,
            "rolling_recommended_count": rolling_count,
            "avg_stable_r2": avg_stable_r2,
            "avg_rolling_r2": avg_rolling_r2,
            "avg_cv_duration": avg_cv_duration,
        },
        "sectors": {
            sector_id: {
                "stable_beta_duration": analysis.stable_beta_duration,
                "stable_beta_credit": analysis.stable_beta_credit,
                "stable_beta_liquidity": analysis.stable_beta_liquidity,
                "stable_r_squared": analysis.stable_r_squared,
                "rolling_beta_mean_duration": analysis.rolling_beta_mean_duration,
                "rolling_beta_std_duration": analysis.rolling_beta_std_duration,
                "rolling_beta_mean_credit": analysis.rolling_beta_mean_credit,
                "rolling_beta_std_credit": analysis.rolling_beta_std_credit,
                "rolling_beta_mean_liquidity": analysis.rolling_beta_mean_liquidity,
                "rolling_beta_std_liquidity": analysis.rolling_beta_std_liquidity,
                "beta_duration_cv": analysis.beta_duration_cv,
                "beta_credit_cv": analysis.beta_credit_cv,
                "beta_liquidity_cv": analysis.beta_liquidity_cv,
                "stable_forecast_r2": analysis.stable_forecast_r2,
                "rolling_forecast_r2": analysis.rolling_forecast_r2,
                "recommended_approach": analysis.recommended_approach,
                "rationale": analysis.rationale,
            }
            for sector_id, analysis in stability_results.items()
        },
    }

    json_path = output_dir / "beta_comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)

    console.print(f"\n[green]Results saved to:[/green] {json_path}")

    # Generate Markdown report
    md_path = output_dir / "beta_comparison_report.md"
    with open(md_path, "w") as f:
        f.write("# Rolling vs Stable Beta Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now(tz=UTC).isoformat()}\n\n")
        f.write(f"**Test Period Start:** {test_period_start.isoformat()}\n\n")
        f.write(f"**Number of Sectors:** {len(stability_results)}\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(f"- **Average Stable R² (out-of-sample):** {avg_stable_r2:.4f}\n")
        f.write(f"- **Average Rolling R² (out-of-sample):** {avg_rolling_r2:.4f}\n")
        f.write(f"- **Average Duration Beta CV:** {avg_cv_duration:.4f}\n")
        f.write(f"- **Stable Approach Recommended:** {stable_count} sectors\n")
        f.write(f"- **Rolling Approach Recommended:** {rolling_count} sectors\n\n")

        f.write("## Overall Recommendation\n\n")
        if stable_count > rolling_count:
            f.write(
                "**Recommendation: Use STABLE betas for most sectors.**\n\n"
                f"The majority ({stable_count}/{len(stability_results)}) of sectors show "
                "stable betas with comparable or better forecast accuracy using the full-sample approach.\n\n"
            )
        else:
            f.write(
                "**Recommendation: Use ROLLING betas for most sectors.**\n\n"
                f"The majority ({rolling_count}/{len(stability_results)}) of sectors show "
                "time-varying betas with better forecast accuracy using rolling windows.\n\n"
            )

        f.write("## Sector-Level Results\n\n")
        f.write("| Sector | β_dur (Stable) | β_dur CV | β_cred CV | β_liq CV | Stable R² | Rolling R² | Recommendation |\n")
        f.write("|--------|----------------|----------|-----------|----------|-----------|------------|----------------|\n")

        for sector_id, analysis in sorted(stability_results.items()):
            f.write(
                f"| {sector_id} | "
                f"{analysis.stable_beta_duration:.3f} | "
                f"{analysis.beta_duration_cv:.2f} | "
                f"{analysis.beta_credit_cv:.2f} | "
                f"{analysis.beta_liquidity_cv:.2f} | "
                f"{analysis.stable_forecast_r2:.3f} | "
                f"{analysis.rolling_forecast_r2:.3f} | "
                f"**{analysis.recommended_approach.upper()}** |\n"
            )

        f.write("\n## Detailed Rationale\n\n")
        for sector_id, analysis in sorted(stability_results.items()):
            f.write(f"### {sector_id}\n\n")
            f.write(f"**Recommendation:** {analysis.recommended_approach.upper()}\n\n")
            f.write(f"{analysis.rationale}\n\n")

    console.print(f"[green]Markdown report saved to:[/green] {md_path}")

    console.print("\n[bold green]Analysis complete![/bold green]\n")


if __name__ == "__main__":
    main()
