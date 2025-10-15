#!/usr/bin/env python3
"""
Generate comprehensive regression diagnostics report for the 3D Risk Model.

This script:
1. Loads sector returns and factor data
2. Computes regression diagnostics for each sector
3. Creates summary statistics and acceptance criteria validation
4. Generates diagnostic plots
5. Exports results to JSON and markdown

Usage:
    python playground/scripts/generate_diagnostics_report.py --start-date 2010-01-01 --end-date 2024-12-31
"""

from __future__ import annotations

import json
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import structlog
from rich.console import Console
from rich.table import Table


if TYPE_CHECKING:
    from playground.risk_model.diagnostics import SectorDiagnosticsReport

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from playground.exposure.factor_exposure import prepare_factor_returns
from playground.risk_model.diagnostics import compute_regression_diagnostics
from playground.risk_model.diagnostics import create_diagnostics_summary
from playground.risk_model.fetchers import fetch_factor_data
from playground.risk_model.fetchers import fetch_sector_prices


LOGGER = structlog.get_logger(__name__)


def load_sector_and_factor_data(
    start_date: str,
    end_date: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load sector returns and factor data for the specified period.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Sector returns and factor returns DataFrames.
    """
    LOGGER.info("Loading data", start_date=start_date, end_date=end_date)

    # Fetch sector prices and compute returns
    sector_prices = fetch_sector_prices(start_date=start_date, end_date=end_date)

    # Compute daily returns
    sector_returns = (
        sector_prices.sort(["symbol", "timestamp"])
        .with_columns(
            pl.col("close").pct_change().over("symbol").alias("return")
        )
        .drop_nulls(subset=["return"])
        .select(["timestamp", "symbol", "return"])
    )

    LOGGER.info(
        "Loaded sector returns",
        n_sectors=sector_returns["symbol"].n_unique(),
        n_observations=len(sector_returns),
    )

    # Fetch factor data
    factor_data = fetch_factor_data(start_date=start_date, end_date=end_date)

    # Compute factor returns
    factor_columns = ["factor_duration", "factor_credit", "factor_liquidity"]
    factor_returns = prepare_factor_returns(
        factor_data,
        columns=factor_columns,
        method="difference",
        winsorize_percentile=0.99,
    )

    LOGGER.info(
        "Computed factor returns",
        n_observations=len(factor_returns),
    )

    return sector_returns, factor_returns


def print_summary_table(report: SectorDiagnosticsReport) -> None:
    """
    Print summary table to console.

    Parameters
    ----------
    report : SectorDiagnosticsReport
        Diagnostics report to display.
    """
    console = Console()

    # Print summary statistics
    console.print("\n[bold cyan]Summary Statistics[/bold cyan]\n")

    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right", style="green")

    stats = report.summary_stats
    stats_table.add_row("Mean R²", f"{stats['mean_r_squared']:.4f}")
    stats_table.add_row("Median R²", f"{stats['median_r_squared']:.4f}")
    stats_table.add_row("Std R²", f"{stats['std_r_squared']:.4f}")
    stats_table.add_row("Min R²", f"{stats['min_r_squared']:.4f}")
    stats_table.add_row("Max R²", f"{stats['max_r_squared']:.4f}")
    stats_table.add_row("% R² > 0.30", f"{stats['pct_r2_above_threshold']:.1f}%")
    stats_table.add_row("% with 2/3 Sig Betas", f"{stats['pct_2_3_sig_betas']:.1f}%")
    stats_table.add_row("Mean VIF", f"{stats['mean_vif']:.2f}")
    stats_table.add_row("Max VIF", f"{stats['max_vif']:.2f}")
    stats_table.add_row("Mean Durbin-Watson", f"{stats['mean_durbin_watson']:.2f}")
    stats_table.add_row("% DW Acceptable", f"{stats['pct_dw_acceptable']:.1f}%")
    stats_table.add_row("% Significant F-stat", f"{stats['pct_significant_f']:.1f}%")

    console.print(stats_table)

    # Print acceptance criteria
    console.print("\n[bold cyan]Acceptance Criteria[/bold cyan]\n")

    criteria_table = Table(show_header=True, header_style="bold magenta")
    criteria_table.add_column("Criterion", style="cyan")
    criteria_table.add_column("Status", justify="center")
    criteria_table.add_column("Threshold", style="yellow")

    status = report.acceptance_status
    criteria_table.add_row(
        "R² > 0.30 for 70%+ sectors",
        "[green]PASS[/green]" if status["r2_criterion"] else "[red]FAIL[/red]",
        "70%",
    )
    criteria_table.add_row(
        "2/3 Sig Betas for 70%+ sectors",
        "[green]PASS[/green]" if status["significant_betas"] else "[red]FAIL[/red]",
        "70%",
    )
    criteria_table.add_row(
        "VIF < 5 for all factors",
        "[green]PASS[/green]" if status["multicollinearity"] else "[red]FAIL[/red]",
        "< 5.0",
    )
    criteria_table.add_row(
        "DW in [1.5, 2.5] for 70%+ sectors",
        "[green]PASS[/green]" if status["autocorrelation"] else "[red]FAIL[/red]",
        "70%",
    )
    criteria_table.add_row(
        "Overall",
        "[green]PASS[/green]" if status["overall"] else "[red]FAIL[/red]",
        "All",
    )

    console.print(criteria_table)

    # Print per-sector diagnostics
    console.print("\n[bold cyan]Per-Sector Diagnostics[/bold cyan]\n")

    sector_table = Table(show_header=True, header_style="bold magenta")
    sector_table.add_column("Sector", style="cyan")
    sector_table.add_column("R²", justify="right")
    sector_table.add_column("F-stat", justify="right")
    sector_table.add_column("Sig Betas", justify="center")
    sector_table.add_column("Max VIF", justify="right")
    sector_table.add_column("DW", justify="right")
    sector_table.add_column("N", justify="right")

    for sector_id, diag in sorted(report.diagnostics.items()):
        n_sig_betas = sum(
            [
                diag.p_value_duration < 0.05,
                diag.p_value_credit < 0.05,
                diag.p_value_liquidity < 0.05,
            ]
        )
        max_vif = max(diag.vif_duration, diag.vif_credit, diag.vif_liquidity)

        # Color code R² and DW
        r2_str = f"{diag.r_squared:.4f}"
        if diag.r_squared > 0.30:
            r2_str = f"[green]{r2_str}[/green]"
        else:
            r2_str = f"[red]{r2_str}[/red]"

        dw_str = f"{diag.durbin_watson:.2f}"
        if 1.5 <= diag.durbin_watson <= 2.5:
            dw_str = f"[green]{dw_str}[/green]"
        else:
            dw_str = f"[yellow]{dw_str}[/yellow]"

        sector_table.add_row(
            sector_id,
            r2_str,
            f"{diag.f_statistic:.2f}",
            f"{n_sig_betas}/3",
            f"{max_vif:.2f}",
            dw_str,
            str(diag.n_observations),
        )

    console.print(sector_table)


def export_report(
    report: SectorDiagnosticsReport,
    output_dir: Path,
) -> None:
    """
    Export report to JSON and markdown.

    Parameters
    ----------
    report : SectorDiagnosticsReport
        Diagnostics report to export.
    output_dir : Path
        Output directory for reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to JSON
    json_path = output_dir / "regression_diagnostics.json"

    # Convert diagnostics to dict
    diagnostics_dict = {}
    for sector_id, diag in report.diagnostics.items():
        diagnostics_dict[sector_id] = {
            "sector_id": diag.sector_id,
            "r_squared": diag.r_squared,
            "adj_r_squared": diag.adj_r_squared,
            "f_statistic": diag.f_statistic,
            "f_pvalue": diag.f_pvalue,
            "durbin_watson": diag.durbin_watson,
            "betas": {
                "duration": diag.beta_duration,
                "credit": diag.beta_credit,
                "liquidity": diag.beta_liquidity,
                "alpha": diag.alpha,
            },
            "p_values": {
                "duration": diag.p_value_duration,
                "credit": diag.p_value_credit,
                "liquidity": diag.p_value_liquidity,
                "alpha": diag.p_value_alpha,
            },
            "vifs": {
                "duration": diag.vif_duration,
                "credit": diag.vif_credit,
                "liquidity": diag.vif_liquidity,
            },
            "bp_test": {
                "statistic": diag.bp_test_statistic,
                "p_value": diag.bp_p_value,
            },
            "residuals": {
                "mean": diag.residual_mean,
                "std": diag.residual_std,
                "skewness": diag.residual_skewness,
                "kurtosis": diag.residual_kurtosis,
            },
            "n_observations": diag.n_observations,
            "date_range": {
                "start": diag.date_range_start.isoformat(),
                "end": diag.date_range_end.isoformat(),
            },
        }

    report_dict = {
        "summary_stats": report.summary_stats,
        "acceptance_status": report.acceptance_status,
        "diagnostics": diagnostics_dict,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }

    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    LOGGER.info("Exported JSON report", path=str(json_path))

    # Export to Markdown
    md_path = output_dir / "regression_diagnostics.md"

    with open(md_path, "w") as f:
        f.write("# Regression Diagnostics Report\n\n")
        f.write(f"**Generated:** {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")

        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for key, value in report.summary_stats.items():
            if isinstance(value, float):
                f.write(f"| {key.replace('_', ' ').title()} | {value:.4f} |\n")
            else:
                f.write(f"| {key.replace('_', ' ').title()} | {value} |\n")

        f.write("\n## Acceptance Criteria\n\n")
        f.write("| Criterion | Status | Threshold |\n")
        f.write("|-----------|--------|----------|\n")
        f.write(f"| R² > 0.30 for 70%+ sectors | {'✅ PASS' if report.acceptance_status['r2_criterion'] else '❌ FAIL'} | 70% |\n")
        f.write(f"| 2/3 Sig Betas for 70%+ sectors | {'✅ PASS' if report.acceptance_status['significant_betas'] else '❌ FAIL'} | 70% |\n")
        f.write(f"| VIF < 5 for all factors | {'✅ PASS' if report.acceptance_status['multicollinearity'] else '❌ FAIL'} | < 5.0 |\n")
        f.write(f"| DW in [1.5, 2.5] for 70%+ sectors | {'✅ PASS' if report.acceptance_status['autocorrelation'] else '❌ FAIL'} | 70% |\n")
        f.write(f"| **Overall** | {'✅ PASS' if report.acceptance_status['overall'] else '❌ FAIL'} | All |\n")

        f.write("\n## Per-Sector Diagnostics\n\n")
        f.write("| Sector | R² | Adj R² | F-stat | p-value | Sig Betas | Max VIF | DW | N |\n")
        f.write("|--------|-----|--------|--------|---------|-----------|---------|----|-|\n")

        for sector_id, diag in sorted(report.diagnostics.items()):
            n_sig = sum(
                [
                    diag.p_value_duration < 0.05,
                    diag.p_value_credit < 0.05,
                    diag.p_value_liquidity < 0.05,
                ]
            )
            max_vif = max(diag.vif_duration, diag.vif_credit, diag.vif_liquidity)

            f.write(
                f"| {sector_id} | {diag.r_squared:.4f} | {diag.adj_r_squared:.4f} | "
                f"{diag.f_statistic:.2f} | {diag.f_pvalue:.4f} | {n_sig}/3 | "
                f"{max_vif:.2f} | {diag.durbin_watson:.2f} | {diag.n_observations} |\n"
            )

        f.write("\n## Detailed Sector Analysis\n\n")

        for sector_id, diag in sorted(report.diagnostics.items()):
            f.write(f"### {sector_id}\n\n")
            f.write(f"**Date Range:** {diag.date_range_start.date()} to {diag.date_range_end.date()}\n\n")
            f.write(f"**Observations:** {diag.n_observations}\n\n")

            f.write("#### Regression Coefficients\n\n")
            f.write("| Factor | Beta | Std Error | t-stat | p-value | Significant |\n")
            f.write("|--------|------|-----------|--------|---------|-------------|\n")
            f.write(
                f"| Alpha | {diag.alpha:.4f} | {diag.se_alpha:.4f} | "
                f"{diag.t_stat_alpha:.2f} | {diag.p_value_alpha:.4f} | "
                f"{'✓' if diag.p_value_alpha < 0.05 else '✗'} |\n"
            )
            f.write(
                f"| Duration | {diag.beta_duration:.4f} | {diag.se_duration:.4f} | "
                f"{diag.t_stat_duration:.2f} | {diag.p_value_duration:.4f} | "
                f"{'✓' if diag.p_value_duration < 0.05 else '✗'} |\n"
            )
            f.write(
                f"| Credit | {diag.beta_credit:.4f} | {diag.se_credit:.4f} | "
                f"{diag.t_stat_credit:.2f} | {diag.p_value_credit:.4f} | "
                f"{'✓' if diag.p_value_credit < 0.05 else '✗'} |\n"
            )
            f.write(
                f"| Liquidity | {diag.beta_liquidity:.4f} | {diag.se_liquidity:.4f} | "
                f"{diag.t_stat_liquidity:.2f} | {diag.p_value_liquidity:.4f} | "
                f"{'✓' if diag.p_value_liquidity < 0.05 else '✗'} |\n"
            )

            f.write("\n#### Diagnostics\n\n")
            f.write(f"- **R²:** {diag.r_squared:.4f}\n")
            f.write(f"- **Adjusted R²:** {diag.adj_r_squared:.4f}\n")
            f.write(f"- **F-statistic:** {diag.f_statistic:.2f} (p={diag.f_pvalue:.4f})\n")
            f.write(f"- **Durbin-Watson:** {diag.durbin_watson:.2f}\n")
            f.write(f"- **Breusch-Pagan:** {diag.bp_test_statistic:.2f} (p={diag.bp_p_value:.4f})\n")
            f.write(f"- **VIF (Duration):** {diag.vif_duration:.2f}\n")
            f.write(f"- **VIF (Credit):** {diag.vif_credit:.2f}\n")
            f.write(f"- **VIF (Liquidity):** {diag.vif_liquidity:.2f}\n")

            f.write("\n#### Residuals\n\n")
            f.write(f"- **Mean:** {diag.residual_mean:.6f}\n")
            f.write(f"- **Std Dev:** {diag.residual_std:.6f}\n")
            f.write(f"- **Skewness:** {diag.residual_skewness:.4f}\n")
            f.write(f"- **Kurtosis:** {diag.residual_kurtosis:.4f}\n\n")

    LOGGER.info("Exported Markdown report", path=str(md_path))


def main() -> None:
    """Main entry point for diagnostics report generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate regression diagnostics report for 3D Risk Model"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2010-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="playground/reports/regression_diagnostics",
        help="Output directory for reports",
    )

    args = parser.parse_args()

    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    LOGGER.info(
        "Starting diagnostics report generation",
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Load data
    sector_returns, factor_returns = load_sector_and_factor_data(
        args.start_date,
        args.end_date,
    )

    # Compute diagnostics
    factor_columns = ["factor_duration", "factor_credit", "factor_liquidity"]
    diagnostics = compute_regression_diagnostics(
        sector_returns,
        factor_returns,
        factor_columns=factor_columns,
    )

    # Create summary report
    report = create_diagnostics_summary(diagnostics)

    # Print to console
    print_summary_table(report)

    # Export report
    output_dir = Path(args.output_dir)
    export_report(report, output_dir)

    LOGGER.info("Diagnostics report generation complete", output_dir=str(output_dir))

    # Exit with error code if acceptance criteria not met
    if not report.acceptance_status["overall"]:
        LOGGER.error("Acceptance criteria NOT met - some diagnostics failed")
        sys.exit(1)
    else:
        LOGGER.info("Acceptance criteria MET - all diagnostics passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
