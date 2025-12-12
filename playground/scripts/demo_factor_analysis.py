"""
Demo script for factor correlation and PCA analysis.

This script demonstrates:
1. Loading historical factor returns (2010-2024)
2. Computing factor correlation matrix
3. Running PCA analysis
4. Displaying results with rich tables
5. Evaluating acceptance criteria
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

from playground.risk_model.factor_analysis import compute_factor_correlations
from playground.risk_model.factor_analysis import compute_pca_analysis


# Constants
DATA_PATH = Path(__file__).parent.parent / "data" / "sector_dataset" / "factor_returns.parquet"
FACTOR_COLUMNS = ["factor_duration", "factor_credit", "factor_liquidity"]
CORRELATION_THRESHOLD = 0.50
VARIANCE_THRESHOLD = 0.80

console = Console()


def load_factor_returns() -> pl.DataFrame:
    """Load factor returns from parquet file."""
    console.print(f"\n[bold cyan]Loading factor returns from:[/bold cyan] {DATA_PATH}")

    if not DATA_PATH.exists():
        msg = f"Data file not found: {DATA_PATH}"
        raise FileNotFoundError(msg)

    df = pl.read_parquet(DATA_PATH)

    # Validate required columns
    missing_cols = set(FACTOR_COLUMNS) - set(df.columns)
    if missing_cols:
        msg = f"Missing factor columns: {sorted(missing_cols)}"
        raise ValueError(msg)

    console.print(f"[green]✓[/green] Loaded {len(df):,} observations")
    console.print(f"[green]✓[/green] Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def display_correlation_results(
    correlation_analysis,
) -> None:
    """Display correlation analysis results in formatted tables."""
    console.print("\n[bold yellow]═" * 40)
    console.print("[bold yellow]CORRELATION ANALYSIS RESULTS")
    console.print("[bold yellow]═" * 40)

    # Summary metrics
    console.print("\n[bold]Summary Metrics:[/bold]")
    console.print(f"  • Observations: {correlation_analysis.n_observations:,}")
    console.print(f"  • Max |r| (off-diagonal): {correlation_analysis.max_abs_correlation:.4f}")
    console.print(f"  • Mean |r| (off-diagonal): {correlation_analysis.mean_abs_correlation:.4f}")
    console.print(
        f"  • Threshold: {CORRELATION_THRESHOLD:.2f}",
    )

    # Orthogonality status
    if correlation_analysis.is_orthogonal:
        console.print(f"  • Status: [bold green]✓ ORTHOGONAL[/bold green] (max |r| < {CORRELATION_THRESHOLD})")
    else:
        console.print(f"  • Status: [bold red]✗ NOT ORTHOGONAL[/bold red] (max |r| ≥ {CORRELATION_THRESHOLD})")

    # Correlation matrix table
    console.print("\n[bold]Correlation Matrix:[/bold]")
    table = Table(show_header=True, header_style="bold cyan", show_lines=True)
    table.add_column("Factor", style="bold")

    for factor in correlation_analysis.factor_names:
        table.add_column(factor, justify="right")

    for factor1 in correlation_analysis.factor_names:
        row = [factor1]
        for factor2 in correlation_analysis.factor_names:
            corr = correlation_analysis.correlation_matrix[factor1][factor2]

            # Color code based on magnitude
            if factor1 == factor2:
                # Diagonal (should be 1.0)
                cell = f"[bold white]{corr:.2f}[/bold white]"
            elif abs(corr) < 0.30:
                # Weak correlation (good)
                cell = f"[green]{corr:.2f}[/green]"
            elif abs(corr) < 0.50:
                # Moderate correlation (acceptable)
                cell = f"[yellow]{corr:.2f}[/yellow]"
            else:
                # Strong correlation (problematic)
                cell = f"[red]{corr:.2f}[/red]"

            row.append(cell)

        table.add_row(*row)

    console.print(table)


def display_pca_results(pca_analysis) -> None:
    """Display PCA analysis results in formatted tables."""
    console.print("\n[bold yellow]═" * 40)
    console.print("[bold yellow]PCA ANALYSIS RESULTS")
    console.print("[bold yellow]═" * 40)

    # Summary metrics
    console.print("\n[bold]Summary Metrics:[/bold]")
    console.print(f"  • Number of components: {pca_analysis.n_components}")
    console.print(f"  • Variance by 3 PCs: {pca_analysis.variance_captured_by_3pc:.2%}")
    console.print(f"  • Threshold: {VARIANCE_THRESHOLD:.0%}")

    # Adequacy status
    if pca_analysis.is_adequate:
        console.print(
            f"  • Status: [bold green]✓ ADEQUATE[/bold green] (variance > {VARIANCE_THRESHOLD:.0%})"
        )
    else:
        console.print(
            f"  • Status: [bold red]✗ INADEQUATE[/bold red] (variance ≤ {VARIANCE_THRESHOLD:.0%})"
        )

    # Explained variance table
    console.print("\n[bold]Explained Variance by Component:[/bold]")
    var_table = Table(show_header=True, header_style="bold cyan")
    var_table.add_column("Component", style="bold")
    var_table.add_column("Variance Explained", justify="right")
    var_table.add_column("Cumulative Variance", justify="right")
    var_table.add_column("Eigenvalue", justify="right")

    for i in range(pca_analysis.n_components):
        pc_name = f"PC{i+1}"
        var_explained = pca_analysis.explained_variance_ratio[i]
        cum_var = pca_analysis.cumulative_variance[i]
        eigenvalue = pca_analysis.eigenvalues[i]

        # Color code based on variance explained
        if var_explained > 0.40:
            var_style = "red"  # Too dominant
        elif var_explained > 0.25:
            var_style = "green"  # Good balance
        else:
            var_style = "yellow"  # Low contribution

        var_table.add_row(
            pc_name,
            f"[{var_style}]{var_explained:.2%}[/{var_style}]",
            f"{cum_var:.2%}",
            f"{eigenvalue:.4f}",
        )

    console.print(var_table)

    # Loadings table
    console.print("\n[bold]Principal Component Loadings:[/bold]")
    console.print("[dim](Correlation between original factors and PCs)[/dim]")

    loadings_table = Table(show_header=True, header_style="bold cyan")
    loadings_table.add_column("Principal Component", style="bold")

    # Get factor names from first PC
    first_pc = "PC1"
    factor_names = list(pca_analysis.loadings[first_pc].keys())

    for factor in factor_names:
        loadings_table.add_column(factor, justify="right")

    for i in range(pca_analysis.n_components):
        pc_name = f"PC{i+1}"
        row = [pc_name]

        for factor in factor_names:
            loading = pca_analysis.loadings[pc_name][factor]

            # Color code based on magnitude
            if abs(loading) > 0.5:
                style = "green"  # Strong loading
            elif abs(loading) > 0.3:
                style = "yellow"  # Moderate loading
            else:
                style = "dim"  # Weak loading

            row.append(f"[{style}]{loading:.4f}[/{style}]")

        loadings_table.add_row(*row)

    console.print(loadings_table)


def display_acceptance_criteria(
    correlation_analysis,
    pca_analysis,
) -> None:
    """Display overall acceptance criteria status."""
    console.print("\n[bold yellow]═" * 40)
    console.print("[bold yellow]ACCEPTANCE CRITERIA")
    console.print("[bold yellow]═" * 40)

    # Create criteria table
    criteria_table = Table(show_header=True, header_style="bold cyan")
    criteria_table.add_column("Criterion", style="bold")
    criteria_table.add_column("Requirement", justify="center")
    criteria_table.add_column("Actual", justify="center")
    criteria_table.add_column("Status", justify="center")

    # Factor independence criterion
    independence_pass = correlation_analysis.is_orthogonal
    criteria_table.add_row(
        "Factor Independence",
        f"max |r| < {CORRELATION_THRESHOLD}",
        f"{correlation_analysis.max_abs_correlation:.4f}",
        "[bold green]✓ PASS[/bold green]" if independence_pass else "[bold red]✗ FAIL[/bold red]",
    )

    # PCA variance criterion
    pca_pass = pca_analysis.is_adequate
    criteria_table.add_row(
        "PCA Dimensionality",
        f"3 PCs > {VARIANCE_THRESHOLD:.0%}",
        f"{pca_analysis.variance_captured_by_3pc:.2%}",
        "[bold green]✓ PASS[/bold green]" if pca_pass else "[bold red]✗ FAIL[/bold red]",
    )

    # Overall status
    overall_pass = independence_pass and pca_pass
    criteria_table.add_row(
        "[bold]OVERALL[/bold]",
        "[bold]All criteria[/bold]",
        "[bold]-[/bold]",
        "[bold green]✓ PASS[/bold green]" if overall_pass else "[bold red]✗ FAIL[/bold red]",
    )

    console.print(criteria_table)

    # Final verdict
    console.print()
    if overall_pass:
        console.print(
            "[bold green]"
            "═" * 60 + "\n"
            "VERDICT: FACTORS ARE INDEPENDENT AND COMPREHENSIVE\n"
            "The 3D risk model factors meet all acceptance criteria.\n"
            "Proceed with factor-based risk decomposition.\n"
            + "═" * 60
            + "[/bold green]"
        )
    else:
        console.print(
            "[bold red]"
            "═" * 60 + "\n"
            "VERDICT: FACTORS DO NOT MEET ACCEPTANCE CRITERIA\n"
            "Review factor construction and consider remediation:\n"
            "  • Orthogonalization (Gram-Schmidt)\n"
            "  • Alternative factor proxies\n"
            "  • Higher-frequency data\n"
            + "═" * 60
            + "[/bold red]"
        )


def main() -> None:
    """Run factor correlation and PCA analysis demo."""
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]FACTOR CORRELATION & ORTHOGONALITY ANALYSIS DEMO[/bold magenta]")
    console.print("[bold magenta]3D Risk Model (Duration, Credit, Liquidity)[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    try:
        # Load data
        factor_returns = load_factor_returns()

        # Compute correlation analysis
        console.print("\n[bold cyan]Computing factor correlations...[/bold cyan]")
        correlation_analysis = compute_factor_correlations(
            factor_returns,
            factor_columns=FACTOR_COLUMNS,
            correlation_threshold=CORRELATION_THRESHOLD,
        )
        console.print("[green]✓[/green] Correlation analysis complete")

        # Compute PCA analysis
        console.print("\n[bold cyan]Computing PCA analysis...[/bold cyan]")
        pca_analysis = compute_pca_analysis(
            factor_returns,
            factor_columns=FACTOR_COLUMNS,
            n_components=3,
            variance_threshold=VARIANCE_THRESHOLD,
        )
        console.print("[green]✓[/green] PCA analysis complete")

        # Display results
        display_correlation_results(correlation_analysis)
        display_pca_results(pca_analysis)
        display_acceptance_criteria(correlation_analysis, pca_analysis)

        console.print("\n[dim]For detailed methodology, see: playground/docs/factor_correlation_analysis.md[/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]ERROR:[/bold red] {e}")
        raise


if __name__ == "__main__":
    main()
