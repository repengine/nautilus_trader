#!/usr/bin/env python3
"""
Demonstration of regression diagnostics with synthetic data.

This script creates synthetic sector and factor data to demonstrate
the diagnostics module and validate acceptance criteria.
"""

from __future__ import annotations

import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import structlog


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from playground.risk_model.diagnostics import compute_regression_diagnostics
from playground.risk_model.diagnostics import create_diagnostics_summary


structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)

LOGGER = structlog.get_logger(__name__)


def create_synthetic_data(n_sectors: int = 9, n_observations: int = 500) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Create synthetic sector and factor data with realistic characteristics.

    Parameters
    ----------
    n_sectors : int
        Number of sectors to generate (default: 9 for sector ETFs).
    n_observations : int
        Number of daily observations (default: 500 ≈ 2 years).

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Sector returns and factor returns.
    """
    LOGGER.info("Creating synthetic data", n_sectors=n_sectors, n_observations=n_observations)

    np.random.seed(42)

    # Generate timestamps
    timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + np.timedelta64(i, "D") for i in range(n_observations)]

    # Generate factor returns (mean=0, realistic volatility)
    factor_duration = np.random.randn(n_observations) * 0.02  # 2% daily std
    factor_credit = np.random.randn(n_observations) * 0.015  # 1.5% daily std
    factor_liquidity = np.random.randn(n_observations) * 0.01  # 1% daily std

    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": factor_duration,
            "factor_credit": factor_credit,
            "factor_liquidity": factor_liquidity,
        }
    )

    # Generate sector returns with different factor exposures and R² levels
    sector_configs = [
        # (name, beta_dur, beta_cred, beta_liq, noise_std, description)
        ("XLU", 0.85, 0.42, -0.31, 0.003, "Utilities - High fit, defensive"),
        ("XLF", 0.65, 0.78, 0.15, 0.004, "Financials - High fit, credit sensitive"),
        ("XLE", 0.45, 0.55, 0.60, 0.005, "Energy - Medium fit, commodity exposed"),
        ("XLI", 0.50, 0.35, 0.25, 0.005, "Industrials - Medium fit, cyclical"),
        ("XLB", 0.40, 0.45, 0.35, 0.006, "Materials - Medium fit, commodity"),
        ("XLY", -0.30, -0.50, 0.20, 0.006, "Consumer Disc - Medium fit, growth"),
        ("XLK", -0.45, -0.28, 0.12, 0.008, "Technology - Low fit, idiosyncratic"),
        ("XLV", 0.35, 0.15, -0.10, 0.007, "Healthcare - Medium fit, defensive"),
        ("XLP", 0.50, 0.20, -0.20, 0.005, "Consumer Staples - Medium fit, defensive"),
    ]

    sector_records = []
    for sector_name, beta_dur, beta_cred, beta_liq, noise_std, description in sector_configs:
        # Generate returns based on factor model + idiosyncratic noise
        sector_returns = (
            0.0005  # Small positive drift
            + beta_dur * factor_duration
            + beta_cred * factor_credit
            + beta_liq * factor_liquidity
            + np.random.randn(n_observations) * noise_std
        )

        for ts, ret in zip(timestamps, sector_returns):
            sector_records.append(
                {
                    "timestamp": ts,
                    "symbol": sector_name,
                    "return": ret,
                }
            )

        LOGGER.info(
            "Generated sector",
            sector=sector_name,
            description=description,
            betas={"duration": beta_dur, "credit": beta_cred, "liquidity": beta_liq},
        )

    sector_returns_df = pl.DataFrame(sector_records)

    return sector_returns_df, factor_returns


def main() -> None:
    """Main entry point for demo."""
    LOGGER.info("Starting regression diagnostics demo")

    # Create synthetic data
    sector_returns, factor_returns = create_synthetic_data()

    # Compute diagnostics
    LOGGER.info("Computing regression diagnostics")
    diagnostics = compute_regression_diagnostics(
        sector_returns,
        factor_returns,
        factor_columns=["factor_duration", "factor_credit", "factor_liquidity"],
    )

    # Create summary report
    LOGGER.info("Creating summary report")
    report = create_diagnostics_summary(diagnostics)

    # Display results
    print("\n" + "=" * 80)
    print("REGRESSION DIAGNOSTICS SUMMARY")
    print("=" * 80)

    print("\n--- Summary Statistics ---")
    for key, value in report.summary_stats.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")

    print("\n--- Acceptance Criteria ---")
    for criterion, passed in report.acceptance_status.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {criterion.replace('_', ' ').title()}: {status}")

    print("\n--- Per-Sector Results ---")
    print(f"{'Sector':<6} {'R²':>8} {'Adj R²':>8} {'F-stat':>10} {'Sig Betas':>10} {'Max VIF':>10} {'DW':>8}")
    print("-" * 80)

    for sector_id in sorted(diagnostics.keys()):
        diag = diagnostics[sector_id]
        n_sig_betas = sum(
            [
                diag.p_value_duration < 0.05,
                diag.p_value_credit < 0.05,
                diag.p_value_liquidity < 0.05,
            ]
        )
        max_vif = max(diag.vif_duration, diag.vif_credit, diag.vif_liquidity)

        print(
            f"{sector_id:<6} {diag.r_squared:8.4f} {diag.adj_r_squared:8.4f} "
            f"{diag.f_statistic:10.2f} {n_sig_betas:>10}/3 {max_vif:10.2f} {diag.durbin_watson:8.2f}"
        )

    # Detailed example for one sector
    example_sector = "XLU"
    if example_sector in diagnostics:
        diag = diagnostics[example_sector]
        print(f"\n--- Detailed Example: {example_sector} (Utilities) ---")
        print(f"  R²: {diag.r_squared:.4f}")
        print(f"  Adjusted R²: {diag.adj_r_squared:.4f}")
        print(f"  F-statistic: {diag.f_statistic:.2f} (p={diag.f_pvalue:.6f})")
        print("\n  Factor Coefficients:")
        print(
            f"    Duration:   β={diag.beta_duration:7.4f}, t={diag.t_stat_duration:7.2f}, "
            f"p={diag.p_value_duration:.6f} {'✓' if diag.p_value_duration < 0.05 else '✗'}"
        )
        print(
            f"    Credit:     β={diag.beta_credit:7.4f}, t={diag.t_stat_credit:7.2f}, "
            f"p={diag.p_value_credit:.6f} {'✓' if diag.p_value_credit < 0.05 else '✗'}"
        )
        print(
            f"    Liquidity:  β={diag.beta_liquidity:7.4f}, t={diag.t_stat_liquidity:7.2f}, "
            f"p={diag.p_value_liquidity:.6f} {'✓' if diag.p_value_liquidity < 0.05 else '✗'}"
        )
        print("\n  Diagnostics:")
        print(f"    VIF (Duration): {diag.vif_duration:.2f}")
        print(f"    VIF (Credit): {diag.vif_credit:.2f}")
        print(f"    VIF (Liquidity): {diag.vif_liquidity:.2f}")
        print(f"    Durbin-Watson: {diag.durbin_watson:.2f}")
        print(f"    Breusch-Pagan: {diag.bp_test_statistic:.2f} (p={diag.bp_p_value:.4f})")
        print("\n  Residuals:")
        print(f"    Mean: {diag.residual_mean:.6f}")
        print(f"    Std Dev: {diag.residual_std:.6f}")
        print(f"    Skewness: {diag.residual_skewness:.4f}")
        print(f"    Kurtosis: {diag.residual_kurtosis:.4f}")

    print("\n" + "=" * 80)

    # Final verdict
    if report.acceptance_status["overall"]:
        LOGGER.info("✅ All acceptance criteria PASSED")
        print("\n✅ SUCCESS: The 3D Factor Risk Model meets all acceptance criteria!")
        print("   The model has sufficient explanatory power and statistical significance.")
    else:
        LOGGER.warning("❌ Some acceptance criteria FAILED")
        print("\n❌ CAUTION: Some acceptance criteria were not met.")
        print("   Review the diagnostics above to identify issues.")

        # Identify specific failures
        failures = [k for k, v in report.acceptance_status.items() if k != "overall" and not v]
        if failures:
            print("\n   Failed criteria:")
            for criterion in failures:
                print(f"     - {criterion.replace('_', ' ').title()}")


if __name__ == "__main__":
    main()
