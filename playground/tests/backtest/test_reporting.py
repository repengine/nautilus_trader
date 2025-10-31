"""
Tests for reporting utilities that transform sensitivity outputs into PDFs.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from playground.backtest.reporting import generate_sensitivity_summary_pdf


def test_generate_sensitivity_summary_pdf(tmp_path: Path) -> None:
    """PDF generation should materialise a readable document."""
    summary_path = tmp_path / "summary.csv"
    frame = pl.DataFrame({
        "spec": ["Rolling Window Sensitivity", "Transaction Cost Sensitivity"],
        "slug": ["rolling-window-sensitivity", "transaction-cost-sensitivity"],
        "strategy": ["3D Factor (Rolling Betas)", "3D Factor (Rolling Betas)"],
        "metric": ["sharpe_ratio", "sharpe_ratio"],
        "metric_value": [0.9123, 0.8876],
        "metric_spread": [0.05, 0.02],
        "metric_spread_tolerance": [0.15, 0.15],
        "metric_spread_ok": [True, True],
        "evaluated_combinations": [3, 3],
        "best_config": ['{"rolling_window": 252}', '{"config.transaction_cost_bps": 10.0}'],
    })
    frame.write_csv(summary_path)

    output_path = tmp_path / "sensitivity.pdf"
    generate_sensitivity_summary_pdf(summary_path=summary_path, output_path=output_path, title="Test Report")

    assert output_path.exists()
    header = output_path.read_bytes()[:4]
    assert header == b"%PDF"
