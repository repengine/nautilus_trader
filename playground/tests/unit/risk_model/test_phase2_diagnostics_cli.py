"""Tests for the Phase 2 regression diagnostics CLI."""

from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from playground.scripts import run_phase2_regression_diagnostics as diagnostics_cli


def _build_sample_dataset(root: Path) -> None:
    """Persist a deterministic sector/factor dataset for CLI tests."""
    root.mkdir(parents=True, exist_ok=True)
    timestamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(64)]
    rng = np.random.default_rng(42)
    duration_values = rng.normal(0.0, 0.002, size=len(timestamps))
    credit_values = 0.5 * duration_values + rng.normal(0.0, 0.0008, size=len(timestamps))
    liquidity_values = rng.normal(0.0, 0.0015, size=len(timestamps))

    duration = pl.Series("factor_duration", duration_values.tolist())
    credit = pl.Series("factor_credit", credit_values.tolist())
    liquidity = pl.Series("factor_liquidity", liquidity_values.tolist())
    factor_returns = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": duration,
            "factor_credit": credit,
            "factor_liquidity": liquidity,
        }
    )

    sector_rows: list[dict[str, object]] = []
    for sector_index, sector in enumerate(diagnostics_cli.DEFAULT_SECTORS):
        sector_noise = rng.normal(0.0, 0.0005, size=len(timestamps))
        sector_offset = 0.00005 * float(sector_index)
        for idx, ts in enumerate(timestamps):
            base = 0.0005 + 1.2 * duration_values[idx] - 0.8 * credit_values[idx] + 0.6 * liquidity_values[idx]
            sector_rows.append(
                {
                    "timestamp": ts,
                    "symbol": sector,
                    "return": base + sector_offset + sector_noise[idx],
                }
            )

    sector_returns = pl.DataFrame(sector_rows)

    factor_returns.write_parquet(root / diagnostics_cli.FACTOR_RETURNS_FILENAME)
    sector_returns.write_parquet(root / diagnostics_cli.SECTOR_RETURNS_FILENAME)


@pytest.mark.parametrize("run_tag", ["unit-test-run"])
def test_phase2_diagnostics_cli_generates_reports(tmp_path: Path, run_tag: str) -> None:
    """End-to-end test ensuring CLI emits diagnostics artefacts."""
    dataset_root = tmp_path / "dataset"
    _build_sample_dataset(dataset_root)

    output_root = tmp_path / "reports"
    diagnostics_cli.main(
        [
            "--dataset-path",
            str(dataset_root),
            "--output-dir",
            str(output_root),
            "--run-tag",
            run_tag,
            "--start",
            "2020-01-01",
            "--end",
            "2020-03-01",
            "--sectors",
            *diagnostics_cli.DEFAULT_SECTORS,
        ]
    )

    run_directory = output_root / run_tag
    summary_path = run_directory / "phase2_regression_summary.json"
    diagnostics_csv = run_directory / "sector_regression_diagnostics.csv"
    diagnostics_parquet = run_directory / "sector_regression_diagnostics.parquet"

    assert summary_path.exists()
    assert diagnostics_csv.exists()
    assert diagnostics_parquet.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["acceptance_status"]["overall"] is True
    assert summary["n_sectors"] == len(diagnostics_cli.DEFAULT_SECTORS)
    assert summary["config"]["thresholds"]["r2_threshold"] == pytest.approx(0.30)

    diagnostics_frame = pl.read_csv(diagnostics_csv)
    assert set(diagnostics_frame["sector_id"].to_list()) == set(diagnostics_cli.DEFAULT_SECTORS)
    assert (diagnostics_frame["r_squared"] > 0.90).all()
