"""
Tests for the Phase 4 sensitivity CLI helpers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from playground.scripts.run_phase4_sensitivity import main as sensitivity_main


class _StubSuiteResult:
    """Stubbed sensitivity suite result used to isolate the CLI."""

    output_dirname = "sensitivity"
    runs: tuple[Any, ...] = tuple()

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def summary_frame(self) -> pl.DataFrame:
        """Return the precomputed summary frame."""
        return self._frame


@pytest.fixture()
def stub_summary_frame() -> pl.DataFrame:
    """Generate a minimal sensitivity summary dataframe."""
    return pl.DataFrame({
        "spec": ["Rolling Window Sensitivity"],
        "slug": ["rolling-window-sensitivity"],
        "strategy": ["3D Factor (Rolling Betas)"],
        "metric": ["sharpe_ratio"],
        "metric_value": [0.94],
        "metric_spread": [0.01],
        "metric_spread_tolerance": [0.15],
        "metric_spread_ok": [True],
        "evaluated_combinations": [3],
        "best_config": ['{"rolling_window": 252}'],
    })


def test_run_phase4_sensitivity_cli_writes_pdf(
    tmp_path: Path,
    stub_summary_frame: pl.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should render a sensitivity PDF when a summary is available."""
    output_dir = tmp_path / "reports" / "backtesting"
    summary_dir = output_dir / "sensitivity"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summary_dir / "summary.csv"
    stub_summary_frame.write_csv(summary_path)

    def _stub_run_parameter_sensitivity_suite(
        *,
        dataset_path: Path,
        output_dir: Path,
        spec_slugs: Sequence[str] | None = None,
    ) -> _StubSuiteResult:
        assert output_dir == tmp_path / "reports" / "backtesting"
        return _StubSuiteResult(frame=stub_summary_frame)

    monkeypatch.setattr(
        "playground.scripts.run_phase4_sensitivity.run_parameter_sensitivity_suite",
        _stub_run_parameter_sensitivity_suite,
    )

    args = [
        "--dataset-path",
        str(tmp_path / "dataset"),
        "--output-dir",
        str(output_dir),
    ]
    sensitivity_main(args)

    pdf_path = summary_dir / "sensitivity_analysis.pdf"
    assert pdf_path.exists()
    header = pdf_path.read_bytes()[:4]
    assert header == b"%PDF"
