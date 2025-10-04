"""CLI contract tests for the playground risk pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

from playground.risk_model.cli import main as cli_main
from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.pipeline import RiskPipelineResult
from playground.risk_model.visualization import VisualizationPayload


@pytest.fixture()
def _stub_result() -> RiskPipelineResult:
    """Provide a deterministic pipeline result for CLI contract tests."""
    coverage_summary = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=252,
        factor_expected_days=252,
        sector_coverage={"XLF": 0.91},
        factor_coverage={"factor_duration": 0.88},
        composite_coverage={"macro_liquidity": 0.62},
    )
    visualization_payloads: dict[int, VisualizationPayload] = {}
    alerts = {
        "sector": {},
        "factor": {"factor_duration": 0.88},
        "composite": {"macro_liquidity": 0.62},
    }
    return RiskPipelineResult(
        sector_returns=pl.DataFrame({"timestamp": [], "symbol": [], "return": []}),
        factor_levels=pl.DataFrame({"timestamp": [], "factor_duration": []}),
        factor_returns=pl.DataFrame({"timestamp": [], "factor_duration": []}),
        exposures=pl.DataFrame({"asset_id": [], "benchmark_id": [], "ewma_beta": []}),
        profiles=[],
        distance_reports={},
        visualization_payloads=visualization_payloads,
        coverage_summary=coverage_summary,
        eigenvalue_trends={"2020s": {"lambda_1": 1.0}},
        coverage_alerts=alerts,
        optimizer_recommendations={2024: {"XLF": 0.6, "XLK": 0.4}},
        beta_persisted_rows=42,
    )


def test_cli_writes_coverage_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _stub_result: RiskPipelineResult, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --coverage-report emits the expected JSON structure."""

    def _fake_run_pipeline(_config: object) -> RiskPipelineResult:
        return _stub_result

    monkeypatch.setattr("playground.risk_model.cli.run_risk_pipeline", _fake_run_pipeline)

    report_path = tmp_path / "coverage.json"
    persist_dir = tmp_path / "persist"
    cache_dir = tmp_path / "cache"
    vis_dir = tmp_path / "vis"

    argv = [
        "python",
        "--start",
        "2020-01-01",
        "--end",
        "2020-12-31",
        "--coverage-report",
        str(report_path),
        "--persist-dir",
        str(persist_dir),
        "--cache-dir",
        str(cache_dir),
        "--visualization-dir",
        str(vis_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli_main()

    out, _ = capsys.readouterr()
    assert "Completed pipeline" in out
    assert "[coverage]" in out
    assert "[alerts]" in out
    assert "macro_liquidity (62.0%)" in out
    assert "factor_duration (88.0%)" in out
    assert "[optimizer]" in out
    assert "beta_rows" in out

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["coverage"]["calendar_name"] == "XNYS"
    assert payload["coverage"]["composite_coverage"]["macro_liquidity"] == pytest.approx(0.62)
    assert payload["coverage_alerts"]["composite"]["macro_liquidity"] == pytest.approx(0.62)
