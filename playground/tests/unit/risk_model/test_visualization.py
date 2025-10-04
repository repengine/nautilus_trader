"""Tests for visualization payload helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from playground.exposure.optimizer import RiskPoint
from playground.risk_model.analysis import AnnualRiskProfile
from playground.risk_model.analysis import SectorDistanceReport
from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.visualization import VisualizationPayload
from playground.risk_model.visualization import build_visualization_payload


def test_build_visualization_payload_serializes(tmp_path: Path) -> None:
    profile = AnnualRiskProfile(
        year=2020,
        weights={"XLF": 0.6, "XLK": 0.4},
        sharpe_scores={"XLF": 1.2, "XLK": 0.8},
        risk_point=RiskPoint(
            {
                "factor_duration": 0.45,
                "factor_credit": 0.35,
                "factor_liquidity": 0.20,
            },
        ),
    )
    reports = [
        SectorDistanceReport(
            sector_id="XLF",
            distance=0.05,
            coordinates={
                "factor_duration": 0.40,
                "factor_credit": 0.33,
                "factor_liquidity": 0.18,
            },
            deltas={
                "factor_duration": -0.05,
                "factor_credit": -0.02,
                "factor_liquidity": -0.02,
            },
            recommended_weight=0.6,
        ),
    ]

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=252,
        factor_expected_days=252,
        sector_coverage={"XLF": 1.0},
        factor_coverage={"factor_duration": 1.0},
    )
    eigen_trends = {"2020s": {"eig_1": 1.23}}
    coverage_alerts = {
        "sector": {},
        "factor": {},
        "composite": {},
    }

    payload = build_visualization_payload(
        profile,
        reports,
        notes="unit-test",
        coverage=coverage,
        eigenvalue_trends=eigen_trends,
        coverage_alerts=coverage_alerts,
    )
    json_text = payload.to_json(tmp_path / "payload.json")

    assert isinstance(payload, VisualizationPayload)
    assert payload.year == 2020
    assert (tmp_path / "payload.json").exists()

    parsed = json.loads(json_text)
    assert parsed["ideal_point"]["factor_duration"] == pytest.approx(0.45)
    assert parsed["metadata"]["notes"] == "unit-test"
    assert parsed["metadata"]["status"] == "success"
    assert parsed["sectors"][0]["sector"] == "XLF"
    assert parsed["metadata"]["coverage"]["calendar_name"] == "XNYS"
    assert parsed["metadata"]["eigenvalue_trends"] == eigen_trends
    assert parsed["metadata"]["coverage_alerts"] == coverage_alerts
