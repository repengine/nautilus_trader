"""Serialization helpers for the Three.js 3D risk view."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from playground.risk_model.analysis import AnnualRiskProfile
from playground.risk_model.analysis import SectorDistanceReport
from playground.risk_model.dataset import CoverageSummary


LOGGER = structlog.get_logger(__name__)


@dataclass(slots=True)
class VisualizationPayload:
    """Data structure compatible with the playground Three.js renderer."""

    year: int
    ideal_point: dict[str, float]
    sectors: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_json(self, path: Path | None = None) -> str:
        """Serialize the payload to JSON and optionally persist to disk."""
        json_payload = json.dumps(asdict(self), indent=2, sort_keys=True)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_payload, encoding="utf-8")
        return json_payload


def build_visualization_payload(
    profile: AnnualRiskProfile,
    reports: Sequence[SectorDistanceReport],
    *,
    notes: str | None = None,
    coverage: CoverageSummary | None = None,
    eigenvalue_trends: dict[str, dict[str, float]] | None = None,
    coverage_alerts: dict[str, dict[str, float]] | None = None,
) -> VisualizationPayload:
    """Create a Three.js-friendly payload for a single year view."""
    sectors = [
        {
            "sector": report.sector_id,
            "coordinates": report.coordinates,
            "distance": report.distance,
            "deltas": report.deltas,
            "recommended_weight": report.recommended_weight,
            "mahalanobis_distance": report.mahalanobis_distance,
        }
        for report in reports
    ]

    metadata = {
        "notes": notes or "",
        "weights": profile.weights,
        "sharpe_scores": profile.sharpe_scores,
        "status": profile.status,
    }
    if profile.diagnostics is not None:
        metadata["diagnostics"] = profile.diagnostics
    if coverage is not None:
        metadata["coverage"] = coverage.to_dict()
    if eigenvalue_trends:
        metadata["eigenvalue_trends"] = eigenvalue_trends
    if coverage_alerts:
        metadata["coverage_alerts"] = coverage_alerts
    LOGGER.debug(
        "Built visualization payload",
        year=profile.year,
        sector_count=len(sectors),
    )
    return VisualizationPayload(
        year=profile.year,
        ideal_point=dict(profile.risk_point.coordinates),
        sectors=sectors,
        metadata=metadata,
    )


__all__ = ["VisualizationPayload", "build_visualization_payload"]
