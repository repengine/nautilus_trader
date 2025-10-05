"""Serialization helpers for the Three.js 3D risk view."""

from __future__ import annotations

import json
import math
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


def _compute_cloud_radius(
    current_position: dict[str, float],
    stable_center: dict[str, float],
) -> float:
    """
    Compute Euclidean distance between current position and stable center.

    Parameters
    ----------
    current_position : dict[str, float]
        Current sector position in factor space.
    stable_center : dict[str, float]
        Stable long-term center for the sector.

    Returns
    -------
    float
        Euclidean distance (cloud radius).

    Notes
    -----
    The cloud radius quantifies how far a sector has drifted from its
    long-term stable position. Larger values indicate regime shifts or
    unusual market conditions.

    The calculation uses all factors present in either dictionary,
    defaulting to 0.0 for missing values.
    """
    all_factors = set(current_position.keys()) | set(stable_center.keys())
    squared_sum = sum(
        (current_position.get(factor, 0.0) - stable_center.get(factor, 0.0)) ** 2
        for factor in all_factors
    )
    return math.sqrt(squared_sum)


def build_visualization_payload(
    profile: AnnualRiskProfile,
    reports: Sequence[SectorDistanceReport],
    *,
    notes: str | None = None,
    coverage: CoverageSummary | None = None,
    eigenvalue_trends: dict[str, dict[str, float]] | None = None,
    coverage_alerts: dict[str, dict[str, float]] | None = None,
    stable_positions: dict[str, dict[str, float]] | None = None,
    annual_positions: dict[str, dict[str, float]] | None = None,
) -> VisualizationPayload:
    """
    Create a Three.js-friendly payload for a single year view.

    Parameters
    ----------
    profile : AnnualRiskProfile
        Annual risk profile containing ideal point, weights, and diagnostics.
    reports : Sequence[SectorDistanceReport]
        Sector distance reports for the year.
    notes : str | None, optional
        Additional notes to include in metadata.
    coverage : CoverageSummary | None, optional
        Coverage summary for data quality monitoring.
    eigenvalue_trends : dict[str, dict[str, float]] | None, optional
        Eigenvalue trends aggregated by time period.
    coverage_alerts : dict[str, dict[str, float]] | None, optional
        Coverage alerts for factors/sectors below thresholds.
    stable_positions : dict[str, dict[str, float]] | None, optional
        Long-term stable sector positions from compute_stable_sector_positions().
        Structure: {sector_id: {factor: value}}
        Example: {"XLK": {"factor_duration": -0.12, "factor_credit": -0.20}}
    annual_positions : dict[str, dict[str, float]] | None, optional
        Annual sector positions for this specific year from compute_annual_sector_positions().
        Structure: {sector_id: {factor: value}}
        Used as fallback if stable_positions is None.

    Returns
    -------
    VisualizationPayload
        Payload containing year, ideal point, sector data, and metadata.

    Notes
    -----
    The payload includes stable coordinate system data when available:
    - stable_center: Long-term "home" position for the sector
    - cloud_radius: Euclidean distance from current position to stable center
    - deviation_from_stable: Same as cloud_radius (clearer naming for frontend)

    These fields enable visualization of sector drift from their stable positions
    and help identify regime changes where sectors move significantly from their
    long-term averages.
    """
    sectors = []
    for report in reports:
        sector_data = {
            "sector": report.sector_id,
            "coordinates": report.coordinates,
            "distance": report.distance,
            "deltas": report.deltas,
            "recommended_weight": report.recommended_weight,
            "mahalanobis_distance": report.mahalanobis_distance,
        }

        # Add stable coordinate system fields
        stable_center: dict[str, float] | None = None
        cloud_radius: float | None = None
        deviation: float | None = None

        if stable_positions is not None and report.sector_id in stable_positions:
            stable_center = stable_positions[report.sector_id]

            # Determine current position: use annual_positions if available, else report.coordinates
            if annual_positions is not None and report.sector_id in annual_positions:
                current_position = annual_positions[report.sector_id]
            else:
                current_position = report.coordinates

            # Compute cloud radius (Euclidean distance from current to stable center)
            cloud_radius = _compute_cloud_radius(current_position, stable_center)
            deviation = cloud_radius

        sector_data["stable_center"] = stable_center
        sector_data["cloud_radius"] = cloud_radius
        sector_data["deviation_from_stable"] = deviation

        sectors.append(sector_data)

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
