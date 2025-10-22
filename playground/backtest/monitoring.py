"""
Monitoring utilities for Phase 3 backtest artefacts.

These helpers validate exported walk-forward metadata against the shared
configuration defaults and emit structlog alerts when discrepancies occur.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import structlog

from ml.config.playground import ThreeDRiskBacktestDefaults


LOGGER = structlog.get_logger(__name__)


@dataclass(slots=True)
class MetadataDrift:
    """
    Drift record describing a mismatch between metadata and defaults.

    Attributes
    ----------
    field : str
        Name of the field that differs.
    expected : object
        Expected/default value.
    actual : object
        Value observed in the metadata payload.
    """

    field: str
    expected: object
    actual: object


def validate_walk_forward_metadata(
    metadata: Mapping[str, object],
    *,
    defaults: ThreeDRiskBacktestDefaults | None = None,
) -> list[MetadataDrift]:
    """
    Compare walk-forward metadata with defaults and return drift records.
    """
    resolved_defaults = defaults or ThreeDRiskBacktestDefaults()
    drifts: list[MetadataDrift] = []

    risk_free = metadata.get("risk_free_rate")
    if not _float_matches(risk_free, resolved_defaults.risk_free_rate):
        drifts.append(MetadataDrift("risk_free_rate", resolved_defaults.risk_free_rate, risk_free))

    turnover = metadata.get("turnover_smoothing")
    expected_turnover = {
        "stable": resolved_defaults.stable_turnover_smoothing,
        "rolling": resolved_defaults.rolling_turnover_smoothing,
    }
    if not _mapping_matches(turnover, expected_turnover):
        drifts.append(MetadataDrift("turnover_smoothing", expected_turnover, turnover))

    liquidity = metadata.get("liquidity_config")
    expected_liquidity = resolved_defaults.liquidity_scaling.to_kwargs()
    if not _mapping_matches(liquidity, expected_liquidity):
        drifts.append(MetadataDrift("liquidity_config", expected_liquidity, liquidity))

    baseline = metadata.get("baseline_strategies")
    expected_baseline = list(resolved_defaults.baseline_strategies)
    if not _sequence_matches(baseline, expected_baseline):
        drifts.append(MetadataDrift("baseline_strategies", expected_baseline, baseline))

    return drifts


def log_walk_forward_metadata(metadata_path: Path) -> list[MetadataDrift]:
    """
    Load metadata from ``metadata_path`` and log drift warnings if present.
    """
    if not metadata_path.exists():
        LOGGER.warning("Walk-forward metadata file missing", path=str(metadata_path))
        return []

    try:
        metadata_text = metadata_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_text)
    except (OSError, ValueError) as exc:
        LOGGER.exception("Unable to read walk-forward metadata", path=str(metadata_path), error=str(exc))
        return []

    drifts = validate_walk_forward_metadata(metadata)
    if not drifts:
        LOGGER.info("Walk-forward metadata matches defaults", path=str(metadata_path))
    else:
        for drift in drifts:
            LOGGER.warning(
                "Walk-forward metadata drift detected",
                path=str(metadata_path),
                field=drift.field,
                expected=drift.expected,
                actual=drift.actual,
            )
    return drifts


def _float_matches(candidate: object, expected: float) -> bool:
    if isinstance(candidate, (int, float)):
        return abs(float(candidate) - expected) <= 1e-9
    return False


def _mapping_matches(candidate: object, expected: Mapping[str, object]) -> bool:
    if not isinstance(candidate, Mapping):
        return False
    for key, expected_value in expected.items():
        actual = candidate.get(key)
        if isinstance(expected_value, (int, float)):
            if not _float_matches(actual, float(expected_value)):
                return False
        else:
            if actual != expected_value:
                return False
    return True


def _sequence_matches(candidate: object, expected: Sequence[object]) -> bool:
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
        return False
    return list(candidate) == list(expected)
