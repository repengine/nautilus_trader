"""
Model sidecar metadata helpers.

Sidecar JSON files capture technical metadata produced during training/export.
These helpers load and normalize sidecar metadata for registry and inference
parity (output schema + calibration params).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def load_sidecar_metadata(path: Path) -> dict[str, Any] | None:
    """
    Load JSON sidecar metadata from a given path.

    Parameters
    ----------
    path : Path
        Sidecar file path.

    Returns
    -------
    dict[str, Any] | None
        Parsed metadata mapping or None when unavailable/invalid.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug(
            "Failed to read sidecar metadata from %s: %s",
            path,
            exc,
            exc_info=True,
        )
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def resolve_model_sidecar_metadata(model_path: Path) -> dict[str, Any] | None:
    """
    Resolve a sidecar metadata payload for a model artifact.

    This probes common sidecar naming patterns such as:
    - model.meta.json
    - model.onnx.meta.json
    - model.meta (legacy)

    Parameters
    ----------
    model_path : Path
        Path to the model artifact.

    Returns
    -------
    dict[str, Any] | None
        Parsed metadata mapping or None if no sidecar found.
    """
    candidates = [
        model_path.with_suffix(".meta.json"),
        model_path.with_suffix(model_path.suffix + ".meta.json"),
        model_path.with_suffix(".meta"),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        payload = load_sidecar_metadata(candidate)
        if payload is not None:
            return payload
    return None


def extract_inference_metadata(
    sidecar: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Extract output schema and calibration metadata from a sidecar payload.

    Parameters
    ----------
    sidecar : Mapping[str, Any]
        Sidecar metadata mapping.

    Returns
    -------
    tuple[dict[str, Any] | None, dict[str, Any] | None]
        Tuple of (output_schema, calibration) mappings when present.
    """
    output_schema: dict[str, Any] | None = None
    raw_schema = sidecar.get("output_schema")
    if isinstance(raw_schema, Mapping):
        output_schema = dict(raw_schema)

    calibration: dict[str, Any] | None = None
    raw_calibration = sidecar.get("calibration")
    if raw_calibration is None:
        raw_calibration = sidecar.get("calibration_params")
    if raw_calibration is None:
        raw_calibration = sidecar.get("calibration_config")
    if isinstance(raw_calibration, Mapping):
        calibration = dict(raw_calibration)
    else:
        kind = sidecar.get("calibrator_kind")
        params = sidecar.get("calibrator_params")
        if kind is not None or isinstance(params, Mapping):
            calibration = {}
            if kind is not None:
                calibration["kind"] = str(kind)
            if isinstance(params, Mapping):
                calibration["params"] = dict(params)

    return output_schema, calibration


__all__ = [
    "extract_inference_metadata",
    "load_sidecar_metadata",
    "resolve_model_sidecar_metadata",
]
