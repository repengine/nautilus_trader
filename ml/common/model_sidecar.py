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
    metadata_scope = _coerce_mapping(sidecar.get("inference")) or _coerce_mapping(
        sidecar.get("inference_metadata"),
    )
    raw_schema = sidecar.get("output_schema")
    if raw_schema is None and metadata_scope is not None:
        raw_schema = metadata_scope.get("output_schema")
    if isinstance(raw_schema, Mapping):
        output_schema = dict(raw_schema)

    calibration: dict[str, Any] | None = None
    raw_calibration = sidecar.get("calibration")
    if raw_calibration is None and metadata_scope is not None:
        raw_calibration = metadata_scope.get("calibration")
    if raw_calibration is None:
        raw_calibration = sidecar.get("calibration_params")
    if raw_calibration is None and metadata_scope is not None:
        raw_calibration = metadata_scope.get("calibration_params")
    if raw_calibration is None:
        raw_calibration = sidecar.get("calibration_config")
    if raw_calibration is None and metadata_scope is not None:
        raw_calibration = metadata_scope.get("calibration_config")
    if isinstance(raw_calibration, Mapping):
        calibration = dict(raw_calibration)
    else:
        kind = sidecar.get("calibrator_kind")
        params = sidecar.get("calibrator_params")
        if kind is None and metadata_scope is not None:
            kind = metadata_scope.get("calibrator_kind")
        if params is None and metadata_scope is not None:
            params = metadata_scope.get("calibrator_params")
        if kind is not None or isinstance(params, Mapping):
            calibration = {}
            if kind is not None:
                calibration["kind"] = str(kind)
            if isinstance(params, Mapping):
                calibration["params"] = dict(params)

    return output_schema, calibration


def extract_artifact_digest(sidecar: Mapping[str, Any]) -> str | None:
    """
    Extract SHA-256 artifact digest from sidecar payload variants.

    Parameters
    ----------
    sidecar : Mapping[str, Any]
        Sidecar metadata mapping.

    Returns
    -------
    str | None
        Digest string when present, otherwise ``None``.
    """
    direct_value = _first_non_empty_string(
        sidecar.get("artifact_sha256_digest"),
        sidecar.get("artifact_digest"),
        sidecar.get("sha256_digest"),
        sidecar.get("sha256"),
    )
    if direct_value is not None:
        return direct_value

    nested_keys = ("artifact", "integrity", "security", "manifest")
    for nested_key in nested_keys:
        nested = _coerce_mapping(sidecar.get(nested_key))
        if nested is None:
            continue
        nested_value = _first_non_empty_string(
            nested.get("artifact_sha256_digest"),
            nested.get("sha256_digest"),
            nested.get("sha256"),
            nested.get("digest"),
        )
        if nested_value is not None:
            return nested_value

    return None


def _coerce_mapping(value: object) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _first_non_empty_string(*values: object) -> str | None:
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return None


__all__ = [
    "extract_artifact_digest",
    "extract_inference_metadata",
    "load_sidecar_metadata",
    "resolve_model_sidecar_metadata",
]
