"""Shared helpers for deterministic test model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from typing import Mapping

from ml.registry import FeatureRegistry
from ml.tests.builders import RegistryBuilder


_DEFAULT_OUTPUT_SCHEMA: dict[str, Any] = {
    "kind": "binary_proba",
    "shape": [None, 1],
}
_DEFAULT_CALIBRATION: dict[str, Any] = {
    "kind": "platt",
    "params": {"coef": 1.0},
}


def default_output_schema() -> dict[str, Any]:
    """Return a copy of the canonical strict-valid output schema payload."""
    return dict(_DEFAULT_OUTPUT_SCHEMA)


def default_calibration() -> dict[str, Any]:
    """Return a copy of the canonical strict-valid calibration payload."""
    return {
        "kind": str(_DEFAULT_CALIBRATION["kind"]),
        "params": dict(_DEFAULT_CALIBRATION["params"]),
    }


def register_feature_set_for_schema(
    *,
    registry_path: Path,
    schema_hash: str,
    feature_set_id: str = "",
) -> str:
    """Register and return a feature set linked to a model schema hash.

    Args:
        registry_path: Directory used by the local feature registry.
        schema_hash: Feature schema hash that must match the model manifest.
        feature_set_id: Optional explicit feature-set identifier. When empty, the
            registry assigns a canonical identifier.

    Returns:
        Registered feature-set identifier.
    """

    feature_registry = FeatureRegistry(registry_path)
    feature_manifest = RegistryBuilder.feature_manifest(
        feature_set_id=feature_set_id,
        schema_hash=schema_hash,
    )
    return feature_registry.register_feature_set(feature_manifest)


def write_stub_onnx_artifact(
    model_path: Path,
    *,
    content: bytes = b"mock-onnx",
    output_schema: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
) -> Path:
    """Write a stub ONNX artifact and policy-compatible sidecar metadata.

    Args:
        model_path: Target path for the ONNX artifact.
        content: Raw bytes to write as model payload.
        output_schema: Optional output schema override written to sidecar metadata.
        calibration: Optional calibration override written to sidecar metadata.

    Returns:
        The artifact path that was written.
    """

    model_path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    sidecar_payload: dict[str, Any] = {
        "artifact_sha256_digest": digest,
        "output_schema": (
            dict(output_schema)
            if output_schema is not None
            else dict(_DEFAULT_OUTPUT_SCHEMA)
        ),
    }
    if calibration is not None:
        sidecar_payload["calibration"] = dict(calibration)
    model_path.with_suffix(".meta.json").write_text(
        json.dumps(sidecar_payload),
        encoding="utf-8",
    )
    return model_path


def ensure_strict_onnx_sidecar(
    model_path: Path,
    *,
    output_schema: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
) -> Path:
    """Attach strict-valid sidecar metadata to an existing ONNX artifact.

    Args:
        model_path: Existing ONNX artifact path.
        output_schema: Optional output schema override for sidecar metadata.
        calibration: Optional calibration override for sidecar metadata.

    Returns:
        The artifact path with refreshed strict-valid sidecar metadata.
    """

    return write_stub_onnx_artifact(
        model_path=model_path,
        content=model_path.read_bytes(),
        output_schema=output_schema,
        calibration=calibration,
    )


__all__ = [
    "default_calibration",
    "default_output_schema",
    "ensure_strict_onnx_sidecar",
    "register_feature_set_for_schema",
    "write_stub_onnx_artifact",
]
