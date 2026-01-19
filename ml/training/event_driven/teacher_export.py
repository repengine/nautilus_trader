"""
Validation helpers for streaming teacher exports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ml.common.security import calculate_file_sha256
from ml.registry import ModelRegistry


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class StreamingExportValidationResult:
    """Result of validating a streaming teacher export manifest."""

    model_id: str | None
    registry_root: Path | None
    artifact_path: Path | None
    errors: tuple[str, ...]

    def raise_for_errors(self) -> None:
        """Raise a RuntimeError when validation errors are present."""
        if self.errors:
            joined = "; ".join(self.errors)
            raise RuntimeError(f"Streaming export validation failed: {joined}")


def _load_manifest_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest payload must be a JSON object: {path}")
    return payload


def validate_streaming_teacher_export(
    manifest_path: Path,
    *,
    registry_root: Path | None = None,
    require_registry: bool = True,
) -> StreamingExportValidationResult:
    """
    Validate a streaming teacher export manifest against the model registry.

    Parameters
    ----------
    manifest_path : Path
        Path to the streaming manifest JSON emitted by the runner.
    registry_root : Path | None, optional
        Optional registry override path (defaults to manifest registry_root).
    require_registry : bool, default True
        Whether to require the model registry entry to exist.

    Returns
    -------
    StreamingExportValidationResult
        Validation result with collected errors.
    """
    payload = _load_manifest_payload(manifest_path)
    errors: list[str] = []

    cohort = payload.get("cohort_run")
    if not isinstance(cohort, dict):
        errors.append("manifest missing cohort_run")
        return StreamingExportValidationResult(None, None, None, tuple(errors))

    registry_payload = cohort.get("model_registry")
    if not isinstance(registry_payload, dict):
        errors.append("manifest missing cohort_run.model_registry")
        return StreamingExportValidationResult(None, None, None, tuple(errors))

    model_id = registry_payload.get("model_id")
    registry_root_str = registry_payload.get("registry_root")
    artifact_rel = registry_payload.get("artifact_rel_path")
    expected_sha = registry_payload.get("artifact_sha256")

    if not isinstance(model_id, str) or not model_id.strip():
        errors.append("model_id missing or invalid")
    if not isinstance(registry_root_str, str) or not registry_root_str.strip():
        errors.append("registry_root missing or invalid")
    if not isinstance(artifact_rel, str) or not artifact_rel.strip():
        errors.append("artifact_rel_path missing or invalid")

    if errors:
        return StreamingExportValidationResult(
            model_id if isinstance(model_id, str) else None,
            None,
            None,
            tuple(errors),
        )

    resolved_registry_root = (registry_root or Path(cast(str, registry_root_str))).expanduser()
    artifact_path = resolved_registry_root / cast(str, artifact_rel)

    if require_registry and not resolved_registry_root.exists():
        errors.append(f"registry_root not found: {resolved_registry_root}")

    if not artifact_path.exists():
        errors.append(f"artifact not found: {artifact_path}")
    elif isinstance(expected_sha, str) and expected_sha.strip():
        try:
            actual = calculate_file_sha256(artifact_path)
            if actual.lower() != expected_sha.lower():
                errors.append("artifact_sha256 mismatch")
        except Exception as exc:
            errors.append(f"artifact_sha256 check failed: {exc}")

    if require_registry and resolved_registry_root.exists() and isinstance(model_id, str):
        registry = ModelRegistry(resolved_registry_root)
        model_info = registry.get_model(model_id)
        if model_info is None:
            errors.append(f"model_id {model_id} not found in registry")
        else:
            try:
                registry._validate_registration_inputs(artifact_path, model_info.manifest)
            except Exception as exc:
                logger.debug("registry validation failed", exc_info=True)
                errors.append(f"registry validation failed: {exc}")

    return StreamingExportValidationResult(
        model_id=model_id if isinstance(model_id, str) else None,
        registry_root=resolved_registry_root,
        artifact_path=artifact_path,
        errors=tuple(errors),
    )


__all__ = [
    "StreamingExportValidationResult",
    "validate_streaming_teacher_export",
]
