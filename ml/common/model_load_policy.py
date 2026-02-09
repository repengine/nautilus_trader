"""
Direct model-path loading policy helpers.

This module applies policy-driven compatibility and integrity checks for
actor direct-path model loads, while preserving permissive defaults.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.common.metrics import registry_compatibility_migration_bypass_total
from ml.common.metrics import registry_unsigned_artifact_override_total
from ml.common.model_sidecar import extract_artifact_digest
from ml.common.model_sidecar import extract_inference_metadata
from ml.common.model_sidecar import resolve_model_sidecar_metadata
from ml.common.output_semantics import validate_output_semantics
from ml.config.policy import RegistryCompatibilityPolicyConfig
from ml.config.registry import RegistryPolicyConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DirectModelLoadPolicyResult:
    """
    Result of applying compatibility/integrity policy to direct path loads.

    Parameters
    ----------
    expected_digest : str | None
        Normalized expected SHA-256 artifact digest when available.
    strict_integrity : bool
        Whether digest mismatch should fail closed in secure loader paths.
    metadata : dict[str, Any]
        Updated model metadata after sidecar merge + semantics normalization.
    """

    expected_digest: str | None
    strict_integrity: bool
    metadata: dict[str, Any]


def apply_direct_model_load_policy(
    *,
    model_path: Path,
    model_metadata: Mapping[str, Any] | None = None,
    model_id: str | None = None,
    env: Mapping[str, str] | None = None,
    context: str = "direct_model_load",
) -> DirectModelLoadPolicyResult:
    """
    Apply direct-path compatibility/integrity policy and return load settings.

    Parameters
    ----------
    model_path : Path
        Path to the model artifact being loaded.
    model_metadata : Mapping[str, Any] | None, optional
        Existing metadata payload to enrich and validate.
    model_id : str | None, optional
        Optional model identifier used for diagnostics and metric labels.
    env : Mapping[str, str] | None, optional
        Environment override source for policy resolution in tests.
    context : str, default "direct_model_load"
        Short operation label for warning/error context.

    Returns
    -------
    DirectModelLoadPolicyResult
        Resolved digest policy, integrity strictness, and normalized metadata.

    Raises
    ------
    ValueError
        If strict policy gates reject missing digest/output semantics.
    """
    policy = RegistryPolicyConfig.from_env(env=env).compatibility_policy
    metadata = dict(model_metadata) if model_metadata is not None else {}

    sidecar = resolve_model_sidecar_metadata(model_path)
    if sidecar is not None:
        _merge_sidecar_semantics_and_digest(metadata=metadata, sidecar=sidecar)

    expected_digest = _normalize_expected_digest(
        _coerce_str(metadata.get("artifact_sha256_digest")),
    )
    if expected_digest is not None:
        metadata["artifact_sha256_digest"] = expected_digest

    model_label = model_id or model_path.stem or model_path.name
    _enforce_direct_digest_policy(
        policy=policy,
        model_label=model_label,
        model_path=model_path,
        expected_digest=expected_digest,
        context=context,
    )
    _validate_direct_output_semantics(
        policy=policy,
        metadata=metadata,
        model_label=model_label,
        context=context,
    )

    # Digest mismatch should fail closed whenever digest metadata is available.
    strict_integrity = expected_digest is not None
    return DirectModelLoadPolicyResult(
        expected_digest=expected_digest,
        strict_integrity=strict_integrity,
        metadata=metadata,
    )


def _merge_sidecar_semantics_and_digest(
    *,
    metadata: dict[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    output_schema, calibration = extract_inference_metadata(sidecar)
    if output_schema is not None and "output_schema" not in metadata:
        metadata["output_schema"] = output_schema
    if calibration is not None and "calibration" not in metadata:
        metadata["calibration"] = calibration

    if "artifact_sha256_digest" in metadata:
        return
    digest = _normalize_expected_digest(extract_artifact_digest(sidecar))
    if digest is not None:
        metadata["artifact_sha256_digest"] = digest


def _validate_direct_output_semantics(
    *,
    policy: RegistryCompatibilityPolicyConfig,
    metadata: dict[str, Any],
    model_label: str,
    context: str,
) -> None:
    validation = validate_output_semantics(
        output_schema=metadata.get("output_schema"),
        calibration=metadata.get("calibration"),
        require_output_semantics=policy.require_output_semantics,
    )
    if validation.is_valid:
        if validation.normalized_output_schema is not None:
            metadata["output_schema"] = validation.normalized_output_schema
        if validation.normalized_calibration is not None:
            metadata["calibration"] = validation.normalized_calibration
        return

    _handle_compatibility_violation(
        policy=policy,
        message=(
            f"Output semantics validation failed during {context} for model "
            f"{model_label}: {'; '.join(validation.errors)}"
        ),
        reason="output_semantics_validation_failed",
        model_label=model_label,
        strict_gate=(policy.strict_model_compatibility or policy.require_output_semantics),
    )


def _enforce_direct_digest_policy(
    *,
    policy: RegistryCompatibilityPolicyConfig,
    model_label: str,
    model_path: Path,
    expected_digest: str | None,
    context: str,
) -> None:
    if expected_digest is not None:
        return

    message = (
        f"No SHA-256 digest available for {model_label} ({model_path.name}) during "
        f"{context}; artifact integrity verification is unavailable"
    )

    if policy.allow_unsigned_artifacts:
        registry_unsigned_artifact_override_total.labels(
            model_id=model_label,
            reason="missing_digest",
        ).inc()
        logger.warning("%s; allowing load due unsigned artifact override policy", message)
        return

    _handle_compatibility_violation(
        policy=policy,
        message=message,
        reason="missing_digest",
        model_label=model_label,
        strict_gate=policy.strict_model_compatibility,
    )


def _handle_compatibility_violation(
    *,
    policy: RegistryCompatibilityPolicyConfig,
    message: str,
    reason: str,
    model_label: str,
    strict_gate: bool,
) -> None:
    if not strict_gate:
        logger.warning(message)
        return

    if policy.allow_compatibility_migration_override:
        registry_compatibility_migration_bypass_total.labels(
            model_id=model_label,
            reason=reason,
        ).inc()
        logger.warning("%s; bypassed due compatibility migration override policy", message)
        return

    raise ValueError(message)


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _normalize_expected_digest(expected_digest: str | None) -> str | None:
    if expected_digest is None:
        return None
    normalized = expected_digest.strip()
    if not normalized:
        return None
    if normalized.lower().startswith("sha256:"):
        normalized = normalized.split(":", maxsplit=1)[1].strip()
    if not normalized:
        return None
    return normalized


__all__ = ["DirectModelLoadPolicyResult", "apply_direct_model_load_policy"]
