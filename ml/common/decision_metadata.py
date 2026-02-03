"""
Decision metadata normalization helpers.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any

from ml.schema import DecisionMetadataV1


__all__ = [
    "decision_metadata_from_model_metadata",
    "normalize_decision_metadata",
]


_HORIZON_KEYS: tuple[tuple[str, str], ...] = (
    ("horizon_ms", "ms"),
    ("horizon_seconds", "seconds"),
    ("horizon_minutes", "minutes"),
    ("prediction_horizon_ms", "ms"),
    ("prediction_horizon_seconds", "seconds"),
    ("prediction_horizon_minutes", "minutes"),
    ("label_horizon_ms", "ms"),
    ("label_horizon_minutes", "minutes"),
    ("target_horizon_ms", "ms"),
    ("target_horizon_minutes", "minutes"),
    ("horizon", "unknown"),
    ("prediction_horizon", "unknown"),
    ("label_horizon", "unknown"),
    ("target_horizon", "unknown"),
)

_LEGACY_DECISION_KEYS: set[str] = {
    "decision_policy",
    "decision_config",
    "label_name",
    "target",
    "target_name",
    "target_col",
    "target_column",
    "calibration_params",
    "calibration_config",
}
for _key, _unit in _HORIZON_KEYS:
    if _key != "horizon":
        _LEGACY_DECISION_KEYS.add(_key)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    value_attr = getattr(value, "value", None)
    if isinstance(value_attr, str):
        return value_attr
    return str(value)


def _first_value(stack: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> Any | None:
    for metadata in stack:
        for key in keys:
            if key in metadata and metadata[key] is not None:
                return metadata[key]
    return None


def _extract_policy(stack: Iterable[Mapping[str, Any]]) -> tuple[str | None, Mapping[str, Any] | None]:
    policy = _coerce_str(_first_value(stack, ("policy", "decision_policy")))
    cfg_raw = _first_value(stack, ("policy_config", "decision_config"))
    return policy, _as_mapping(cfg_raw)


def _extract_label(stack: Iterable[Mapping[str, Any]]) -> str | None:
    raw = _first_value(
        stack,
        (
            "label",
            "label_name",
            "target",
            "target_name",
            "target_col",
            "target_column",
        ),
    )
    return _coerce_str(raw)


def _extract_calibration(stack: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    raw = _first_value(stack, ("calibration", "calibration_params", "calibration_config"))
    return _as_mapping(raw)


def _extract_horizon(stack: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for metadata in stack:
        for key, unit in _HORIZON_KEYS:
            if key in metadata and metadata[key] is not None:
                raw = metadata[key]
                mapped = _as_mapping(raw)
                if mapped is not None:
                    return dict(mapped)
                return {"value": raw, "unit": unit}
    return None


def _build_model_lineage(
    model_metadata: Mapping[str, Any] | None,
    *,
    model_id: str | None,
    model_version: str | None,
    parent_id: str | None,
    role: str | None,
) -> dict[str, Any] | None:
    lineage: dict[str, Any] = {}
    if model_metadata:
        model_id = model_id or _coerce_str(model_metadata.get("model_id"))
        model_version = model_version or _coerce_str(model_metadata.get("version"))
        parent_id = parent_id or _coerce_str(model_metadata.get("parent_id"))
        role = role or _coerce_str(model_metadata.get("role"))
    if model_id is not None:
        lineage["model_id"] = model_id
    if model_version is not None:
        lineage["version"] = model_version
    if parent_id is not None:
        lineage["parent_id"] = parent_id
    if role is not None:
        lineage["role"] = role
    return lineage or None


def _collect_metadata_stack(
    primary: Mapping[str, Any] | None,
    model_metadata: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    stack: list[Mapping[str, Any]] = []
    if primary:
        stack.append(primary)
        decision_cfg = _as_mapping(primary.get("decision_config"))
        if decision_cfg:
            stack.append(decision_cfg)
        training_cfg = _as_mapping(primary.get("training_config"))
        if training_cfg:
            stack.append(training_cfg)
    if model_metadata:
        stack.append(model_metadata)
        decision_cfg = _as_mapping(model_metadata.get("decision_config"))
        if decision_cfg:
            stack.append(decision_cfg)
        training_cfg = _as_mapping(model_metadata.get("training_config"))
        if training_cfg:
            stack.append(training_cfg)
    return stack


def normalize_decision_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    model_metadata: Mapping[str, Any] | None = None,
    model_id: str | None = None,
    model_version: str | None = None,
    parent_id: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """
    Normalize decision metadata into the v1 payload schema.

    Requires an explicit decision-metadata payload. When metadata is omitted,
    the payload is derived from model metadata (manifest/config) only.
    """
    if metadata is None:
        if model_metadata is None:
            raise ValueError("decision_metadata is required")
        stack = _collect_metadata_stack(None, model_metadata)
        policy, policy_config = _extract_policy(stack)
        label = _extract_label(stack)
        horizon = _extract_horizon(stack)
        calibration = _extract_calibration(stack)
        lineage = _build_model_lineage(
            model_metadata,
            model_id=model_id,
            model_version=model_version,
            parent_id=parent_id,
            role=role,
        )
        payload = DecisionMetadataV1(
            policy=policy,
            policy_config=policy_config,
            horizon=horizon,
            label=label,
            calibration=calibration,
            model_lineage=lineage,
        ).to_payload()
        if "version" not in payload:
            payload["version"] = "v1"
        return payload

    if not isinstance(metadata, Mapping):
        raise ValueError("decision_metadata must be a mapping")
    if "decision_metadata" in metadata:
        raise ValueError("decision_metadata must be provided directly, not nested")

    legacy_keys = _LEGACY_DECISION_KEYS.intersection(metadata.keys())
    if legacy_keys:
        raise ValueError(
            "decision_metadata contains legacy fields; provide explicit v1 payload",
        )

    payload = dict(metadata)
    if "version" not in payload:
        payload["version"] = "v1"

    lineage = _build_model_lineage(
        model_metadata,
        model_id=model_id,
        model_version=model_version,
        parent_id=parent_id,
        role=role,
    )
    if "model_lineage" not in payload and lineage:
        payload["model_lineage"] = lineage
    return payload


def decision_metadata_from_model_metadata(
    model_metadata: Mapping[str, Any] | None,
    *,
    model_id: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """
    Build decision metadata payload from model metadata only.
    """
    return normalize_decision_metadata(
        None,
        model_metadata=model_metadata,
        model_id=model_id,
        model_version=model_version,
    )
