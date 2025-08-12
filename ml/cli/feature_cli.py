#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.registry.base import DataRequirements
from ml.registry.dataclasses import QualityGate
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import LocalFeatureRegistry


def cli_register_default(
    registry_path: str,
    name: str = "default",
    version: str = "1.0.0",
    role: str = "student",
    data_requirements: str = "l1_only",
) -> str:
    """
    Register the default FeatureConfig as a feature set.

    Returns the new feature_set_id.

    """
    role_enum = FeatureRole(role)
    req_enum = DataRequirements(data_requirements)
    eng = FeatureEngineer(FeatureConfig())
    manifest = eng.generate_feature_manifest(
        name=name,
        version=version,
        role=role_enum,
        data_requirements=req_enum,
    )
    reg = LocalFeatureRegistry(Path(registry_path))
    return reg.register_feature_set(manifest)


def cli_promote_with_gates(
    registry_path: str,
    feature_set_id: str,
    gates: list[dict[str, Any]],
) -> bool:
    """
    Validate and promote a feature set using simple dict-based gates.

    Each gate dict should include: {"metric_name": str, "threshold": float, "comparison": str}

    """
    reg = LocalFeatureRegistry(Path(registry_path))
    gate_objs = [
        QualityGate(
            metric_name=g["metric_name"],
            threshold=float(g["threshold"]),
            comparison=str(g.get("comparison", "gte")),
            required=bool(g.get("required", True)),
        )
        for g in gates
    ]
    return reg.validate_and_promote(feature_set_id, gate_objs)


def cli_deprecate(
    registry_path: str,
    feature_set_id: str,
    reason: str | None = None,
) -> None:
    """
    Deprecate a feature set in the registry.

    Parameters
    ----------
    registry_path : str
        Path to the feature registry.
    feature_set_id : str
        ID of the feature set to deprecate.
    reason : str | None
        Optional reason for deprecation.

    """
    reg = LocalFeatureRegistry(Path(registry_path))
    reg.deprecate(feature_set_id, reason=reason)
