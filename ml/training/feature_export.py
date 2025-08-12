#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import LocalFeatureRegistry


def register_feature_set_from_engineer(
    registry_path: Path,
    name: str,
    version: str,
    role: FeatureRole,
    data_requirements: DataRequirements,
    feature_config: FeatureConfig | None = None,
    parity_report: dict[str, Any] | None = None,
    perf_report: dict[str, Any] | None = None,
    capability_flags: dict[str, bool] | None = None,
    constraints: dict[str, Any] | None = None,
) -> str:
    """
    Build a FeatureManifest from FeatureEngineer and register it locally.

    Returns the generated feature_set_id.

    """
    engineer = FeatureEngineer(feature_config or FeatureConfig())
    manifest = engineer.generate_feature_manifest(
        name=name,
        version=version,
        role=role,
        data_requirements=data_requirements,
        capability_flags=capability_flags,
        constraints=constraints,
        parity_tolerance=(parity_report or {}).get("tolerance", 0.0),
        parity_digest=parity_report or {},
        perf_digest=perf_report or {},
    )
    registry = LocalFeatureRegistry(registry_path)
    feature_set_id = registry.register_feature_set(manifest)
    return feature_set_id
