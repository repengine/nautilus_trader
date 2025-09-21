"""
Registry operations tasks (feature/model promotion, artifact updates).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ml.config.constants import Versions
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.registry.base import DataRequirements
from ml.registry.dataclasses import QualityGate
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole


@dataclass(slots=True, frozen=True)
class FeaturePromotionGate:
    """
    Typed representation of a feature validation gate.
    """

    metric_name: str
    threshold: float
    comparison: str = "gte"
    required: bool = True

    def to_quality_gate(self) -> QualityGate:
        return QualityGate(
            metric_name=self.metric_name,
            threshold=self.threshold,
            comparison=self.comparison,
            required=self.required,
        )


def register_default_feature_set(
    registry_path: Path,
    *,
    name: str = "default",
    version: str | None = None,
    role: FeatureRole = FeatureRole.STUDENT,
    data_requirements: DataRequirements = DataRequirements.L1_ONLY,
) -> str:
    """
    Register the default :class:`FeatureConfig` manifest in the registry.
    """
    engineer = FeatureEngineer(FeatureConfig())
    manifest = engineer.generate_feature_manifest(
        name=name,
        version=version or Versions.DEFAULT_MANIFEST_VERSION,
        role=role,
        data_requirements=data_requirements,
    )
    registry = FeatureRegistry(registry_path)
    return registry.register_feature_set(manifest)


def promote_feature_set(
    registry_path: Path,
    *,
    feature_set_id: str,
    gates: Sequence[FeaturePromotionGate],
) -> bool:
    """
    Validate and promote a feature set using typed gates.
    """
    registry = FeatureRegistry(registry_path)
    return registry.validate_and_promote(
        feature_set_id,
        [gate.to_quality_gate() for gate in gates],
    )


def deprecate_feature_set(
    registry_path: Path,
    *,
    feature_set_id: str,
    reason: str | None = None,
) -> None:
    """
    Deprecate a feature set in the registry.
    """
    registry = FeatureRegistry(registry_path)
    registry.deprecate(feature_set_id, reason=reason)


__all__ = [
    "FeaturePromotionGate",
    "deprecate_feature_set",
    "promote_feature_set",
    "register_default_feature_set",
]
