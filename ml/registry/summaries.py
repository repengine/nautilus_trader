"""
Public helpers for summarizing model registry contents.

These utilities produce lightweight, serializable views of registry state for
user interfaces and APIs. They are cold-path only helpers that should be
invoked by dashboards, CLIs, or orchestration jobs to avoid duplicating
registry traversal logic.

Examples
--------
>>> from pathlib import Path
>>> from ml.registry.base import DataRequirements, ModelInfo, ModelManifest, ModelRole, DeploymentStatus
>>> manifest = ModelManifest(
...     model_id="model_v1",
...     role=ModelRole.INFERENCE,
...     data_requirements=DataRequirements.L1_ONLY,
...     architecture="lightgbm",
...     feature_schema={"f0": "float"},
...     feature_schema_hash="abc123",
... )
>>> info = ModelInfo(
...     manifest=manifest,
...     model_path=Path("./models/model_v1.onnx"),
...     deployment_status=DeploymentStatus.ACTIVE,
...     deployed_to=["ml_signal_actor"],
...     performance_history=[{"metric": "roc_auc", "value": 0.76}],
...     metadata={"notes": "production"},
... )
>>> summaries = build_model_summaries([info])
>>> summaries[0].model_id
'model_v1'

"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ml.registry.base import ModelInfo


@dataclass(frozen=True, slots=True)
class ModelSummary:
    """Immutable representation of a model registry entry."""

    model_id: str
    role: str
    version: str
    deployment_status: str
    deployed_to: tuple[str, ...]
    architecture: str | None
    feature_schema_hash: str | None
    performance_history: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]


def build_model_summaries(models: Sequence[ModelInfo] | None) -> tuple[ModelSummary, ...]:
    """
    Return immutable summaries for the provided model metadata.

    Parameters
    ----------
    models : Sequence[ModelInfo] | None
        Model registry entries to summarise; ``None`` yields an empty result.

    Returns
    -------
    tuple[ModelSummary, ...]
        Immutable summaries ready for serialization or API responses.
    """
    if not models:
        return ()

    summaries: list[ModelSummary] = []
    for info in models:
        summaries.append(
            ModelSummary(
                model_id=info.manifest.model_id,
                role=info.manifest.role.value,
                version=info.manifest.version,
                deployment_status=info.deployment_status.value,
                deployed_to=tuple(info.deployed_to),
                architecture=info.manifest.architecture,
                feature_schema_hash=info.manifest.feature_schema_hash,
                performance_history=tuple(dict(record) for record in info.performance_history),
                metadata=dict(info.metadata),
            ),
        )
    return tuple(summaries)


__all__ = [
    "ModelSummary",
    "build_model_summaries",
]
