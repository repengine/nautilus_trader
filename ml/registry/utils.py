from __future__ import annotations

from typing import Any

from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole


def build_feature_schema(feature_names: list[str], dtypes: list[str]) -> dict[str, str]:
    """
    Build feature schema from names and data types.

    Parameters
    ----------
    feature_names : list[str]
        Feature names.
    dtypes : list[str]
        Data types for each feature.

    Returns
    -------
    dict[str, str]
        Feature schema mapping names to types.

    Raises
    ------
    ValueError
        If feature_names and dtypes have different lengths.

    """
    if len(feature_names) != len(dtypes):
        raise ValueError("feature_names and dtypes must be same length")
    return dict(zip(feature_names, dtypes))


def build_student_manifest(
    *,
    model_id: str,
    architecture: str,
    feature_schema: dict[str, str],
    feature_schema_hash: str,
    parent_id: str,
    performance_metrics: dict[str, float] | None = None,
    deployment_constraints: dict[str, Any] | None = None,
    version: str = "1.0.0",
) -> ModelManifest:
    """
    Build a student model manifest.

    Parameters
    ----------
    model_id : str
        Unique model identifier.
    architecture : str
        Model architecture (e.g., 'LightGBM').
    feature_schema : dict[str, str]
        Feature names to types mapping.
    feature_schema_hash : str
        Hash of the feature schema.
    parent_id : str
        Parent teacher model ID.
    performance_metrics : dict[str, float] | None
        Performance metrics.
    deployment_constraints : dict[str, Any] | None
        Deployment constraints.
    version : str
        Model version.

    Returns
    -------
    ModelManifest
        Configured student model manifest.

    """
    return ModelManifest(
        model_id=model_id,
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture=architecture,
        feature_schema=feature_schema,
        feature_schema_hash=feature_schema_hash,
        parent_id=parent_id,
        training_config={},
        performance_metrics=performance_metrics or {},
        deployment_constraints=deployment_constraints or {},
        version=version,
    )
