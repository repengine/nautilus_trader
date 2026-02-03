from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole


REGISTRY_PATH_ENV_VAR = "ML_REGISTRY_PATH"


def get_default_registry_path() -> Path:
    """Return the default registry path honoring environment overrides."""
    env_value = os.getenv(REGISTRY_PATH_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return Path.home() / ".nautilus" / "ml" / "registry"


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



def compute_dataset_schema_hash(
    *,
    schema: Mapping[str, str],
    primary_keys: Sequence[str],
    ts_field: str,
    seq_field: str | None = None,
    pipeline_signature: str | None = None,
) -> str:
    """Compute stable schema hash used by dataset manifests and DataStore validation."""
    hash_builder = hashlib.sha256()

    for column_name in sorted(schema.keys()):
        dtype = schema[column_name]
        hash_builder.update(column_name.encode("utf-8"))
        hash_builder.update(b"::")
        hash_builder.update(dtype.encode("utf-8"))
        hash_builder.update(b"\n")

    hash_builder.update(b"|keys|")
    for key in sorted(str(item) for item in primary_keys):
        hash_builder.update(key.encode("utf-8"))
        hash_builder.update(b",")

    hash_builder.update(b"|ts|")
    hash_builder.update(ts_field.encode("utf-8"))

    if seq_field:
        hash_builder.update(b"|seq|")
        hash_builder.update(seq_field.encode("utf-8"))

    signature = pipeline_signature or ""
    hash_builder.update(b"|pipeline|")
    hash_builder.update(signature.encode("utf-8"))

    return hash_builder.hexdigest()

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
    feature_set_id: str | None = None,
    pipeline_signature: str | None = None,
    pipeline_version: str | None = None,
    decision_policy: str | None = None,
    decision_config: Mapping[str, Any] | None = None,
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
    decision_policy : str | None
        Optional decision policy adapter path for inference.
    decision_config : Mapping[str, Any] | None
        Optional decision adapter configuration (e.g., positive_class_index).

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
        feature_set_id=feature_set_id,
        pipeline_signature=pipeline_signature,
        pipeline_version=pipeline_version,
        decision_policy=decision_policy,
        decision_config=dict(decision_config or {}),
    )


def assert_features_compatible(
    manifest: ModelManifest,
    feature_names: list[str],
    feature_dtypes: list[str] | None = None,
) -> None:
    """
    Validate that the provided feature order (and optional dtypes) match the model
    manifest.

    Raises ValueError on mismatch.

    """
    expected_names = list(manifest.feature_schema.keys())
    if feature_names != expected_names:
        raise ValueError(
            "Feature names/order mismatch with model manifest: "
            f"expected={expected_names}, got={feature_names}",
        )
    if feature_dtypes is not None:
        expected_types = [manifest.feature_schema[n] for n in expected_names]
        if feature_dtypes != expected_types:
            raise ValueError(
                "Feature dtypes mismatch with model manifest: "
                f"expected={expected_types}, got={feature_dtypes}",
            )
