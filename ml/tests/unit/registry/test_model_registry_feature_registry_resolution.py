"""
ModelRegistry resolves FeatureRegistry when stored as a sibling directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.registry import ModelRegistry
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import compute_schema_hash
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.tests.builders import RegistryBuilder
from ml.tests.utils.model_artifacts import default_calibration
from ml.tests.utils.model_artifacts import default_output_schema


def test_model_registry_resolves_feature_registry_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ML_STRICT_FEATURE_PARITY", "1")

    registry_root = tmp_path / "registry"
    feature_dir = registry_root / "features"
    model_dir = registry_root / "models"

    feature_names = ["feature_a", "feature_b"]
    feature_dtypes = ["float32", "float32"]
    schema_hash = compute_schema_hash(feature_names, feature_dtypes, "pipeline_sig")
    feature_manifest = RegistryBuilder.feature_manifest(
        feature_set_id="feature_set_1",
        feature_names=feature_names,
        feature_dtypes=feature_dtypes,
        schema_hash=schema_hash,
        pipeline_signature="pipeline_sig",
        pipeline_version="1.0.0",
    )
    feature_registry = FeatureRegistry(
        feature_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=feature_dir),
    )
    feature_set_id = feature_registry.register_feature_set(feature_manifest)

    model_registry = ModelRegistry(
        registry_path=model_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=model_dir),
    )
    model_path = model_dir / "m.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"onnx")

    manifest = RegistryBuilder.model_manifest(
        model_id="model_1",
        architecture="LightGBM",
        feature_schema={"feature_a": "float32", "feature_b": "float32"},
        feature_schema_hash=schema_hash,
        feature_set_id=feature_set_id,
        pipeline_signature=None,
        pipeline_version=None,
        output_schema=default_output_schema(),
        calibration=default_calibration(),
    )
    model_registry.register_model(model_path=model_path, manifest=manifest, auto_deploy=False)

    info = model_registry.get_model("model_1")
    assert info is not None
    assert info.manifest.pipeline_signature == feature_manifest.pipeline_signature
