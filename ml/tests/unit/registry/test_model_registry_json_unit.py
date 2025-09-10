"""
Happy-path registration tests for ModelRegistry using JSON backend.

Validates that a serveable ONNX model with matching feature schema hash registers
successfully when a corresponding FeatureRegistry entry exists.

"""

from __future__ import annotations

from pathlib import Path

from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash
from ml.registry.model_registry import ModelRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


def test_model_registry_register_happy_path(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    reg_dir.mkdir(parents=True, exist_ok=True)

    # Prepare FeatureRegistry with a schema
    freg = FeatureRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )
    feature_names = ["f1", "f2"]
    feature_dtypes = ["float32", "float32"]
    pipeline_sig = "sig_v1"
    schema_hash = compute_schema_hash(feature_names, feature_dtypes, pipeline_sig)

    fm = FeatureManifest(
        feature_set_id="feat_v1",
        name="test_features",
        version="1.0.0",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=feature_names,
        feature_dtypes=feature_dtypes,
        schema_hash=schema_hash,
        pipeline_signature=pipeline_sig,
        pipeline_version="1",
        parity_tolerance=1e-10,
    )
    freg.register_feature_set(fm)

    # Prepare a dummy ONNX model file
    model_path = reg_dir / "m.onnx"
    model_path.write_bytes(b"onnx")

    # Register model with matching schema hash
    mreg = ModelRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )
    manifest = ModelManifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"f1": "float32", "f2": "float32"},
        feature_schema_hash=schema_hash,
        serveable=True,
        artifact_format="onnx",
        feature_set_id="feat_v1",
    )
    model_id = mreg.register_model(model_path=model_path, manifest=manifest, auto_deploy=False)
    # Force immediate save to avoid flakiness from batch timer
    mreg._save_registry(immediate=True)  # type: ignore[attr-defined]
    assert isinstance(model_id, str) and model_id
    # Registry JSON should be created
    assert (reg_dir / "registry.json").exists()
