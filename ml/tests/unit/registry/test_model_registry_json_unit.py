"""
Happy-path registration tests for ModelRegistry using JSON backend.

Validates that a serveable ONNX model with matching feature schema hash registers
successfully when a corresponding FeatureRegistry entry exists.

"""

from __future__ import annotations

from pathlib import Path

from ml.registry import DataRequirements
from ml.registry import FeatureRegistry
from ml.registry import FeatureRole
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.registry import compute_schema_hash
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.tests.builders import RegistryBuilder
from ml.tests.utils.model_artifacts import default_calibration
from ml.tests.utils.model_artifacts import default_output_schema


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

    fm = RegistryBuilder.feature_manifest(
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
    manifest = RegistryBuilder.model_manifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"f1": "float32", "f2": "float32"},
        feature_schema_hash=schema_hash,
        serveable=True,
        artifact_format="onnx",
        feature_set_id="feat_v1",
        output_schema=default_output_schema(),
        calibration=default_calibration(),
    )
    model_id = mreg.register_model(model_path=model_path, manifest=manifest, auto_deploy=False)
    # Force immediate save to avoid flakiness from batch timer
    if hasattr(mreg, "flush"):
        mreg.flush()  # Facade API
    elif hasattr(mreg, "_save_registry"):
        mreg._save_registry(immediate=True)  # Legacy API
    assert isinstance(model_id, str) and model_id
    # Registry JSON should be created
    assert (reg_dir / "registry.json").exists()
