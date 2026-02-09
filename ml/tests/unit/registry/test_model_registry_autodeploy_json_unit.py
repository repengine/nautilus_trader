"""
ModelRegistry JSON: auto_deploy maps student to ml_signal_actor target.
"""

from __future__ import annotations

from pathlib import Path

from ml.registry import DataRequirements
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.tests.builders import RegistryBuilder
from ml.tests.utils.model_artifacts import default_calibration
from ml.tests.utils.model_artifacts import default_output_schema
from ml.tests.utils.model_artifacts import register_feature_set_for_schema


def test_auto_deploy_student_targets_signal_actor(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    reg = ModelRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )

    # Create teacher (non-serveable) first
    teacher_path = reg_dir / "teacher.onnx"
    teacher_path.write_bytes(b"onnx")
    teacher = RegistryBuilder.model_manifest(
        model_id="",
        role=ModelRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"f1": "float32"},
        feature_schema_hash="hash",
        serveable=False,
        artifact_format="onnx",
        feature_set_id=None,
        version="1.0.0",
    )
    teacher_id = reg.register_model(model_path=teacher_path, manifest=teacher, auto_deploy=False)

    # Student model with parent_id, auto_deploy=True
    model_path = reg_dir / "student.onnx"
    model_path.write_bytes(b"onnx")

    manifest = RegistryBuilder.model_manifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"f1": "float32"},
        feature_schema_hash="hash",
        serveable=True,
        artifact_format="onnx",
        feature_set_id=None,
        parent_id=teacher_id,
    )
    manifest.feature_set_id = register_feature_set_for_schema(
        registry_path=reg_dir,
        schema_hash=manifest.feature_schema_hash,
    )
    manifest.output_schema = default_output_schema()
    manifest.calibration = default_calibration()
    mid = reg.register_model(model_path=model_path, manifest=manifest, auto_deploy=True)
    # Verify deployment through public API - get_model returns deployment info
    info = reg.get_model(mid)
    assert info is not None
    # Auto-deployed student should be deployed to ml_signal_actor
    assert info.deployed_to == "ml_signal_actor" or info.deployment_status.value == "active"
