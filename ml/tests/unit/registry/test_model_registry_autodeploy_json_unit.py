"""
ModelRegistry JSON: auto_deploy maps student to ml_signal_actor target.
"""

from __future__ import annotations

from pathlib import Path

from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.model_registry import ModelRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


def test_auto_deploy_student_targets_signal_actor(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    reg = ModelRegistry(registry_path=reg_dir, persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir))

    # Create teacher (non-serveable) first
    teacher_path = reg_dir / "teacher.onnx"
    teacher_path.write_bytes(b"onnx")
    teacher = ModelManifest(
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

    manifest = ModelManifest(
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
    mid = reg.register_model(model_path=model_path, manifest=manifest, auto_deploy=True)
    reg._save_registry(immediate=True)  # type: ignore[attr-defined]
    # Deployment mapping contains student for ml_signal_actor
    deployments = getattr(reg, "_deployments", {})
    assert "ml_signal_actor" in deployments and mid in deployments["ml_signal_actor"]
