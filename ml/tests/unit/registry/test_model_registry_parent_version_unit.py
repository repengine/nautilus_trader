"""
ModelRegistry JSON: parent-child linkage and auto-version tests.
"""

from __future__ import annotations

from pathlib import Path

from ml.registry import DataRequirements
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.tests.builders import RegistryBuilder


def test_model_registry_parent_child_and_auto_version(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    reg = ModelRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )

    # Prepare dummy model files
    teacher_path = reg_dir / "teacher.onnx"
    student_path = reg_dir / "student.onnx"
    teacher_path.write_bytes(b"onnx")
    student_path.write_bytes(b"onnx")

    # Register teacher with explicit version
    t_manifest = RegistryBuilder.model_manifest(
        model_id="",
        role=ModelRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"f1": "float32"},
        feature_schema_hash="hash1",
        serveable=False,
        artifact_format="onnx",
        feature_set_id=None,
        version="1.0.0",
    )
    teacher_id = reg.register_model(model_path=teacher_path, manifest=t_manifest, auto_deploy=False)

    # Register student with parent, blank version (auto-increment patch)
    s_manifest = RegistryBuilder.model_manifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"f1": "float32"},
        feature_schema_hash="hash1",
        serveable=True,
        artifact_format="onnx",
        feature_set_id=None,
        parent_id=teacher_id,
        version="",
    )
    student_id = reg.register_model(model_path=student_path, manifest=s_manifest, auto_deploy=False)

    # Save and fetch - use flush() for facade, _save_registry for legacy
    if hasattr(reg, "flush"):
        reg.flush()  # Facade API
    elif hasattr(reg, "_save_registry"):
        reg._save_registry(immediate=True)  # Legacy API
    t_info = reg.get_model(teacher_id)
    s_info = reg.get_model(student_id)

    assert t_info is not None and s_info is not None
    # Parent should include child id
    assert student_id in t_info.manifest.children_ids
    # Student version auto-increments patch component when blank
    assert s_info.manifest.version.endswith(".1")
