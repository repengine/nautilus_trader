from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.registry.base import DataRequirements, ModelManifest, ModelRole
from ml.registry.model_registry import ModelRegistry


def test_register_non_serveable_teacher_artifact(tmp_path: Path) -> None:
    reg_dir = tmp_path / "models"
    reg = ModelRegistry(reg_dir)

    # Create a fake non-ONNX artifact (e.g., TorchScript or pickle)
    art_path = reg_dir / "teacher.pt"
    art_path.parent.mkdir(parents=True, exist_ok=True)
    art_path.write_bytes(b"fake_torchscript")

    manifest = ModelManifest(
        model_id="",  # let registry assign
        role=ModelRole.TEACHER,
        data_requirements=DataRequirements.HISTORICAL,
        architecture="TFT",
        feature_schema={"f1": "float32"},
        feature_schema_hash="abc123",
        serveable=False,
        artifact_format="torchscript",
    )

    model_id = reg.register_model(art_path, manifest)
    assert model_id

    # load_model should not load non-ONNX models
    assert reg.load_model(model_id) is None

    # get_artifact_path should return the path as a safe reference
    p = reg.get_artifact_path(model_id)
    assert p and p.exists() and p.suffix == ".pt"


def test_register_student_requires_onnx(tmp_path: Path) -> None:
    reg_dir = tmp_path / "models2"
    reg = ModelRegistry(reg_dir)

    bad_path = reg_dir / "student.pkl"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"pickle")

    manifest = ModelManifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="LightGBM",
        feature_schema={"f1": "float32"},
        feature_schema_hash="abc123",
        serveable=True,
        artifact_format="pickle",
    )

    try:
        reg.register_model(bad_path, manifest)
        assert False, "Expected ValueError for non-ONNX serveable model"
    except ValueError as e:
        assert "Only ONNX models" in str(e)
