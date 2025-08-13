from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.model_registry import LocalModelRegistry
from ml.registry.utils import build_feature_schema
from ml.registry.utils import build_student_manifest


class DummyOrtSession:
    def __init__(
        self,
        path: str,
        sess_options: Any | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self._path = path
        self._inputs = [type("I", (), {"name": "input"})()]

    def get_inputs(self) -> list[Any]:
        return self._inputs

    def run(self, _: Any, __: dict[str, Any]) -> list[np.ndarray[Any, np.dtype[np.float32]]]:
        # Return deterministic probabilities
        return [np.array([[0.25], [0.75]], dtype=np.float32)]


def test_registry_registers_teacher_and_student_and_loads_onnx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prepare registry
    reg_dir = tmp_path / "registry"
    registry = LocalModelRegistry(reg_dir)

    # Prepare dummy ONNX files
    teacher_path = reg_dir / "teacher.onnx"
    student_path = reg_dir / "student.onnx"
    teacher_path.write_bytes(b"dummy")
    student_path.write_bytes(b"dummy")

    # Monkeypatch onnxruntime in ml._imports used by LocalModelRegistry.load_model
    import ml._imports as imports

    class DummyOrtModule:
        InferenceSession = DummyOrtSession

        class SessionOptions:
            def __init__(self) -> None:
                pass

        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = 0

        class ExecutionMode:
            ORT_SEQUENTIAL = 0

    monkeypatch.setattr(imports, "HAS_ONNX", True, raising=False)
    monkeypatch.setattr(imports, "ort", DummyOrtModule(), raising=False)

    # Teacher manifest
    teacher_manifest = ModelManifest(
        model_id="teacher_v1",
        role=ModelRole.TEACHER,
        data_requirements=DataRequirements.L1_L2_L3,
        architecture="TFT",
        feature_schema={"f_l3": "float32", "f_l2": "float32"},
        feature_schema_hash="abc123",
        version="1.0.0",
    )

    teacher_id = registry.register_model(teacher_path, teacher_manifest)

    # Student manifest (L1-only)
    feature_names = ["bid", "ask"]
    dtypes = ["float32", "float32"]
    feature_schema = build_feature_schema(feature_names, dtypes)
    # Minimal hash; not tested here beyond presence
    feature_schema_hash = "deadbeef"

    student_manifest = build_student_manifest(
        model_id="student_v1",
        architecture="LightGBM",
        feature_schema=feature_schema,
        feature_schema_hash=feature_schema_hash,
        parent_id=teacher_id,
        performance_metrics={"inference_latency_ms": 1.2},
    )

    student_id = registry.register_model(student_path, student_manifest, auto_deploy=True)

    # Parent updated with child
    lineage = registry.get_model_lineage(teacher_id)
    # lineage = [teacher, student]
    assert len(lineage) >= 2
    assert lineage[0].manifest.model_id == teacher_id
    assert any(mi.manifest.model_id == student_id for mi in lineage[1:])

    # Load student ONNX via registry
    session = registry.load_model(student_id)
    assert session is not None
    outs = session.run(None, {"input": np.zeros((2, 2), dtype=np.float32)})
    assert isinstance(outs, list)
    assert outs[0].dtype == np.float32
