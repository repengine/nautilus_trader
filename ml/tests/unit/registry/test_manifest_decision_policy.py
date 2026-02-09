from __future__ import annotations

from pathlib import Path

from ml.registry import DataRequirements
from ml.registry import ModelManifest
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.tests.utils.model_artifacts import default_calibration
from ml.tests.utils.model_artifacts import default_output_schema
from ml.tests.utils.model_artifacts import register_feature_set_for_schema


def test_manifest_round_trip_with_decision_policy(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path)
    model_file = tmp_path / "student.onnx"
    model_file.write_bytes(b"\x08onnx-dummy")

    manifest = ModelManifest(
        model_id="student_demo",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="LightGBM",
        feature_schema={"f1": "float32", "f2": "float32"},
        feature_schema_hash="abc123",
        decision_policy="ml.actors.adapters.DynamicThresholdAdapter",
        decision_config={"alpha": 0.1},
    )
    manifest.feature_set_id = register_feature_set_for_schema(
        registry_path=tmp_path,
        schema_hash=manifest.feature_schema_hash,
    )
    manifest.output_schema = default_output_schema()
    manifest.calibration = default_calibration()

    reg.register_model(model_file, manifest, auto_deploy=False)

    info = reg.get_model("student_demo")
    assert info is not None
    assert info.manifest.decision_policy == "ml.actors.adapters.DynamicThresholdAdapter"
    assert info.manifest.decision_config == {"alpha": 0.1}
