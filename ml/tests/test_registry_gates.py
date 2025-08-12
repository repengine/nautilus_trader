from __future__ import annotations

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.model_registry import LocalModelRegistry


@given(
    role=st.sampled_from([ModelRole.TEACHER, ModelRole.STUDENT, ModelRole.INFERENCE]),
    req=st.sampled_from(
        [DataRequirements.L1_ONLY, DataRequirements.L1_L2, DataRequirements.L1_L2_L3],
    ),
    has_parent=st.booleans(),
)
def test_auto_deploy_gates(
    tmp_path: Path,
    role: ModelRole,
    req: DataRequirements,
    has_parent: bool,
) -> None:
    reg_dir = tmp_path / "reg"
    registry = LocalModelRegistry(reg_dir)
    # Ensure model files exist and are ONNX
    model_path = reg_dir / "model.onnx"
    model_path.write_bytes(b"dummy")

    parent_id = None
    # Optionally register a parent teacher
    if has_parent:
        parent_path = reg_dir / "parent.onnx"
        parent_path.write_bytes(b"dummy")
        parent_manifest = ModelManifest(
            model_id="t",
            role=ModelRole.TEACHER,
            data_requirements=DataRequirements.L1_L2_L3,
            architecture="TFT",
            feature_schema={},
            feature_schema_hash="h",
            version="1.0.0",
        )
        parent_id = registry.register_model(parent_path, parent_manifest)

    manifest = ModelManifest(
        model_id="m",
        role=role,
        data_requirements=req,
        architecture="LightGBM" if role != ModelRole.TEACHER else "TFT",
        feature_schema={"f": "float32"},
        feature_schema_hash="h2",
        parent_id=parent_id if role == ModelRole.STUDENT else None,
        version="1.0.0",
    )

    # Monkeypatch ort for load_model used by auto-deploy
    import ml._imports as imports

    class DummyOrtModule:
        class InferenceSession:  # type: ignore[no-redef]
            def __init__(self, *args, **kwargs) -> None:
                pass

        class SessionOptions:
            def __init__(self) -> None:
                pass

        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = 0

        class ExecutionMode:
            ORT_SEQUENTIAL = 0

    # Ensure HAS_ONNX true for auto-deploy load
    imports.HAS_ONNX = True  # type: ignore[assignment]
    imports.ort = DummyOrtModule()  # type: ignore[assignment]

    model_id = registry.register_model(model_path, manifest, auto_deploy=True)

    mi = registry._models[model_id]
    # Expected: auto-deploy only for student with L1_ONLY and parent
    should_deploy = role == ModelRole.STUDENT and req == DataRequirements.L1_ONLY and has_parent
    assert (mi.deployment_status == DeploymentStatus.ACTIVE) == should_deploy
    assert ("ml_signal_actor" in mi.deployed_to) == should_deploy
