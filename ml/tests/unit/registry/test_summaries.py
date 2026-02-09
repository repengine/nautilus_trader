from __future__ import annotations

from pathlib import Path

from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.summaries import build_model_summaries


def _make_model_info(*, model_id: str = "model_a") -> ModelInfo:
    manifest = ModelManifest(
        model_id=model_id,
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"f0": "float32"},
        feature_schema_hash="schema_hash",
        version="1.0.0",
    )
    return ModelInfo(
        manifest=manifest,
        model_path=Path("/tmp/model.onnx"),
        deployment_status=DeploymentStatus.ACTIVE,
        deployed_to=["ml_signal_actor"],
        performance_history=[{"metric": "roc_auc", "value": 0.81}],
        metadata={"owner": "qa"},
    )


def test_build_model_summaries_returns_empty_tuple_for_none_and_empty() -> None:
    assert build_model_summaries(None) == ()
    assert build_model_summaries([]) == ()


def test_build_model_summaries_copies_mutable_manifest_fields() -> None:
    model_info = _make_model_info()

    summaries = build_model_summaries([model_info])
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.model_id == "model_a"
    assert summary.role == ModelRole.INFERENCE.value
    assert summary.deployment_status == DeploymentStatus.ACTIVE.value
    assert summary.deployed_to == ("ml_signal_actor",)
    assert summary.performance_history[0]["metric"] == "roc_auc"
    assert summary.metadata["owner"] == "qa"

    model_info.deployed_to.append("secondary_target")
    model_info.performance_history[0]["value"] = 0.5
    model_info.metadata["owner"] = "ops"

    assert summary.deployed_to == ("ml_signal_actor",)
    assert summary.performance_history[0]["value"] == 0.81
    assert summary.metadata["owner"] == "qa"
