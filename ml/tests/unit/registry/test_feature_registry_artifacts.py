from __future__ import annotations

from pathlib import Path

from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.tests.builders import RegistryBuilder


def test_attach_artifact_persists_json(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    m = RegistryBuilder.feature_manifest(
        feature_set_id="",
        name="fs",
        version="1.0.0",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=["f1"],
        feature_dtypes=["float32"],
        schema_hash="abc",
        pipeline_signature="sig",
        pipeline_version="1",
    )
    fid = reg.register_feature_set(m)

    artifact_path = str(tmp_path / "report.md")
    reg.attach_artifact(fid, "dataset_report", artifact_path)

    reg2 = FeatureRegistry(tmp_path)
    info = reg2.get_feature_set(fid)
    assert info is not None
    assert info.artifacts.get("dataset_report") == artifact_path
