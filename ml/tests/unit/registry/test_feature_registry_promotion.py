from __future__ import annotations

from pathlib import Path

from ml.registry.base import DataRequirements
from ml.registry.dataclasses import QualityGate
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage


def test_feature_registry_validate_and_promote(tmp_path: Path) -> None:
    freg = FeatureRegistry(tmp_path)
    manifest = FeatureManifest(
        feature_set_id="",
        name="test_features",
        version="1.0.0",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=["f1", "f2"],
        feature_dtypes=["float32", "float32"],
        schema_hash="abc123",
        pipeline_signature="sig",
        pipeline_version="1.0.0",
    )
    fid = freg.register_feature_set(manifest)
    # Inject performance metrics into manifest for promotion
    info = freg.get_feature_set(fid)
    assert info is not None
    info.manifest.perf_digest["pr_auc"] = 0.8
    info.manifest.perf_digest["logloss"] = 0.5
    gates = [
        QualityGate(metric_name="pr_auc", threshold=0.7, comparison="gte", required=True),
        QualityGate(metric_name="logloss", threshold=0.6, comparison="lte", required=True),
    ]
    ok = freg.validate_and_promote(fid, gates)
    assert ok is True
    promoted = freg.get_feature_set(fid)
    assert promoted is not None and promoted.manifest.stage == FeatureStage.PROD

