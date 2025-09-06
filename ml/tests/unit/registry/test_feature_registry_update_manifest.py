from __future__ import annotations

import json
from pathlib import Path

from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage


def test_update_manifest_persists_perf_digest_json(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    m = FeatureManifest(
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
        stage=FeatureStage.CANDIDATE,
    )
    fid = reg.register_feature_set(m)

    reg.update_manifest(fid, perf_digest={"pr_auc": 0.71, "logloss": 0.59})

    # Reload registry and verify persistence
    reg2 = FeatureRegistry(tmp_path)
    info = reg2.get_feature_set(fid)
    assert info is not None
    assert abs(info.manifest.perf_digest.get("pr_auc", 0.0) - 0.71) < 1e-12
    assert abs(info.manifest.perf_digest.get("logloss", 1.0) - 0.59) < 1e-12

