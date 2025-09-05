from __future__ import annotations

import json
from pathlib import Path

from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage
from ml.scripts.promote_features import main as promote_main


def test_promote_features_cli_promotes_on_gates(tmp_path: Path) -> None:
    # Prepare a minimal feature set in a temp registry
    reg_dir = tmp_path / "features"
    freg = FeatureRegistry(reg_dir)
    m = FeatureManifest(
        feature_set_id="",
        name="fs",
        version="1.0.0",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=["f1", "f2"],
        feature_dtypes=["float32", "float32"],
        schema_hash="abc",
        pipeline_signature="sig",
        pipeline_version="1",
    )
    fid = freg.register_feature_set(m)

    # Metrics JSON
    metrics = {"pr_auc": 0.75, "logloss": 0.55}
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    # Run CLI with inline gates
    args = [
        "--feature_registry_dir",
        str(reg_dir),
        "--feature_set_id",
        fid,
        "--metrics_json",
        str(metrics_path),
        "--gate",
        "pr_auc",
        "gte",
        "0.70",
        "required",
        "logloss",
        "lte",
        "0.60",
        "required",
    ]
    rc = promote_main(args)
    assert rc == 0

    # Reload registry to reflect CLI-side mutation
    freg2 = FeatureRegistry(reg_dir)
    info = freg2.get_feature_set(fid)
    assert info is not None
    assert info.manifest.stage == FeatureStage.PROD
