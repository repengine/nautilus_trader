from __future__ import annotations

from pathlib import Path

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole


def test_generate_and_register_manifest(tmp_path: Path) -> None:
    cfg = FeatureConfig()
    eng = FeatureEngineer(cfg)
    manifest = eng.generate_feature_manifest(
        name="default",
        version="1.0.0",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        pipeline_version="1.0.0",
    )
    # Names must match config order
    assert manifest.feature_names == cfg.get_feature_names()
    # Dtypes match float32 length
    assert manifest.feature_dtypes == ["float32"] * len(manifest.feature_names)
    # Register
    reg = FeatureRegistry(tmp_path)
    fid = reg.register_feature_set(manifest)
    got = reg.get_feature_set(fid)
    assert got is not None and got.manifest.schema_hash == manifest.schema_hash
