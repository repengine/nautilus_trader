from __future__ import annotations

from pathlib import Path

import polars as pl

from ml.data.feature_manifest_export import FeatureExportConfig
from ml.data.feature_manifest_export import export_feature_manifest
from ml.data.feature_manifest_export import infer_feature_columns
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.tests.builders import DataBuilder


def test_export_feature_manifest_roundtrip(tmp_path: Path) -> None:
    # Use DataBuilder for consistent test data generation
    timestamps = DataBuilder.time_series(n_points=3, start_time=1, interval_ns=1)
    feature_data = DataBuilder.feature_data(n_samples=3, n_features=2)
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "instrument_id": ["SPY", "SPY", "SPY"],
            "time_index": [0, 1, 2],
            "f1": feature_data[:, 0],
            "f2": feature_data[:, 1],
            "y": [0, 1, 0],
        },
    )
    feature_names = infer_feature_columns(df)
    cfg = FeatureExportConfig(
        registry_path=tmp_path / "features",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
    )
    fid = export_feature_manifest(
        feature_names=feature_names,
        feature_dtypes=["float32"] * len(feature_names),
        flags={"include_macro": True},
        cfg=cfg,
    )
    assert isinstance(fid, str) and fid
    freg = FeatureRegistry(tmp_path / "features")
    info = freg.get_feature_set(fid)
    assert info is not None
    assert info.manifest.feature_names == feature_names
    assert info.manifest.capability_flags.get("include_macro") is True
