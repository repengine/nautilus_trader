from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.features.materialize_cli import main as materialize_main
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash


def _register_feature_manifest(tmp_path: Path, feature_names: list[str]) -> tuple[Path, str]:
    reg_dir = tmp_path / "feature_registry"
    freg = FeatureRegistry(reg_dir)
    dtypes = ["float32"] * len(feature_names)
    schema = compute_schema_hash(feature_names, dtypes, pipeline_signature="sig_v1")
    manifest = FeatureManifest(
        feature_set_id="",
        name="fs",
        version="0.0.1",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=feature_names,
        feature_dtypes=dtypes,
        schema_hash=schema,
        pipeline_signature="sig_v1",
        pipeline_version="1",
    )
    fid = freg.register_feature_set(manifest)
    return reg_dir, fid


@pytest.mark.parallel_safe
def test_materialize_reorder_only(tmp_path: Path) -> None:
    feature_names = ["f1", "f2", "f3"]
    reg_dir, fid = _register_feature_manifest(tmp_path, feature_names)

    # Input CSV with shuffled columns
    df = pd.DataFrame(
        {
            "instrument_id": ["A", "A"],
            "time_index": [0, 1],
            "y": [1, 0],
            "f2": [0.2, 0.3],
            "f1": [1.0, 2.0],
            "f3": [5.0, 6.0],
        },
    )
    input_csv = tmp_path / "in.csv"
    output_csv = tmp_path / "out.csv"
    df.to_csv(input_csv, index=False)

    args = [
        "--feature_registry_dir",
        str(reg_dir),
        "--feature_set_id",
        fid,
        "--input_csv",
        str(input_csv),
        "--output_csv",
        str(output_csv),
        "--target_col",
        "y",
    ]
    rc = materialize_main(args)
    assert rc == 0

    out_df = pd.read_csv(output_csv)
    # Expect time_index, instrument_id (if present), then features ordered per manifest, then y
    expected_prefix = ["time_index", "instrument_id"]
    assert list(out_df.columns[:2]) == expected_prefix
    assert list(out_df.columns[2:5]) == feature_names
    assert out_df.columns[-1] == "y"
