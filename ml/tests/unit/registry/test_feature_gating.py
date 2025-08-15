from __future__ import annotations

from pathlib import Path

from ml.registry.base import DataRequirements
from ml.registry.dataclasses import QualityGate
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash


def _manifest_with_digests(names: list[str], dtypes: list[str]) -> FeatureManifest:
    sig = "sig"
    return FeatureManifest(
        feature_set_id="",
        name="fs",
        version="1.0.0",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=names,
        feature_dtypes=dtypes,
        schema_hash=compute_schema_hash(names, dtypes, sig),
        pipeline_signature=sig,
        pipeline_version="1.0.0",
        parity_digest={"max_difference": 1e-12, "tolerance": 1e-10},
        perf_digest={"p99_feature_ms": 0.4},
    )


def test_validate_and_promote(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    names = ["a", "b"]
    dtypes = ["float32", "float32"]
    m = _manifest_with_digests(names, dtypes)
    fid = reg.register_feature_set(m)

    gates = [
        QualityGate(metric_name="tolerance", threshold=1e-10, comparison="gte", required=True),
        QualityGate(metric_name="max_difference", threshold=1e-10, comparison="lte", required=True),
        QualityGate(metric_name="p99_feature_ms", threshold=0.5, comparison="lte", required=True),
    ]
    ok = reg.validate_and_promote(fid, gates)
    assert ok is True
    got = reg.get_feature_set(fid)
    assert got is not None and got.manifest.stage.value == "prod"
