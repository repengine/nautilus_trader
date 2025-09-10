"""
FeatureRegistry JSON backend functional tests.

Covers register/get, promote, resolve_by_schema_hash, and validate_and_promote with
basic quality gate conditions. Focus on functional outcomes.

"""

from __future__ import annotations

from pathlib import Path

from ml.registry.base import DataRequirements
from ml.registry.dataclasses import QualityGate
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage
from ml.registry.feature_registry import compute_schema_hash
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


def _mk_manifest(schema_hash: str) -> FeatureManifest:
    return FeatureManifest(
        feature_set_id="fs1",
        name="features",
        version="1.0.0",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=["f1", "f2"],
        feature_dtypes=["float32", "float32"],
        schema_hash=schema_hash,
        pipeline_signature="sig",
        pipeline_version="1",
        parity_tolerance=1e-10,
    )


def test_feature_registry_register_get_promote(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    reg = FeatureRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )

    schema_hash = compute_schema_hash(["f1", "f2"], ["float32", "float32"], "sig")
    fid = reg.register_feature_set(_mk_manifest(schema_hash))
    info = reg.get_feature_set(fid)
    assert info is not None
    assert info.manifest.schema_hash == schema_hash

    # Promote to PROD and verify stage
    reg.promote(fid, FeatureStage.PROD)
    assert reg.get_feature_set(fid).manifest.stage == FeatureStage.PROD  # type: ignore[union-attr]

    # Resolve by schema hash
    matches = reg.resolve_by_schema_hash(schema_hash)
    assert any(fi.manifest.feature_set_id == fid for fi in matches)


def test_validate_and_promote_with_quality_gates(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    reg = FeatureRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )

    schema_hash = compute_schema_hash(["f1"], ["float32"], "sig2")
    m = _mk_manifest(schema_hash)
    m.feature_set_id = "fsX"
    # Provide digests so gates can read values
    m.perf_digest = {"p99_latency_ms": 3.5}
    m.parity_digest = {"max_difference": 1e-12}
    m.constraints = {"min_bars_warmup": 20}
    reg.register_feature_set(m)

    gates = [
        QualityGate(metric_name="p99_latency_ms", threshold=5.0, comparison="lte", required=True),
        QualityGate(metric_name="max_difference", threshold=1e-10, comparison="lte", required=True),
    ]
    ok = reg.validate_and_promote("fsX", gates)
    assert ok is True
    assert reg.get_feature_set("fsX").manifest.stage == FeatureStage.PROD  # type: ignore[union-attr]
