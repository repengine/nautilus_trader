from __future__ import annotations

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage
from ml.registry.feature_registry import compute_schema_hash


def _manifest_from(names: list[str], dtypes: list[str], sig: str = "sigA") -> FeatureManifest:
    schema_hash = compute_schema_hash(names, dtypes, sig)
    return FeatureManifest(
        feature_set_id="",
        name="testset",
        version="1.0.0",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=list(names),
        feature_dtypes=list(dtypes),
        schema_hash=schema_hash,
        pipeline_signature=sig,
        pipeline_version="0.1.0",
    )


@given(st.lists(st.text(min_size=1, max_size=12), min_size=1, max_size=16, unique=True))
def test_manifest_schema_hash_determinism(names: list[str]) -> None:
    dtypes = ["float32"] * len(names)
    m1 = _manifest_from(names, dtypes, "sigX")
    m2 = _manifest_from(list(names), list(dtypes), "sigX")
    assert m1.schema_hash == m2.schema_hash
    if len(names) > 1:
        swapped = list(reversed(names))
        m3 = _manifest_from(swapped, dtypes, "sigX")
        assert m1.schema_hash != m3.schema_hash


def test_registry_lifecycle(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    names = ["return_1", "rsi"]
    dtypes = ["float32", "float32"]
    m = _manifest_from(names, dtypes)
    fid = reg.register_feature_set(m, artifacts={"scaler": "scaler.npz"})
    got = reg.get_feature_set(fid)
    assert got is not None
    assert got.manifest.stage == FeatureStage.CANDIDATE
    reg.promote(fid, FeatureStage.PROD)
    got2 = reg.get_feature_set(fid)
    assert got2 is not None and got2.manifest.stage == FeatureStage.PROD
    reg.deprecate(fid, reason="replaced")
    got3 = reg.get_feature_set(fid)
    assert got3 is not None and got3.manifest.stage == FeatureStage.DEPRECATED
    # scrapped should still be retrievable but marked
    reg.scrap(fid)
    got4 = reg.get_feature_set(fid)
    assert got4 is not None and got4.manifest.stage == FeatureStage.SCRAPPED


def test_resolve_by_schema_hash(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    names = ["a", "b"]
    dtypes = ["float32", "float32"]
    m1 = _manifest_from(names, dtypes, sig="S")
    m2 = _manifest_from(names, dtypes, sig="S")
    id1 = reg.register_feature_set(m1)
    id2 = reg.register_feature_set(m2)
    matches = reg.resolve_by_schema_hash(m1.schema_hash)
    assert {x.manifest.feature_set_id for x in matches} == {id1, id2}
