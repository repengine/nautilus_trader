#!/usr/bin/env python3
"""
Consolidated Feature Registry Tests.

This file consolidates tests from:
- ml/tests/unit/registry/test_feature_registry.py (original)
- ml/tests/unit/registry/test_feature_registry_basic.py (merged)

Consolidation performed on 2025-08-25.

"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.registry.base import DataRequirements
from ml.registry.dataclasses import QualityGate
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage
from ml.registry.feature_registry import compute_schema_hash


@given(
    names=st.lists(
        st.from_regex(r"[a-z_][a-z0-9_]{0,10}", fullmatch=True),
        min_size=1,
        max_size=8,
        unique=True,
    ),
)
@pytest.mark.property
@pytest.mark.parallel_safe
@pytest.mark.unit
def test_feature_registry_roundtrip(names: list[str]) -> None:
    dtypes = ["float32" for _ in names]
    signature = "sig_v1"
    schema_hash = compute_schema_hash(names, dtypes, signature)

    with tempfile.TemporaryDirectory() as td:
        reg = FeatureRegistry(Path(td))

        manifest = FeatureManifest(
            feature_set_id="",
            name="student_features",
            version="1.0.0",
            role=FeatureRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            feature_names=names,
            feature_dtypes=dtypes,
            schema_hash=schema_hash,
            pipeline_signature=signature,
            pipeline_version="1.0.0",
        )

        fid = reg.register_feature_set(manifest)
        got = reg.get_feature_set(fid)
        assert got is not None
        assert got.manifest.schema_hash == schema_hash
        assert got.manifest.role == FeatureRole.STUDENT
        assert got.manifest.stage == FeatureStage.CANDIDATE

        # Promote with gates
        ok = reg.validate_and_promote(
            fid,
            gates=[QualityGate("p99_latency_ms", 5.0, comparison="lte", required=True)],
        )
        # No perf_digest, so gate fails and stage remains candidate
        assert ok is False
        feature_set = reg.get_feature_set(fid)
        assert feature_set is not None
        assert feature_set.manifest.stage == FeatureStage.CANDIDATE

        # Add perf and re-promote
        info = reg._features[fid]
        info.manifest.perf_digest["p99_latency_ms"] = 0.4
        assert (
            reg.validate_and_promote(fid, [QualityGate("p99_latency_ms", 1.0, "lte", True)]) is True
        )
        feature_set = reg.get_feature_set(fid)
        assert feature_set is not None
        assert feature_set.manifest.stage == FeatureStage.PROD


# =================================================================================================
# Tests merged from test_feature_registry_basic.py
# =================================================================================================


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
