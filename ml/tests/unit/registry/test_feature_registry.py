#!/usr/bin/env python3

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from ml.registry.base import DataRequirements
from ml.registry.dataclasses import QualityGate
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage
from ml.registry.feature_registry import LocalFeatureRegistry
from ml.registry.feature_registry import compute_schema_hash


@given(
    names=st.lists(
        st.from_regex(r"[a-z_][a-z0-9_]{0,10}", fullmatch=True),
        min_size=1,
        max_size=8,
        unique=True,
    ),
)
def test_feature_registry_roundtrip(names: list[str]) -> None:
    dtypes = ["float32" for _ in names]
    signature = "sig_v1"
    schema_hash = compute_schema_hash(names, dtypes, signature)

    with tempfile.TemporaryDirectory() as td:
        reg = LocalFeatureRegistry(Path(td))

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
