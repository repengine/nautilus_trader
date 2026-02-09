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
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml.registry.base import DataRequirements
from ml.registry.dataclasses import QualityGate
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureInfo
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import FeatureStage
from ml.registry.feature_registry import compute_schema_hash
from ml.registry.persistence import BackendType
from ml.tests.builders import RegistryBuilder


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

        manifest = RegistryBuilder.feature_manifest(
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


def test_register_feature_set_records_capability_flag_diff(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    base_manifest = RegistryBuilder.feature_manifest(
        feature_set_id="",
        name="streaming_teacher",
        version="1.0.0",
        role=FeatureRole.TEACHER,
        capability_flags={
            "include_macro": True,
            "include_l2": False,
        },
    )
    first_id = reg.register_feature_set(base_manifest)
    first_info = reg.get_feature_set(first_id)
    assert first_info is not None
    assert "capability_flags_diff" not in first_info.manifest.parity_digest

    updated_manifest = RegistryBuilder.feature_manifest(
        feature_set_id="",
        name="streaming_teacher",
        version="1.0.1",
        role=FeatureRole.TEACHER,
        capability_flags={
            "include_macro": True,
            "include_l2": True,
        },
    )
    second_id = reg.register_feature_set(updated_manifest)
    second_info = reg.get_feature_set(second_id)
    assert second_info is not None
    diff = second_info.manifest.parity_digest.get("capability_flags_diff")
    assert diff == {
        "include_l2": {
            "previous": False,
            "current": True,
        },
    }


# =================================================================================================
# Tests merged from test_feature_registry_basic.py
# =================================================================================================


def _manifest_from(names: list[str], dtypes: list[str], sig: str = "sigA"):
    schema_hash = compute_schema_hash(names, dtypes, sig)
    return RegistryBuilder.feature_manifest(
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


def test_validate_and_promote_handles_all_comparisons(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    manifest = RegistryBuilder.feature_manifest(
        feature_set_id="",
        name="comparison_gates",
        version="1.0.0",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=["f1"],
        feature_dtypes=["float32"],
        schema_hash="comparison_hash",
        pipeline_signature="sig",
        pipeline_version="v1",
        perf_digest={
            "gte_metric": 1.0,
            "lte_metric": 1.0,
            "gt_metric": 2.0,
            "lt_metric": 0.5,
            "eq_metric": 3.0,
            "unknown_metric": 10.0,
        },
    )
    feature_id = reg.register_feature_set(manifest)

    gates = [
        QualityGate(metric_name="gte_metric", threshold=1.0, comparison="gte", required=True),
        QualityGate(metric_name="lte_metric", threshold=1.1, comparison="lte", required=True),
        QualityGate(metric_name="gt_metric", threshold=1.0, comparison="gt", required=True),
        QualityGate(metric_name="lt_metric", threshold=1.0, comparison="lt", required=True),
        QualityGate(metric_name="eq_metric", threshold=3.0, comparison="eq", required=True),
        QualityGate(metric_name="missing_metric", threshold=0.0, comparison="gte", required=False),
        QualityGate(
            metric_name="unknown_metric", threshold=5.0, comparison="unsupported", required=False
        ),
    ]
    assert reg.validate_and_promote(feature_id, gates) is True
    info = reg.get_feature_set(feature_id)
    assert info is not None
    assert info.manifest.stage == FeatureStage.PROD


def test_update_manifest_raises_for_unknown_feature_set(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    with pytest.raises(KeyError, match="Unknown feature_set_id"):
        reg.update_manifest("missing", perf_digest={"p99": 1.0})


def test_scrap_is_noop_for_unknown_feature_set(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    reg.scrap("missing")
    assert reg.list_all() == []


def test_postgres_paths_handle_none_session_and_persist_through_helpers(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    reg.backend = BackendType.POSTGRES
    reg.persistence = SimpleNamespace(
        get_session=lambda: None,
        log_audit=lambda **_: None,
    )

    manifest = RegistryBuilder.feature_manifest(
        feature_set_id="postgres_feature",
        name="postgres_feature",
        version="1.0.0",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=["f1"],
        feature_dtypes=["float32"],
        schema_hash="postgres_hash",
        pipeline_signature="sig",
        pipeline_version="v1",
    )

    reg._load()
    reg._save()
    feature_id = reg.register_feature_set(manifest)
    reg.promote(feature_id, FeatureStage.STAGING)
    reg.deprecate(feature_id, reason="legacy")
    reg.update_manifest(feature_id, constraints={"warmup": 3.0})
    reg.scrap(feature_id)

    info = reg.get_feature_set(feature_id)
    assert info is not None
    assert info.manifest.stage == FeatureStage.SCRAPPED


def test_db_to_feature_info_and_health_snapshot_value_error_branch(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    db_feature = SimpleNamespace(
        feature_set_id="db_feature",
        name="db_feature",
        version="2.0.0",
        role=FeatureRole.INFERENCE_SUPPORT.value,
        data_requirements=DataRequirements.L1_L2.value,
        feature_names=["f1", "f2"],
        feature_dtypes=["float32", "int64"],
        schema_hash="db_hash",
        pipeline_signature="sig",
        pipeline_version="v2",
        capability_flags={"flag": True},
        constraints={"max_memory_mb": 32.0},
        parity_tolerance=0.01,
        parity_digest={"max_diff": 0.0},
        perf_digest={"latency_ms": 0.9},
        parent_feature_set_id=None,
        extra_metadata={"owner": "qa", "artifacts": {"report": "report.md"}},
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        last_modified=datetime(2024, 1, 2, tzinfo=UTC),
        stage=FeatureStage.STAGING.value,
    )

    info = reg._db_to_feature_info(db_feature)
    assert info.manifest.feature_set_id == "db_feature"
    assert info.manifest.stage == FeatureStage.STAGING
    assert info.artifacts == {"report": "report.md"}

    empty_reg = FeatureRegistry(tmp_path / "empty")
    assert empty_reg._health_snapshot() == (0, None)

    manifest = RegistryBuilder.feature_manifest(
        feature_set_id="",
        name="health_case",
        version="1.0.0",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=["f1"],
        feature_dtypes=["float32"],
        schema_hash="health_hash",
        pipeline_signature="sig",
        pipeline_version="v1",
    )
    reg.register_feature_set(manifest)
    with patch("builtins.max", side_effect=ValueError()):
        count, last_modified = reg._health_snapshot()
    assert count == 1
    assert last_modified is None


def test_save_feature_to_db_update_insert_and_error_paths(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    reg.backend = BackendType.POSTGRES
    feature_info = FeatureInfo(
        manifest=FeatureManifest(
            feature_set_id="feature_db_save",
            name="feature_db_save",
            version="1.0.0",
            role=FeatureRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            feature_names=["f1"],
            feature_dtypes=["float32"],
            schema_hash="save_hash",
            pipeline_signature="sig",
            pipeline_version="v1",
            stage=FeatureStage.CANDIDATE,
        ),
        artifacts={"artifact": "path"},
    )

    existing_row = SimpleNamespace()
    update_session = MagicMock()
    update_session.query.return_value.filter_by.return_value.first.return_value = existing_row
    reg.persistence = SimpleNamespace(get_session=lambda: update_session)
    reg._save_feature_to_db(feature_info)
    assert existing_row.name == "feature_db_save"
    assert existing_row.stage == FeatureStage.CANDIDATE.value
    update_session.commit.assert_called_once()
    update_session.close.assert_called_once()

    insert_session = MagicMock()
    insert_session.query.return_value.filter_by.return_value.first.return_value = None
    reg.persistence = SimpleNamespace(get_session=lambda: insert_session)
    reg._save_feature_to_db(feature_info)
    insert_session.add.assert_called_once()
    insert_session.commit.assert_called_once()
    insert_session.close.assert_called_once()

    failing_session = MagicMock()
    failing_session.query.return_value.filter_by.return_value.first.return_value = None
    failing_session.commit.side_effect = RuntimeError("db error")
    reg.persistence = SimpleNamespace(get_session=lambda: failing_session)
    with pytest.raises(RuntimeError, match="Failed to save feature to database: db error"):
        reg._save_feature_to_db(feature_info)
    failing_session.rollback.assert_called_once()
    failing_session.close.assert_called_once()


def test_postgres_load_error_resets_features_and_closes_session(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    reg.backend = BackendType.POSTGRES
    reg._features = {
        "stale_feature": FeatureInfo(
            manifest=FeatureManifest(
                feature_set_id="stale_feature",
                name="stale_feature",
                version="1.0.0",
                role=FeatureRole.STUDENT,
                data_requirements=DataRequirements.L1_ONLY,
                feature_names=["f1"],
                feature_dtypes=["float32"],
                schema_hash="stale_hash",
                pipeline_signature="sig",
                pipeline_version="v1",
            ),
            artifacts={},
        ),
    }

    session = MagicMock()
    session.query.return_value.all.side_effect = RuntimeError("load failed")
    reg.persistence = SimpleNamespace(get_session=lambda: session)

    reg._load()

    assert reg._features == {}
    session.close.assert_called_once()


def test_manifest_accessors_and_lineage_helpers_cover_missing_and_present_paths(
    tmp_path: Path,
) -> None:
    reg = FeatureRegistry(tmp_path)

    parent_manifest = RegistryBuilder.feature_manifest(
        feature_set_id="parent_feature",
        name="feature_lineage",
        version="1.0.0",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=["f_parent"],
        feature_dtypes=["float32"],
        schema_hash="parent_hash",
        pipeline_signature="sig",
        pipeline_version="v1",
    )
    child_manifest = RegistryBuilder.feature_manifest(
        feature_set_id="child_feature",
        name="feature_lineage_child",
        version="1.0.1",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=["f_child"],
        feature_dtypes=["float32"],
        schema_hash="child_hash",
        pipeline_signature="sig",
        pipeline_version="v1",
        parent_feature_set_id="parent_feature",
    )

    parent_id = reg.register_feature_set(parent_manifest)
    child_id = reg.register_feature_set(child_manifest)

    assert reg.get_feature_manifest("missing") is None
    parent_loaded = reg.get_feature_manifest(parent_id)
    assert parent_loaded is not None
    assert parent_loaded.feature_set_id == parent_id

    listed = reg.list_by_role(FeatureRole.STUDENT)
    assert {item.manifest.feature_set_id for item in listed} == {child_id}

    all_ids = {item.manifest.feature_set_id for item in reg.list_all()}
    assert all_ids == {parent_id, child_id}

    assert reg.get_lineage("missing") == []
    child_lineage = reg.get_lineage(child_id)
    assert [manifest.feature_set_id for manifest in child_lineage] == [parent_id, child_id]


def test_update_manifest_merges_parity_stage_and_attach_artifact_helpers(tmp_path: Path) -> None:
    reg = FeatureRegistry(tmp_path)
    manifest = RegistryBuilder.feature_manifest(
        feature_set_id="feature_update_paths",
        name="feature_update_paths",
        version="1.0.0",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=["f1"],
        feature_dtypes=["float32"],
        schema_hash="update_hash",
        pipeline_signature="sig",
        pipeline_version="v1",
    )
    feature_id = reg.register_feature_set(manifest)

    reg.update_manifest(
        feature_id,
        parity_digest={"max_abs_diff": 0.1},
        stage=FeatureStage.STAGING,
    )
    reg.attach_artifact(feature_id, "report", "/tmp/report.json")
    reg.attach_artifacts(feature_id, {"drift": "/tmp/drift.json"})

    info = reg.get_feature_set(feature_id)
    assert info is not None
    assert info.manifest.parity_digest["max_abs_diff"] == 0.1
    assert info.manifest.stage == FeatureStage.STAGING
    assert info.artifacts["report"] == "/tmp/report.json"
    assert info.artifacts["drift"] == "/tmp/drift.json"
