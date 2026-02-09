"""
Focused unit tests for DataRegistry JSON backend and schema/parity enforcement.

These tests avoid database dependencies by using the JSON backend and temporary
directories. They validate event emission, watermark tracking, lineage linking, and
model-feature schema enforcement via environment flags.

"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from sqlalchemy.exc import IntegrityError

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.base import DataRequirements
from ml.registry.base import ModelRole
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.tests.builders import RegistryBuilder


def _mk_dataset_manifest(dataset_id: str) -> DatasetManifest:
    schema = {
        "instrument_id": "str",
        "ts_event": "int64",
        "ts_init": "int64",
        "value": "float64",
    }
    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.PARQUET,
        location="/tmp",
        partitioning={"by": ["date", "instrument_id"]},
        retention_days=7,
        schema=schema,
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints={},
        lineage=[],
        pipeline_signature="test_pipeline",
        version="1.0.0",
    )


def _mk_manifest_row(dataset_id: str) -> dict[str, Any]:
    manifest = _mk_dataset_manifest(dataset_id)
    return {
        "dataset_id": dataset_id,
        "dataset_type": manifest.dataset_type.value,
        "storage_kind": manifest.storage_kind.value,
        "location": manifest.location,
        "partitioning": json.dumps(manifest.partitioning),
        "retention_days": manifest.retention_days,
        "schema": json.dumps(manifest.schema),
        "schema_hash": manifest.schema_hash,
        "constraints": json.dumps(manifest.constraints),
        "lineage": json.dumps(manifest.lineage),
        "pipeline_signature": manifest.pipeline_signature,
        "version": manifest.version,
        "created_at": manifest.created_at,
        "last_modified": manifest.last_modified,
        "metadata": json.dumps({"ts_field": "ts_event", "primary_keys": manifest.primary_keys}),
    }


def _mk_postgres_registry(
    tmp_path: Path,
    *,
    session: Any | None,
) -> tuple[DataRegistry, Any]:
    registry_path = tmp_path / "registry_postgres_stub"
    registry = DataRegistry(
        registry_path=registry_path,
        batch_save_interval=0.0,
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=registry_path,
        ),
    )

    persistence_stub = SimpleNamespace(
        get_session=(lambda: session),
        log_audit=MagicMock(),
        close=MagicMock(),
    )
    registry_runtime = cast(Any, registry)
    registry_runtime.backend = BackendType.POSTGRES
    registry_runtime.persistence = persistence_stub
    return registry, persistence_stub


def test_data_registry_json_emit_event_and_watermark(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    cfg = PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir)
    reg = DataRegistry(registry_path=reg_dir, persistence_config=cfg)

    # Register manifest and flush
    manifest = _mk_dataset_manifest("features_test")
    reg.register_dataset(manifest)
    reg.flush()

    # Emit event and ensure it persists
    reg.emit_event(
        dataset_id="features_test",
        instrument_id="EUR/USD",
        stage=Stage.CATALOG_WRITTEN,
        source=Source.HISTORICAL,
        run_id="run_1",
        ts_min=1,
        ts_max=2,
        count=10,
        status=EventStatus.SUCCESS,
        metadata={"foo": "bar"},
    )
    # The JSON backend stores events in memory and flushes to file immediately
    assert len(reg._events) >= 1

    # Update watermark and verify retrieval
    reg.update_watermark(
        dataset_id="features_test",
        instrument_id="EUR/USD",
        source=Source.LIVE,
        last_success_ns=2,
        count=10,
        completeness_pct=100.0,
    )
    wm = reg.get_watermark("features_test", "EUR/USD", Source.LIVE)
    assert wm is not None
    assert wm.last_success_ns == 2
    assert wm.last_count == 10

    # Link lineage entries and ensure they are recorded
    reg.link_lineage(
        child_dataset_id="features_test",
        parent_ids=["bars_eurusd_1m"],
        transform_id="feature_pipeline_v1",
        ts_range={"start_ns": 1, "end_ns": 2},
        params={"lookback": 20},
    )
    assert len(reg._lineage) >= 1


def test_model_feature_schema_enforcement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Prepare feature registry JSON with a known schema_hash
    reg_dir = tmp_path / "registry"
    freg = FeatureRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )

    feature_names = ["f1", "f2"]
    feature_dtypes = ["float32", "float32"]
    pipeline_sig = "sig_v1"
    schema_hash = compute_schema_hash(feature_names, feature_dtypes, pipeline_sig)

    fmanifest = RegistryBuilder.feature_manifest(
        feature_set_id="feat_v1",
        name="test_features",
        version="1.0.0",
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=feature_names,
        feature_dtypes=feature_dtypes,
        schema_hash=schema_hash,
        pipeline_signature=pipeline_sig,
        pipeline_version="1",
        parity_tolerance=1e-10,
    )
    freg.register_feature_set(fmanifest)

    # Build a model manifest with a mismatched feature_schema_hash
    mmanifest = RegistryBuilder.model_manifest(
        model_id="",
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="xgboost",
        feature_schema={"f1": "float32", "f2": "float32"},
        feature_schema_hash="deadbeef",
        serveable=True,
        artifact_format="onnx",
        feature_set_id="feat_v1",
    )

    # Strict parity: mismatch should raise
    monkeypatch.setenv("ML_STRICT_FEATURE_PARITY", "1")

    # Prepare a dummy ONNX file (minimal, existence is validated only)
    model_path = reg_dir / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"onnx")

    # Register using the concrete ModelRegistry
    from ml.registry import ModelRegistry

    mreg = ModelRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )
    with pytest.raises(ValueError):
        mreg.register_model(model_path=model_path, manifest=mmanifest, auto_deploy=False)


def test_register_dataset_postgres_success_records_manifest_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    registry, persistence_stub = _mk_postgres_registry(tmp_path, session=session)
    emit_mock = MagicMock()
    monkeypatch.setattr(registry, "emit_event", emit_mock)

    manifest = _mk_dataset_manifest("pg_features")
    dataset_id = registry.register_dataset(manifest)

    assert dataset_id == "pg_features"
    execute_args = session.execute.call_args.args
    assert "INSERT INTO ml_dataset_registry" in str(execute_args[0])
    payload = cast(dict[str, Any], execute_args[1])
    assert payload["dataset_type"] == "FEATURES"
    metadata_payload = json.loads(str(payload["metadata"]))
    assert metadata_payload["ts_field"] == "ts_event"
    assert metadata_payload["primary_keys"] == ["instrument_id", "ts_event"]
    assert registry._manifests["pg_features"] == manifest
    session.commit.assert_called_once()
    session.close.assert_called_once()
    persistence_stub.log_audit.assert_called_once()
    emit_mock.assert_called_once()


def test_register_dataset_postgres_handles_integrity_error_by_hydrating_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    session.execute.side_effect = IntegrityError("insert", {}, Exception("duplicate"))
    registry, _ = _mk_postgres_registry(tmp_path, session=session)
    monkeypatch.setattr(registry, "emit_event", MagicMock())
    existing = _mk_dataset_manifest("pg_existing")
    monkeypatch.setattr(registry, "get_manifest", lambda _dataset_id: existing)

    dataset_id = registry.register_dataset(existing)

    assert dataset_id == "pg_existing"
    assert registry._manifests["pg_existing"] == existing
    session.rollback.assert_called_once()
    session.close.assert_called_once()


def test_register_dataset_postgres_rolls_back_on_generic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    session.execute.side_effect = RuntimeError("insert-failed")
    registry, _ = _mk_postgres_registry(tmp_path, session=session)
    monkeypatch.setattr(registry, "emit_event", MagicMock())

    with pytest.raises(RuntimeError, match="insert-failed"):
        registry.register_dataset(_mk_dataset_manifest("pg_fail"))

    session.rollback.assert_called_once()
    session.close.assert_called_once()


def test_update_manifest_json_raises_for_missing_dataset(tmp_path: Path) -> None:
    registry = DataRegistry(
        registry_path=tmp_path / "registry_json_missing_update",
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_json_missing_update",
        ),
        batch_save_interval=0.0,
    )
    with pytest.raises(ValueError, match="Dataset 'missing' not found"):
        registry.update_manifest("missing", {"version": "1.1.0"})


def test_update_manifest_postgres_validates_changes_and_row_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_fields_session = MagicMock()
    registry_no_fields, _ = _mk_postgres_registry(tmp_path, session=no_fields_session)
    monkeypatch.setattr(registry_no_fields, "emit_event", MagicMock())
    with pytest.raises(ValueError, match="No valid fields to update"):
        registry_no_fields.update_manifest("pg_dataset", {"unsupported": "value"})
    no_fields_session.rollback.assert_called_once()
    no_fields_session.close.assert_called_once()

    missing_row_session = MagicMock()
    missing_row_session.execute.return_value = SimpleNamespace(rowcount=0)
    registry_missing_row, _ = _mk_postgres_registry(tmp_path, session=missing_row_session)
    monkeypatch.setattr(registry_missing_row, "emit_event", MagicMock())
    with pytest.raises(ValueError, match="Dataset 'pg_dataset' not found"):
        registry_missing_row.update_manifest("pg_dataset", {"version": "2.0.0"})
    missing_row_session.rollback.assert_called_once()
    missing_row_session.close.assert_called_once()


def test_update_manifest_postgres_success_serializes_json_fields_and_updates_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    session.execute.return_value = SimpleNamespace(rowcount=1)
    registry, persistence_stub = _mk_postgres_registry(tmp_path, session=session)
    emit_mock = MagicMock()
    monkeypatch.setattr(registry, "emit_event", emit_mock)

    manifest = _mk_dataset_manifest("pg_dataset")
    registry._manifests[manifest.dataset_id] = manifest

    registry.update_manifest(
        manifest.dataset_id,
        {
            "version": "2.0.0",
            "metadata": {"owner": "registry-tests"},
            "constraints": {"nullability": {"instrument_id": False}},
        },
    )

    execute_params = cast(dict[str, Any], session.execute.call_args.args[1])
    assert execute_params["version"] == "2.0.0"
    assert json.loads(str(execute_params["metadata"])) == {"owner": "registry-tests"}
    assert json.loads(str(execute_params["constraints"])) == {
        "nullability": {"instrument_id": False},
    }
    assert registry._manifests[manifest.dataset_id].version == "2.0.0"
    session.commit.assert_called_once()
    session.close.assert_called_once()
    persistence_stub.log_audit.assert_called_once()
    emit_mock.assert_called_once()


def test_list_and_get_manifest_postgres_paths(tmp_path: Path) -> None:
    cached_manifest = _mk_dataset_manifest("cached_dataset")
    registry_none, _ = _mk_postgres_registry(tmp_path, session=None)
    registry_none._manifests[cached_manifest.dataset_id] = cached_manifest

    manifests = registry_none.list_manifests()
    assert manifests == [cached_manifest]
    assert registry_none.get_manifest(cached_manifest.dataset_id) == cached_manifest

    session_rows = MagicMock()
    session_rows.execute.return_value.fetchall.return_value = [_mk_manifest_row("from_rows")]
    registry_rows, _ = _mk_postgres_registry(tmp_path, session=session_rows)
    listed = registry_rows.list_manifests()
    assert len(listed) == 1
    assert listed[0].dataset_id == "from_rows"
    assert registry_rows._manifests["from_rows"].dataset_id == "from_rows"
    session_rows.close.assert_called_once()

    session_get = MagicMock()
    session_get.execute.return_value.fetchone.return_value = _mk_manifest_row("from_get")
    registry_get, _ = _mk_postgres_registry(tmp_path, session=session_get)
    fetched = registry_get.get_manifest("from_get")
    assert fetched.dataset_id == "from_get"
    assert registry_get._manifests["from_get"].dataset_id == "from_get"
    session_get.close.assert_called_once()

    session_missing = MagicMock()
    session_missing.execute.return_value.fetchone.return_value = None
    registry_missing, _ = _mk_postgres_registry(tmp_path, session=session_missing)
    with pytest.raises(ValueError, match="Dataset 'missing' not found"):
        registry_missing.get_manifest("missing")
    session_missing.close.assert_called_once()


def test_get_contract_postgres_uses_cache_then_manifest(tmp_path: Path) -> None:
    registry, _ = _mk_postgres_registry(tmp_path, session=MagicMock())
    manifest = _mk_dataset_manifest("contract_dataset")

    contract = registry._create_contract_from_manifest(manifest)
    registry._contracts[manifest.dataset_id] = contract
    assert registry.get_contract(manifest.dataset_id) == contract

    del registry._contracts[manifest.dataset_id]
    registry._manifests[manifest.dataset_id] = manifest
    created = registry.get_contract(manifest.dataset_id)
    assert created.dataset_id == manifest.dataset_id
    assert registry._contracts[manifest.dataset_id] == created


def test_emit_event_postgres_fallback_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ml.registry.data_registry.set_instrumentation_search_path", lambda _session: None)

    success_session = MagicMock()
    success_registry, _ = _mk_postgres_registry(tmp_path, session=success_session)
    success_registry.emit_event(
        dataset_id="pg.features",
        instrument_id="EUR/USD",
        stage=Stage.CATALOG_WRITTEN,
        source=Source.HISTORICAL,
        run_id="run-1",
        ts_min=1,
        ts_max=2,
        count=3,
        status=EventStatus.SUCCESS,
        metadata={"foo": "bar"},
    )
    assert "emit_data_event_ext" in str(success_session.execute.call_args_list[0].args[0])
    success_session.commit.assert_called_once()

    legacy_session = MagicMock()
    legacy_session.execute.side_effect = [RuntimeError("ext failed"), None]
    legacy_registry, _ = _mk_postgres_registry(tmp_path, session=legacy_session)
    legacy_registry.emit_event(
        dataset_id="pg.features",
        instrument_id="EUR/USD",
        stage=Stage.CATALOG_WRITTEN,
        source=Source.HISTORICAL,
        run_id="run-2",
        ts_min=1,
        ts_max=2,
        count=3,
        status=EventStatus.SUCCESS,
    )
    assert legacy_session.execute.call_count == 2
    assert "emit_data_event(" in str(legacy_session.execute.call_args_list[1].args[0])
    legacy_session.rollback.assert_called_once()
    legacy_session.commit.assert_called_once()

    insert_session = MagicMock()
    insert_session.execute.side_effect = [
        RuntimeError("ext failed"),
        RuntimeError("legacy failed"),
        None,
    ]
    insert_registry, _ = _mk_postgres_registry(tmp_path, session=insert_session)
    insert_registry.emit_event(
        dataset_id="pg.features",
        instrument_id="EUR/USD",
        stage=Stage.CATALOG_WRITTEN,
        source=Source.HISTORICAL,
        run_id="run-3",
        ts_min=1,
        ts_max=2,
        count=3,
        status=EventStatus.SUCCESS,
    )
    assert "INSERT INTO ml_data_events" in str(insert_session.execute.call_args_list[2].args[0])
    assert insert_session.rollback.call_count == 2
    insert_session.commit.assert_called_once()
    insert_session.close.assert_called_once()

    failing_insert_session = MagicMock()
    failing_insert_session.execute.side_effect = [
        RuntimeError("ext failed"),
        RuntimeError("legacy failed"),
        RuntimeError("insert failed"),
    ]
    failing_registry, _ = _mk_postgres_registry(tmp_path, session=failing_insert_session)
    with pytest.raises(RuntimeError, match="insert failed"):
        failing_registry.emit_event(
            dataset_id="pg.features",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run-4",
            ts_min=1,
            ts_max=2,
            count=3,
            status=EventStatus.SUCCESS,
        )
    assert failing_insert_session.rollback.call_count == 3
    failing_insert_session.close.assert_called_once()


def test_watermark_and_lineage_postgres_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ml.registry.data_registry.set_instrumentation_search_path", lambda _session: None)

    watermark_session = MagicMock()
    registry, _ = _mk_postgres_registry(tmp_path, session=watermark_session)
    registry.update_watermark(
        dataset_id="pg.features",
        instrument_id="EUR/USD",
        source=Source.HISTORICAL,
        last_success_ns=5,
        count=2,
        completeness_pct=98.0,
    )
    watermark_session.commit.assert_called_once()
    watermark_session.close.assert_called_once()
    assert "pg.features:EUR/USD:historical" in registry._watermarks

    failing_watermark_session = MagicMock()
    failing_watermark_session.execute.side_effect = RuntimeError("watermark failed")
    failing_registry, _ = _mk_postgres_registry(tmp_path, session=failing_watermark_session)
    with pytest.raises(RuntimeError, match="watermark failed"):
        failing_registry.update_watermark(
            dataset_id="pg.features",
            instrument_id="EUR/USD",
            source=Source.HISTORICAL,
            last_success_ns=6,
            count=2,
            completeness_pct=97.0,
        )
    failing_watermark_session.rollback.assert_called_once()
    failing_watermark_session.close.assert_called_once()

    get_session = MagicMock()
    get_session.execute.return_value.fetchone.return_value = {
        "dataset_id": "pg.features",
        "instrument_id": "EUR/USD",
        "source": "historical",
        "last_success_ns": 6,
        "last_attempt_ns": 6,
        "last_count": 2,
        "completeness_pct": 97.0,
        "updated_at": 1.0,
    }
    get_registry, _ = _mk_postgres_registry(tmp_path, session=get_session)
    watermark = get_registry.get_watermark("pg.features", "EUR/USD", Source.HISTORICAL)
    assert watermark is not None
    assert watermark.last_success_ns == 6
    get_session.close.assert_called_once()

    iter_session = MagicMock()
    iter_session.execute.return_value.fetchall.return_value = [
        {
            "dataset_id": "pg.features",
            "instrument_id": "EUR/USD",
            "source": "historical",
            "last_success_ns": 6,
            "last_attempt_ns": 6,
            "last_count": 2,
            "completeness_pct": 97.0,
            "updated_at": 1.0,
        },
    ]
    iter_registry, _ = _mk_postgres_registry(tmp_path, session=iter_session)
    iterated = list(
        iter_registry.iter_watermarks(
            dataset_id="pg.features",
            instrument_id="EUR/USD",
            source=Source.HISTORICAL,
            limit=1,
        ),
    )
    assert len(iterated) == 1
    assert iterated[0].dataset_id == "pg.features"
    assert "LIMIT :limit" in str(iter_session.execute.call_args.args[0])
    iter_session.close.assert_called_once()

    lineage_session = MagicMock()
    lineage_registry, _ = _mk_postgres_registry(tmp_path, session=lineage_session)
    lineage_registry.link_lineage(
        child_dataset_id="child",
        parent_ids=["parent_a", "parent_b"],
        transform_id="transform",
        ts_range={"start_ns": 1, "end_ns": 2},
        params={"lookback": 10},
    )
    assert lineage_session.execute.call_count == 2
    lineage_session.commit.assert_called_once()
    lineage_session.close.assert_called_once()

    iter_lineage_session = MagicMock()
    iter_lineage_session.execute.return_value.fetchall.return_value = [
        {
            "transform_id": "transform",
            "child_dataset_id": "child",
            "parent_dataset_id": "parent_a",
            "ts_range": json.dumps({"start_ns": 1, "end_ns": 2}),
            "parameters": json.dumps({"lookback": 10}),
            "created_at": 12.0,
        },
    ]
    iter_lineage_registry, _ = _mk_postgres_registry(tmp_path, session=iter_lineage_session)
    lineage = list(iter_lineage_registry.iter_lineage(child="child", parent="parent_a", limit=1))
    assert len(lineage) == 1
    assert lineage[0].transform_id == "transform"
    assert "LIMIT :limit" in str(iter_lineage_session.execute.call_args.args[0])
    iter_lineage_session.close.assert_called_once()


def test_row_conversion_helpers_handle_invalid_json_payloads(tmp_path: Path) -> None:
    registry = DataRegistry(
        registry_path=tmp_path / "registry_rows",
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_rows",
        ),
        batch_save_interval=0.0,
    )

    row = _mk_manifest_row("row_dataset")
    row["partitioning"] = "{invalid_json"
    row["constraints"] = "{invalid_json"
    manifest = registry._manifest_from_row(row)
    assert manifest.dataset_id == "row_dataset"
    assert manifest.pipeline_signature == "test_pipeline"

    lineage_record = registry._lineage_from_row(
        {
            "transform_id": "transform_a",
            "child_dataset_id": "child_dataset",
            "parent_dataset_id": "parent_dataset",
            "ts_range": "{invalid_json",
            "parameters": "{invalid_json",
            "created_at": None,
        },
    )
    assert lineage_record.ts_range == {}
    assert lineage_record.parameters == {}
    assert lineage_record.created_at == 0.0


def test_iter_watermarks_json_filters_and_limit_paths(tmp_path: Path) -> None:
    registry = DataRegistry(
        registry_path=tmp_path / "registry_watermarks",
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_watermarks",
        ),
        batch_save_interval=0.0,
    )

    registry.update_watermark(
        dataset_id="dataset_a",
        instrument_id="EUR/USD",
        source=Source.HISTORICAL,
        last_success_ns=10,
        count=1,
        completeness_pct=99.0,
    )
    registry.update_watermark(
        dataset_id="dataset_a",
        instrument_id="EUR/USD",
        source=Source.LIVE,
        last_success_ns=11,
        count=2,
        completeness_pct=98.0,
    )
    registry.update_watermark(
        dataset_id="dataset_b",
        instrument_id="EUR/USD",
        source=Source.HISTORICAL,
        last_success_ns=12,
        count=3,
        completeness_pct=97.0,
    )

    filtered = list(
        registry.iter_watermarks(
            dataset_id="dataset_a",
            instrument_id="EUR/USD",
            source=Source.HISTORICAL,
            limit=1,
        ),
    )
    assert len(filtered) == 1
    assert filtered[0].dataset_id == "dataset_a"
    assert filtered[0].source == Source.HISTORICAL.value


def test_iter_lineage_json_filter_and_invalid_string_payload_paths(tmp_path: Path) -> None:
    registry = DataRegistry(
        registry_path=tmp_path / "registry_lineage_json",
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_lineage_json",
        ),
        batch_save_interval=0.0,
    )

    registry._lineage = [
        {
            "transform_id": "skip_child",
            "child_dataset_id": "other_child",
            "parent_dataset_id": "target_parent",
            "ts_range": {"start_ns": 1, "end_ns": 2},
            "parameters": {"k": "v"},
            "created_at": 3.0,
        },
        {
            "transform_id": "skip_parent",
            "child_dataset_id": "target_child",
            "parent_dataset_id": "other_parent",
            "ts_range": {"start_ns": 2, "end_ns": 3},
            "parameters": {"k": "v"},
            "created_at": 2.0,
        },
        {
            "transform_id": "keep_record",
            "child_dataset_id": "target_child",
            "parent_dataset_id": "target_parent",
            "ts_range": "{invalid_json",
            "parameters": "{invalid_json",
            "created_at": 1.0,
        },
    ]

    records = list(
        registry.iter_lineage(
            child="target_child",
            parent="target_parent",
            limit=1,
        ),
    )
    assert len(records) == 1
    assert records[0].transform_id == "keep_record"
    assert records[0].ts_range == {}
    assert records[0].parameters == {}


def test_pipeline_signature_helpers_cover_validation_and_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = DataRegistry(
        registry_path=tmp_path / "registry_pipeline_signature",
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_pipeline_signature",
        ),
        batch_save_interval=0.0,
    )

    manifest = _mk_dataset_manifest("signature_dataset")
    registry.register_dataset(manifest)
    assert registry.get_pipeline_signature("signature_dataset") == "test_pipeline"

    registry.set_pipeline_signature("signature_dataset", "sig_updated")
    assert registry.get_pipeline_signature("signature_dataset") == "sig_updated"
    assert registry.get_manifest("signature_dataset").metadata["pipeline_signature"] == "sig_updated"

    with pytest.raises(ValueError, match="Pipeline signature cannot be empty"):
        registry.set_pipeline_signature("signature_dataset", "")

    def _raise_key_error(_dataset_id: str) -> DatasetManifest:
        raise KeyError("missing")

    monkeypatch.setattr(registry, "get_manifest", _raise_key_error)
    assert registry.get_pipeline_signature("missing_dataset") is None

    def _raise_runtime_error(_dataset_id: str) -> DatasetManifest:
        raise RuntimeError("boom")

    monkeypatch.setattr(registry, "get_manifest", _raise_runtime_error)
    assert registry.get_pipeline_signature("errored_dataset") is None


def test_emit_event_json_trims_old_records(tmp_path: Path) -> None:
    registry = DataRegistry(
        registry_path=tmp_path / "registry_trim_events",
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_trim_events",
        ),
        batch_save_interval=0.0,
    )

    registry._events = [{"dataset_id": f"old_{idx}"} for idx in range(10000)]
    registry.emit_event(
        dataset_id="dataset_trim",
        instrument_id="EUR/USD",
        stage=Stage.CATALOG_WRITTEN,
        source=Source.HISTORICAL,
        run_id="trim-run",
        ts_min=1,
        ts_max=2,
        count=1,
        status=EventStatus.SUCCESS,
    )
    assert len(registry._events) == 10000
    assert registry._events[-1]["dataset_id"] == "dataset_trim"


def test_destructor_cancels_pending_timer_and_closes_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = DataRegistry(
        registry_path=tmp_path / "registry_destructor",
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_destructor",
        ),
        batch_save_interval=0.0,
    )

    timer = MagicMock()
    do_save_mock = MagicMock()
    close_mock = MagicMock()

    registry._save_timer = timer
    registry._pending_save = True
    monkeypatch.setattr(registry, "_do_save", do_save_mock)
    registry.persistence = cast(
        Any,
        SimpleNamespace(
            close=close_mock,
            save_json=lambda *_args, **_kwargs: None,
        ),
    )

    registry.__del__()

    timer.cancel.assert_called_once()
    do_save_mock.assert_called_once()
    close_mock.assert_called_once()
    registry._pending_save = False


def test_default_init_and_manifest_enum_normalization_paths(tmp_path: Path) -> None:
    registry = DataRegistry(
        registry_path=tmp_path / "registry_default_init",
        batch_save_interval=0.0,
    )

    manifest_payload = registry._manifest_to_dict(_mk_dataset_manifest("normalize_dataset"))
    manifest_payload["dataset_type"] = "FEATURES"
    manifest_payload["storage_kind"] = "PARQUET"
    normalized_manifest = registry._dict_to_manifest(manifest_payload)

    assert normalized_manifest.dataset_type == DatasetType.FEATURES
    assert normalized_manifest.storage_kind == StorageKind.PARQUET


def test_postgres_session_missing_paths_raise_runtime_errors(tmp_path: Path) -> None:
    registry, _ = _mk_postgres_registry(tmp_path, session=None)
    registry._load_registry()
    registry.flush()

    with pytest.raises(RuntimeError, match="Failed to get database session"):
        registry.register_dataset(_mk_dataset_manifest("pg_missing_session_dataset"))

    with pytest.raises(RuntimeError, match="Failed to get database session"):
        registry.update_manifest("pg_missing_session_dataset", {"version": "2.0.0"})

    with pytest.raises(RuntimeError, match="Failed to get database session"):
        registry.get_manifest("pg_missing_session_dataset")

    with pytest.raises(RuntimeError, match="Failed to get database session"):
        registry.emit_event(
            dataset_id="pg_missing_session_dataset",
            instrument_id="EUR/USD",
            stage=Stage.CATALOG_WRITTEN,
            source=Source.HISTORICAL,
            run_id="run-missing-session",
            ts_min=1,
            ts_max=2,
            count=1,
            status=EventStatus.SUCCESS,
        )

    with pytest.raises(RuntimeError, match="Failed to get database session"):
        registry.update_watermark(
            dataset_id="pg_missing_session_dataset",
            instrument_id="EUR/USD",
            source=Source.HISTORICAL,
            last_success_ns=1,
            count=1,
            completeness_pct=100.0,
        )

    with pytest.raises(RuntimeError, match="Failed to get database session"):
        registry.get_watermark("pg_missing_session_dataset", "EUR/USD", Source.HISTORICAL)

    with pytest.raises(RuntimeError, match="Failed to get database session"):
        list(
            registry.iter_watermarks(
                dataset_id="pg_missing_session_dataset",
                instrument_id="EUR/USD",
                source=Source.HISTORICAL,
                limit=1,
            ),
        )

    with pytest.raises(RuntimeError, match="Failed to get database session"):
        registry.link_lineage(
            child_dataset_id="pg_missing_session_dataset",
            parent_ids=["parent_dataset"],
            transform_id="transform_missing_session",
            ts_range={"start_ns": 1, "end_ns": 2},
            params={"window": 10},
        )

    with pytest.raises(RuntimeError, match="Failed to get database session"):
        list(registry.iter_lineage(child="child", parent="parent", limit=1))


def test_register_update_and_deprecate_swallow_emit_event_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = DataRegistry(
        registry_path=tmp_path / "registry_emit_failures",
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_emit_failures",
        ),
        batch_save_interval=0.0,
    )

    monkeypatch.setattr(
        registry,
        "emit_event",
        MagicMock(side_effect=RuntimeError("emit failed")),
    )

    manifest = _mk_dataset_manifest("emit_failure_dataset")
    registry.register_dataset(manifest)
    registry.update_manifest("emit_failure_dataset", {"version": "1.0.1"})
    registry.deprecate("emit_failure_dataset")

    assert registry.get_manifest("emit_failure_dataset").metadata.get("deprecated") is True


def test_contract_resolution_and_bootstrap_fallback_paths(tmp_path: Path) -> None:
    registry = DataRegistry(
        registry_path=tmp_path / "registry_contracts",
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_contracts",
        ),
        batch_save_interval=0.0,
    )

    manifest = replace(
        _mk_dataset_manifest("contract_rules_dataset"),
        constraints={
            "ranges": {"value": {"min": 0.0, "max": 10.0}},
            "nullability": {"instrument_id": False},
            "regex": {"instrument_id": r"^[A-Z/]+$"},
            "null_rate_threshold": 0.1,
        },
    )
    contract = registry._create_contract_from_manifest(manifest)
    assert len(contract.validation_rules) >= 3
    assert contract.quality_thresholds["null_rate"] == 0.1

    with patch(
        "ml.registry.bootstrap_datasets.create_standard_contracts",
        side_effect=RuntimeError("bootstrap error"),
    ):
        fallback_contract = registry._resolve_bootstrap_contract(manifest)
    assert fallback_contract.dataset_id == manifest.dataset_id


def test_register_dataset_postgres_bootstrap_contract_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    registry, _ = _mk_postgres_registry(tmp_path, session=session)
    monkeypatch.setattr(registry, "emit_event", MagicMock())

    manifest = replace(_mk_dataset_manifest("ml.earnings_actuals"), seq_field="ts_event")
    dataset_id = registry.register_dataset(manifest)

    assert dataset_id == "ml.earnings_actuals"
    assert dataset_id in registry._contracts
    metadata_payload = json.loads(str(session.execute.call_args.args[1]["metadata"]))
    assert metadata_payload["seq_field"] == "ts_event"


def test_get_pipeline_signature_returns_none_when_no_signature_fields(tmp_path: Path) -> None:
    registry = DataRegistry(
        registry_path=tmp_path / "registry_signature_none",
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry_signature_none",
        ),
        batch_save_interval=0.0,
    )

    manifest = replace(_mk_dataset_manifest("no_signature_dataset"), pipeline_signature="", metadata={})
    registry.register_dataset(manifest)

    assert registry.get_pipeline_signature("no_signature_dataset") is None
