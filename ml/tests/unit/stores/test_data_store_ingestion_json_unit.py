from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from ml.common.message_bus import MessagePublisherProtocol
from ml.config.events import EventStatus, Stage
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DatasetManifest, DatasetType, StorageKind
from ml.registry.persistence import BackendType, PersistenceConfig
from ml.stores.data_store import DataStore


def _make_manifest(dataset_id: str, location: Path) -> DatasetManifest:
    schema: dict[str, str] = {
        "instrument_id": "str",
        "ts_event": "int64",
        "ts_init": "int64",
    }
    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.PARQUET,
        location=str(location),
        partitioning={},
        retention_days=1,
        schema=schema,
        ts_field="ts_event",
        seq_field=None,
        primary_keys=["instrument_id", "ts_event"],
        schema_hash="",
        constraints={},
        lineage=[],
        pipeline_signature="test",
        version="1.0.0",
    )


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


def test_write_ingestion_failure_emits_failed_event_and_raises(tmp_path: Path) -> None:
    reg_dir = tmp_path / "reg"
    reg = DataRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )
    manifest = _make_manifest("features_ds_fail", tmp_path / "features.parquet")
    reg.register_dataset(manifest)

    # DataStore with feature_store raising after preflight to exercise failure path
    feature_store = cast(Any, MagicMock())
    feature_store.write_features = MagicMock(side_effect=RuntimeError("boom"))
    store = DataStore(
        connection_string="sqlite:///:memory:",
        registry=reg,
        feature_store=feature_store,
        model_store=cast(Any, MagicMock()),
        strategy_store=cast(Any, MagicMock()),
    )

    # Valid records so preflight passes; write_features then fails to exercise emit_event(status=failed)
    records: list[dict[str, Any]] = [
        {"instrument_id": "EURUSD.SIM", "ts_event": 1, "ts_init": 1},
    ]

    # Bypass full preflight to keep unit deterministic
    store.preflight_check = lambda *a, **k: (True, None, {"warnings": []})  # type: ignore[assignment]

    with pytest.raises(RuntimeError):  # raised by DataStore after emitting failed event
        store.write_ingestion(
            dataset_id=manifest.dataset_id,
            records=records,
            source="live",
            run_id="run_fail",
            instrument_id="EURUSD.SIM",
        )

    data = json.loads((reg_dir / "data_registry.json").read_text())
    events = data.get("events", [])
    assert any(
        e.get("dataset_id") == manifest.dataset_id and e.get("status") == EventStatus.FAILED.value
        for e in events
    )


def test_write_ingestion_updates_watermark_json(tmp_path: Path) -> None:
    reg_dir = tmp_path / "reg"
    reg = DataRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )
    manifest = _make_manifest("features_ds_ok", tmp_path / "features.parquet")
    reg.register_dataset(manifest)

    pub = CapturePublisher()
    store = DataStore(
        connection_string="sqlite:///:memory:",
        registry=reg,
        feature_store=cast(Any, MagicMock(write_features=MagicMock())),
        model_store=cast(Any, MagicMock()),
        strategy_store=cast(Any, MagicMock()),
        publisher=pub,
    )

    records: list[dict[str, Any]] = [
        {"instrument_id": "EURUSD.SIM", "ts_event": 1000, "ts_init": 1000},
        {"instrument_id": "EURUSD.SIM", "ts_event": 2000, "ts_init": 2000},
    ]

    # Bypass preflight schema hash for deterministic test
    store.preflight_check = lambda *a, **k: (True, None, {"warnings": []})  # type: ignore[assignment]

    store.write_ingestion(
        dataset_id=manifest.dataset_id,
        records=records,
        source="live",
        run_id="run_ok",
        instrument_id="EURUSD.SIM",
    )

    # Verify event published and registry updated
    assert pub.calls
    topic, payload = pub.calls[0]
    assert topic == "ml.features.updated.EURUSD.SIM"
    assert payload["stage"] == Stage.FEATURE_COMPUTED.value
    assert payload["status"] == EventStatus.SUCCESS.value

    data = json.loads((reg_dir / "data_registry.json").read_text())
    watermarks = data.get("watermarks", {})
    # key format: dataset:instrument:source
    key = f"{manifest.dataset_id}:EURUSD.SIM:live"
    assert key in watermarks
    # last_success_ns should reflect the maximum ts_event in records
    assert int(watermarks[key]["last_success_ns"]) == 2000
