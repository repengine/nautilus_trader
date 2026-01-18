from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from ml.common.message_bus import MessagePublisherProtocol
from ml.config.events import Stage
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.features.earnings.store import DummyEarningsStore
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


class TestDataStoreWriteIngestion:
    def test_write_ingestion_success_emits_event_and_publishes(self, tmp_path: Path) -> None:
        # Registry (JSON) and manifest
        reg_dir = tmp_path / "reg"
        reg = DataRegistry(
            registry_path=reg_dir,
            persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
        )
        manifest = _make_manifest("features_ds", tmp_path / "features.parquet")
        reg.register_dataset(manifest)

        # DataStore with mocked stores and capture publisher
        feature_store = cast(Any, MagicMock(write_features=MagicMock()))
        model_store = cast(Any, MagicMock())
        strategy_store = cast(Any, MagicMock())
        pub = CapturePublisher()
        store = cast(Any, DataStore)(
            connection_string="sqlite:///:memory:",
            registry=reg,  # use JSON backend
            feature_store=cast(Any, feature_store),
            model_store=cast(Any, model_store),
            strategy_store=cast(Any, strategy_store),
            earnings_store=DummyEarningsStore(),
            publisher=pub,
            enable_publishing=True,
            fail_on_validation_error=False,
        )

        # Records satisfy manifest
        records: list[dict[str, Any]] = [
            {"instrument_id": "EURUSD.SIM", "ts_event": 1000, "ts_init": 1000},
            {"instrument_id": "EURUSD.SIM", "ts_event": 2000, "ts_init": 2000},
        ]

        event = store.write_ingestion(
            dataset_id=manifest.dataset_id,
            records=records,
            source="live",
            run_id="run_ok",
            instrument_id="EURUSD.SIM",
        )

        # Publisher should have been called with canonical topic
        assert len(pub.calls) >= 1
        topic, payload = pub.calls[0]
        assert topic == "ml.features.updated.EURUSD.SIM"
        assert payload["dataset_id"] == manifest.dataset_id
        assert payload["instrument_id"] == "EURUSD.SIM"
        assert payload["stage"] == Stage.FEATURE_COMPUTED.value
        assert payload["status"] == "success"

        # Verify JSON registry has persisted events
        reg.flush()  # Explicit flush required for tests that verify persistence
        data = json.loads((reg_dir / "data_registry.json").read_text())
        assert any(evt.get("dataset_id") == manifest.dataset_id for evt in data.get("events", []))
        assert event.record_count == 2

    def test_write_ingestion_preflight_failure_raises(self, tmp_path: Path) -> None:
        reg_dir = tmp_path / "reg"
        reg = DataRegistry(
            registry_path=reg_dir,
            persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
        )
        manifest = _make_manifest("features_ds_bad", tmp_path / "features.parquet")
        reg.register_dataset(manifest)

        store = cast(Any, DataStore)(
            connection_string="sqlite:///:memory:",
            registry=reg,
            feature_store=cast(Any, object()),
            model_store=cast(Any, object()),
            strategy_store=cast(Any, object()),
            earnings_store=DummyEarningsStore(),
        )

        # Missing required instrument_id column -> preflight should fail
        bad_records: list[dict[str, Any]] = [
            {"ts_event": 1000, "ts_init": 1000},
        ]

        from pytest import raises

        with raises(ValueError):
            store.write_ingestion(
                dataset_id=manifest.dataset_id,
                records=bad_records,
                source="live",
                run_id="run_bad",
                instrument_id="EURUSD.SIM",
            )
