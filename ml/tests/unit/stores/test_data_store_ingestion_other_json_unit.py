from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
import types
from unittest.mock import MagicMock

from ml.common.message_bus import MessagePublisherProtocol
from ml.config.events import EventStatus, Stage
from ml.features.earnings.store import DummyEarningsStore
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DatasetManifest, DatasetType, StorageKind
from ml.registry.persistence import BackendType, PersistenceConfig
from ml.stores.data_store import DataStore
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


def _set_noop_preflight(store: DataStore) -> None:
    def _noop_preflight(
        self: DataStore,
        *args: object,
        **kwargs: object,
    ) -> tuple[bool, None, dict[str, object]]:
        return True, None, {"warnings": []}

    def _noop_validator_preflight(
        *args: object,
        **kwargs: object,
    ) -> tuple[bool, None, dict[str, object]]:
        return True, None, {"warnings": []}

    store_any = cast(Any, store)
    store_any.preflight_check = types.MethodType(_noop_preflight, store)
    if hasattr(store_any, "_schema_validator"):
        store_any._schema_validator.preflight_check = _noop_validator_preflight
    if hasattr(store_any, "_data_writer"):
        store_any._data_writer._validator.preflight_check = _noop_validator_preflight


def _manifest(dataset_id: str, dataset_type: DatasetType, location: Path) -> DatasetManifest:
    schema: dict[str, str] = {
        "instrument_id": "str",
        "ts_event": "int64",
        "ts_init": "int64",
    }
    if dataset_type is DatasetType.SIGNALS:
        schema["decision_metadata"] = "json"
    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
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


def test_ingestion_predictions_emits_event_and_updates_watermark(tmp_path: Path) -> None:
    reg_dir = tmp_path / "reg"
    reg = DataRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )
    ds_id = "predictions_ds"
    reg.register_dataset(_manifest(ds_id, DatasetType.PREDICTIONS, tmp_path / "p.parquet"))

    pub = CapturePublisher()
    model_store = cast(ModelStore, MagicMock(spec=ModelStore))
    cast(MagicMock, model_store).write_batch = MagicMock()
    store = DataStore(
        connection_string="sqlite:///:memory:",
        registry=reg,
        feature_store=cast(FeatureStore, MagicMock(spec=FeatureStore)),
        model_store=model_store,
        strategy_store=cast(StrategyStore, MagicMock(spec=StrategyStore)),
        earnings_store=DummyEarningsStore(),
        publisher=pub,
        enable_publishing=True,
    )
    # Bypass schema preflight for deterministic behavior
    def _noop_preflight(self: DataStore, *args: object, **kwargs: object) -> tuple[bool, None, dict[str, object]]:
        return True, None, {"warnings": []}

    _set_noop_preflight(store)

    records: list[dict[str, Any]] = [
        {
            "instrument_id": "EURUSD.SIM",
            "model_id": "student_v1",
            "ts_event": 111,
            "ts_init": 111,
            "prediction": 0.6,
            "confidence": 0.8,
        },
        {
            "instrument_id": "EURUSD.SIM",
            "model_id": "student_v1",
            "ts_event": 222,
            "ts_init": 222,
            "prediction": 0.7,
            "confidence": 0.9,
        },
    ]
    store.write_ingestion(
        dataset_id=ds_id,
        records=records,
        source="live",
        run_id="run_pred",
        instrument_id="EURUSD.SIM",
    )

    # Publisher topic and payload
    assert pub.calls
    topic, payload = pub.calls[0]
    assert topic == "ml.models.created.EURUSD.SIM"
    assert payload["stage"] == Stage.PREDICTION_EMITTED.value
    assert payload["status"] == EventStatus.SUCCESS.value

    # Watermark updated
    reg.flush()  # Explicit flush required for tests that verify persistence
    data = json.loads((reg_dir / "data_registry.json").read_text())
    key = f"{ds_id}:EURUSD.SIM:live"
    assert key in data.get("watermarks", {})
    assert int(data["watermarks"][key]["last_success_ns"]) == 222 * 1_000_000_000


def test_ingestion_signals_emits_event_and_updates_watermark(tmp_path: Path) -> None:
    reg_dir = tmp_path / "reg"
    reg = DataRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )
    ds_id = "signals_ds"
    reg.register_dataset(_manifest(ds_id, DatasetType.SIGNALS, tmp_path / "s.parquet"))

    pub = CapturePublisher()
    strategy_store = cast(StrategyStore, MagicMock(spec=StrategyStore))
    cast(MagicMock, strategy_store).write_batch = MagicMock()
    store = DataStore(
        connection_string="sqlite:///:memory:",
        registry=reg,
        feature_store=cast(FeatureStore, MagicMock(spec=FeatureStore)),
        model_store=cast(ModelStore, MagicMock(spec=ModelStore)),
        strategy_store=strategy_store,
        earnings_store=DummyEarningsStore(),
        publisher=pub,
        enable_publishing=True,
    )
    _set_noop_preflight(store)

    records: list[dict[str, Any]] = [
        {
            "instrument_id": "EURUSD.SIM",
            "strategy_id": "ml_strategy",
            "signal_type": "long",
            "strength": 1.0,
            "ts_event": 333,
            "ts_init": 333,
            "decision_metadata": {"version": "v1"},
        },
        {
            "instrument_id": "EURUSD.SIM",
            "strategy_id": "ml_strategy",
            "signal_type": "short",
            "strength": 0.5,
            "ts_event": 444,
            "ts_init": 444,
            "decision_metadata": {"version": "v1"},
        },
    ]
    store.write_ingestion(
        dataset_id=ds_id,
        records=records,
        source="live",
        run_id="run_sig",
        instrument_id="EURUSD.SIM",
    )

    assert pub.calls
    topic, payload = pub.calls[0]
    assert topic == "ml.strategies.created.EURUSD.SIM"
    assert payload["stage"] == Stage.SIGNAL_EMITTED.value
    assert payload["status"] == EventStatus.SUCCESS.value

    reg.flush()  # Explicit flush required for tests that verify persistence
    data = json.loads((reg_dir / "data_registry.json").read_text())
    key = f"{ds_id}:EURUSD.SIM:live"
    assert key in data.get("watermarks", {})
    assert int(data["watermarks"][key]["last_success_ns"]) == 444 * 1_000_000_000
