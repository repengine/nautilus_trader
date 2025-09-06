from __future__ import annotations

import time
from pathlib import Path

import pytest

from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.base import ModelPrediction
from ml.stores.model_store import ModelStore


def test_model_store_emits_event_to_data_registry_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Shared DataRegistry (JSON backend)
    reg = DataRegistry(tmp_path / "datasets", persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=tmp_path / "datasets"))

    # ModelStore using Postgres for storage (or default); events go to JSON registry via injection
    ms = ModelStore(persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=tmp_path / "json"))
    ms.set_data_registry(reg)
    # Avoid DB dependency in unit test by stubbing out the actual write
    monkeypatch.setattr(ms, "_execute_write", lambda values: None)

    now_ns = int(time.time_ns())
    batch = [
        ModelPrediction(
            model_id="m1",
            instrument_id="SPY",
            prediction=0.6,
            confidence=0.7,
            features_used={"a": 1.0},
            inference_time_ms=1.2,
            _ts_event=now_ns,
            _ts_init=now_ns,
        ),
        ModelPrediction(
            model_id="m1",
            instrument_id="SPY",
            prediction=0.55,
            confidence=0.65,
            features_used={"a": 2.0},
            inference_time_ms=1.1,
            _ts_event=now_ns + 1,
            _ts_init=now_ns + 1,
        ),
    ]

    ms.write_batch(batch, emit_events=True)

    # Verify event written to JSON registry (data_registry.json exists)
    reg.flush()
    content = (tmp_path / "datasets" / "data_registry.json").read_text(encoding="utf-8")
    assert '"dataset_id": "predictions"' in content
    assert '"model_id": "m1"' in content
