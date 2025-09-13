from __future__ import annotations

import time
from pathlib import Path

import pytest

from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.base import ModelPrediction
from ml.stores.model_store import ModelStore


def test_model_store_emits_event_to_data_registry_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    default_instrument_id,
    test_timestamps,
    sample_features,
) -> None:
    # Shared DataRegistry (JSON backend)
    reg = DataRegistry(
        tmp_path / "datasets",
        persistence_config=PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "datasets",
        ),
    )

    # ModelStore using Postgres for storage (or default); events go to JSON registry via injection
    ms = ModelStore(
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=tmp_path / "json"),
    )
    ms.set_data_registry(reg)
    # Avoid DB dependency in unit test by stubbing out the actual write
    monkeypatch.setattr(ms, "_execute_write", lambda values: None)

    ts_event, ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id).split(".")[0]  # Get "EUR/USD" from "EUR/USD.SIM"

    batch = [
        ModelPrediction(
            model_id="m1",
            instrument_id=instrument_id_str,
            prediction=0.6,
            confidence=0.7,
            features_used=sample_features,
            inference_time_ms=1.2,
            _ts_event=ts_event,
            _ts_init=ts_init,
        ),
        ModelPrediction(
            model_id="m1",
            instrument_id=instrument_id_str,
            prediction=0.55,
            confidence=0.65,
            features_used=sample_features,
            inference_time_ms=1.1,
            _ts_event=ts_event + 1,
            _ts_init=ts_init + 1,
        ),
    ]

    ms.write_batch(batch, emit_events=True)

    # Verify event written to JSON registry (data_registry.json exists)
    reg.flush()
    content = (tmp_path / "datasets" / "data_registry.json").read_text(encoding="utf-8")
    assert '"dataset_id": "predictions"' in content
    assert '"model_id": "m1"' in content
