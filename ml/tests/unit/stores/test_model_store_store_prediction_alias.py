from __future__ import annotations

from typing import Any

import pytest

from ml.stores.model_store import ModelStore


def test_store_prediction_kwargs_alias_calls_write_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ml.stores.model_store.ModelStore._init_engine_and_tables", lambda self: None
    )
    store = ModelStore(connection_string=None)

    captured: dict[str, Any] = {}

    def fake_write_prediction(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(store, "write_prediction", fake_write_prediction)

    store.store_prediction(
        model_id=123,  # coerced to str
        instrument_id="EURUSD.SIM",
        ts_event=1700000000,  # seconds -> alias passes through; write handles normalization
        prediction=0.42,
        confidence=0.9,
        # omit features/inference_time_ms to hit defaults
    )

    assert captured["model_id"] == "123"
    assert captured["instrument_id"] == "EURUSD.SIM"
    assert isinstance(captured["ts_event"], int)
    assert captured["prediction"] == 0.42
    assert captured["confidence"] == 0.9
    assert captured["features"] == {}
    assert captured["inference_time_ms"] == 0.0
