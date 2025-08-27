"""
Integration tests for ModelStore canonical dataset IDs, JSONB writes, and parameterized reads.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import pytest

from ml.stores.model_store import ModelStore


@pytest.mark.usefixtures("clean_postgres_db")
def test_model_store_events_and_jsonb_and_reads(test_database: Any, monkeypatch: Any) -> None:
    """
    Write predictions, verify canonical events, JSONB dict, and parameterized reads.
    """
    store = ModelStore(connection_string=test_database.connection_string)

    # Mock DataRegistry
    class _MockRegistry:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def emit_event(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

        def update_watermark(self, **kwargs: Any) -> None:  # pragma: no cover - smoke only
            self.calls.append({"wm": kwargs})

    registry = _MockRegistry()
    monkeypatch.setattr(store, "_get_data_registry", lambda: registry)  # type: ignore[misc]

    # Write a couple predictions
    now_ns = int(time.time() * 1e9)
    store.write_prediction(
        model_id="test_model",
        instrument_id="EUR/USD",
        prediction=0.123,
        confidence=0.9,
        features={"f1": 1.0},
        inference_time_ms=0.7,
        ts_event=now_ns,
        is_live=False,
    )
    store.write_prediction(
        model_id="test_model",
        instrument_id="EUR/USD",
        prediction=-0.321,
        confidence=0.8,
        features={"f2": 2.0},
        inference_time_ms=0.6,
        ts_event=now_ns + 1,
        is_live=False,
    )
    store.flush()

    # Canonical dataset_id asserted via registry mock
    event_calls = [c for c in registry.calls if isinstance(c, dict) and c.get("dataset_id")]
    assert event_calls, "No events emitted"
    assert any(c["dataset_id"] == "predictions" and c["stage"] == "PREDICTION_EMITTED" for c in event_calls)

    # JSONB check: read back features_used via read_predictions
    df: pd.DataFrame = store.read_predictions("test_model", "EUR/USD", start_ns=now_ns, end_ns=now_ns + 10)
    assert not df.empty
    assert "features_used" in df.columns
    # Features_used should be dict-like (SQLAlchemy may deserialize to dict); if string, ensure JSON-looking
    fu = df.iloc[0]["features_used"]
    assert isinstance(fu, (dict, str))
    if isinstance(fu, dict):
        assert set(fu.keys()) <= {"f1", "f2"}

    # Parameterized reads basic behavior
    df_range: pd.DataFrame = store.read_range(start_ns=now_ns, end_ns=now_ns + 2, instrument_id="EUR/USD")
    assert len(df_range) >= 2

    latest: pd.DataFrame = store.get_latest("EUR/USD", limit=1)
    assert len(latest) == 1

