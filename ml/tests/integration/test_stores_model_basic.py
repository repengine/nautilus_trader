from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import text

from ml.stores.model_store import ModelStore
from nautilus_trader.model.identifiers import InstrumentId

# Integration DB-backed read/write
pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.usefixtures("clean_postgres_db_module"),
]


def test_model_store_write_and_read(
    clean_postgres_db: Any,
    postgres_connection: str,
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ModelStore(
        connection_string=postgres_connection,
        batch_size=2,
        flush_interval_seconds=0.01,
    )
    # Avoid DataRegistry emissions in this unit test
    monkeypatch.setattr(store, "_emit_events", lambda predictions: None)

    ts_event, _ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id).split(".")[0]  # Get "EUR/USD" from "EUR/USD.SIM"

    # Two buffered writes should trigger flush by batch size
    store.write_prediction(
        model_id="model-A",
        instrument_id=instrument_id_str,
        prediction=0.7,
        confidence=0.9,
        features={},
        inference_time_ms=1.0,
        ts_event=ts_event,
        is_live=False,
    )
    store.write_prediction(
        model_id="model-A",
        instrument_id=instrument_id_str,
        prediction=0.2,
        confidence=0.5,
        features={},
        inference_time_ms=1.0,
        ts_event=ts_event + 1,
        is_live=False,
    )

    store.flush()

    table_name = store._qualified_table("ml_model_predictions")
    stmt = text(
        f"""
        SELECT prediction
        FROM {table_name}
        WHERE model_id = :model_id
          AND instrument_id = :instrument_id
        ORDER BY ts_event
        """
    )
    with store.engine.connect() as conn:
        rows = conn.execute(
            stmt,
            {
                "model_id": "model-A",
                "instrument_id": instrument_id_str,
            },
        ).fetchall()

    predictions = [float(value) for (value,) in rows]
    assert predictions == [0.7, 0.2]
