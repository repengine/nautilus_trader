from __future__ import annotations

from typing import Any

import pytest

from ml.stores.model_store import ModelStore
from nautilus_trader.model.identifiers import InstrumentId

# Integration DB-backed read/write
pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.usefixtures("clean_postgres_db_module"),
]


@pytest.mark.skip(
    reason=(
        "Flaky on some hosts due to pandas read_sql connection reuse; covered by publish-mode and flush-timer tests"
    ),
)
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

    # Read back in a wide range
    start_ns = ts_event - 1000
    end_ns = ts_event + 10_000
    df = store.read_predictions("model-A", instrument_id_str, start_ns, end_ns)
    assert len(df) == 2
    assert list(df["prediction"]) == [0.7, 0.2]
