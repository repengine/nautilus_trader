# ruff: noqa: I001
import time

from sqlalchemy import func, select, text

from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore


def test_modelstore_sanitizes_seconds_to_ns(monkeypatch, tmp_path) -> None:
    captured: list[dict] = []

    db_path = tmp_path / "model_store.db"
    store = ModelStore(connection_string=f"sqlite:///{db_path}", batch_size=1)

    def fake_execute(values):
        captured.extend(values)

    monkeypatch.setattr(store, "_execute_write", fake_execute)

    secs = int(time.time())  # seconds
    store.write_prediction(
        model_id="m1",
        instrument_id="EUR/USD",
        prediction=0.5,
        confidence=0.8,
        features={"f": 1.0},
        inference_time_ms=1.2,
        ts_event=secs,
        is_live=False,
    )

    assert captured, "write_prediction should have invoked _execute_write"
    ts_event = captured[0]["ts_event"]
    assert ts_event != secs
    assert ts_event >= 10**12  # should be nanoseconds scale


def test_featurestore_upsert_idempotent(tmp_path) -> None:
    db_path = tmp_path / "feature_store.db"
    store = FeatureStore(connection_string=f"sqlite:///{db_path}")

    row = {
        "feature_set_id": "fs_test",
        "instrument_id": "EUR/USD",
        "ts_event": int(time.time()),  # seconds (will be sanitized)
        "ts_init": int(time.time()),
        "values": {"x": 1.0},
        "is_live": False,
        "source": "computed",
    }

    # Upsert twice with identical key
    store._execute_write(row.copy())
    store._execute_write(row.copy())

    # Verify only one logical row exists
    with store.engine.connect() as conn:
        try:
            result = conn.execute(
                select(func.count()).select_from(store.feature_values_table),
            ).scalar()
        except Exception:
            result = conn.execute(text("SELECT COUNT(*) FROM ml_feature_values")).scalar()

    assert int(result or 0) == 1
