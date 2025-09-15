"""
DB-backed read/stat/performance tests for ModelStore.

Writes a few predictions, flushes, and asserts read APIs and stats populate
expected fields and types.
"""

from __future__ import annotations

import time

import pytest

from ml.stores.model_store import ModelStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.usefixtures("clean_postgres_db_module"),
]


def test_model_store_reads_stats_and_performance(test_database) -> None:
    store = ModelStore(connection_string=test_database.connection_string)

    model_id = "modelA"
    instrument_id = "EUR/USD"
    base_ts = int(time.time_ns())

    preds = [
        (0.55, 0.80, 12.0, base_ts + 1),
        (0.60, 0.75, 8.0, base_ts + 2),
        (0.58, 0.70, 20.0, base_ts + 3),
    ]
    for pred, conf, latency_ms, ts in preds:
        store.write_prediction(
            model_id=model_id,
            instrument_id=instrument_id,
            prediction=pred,
            confidence=conf,
            features={"f": pred},
            inference_time_ms=latency_ms,
            ts_event=ts,
        )

    store.flush()

    # Latest returns most recent first
    latest = store.get_latest(instrument_id=instrument_id, limit=2)
    assert len(latest.index) == 2
    assert int(latest.iloc[0]["ts_event"]) == preds[-1][3]
    assert int(latest.iloc[1]["ts_event"]) == preds[-2][3]

    # Stats have expected counts and timestamp bounds
    stats = store.get_statistics(start_ns=preds[0][3], end_ns=preds[-1][3] + 1)
    assert stats["total_predictions"] == 3
    assert stats["unique_models"] >= 1
    assert stats["unique_instruments"] >= 1
    assert isinstance(stats["avg_inference_ms"], float)
    assert isinstance(stats["max_inference_ms"], float)
    assert int(stats["min_timestamp_ns"]) == preds[0][3]
    assert int(stats["max_timestamp_ns"]) == preds[-1][3]

    # Model performance fields are populated with correct types
    perf = store.get_model_performance(model_id=model_id)
    assert perf["prediction_count"] == 3
    for key in [
        "avg_confidence",
        "std_confidence",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
    ]:
        assert isinstance(perf[key], float)
        assert perf[key] >= 0.0

