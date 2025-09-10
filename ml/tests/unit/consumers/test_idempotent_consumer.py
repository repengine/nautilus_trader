from __future__ import annotations

from typing import Any

from ml.consumers.idempotent import IdempotentConsumer


def make_payload(**kwargs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset_id": "features",
        "instrument_id": "EURUSD.SIM",
        "source": "historical",
        "ts_max": 200,
        "metadata": {"correlation_id": "cid-1"},
    }
    payload.update(kwargs)
    return payload


def test_idempotent_consumer_deduplicates_and_updates_watermark() -> None:
    c = IdempotentConsumer()
    p1 = make_payload(ts_max=100, metadata={"correlation_id": "A"})
    p2 = make_payload(ts_max=90, metadata={"correlation_id": "B"})
    p3 = make_payload(ts_max=120, metadata={"correlation_id": "A"})  # duplicate correlation
    p4 = make_payload(ts_max=130, metadata={"correlation_id": "C"})

    assert c.process(p1) is True
    # Lower ts_max than watermark -> drop
    assert c.process(p2) is False
    # Duplicate correlation -> drop
    assert c.process(p3) is False
    # New correlation and higher ts_max -> accept
    assert c.process(p4) is True
