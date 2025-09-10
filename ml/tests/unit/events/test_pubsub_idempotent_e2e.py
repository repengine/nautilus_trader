from __future__ import annotations

from typing import Any

from ml.common.in_memory_bus import InMemoryPublisher
from ml.common.message_topics import build_stage_topic
from ml.consumers.idempotent import IdempotentConsumer


def test_e2e_stage_first_pubsub_with_idempotent_consumer() -> None:
    bus = InMemoryPublisher()
    consumer = IdempotentConsumer()
    processed: list[dict[str, Any]] = []

    def handler(_topic: str, payload: dict[str, Any]) -> None:
        if consumer.process(payload):
            processed.append(payload)

    bus.subscribe("events.ml.FEATURE_COMPUTED.#", handler)

    # Events with same correlation should be deduped; lower ts_max should drop via watermark gate
    p_ok = {
        "dataset_id": "features",
        "instrument_id": "EURUSD.SIM",
        "source": "historical",
        "ts_min": 0,
        "ts_max": 100,
        "count": 1,
        "status": "success",
        "metadata": {"correlation_id": "CID-1"},
    }
    p_dup = dict(p_ok)
    p_dup["ts_max"] = 100
    p_dup["metadata"] = {"correlation_id": "CID-1"}
    p_low = dict(p_ok)
    p_low["ts_max"] = 90
    p_low["metadata"] = {"correlation_id": "CID-2"}

    topic = build_stage_topic("FEATURE_COMPUTED", "EURUSD.SIM")
    assert bus.publish(topic, p_ok) is True
    assert bus.publish(topic, p_dup) is True
    assert bus.publish(topic, p_low) is True

    assert len(processed) == 1 and processed[0]["metadata"]["correlation_id"] == "CID-1"
