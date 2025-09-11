from __future__ import annotations

from typing import Any

from ml.common.cascade import emit_cascade
from ml.common.in_memory_bus import InMemoryPublisher
from ml.common.message_topics import build_stage_topic
from ml.consumers.idempotent import IdempotentConsumer


def test_end_to_end_cascade_preserves_correlation_and_order() -> None:
    """
    End-to-end cascade across domains preserves correlation and timestamp order.

    Chain: data -> features -> models -> strategies, with positive delays.

    """
    src = {
        "domain": "data",
        "event_type": "INGESTED",
        "correlation_id": "CID-123",
        "instrument_id": "EURUSD.SIM",
        "ts_event": 1_000,
        "event_id": "E-1",
        "payload": {"dataset_id": "data"},
    }

    e_feat = emit_cascade(src, target_domain="features", delay_ns=10)
    e_model = emit_cascade(dict(e_feat), target_domain="models", delay_ns=20)
    e_strat = emit_cascade(dict(e_model), target_domain="strategies", delay_ns=30)

    # Correlation preserved and timestamps monotonic
    assert e_feat["correlation_id"] == src["correlation_id"]
    assert e_model["correlation_id"] == src["correlation_id"]
    assert e_strat["correlation_id"] == src["correlation_id"]
    assert e_feat["ts_event"] > src["ts_event"]
    assert e_model["ts_event"] > e_feat["ts_event"]
    assert e_strat["ts_event"] > e_model["ts_event"]


def test_idempotent_consumer_with_cascaded_feature_events() -> None:
    """
    IdempotentConsumer gates duplicate correlation IDs and decreasing watermarks for
    cascaded events.
    """
    bus = InMemoryPublisher()
    consumer = IdempotentConsumer()
    processed: list[dict[str, Any]] = []

    def handler(_topic: str, payload: dict[str, Any]) -> None:
        if consumer.process(payload):
            processed.append(payload)

    bus.subscribe("events.ml.FEATURE_COMPUTED.#", handler)

    # Create a base data event and cascade to a feature event
    base = {
        "domain": "data",
        "event_type": "INGESTED",
        "correlation_id": "CID-999",
        "instrument_id": "EURUSD.SIM",
        "ts_event": 10_000,
        "event_id": "E-DATA",
        "payload": {"dataset_id": "data"},
    }
    feat = emit_cascade(base, target_domain="features", delay_ns=100)

    # Build feature-computed payload for consumer schema
    p1 = {
        "dataset_id": "features",
        "instrument_id": feat["instrument_id"],
        "source": "historical",
        "ts_min": feat["ts_event"],
        "ts_max": feat["ts_event"],
        "count": 1,
        "status": "success",
        "metadata": {"correlation_id": feat["correlation_id"]},
    }
    # Duplicate correlation
    p_dup = dict(p1)
    p_dup["metadata"] = {"correlation_id": feat["correlation_id"]}
    # Lower watermark
    p_low = dict(p1)
    p_low["ts_max"] = feat["ts_event"] - 1
    p_low["metadata"] = {"correlation_id": "CID-OTHER"}

    topic = build_stage_topic("FEATURE_COMPUTED", feat["instrument_id"])  # stage-first
    assert bus.publish(topic, p1) is True
    assert bus.publish(topic, p_dup) is True
    assert bus.publish(topic, p_low) is True

    assert len(processed) == 1
    assert processed[0]["metadata"]["correlation_id"] == feat["correlation_id"]
