from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

import pytest
from typing import Any, cast

from ml.common.in_memory_bus import InMemoryPublisher
from ml.consumers.aggregator import AggregatingConsumer
from ml.consumers.lineage_writer import LineageWriter
from ml.consumers.protocols import Envelope
from ml.observability.service import ObservabilityService


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

def test_aggregator_to_lineage_writer_flow() -> None:
    svc = ObservabilityService()
    writer = LineageWriter(service=svc)

    bus = InMemoryPublisher()

    def _forward(topic: str, payload: dict[str, Any]) -> None:
        writer.handle(topic, cast(Envelope, payload))

    # Wire a subscriber that forwards to the writer
    bus.subscribe("aggregated.#", _forward)

    agg = AggregatingConsumer(downstream=bus, topic_mapper=lambda _stage: "aggregated.lineage")

    # Two out-of-order envelopes for same instrument
    e1: Envelope = {
        "id": "e1",
        "parent_id": None,
        "instrument_id": "EURUSD.SIM",
        "ts_event": 20,
        "stage": "FEATURE_COMPUTED",
        "correlation_id": "c1",
        "payload": {},
    }
    e0: Envelope = {**e1, "id": "e0", "ts_event": 10}

    agg.handle("events.ml.FEATURE_COMPUTED", e1)
    agg.handle("events.ml.FEATURE_COMPUTED", e0)

    # Flush with watermark=100 to emit both, in order
    flushed = agg.advance_watermark("EURUSD.SIM", 100)
    assert [e["id"] for e in flushed] == ["e0", "e1"]

    df = svc.event_correlation_df()
    assert set(df.columns) >= {
        "correlation_id",
        "event_id",
        "parent_event_id",
        "instrument_id",
        "domain",
        "lineage_depth",
        "ts_event",
        "propagation_path",
    }
    assert list(df.sort_values("ts_event")["event_id"]) == ["e0", "e1"]
