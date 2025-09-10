from __future__ import annotations

from typing import Any

from ml.common.in_memory_bus import InMemoryPublisher


def test_in_memory_bus_wildcard_delivery() -> None:
    bus = InMemoryPublisher()
    received: list[tuple[str, dict[str, Any]]] = []

    def handler(topic: str, payload: dict[str, Any]) -> None:
        received.append((topic, payload))

    bus.subscribe("events.ml.FEATURE_COMPUTED.#", handler)
    delivered1 = bus.publish("events.ml.FEATURE_COMPUTED.EURUSD.SIM", {"x": 1})
    delivered2 = bus.publish("events.ml.PREDICTION_EMITTED.EURUSD.SIM", {"x": 2})

    assert delivered1 is True and delivered2 is False
    assert len(received) == 1 and received[0][1]["x"] == 1
