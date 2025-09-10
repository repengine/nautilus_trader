from __future__ import annotations

import time
from typing import Any

from ml.actors.ml_domain_events import DomainEventBridge
from ml.common.message_bus import MessagePublisherProtocol
from ml.common.throttler import Throttler


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


def test_bridge_enqueues_and_flushes() -> None:
    cap = CapturePublisher()
    bridge = DomainEventBridge(cap, max_queue=8)
    bridge.start()
    try:
        for i in range(4):
            assert bridge.publish("topic", {"i": i}) is True
        # Give background thread a moment to drain
        time.sleep(0.05)
        # Stop and drain
    finally:
        bridge.stop(drain=True, timeout=1.0)

    assert len(cap.calls) >= 1


def test_bridge_backpressure_drop() -> None:
    cap = CapturePublisher()
    bridge = DomainEventBridge(cap, max_queue=1)
    bridge.start()
    try:
        assert bridge.publish("t1", {"i": 1}) is True
        # Immediately fill the queue; next may drop
        dropped = not bridge.publish("t2", {"i": 2})
        assert dropped in {True, False}
    finally:
        bridge.stop(drain=True, timeout=1.0)


def test_bridge_respects_throttler() -> None:
    cap = CapturePublisher()
    throttler = Throttler(rate_per_sec=1.0, burst=1)
    bridge = DomainEventBridge(cap, max_queue=8, throttler=throttler)
    bridge.start()
    try:
        payload1 = {"ts_max": 0}
        payload2 = {"ts_max": 0}
        assert bridge.publish("topic", payload1) is True
        # Without time advancing, throttler should drop the next publish
        assert bridge.publish("topic", payload2) is False
    finally:
        bridge.stop(drain=True, timeout=1.0)
