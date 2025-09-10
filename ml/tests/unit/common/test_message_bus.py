from __future__ import annotations

from typing import Any

from ml.common.message_bus import MessagePublisherProtocol
from ml.common.message_bus import NoopPublisher


class DummyPublisher:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.published.append((topic, payload))
        return True


class TestMessageBus:
    def test_noop_publisher_returns_false(self) -> None:
        pub = NoopPublisher()
        ok = pub.publish("ml.data.created.EURUSD.SIM", {"x": 1})
        assert ok is False

    def test_protocol_is_respected(self) -> None:
        pub: MessagePublisherProtocol = DummyPublisher()
        ok = pub.publish("ml.data.created.EURUSD.SIM", {"x": 1})
        assert ok is True
