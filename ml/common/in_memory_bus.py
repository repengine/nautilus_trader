"""
In-memory publisher with wildcard subscriptions for tests/examples.

This lightweight pub/sub helper implements MessagePublisherProtocol and adds a
``subscribe`` method to register handlers by pattern using topic_filters.match_topic.
It is intended for unit/integration tests and local examples.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from ml.common.message_bus import MessagePublisherProtocol
from ml.common.topic_filters import match_topic


Handler = Callable[[str, dict[str, Any]], None]


class InMemoryPublisher(MessagePublisherProtocol):
    """
    In-memory pub/sub implementation for testing.
    """

    def __init__(self) -> None:
        self._subs: list[tuple[str, Handler]] = []

    def subscribe(self, pattern: str, handler: Handler) -> None:
        self._subs.append((pattern, handler))

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        delivered = False
        for pattern, handler in self._subs:
            if match_topic(pattern, topic):
                handler(topic, payload)
                delivered = True
        return delivered


__all__: Final[list[str]] = ["Handler", "InMemoryPublisher"]
