"""
Lightweight message bus publisher protocol and adapters.

This module defines a minimal, typed interface for publishing ML events to an
external message bus. A `NoopPublisher` is provided as a safe default, and an
optional Redis Streams adapter can be enabled via configuration.

"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from ml.config.bus import MessageBusConfig


@runtime_checkable
class MessagePublisherProtocol(Protocol):
    """
    Protocol for message bus publishers.
    """

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        """
        Publish a payload to a topic; returns True on success.
        """
        ...


class NoopPublisher:
    """
    No-op publisher implementation (safe default).
    """

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        return False


class RedisStreamsPublisher:
    """
    Redis Streams publisher (optional backend).

    Publishes events to a configured Redis stream using XADD with fields {"topic":
    topic, "payload": json-string}. This adapter is best-effort; errors are caught and
    reported via return value False.

    """

    def __init__(self, *, url: str, stream: str, maxlen: int | None = None) -> None:
        self._url = url
        self._stream = stream
        self._maxlen = maxlen
        self._client: Any | None = None
        try:
            import redis

            # Create a simple client; decode_responses for str payloads
            self._client = redis.Redis.from_url(url, decode_responses=True)
        except Exception:  # pragma: no cover - import failure path
            self._client = None

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        try:
            if self._client is None:
                return False
            fields: dict[str, str] = {
                "topic": topic,
                "payload": json.dumps(payload, separators=(",", ":")),
            }
            if self._maxlen is not None and self._maxlen > 0:
                self._client.xadd(self._stream, fields, maxlen=self._maxlen, approximate=True)
            else:
                self._client.xadd(self._stream, fields)
            return True
        except Exception:
            return False


def publisher_from_config(cfg: MessageBusConfig) -> MessagePublisherProtocol:
    """
    Construct a publisher from the given configuration.

    Returns a `NoopPublisher` when disabled or for unknown backends to keep
    callers safe by default.

    """
    if not cfg.enabled:
        return NoopPublisher()
    if cfg.backend == "redis":
        return RedisStreamsPublisher(
            url=cfg.redis_url,
            stream=cfg.redis_stream,
            maxlen=cfg.redis_maxlen,
        )
    return NoopPublisher()


__all__ = [
    "MessagePublisherProtocol",
    "NoopPublisher",
    "RedisStreamsPublisher",
    "publisher_from_config",
]
