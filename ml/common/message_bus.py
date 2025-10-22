"""
Lightweight message bus publisher protocol and adapters.

This module defines a minimal, typed interface for publishing ML events to an
external message bus. A `NoopPublisher` is provided as a safe default, and an
optional Redis Streams adapter can be enabled via configuration.

"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from ml import _imports as ml_imports
from ml.config.bus import MessageBusConfig


if TYPE_CHECKING:
    from redis import Redis as _RedisClient
else:
    _RedisClient = Any


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
        self._client: _RedisClient | None = None
        redis_module = ml_imports.redis
        if not ml_imports.HAS_REDIS or redis_module is None:
            # Optional dependency missing; publisher remains inactive.
            return
        try:
            client = redis_module.Redis.from_url(url, decode_responses=True)
            self._client = cast(_RedisClient, client)
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
            redis_fields = cast(
                dict[
                    bytes | bytearray | memoryview | str | int | float,
                    bytes | bytearray | memoryview | str | int | float,
                ],
                fields,
            )
            if self._maxlen is not None and self._maxlen > 0:
                self._client.xadd(
                    self._stream,
                    redis_fields,
                    maxlen=self._maxlen,
                    approximate=True,
                )
            else:
                self._client.xadd(self._stream, redis_fields)
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
        if not ml_imports.HAS_REDIS or ml_imports.redis is None:
            return NoopPublisher()
        return RedisStreamsPublisher(
            url=cfg.redis_url,
            stream=cfg.redis_stream,
            maxlen=cfg.redis_maxlen,
        )
    return NoopPublisher()


class BusPublisherMixin:
    """
    Mixin providing standardized bus publishing attributes and initialization.

    Call `_init_bus_publishing(...)` in a constructor to set:
    - `_enable_publishing`: bool
    - `publisher`: MessagePublisherProtocol | None
    - `_publish_mode`: Literal["batch", "row", "both"]
    - `_topic_scheme`: str
    - `_topic_prefix`: str

    """

    def _init_bus_publishing(
        self,
        *,
        enable_publishing: bool,
        publisher: MessagePublisherProtocol | None,
        publish_mode: Literal["batch", "row", "both"] = "batch",
    ) -> None:
        # Basic flags
        self._enable_publishing = bool(enable_publishing)
        self.publisher = publisher
        self._publish_mode = publish_mode

        # Topic scheme/prefix (env-driven defaults)
        try:
            from ml.config.bus import MessageBusConfig as _MBC

            _cfg = _MBC.from_env()
            self._topic_scheme = str(_cfg.scheme)
            self._topic_prefix = str(_cfg.topic_prefix)
        except Exception:  # pragma: no cover - defensive
            # Sensible defaults consistent with existing code
            self._topic_scheme = "domain_op"
            self._topic_prefix = "events.ml"


__all__ = [
    "BusPublisherMixin",
    "MessagePublisherProtocol",
    "NoopPublisher",
    "RedisStreamsPublisher",
    "publisher_from_config",
]
