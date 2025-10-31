"""
Redis Streams consumer example with idempotent gating.

This consumer reads from a Redis stream (fields: topic, payload JSON), applies
idempotent + watermark gating via IdempotentConsumer, and invokes a handler only for
accepted events. It is intended as a minimal, safe example and is not meant to run on
the hot path.

"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from ml import _imports as ml_imports
from ml.consumers.idempotent import IdempotentConsumer


if TYPE_CHECKING:
    from redis import Redis as _RedisClient
else:
    _RedisClient = Any


OnEvent = Callable[[str, dict[str, Any]], None]


class Gate(Protocol):
    """Protocol describing the gating interface for Redis stream consumers."""

    def process(self, payload: Mapping[str, Any]) -> bool:
        """Return True when the payload should be processed."""


class RedisStreamsConsumer:
    """
    Minimal Redis Streams consumer that gates events and invokes a handler.
    """

    def __init__(
        self,
        *,
        url: str,
        stream: str,
        handler: OnEvent,
        gate: Gate | None = None,
    ) -> None:
        self._url = url
        self._stream = stream
        self._handler = handler
        self._gate = gate or IdempotentConsumer()
        self._client: _RedisClient | None = None
        self._last_entry_id: str | None = None
        self._logger = logging.getLogger(__name__)
        redis_module = ml_imports.redis
        if redis_module is None:
            try:
                import importlib

                redis_module = importlib.import_module("redis")
                setattr(ml_imports, "redis", redis_module)
                setattr(ml_imports, "HAS_REDIS", True)
            except Exception:  # pragma: no cover - redis genuinely unavailable
                redis_module = None
        if redis_module is None:
            self._logger.debug(
                "redis dependency unavailable; consumer disabled",
                extra={"url": url},
            )
            return
        try:
            client = redis_module.Redis.from_url(url, decode_responses=True)
            self._client = cast(_RedisClient, client)
        except Exception:  # pragma: no cover - import failure path
            self._client = None

    @property
    def last_entry_id(self) -> str | None:
        """Return the last processed Redis stream ID."""
        return self._last_entry_id

    def poll_once(self, *, count: int = 100, block_ms: int = 0, last_id: str = "$") -> int:
        """
        Read a batch via XREAD and process accepted events. Returns processed count.

        When no client is available, returns 0.

        """
        if self._client is None:
            return 0
        try:
            raw_results = self._client.xread({self._stream: last_id}, count=count, block=block_ms)
        except Exception:
            return 0

        results = cast(
            list[tuple[str, list[tuple[str, dict[str, str]]]]],
            raw_results or [],
        )

        processed = 0
        last_processed: str | None = None
        for _stream_name, entries in results:
            for entry_id, fields in entries:
                last_processed = entry_id
                topic = fields.get("topic", "")
                payload_raw = fields.get("payload", "{}")
                try:
                    payload = json.loads(payload_raw)
                except Exception:
                    payload = {}
                # Only pass through accepted events
                if self._gate.process(payload):
                    try:
                        self._handler(topic, payload)
                        processed += 1
                    except Exception as exc:
                        # Log and continue to keep the loop resilient in example code.
                        self._logger.debug("consumer handler error: %s", exc, exc_info=True)
                        continue
        if last_processed is not None:
            self._last_entry_id = last_processed
        return processed


__all__: Final[list[str]] = ["Gate", "OnEvent", "RedisStreamsConsumer"]
