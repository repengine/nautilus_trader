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
from typing import Any, Final

from ml.consumers.idempotent import IdempotentConsumer


OnEvent = Callable[[str, dict[str, Any]], None]


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
        gate: IdempotentConsumer | None = None,
    ) -> None:
        self._url = url
        self._stream = stream
        self._handler = handler
        self._gate = gate or IdempotentConsumer()
        try:
            import redis

            self._client = redis.Redis.from_url(url, decode_responses=True)
        except Exception:  # pragma: no cover - import failure path
            self._client = None

        self._logger = logging.getLogger(__name__)

    def poll_once(self, *, count: int = 100, block_ms: int = 0, last_id: str = "$") -> int:
        """
        Read a batch via XREAD and process accepted events. Returns processed count.

        When no client is available, returns 0.

        """
        if self._client is None:
            return 0
        try:
            results = self._client.xread({self._stream: last_id}, count=count, block=block_ms)
        except Exception:
            return 0

        processed = 0
        for _stream_name, entries in results or []:
            for _entry_id, fields in entries:
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
        return processed


__all__: Final[list[str]] = ["OnEvent", "RedisStreamsConsumer"]
