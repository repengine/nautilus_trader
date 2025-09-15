from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import Any

from ml.consumers.redis_streams_consumer import RedisStreamsConsumer


class DummyRedis:
    class Redis:
        """
        Simple stub of redis.Redis with a primed xread queue.
        """

        def __init__(self) -> None:
            self._batches: list[list[tuple[str, list[tuple[str, dict[str, str]]]]]] = []

        @classmethod
        def from_url(cls, url: str, decode_responses: bool = False) -> DummyRedis.Redis:
            return cls()

        def prime(self, batch: list[tuple[str, list[tuple[str, dict[str, str]]]]]) -> None:
            self._batches.append(batch)

        def xread(
            self,
            *_args: Any,
            **_kwargs: Any,
        ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
            return self._batches.pop(0) if self._batches else []


def test_redis_streams_consumer_gates_and_handles() -> None:
    # Inject dummy redis module
    dummy_module = ModuleType("redis")
    dummy = DummyRedis.Redis()
    setattr(dummy_module, "Redis", DummyRedis.Redis)
    sys.modules["redis"] = dummy_module

    received: list[tuple[str, dict[str, Any]]] = []

    def handler(topic: str, payload: dict[str, Any]) -> None:
        received.append((topic, payload))

    try:
        # Prime xread with duplicates and lower watermark
        payload_ok = {
            "dataset_id": "features",
            "instrument_id": "EURUSD.SIM",
            "source": "historical",
            "ts_max": 100,
            "metadata": {"correlation_id": "CID-1"},
        }
        payload_dup = dict(payload_ok)
        payload_low = dict(payload_ok)
        payload_low["ts_max"] = 90
        batch = [
            (
                "ml-events",
                [
                    (
                        "1-0",
                        {
                            "topic": "events.ml.FEATURE_COMPUTED.EURUSD.SIM",
                            "payload": json.dumps(payload_ok),
                        },
                    ),
                    (
                        "2-0",
                        {
                            "topic": "events.ml.FEATURE_COMPUTED.EURUSD.SIM",
                            "payload": json.dumps(payload_dup),
                        },
                    ),
                    (
                        "3-0",
                        {
                            "topic": "events.ml.FEATURE_COMPUTED.EURUSD.SIM",
                            "payload": json.dumps(payload_low),
                        },
                    ),
                ],
            ),
        ]
        # Access the dummy to prime
        client = DummyRedis.Redis()
        client.prime(batch)
        # Monkeypatch is not used here; wire via module-level attribute
        # Construct consumer (will use DummyRedis.Redis.from_url)
        consumer = RedisStreamsConsumer(url="redis://", stream="ml-events", handler=handler)
        # Attach our primed client to the instance
        consumer._client = client
        processed = consumer.poll_once()
        assert processed == 1
        assert len(received) == 1 and received[0][0].startswith("events.ml.FEATURE_COMPUTED")
    finally:
        sys.modules.pop("redis", None)
