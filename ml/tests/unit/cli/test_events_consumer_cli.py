from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from types import ModuleType
from typing import Any

from ml.cli.events_consumer import main


class DummyRedis:
    class Redis:
        """
        Stub of redis.Redis with primed xread.
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


def test_events_consumer_cli_prints_filtered_events() -> None:
    # Inject dummy redis module and prime entries
    dummy_module = ModuleType("redis")
    setattr(dummy_module, "Redis", DummyRedis.Redis)
    sys.modules["redis"] = dummy_module

    try:
        client = DummyRedis.Redis()
        payload = {
            "dataset_id": "features",
            "instrument_id": "EURUSD.SIM",
            "source": "historical",
            "ts_max": 100,
            "metadata": {"correlation_id": "CID-1"},
        }
        batch = [
            (
                "ml-events",
                [
                    (
                        "1-0",
                        {
                            "topic": "events.ml.FEATURE_COMPUTED.EURUSD.SIM",
                            "payload": json.dumps(payload),
                        },
                    ),
                ],
            ),
        ]
        # Monkeypatch by attaching to Dummy instance used after construction
        out = io.StringIO()
        with redirect_stdout(out):
            # Call CLI main with one iteration and a pattern filter that matches
            rc = main(
                [
                    "--redis-url",
                    "redis://",
                    "--stream",
                    "ml-events",
                    "--pattern",
                    "events.ml.FEATURE_COMPUTED.#",
                    "--iterations",
                    "1",
                    "--count",
                    "100",
                ],
            )
            # Attach primed client after consumer construction inside main
            # Not applicable here: main() constructs its own consumer and immediately polls.
            # Instead, we simulate the redis module returning a client with primed batches
            # by setting Redis.from_url to return our client instance.
        # Re-run with patched constructor
        sys.modules["redis"].Redis.from_url = lambda *args, **kwargs: client  # type: ignore[attr-defined]
        client.prime(batch)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(
                [
                    "--redis-url",
                    "redis://",
                    "--stream",
                    "ml-events",
                    "--pattern",
                    "events.ml.FEATURE_COMPUTED.#",
                    "--iterations",
                    "1",
                    "--count",
                    "100",
                ],
            )
        assert rc == 0
        lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
        assert len(lines) == 1
        doc = json.loads(lines[0])
        assert doc["topic"].startswith("events.ml.FEATURE_COMPUTED")
        assert doc["payload"]["metadata"]["correlation_id"] == "CID-1"
    finally:
        sys.modules.pop("redis", None)
