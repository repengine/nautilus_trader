"""
Events Consumer CLI (Redis Streams + Idempotent Gating).
This CLI subscribes to a Redis stream (fields: topic, payload JSON), applies
idempotent + watermark gating, and prints accepted events. It supports optional
topic pattern filtering using wildcard semantics (see ml.common.topic_filters).

Examples
--------
uv run -m ml.cli.events_consumer \
  --redis-url redis://localhost:6379/0 \
  --stream ml-events \
  --pattern events.ml.FEATURE_COMPUTED.# \
  --iterations 1 --count 100

Notes: This is an example tool and is not designed for hot-path usage.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from ml.common.topic_filters import match_topic
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer


def _env(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val is not None and val != "" else default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Events consumer (Redis Streams)")
    parser.add_argument(
        "--redis-url",
        dest="redis_url",
        type=str,
        default=_env("ML_BUS_REDIS_URL", "redis://localhost:6379/0"),
        help="Redis connection URL (default: env ML_BUS_REDIS_URL or redis://localhost:6379/0)",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        type=str,
        default=_env("ML_BUS_REDIS_STREAM", "ml-events"),
        help="Redis stream name (default: env ML_BUS_REDIS_STREAM or ml-events)",
    )
    parser.add_argument(
        "--pattern",
        dest="patterns",
        action="append",
        default=[],
        help="Topic pattern to filter (wildcards: * and #). May be repeated.",
    )
    parser.add_argument(
        "--count",
        dest="count",
        type=int,
        default=100,
        help="Max messages to read per iteration (default: 100)",
    )
    parser.add_argument(
        "--block-ms",
        dest="block_ms",
        type=int,
        default=0,
        help="XREAD block duration in ms (default: 0 = non-blocking)",
    )
    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        default=1,
        help="Number of poll iterations (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    patterns: list[str] = [str(p) for p in (args.patterns or [])]

    def handler(topic: str, payload: dict[str, Any]) -> None:
        if patterns and not any(match_topic(p, topic) for p in patterns):
            return
        out = {"topic": topic, "payload": payload}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()

    consumer = RedisStreamsConsumer(
        url=str(args.redis_url),
        stream=str(args.stream),
        handler=handler,
    )

    iterations = int(args.iterations)
    for _ in range(max(iterations, 1)):
        consumer.poll_once(count=int(args.count), block_ms=int(args.block_ms))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
