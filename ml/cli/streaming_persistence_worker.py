#!/usr/bin/env python3
"""CLI entrypoint for the streaming training persistence worker."""

from __future__ import annotations

import argparse
import signal
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from ml.config.bus import MessageBusConfig
from ml.config.streaming_pipeline import StreamingPersistenceConfig
from ml.consumers.streaming_training_worker import StreamingTrainingPersistenceWorker


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the persistence worker."""
    parser = argparse.ArgumentParser(
        description="Run the streaming training persistence worker against Redis Streams.",
    )
    parser.add_argument(
        "--state-path",
        dest="state_path",
        type=str,
        default=None,
        help="Override persistence snapshot path (optional).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="Maximum number of entries to process per poll (optional).",
    )
    parser.add_argument(
        "--block-ms",
        dest="block_ms",
        type=int,
        default=None,
        help="Block duration in milliseconds for Redis XREAD (optional).",
    )
    parser.add_argument(
        "--poll-interval",
        dest="poll_interval_seconds",
        type=float,
        default=None,
        help="Idle interval in seconds between polls when no messages are processed.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--enable",
        dest="enable",
        action="store_true",
        help="Force-enable the worker regardless of environment configuration.",
    )
    group.add_argument(
        "--disable",
        dest="disable",
        action="store_true",
        help="Disable the worker regardless of environment configuration.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def build_config(
    args: argparse.Namespace,
    *,
    env: Mapping[str, str] | None = None,
) -> StreamingPersistenceConfig:
    """Construct a streaming persistence config from env defaults and CLI overrides."""
    base = StreamingPersistenceConfig.from_env(env)
    enabled = base.enabled
    if getattr(args, "enable", False):
        enabled = True
    if getattr(args, "disable", False):
        enabled = False

    state_path = str(getattr(args, "state_path")) if getattr(args, "state_path", None) else base.state_path
    batch_size = int(getattr(args, "batch_size")) if getattr(args, "batch_size", None) is not None else int(base.batch_size)
    block_ms = int(getattr(args, "block_ms")) if getattr(args, "block_ms", None) is not None else int(base.block_ms)
    poll_interval = (
        float(getattr(args, "poll_interval_seconds"))
        if getattr(args, "poll_interval_seconds", None) is not None
        else float(base.poll_interval_seconds)
    )
    return StreamingPersistenceConfig(
        enabled=enabled,
        state_path=state_path,
        batch_size=batch_size,
        block_ms=block_ms,
        poll_interval_seconds=poll_interval,
    )


def _install_signal_handlers(worker: StreamingTrainingPersistenceWorker) -> None:
    """Register signal handlers for graceful shutdown."""

    def _stop_handler(_signum: int, _frame: Any) -> None:
        worker.stop()

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the streaming persistence worker CLI."""
    args = parse_args(argv)
    config = build_config(args)
    bus_config = MessageBusConfig.from_env()
    worker = StreamingTrainingPersistenceWorker(
        config=config,
        message_bus_config=bus_config,
    )
    _install_signal_handlers(worker)
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        worker.stop()
    return 0


__all__ = ["build_config", "main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
