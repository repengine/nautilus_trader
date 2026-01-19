#!/usr/bin/env python3
"""Replay message bus JSONL payloads at live pace."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ml.common.logging_config import configure_logging
from ml.common.message_bus import publisher_from_config
from ml.config.bus import MessageBusConfig
from ml.config.replay import LiveReplayConfig
from ml.deployment.live_paced_replay import run_live_paced_replay


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay JSONL bus payloads at live pace.")
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        required=True,
        help="Path to JSONL file containing topic/payload records.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Speed multiplier for pacing (1.0 = real time).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events to replay.",
    )
    parser.add_argument(
        "--timestamp-field",
        type=str,
        default=None,
        help="Primary payload field used for pacing (default: ts_min).",
    )
    return parser


def main() -> None:
    configure_logging()
    args = _build_parser().parse_args()

    config = LiveReplayConfig(
        speed_multiplier=args.speed if args.speed is not None else 1.0,
        max_events=args.max_events,
        timestamp_field=args.timestamp_field or "ts_min",
    )
    bus_config = MessageBusConfig.from_env()
    publisher = publisher_from_config(bus_config)

    summary = run_live_paced_replay(
        args.input_path,
        publisher,
        config=config,
    )

    logger.info(
        "live_replay_complete",
        extra={
            "loaded": summary.loaded,
            "published": summary.published,
            "failed": summary.failed,
            "skipped": summary.skipped,
            "duration_seconds": summary.duration_seconds,
        },
    )


if __name__ == "__main__":
    main()
