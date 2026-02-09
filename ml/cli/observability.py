#!/usr/bin/env python3
"""
CLI entrypoint for observability flush operations.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ml.core.common.observability import ObservabilityComponent
from ml.core.common.observability import normalize_async_worker_status
from ml.core.common.observability import seed_sample_observability
from ml.observability.scheduler import ObservabilityStartConfig
from ml.observability.scheduler import run_observability_start


__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Observability flush CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_jsonl = sub.add_parser("flush-jsonl", help="Flush to JSONL/CSV")
    p_jsonl.add_argument("--base-path", required=True)
    p_jsonl.add_argument("--format", default="jsonl", choices=["jsonl", "csv"])
    p_jsonl.add_argument("--seed-sample", action="store_true")

    p_db = sub.add_parser("flush-db", help="Flush to DB")
    p_db.add_argument("--db-url", required=True)
    p_db.add_argument("--seed-sample", action="store_true")

    p_start = sub.add_parser("start", help="Start background flushing (thread or async)")
    p_start.add_argument("--sink", default="file", choices=["file", "db"], help="file or db sink")
    p_start.add_argument("--base-path", default="./observability", help="base path when sink=file")
    p_start.add_argument("--format", default="jsonl", choices=["jsonl", "csv"], help="file format")
    p_start.add_argument("--db-url", default=None, help="DB URL when sink=db")
    p_start.add_argument("--interval", type=float, default=60.0, help="flush interval seconds")
    p_start.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="run duration seconds (0=until SIGTERM)",
    )
    p_start.add_argument(
        "--seed-sample",
        action="store_true",
        help="seed a sample set of rows before start",
    )
    p_start.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="use async worker instead of thread flusher",
    )
    p_start.add_argument(
        "--async-queue",
        type=int,
        default=4096,
        help="async worker bounded queue size",
    )
    p_start.add_argument(
        "--async-component",
        type=str,
        default="obs_async_worker",
        help="component label for metrics",
    )

    sub.add_parser("status", help="Show observability async worker status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    component = ObservabilityComponent()

    if args.cmd == "flush-jsonl":
        if args.seed_sample:
            seed_sample_observability(component)
        out = component.flush_observability_to_path(
            base_path=Path(str(args.base_path)),
            file_format=str(args.format),
        )
        for table_name, file_path in out.items():
            print(f"{table_name}: {file_path}")
        return 0

    if args.cmd == "flush-db":
        if args.seed_sample:
            seed_sample_observability(component)
        out_db = component.flush_observability_to_db(
            connection_string=str(args.db_url),
        )
        for table_name, row_count in out_db.items():
            print(f"{table_name}: {row_count}")
        return 0

    if args.cmd == "start":
        if args.seed_sample:
            seed_sample_observability(component)
        config = ObservabilityStartConfig(
            sink=str(args.sink),
            base_path=Path(str(args.base_path)),
            file_format=str(args.format),
            db_url=str(args.db_url) if args.db_url is not None else None,
            interval_seconds=float(args.interval),
            duration_seconds=float(args.duration),
            async_enabled=bool(args.async_mode),
            async_queue_maxsize=int(args.async_queue),
            async_component_label=str(args.async_component),
        )
        return run_observability_start(component, config)

    status = component.get_observability_async_status()
    running, queue_size = normalize_async_worker_status(status)
    print(f"async_worker_running={running} queue_size={queue_size}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
