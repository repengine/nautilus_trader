#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from collections.abc import Sequence
from typing import cast

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.tasks.pipelines import PipelineScheduleConfig
from ml.tasks.pipelines import run_pipeline_schedule


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the ML pipeline orchestrator on a schedule")
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--schedule-time",
        dest="schedule_time",
        help="Daily UTC time HH:MMZ (e.g., 02:30Z)",
    )
    group.add_argument(
        "--interval-min",
        dest="interval_min",
        type=int,
        help="Interval in minutes (e.g., 1440)",
    )
    ap.add_argument(
        "--config",
        dest="config_path",
        help="Path to orchestrator JSON/TOML config",
        default=None,
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without invoking orchestrator",
    )
    ap.add_argument("--force", action="store_true", help="Ignore existing outputs and run anyway")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()
    bind_log_context(run_id=f"cli_pipeline_scheduler_{os.getpid()}", component="ml.cli.pipeline_scheduler")

    from ml.cli.pipeline_orchestrator import main as _orch_main

    schedule_config = PipelineScheduleConfig(
        schedule_time=args.schedule_time,
        interval_minutes=args.interval_min,
        config_path=args.config_path,
        dry_run=args.dry_run,
        force=args.force,
    )
    run_pipeline_schedule(
        schedule_config,
        invoke_pipeline=cast(Callable[[Sequence[str] | None], int], _orch_main),
        sleep_fn=time.sleep,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    from typing import cast

    raise SystemExit(main())
