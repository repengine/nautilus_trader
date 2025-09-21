#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from collections.abc import Sequence
from typing import cast

from ml.orchestration import config_loader as _config_loader
from ml.orchestration.scheduler import run_forever as _run_forever


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

    # Configure environment fallbacks for the scheduler loop
    if args.schedule_time:
        os.environ["ORCH_SCHEDULE_TIME"] = str(args.schedule_time)
        os.environ.pop("ORCH_INTERVAL_MIN", None)
    elif args.interval_min is not None:
        os.environ["ORCH_INTERVAL_MIN"] = str(int(args.interval_min))
        os.environ.pop("ORCH_SCHEDULE_TIME", None)

    if args.config_path:
        os.environ["ORCH_CONFIG"] = str(args.config_path)
    if args.dry_run:
        os.environ["ORCH_DRY_RUN"] = "1"
    if args.force:
        os.environ["ORCH_FORCE"] = "1"

    # Invoke forever using orchestrator CLI main
    from ml.cli.pipeline_orchestrator import main as _orch_main

    _run_forever(
        config_loader=_config_loader,
        invoke_pipeline=cast(Callable[[Sequence[str] | None], int], _orch_main),
        sleep_fn=time.sleep,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    from typing import cast

    raise SystemExit(main())
