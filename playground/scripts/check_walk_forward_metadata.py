"""
CLI utility to validate walk-forward metadata against shared defaults.

Usage:
    poetry run python -m playground.scripts.check_walk_forward_metadata \
        --metadata playground/reports/backtesting/walk_forward/metadata.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from playground.backtest.monitoring import log_walk_forward_metadata


LOGGER = structlog.get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check walk-forward metadata for drift against defaults.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("playground/reports/backtesting/walk_forward/metadata.json"),
        help="Path to walk-forward metadata JSON (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    drifts = log_walk_forward_metadata(args.metadata)
    if drifts:
        LOGGER.error(
            "Walk-forward metadata drift detected",
            path=str(args.metadata),
            drift_count=len(drifts),
        )
        return 1
    LOGGER.info("Walk-forward metadata validated successfully", path=str(args.metadata))
    return 0


if __name__ == "__main__":
    sys.exit(main())
