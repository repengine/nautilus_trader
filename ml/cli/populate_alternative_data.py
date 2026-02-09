#!/usr/bin/env python3
"""
Thin CLI wrapper for alternative data population.
"""

from __future__ import annotations

import argparse
import sys
import uuid as _uuid
from pathlib import Path

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.data.ingest import PopulateAlternativeDataTaskConfig
from ml.data.ingest import populate_alternative_data_task
from ml.data.loaders.alternative import AlternativeDataResult
from ml.data.loaders.alternative import AlternativeSource


__all__ = ["main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate alternative data sources")
    parser.add_argument("--all", action="store_true", help="Populate all data sources")
    parser.add_argument(
        "--source",
        action="append",
        choices=[member.value for member in AlternativeSource],
        help="Specific data source to populate (can be repeated)",
    )
    parser.add_argument("--symbols", nargs="+", help="Symbols to populate (defaults to Tier 1)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/alternative"),
        help="Directory where parquet outputs are written",
    )
    parser.add_argument(
        "--tier1-progress",
        type=Path,
        default=None,
        help="Optional path to Tier 1 progress file",
    )
    return parser.parse_args(argv)


def _print_summary(result: AlternativeDataResult) -> None:
    if not result.frames:
        print("No data sources populated")
        return
    for name, frame in result.frames.items():
        rows = getattr(frame, "height", 0)
        print(f"{name}: {int(rows)} rows")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    bind_log_context(
        run_id=f"cli_populate_alternative_data_{_uuid.uuid4().hex[:8]}",
        component="ml.cli.populate_alternative_data",
    )

    config = PopulateAlternativeDataTaskConfig(
        output_dir=args.output_dir,
        symbols=tuple(args.symbols) if args.symbols else None,
        sources=tuple(args.source) if args.source else None,
        populate_all=args.all,
        tier1_progress_path=args.tier1_progress,
    )
    try:
        result = populate_alternative_data_task(config)
    except Exception as exc:  # pragma: no cover - surface CLI errors
        print(str(exc), file=sys.stderr)
        return 1

    _print_summary(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
