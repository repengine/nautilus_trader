#!/usr/bin/env python3
"""
Thin CLI wrapper for efficient L2 data population tasks.
"""

from __future__ import annotations

import argparse
import sys
import uuid as _uuid
from datetime import datetime
from pathlib import Path

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.data.ingest import PopulateL2TaskConfig
from ml.data.ingest import populate_l2_efficient
from ml.data.ingest.l2_efficient import L2PopulateResult


__all__ = ["main"]


def _parse_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argument guard
        raise SystemExit(f"Invalid date '{value}': {exc}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Efficient L2 data downloader")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to download")
    parser.add_argument("--tier", type=int, choices=[1], help="Use Tier 1 symbols")
    parser.add_argument("--days", type=int, default=30, help="Number of days to download")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tier1"),
        help="Data directory",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("data/tier1/.l2_progress.json"),
        help="Path to JSON progress file for resume",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing data",
    )
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--check-gaps",
        action="store_true",
        default=True,
        help="Check and fill gaps",
    )
    parser.add_argument("--force", action="store_true", help="Re-download all data")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Limit symbols processed in run",
    )
    parser.add_argument(
        "--symbol-offset",
        type=int,
        default=0,
        help="Start offset into symbol list",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle symbol order")
    parser.add_argument("--rate-limit", type=int, default=60, help="API calls per minute throttle")
    parser.add_argument("--dataset", type=str, default="DBEQ.BASIC", help="Databento dataset")
    parser.add_argument(
        "--schema",
        type=str,
        default="mbp-1",
        help="Depth schema: mbp-10|mbp-1|mbo (default: mbp-1)",
    )
    parser.add_argument(
        "--sleep-between-symbols",
        type=float,
        default=0.0,
        help="Seconds to sleep between symbols",
    )
    return parser.parse_args(argv)


def _print_summary(result: L2PopulateResult) -> None:
    print(
        "Total records:",
        f"{result.total_records:,}",
    )
    print(f"Total size (MB): {result.total_size_mb:.1f}")
    print(f"Symbols processed: {result.symbols_processed}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    bind_log_context(
        run_id=f"cli_populate_l2_efficient_{_uuid.uuid4().hex[:8]}",
        component="ml.cli.populate_l2_efficient",
    )

    config = PopulateL2TaskConfig(
        data_dir=args.data_dir,
        progress_file=args.progress_file,
        symbols=tuple(args.symbols) if args.symbols else None,
        tier=args.tier,
        days=args.days,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        resume=args.resume,
        check_gaps=args.check_gaps,
        force=args.force,
        max_symbols=args.max_symbols,
        symbol_offset=args.symbol_offset,
        shuffle=args.shuffle,
        rate_limit=args.rate_limit,
        dataset=args.dataset,
        schema=args.schema,
        sleep_between_symbols=args.sleep_between_symbols,
    )
    try:
        result = populate_l2_efficient(config)
    except Exception as exc:  # pragma: no cover - surface CLI errors
        print(str(exc), file=sys.stderr)
        return 1

    _print_summary(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
