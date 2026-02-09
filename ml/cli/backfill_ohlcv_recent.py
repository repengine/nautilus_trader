#!/usr/bin/env python3
"""
Thin CLI wrapper for backfilling recent OHLCV bars via tasks.
"""

from __future__ import annotations

import argparse
import sys
import uuid as _uuid
from datetime import datetime
from pathlib import Path

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.data.ingest import BackfillRecentOhlcvTaskConfig
from ml.data.ingest import OhlcvRecentBackfillResult
from ml.data.ingest import SymbolBackfillStatus
from ml.data.ingest import backfill_recent_ohlcv


__all__ = ["main"]


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argument guard
        raise SystemExit(f"Invalid datetime value '{value}': {exc}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill recent OHLCV-1m bars via Databento")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tier1"))
    parser.add_argument("--symbols", nargs="*", help="Symbols to backfill (default: directories)")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--days", type=int, default=14, help="Window when start/end not provided")
    return parser.parse_args(argv)


def _print_summary(result: OhlcvRecentBackfillResult) -> None:
    for summary in result.summaries:
        start = summary.requested_start.strftime("%Y-%m-%d") if summary.requested_start else "-"
        end = summary.requested_end.strftime("%Y-%m-%d") if summary.requested_end else "-"
        if summary.status is SymbolBackfillStatus.SUCCESS:
            rows = summary.rows_downloaded
            print(f"{summary.symbol}: downloaded {rows} rows from {start} to {end}")
        elif summary.status is SymbolBackfillStatus.EMPTY:
            print(f"{summary.symbol}: no rows in requested window")
        elif summary.status is SymbolBackfillStatus.SKIPPED:
            reason = summary.message or "skipped"
            print(f"{summary.symbol}: skipped ({reason})")
        else:
            print(f"{summary.symbol}: error {summary.message}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    bind_log_context(
        run_id=f"cli_backfill_ohlcv_recent_{_uuid.uuid4().hex[:8]}",
        component="ml.cli.backfill_ohlcv_recent",
    )

    config = BackfillRecentOhlcvTaskConfig(
        data_dir=args.data_dir,
        symbols=tuple(args.symbols) if args.symbols else None,
        tier=args.tier,
        start=_parse_datetime(args.start),
        end=_parse_datetime(args.end),
        lookback_days=args.days,
    )
    try:
        result = backfill_recent_ohlcv(config)
    except Exception as exc:  # pragma: no cover - surface CLI errors
        print(str(exc), file=sys.stderr)
        return 1

    _print_summary(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
