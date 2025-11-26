#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ml.features import FeatureConfig
from ml.stores.feature_store import FeatureStore


def _parse_instruments(arg: str | None) -> list[str]:
    if not arg:
        return []
    # Support comma-separated or file paths
    path = Path(arg)
    if path.exists() and path.is_file():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [s.strip() for s in arg.split(",") if s.strip()]


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    # Accept YYYY-MM-DD or full ISO 8601
    try:
        if len(value) == 10:
            return datetime.fromisoformat(value)
        return datetime.fromisoformat(value)
    except Exception as exc:  # pragma: no cover - argparse error path
        raise argparse.ArgumentTypeError(f"Invalid datetime format: {value} ({exc})")


def run(
    connection_string: str,
    instruments: list[str],
    start: datetime | None,
    end: datetime | None,
    *,
    force_recompute: bool,
    max_workers: int,
    feature_config: FeatureConfig | None = None,
) -> dict[str, int]:
    store = FeatureStore(connection_string, feature_config=(feature_config or FeatureConfig()))
    return store.compute_historical_parallel(
        instrument_ids=instruments,
        start=start,
        end=end,
        force_recompute=force_recompute,
        max_workers=max_workers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parallel backfill of features via FeatureStore")
    parser.add_argument("--db", dest="db", required=True, help="PostgreSQL connection string")
    parser.add_argument(
        "--instruments",
        dest="instruments",
        required=True,
        help="Comma-separated instruments or a path to a file (one per line)",
    )
    parser.add_argument(
        "--start",
        dest="start",
        default=None,
        help="Start ISO date/time (optional)",
    )
    parser.add_argument("--end", dest="end", default=None, help="End ISO date/time (optional)")
    parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="Force recompute even if data present",
    )
    parser.add_argument(
        "--max-workers",
        dest="max_workers",
        type=int,
        default=4,
        help="Max parallel workers (default 4)",
    )

    args = parser.parse_args(argv)

    instruments = _parse_instruments(args.instruments)
    if not instruments:
        raise SystemExit("No instruments specified")

    start_dt = _parse_dt(args.start)
    end_dt = _parse_dt(args.end)

    results = run(
        connection_string=args.db,
        instruments=instruments,
        start=start_dt,
        end=end_dt,
        force_recompute=args.force,
        max_workers=int(args.max_workers),
    )

    # Simple textual report
    total_rows = sum(results.values())
    completed = sum(1 for v in results.values() if v > 0)
    failed = sum(1 for v in results.values() if v <= 0)
    print(f"Completed: {completed}, Failed: {failed}, Total rows: {total_rows}")
    for inst, rows in sorted(results.items()):
        print(f"  {inst}: {rows}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
