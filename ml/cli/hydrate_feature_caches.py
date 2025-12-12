#!/usr/bin/env python3
"""
CLI wrapper for micro/L2 cache hydration tasks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import cast

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.config.universes import TIER1_SYMBOL_SETS
from ml.stores.data_store import DataStore
from ml.stores.feature_raw_writer import FeatureDatasetParquetRawWriter
from ml.stores.protocols import DataStoreFacadeProtocol
from ml.tasks.caches import CacheHydrationResult
from ml.tasks.caches import L2CacheHydrationConfig
from ml.tasks.caches import MicroCacheHydrationConfig
from ml.tasks.caches import hydrate_l2_caches
from ml.tasks.caches import hydrate_micro_caches
from ml.tasks.caches import ingest_l2_cache_partitions
from ml.tasks.caches import ingest_micro_cache_partitions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hydrate microstructure and L2 caches")
    parser.add_argument(
        "--symbols",
        default="@tier1_full",
        help="Comma-separated symbols or @alias (default: @tier1_full)",
    )
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--raw-dir",
        default="data/tier1",
        help="Raw tier1 data directory (default: data/tier1)",
    )
    parser.add_argument(
        "--micro-cache-dir",
        default="data/features/micro_minute",
        help="Micro cache directory (default: data/features/micro_minute)",
    )
    parser.add_argument(
        "--l2-cache-dir",
        default="data/features/l2_minute",
        help="L2 cache directory (default: data/features/l2_minute)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers (default: 4)",
    )
    parser.add_argument(
        "--micro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable micro cache hydration (default: enabled)",
    )
    parser.add_argument(
        "--l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable L2 cache hydration (default: enabled)",
    )
    parser.add_argument(
        "--force-micro",
        action="store_true",
        help="Force micro cache rebuild even if partitions exist",
    )
    parser.add_argument(
        "--force-l2",
        action="store_true",
        help="Force L2 cache rebuild even if partitions exist",
    )
    parser.add_argument(
        "--dsn",
        help="PostgreSQL DSN for SQL ingestion (default: DB_CONNECTION / NAUTILUS_DB)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    bind_log_context(component="ml.cli.hydrate_feature_caches")
    symbols = _parse_symbols(args.symbols)
    if not symbols:
        print("No symbols resolved from --symbols argument", file=sys.stderr)
        return 1
    start = _parse_date(args.start_date, flag="--start-date")
    end = _parse_date(args.end_date, flag="--end-date")
    if end < start:
        raise SystemExit("--end-date must be on or after --start-date")
    max_workers = max(1, int(args.max_workers))

    dsn = _resolve_dsn(args)
    results: list[CacheHydrationResult] = []
    feature_raw_writer = FeatureDatasetParquetRawWriter(
        micro_base_dir=Path(args.micro_cache_dir),
        l2_base_dir=Path(args.l2_cache_dir),
    )
    data_store = DataStore(connection_string=dsn, raw_writer=feature_raw_writer)
    if args.micro:
        micro_cfg = MicroCacheHydrationConfig(
            symbols=symbols,
            start_date=start,
            end_date=end,
            raw_base_dir=Path(args.raw_dir),
            cache_dir=Path(args.micro_cache_dir),
            max_workers=max_workers,
            force_rebuild=bool(args.force_micro),
        )
        results.append(hydrate_micro_caches(micro_cfg))
    if args.l2:
        l2_cfg = L2CacheHydrationConfig(
            symbols=symbols,
            start_date=start,
            end_date=end,
            raw_base_dir=Path(args.raw_dir),
            cache_dir=Path(args.l2_cache_dir),
            max_workers=max_workers,
            force_rebuild=bool(args.force_l2),
        )
        results.append(hydrate_l2_caches(l2_cfg))

    if not results:
        print("Nothing to do (enable --micro and/or --l2)", file=sys.stderr)
        return 1

    ingest_run_id = f"cache_cli_{int(time.time())}"
    if args.micro:
        ingest_micro_cache_partitions(
            data_store=cast(DataStoreFacadeProtocol, data_store),
            symbols=symbols,
            start_date=start,
            end_date=end,
            cache_dir=Path(args.micro_cache_dir),
            run_id=f"{ingest_run_id}_micro",
        )
    if args.l2:
        ingest_l2_cache_partitions(
            data_store=cast(DataStoreFacadeProtocol, data_store),
            symbols=symbols,
            start_date=start,
            end_date=end,
            cache_dir=Path(args.l2_cache_dir),
            run_id=f"{ingest_run_id}_l2",
        )

    for result in results:
        _print_summary(result)
    return 0


def _parse_symbols(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for entry in value.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        if trimmed.startswith("@"):
            alias = trimmed[1:].lower()
            resolved = _resolve_alias(alias)
            tokens.extend(resolved)
        else:
            tokens.append(trimmed)
    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = token.strip().upper()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _resolve_alias(alias: str) -> Iterable[str]:
    resolved = TIER1_SYMBOL_SETS.get(alias)
    if resolved is not None:
        return resolved
    if alias.startswith("tier1_"):
        fallback = alias.replace("tier1_", "", 1)
        resolved = TIER1_SYMBOL_SETS.get(fallback)
        if resolved is not None:
            return resolved
    raise SystemExit(f"Unknown universe alias '{alias}'")


def _parse_date(payload: str, *, flag: str) -> date:
    try:
        return date.fromisoformat(payload)
    except ValueError as exc:  # pragma: no cover - argparse guard
        raise SystemExit(f"{flag} must be YYYY-MM-DD ({exc})") from exc


def _print_summary(result: CacheHydrationResult) -> None:
    print(f"{result.label.upper()} hydration summary")
    print(f"  symbols processed: {len(result.results)}")
    print(f"  partitions requested: {result.total_requested}")
    print(f"  partitions written:   {result.total_written}")
    print(f"  partitions skipped:   {result.total_skipped}")
    print(f"  empty partitions:     {result.total_empty}")
    if result.failed:
        print("  failures:")
        for failure in result.failed:
            print(f"    - {failure.symbol}: {'; '.join(failure.errors)}")
    else:
        print("  failures: none")


def _resolve_dsn(args: argparse.Namespace) -> str:
    candidates = (
        getattr(args, "dsn", None),
        os.getenv("DB_CONNECTION"),
        os.getenv("NAUTILUS_DB"),
        os.getenv("DATABASE_URL"),
    )
    for candidate in candidates:
        if candidate:
            return candidate
    raise SystemExit("Database connection string required (use --dsn or set DB_CONNECTION/NAUTILUS_DB).")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
