#!/usr/bin/env python
"""
CLI to migrate tier1 parquet shards into a ParquetDataCatalog.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from ml.common.logging_config import configure_logging
from ml.data.migration.tier1_to_catalog import MigrationStats
from ml.data.migration.tier1_to_catalog import migrate_tier1_to_catalog


logger = logging.getLogger(__name__)
configure_logging(level="INFO")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier1-root",
        type=Path,
        default=Path("/home/nate/nautilus_data/ml_data/tier1"),
        help="Root directory containing per-symbol tier1 shards.",
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=Path("/home/nate/nautilus_data/ml_data/catalog"),
        help="Target ParquetDataCatalog path.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="bars,quotes,trades",
        help="Comma-separated datasets to migrate (bars,quotes,trades).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Optional comma-separated symbols allowlist (base symbols). Empty means all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20000,
        help="Parquet batch size when streaming.",
    )
    parser.add_argument(
        "--default-venue",
        type=str,
        default="XNAS",
        help="Venue suffix used when tier1 symbol lacks a venue token.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Do not write; just plan and log file dispositions using parquet footers.",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=None,
        help="Optional path to write the analyzed plan as NDJSON (analyze-only).",
    )
    parser.add_argument(
        "--plan-input",
        type=Path,
        default=None,
        help="Optional plan file to drive a targeted migration (processes non-fully-overlapped entries).",
    )
    parser.add_argument(
        "--process-statuses",
        type=str,
        default="",
        help="Comma-separated statuses to process when using a plan (default: partial_overlap,uncovered,no_bounds).",
    )
    return parser.parse_args(argv)


def _format_stats(stats: MigrationStats) -> str:
    return (
        f"bars: files={stats.bars_files}, rows={stats.bars_rows}; "
        f"quotes: files={stats.quotes_files}, rows={stats.quotes_rows}; "
        f"trades: files={stats.trades_files}, rows={stats.trades_rows}; "
        f"skipped={len(stats.skipped)} analyzed={len(stats.analyzed)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_tokens = {
        token.strip().lower() for token in args.datasets.split(",") if token.strip()
    }
    invalid = dataset_tokens - {"bars", "quotes", "trades"}
    if invalid:
        raise SystemExit(f"Unsupported datasets requested: {sorted(invalid)}")
    symbols = {s.strip().upper() for s in args.symbols.split(",") if s.strip()} or None
    statuses = {s.strip() for s in args.process_statuses.split(",") if s.strip()} or None

    logger.info("Starting tier1 -> catalog migration")
    stats = migrate_tier1_to_catalog(
        tier1_root=args.tier1_root,
        catalog_path=args.catalog_path,
        datasets=set(dataset_tokens),
        symbols=symbols,
        batch_size=args.batch_size,
        default_venue=args.default_venue,
        analyze_only=args.analyze_only,
        plan_output=args.plan_output,
        plan_input=args.plan_input,
        process_statuses=statuses,
    )
    logger.info("Migration complete: %s", _format_stats(stats))
    if stats.skipped:
        logger.info("Skipped entries (first 10): %s", stats.skipped[:10])
    if stats.analyzed:
        logger.info("Analyzed entries (first 10): %s", stats.analyzed[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
