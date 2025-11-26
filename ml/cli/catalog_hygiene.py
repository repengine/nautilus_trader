#!/usr/bin/env python3
"""
CLI utilities for catalog hygiene.

Operators can invoke this tool before Tier-1 orchestrator runs to archive
stale catalog contents (which contain overlapping intervals) and recreate a
clean directory for future parquet fan-out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from ml.common.logging_config import configure_logging
from ml.data.catalog_hygiene import archive_catalog
from ml.data.catalog_hygiene import prepare_clean_catalog_path


logger = structlog.get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Catalog hygiene helper")
    parser.add_argument("--catalog-path", required=True, help="Target Parquet catalog directory")
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="Optional directory for catalog archives (defaults to parent directory)",
    )
    parser.add_argument(
        "--mode",
        default="clean",
        choices=("clean", "archive"),
        help="Operation to perform (default: clean)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    catalog_path = Path(args.catalog_path)
    backup_dir = Path(args.backup_dir).expanduser() if args.backup_dir else None
    if args.mode == "archive":
        try:
            destination = archive_catalog(catalog_path=catalog_path, backup_root=backup_dir)
        except FileNotFoundError:
            logger.info(
                "catalog_hygiene.archive_skipped",
                catalog=str(catalog_path),
                reason="catalog_missing",
            )
            return 0
        logger.info(
            "catalog_hygiene.archived",
            catalog=str(catalog_path),
            archive=str(destination),
        )
        return 0
    prepare_clean_catalog_path(catalog_path=catalog_path, backup_root=backup_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
