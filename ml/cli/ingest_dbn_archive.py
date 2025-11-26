#!/usr/bin/env python3
"""
CLI entrypoint for ingesting Databento DBN archives from disk.

The command processes one or more ``.zip`` bundles under ``data/batch/`` (or a
user-specified path), decodes the contained ``.dbn.zst`` members, and writes the
resulting frames to the canonical SQL market-data store with an optional
DataStore mirror.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import structlog

from ml.data.ingest.dbn_archive import DBNArchiveIngestionConfig
from ml.data.ingest.dbn_archive import DBNArchiveIngestor
from ml.stores import DataStore
from ml.stores.providers import SqlMarketDataWriter
from ml.stores.writers import DataStoreMarketDataWriter


__all__ = ["main"]


logger = structlog.get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest Databento DBN archives into the market data store.",
    )
    parser.add_argument(
        "input",
        metavar="PATH",
        help="Path to a .zip archive or a directory containing archives.",
    )
    parser.add_argument(
        "--db-url",
        dest="db_url",
        required=True,
        help="PostgreSQL connection URL for the canonical market_data table.",
    )
    parser.add_argument(
        "--dataset",
        help="Override dataset identifier (defaults to metadata.json value).",
    )
    parser.add_argument(
        "--schema",
        help="Override schema identifier (defaults to metadata.json value).",
    )
    parser.add_argument(
        "--source-dataset",
        dest="source_dataset",
        help="Source dataset tag used for provenance (defaults to dataset).",
    )
    parser.add_argument(
        "--instrument-suffix",
        dest="instrument_suffix",
        help="Optional suffix appended to symbols when deriving instrument IDs",
    )
    parser.add_argument(
        "--mirror-data-store",
        dest="mirror",
        action="store_true",
        help="Also write records to the ML DataStore (requires DB connection).",
    )
    parser.add_argument(
        "--table-name",
        dest="table_name",
        default="market_data",
        help="Market data table name (default: market_data).",
    )
    return parser


def _resolve_archives(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix == ".zip")
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path does not exist: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    archives = _resolve_archives(Path(args.input).expanduser())
    if not archives:
        logger.warning("offline.ingest.no_archives", input=args.input)
        return 0

    writer = SqlMarketDataWriter(
        connection_string=args.db_url,
        table_name=args.table_name,
    )

    mirror_writer = None
    if args.mirror:
        mirror_writer = DataStoreMarketDataWriter(
            store=DataStore(connection_string=args.db_url),
        )

    ingestor = DBNArchiveIngestor(writer=writer, mirror_writer=mirror_writer)

    for archive_path in archives:
        config = DBNArchiveIngestionConfig(
            archive_path=archive_path,
            dataset=args.dataset,
            schema=args.schema,
            source_dataset=args.source_dataset,
            instrument_suffix=args.instrument_suffix,
        )
        result = ingestor.ingest_archive(config)
        logger.info(
            "offline.ingest.archive.completed",
            archive=str(archive_path),
            dataset=result.dataset,
            schema=result.schema,
            source_dataset=result.source_dataset,
            instruments=len(result.instruments),
            frames=result.total_frames,
            rows=result.total_rows,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
