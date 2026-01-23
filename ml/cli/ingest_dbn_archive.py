#!/usr/bin/env python3
"""
CLI entrypoint for ingesting Databento DBN archives from disk.

The command processes one or more ``.zip`` bundles under ``data/batch/`` (or a
user-specified path), decodes the contained ``.dbn.zst`` members, and writes the
resulting frames to the canonical SQL market-data store with optional mirrors
(DataStore and/or Parquet catalog). A catalog-only mode is available for offline
rehydration workflows where SQL will be restored later from Parquet.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import structlog

from ml.data.ingest.dbn_archive import DBNArchiveIngestionConfig
from ml.data.ingest.dbn_archive import DBNArchiveIngestor
from ml.deployment.scheduling_utils import parse_dataset_template_map_env
from ml.schema import validate_dataset_type_templates
from ml.stores.data_store import DataStore
from ml.stores.protocols import MarketDataWriterProtocol
from ml.stores.providers import SqlMarketDataWriter
from ml.stores.writers import DataStoreMarketDataWriter
from ml.stores.writers import FanoutMarketDataWriter
from ml.stores.writers import ParquetCatalogRawMarketDataWriter


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
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL for the canonical market_data table (required unless --catalog-only).",
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
        "--catalog-path",
        dest="catalog_path",
        default=os.environ.get("CATALOG_PATH"),
        help="Optional Parquet catalog root for mirroring writes.",
    )
    parser.add_argument(
        "--catalog-overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prune overlapping Parquet catalog files before writes "
            "(use --no-catalog-overwrite to preserve existing files and skip overlaps)."
        ),
    )
    parser.add_argument(
        "--catalog-only",
        dest="catalog_only",
        action="store_true",
        help="Write only to the Parquet catalog (skip SQL/DataStore). Requires --catalog-path.",
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

    if args.catalog_only and not args.catalog_path:
        parser.error("--catalog-only requires --catalog-path")
    if not args.catalog_only and not args.db_url:
        parser.error("--db-url is required unless --catalog-only is set")
    if args.catalog_only and args.mirror:
        parser.error("--mirror-data-store cannot be used with --catalog-only")

    archives = _resolve_archives(Path(args.input).expanduser())
    if not archives:
        logger.warning("offline.ingest.no_archives", input=args.input)
        return 0

    catalog_writer: ParquetCatalogRawMarketDataWriter | None = None
    if args.catalog_path:
        catalog_path = Path(args.catalog_path).expanduser()
        catalog_path.mkdir(parents=True, exist_ok=True)
        try:
            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
        except Exception as exc:  # pragma: no cover - defensive import
            logger.error("Unable to import ParquetDataCatalog", exc_info=True)
            raise RuntimeError("ParquetDataCatalog import failed") from exc
        catalog = ParquetDataCatalog(str(catalog_path))
        dataset_templates = validate_dataset_type_templates(
            parse_dataset_template_map_env(
                os.environ.get("CATALOG_REHYDRATE_DATASET_TYPE_TEMPLATES"),
            ),
        )
        catalog_writer = ParquetCatalogRawMarketDataWriter(
            catalog=catalog,
            replace_on_overlap=bool(args.catalog_overwrite),
            dataset_type_identifier_templates=dataset_templates or None,
        )

    writer: MarketDataWriterProtocol
    if args.catalog_only:
        if catalog_writer is None:  # pragma: no cover - parser guards but keep defensive
            parser.error("--catalog-only requires --catalog-path")
        assert catalog_writer is not None
        writer = catalog_writer
    else:
        primary_writer = SqlMarketDataWriter(
            connection_string=args.db_url,
            table_name=args.table_name,
        )

        mirrors: list[MarketDataWriterProtocol] = []
        if args.mirror:
            mirrors.append(
                DataStoreMarketDataWriter(
                    store=DataStore(connection_string=args.db_url),
                ),
            )
        if catalog_writer is not None:
            mirrors.append(catalog_writer)

        if mirrors:
            writer = FanoutMarketDataWriter(
                primary=primary_writer,
                mirrors=tuple(mirrors),
            )
        else:
            writer = primary_writer

    ingestor = DBNArchiveIngestor(writer=writer, mirror_writer=None)

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
