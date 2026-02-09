#!/usr/bin/env python3
"""
CLI entrypoint for market data backfill ingestion.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from ml.common.cli_parsers import parse_market_inputs_json
from ml.orchestration.ingestion_coordinator import IngestBackfillRuntimeConfig
from ml.orchestration.ingestion_coordinator import run_ingest_backfill


__all__ = ["main"]


def _env_default(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _parse_instruments(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    path = Path(value)
    if path.exists() and path.is_file():
        return tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return tuple(token.strip() for token in value.split(",") if token.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill gaps using orchestrator with pluggable coverage+writer.",
    )
    parser.add_argument(
        "--db",
        dest="db",
        default=_env_default("DB_CONNECTION"),
        help="PostgreSQL connection URL",
    )
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier (e.g., EQUS.MINI)")
    parser.add_argument("--schema", required=True, help="Schema (e.g., bars, tbbo, trades)")
    parser.add_argument("--instruments", required=True, help="Comma list or file with instrument IDs")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=int(_env_default("BACKFILL_LOOKBACK_DAYS", "7") or "7"),
    )
    parser.add_argument("--table-name", default=_env_default("TABLE_NAME", "market_data"))
    parser.add_argument("--catalog-path", default=_env_default("CATALOG_PATH"))
    parser.add_argument(
        "--also-write-catalog",
        action="store_true",
        help="When set, write domain objects into ParquetDataCatalog (requires --catalog-path)",
    )
    parser.add_argument(
        "--state-path",
        default=_env_default("INGEST_STATE_PATH", "checkpoints/ingest_state.json"),
    )
    parser.add_argument(
        "--coverage-mode",
        choices=["sql", "catalog"],
        default=_env_default("COVERAGE_MODE", "sql"),
        help="Gap detection source (default sql)",
    )
    parser.add_argument(
        "--write-mode",
        choices=["sql", "parquet"],
        default=_env_default("WRITE_MODE", "sql"),
        help="Persistence target (default sql)",
    )
    parser.add_argument(
        "--client-mode",
        choices=["catalog", "databento", "noop"],
        default=_env_default("INGEST_CLIENT_MODE", "catalog"),
        help="Ingestion client (catalog|databento|noop)",
    )
    parser.add_argument(
        "--api-key",
        default=_env_default("DATABENTO_API_KEY"),
        help="Databento API key (for client-mode databento)",
    )
    parser.add_argument(
        "--market-dataset-id",
        help="Optional dataset identifier used when resolving market feed bindings",
    )
    parser.add_argument(
        "--market-inputs-json",
        help="JSON payload describing MarketDatasetInput descriptors",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan gaps only (no ingestion/writes)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse CLI arguments and run the canonical ingest backfill flow.
    """
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        market_inputs = parse_market_inputs_json(getattr(args, "market_inputs_json", None))
        config = IngestBackfillRuntimeConfig(
            db=args.db,
            dataset_id=args.dataset_id,
            schema=args.schema,
            instruments=_parse_instruments(args.instruments),
            lookback_days=int(args.lookback_days),
            table_name=args.table_name,
            catalog_path=args.catalog_path,
            also_write_catalog=bool(args.also_write_catalog),
            state_path=args.state_path,
            coverage_mode=args.coverage_mode,
            write_mode=args.write_mode,
            client_mode=args.client_mode,
            api_key=args.api_key,
            market_dataset_id=args.market_dataset_id,
            market_inputs=market_inputs,
            dry_run=bool(args.dry_run),
        )
        run_ingest_backfill(config, emit=print)
    except ValueError as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
