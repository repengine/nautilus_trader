#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ml.data.ingest.nautilus_adapters import to_df_bars
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.resume import DatabentoLikeClient
from ml.data.ingest.state import load_state
from ml.data.ingest.state import save_state
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.coverage_catalog import CatalogCoverageProvider
from ml.stores.coverage_sql import SqlCoverageProvider
from ml.stores.coverage_sql import SqlMarketDataWriter


if TYPE_CHECKING:  # pragma: no cover
    from nautilus_trader.model.data import Bar as NautilusBar
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
else:  # pragma: no cover - avoid hard dependency for tools
    NautilusBar = object  # type: ignore[assignment]
    ParquetDataCatalog = object  # type: ignore[assignment]


def _parse_instruments(arg: str | None) -> list[str]:
    if not arg:
        return []
    p = Path(arg)
    if p.exists() and p.is_file():
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return [s.strip() for s in arg.split(",") if s.strip()]


def _env_default(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


class _CatalogIngestClient:
    """
    Minimal DatabentoLikeClient backed by ParquetDataCatalog for bars.

    Useful for offline backfills when a Databento API client is not available.

    """

    def __init__(self, catalog_path: str) -> None:
        self._catalog = ParquetDataCatalog(catalog_path)

    def get_data(
        self,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
        **_: object,
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        instrument_id = symbols[0]
        # Only bars supported in this adapter; convert datetime to ns timestamp if needed
        s_val: int | str | float
        e_val: int | str | float
        if isinstance(start, datetime):
            s_val = int(start.timestamp() * 1e9)
        else:
            s_val = start
        if isinstance(end, datetime):
            e_val = int(end.timestamp() * 1e9)
        else:
            e_val = end
        data = self._catalog.query(
            data_cls=NautilusBar,
            identifiers=[instrument_id],
            start=s_val,
            end=e_val,
        )
        return to_df_bars(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Backfill gaps using orchestrator with pluggable coverage+writer.",
    )
    ap.add_argument(
        "--db",
        dest="db",
        default=_env_default("DB_CONNECTION"),
        help="PostgreSQL connection URL",
    )
    ap.add_argument("--dataset-id", required=True, help="Dataset identifier (e.g., EQUS.MINI)")
    ap.add_argument("--schema", required=True, help="Schema (e.g., bars, tbbo, trades)")
    ap.add_argument("--instruments", required=True, help="Comma list or file with instrument IDs")
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=int(_env_default("BACKFILL_LOOKBACK_DAYS", "7") or "7"),
    )
    ap.add_argument("--table-name", default=_env_default("TABLE_NAME", "market_data"))
    ap.add_argument("--catalog-path", default=_env_default("CATALOG_PATH"))
    ap.add_argument(
        "--state-path",
        default=_env_default("INGEST_STATE_PATH", "checkpoints/ingest_state.json"),
    )
    ap.add_argument(
        "--coverage-mode",
        choices=["sql", "catalog"],
        default=_env_default("COVERAGE_MODE", "sql"),
        help="Gap detection source (default sql)",
    )
    ap.add_argument(
        "--write-mode",
        choices=["sql", "parquet"],
        default=_env_default("WRITE_MODE", "sql"),
        help="Persistence target (default sql)",
    )
    ap.add_argument(
        "--client-mode",
        choices=["catalog", "databento", "noop"],
        default=_env_default("INGEST_CLIENT_MODE", "catalog"),
        help="Ingestion client (catalog|databento|noop)",
    )
    ap.add_argument(
        "--api-key",
        default=_env_default("DATABENTO_API_KEY"),
        help="Databento API key (for client-mode databento)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Plan gaps only (no ingestion/writes)")

    args = ap.parse_args(argv)

    instruments = _parse_instruments(args.instruments)
    if not instruments:
        raise SystemExit("No instruments provided")

    # Coverage
    from ml.stores.protocols import CoverageProviderProtocol  # local import to avoid cycles

    coverage: CoverageProviderProtocol
    if args.coverage_mode == "sql":
        if not args.db:
            raise SystemExit("--db is required for SQL coverage")
        coverage = SqlCoverageProvider(connection_string=args.db, table_name=args.table_name)
    else:
        if not args.catalog_path:
            raise SystemExit("--catalog-path is required for catalog coverage")
        coverage = CatalogCoverageProvider(catalog_path=args.catalog_path)

    # Writer
    if args.write_mode == "sql":
        if not args.db:
            raise SystemExit("--db is required for SQL write mode")
        writer = SqlMarketDataWriter(connection_string=args.db, table_name=args.table_name)
    else:
        raise SystemExit("parquet write-mode not implemented; use --write-mode sql")

    # Registry (DB authoritative)
    if not args.db:
        raise SystemExit("--db is required (registry + sql write mode)")
    registry = DataRegistry(
        registry_path=Path("ml_registry"),
        persistence_config=PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string=args.db,
        ),
    )

    # Ingestion client
    if args.client_mode == "catalog":
        if not args.catalog_path:
            raise SystemExit("--catalog-path is required for client-mode catalog")
        client: DatabentoLikeClient = _CatalogIngestClient(args.catalog_path)
    elif args.client_mode == "databento":
        if not args.api_key:
            raise SystemExit(
                "--api-key (or DATABENTO_API_KEY) is required for client-mode databento",
            )
        from ml.data.ingest.databento_adapter import DatabentoAPIClient

        client = DatabentoAPIClient(api_key=str(args.api_key))
    else:

        class _NoopClient:
            def get_data(
                self,
                dataset: str,
                symbols: list[str],
                schema: str,
                start: str | datetime,
                end: str | datetime,
                **kwargs: object,
            ) -> pd.DataFrame:
                return pd.DataFrame()

        client = _NoopClient()

    ingestor = DatabentoIngestor(client=client)
    orch = IngestionOrchestrator(
        coverage=coverage,
        writer=writer,
        registry=registry,
        ingestor=ingestor,
    )

    # State
    state = load_state(args.state_path)

    total_gaps: list[tuple[int, int]] = []
    for inst in instruments:
        gaps = orch.backfill_gaps(
            dataset_id=args.dataset_id,
            schema=args.schema,
            instrument_id=inst,
            lookback_days=int(args.lookback_days),
            state=None if args.dry_run else state,
        )
        total_gaps.extend(gaps)
        print(f"{inst}: planned {len(gaps)} day window(s)")

    if not args.dry_run:
        save_state(args.state_path, state)
        print(f"State saved to {args.state_path}")

    print(f"Total windows planned: {len(total_gaps)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
