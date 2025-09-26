from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import coerce_storage_kind
from ml.data.ingest.api import ensure_service
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.nautilus_adapters import to_df_bars
from ml.data.ingest.orchestrator import DomainWindowLoaderProtocol
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.resume import DatabentoLikeClient
from ml.data.ingest.service import build_like_client
from ml.data.ingest.state import load_state
from ml.data.ingest.state import save_state
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.providers import CatalogCoverageProvider
from ml.stores.providers import SqlCoverageProvider
from ml.stores.providers import SqlMarketDataWriter


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


def _parse_market_inputs(value: str | None) -> tuple[MarketDatasetInput, ...] | None:
    if value is None:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise argparse.ArgumentTypeError("market_inputs_json must be valid JSON") from exc

    if isinstance(payload, (str, dict)):
        items: list[object] = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise argparse.ArgumentTypeError(
            "market_inputs_json must encode a descriptor string or list of descriptor objects",
        )

    parsed: list[MarketDatasetInput] = []
    for entry in items:
        if isinstance(entry, str):
            parsed.append(MarketDatasetInput(descriptor_id=entry))
            continue
        if isinstance(entry, dict):
            descriptor_id = entry.get("descriptor_id")
            dataset_id = entry.get("dataset_id")
            symbols_field = entry.get("symbols")
            if symbols_field is None:
                symbols_tuple = None
            elif isinstance(symbols_field, str):
                symbols_tuple = tuple(
                    token.strip().upper()
                    for token in symbols_field.split(",")
                    if token.strip()
                )
            elif isinstance(symbols_field, (list, tuple)):
                symbols_tuple = tuple(
                    str(token).strip().upper()
                    for token in symbols_field
                    if str(token).strip()
                )
            else:
                raise argparse.ArgumentTypeError(
                    "symbols in market_inputs_json must be a list or comma-separated string",
                )

            schema_override = entry.get("schema") or entry.get("schema_override")
            storage_raw = entry.get("storage_kind") or entry.get("storage_kind_override")
            storage_kind = None
            if storage_raw is not None:
                try:
                    storage_kind = coerce_storage_kind(storage_raw)
                except ValueError as exc:
                    raise argparse.ArgumentTypeError(
                        f"Invalid storage_kind '{storage_raw}' in market_inputs_json",
                    ) from exc

            parsed.append(
                MarketDatasetInput(
                    descriptor_id=str(descriptor_id) if descriptor_id is not None else None,
                    dataset_id=str(dataset_id) if dataset_id is not None else None,
                    symbols=symbols_tuple,
                    schema_override=str(schema_override) if schema_override is not None else None,
                    storage_kind_override=storage_kind,
                    start=str(entry.get("start")) if entry.get("start") is not None else None,
                    end=str(entry.get("end")) if entry.get("end") is not None else None,
                ),
            )
            continue
        raise argparse.ArgumentTypeError(
            "market_inputs_json entries must be descriptor strings or mapping objects",
        )

    return tuple(parsed) if parsed else None


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
            from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

            s_val = _sanitize(int(start.timestamp() * 1e9), context="cli.ingest_backfill:start")
        else:
            s_val = start
        if isinstance(end, datetime):
            e_val = _sanitize(int(end.timestamp() * 1e9), context="cli.ingest_backfill:end")
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
        "--also-write-catalog",
        action="store_true",
        help="When set, write domain objects into ParquetDataCatalog (requires --catalog-path)",
    )
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
    ap.add_argument(
        "--market-dataset-id",
        help="Optional dataset identifier used when resolving market feed bindings",
    )
    ap.add_argument(
        "--market-inputs-json",
        help="JSON payload describing MarketDatasetInput descriptors",
    )
    ap.add_argument("--dry-run", action="store_true", help="Plan gaps only (no ingestion/writes)")

    args = ap.parse_args(argv)

    instruments = _parse_instruments(args.instruments)
    if not instruments:
        raise SystemExit("No instruments provided")

    market_inputs = _parse_market_inputs(getattr(args, "market_inputs_json", None))
    binding_dataset_id = getattr(args, "market_dataset_id", None) or args.dataset_id
    use_bindings = bool(market_inputs or getattr(args, "market_dataset_id", None))

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

    # Writer(s)
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
        api_key = args.api_key or os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise SystemExit(
                "--api-key (or DATABENTO_API_KEY) is required for client-mode databento",
            )
        os.environ.setdefault("DATABENTO_API_KEY", str(api_key))
        try:
            service = ensure_service()
        except Exception as exc:
            raise SystemExit(f"Failed to initialise Databento ingestion service: {exc}")
        client = build_like_client(service)
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

    # Optional dual-write to catalog using domain objects
    raw_writer = None
    domain_loader: DomainWindowLoaderProtocol | None = None
    if args.also_write_catalog:
        if not args.catalog_path:
            raise SystemExit("--also-write-catalog requires --catalog-path")
        try:
            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog as _PDC
        except Exception as exc:  # pragma: no cover - env dependent
            raise SystemExit(f"ParquetDataCatalog unavailable: {exc}")

        from ml.stores.io_raw import ParquetCatalogRawWriter

        catalog_obj = _PDC(args.catalog_path)
        raw_writer = ParquetCatalogRawWriter(catalog_obj)
        # Service-backed ingestion writes Pandas frames directly; domain_loader remains None.

    orch = IngestionOrchestrator(
        coverage=coverage,
        writer=writer,
        registry=registry,
        ingestor=ingestor,
        raw_writer=raw_writer,
        domain_loader=domain_loader,
    )

    resolved_bindings: tuple[ResolvedMarketBinding, ...] = ()
    binding_lookup: dict[str, ResolvedMarketBinding] = {}
    binding_ids_processed: set[str] = set()
    if use_bindings:
        base_symbols = sorted({inst.split(".")[0].upper() for inst in instruments})
        resolved_bindings = IngestionOrchestrator.resolve_market_bindings(
            symbols=base_symbols,
            instrument_ids=tuple(instruments),
            market_dataset_id=binding_dataset_id,
            market_inputs=market_inputs,
        )
        binding_lookup = {
            binding_item.symbol.upper(): binding_item
            for binding_item in resolved_bindings
        }
        for binding_item in resolved_bindings:
            for inst_id in binding_item.instrument_ids:
                binding_lookup.setdefault(inst_id.upper(), binding_item)

    # State
    state = load_state(args.state_path)

    total_gaps: list[tuple[int, int]] = []
    for inst in instruments:
        binding: ResolvedMarketBinding | None = None
        if use_bindings:
            binding = binding_lookup.get(inst.upper()) or binding_lookup.get(inst.split(".")[0].upper())
        if binding is not None:
            if binding.binding_id in binding_ids_processed:
                continue
            binding_results = orch.backfill_binding(
                binding=binding,
                lookback_days=int(args.lookback_days),
                state=None if args.dry_run else state,
            )
            binding_ids_processed.add(binding.binding_id)
            for instrument_id, windows in binding_results.items():
                total_gaps.extend(windows)
                print(
                    f"{instrument_id}: planned {len(windows)} day window(s) via binding"
                    f" {binding.binding_id}",
                )
            continue

        gaps = orch.backfill_gaps(
            dataset_id=args.dataset_id,
            schema=args.schema,
            instrument_id=inst,
            lookback_days=int(args.lookback_days),
            state=None if args.dry_run else state,
        )
        if use_bindings and binding is None:
            print(
                "Warning: no binding resolved for"
                f" {inst}; using legacy dataset {args.dataset_id}/{args.schema}",
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
