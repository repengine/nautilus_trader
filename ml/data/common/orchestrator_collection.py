"""
Orchestrator-based collection component extracted from DataScheduler.

This component handles collection logic using the IngestionOrchestrator:
- API key validation
- Database connection validation
- SQL coverage/writer initialization
- Optional dual-write to ParquetDataCatalog
- Market binding resolution and backfill

Extracted from legacy DataScheduler (lines 1042-1172):
- _collect_via_orchestrator()

"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:

    from ml.config.scheduler_config import SchedulerConfig
    from ml.data.ingest.market_bindings import ResolvedMarketBinding
    from ml.data.ingest.orchestrator import DomainWindowLoaderProtocol
    from ml.data.ingest.orchestrator import IngestionOrchestrator
    from ml.registry.dataclasses import DatasetType
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.io_raw import RawIngestionWriterProtocol
    from ml.stores.protocols import CoverageProviderProtocol
    from ml.stores.protocols import MarketDataWriterProtocol
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


logger = logging.getLogger(__name__)


class OrchestratorCollectionProtocol(Protocol):
    """
    Protocol for orchestrator-based collection operations.

    This protocol defines the contract for orchestrator collection components,
    enabling duck typing for testing and alternative implementations.

    Methods
    -------
    collect_via_orchestrator
        Collect data using IngestionOrchestrator with optional dual-write.
    create_sql_coverage_provider
        Create a SqlCoverageProvider instance.
    create_sql_writer
        Create a SqlMarketDataWriter instance.
    create_raw_writer
        Create a ParquetCatalogRawWriter for dual-write scenarios.

    """

    def collect_via_orchestrator(
        self,
        config: SchedulerConfig,
        connection: str | None,
        registry: RegistryProtocol,
        catalog: ParquetDataCatalog,
        dual_write: bool,
        dual_write_dataset_types: Mapping[DatasetType, bool] | None = None,
        dataset_type_identifier_templates: Mapping[DatasetType, str] | None = None,
    ) -> None:
        """
        Collect data via IngestionOrchestrator with optional dual-write.

        Args:
            config: Scheduler configuration with Databento and symbol settings.
            connection: Database connection string for SQL coverage/writer.
            registry: DataRegistry for event tracking and watermarks.
            catalog: ParquetDataCatalog for dual-write scenarios.
            dual_write: Whether to mirror domain objects to catalog.
            dual_write_dataset_types: Optional dataset-type toggles for mirroring.
            dataset_type_identifier_templates: Optional identifier templates keyed by DatasetType.
            dual_write_dataset_types: Optional dataset-type toggles for mirroring.
            dataset_type_identifier_templates: Optional identifier templates keyed by DatasetType.

        Raises:
            ValueError: If API key or DB connection is missing.
            RuntimeError: If registry is not initialized.

        """
        ...

    def create_sql_coverage_provider(
        self,
        connection: str,
        table_name: str,
    ) -> CoverageProviderProtocol:
        """
        Create a SqlCoverageProvider instance.

        Args:
            connection: Database connection string.
            table_name: Name of the market data table.

        Returns:
            Configured SqlCoverageProvider.

        """
        ...

    def create_sql_writer(
        self,
        connection: str,
        table_name: str,
    ) -> MarketDataWriterProtocol:
        """
        Create a SqlMarketDataWriter instance.

        Args:
            connection: Database connection string.
            table_name: Name of the market data table.

        Returns:
            Configured SqlMarketDataWriter.

        """
        ...

    def create_raw_writer(
        self,
        catalog: ParquetDataCatalog,
        dataset_type_toggles: Mapping[DatasetType, bool] | None = None,
        dataset_type_identifier_templates: Mapping[DatasetType, str] | None = None,
    ) -> RawIngestionWriterProtocol:
        """
        Create a ParquetCatalogRawWriter for dual-write scenarios.

        Args:
            catalog: ParquetDataCatalog instance.
            dataset_type_toggles: Optional dataset-type toggles for mirroring.
            dataset_type_identifier_templates: Optional identifier templates keyed by DatasetType.

        Returns:
            Configured ParquetCatalogRawWriter.

        """
        ...


class OrchestratorCollectionComponent:
    """
    Component for orchestrator-based data collection logic.

    This component extracts collection responsibilities from DataScheduler,
    providing focused methods for:
    - API key and connection validation
    - SQL coverage provider and writer creation
    - Optional dual-write with domain loader
    - Market binding resolution and backfill execution

    All methods are designed to handle errors appropriately with clear
    error messages for missing dependencies.

    Example:
        >>> from ml.config.scheduler_config import SchedulerConfig
        >>> component = OrchestratorCollectionComponent()
        >>> config = SchedulerConfig()
        >>> # Requires valid connection, registry, and catalog
        >>> component.collect_via_orchestrator(config, conn, registry, catalog, False)

    """

    def collect_via_orchestrator(
        self,
        config: SchedulerConfig,
        connection: str | None,
        registry: RegistryProtocol | None,
        catalog: ParquetDataCatalog,
        dual_write: bool,
        dual_write_dataset_types: Mapping[DatasetType, bool] | None = None,
        dataset_type_identifier_templates: Mapping[DatasetType, str] | None = None,
    ) -> None:
        """
        Collect data via IngestionOrchestrator with optional dual-write.

        Uses SQL coverage and SQL writer for canonical market data storage.
        When dual_write=True, also mirrors domain objects into the
        ParquetDataCatalog using a lightweight domain loader.

        Args:
            config: Scheduler configuration with Databento and symbol settings.
            connection: Database connection string for SQL coverage/writer.
            registry: DataRegistry for event tracking and watermarks.
            catalog: ParquetDataCatalog for dual-write scenarios.
            dual_write: Whether to mirror domain objects to catalog.

        Raises:
            ValueError: If API key or DB connection is missing.
            RuntimeError: If registry is not initialized.

        Example:
            >>> component = OrchestratorCollectionComponent()
            >>> component.collect_via_orchestrator(
            ...     config=scheduler_config,
            ...     connection="postgresql://user:pass@host:5432/db",
            ...     registry=data_registry,
            ...     catalog=parquet_catalog,
            ...     dual_write=True,
            ... )

        """
        # Validate API key
        api_key = config.databento.api_key or os.getenv("DATABENTO_API_KEY")
        if not api_key:
            logger.error("DATABENTO_API_KEY environment variable not set")
            raise ValueError("DATABENTO_API_KEY required for orchestrator ingestion")

        # Validate DB connection
        db_conn = self._resolve_db_connection(connection)
        if not db_conn:
            raise ValueError("DB connection required for orchestrator coverage/writer")

        # Validate registry
        if registry is None:
            raise RuntimeError("DataRegistry not initialized")

        # Create SQL providers
        coverage = self.create_sql_coverage_provider(db_conn, "market_data")
        writer = self.create_sql_writer(db_conn, "market_data")

        # Create ingestor
        ingestor = self._create_ingestor(api_key)

        # Setup optional dual-write
        raw_writer: RawIngestionWriterProtocol | None = None
        domain_loader: DomainWindowLoaderProtocol | None = None
        if dual_write:
            raw_writer = self.create_raw_writer(
                catalog,
                dataset_type_toggles=dual_write_dataset_types,
                dataset_type_identifier_templates=dataset_type_identifier_templates,
            )
            domain_loader = self._create_domain_loader(api_key, config)

        # Create orchestrator
        orchestrator = self._create_orchestrator(
            coverage=coverage,
            writer=writer,
            registry=registry,
            ingestor=ingestor,
            raw_writer=raw_writer,
            domain_loader=domain_loader,
        )

        # Resolve market bindings if configured
        bindings = self._resolve_bindings(config, orchestrator)

        # Execute backfill
        self._execute_backfill(config, orchestrator, bindings)

    def create_sql_coverage_provider(
        self,
        connection: str,
        table_name: str,
    ) -> CoverageProviderProtocol:
        """
        Create a SqlCoverageProvider instance.

        Args:
            connection: Database connection string.
            table_name: Name of the market data table.

        Returns:
            Configured SqlCoverageProvider.

        Example:
            >>> component = OrchestratorCollectionComponent()
            >>> provider = component.create_sql_coverage_provider(
            ...     "postgresql://user:pass@host:5432/db",
            ...     "market_data",
            ... )

        """
        from ml.stores.providers import SqlCoverageProvider

        return SqlCoverageProvider(
            connection_string=connection,
            table_name=table_name,
        )

    def create_sql_writer(
        self,
        connection: str,
        table_name: str,
    ) -> MarketDataWriterProtocol:
        """
        Create a SqlMarketDataWriter instance.

        Args:
            connection: Database connection string.
            table_name: Name of the market data table.

        Returns:
            Configured SqlMarketDataWriter.

        Example:
            >>> component = OrchestratorCollectionComponent()
            >>> writer = component.create_sql_writer(
            ...     "postgresql://user:pass@host:5432/db",
            ...     "market_data",
            ... )

        """
        from ml.stores.providers import SqlMarketDataWriter

        return SqlMarketDataWriter(
            connection_string=connection,
            table_name=table_name,
        )

    def create_raw_writer(
        self,
        catalog: ParquetDataCatalog,
        dataset_type_toggles: Mapping[DatasetType, bool] | None = None,
        dataset_type_identifier_templates: Mapping[DatasetType, str] | None = None,
    ) -> RawIngestionWriterProtocol:
        """
        Create a ParquetCatalogRawWriter for dual-write scenarios.

        Args:
            catalog: ParquetDataCatalog instance.
            dataset_type_toggles: Optional dataset-type toggles for mirroring.
            dataset_type_identifier_templates: Optional identifier templates keyed by DatasetType.

        Returns:
            Configured ParquetCatalogRawWriter.

        Example:
            >>> component = OrchestratorCollectionComponent()
            >>> writer = component.create_raw_writer(parquet_catalog)

        """
        from ml.registry.dataclasses import DatasetType
        from ml.stores.io_raw import FilteredRawWriter
        from ml.stores.io_raw import ParquetCatalogRawWriter

        base_toggles = {
            DatasetType.BARS: True,
            DatasetType.TRADES: True,
            DatasetType.TBBO: True,
            DatasetType.MBP1: True,
        }
        if dataset_type_toggles:
            base_toggles.update(dataset_type_toggles)

        return FilteredRawWriter(
            ParquetCatalogRawWriter(
                catalog,
                dataset_type_identifier_templates=dataset_type_identifier_templates,
            ),
            enabled=base_toggles,
        )

    def _resolve_db_connection(self, connection: str | None) -> str | None:
        """
        Resolve database connection from parameter or environment.

        Checks in order:
        1. Explicit connection parameter
        2. DB_CONNECTION environment variable
        3. DATABASE_URL environment variable
        4. NAUTILUS_DB_CONNECTION environment variable

        Args:
            connection: Explicit connection string parameter.

        Returns:
            Resolved connection string or None if unavailable.

        """
        if connection:
            return connection

        db_conn = (
            os.getenv("DB_CONNECTION")
            or os.getenv("DATABASE_URL")
            or os.getenv("NAUTILUS_DB_CONNECTION")
        )
        return db_conn

    def _create_ingestor(self, api_key: str) -> Any:
        """
        Create a DatabentoIngestor with the given API key.

        Args:
            api_key: Databento API key.

        Returns:
            Configured DatabentoIngestor.

        """
        from ml.data.ingest.databento_adapter import DatabentoAPIClient
        from ml.data.ingest.resume import DatabentoIngestor

        return DatabentoIngestor(client=DatabentoAPIClient(api_key=api_key))

    def _create_domain_loader(
        self,
        api_key: str,
        config: SchedulerConfig,
    ) -> DomainWindowLoaderProtocol:
        """
        Create a domain loader for dual-write scenarios.

        The domain loader downloads data from Databento and converts it to
        Nautilus domain objects for writing to the ParquetDataCatalog.

        Args:
            api_key: Databento API key.
            config: Scheduler configuration for price precision.

        Returns:
            Domain loader implementing DomainWindowLoaderProtocol.

        """
        from ml.data.ingest.orchestrator import DomainWindowLoaderProtocol

        class _DomainLoader(DomainWindowLoaderProtocol):
            """Internal domain loader for dual-write scenarios."""

            def __init__(
                self,
                key: str,
                price_precision: int | None,
            ) -> None:
                self._key = key
                self._price_precision = price_precision

            def load(
                self,
                *,
                dataset_id: str,
                schema: str,
                instrument_id: str,
                start_ns: int,
                end_ns: int,
            ) -> list[Any]:
                """Load domain objects for a time window."""
                import databento as db
                from nautilus_trader.model.identifiers import InstrumentId as _IID

                from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader as _DBL

                sym, venue = (
                    instrument_id.split(".")
                    if "." in instrument_id
                    else (instrument_id, "")
                )
                s_dt = datetime.fromtimestamp(start_ns / 1e9, tz=UTC)
                e_dt = datetime.fromtimestamp(end_ns / 1e9, tz=UTC)
                client_h = db.Historical(self._key)

                with tempfile.TemporaryDirectory() as td:
                    path = f"{td}/{sym}_{s_dt:%Y%m%d%H%M%S}_{schema}.dbn"
                    resp = client_h.timeseries.get_range(
                        dataset=dataset_id,
                        symbols=[sym],
                        schema=schema,
                        start=s_dt,
                        end=e_dt,
                    )
                    resp.to_file(path)

                    venue_map = {
                        "XNAS": "NASDAQ",
                        "XNYS": "NYSE",
                        "ARCX": "ARCA",
                        "BATS": "BATS",
                        "GLBX": "GLBX",
                    }
                    _venue = venue_map.get(venue, venue) if venue else ""
                    inst = _IID.from_str(f"{sym}.{_venue}" if _venue else sym)
                    loader = _DBL()
                    items = loader.from_dbn_file(
                        path=path,
                        instrument_id=inst,
                        price_precision=self._price_precision,
                        bars_timestamp_on_close=(
                            True if "ohlcv" in schema or "bar" in schema else False
                        ),
                        include_trades=True if "trade" in schema else False,
                        as_legacy_cython=True,
                    )
                    return list(items) if items else []

        return _DomainLoader(api_key, config.databento.price_precision)

    def _create_orchestrator(
        self,
        *,
        coverage: CoverageProviderProtocol,
        writer: MarketDataWriterProtocol,
        registry: RegistryProtocol,
        ingestor: Any,
        raw_writer: RawIngestionWriterProtocol | None,
        domain_loader: DomainWindowLoaderProtocol | None,
    ) -> IngestionOrchestrator:
        """
        Create an IngestionOrchestrator with the given components.

        Args:
            coverage: Coverage provider for gap detection.
            writer: Market data writer for persistence.
            registry: DataRegistry for event tracking.
            ingestor: DatabentoIngestor for data fetching.
            raw_writer: Optional raw writer for dual-write.
            domain_loader: Optional domain loader for dual-write.

        Returns:
            Configured IngestionOrchestrator.

        """
        from ml.data.ingest.orchestrator import IngestionOrchestrator

        return IngestionOrchestrator(
            coverage=coverage,
            writer=writer,
            registry=registry,
            ingestor=ingestor,
            raw_writer=raw_writer,
            domain_loader=domain_loader,
        )

    def _resolve_bindings(
        self,
        config: SchedulerConfig,
        orchestrator: IngestionOrchestrator,
    ) -> tuple[ResolvedMarketBinding, ...]:
        """
        Resolve market bindings from config if configured.

        Args:
            config: Scheduler configuration with market_inputs or market_dataset_id.
            orchestrator: IngestionOrchestrator for binding resolution.

        Returns:
            Tuple of resolved market bindings, empty if not configured.

        """
        from ml.data.ingest.orchestrator import IngestionOrchestrator

        if not config.market_inputs and not config.market_dataset_id:
            return ()

        base_symbols = sorted({sym.split(".")[0].upper() for sym in config.symbols})
        return IngestionOrchestrator.resolve_market_bindings(
            symbols=base_symbols,
            instrument_ids=tuple(config.symbols),
            market_dataset_id=config.market_dataset_id or config.databento.dataset,
            market_inputs=config.market_inputs,
        )

    def _execute_backfill(
        self,
        config: SchedulerConfig,
        orchestrator: IngestionOrchestrator,
        bindings: tuple[ResolvedMarketBinding, ...],
    ) -> None:
        """
        Execute backfill using orchestrator with resolved bindings or per-symbol.

        Args:
            config: Scheduler configuration with Databento settings.
            orchestrator: IngestionOrchestrator for backfill execution.
            bindings: Resolved market bindings or empty tuple.

        """
        if bindings:
            processed: set[str] = set()
            for binding in bindings:
                if binding.binding_id in processed:
                    continue
                orchestrator.backfill_binding(binding=binding, lookback_days=1)
                processed.add(binding.binding_id)
        else:
            for symbol in config.symbols:
                orchestrator.backfill_gaps(
                    dataset_id=config.databento.dataset,
                    schema=config.databento.schema,
                    instrument_id=symbol,
                    lookback_days=1,
                    state=None,
                )


__all__ = [
    "OrchestratorCollectionComponent",
    "OrchestratorCollectionProtocol",
]
