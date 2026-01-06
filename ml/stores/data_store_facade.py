#!/usr/bin/env python3

"""
DataStore facade integrating all 6 decomposed components.

This module provides the final facade layer for Phase 2.4.7, wiring together:
- SchemaValidatorComponent (Phase 2.4.1)
- DataWriterComponent (Phase 2.4.2)
- DataReaderComponent (Phase 2.4.3)
- EventEmitterComponent (Phase 2.4.4)
- ContractEnforcerComponent (Phase 2.4.5)
- StoreOperationsComponent (Phase 2.4.6)

The facade maintains 100% backward compatibility with the legacy DataStore API
while delegating all operations to specialized components.

Phase 2.4.7 - Final Facade Integration

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ml._imports import HAS_POLARS
from ml.common.message_bus import MessagePublisherProtocol
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DatasetType
from ml.stores.io_raw import RawIngestionWriterProtocol


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.common.protocols import ContractEnforcerProtocol
    from ml.stores.common.protocols import DataReaderProtocol
    from ml.stores.common.protocols import DataWriterProtocol
    from ml.stores.common.protocols import EventEmitterProtocol
    from ml.stores.common.protocols import SchemaValidatorProtocol
    from ml.stores.common.protocols import StoreOperationsProtocol
    from ml.stores.earnings_store import EarningsStore
    from ml.stores.feature_store_facade import FeatureStore
    from ml.stores.model_store import ModelStore
    from ml.stores.strategy_store import StrategyStore

if HAS_POLARS:
    import polars as pl
else:  # pragma: no cover
    pl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================


@dataclass(frozen=True)
class DataStoreConfig:
    """
    Configuration for DataStore facade.

    Attributes
    ----------
    connection_string : str
        PostgreSQL connection string
    registry : RegistryProtocol | None
        Data registry for manifests and contracts
    feature_store : FeatureStore | None
        Feature store instance
    model_store : ModelStore | None
        Model store instance
    strategy_store : StrategyStore | None
        Strategy store instance
    earnings_store : EarningsStore | None
        Earnings store instance
    publisher : MessagePublisherProtocol | None
        Message bus publisher
    enable_publishing : bool
        Enable message bus publishing
    schema_migration_window_hours : int
        Schema migration window duration (hours)
    fail_closed : bool
        Fail-closed mode for validation failures
    auto_register_datasets : bool
        Auto-register missing datasets
    cache_manifests : bool
        Cache manifest lookups
    cache_ttl_seconds : int
        Manifest cache TTL
    batch_size : int
        Default batch size for write operations

    """

    connection_string: str
    registry: RegistryProtocol | None = None
    feature_store: FeatureStore | None = None
    model_store: ModelStore | None = None
    strategy_store: StrategyStore | None = None
    earnings_store: EarningsStore | None = None
    raw_writer: RawIngestionWriterProtocol | None = None
    publisher: MessagePublisherProtocol | None = None
    enable_publishing: bool = False
    schema_migration_window_hours: int = 24
    fail_closed: bool = True
    auto_register_datasets: bool = True
    cache_manifests: bool = True
    cache_ttl_seconds: int = 60
    batch_size: int = 10000

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.schema_migration_window_hours < 0:
            raise ValueError("schema_migration_window_hours must be >= 0")
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be >= 0")
        if self.batch_size < 0:
            raise ValueError("batch_size must be >= 0")


# =========================================================================
# DataStore Facade
# =========================================================================


class DataStoreFacade:
    """
    Facade integrating all 6 DataStore components.

    This facade delegates all operations to specialized components while
    maintaining 100% backward compatibility with the legacy DataStore API.

    Component Delegation:
    - Write operations → DataWriterComponent
    - Read operations → DataReaderComponent
    - Schema validation → SchemaValidatorComponent
    - Event emission → EventEmitterComponent
    - Contract enforcement → ContractEnforcerComponent
    - Health/metrics/close → StoreOperationsComponent

    All 5 Universal ML Architecture Patterns enforced:
    1. 4-Store + 4-Registry Integration (via all components)
    2. Protocol-First Interface Design (component protocols)
    3. Hot/Cold Path Separation (maintained from components)
    4. Progressive Fallback Chains (via StoreOperationsComponent)
    5. Centralized Metrics Bootstrap (all components use metrics_bootstrap)

    Example:
        >>> config = DataStoreConfig(
        ...     connection_string="postgresql://...",
        ...     registry=registry,
        ...     feature_store=feature_store,
        ...     model_store=model_store,
        ...     strategy_store=strategy_store,
        ... )
        >>> store = DataStoreFacade(config)
        >>> event = store.write_ingestion(
        ...     dataset_id="bars_eurusd_1m",
        ...     records=bars,
        ...     source="historical",
        ...     run_id="run_123",
        ... )
        >>> features = store.get_features_at_or_before(
        ...     instrument_id="EURUSD.SIM",
        ...     ts_event=1699999990000000000,
        ... )

    """

    def __init__(
        self,
        config: DataStoreConfig | None = None,
        *,
        # Legacy kwargs for backward compatibility
        connection_string: str | None = None,
        registry: RegistryProtocol | None = None,
        feature_store: FeatureStore | None = None,
        model_store: ModelStore | None = None,
        strategy_store: StrategyStore | None = None,
        earnings_store: EarningsStore | None = None,
        raw_writer: RawIngestionWriterProtocol | None = None,
        publisher: MessagePublisherProtocol | None = None,
        enable_publishing: bool = False,
        fail_on_validation_error: bool = True,
        schema_migration_window_hours: int = 24,
        # Component overrides
        schema_validator: SchemaValidatorProtocol | None = None,
        data_writer: DataWriterProtocol | None = None,
        data_reader: DataReaderProtocol | None = None,
        event_emitter: EventEmitterProtocol | None = None,
        contract_enforcer: ContractEnforcerProtocol | None = None,
        store_operations: StoreOperationsProtocol | None = None,
        # Additional legacy kwargs (ignored but accepted for compatibility)
        **kwargs: Any,
    ) -> None:
        """
        Initialize DataStore facade with components.

        Supports both the new config-based API and legacy kwargs for backward compatibility.

        Args:
            config: DataStore configuration (new API)
            connection_string: PostgreSQL connection string (legacy)
            registry: Data registry (legacy)
            feature_store: Feature store instance (legacy)
            model_store: Model store instance (legacy)
            strategy_store: Strategy store instance (legacy)
            earnings_store: Earnings store instance (legacy)
            publisher: Message bus publisher (legacy)
            enable_publishing: Enable message bus publishing (legacy)
            fail_on_validation_error: Fail-closed mode (legacy)
            schema_migration_window_hours: Schema migration window (legacy)
            batch_size: Default write batch size (legacy, forwarded to config)
            schema_validator: Schema validation component (auto-created if None)
            data_writer: Data writer component (auto-created if None)
            data_reader: Data reader component (auto-created if None)
            event_emitter: Event emitter component (auto-created if None)
            contract_enforcer: Contract enforcer component (auto-created if None)
            store_operations: Store operations component (auto-created if None)
            **kwargs: Additional kwargs (ignored for forward compatibility)

        Raises:
            ValueError: If neither config nor connection_string is provided

        """
        # Handle backward compatibility: build config from legacy kwargs if needed
        if config is None:
            if connection_string is None:
                raise ValueError("Either config or connection_string must be provided")
            config = DataStoreConfig(
                connection_string=connection_string,
                registry=registry,
                feature_store=feature_store,
                model_store=model_store,
                strategy_store=strategy_store,
                earnings_store=earnings_store,
                raw_writer=raw_writer,
                publisher=publisher,
                enable_publishing=enable_publishing,
                fail_closed=fail_on_validation_error,
                schema_migration_window_hours=schema_migration_window_hours,
                batch_size=int(kwargs.get("batch_size", 10000)),
            )

        self._config = config
        self._logger = logger

        # Initialize components (auto-create if not provided)
        self._schema_validator = schema_validator or self._create_schema_validator()
        self._contract_enforcer = contract_enforcer or self._create_contract_enforcer()
        self._event_emitter = event_emitter or self._create_event_emitter()
        self._data_writer = data_writer or self._create_data_writer()
        self._data_reader = data_reader or self._create_data_reader()
        self._store_operations = store_operations or self._create_store_operations()

        self._logger.info(
            "DataStoreFacade initialized with 6 components",
            extra={
                "connection_string": config.connection_string,
                "enable_publishing": config.enable_publishing,
                "fail_closed": config.fail_closed,
            },
        )

    # =====================================================================
    # Component Factory Methods
    # =====================================================================

    def _create_schema_validator(self) -> SchemaValidatorProtocol:
        """
        Create schema validator component.

        Raises:
            ValueError: If registry is not provided.
        """
        from ml.stores.common.schema_validator import SchemaValidatorComponent

        if self._config.registry is None:
            raise ValueError(
                "registry is required for SchemaValidatorComponent. "
                "DataStoreFacade requires explicit registry for schema validation."
            )

        return SchemaValidatorComponent(
            data_registry=self._config.registry,
            allow_schema_migration=True,
            schema_migration_window_hours=self._config.schema_migration_window_hours,
        )

    def _create_contract_enforcer(self) -> ContractEnforcerProtocol:
        """
        Create contract enforcer component.

        Raises:
            ValueError: If registry is not provided.
        """
        from ml.stores.common.contract_enforcer import ContractEnforcerComponent

        if self._config.registry is None:
            raise ValueError(
                "registry is required for ContractEnforcerComponent. "
                "DataStoreFacade requires explicit registry for contract enforcement."
            )

        return ContractEnforcerComponent(
            registry=self._config.registry,
            allow_schema_migration=True,
            schema_migration_window_hours=self._config.schema_migration_window_hours,
            fail_on_validation_error=self._config.fail_closed,
        )

    def _create_event_emitter(self) -> EventEmitterProtocol:
        """
        Create event emitter component.

        Raises:
            ValueError: If registry is not provided.
        """
        from ml.stores.common.event_emitter import EventEmitterComponent

        if self._config.registry is None:
            raise ValueError(
                "registry is required for EventEmitterComponent. "
                "DataStoreFacade requires explicit registry for event emission."
            )

        try:
            from ml.config.bus import MessageBusConfig

            bus_config = MessageBusConfig.from_env()
            topic_scheme = str(bus_config.scheme)
            topic_prefix = str(bus_config.topic_prefix)
        except Exception:  # pragma: no cover - defensive fallback
            topic_scheme = "domain_op"
            topic_prefix = "events.ml"

        self._topic_scheme = topic_scheme
        self._topic_prefix = topic_prefix

        return EventEmitterComponent(
            registry=self._config.registry,
            publisher=self._config.publisher,
            enable_publishing=self._config.enable_publishing,
            topic_scheme=topic_scheme,
            topic_prefix=topic_prefix,
        )

    def _create_data_writer(self) -> DataWriterProtocol:
        """
        Create data writer component.

        Raises:
            ValueError: If registry or any required store is not provided.
        """
        from ml.stores.common.data_writer import DataWriterComponent

        # Validate all required dependencies are provided
        missing: list[str] = []
        if self._config.registry is None:
            missing.append("registry")
        if self._config.feature_store is None:
            missing.append("feature_store")
        if self._config.model_store is None:
            missing.append("model_store")
        if self._config.strategy_store is None:
            missing.append("strategy_store")
        if self._config.earnings_store is None:
            missing.append("earnings_store")

        if missing:
            raise ValueError(
                f"DataWriterComponent requires: {', '.join(missing)}. "
                "DataStoreFacade requires explicit stores and registry for write operations."
            )
        assert self._config.feature_store is not None
        assert self._config.model_store is not None
        assert self._config.strategy_store is not None
        assert self._config.earnings_store is not None
        assert self._config.registry is not None

        return DataWriterComponent(
            feature_store=self._config.feature_store,
            model_store=self._config.model_store,
            strategy_store=self._config.strategy_store,
            earnings_store=self._config.earnings_store,
            raw_writer=self._config.raw_writer,
            validator=self._schema_validator,
            registry=self._config.registry,
            fail_on_validation_error=self._config.fail_closed,
        )

    def _create_data_reader(self) -> DataReaderProtocol:
        """
        Create data reader component.

        Raises:
            ValueError: If registry or any required store is not provided.
        """
        from ml.stores.common.data_reader import DataReaderComponent

        # Validate all required dependencies are provided
        missing: list[str] = []
        if self._config.registry is None:
            missing.append("registry")
        if self._config.feature_store is None:
            missing.append("feature_store")
        if self._config.model_store is None:
            missing.append("model_store")
        if self._config.strategy_store is None:
            missing.append("strategy_store")
        if self._config.earnings_store is None:
            missing.append("earnings_store")

        if missing:
            raise ValueError(
                f"DataReaderComponent requires: {', '.join(missing)}. "
                "DataStoreFacade requires explicit stores and registry for read operations."
            )
        assert self._config.feature_store is not None
        assert self._config.model_store is not None
        assert self._config.strategy_store is not None
        assert self._config.earnings_store is not None
        assert self._config.registry is not None

        return DataReaderComponent(
            feature_store=self._config.feature_store,
            model_store=self._config.model_store,
            strategy_store=self._config.strategy_store,
            earnings_store=self._config.earnings_store,
            registry=self._config.registry,
        )

    def _create_store_operations(self) -> StoreOperationsProtocol:
        """
        Create store operations component.
        """
        from ml.stores.common.store_operations import StoreOperationsComponent

        return StoreOperationsComponent(
            connection_string=self._config.connection_string,
            feature_store=self._config.feature_store,
            model_store=self._config.model_store,
            strategy_store=self._config.strategy_store,
            earnings_store=self._config.earnings_store,
            data_registry=self._config.registry,
        )

    # =====================================================================
    # Write Operations (Delegate to DataWriterComponent)
    # =====================================================================

    def write_ingestion(
        self,
        dataset_id: str,
        records: Any,
        source: str,
        run_id: str,
        instrument_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Write ingestion data with contract validation and event emission.

        Delegates to DataWriterComponent.

        Args:
            dataset_id: Dataset identifier
            records: Data records to write
            source: Data source (live, historical, backfill)
            run_id: Processing run identifier
            instrument_id: Instrument identifier (extracted if not provided)

        Returns:
            DataEvent tracking the write operation

        Raises:
            ValueError: If validation fails in fail-closed mode
            KeyError: If dataset not registered

        Example:
            >>> event = store.write_ingestion(
            ...     dataset_id="bars_eurusd_1m",
            ...     records=bars,
            ...     source="historical",
            ...     run_id="run_123",
            ... )
            >>> assert event.status == "success"
            >>> assert event.record_count == len(bars)

        """
        return self._data_writer.write_ingestion(
            dataset_id=dataset_id,
            records=records,
            source=source,
            run_id=run_id,
            instrument_id=instrument_id,
        )

    def write_features(
        self,
        instrument_id: str,
        features: list[Any],
        source: str = "computed",
        run_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Write features with validation and event emission.

        Delegates to DataWriterComponent.

        Args:
            instrument_id: Instrument identifier
            features: Feature data to store
            source: Data source
            run_id: Processing run identifier

        Returns:
            DataEvent tracking the write operation

        Example:
            >>> from ml.stores.base import FeatureData
            >>> features = [
            ...     FeatureData(
            ...         instrument_id="EURUSD.SIM",
            ...         feature_name="close",
            ...         value=1.0950,
            ...         ts_event=1699999990000000000,
            ...         ts_init=1699999990000000000,
            ...     ),
            ... ]
            >>> event = store.write_features(
            ...     instrument_id="EURUSD.SIM",
            ...     features=features,
            ... )

        """
        return self._data_writer.write_features(
            instrument_id=instrument_id,
            features=features,
            source=source,
            run_id=run_id,
        )

    def write_predictions(
        self,
        predictions: list[Any],
        source: str = "inference",
        run_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Write model predictions with validation and event emission.

        Delegates to DataWriterComponent.

        Args:
            predictions: Model predictions to store
            source: Data source
            run_id: Processing run identifier

        Returns:
            DataEvent tracking the write operation

        """
        return self._data_writer.write_predictions(
            predictions=predictions,
            source=source,
            run_id=run_id,
        )

    def write_signals(
        self,
        signals: list[Any],
        source: str = "strategy",
        run_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Write strategy signals with validation and event emission.

        Delegates to DataWriterComponent.

        Args:
            signals: Strategy signals to store
            source: Data source
            run_id: Processing run identifier

        Returns:
            DataEvent tracking the write operation

        """
        return self._data_writer.write_signals(
            signals=signals,
            source=source,
            run_id=run_id,
        )

    def write_earnings_actual(
        self,
        *,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        eps_basic: float | None = None,
        net_income: float | None = None,
        operating_income: float | None = None,
        shares_outstanding: int | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        fiscal_quarter: int | None = None,
        source: str = "historical",
        run_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Persist an earnings actual record with contract validation.

        Delegates to DataWriterComponent.

        Args:
            ticker: Stock ticker symbol
            period_end: Period end date
            filing_date: Filing date
            eps_diluted: Diluted earnings per share
            revenue: Revenue amount
            ts_event: Event timestamp (nanoseconds)
            ts_init: Initialization timestamp (nanoseconds)
            eps_basic: Basic earnings per share
            net_income: Net income amount
            operating_income: Operating income amount
            shares_outstanding: Shares outstanding
            filing_type: Filing type (10-Q, 10-K, etc.)
            fiscal_year: Fiscal year
            fiscal_quarter: Fiscal quarter
            source: Data source
            run_id: Processing run identifier

        Returns:
            DataEvent tracking the write operation

        """
        return self._data_writer.write_earnings_actual(
            ticker=ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=eps_diluted,
            revenue=revenue,
            ts_event=ts_event,
            ts_init=ts_init,
            eps_basic=eps_basic,
            net_income=net_income,
            operating_income=operating_income,
            shares_outstanding=shares_outstanding,
            filing_type=filing_type,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            source=source,
            run_id=run_id,
        )

    def write_earnings_estimate(
        self,
        *,
        ticker: str,
        estimate_date: str,
        period_end: str,
        eps_consensus: float | None,
        ts_event: int,
        ts_init: int,
        revenue_consensus: float | None = None,
        num_analysts: int | None = None,
        source: str = "historical",
        run_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Persist an earnings estimate record with contract validation.

        Delegates to DataWriterComponent.

        Args:
            ticker: Stock ticker symbol
            estimate_date: Estimate date
            period_end: Period end date
            eps_consensus: Consensus EPS estimate
            ts_event: Event timestamp (nanoseconds)
            ts_init: Initialization timestamp (nanoseconds)
            revenue_consensus: Consensus revenue estimate
            num_analysts: Number of analysts in consensus
            source: Data source
            run_id: Processing run identifier

        Returns:
            DataEvent tracking the write operation

        """
        return self._data_writer.write_earnings_estimate(
            ticker=ticker,
            estimate_date=estimate_date,
            period_end=period_end,
            eps_consensus=eps_consensus,
            ts_event=ts_event,
            ts_init=ts_init,
            revenue_consensus=revenue_consensus,
            num_analysts=num_analysts,
            source=source,
            run_id=run_id,
        )

    # =====================================================================
    # Read Operations (Delegate to DataReaderComponent)
    # =====================================================================

    def get_features_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
    ) -> dict[str, float] | None:
        """
        Return latest feature values at or before the given timestamp.

        HOT PATH: P99 < 5ms requirement
        Delegates to DataReaderComponent.

        Args:
            instrument_id: Instrument identifier
            ts_event: Timestamp in nanoseconds (point-in-time query)

        Returns:
            Dictionary mapping feature names to values (all features),
            or None if no features exist before timestamp.

        Example:
            >>> features = store.get_features_at_or_before(
            ...     instrument_id="EURUSD.SIM",
            ...     ts_event=1699999990000000000,
            ... )
            >>> assert features is not None
            >>> assert "close" in features
            >>> assert features["close"] > 0.0

        """
        return self._data_reader.get_features_at_or_before(
            instrument_id=instrument_id,
            ts_event=ts_event,
        )

    def get_latest_prediction_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        model_id: str | None = None,
    ) -> Any | None:  # PredictionRecord | None
        """
        Return latest prediction at or before ts_event.

        Delegates to DataReaderComponent.

        Args:
            instrument_id: Instrument identifier
            ts_event: Timestamp in nanoseconds (point-in-time query)
            model_id: Optional model identifier filter

        Returns:
            Prediction record or None when not found

        """
        return self._data_reader.get_latest_prediction_at_or_before(
            instrument_id=instrument_id,
            ts_event=ts_event,
            model_id=model_id,
        )

    def get_latest_signal_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        strategy_id: str | None = None,
    ) -> Any | None:  # SignalRecord | None
        """
        Return latest strategy signal at or before ts_event.

        Delegates to DataReaderComponent.

        Args:
            instrument_id: Instrument identifier
            ts_event: Timestamp in nanoseconds (point-in-time query)
            strategy_id: Optional strategy identifier filter

        Returns:
            Signal record or None when not found

        """
        return self._data_reader.get_latest_signal_at_or_before(
            instrument_id=instrument_id,
            ts_event=ts_event,
            strategy_id=strategy_id,
        )

    def read_ingestion_data(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        dataset_type: DatasetType | None = None,
    ) -> Any:  # pl.DataFrame
        """
        Read ingestion data for a specific time range.

        COLD PATH: Bulk read operation
        Delegates to DataReaderComponent.

        Args:
            instrument_id: Instrument identifier
            start_ts: Start timestamp in nanoseconds (inclusive)
            end_ts: End timestamp in nanoseconds (exclusive)
            dataset_type: Optional dataset type filter

        Returns:
            DataFrame with ingestion data

        """
        return self._data_reader.read_ingestion_data(
            instrument_id=instrument_id,
            start_ts=start_ts,
            end_ts=end_ts,
            dataset_type=dataset_type,
        )

    def read_features(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        feature_names: list[str] | None = None,
    ) -> Any:  # pl.DataFrame
        """
        Read feature data for a specific time range.

        COLD PATH: Bulk read operation
        Delegates to DataReaderComponent.

        Args:
            instrument_id: Instrument identifier
            start_ts: Start timestamp in nanoseconds (inclusive)
            end_ts: End timestamp in nanoseconds (exclusive)
            feature_names: Optional list of feature names to retrieve

        Returns:
            DataFrame with feature data

        """
        return self._data_reader.read_features(
            instrument_id=instrument_id,
            start_ts=start_ts,
            end_ts=end_ts,
            feature_names=feature_names,
        )

    def read_predictions(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        model_id: str | None = None,
    ) -> Any:  # pl.DataFrame
        """
        Read model predictions for a specific time range.

        COLD PATH: Bulk read operation
        Delegates to DataReaderComponent.

        Args:
            instrument_id: Instrument identifier
            start_ts: Start timestamp in nanoseconds (inclusive)
            end_ts: End timestamp in nanoseconds (exclusive)
            model_id: Optional model identifier filter

        Returns:
            DataFrame with prediction data

        """
        return self._data_reader.read_predictions(
            instrument_id=instrument_id,
            start_ts=start_ts,
            end_ts=end_ts,
            model_id=model_id,
        )

    def read_signals(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        strategy_id: str | None = None,
    ) -> Any:  # pl.DataFrame
        """
        Read strategy signals for a specific time range.

        COLD PATH: Bulk read operation
        Delegates to DataReaderComponent.

        Args:
            instrument_id: Instrument identifier
            start_ts: Start timestamp in nanoseconds (inclusive)
            end_ts: End timestamp in nanoseconds (exclusive)
            strategy_id: Optional strategy identifier filter

        Returns:
            DataFrame with signal data

        """
        return self._data_reader.read_signals(
            instrument_id=instrument_id,
            start_ts=start_ts,
            end_ts=end_ts,
            strategy_id=strategy_id,
        )

    def read_earnings_actual(
        self,
        *,
        symbol: str,
        start_date: str | None,
        end_date: str | None,
        as_of_ts: int | None = None,
    ) -> Any:  # pl.DataFrame
        """
        Read earnings actuals for a ticker within date range.

        COLD PATH: Bulk read operation
        Delegates to DataReaderComponent.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD) or None
            end_date: End date (YYYY-MM-DD) or None
            as_of_ts: Optional point-in-time timestamp (nanoseconds)

        Returns:
            DataFrame with earnings actuals data

        """
        return self._data_reader.read_earnings_actual(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            as_of_ts=as_of_ts,
        )

    def read_earnings_estimate(
        self,
        *,
        symbol: str,
        start_date: str | None,
        end_date: str | None,
        as_of_ts: int | None = None,
    ) -> Any:  # pl.DataFrame
        """
        Read earnings estimates for a ticker within date range.

        COLD PATH: Bulk read operation
        Delegates to DataReaderComponent.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD) or None
            end_date: End date (YYYY-MM-DD) or None
            as_of_ts: Optional point-in-time timestamp (nanoseconds)

        Returns:
            DataFrame with earnings estimates data

        """
        return self._data_reader.read_earnings_estimate(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            as_of_ts=as_of_ts,
        )

    def get_earnings_actuals_at_or_before(
        self,
        *,
        ticker: str,
        ts_event: int,
        limit: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return earnings actuals visible at ``ts_event``.

        Delegates to the data reader; used in earnings-aware pipelines to enforce
        point-in-time correctness.
        """
        if limit <= 0:
            return []
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize_ts

        as_of_ts = _sanitize_ts(
            int(ts_event),
            context="data_store_facade.get_earnings_actuals_at_or_before:ts_event",
        )
        earnings_store = self._config.earnings_store
        assert earnings_store is not None
        records = earnings_store.get_actuals(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            as_of_ts=as_of_ts,
        )
        if len(records) <= limit:
            return list(records)
        return list(records[:limit])

    def get_earnings_estimate_at_or_before(
        self,
        *,
        ticker: str,
        period_end: str,
        ts_event: int,
    ) -> dict[str, Any] | None:
        """
        Return the latest consensus estimate for ``period_end`` at ``ts_event``.
        """
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize_ts

        as_of_ts = _sanitize_ts(
            int(ts_event),
            context="data_store_facade.get_earnings_estimate_at_or_before:ts_event",
        )
        earnings_store = self._config.earnings_store
        assert earnings_store is not None
        return earnings_store.get_estimates(
            ticker=ticker,
            period_end=period_end,
            as_of_ts=as_of_ts,
        )

    def read_range(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
    ) -> Any:  # pl.DataFrame
        """
        Read data range for any dataset type.

        COLD PATH: Generic bulk read operation
        Delegates to DataReaderComponent (uses internal routing).

        Args:
            dataset_id: Dataset identifier
            instrument_id: Instrument identifier
            start_ts: Start timestamp in nanoseconds (inclusive)
            end_ts: End timestamp in nanoseconds (exclusive)

        Returns:
            DataFrame with data for the specified range

        """
        # Route based on dataset_id prefix (simplified routing logic)
        if "feature" in dataset_id.lower():
            return self.read_features(
                instrument_id=instrument_id,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        elif "prediction" in dataset_id.lower() or "model" in dataset_id.lower():
            return self.read_predictions(
                instrument_id=instrument_id,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        elif "signal" in dataset_id.lower() or "strategy" in dataset_id.lower():
            return self.read_signals(
                instrument_id=instrument_id,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        else:
            # Default to ingestion data
            return self.read_ingestion_data(
                instrument_id=instrument_id,
                start_ts=start_ts,
                end_ts=end_ts,
            )

    # =====================================================================
    # Validation Operations (Delegate to SchemaValidatorComponent)
    # =====================================================================

    def preflight_check(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],
        strict: bool = True,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Perform preflight schema validation before processing data.

        Delegates to SchemaValidatorComponent.

        Args:
            dataset_id: Dataset identifier
            data: Data to validate
            strict: If True, require exact schema match. If False, allow subset.

        Returns:
            (success, error_message, validation_details)

        Example:
            >>> success, error, details = store.preflight_check(
            ...     dataset_id="bars_eurusd_1m",
            ...     data=bars,
            ...     strict=True,
            ... )
            >>> assert success is True
            >>> assert error is None

        """
        return self._schema_validator.preflight_check(
            dataset_id=dataset_id,
            data=data,
            strict=strict,
        )

    def validate_batch(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],
        strict_mode: bool = False,
    ) -> Any:  # QualityReport
        """
        Validate a batch of data against the dataset's contract.

        Delegates to SchemaValidatorComponent.

        Args:
            dataset_id: Dataset identifier
            data: Data to validate
            strict_mode: If True, apply stricter validation rules

        Returns:
            QualityReport with quality score and violations

        Example:
            >>> report = store.validate_batch(
            ...     dataset_id="bars_eurusd_1m",
            ...     data=bars,
            ... )
            >>> assert report.quality_score >= 0.9
            >>> assert report.failed_records == 0

        """
        return self._schema_validator.validate_batch(
            dataset_id=dataset_id,
            data=data,
            strict_mode=strict_mode,
        )

    # =====================================================================
    # Event Emission (Delegate to EventEmitterComponent)
    # =====================================================================

    def emit_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: Stage | str,
        source: Source | str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: str = "success",
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit a dataset processing event via centralized registry.

        COLD PATH: Event emission is async, non-blocking
        Delegates to EventEmitterComponent.

        Args:
            dataset_id: Dataset identifier
            instrument_id: Instrument identifier
            stage: Processing stage (DATA_INGESTED, FEATURES_COMPUTED, etc.)
            source: Event source (LIVE, HISTORICAL, BACKFILL)
            run_id: Unique identifier for this processing run
            ts_min: Minimum timestamp (ns) for covered data
            ts_max: Maximum timestamp (ns) for covered data
            count: Number of records processed
            status: Status string (EventStatus.value)
            error: Error message if status is failed
            metadata: Additional metadata to attach to the event

        """
        self._event_emitter.emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage,
            source=source,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
            status=status,
            error=error,
            metadata=metadata,
        )

    def emit_dataset_event(
        self,
        *,
        dataset_id: str,
        status: EventStatus | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit a dataset-specific event with simplified parameter set.

        COLD PATH: Event emission is async, non-blocking
        Delegates to EventEmitterComponent.

        Args:
            dataset_id: Dataset identifier
            status: Event status (SUCCESS, FAILED, PARTIAL)
            metadata: Additional event metadata

        """
        self._event_emitter.emit_dataset_event(
            dataset_id=dataset_id,
            status=status,
            metadata=metadata,
        )

    # =====================================================================
    # Contract/Manifest Operations (Delegate to ContractEnforcerComponent)
    # =====================================================================

    def _get_manifest(self, dataset_id: str) -> Any:  # DatasetManifest
        """
        Get dataset manifest with caching and version check.

        Delegates to ContractEnforcerComponent (internal use).

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset manifest with schema, constraints, and metadata

        """
        return self._contract_enforcer.get_manifest(dataset_id)

    def _get_contract(self, dataset_id: str) -> Any:  # DataContract
        """
        Get data contract with caching and version check.

        Delegates to ContractEnforcerComponent (internal use).

        Args:
            dataset_id: Dataset identifier

        Returns:
            Data contract with validation rules and enforcement mode

        """
        return self._contract_enforcer.get_contract(dataset_id)

    # =====================================================================
    # Health/Metrics/Operations (Delegate to StoreOperationsComponent)
    # =====================================================================

    def get_health_status(self) -> dict[str, Any]:
        """
        Perform health check across all 4 stores + 4 registries.

        Delegates to StoreOperationsComponent.

        Returns:
            Health status with component-level details

        Example:
            >>> health = store.get_health_status()
            >>> assert health["overall_status"] in ["healthy", "degraded", "unhealthy"]
            >>> assert "components" in health

        """
        health = self._store_operations.health_check()

        # Parity with legacy DataStore and other facades: expose implementation + core components.
        health.setdefault("implementation", "component-based")
        health.setdefault(
            "schema_validator",
            "healthy" if self._schema_validator is not None else "unavailable",
        )
        health.setdefault(
            "contract_enforcer",
            "healthy" if self._contract_enforcer is not None else "unavailable",
        )
        health.setdefault(
            "data_reader",
            "healthy" if self._data_reader is not None else "unavailable",
        )
        health.setdefault(
            "data_writer",
            "healthy" if self._data_writer is not None else "unavailable",
        )
        health.setdefault(
            "event_emitter",
            "healthy" if self._event_emitter is not None else "unavailable",
        )
        health.setdefault(
            "store_operations",
            "healthy" if self._store_operations is not None else "unavailable",
        )

        # Flatten component status keys for legacy expectations
        components = health.get("components", {})
        for key in ("feature_store", "model_store", "strategy_store", "earnings_store"):
            if key in components:
                health.setdefault(key, components[key].get("status"))

        return health

    def get_performance_metrics(self) -> dict[str, float]:
        """
        Aggregate performance metrics from all components.

        Delegates to StoreOperationsComponent.

        Returns:
            Aggregated metrics for observability

        """
        return self._store_operations.get_metrics()

    def validate_configuration(self) -> list[str]:
        """
        Validate configuration and return list of issues.

        Delegates to StoreOperationsComponent.

        Returns:
            List of configuration issues (empty if valid)

        """
        issues: list[str] = []

        # Validate config constraints
        if not self._config.connection_string or not self._config.connection_string.strip():
            issues.append("connection_string must be set")
        if self._config.schema_migration_window_hours < 0:
            issues.append("schema_migration_window_hours must be >= 0")
        if self._config.cache_ttl_seconds < 0:
            issues.append("cache_ttl_seconds must be >= 0")
        if self._config.batch_size <= 0:
            issues.append("batch_size must be positive")

        # Delegate to store operations for store-level validation
        store_issues = self._store_operations.validate_configuration() if hasattr(
            self._store_operations,
            "validate_configuration",
        ) else []
        issues.extend(store_issues)

        return issues

    def close(self) -> None:
        """
        Gracefully shutdown all stores and clean up resources.

        Delegates to StoreOperationsComponent.

        """
        self._logger.info("Closing DataStoreFacade and all components")
        try:
            self._store_operations.close()
        except Exception as e:
            self._logger.error("Error closing store operations", exc_info=True, extra={"error": str(e)})


# Backwards-compatible alias for facade-only deployment.
DataStore = DataStoreFacade
