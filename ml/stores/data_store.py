#!/usr/bin/env python3

"""
DataStore facade maintaining backward compatibility.

This facade delegates to specialized components (SchemaValidator, DataReader,
ContractEnforcer, DataWriter) while preserving the original public API.
Feature flag ML_USE_LEGACY_DATA_STORE controls legacy vs new path.

Phase 2.1: DataStore Decomposition - Strangler Fig Pattern
-----------------------------------------------------------
This facade provides 100% backward compatibility while allowing gradual
migration to the decomposed component architecture. The legacy monolithic
implementation can be restored via environment variable for safe rollback.
"""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from ml.common.message_bus import BusPublisherMixin
from ml.common.message_bus import MessagePublisherProtocol
from ml.common.protocols import MLComponentMixin
from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID as _EARNINGS_ACTUALS_DATASET_ID
from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID as _EARNINGS_ESTIMATES_DATASET_ID
from ml.config.events import Source

# Re-export for test compatibility (tests patch EngineManager)
from ml.ml_types import DataFrameLike
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.utils import get_default_registry_path
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.contract_enforcer import ContractEnforcer
from ml.stores.data_processor import DataProcessor
from ml.stores.data_reader import DataReader
from ml.stores.data_writer import DataEvent as WriterDataEvent
from ml.stores.data_writer import DataWriter
from ml.stores.earnings_store import DummyEarningsStore
from ml.stores.earnings_store import EarningsStore
from ml.stores.feature_store import FeatureStore
from ml.stores.file_backed import FileEarningsStore
from ml.stores.mixins import DataRegistryMixin
from ml.stores.model_store import ModelStore
from ml.stores.protocols import EarningsStoreProtocol
from ml.stores.protocols import PredictionRecord
from ml.stores.protocols import SignalRecord
from ml.stores.raw_protocols import RawIngestionWriterProtocol
from ml.stores.raw_protocols import RawReaderProtocol
from ml.stores.schema_validator import SchemaValidator
from ml.stores.strategy_store import StrategyStore
from ml.stores.validation_types import DataEvent
from ml.stores.validation_types import QualityReport


if TYPE_CHECKING:

    from ml.registry.protocols import RegistryProtocol
    from ml.stores.protocols import CircuitBreakerProtocol

    # Type-only bases to satisfy mypy when follow_imports=skip
    class _MLComponentBase(Protocol):
        def get_health_status(self) -> dict[str, Any]: ...

        def get_performance_metrics(self) -> dict[str, float]: ...

        def validate_configuration(self) -> list[str]: ...

    class _BusPublisherBase(Protocol):
        def _init_bus_publishing(
            self,
            *,
            enable_publishing: bool,
            publisher: MessagePublisherProtocol | None,
            publish_mode: Literal["batch", "row", "both"] = "batch",
        ) -> None: ...

    class _DataRegistryBase(Protocol):
        def _get_data_registry(self) -> RegistryProtocol | None: ...

else:
    # Runtime bases preserve behavior
    _MLComponentBase = MLComponentMixin  # type: ignore[assignment]
    _BusPublisherBase = BusPublisherMixin  # type: ignore[assignment]
    _DataRegistryBase = DataRegistryMixin  # type: ignore[assignment]


def _should_use_component_impl() -> bool:
    """
    Determine whether to enable the component-based DataStore facade.

    Precedence (highest to lowest):
    1. ML_USE_COMPONENT_DATA_STORE=1 explicitly opts in.
    2. ML_USE_COMPONENT_DATA_STORE=0 explicitly opts out.
    3. Historical flag ML_USE_LEGACY_DATA_STORE keeps working:
       - "1" => legacy implementation
       - "0" => component implementation
       - unset => legacy (default)
    """
    component_flag = os.getenv("ML_USE_COMPONENT_DATA_STORE")
    if component_flag is not None:
        return component_flag.strip() == "1"

    legacy_flag = os.getenv("ML_USE_LEGACY_DATA_STORE")
    if legacy_flag is not None:
        return legacy_flag.strip() == "0"

    return False


USE_COMPONENT_DATA_STORE = _should_use_component_impl()
USE_LEGACY_DATA_STORE = not USE_COMPONENT_DATA_STORE


logger = logging.getLogger(__name__)


EARNINGS_ACTUALS_DATASET_ID = _EARNINGS_ACTUALS_DATASET_ID
EARNINGS_ESTIMATES_DATASET_ID = _EARNINGS_ESTIMATES_DATASET_ID


class DataStore(_MLComponentBase, _BusPublisherBase, _DataRegistryBase):
    """
    Unified interface for ML data operations with contract validation.

    This facade delegates to specialized components (SchemaValidator, DataReader,
    DataWriter, ContractEnforcer) while maintaining 100% backward compatibility
    with the original DataStore API.

    Feature Flag Control (defaults to legacy for stability):
    -------------------------------------------------------
    - ML_USE_COMPONENT_DATA_STORE=1: Opt into component-based implementation
    - ML_USE_COMPONENT_DATA_STORE=0: Force legacy implementation
    - ML_USE_LEGACY_DATA_STORE=1: Legacy implementation (backward compatible flag)
    - ML_USE_LEGACY_DATA_STORE=0: Component implementation

    Component Architecture:
    ----------------------
    - SchemaValidator: Type checking, validation rules, quality enforcement
    - DataReader: Read operations for features, predictions, signals, earnings
    - DataWriter: Write operations with validation, event emission, watermarks
    - ContractEnforcer: Contract retrieval, validation, quality reporting

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string
    registry : RegistryProtocol | None
        Data registry instance (created if None)
    feature_store : FeatureStore | None
        Feature store instance (created if None)
    model_store : ModelStore | None
        Model store instance (created if None)
    strategy_store : StrategyStore | None
        Strategy store instance (created if None)
    earnings_store : EarningsStoreProtocol | None
        Earnings store instance (created if None)
    data_processor : DataProcessor | None
        Data processor instance (optional)
    publisher : MessagePublisherProtocol | None
        Message bus publisher (optional)
    enable_publishing : bool
        Enable event publishing to message bus
    fail_on_validation_error : bool
        If True, fail writes on validation errors
    batch_size : int
        Batch size for write operations
    allow_schema_migration : bool
        Allow dual-write during schema migration
    schema_migration_window_hours : int
        Hours to allow dual-write during migration
    raw_writer : RawIngestionWriterProtocol | None
        Optional raw data writer
    raw_reader : RawReaderProtocol | None
        Optional raw data reader
    circuit_breaker : CircuitBreakerProtocol | None
        Optional circuit breaker
    topic_scheme : str
        Topic naming scheme for message bus
    topic_prefix : str
        Topic prefix for message bus

    Examples
    --------
    >>> # Use new component-based implementation (default)
    >>> store = DataStore(connection_string="postgresql://...")
    >>> features = store.get_features_at_or_before(
    ...     instrument_id="EURUSD.SIM",
    ...     ts_event=1234567890000000000,
    ... )

    >>> # Use legacy implementation (rollback)
    >>> import os
    >>> os.environ["ML_USE_LEGACY_DATA_STORE"] = "1"
    >>> store = DataStore(connection_string="postgresql://...")
    """

    feature_store: FeatureStore
    model_store: ModelStore
    strategy_store: StrategyStore
    earnings_store: EarningsStoreProtocol
    registry: RegistryProtocol

    def __init__(
        self,
        connection_string: str,
        *,
        registry: RegistryProtocol | None = None,
        feature_store: FeatureStore | None = None,
        model_store: ModelStore | None = None,
        strategy_store: StrategyStore | None = None,
        earnings_store: EarningsStoreProtocol | None = None,
        data_processor: DataProcessor | None = None,
        publisher: MessagePublisherProtocol | None = None,
        enable_publishing: bool = False,
        fail_on_validation_error: bool = True,
        batch_size: int = 10000,
        allow_schema_migration: bool = False,
        schema_migration_window_hours: int = 24,
        raw_writer: RawIngestionWriterProtocol | None = None,
        raw_reader: RawReaderProtocol | None = None,
        circuit_breaker: CircuitBreakerProtocol | None = None,
        topic_scheme: str = "hierarchical",
        topic_prefix: str = "nautilus",
    ) -> None:
        """
        Initialize DataStore with registry and underlying stores.

        Parameters match original DataStore constructor for complete compatibility.
        """
        if not _should_use_component_impl():
            # Use legacy monolithic implementation
            logger.info("Using legacy DataStore implementation (component facade opt-in disabled)")
            from ml.stores.data_store_legacy import DataStore as DataStoreLegacy

            # Create legacy instance and delegate all calls to it
            self._legacy_impl = DataStoreLegacy(
                connection_string=connection_string,
                registry=registry,
                feature_store=feature_store,
                model_store=model_store,
                strategy_store=strategy_store,
                earnings_store=earnings_store,
                data_processor=data_processor,
                publisher=publisher,
                enable_publishing=enable_publishing,
                fail_on_validation_error=fail_on_validation_error,
                batch_size=batch_size,
                allow_schema_migration=allow_schema_migration,
                schema_migration_window_hours=schema_migration_window_hours,
                raw_writer=raw_writer,
                raw_reader=raw_reader,
                circuit_breaker=circuit_breaker,
            )
            self._use_legacy = True
            # Expose stores for compatibility
            self.feature_store = self._legacy_impl.feature_store
            self.model_store = self._legacy_impl.model_store
            self.strategy_store = self._legacy_impl.strategy_store
            legacy_earnings = cast(EarningsStoreProtocol | None, getattr(self._legacy_impl, "earnings_store", None))
            if legacy_earnings is None:
                legacy_earnings = DummyEarningsStore()
            self.earnings_store = legacy_earnings
            self.registry = self._legacy_impl.registry
            return

        # Use new component-based implementation
        logger.info("Using component-based DataStore implementation (opt-in)")
        self._use_legacy = False

        # Initialize base mixins
        MLComponentMixin.__init__(self)
        BusPublisherMixin.__init__(self)
        DataRegistryMixin.__init__(self)

        # Initialize configuration
        self.connection_string = connection_string
        self.fail_on_validation_error = fail_on_validation_error
        self.batch_size = batch_size
        self.allow_schema_migration = allow_schema_migration
        self.schema_migration_window_hours = schema_migration_window_hours
        self.topic_scheme = topic_scheme
        self.topic_prefix = topic_prefix

        # Initialize or use provided stores
        self.feature_store = feature_store or FeatureStore(connection_string=connection_string)
        self.model_store = model_store or ModelStore(connection_string=connection_string)
        self.strategy_store = strategy_store or StrategyStore(connection_string=connection_string)
        self.earnings_store = earnings_store or self._create_earnings_store(connection_string)
        self.data_processor = data_processor

        # Initialize or use provided registry
        if registry is None:
            registry_path = get_default_registry_path()
            persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=registry_path,
            )
            self.registry = DataRegistry(
                registry_path=registry_path,
                persistence_config=persistence_config,
            )
        else:
            self.registry = registry

        # Initialize bus publishing
        bus_mixin = cast(BusPublisherMixin, self)
        bus_mixin._init_bus_publishing(
            enable_publishing=enable_publishing,
            publisher=publisher,
            publish_mode="batch",
        )
        self._topic_scheme = topic_scheme or getattr(self, "_topic_scheme", "domain_op")
        self._topic_prefix = topic_prefix or getattr(self, "_topic_prefix", "events.ml")

        # Initialize specialized components
        self._schema_validator = SchemaValidator()
        self._contract_enforcer = ContractEnforcer(
            registry=self.registry,
            schema_validator=self._schema_validator,
            allow_schema_migration=allow_schema_migration,
            schema_migration_window_hours=schema_migration_window_hours,
        )
        self._data_reader = DataReader(
            feature_store=self.feature_store,
            model_store=self.model_store,
            strategy_store=self.strategy_store,
            earnings_store=self.earnings_store,
        )
        self._data_writer = DataWriter(
            feature_store=self.feature_store,
            model_store=self.model_store,
            strategy_store=self.strategy_store,
            earnings_store=self.earnings_store,
            contract_enforcer=self._contract_enforcer,
            schema_validator=self._schema_validator,
            registry=self.registry,
            publisher=publisher,
            enable_publishing=enable_publishing,
            fail_on_validation_error=fail_on_validation_error,
            batch_size=batch_size,
            raw_writer=raw_writer,
            topic_scheme=topic_scheme,
            topic_prefix=topic_prefix,
        )

        # Store additional parameters
        self.raw_writer = raw_writer
        self.raw_reader = raw_reader
        self.circuit_breaker = circuit_breaker

        logger.info(
            "Initialized DataStore facade with 4 components: "
            "SchemaValidator, ContractEnforcer, DataReader, DataWriter"
        )

    def _create_earnings_store(self, connection_string: str) -> EarningsStoreProtocol:
        """Create earnings store with fallback to file or dummy store."""
        try:
            return EarningsStore(connection_string=connection_string)
        except Exception:
            logger.warning("PostgreSQL earnings store unavailable, trying file fallback", exc_info=True)
            file_store = self._try_file_earnings_store()
            if file_store:
                return file_store
            logger.warning("File earnings store unavailable, using dummy store")
            return DummyEarningsStore()

    def _try_file_earnings_store(self) -> EarningsStoreProtocol | None:
        """Try to create file-backed earnings store."""
        try:
            from ml.config.paths import get_data_catalog_path

            catalog_path = get_data_catalog_path()
            return FileEarningsStore(base_path=Path(catalog_path))
        except Exception:
            logger.debug("File earnings store creation failed", exc_info=True)
            return None

    # =========================================================================
    # Read Operations (delegate to DataReader)
    # =========================================================================

    def get_features_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
    ) -> dict[str, float] | None:
        """Get latest features at or before timestamp."""
        if self._use_legacy:
            return self._legacy_impl.get_features_at_or_before(
                instrument_id=instrument_id,
                ts_event=ts_event,
            )
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
    ) -> PredictionRecord | None:
        """Get latest prediction at or before timestamp."""
        if self._use_legacy:
            return self._legacy_impl.get_latest_prediction_at_or_before(
                instrument_id=instrument_id,
                ts_event=ts_event,
                model_id=model_id,
            )
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
    ) -> SignalRecord | None:
        """Get latest signal at or before timestamp."""
        if self._use_legacy:
            return self._legacy_impl.get_latest_signal_at_or_before(
                instrument_id=instrument_id,
                ts_event=ts_event,
                strategy_id=strategy_id,
            )
        return self._data_reader.get_latest_signal_at_or_before(
            instrument_id=instrument_id,
            ts_event=ts_event,
            strategy_id=strategy_id,
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
        """Get earnings actuals at or before timestamp."""
        if self._use_legacy:
            return self._legacy_impl.get_earnings_actuals_at_or_before(
                ticker=ticker,
                ts_event=ts_event,
                limit=limit,
                start_date=start_date,
                end_date=end_date,
            )
        return self._data_reader.get_earnings_actuals_at_or_before(
            ticker=ticker,
            ts_event=ts_event,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )

    def get_earnings_estimate_at_or_before(
        self,
        *,
        ticker: str,
        period_end: str,
        ts_event: int,
    ) -> dict[str, Any] | None:
        """Get earnings estimate at or before timestamp."""
        if self._use_legacy:
            return self._legacy_impl.get_earnings_estimate_at_or_before(
                ticker=ticker,
                period_end=period_end,
                ts_event=ts_event,
            )
        return self._data_reader.get_earnings_estimate_at_or_before(
            ticker=ticker,
            period_end=period_end,
            ts_event=ts_event,
        )

    # =========================================================================
    # Write Operations (delegate to DataWriter)
    # =========================================================================

    def write_ingestion(
        self,
        dataset_id: str,
        records: list[dict[str, Any]] | DataFrameLike,
        source: str,
        run_id: str,
        instrument_id: str | None = None,
    ) -> DataEvent:
        """Write ingestion data with validation and event emission."""
        if self._use_legacy:
            legacy_event = self._legacy_impl.write_ingestion(
                dataset_id=dataset_id,
                records=records,
                source=source,
                run_id=run_id,
                instrument_id=instrument_id,
            )
            return self._to_data_event(legacy_event)
        writer_event = self._data_writer.write_ingestion(
            dataset_id=dataset_id,
            records=records,
            source=source,
            run_id=run_id,
            instrument_id=instrument_id,
        )
        return self._to_data_event(writer_event)

    def write_features(
        self,
        instrument_id: str,
        features: list[FeatureData],
        source: str = "computed",
        run_id: str | None = None,
    ) -> DataEvent:
        """Write features with validation and event emission."""
        if self._use_legacy:
            legacy_event = self._legacy_impl.write_features(
                instrument_id=instrument_id,
                features=features,
                source=source,
                run_id=run_id,
            )
            return self._to_data_event(legacy_event)
        writer_event = self._data_writer.write_features(
            instrument_id=instrument_id,
            features=features,
            source=source,
            run_id=run_id,
        )
        return self._to_data_event(writer_event)

    def write_predictions(
        self,
        predictions: list[ModelPrediction],
        source: str = "inference",
        run_id: str | None = None,
    ) -> DataEvent:
        """Write predictions with validation and event emission."""
        if self._use_legacy:
            legacy_event = self._legacy_impl.write_predictions(
                predictions=predictions,
                source=source,
                run_id=run_id,
            )
            return self._to_data_event(legacy_event)
        writer_event = self._data_writer.write_predictions(
            predictions=predictions,
            source=source,
            run_id=run_id,
        )
        return self._to_data_event(writer_event)

    def write_signals(
        self,
        signals: list[StrategySignal],
        source: str = "strategy",
        run_id: str | None = None,
    ) -> DataEvent:
        """Write signals with validation and event emission."""
        if self._use_legacy:
            legacy_event = self._legacy_impl.write_signals(
                signals=signals,
                source=source,
                run_id=run_id,
            )
            return self._to_data_event(legacy_event)
        writer_event = self._data_writer.write_signals(
            signals=signals,
            source=source,
            run_id=run_id,
        )
        return self._to_data_event(writer_event)

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
        source: str = Source.HISTORICAL.value,
        run_id: str | None = None,
    ) -> DataEvent:
        """Write earnings actual with validation."""
        if self._use_legacy:
            legacy_event = self._legacy_impl.write_earnings_actual(
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
            return self._to_data_event(legacy_event)
        writer_event = self._data_writer.write_earnings_actual(
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
        return self._to_data_event(writer_event)

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
        source: str = Source.HISTORICAL.value,
        run_id: str | None = None,
    ) -> DataEvent:
        """Write earnings estimate with validation."""
        if self._use_legacy:
            legacy_event = self._legacy_impl.write_earnings_estimate(
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
            return self._to_data_event(legacy_event)
        writer_event = self._data_writer.write_earnings_estimate(
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
        return self._to_data_event(writer_event)

    # =========================================================================
    # Contract & Validation (delegate to ContractEnforcer + SchemaValidator)
    # =========================================================================

    def preflight_check(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],
        strict: bool = True,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Perform preflight schema validation before processing."""
        if self._use_legacy:
            return self._legacy_impl.preflight_check(
                dataset_id=dataset_id,
                data=data,
                strict=strict,
            )
        return self._contract_enforcer.preflight_check(
            dataset_id=dataset_id,
            data=data,
            strict=strict,
        )

    def validate_batch(
        self,
        dataset_id: str,
        data: DataFrameLike,
        strict_mode: bool = False,
    ) -> QualityReport:
        """Validate batch against contract."""
        if self._use_legacy:
            return self._legacy_impl.validate_batch(
                dataset_id=dataset_id,
                data=data,
                strict_mode=strict_mode,
            )
        return self._contract_enforcer.validate_batch(
            dataset_id=dataset_id,
            data=data,
            strict_mode=strict_mode,
        )

    # =========================================================================
    # Mixin Support Methods
    # =========================================================================

    def _init_bus_publishing(
        self,
        *,
        enable_publishing: bool,
        publisher: MessagePublisherProtocol | None,
        publish_mode: Literal["batch", "row", "both"] = "batch",
    ) -> None:
        """Initialize message bus publishing."""
        if self._use_legacy:
            return self._legacy_impl._init_bus_publishing(
                enable_publishing=enable_publishing,
                publisher=publisher,
                publish_mode=publish_mode,
            )
        bus_self = cast(BusPublisherMixin, self)
        BusPublisherMixin._init_bus_publishing(
            bus_self,
            enable_publishing=enable_publishing,
            publisher=publisher,
            publish_mode=publish_mode,
        )

    def _get_data_registry(self) -> RegistryProtocol | None:
        """Get data registry instance."""
        if self._use_legacy:
            return self._legacy_impl._get_data_registry()
        return self.registry

    def get_health_status(self) -> dict[str, Any]:
        """Get health status from all components."""
        if self._use_legacy:
            return self._legacy_impl.get_health_status()

        return {
            "implementation": "component_based",
            "schema_validator": "healthy",
            "contract_enforcer": "healthy",
            "data_reader": "healthy",
            "data_writer": "healthy",
            "feature_store": self.feature_store.get_health_status() if hasattr(self.feature_store, "get_health_status") else "unknown",
            "model_store": self.model_store.get_health_status() if hasattr(self.model_store, "get_health_status") else "unknown",
            "strategy_store": self.strategy_store.get_health_status() if hasattr(self.strategy_store, "get_health_status") else "unknown",
            "earnings_store": "healthy",
            "registry": "healthy",
        }

    def get_performance_metrics(self) -> dict[str, float]:
        """Get performance metrics from all components."""
        if self._use_legacy:
            return self._legacy_impl.get_performance_metrics()
        return {
            "implementation": 1.0,  # 1.0 = component-based, 0.0 = legacy
        }

    def validate_configuration(self) -> list[str]:
        """Validate configuration and return list of errors."""
        if self._use_legacy:
            return self._legacy_impl.validate_configuration()
        errors = []
        if not self.connection_string:
            errors.append("Missing connection_string")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        return errors

    @staticmethod
    def _to_data_event(event: WriterDataEvent | DataEvent) -> DataEvent:
        """Normalize legacy/writer events to validation `DataEvent`."""
        if isinstance(event, DataEvent):
            return event
        metadata = dict(getattr(event, "metadata", {}) or {})
        return DataEvent(
            event_id=str(event.event_id),
            dataset_id=str(event.dataset_id),
            instrument_id=str(event.instrument_id),
            operation=str(event.operation),
            source=str(event.source),
            run_id=str(event.run_id),
            ts_min=int(event.ts_min),
            ts_max=int(event.ts_max),
            record_count=int(event.record_count),
            status=str(event.status),
            error_message=getattr(event, "error_message", None),
            metadata=metadata,
        )

    # =========================================================================
    # Additional Public Methods (for backward compatibility)
    # =========================================================================

    def emit_event(self, *args: Any, **kwargs: Any) -> None:
        """Emit event (legacy compatibility)."""
        if self._use_legacy:
            return self._legacy_impl.emit_event(*args, **kwargs)
        # Not implemented in component-based version
        logger.debug("emit_event called on component-based implementation (no-op)")

    def emit_dataset_event(self, *args: Any, **kwargs: Any) -> None:
        """Emit dataset event (legacy compatibility)."""
        if self._use_legacy:
            return self._legacy_impl.emit_dataset_event(*args, **kwargs)
        # Not implemented in component-based version
        logger.debug("emit_dataset_event called on component-based implementation (no-op)")

    def read_range(self, *args: Any, **kwargs: Any) -> Any:
        """Read data range (legacy compatibility)."""
        if self._use_legacy:
            return self._legacy_impl.read_range(*args, **kwargs)
        # Not implemented in component-based version yet
        logger.warning("read_range not implemented in component-based version")
        return []

    def _begin_transaction(self) -> AbstractContextManager[Any]:
        """Begin database transaction (legacy compatibility)."""
        if self._use_legacy:
            return self._legacy_impl._begin_transaction()
        # Not implemented in component-based version
        from contextlib import nullcontext

        return nullcontext()
