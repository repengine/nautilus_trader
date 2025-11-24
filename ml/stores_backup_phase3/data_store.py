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
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from ml.common.correlation import make_correlation_id
from ml.common.event_emitter import emit_dataset_event
from ml.common.events_util import build_bus_payload
from ml.common.events_util import to_source_enum
from ml.common.events_util import to_stage_enum
from ml.common.message_bus import BusPublisherMixin
from ml.common.message_bus import MessagePublisherProtocol
from ml.common.message_topics import build_topic_for_stage
from ml.common.metrics import quality_score_histogram
from ml.common.metrics import schema_mismatch_counter
from ml.common.metrics import validation_duration_histogram
from ml.common.metrics import validation_violations_counter
from ml.common.metrics import write_rejection_counter
from ml.common.protocols import MLComponentMixin
from ml.common.timestamps import sanitize_timestamp_ns
from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID as _EARNINGS_ACTUALS_DATASET_ID
from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID as _EARNINGS_ESTIMATES_DATASET_ID
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.core.db_engine import EngineManager
from ml.ml_types import DataFrameLike
from ml.registry.data_registry import DataRegistry
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.utils import compute_dataset_schema_hash
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
from ml.stores.feature_dataset_store import FeatureDatasetStore
from ml.stores.feature_store import FeatureStore
from ml.stores.mixins import DataRegistryMixin
from ml.stores.model_store import ModelStore
from ml.stores.protocols import EarningsStoreProtocol
from ml.stores.protocols import FeatureStoreProtocol
from ml.stores.protocols import ModelStoreProtocol
from ml.stores.protocols import PredictionRecord
from ml.stores.protocols import SignalRecord
from ml.stores.protocols import StrategyStoreProtocol
from ml.stores.raw_protocols import RawIngestionWriterProtocol
from ml.stores.raw_protocols import RawReaderProtocol
from ml.stores.schema_validator import SchemaValidator
from ml.stores.strategy_store import StrategyStore
from ml.stores.validation_types import DataEvent
from ml.stores.validation_types import QualityReport


if TYPE_CHECKING:

    from ml.registry.protocols import RegistryProtocol
    from ml.stores.data_store_legacy import DataStore as DataStoreLegacy
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
    1. ML_USE_LEGACY_DATA_STORE takes precedence for backward compatibility:
       - "1" => force legacy implementation
       - "0" => force component implementation
    2. ML_USE_COMPONENT_DATA_STORE explicit opt-in/out remains supported.
    3. Default remains legacy implementation.
    """
    legacy_flag = os.getenv("ML_USE_LEGACY_DATA_STORE")
    if legacy_flag is not None:
        return legacy_flag.strip() == "0"

    component_flag = os.getenv("ML_USE_COMPONENT_DATA_STORE")
    if component_flag is not None:
        return component_flag.strip() == "1"

    return False


USE_COMPONENT_DATA_STORE = _should_use_component_impl()
USE_LEGACY_DATA_STORE = not USE_COMPONENT_DATA_STORE


logger = logging.getLogger(__name__)


EARNINGS_ACTUALS_DATASET_ID = _EARNINGS_ACTUALS_DATASET_ID
EARNINGS_ESTIMATES_DATASET_ID = _EARNINGS_ESTIMATES_DATASET_ID

__all__ = [
    "EARNINGS_ACTUALS_DATASET_ID",
    "EARNINGS_ESTIMATES_DATASET_ID",
    "DataEvent",
    "DataStore",
    "EngineManager",
    "QualityReport",
    "quality_score_histogram",
    "schema_mismatch_counter",
    "time",
    "validation_duration_histogram",
    "validation_violations_counter",
    "write_rejection_counter",
]


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
        feature_store : FeatureStoreProtocol | None
            Feature store instance (created if None)
        model_store : ModelStoreProtocol | None
            Model store instance (created if None)
        strategy_store : StrategyStoreProtocol | None
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

    feature_store: FeatureStoreProtocol
    model_store: ModelStoreProtocol
    strategy_store: StrategyStoreProtocol
    earnings_store: EarningsStoreProtocol
    registry: RegistryProtocol

    def __init__(
        self,
        connection_string: str,
        *,
        registry: RegistryProtocol | None = None,
        feature_store: FeatureStoreProtocol | None = None,
        model_store: ModelStoreProtocol | None = None,
        strategy_store: StrategyStoreProtocol | None = None,
        earnings_store: EarningsStoreProtocol | None = None,
        feature_dataset_store: FeatureDatasetStore | None = None,
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
                feature_store=cast(FeatureStore | None, feature_store),
                model_store=cast(ModelStore | None, model_store),
                strategy_store=cast(StrategyStore | None, strategy_store),
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
            legacy_earnings = cast(
                EarningsStoreProtocol | None,
                getattr(self._legacy_impl, "_earnings_store", None),
            )
            if legacy_earnings is None:
                legacy_earnings = DummyEarningsStore()
            self.earnings_store = legacy_earnings
            self._earnings_store = self.earnings_store
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
        if feature_store is not None:
            feature_store_impl: FeatureStoreProtocol = feature_store
        else:
            feature_store_impl = cast(
                FeatureStoreProtocol,
                FeatureStore(connection_string=connection_string),
            )
        self.feature_store = feature_store_impl

        if model_store is not None:
            model_store_impl: ModelStoreProtocol = model_store
        else:
            model_store_impl = cast(
                ModelStoreProtocol,
                ModelStore(connection_string=connection_string),
            )
        self.model_store = model_store_impl

        if strategy_store is not None:
            strategy_store_impl: StrategyStoreProtocol = strategy_store
        else:
            strategy_store_impl = cast(
                StrategyStoreProtocol,
                StrategyStore(connection_string=connection_string),
            )
        self.strategy_store = strategy_store_impl
        self.earnings_store = earnings_store or self._create_earnings_store(connection_string)
        self._earnings_store = self.earnings_store
        self.feature_dataset_store: FeatureDatasetStore | None
        if feature_dataset_store is not None:
            self.feature_dataset_store = feature_dataset_store
        else:
            try:
                self.feature_dataset_store = FeatureDatasetStore(connection_string=connection_string)
            except Exception:
                logger.warning("FeatureDatasetStore unavailable; macro/events/micro/L2 writes disabled", exc_info=True)
                self.feature_dataset_store = None
        self._feature_dataset_store = self.feature_dataset_store
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
        # Component facade keeps publishing opt-in, but strict validation mode
        # (fail_on_validation_error=True) retains legacy behaviour when a publisher
        # is provided.
        effective_enable_publishing = bool(
            enable_publishing or (publisher is not None and fail_on_validation_error),
        )
        bus_mixin = cast(BusPublisherMixin, self)
        bus_mixin._init_bus_publishing(
            enable_publishing=effective_enable_publishing,
            publisher=publisher,
            publish_mode="batch",
        )
        self._enable_publishing = effective_enable_publishing  # align with legacy attribute expectations
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
             feature_dataset_store=self.feature_dataset_store,
            contract_enforcer=self._contract_enforcer,
            schema_validator=self._schema_validator,
            registry=self.registry,
            publisher=publisher,
            enable_publishing=effective_enable_publishing,
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

        # Share caches with ContractEnforcer for compatibility helpers
        self._manifest_cache = cast(dict[str, DatasetManifest], getattr(self._contract_enforcer, "_manifest_cache", {}))
        self._contract_cache = cast(dict[str, DataContract], getattr(self._contract_enforcer, "_contract_cache", {}))
        self._schema_migration_state = cast(dict[str, dict[str, Any]], getattr(self._contract_enforcer, "_schema_migration_state", {}))

        logger.info(
            "Initialized DataStore facade with 4 components: "
            "SchemaValidator, ContractEnforcer, DataReader, DataWriter"
        )

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the legacy implementation when enabled.

        Tests and legacy code paths patch private helpers directly on the legacy
        class (e.g., ``_start_migration_window``). To maintain backward
        compatibility while opting into the component facade, we forward missing
        attributes when the legacy implementation is active.
        """
        try:
            use_legacy = object.__getattribute__(self, "_use_legacy")
        except AttributeError:
            use_legacy = False
        if use_legacy:
            try:
                legacy_impl = object.__getattribute__(self, "_legacy_impl")
            except AttributeError as exc:  # pragma: no cover - defensive
                raise AttributeError(
                    f"Legacy implementation not initialized for {type(self).__name__}",
                ) from exc
            return getattr(legacy_impl, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Propagate patched attributes to the legacy implementation when active."""
        object.__setattr__(self, name, value)
        if name.startswith("_"):
            return
        if name == "preflight_check":
            try:
                contract_enforcer = object.__getattribute__(self, "_contract_enforcer")
                setattr(contract_enforcer, "preflight_check", value)
            except AttributeError:
                pass
        try:
            use_legacy = object.__getattribute__(self, "_use_legacy")
        except AttributeError:
            use_legacy = False
        if not use_legacy:
            return
        try:
            legacy_impl = object.__getattribute__(self, "_legacy_impl")
        except AttributeError:
            return
        try:
            setattr(legacy_impl, name, value)
        except Exception:
            # Legacy implementation may not expose attribute; ignore silently
            return

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
            self._record_fallback_metric(level="dummy")
            return DummyEarningsStore()

    def _try_file_earnings_store(self) -> EarningsStoreProtocol | None:
        """Try to create file-backed earnings store."""
        from ml._imports import HAS_POLARS

        if not HAS_POLARS:
            logger.debug("Skipping file earnings fallback because polars is not available")
            return None

        file_root_str = os.getenv("ML_FILE_STORE_PATH")
        file_root = Path(file_root_str) if file_root_str else Path.home() / ".nautilus" / "ml" / "file_store"
        earnings_path = file_root / "earnings"
        try:
            from importlib import import_module

            file_backed = import_module("ml.stores.file_backed")
            file_store_cls: type[Any] = getattr(file_backed, "FileEarningsStore")
            store = cast(EarningsStoreProtocol, file_store_cls(base_path=earnings_path))
            logger.info("Initialized FileEarningsStore fallback at %s", earnings_path)
            self._record_fallback_metric(level="file")
            return store
        except Exception:
            logger.debug("File earnings store creation failed", exc_info=True)
            return None

    @staticmethod
    def _record_fallback_metric(*, level: str) -> None:
        """Increment the fallback activation counter when available."""
        try:
            from ml.common.metrics_bootstrap import get_counter

            counter = get_counter(
                "ml_fallback_activations_total",
                "Fallback activations",
            )
            counter.labels(component="data_store", level=level).inc()
        except Exception:
            logger.debug("Failed to record fallback activation metric", exc_info=True)

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
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
            legacy_event = self._legacy_impl.write_ingestion(
                dataset_id=dataset_id,
                records=records,
                source=source,
                run_id=run_id,
                instrument_id=instrument_id,
            )
            return self._to_data_event(legacy_event)
        writer = getattr(self, "_data_writer", None)
        if writer is None:
            raise AttributeError("DataWriter not initialized for component DataStore")

        try:
            writer_event = writer.write_ingestion(
                dataset_id=dataset_id,
                records=records,
                source=source,
                run_id=run_id,
                instrument_id=instrument_id,
            )
        except Exception as exc:
            inferred_instrument, ts_min, ts_max, count = self._infer_ingestion_failure_context(
                records=records,
                instrument_id=instrument_id,
            )
            stage_value = self._resolve_stage_for_dataset(writer, dataset_id, default=Stage.DATA_INGESTED)

            self._emit_failed_event(
                dataset_id=dataset_id,
                instrument_id=inferred_instrument,
                stage=stage_value,
                source=source,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                error=str(exc),
            )
            raise

        event = self._to_data_event(writer_event)
        if str(event.status).lower() == EventStatus.PARTIAL.value:
            stage_value = self._resolve_stage_for_dataset(writer, dataset_id, default=Stage.DATA_INGESTED)

            reason = str(event.metadata.get("reason", "partial"))
            self._emit_partial_event(
                dataset_id=dataset_id,
                instrument_id=str(event.instrument_id),
                stage=stage_value,
                source=str(event.source),
                run_id=str(event.run_id),
                ts_min=int(event.ts_min),
                ts_max=int(event.ts_max),
                count=int(event.record_count),
                reason=reason,
            )

        monitor_message = event.metadata.get("monitor_only_details")
        if monitor_message:
            logger.info(
                "Data validation issues for %s (monitor-only): %s",
                dataset_id,
                monitor_message,
            )
        return event

    def _infer_ingestion_failure_context(
        self,
        *,
        records: list[dict[str, Any]] | DataFrameLike,
        instrument_id: str | None,
    ) -> tuple[str, int, int, int]:
        """Best-effort metadata extraction for failed ingestion writes."""
        inferred_instrument = instrument_id or "UNKNOWN"
        ts_values: list[int] = []
        count = 0

        if isinstance(records, list):
            count = len(records)
            if records:
                first = records[0]
                if instrument_id is None and isinstance(first, dict) and "instrument_id" in first:
                    inferred_instrument = str(first["instrument_id"])
            for row in records:
                if isinstance(row, dict) and "ts_event" in row:
                    try:
                        ts_values.append(int(row["ts_event"]))
                    except (TypeError, ValueError):
                        continue
        else:
            # Handle DataFrame-like objects in a defensive manner
            if instrument_id is None:
                try:
                    col = records["instrument_id"]
                    if col is not None:
                        inferred_instrument = str(col[0])
                except Exception as exc:
                    logger.debug(
                        "Failed to infer instrument identifier from ingestion records",
                        extra={
                            "component": "DataStore",
                            "field": "instrument_id",
                            "error_type": exc.__class__.__name__,
                        },
                        exc_info=True,
                    )
            try:
                ts_series = records["ts_event"]
                if ts_series is not None:
                    try:
                        count = len(ts_series)
                    except Exception as exc:
                        logger.debug(
                            "Unable to determine record count from ts_event series",
                            extra={
                                "component": "DataStore",
                                "error_type": exc.__class__.__name__,
                            },
                            exc_info=True,
                        )
                        count = max(count, 1)
                    first_ts: int | None = None
                    try:
                        first_ts = int(ts_series[0])
                    except Exception as exc:
                        logger.debug(
                            "Unable to extract first ts_event value from series",
                            extra={
                                "component": "DataStore",
                                "error_type": exc.__class__.__name__,
                            },
                            exc_info=True,
                        )
                    if first_ts is not None:
                        ts_values.append(first_ts)
                    last_ts: int | None = None
                    try:
                        last_ts = int(ts_series[-1])
                    except Exception as exc:
                        logger.debug(
                            "Unable to extract last ts_event value from series",
                            extra={
                                "component": "DataStore",
                                "error_type": exc.__class__.__name__,
                            },
                            exc_info=True,
                        )
                    if last_ts is not None:
                        ts_values.append(last_ts)
            except Exception as exc:
                logger.debug(
                    "Failed to read ts_event series from ingestion records",
                    extra={
                        "component": "DataStore",
                        "error_type": exc.__class__.__name__,
                    },
                    exc_info=True,
                )

        if not ts_values:
            fallback = time.time_ns()
            ts_values = [fallback]
        if count <= 0:
            count = len(ts_values)

        return inferred_instrument or "UNKNOWN", min(ts_values), max(ts_values), count

    def _resolve_stage_for_dataset(
        self,
        writer: Any,
        dataset_id: str,
        *,
        default: Stage | str,
    ) -> Stage | str:
        """Best-effort stage resolution for dataset type."""
        try:
            manifest = self._get_manifest(dataset_id)
            stage_helper = getattr(writer, "_get_stage_for_dataset_type", None)
            if callable(stage_helper):
                result = stage_helper(manifest.dataset_type)
                return cast(Stage | str, result)
        except Exception:
            return default
        return default

    def write_features(
        self,
        instrument_id: str,
        features: list[FeatureData],
        source: str = "computed",
        run_id: str | None = None,
    ) -> DataEvent:
        """Write features with validation and event emission."""
        if getattr(self, "_use_legacy", False):
            legacy_event = self._legacy_impl.write_features(
                instrument_id=instrument_id,
                features=features,
                source=source,
                run_id=run_id,
            )
            return self._to_data_event(legacy_event)
        writer = getattr(self, "_data_writer", None)
        if writer is None:
            return self._write_features_fallback(
                instrument_id=instrument_id,
                features=features,
                source=source,
                run_id=run_id,
            )
        writer_event = writer.write_features(
            instrument_id=instrument_id,
            features=features,
            source=source,
            run_id=run_id,
        )
        return self._to_data_event(writer_event)

    def _write_features_fallback(
        self,
        *,
        instrument_id: str,
        features: list[FeatureData],
        source: str,
        run_id: str | None,
    ) -> DataEvent:
        """
        Compatibility path when the component DataWriter is unavailable.

        This occurs in certain legacy unit tests which construct ``DataStore`` via
        ``object.__new__`` and inject only minimal dependencies. We preserve their
        expectations by executing a reduced write flow: persist via the injected
        FeatureStore, emit best-effort registry events, and return a ``DataEvent``.
        """
        feature_store = getattr(self, "feature_store", None)
        if feature_store is None:
            raise AttributeError("feature_store is required for write_features fallback")

        from ml.stores.protocols import FeatureStoreProtocol as _FeatureStoreProtocol

        protocol_store = cast(_FeatureStoreProtocol, feature_store)

        run_id_value = run_id or f"features_{time.time_ns()}"
        dataset_id = "features"
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        ts_min = min(f.ts_event for f in features)
        ts_max = max(f.ts_event for f in features)
        ts_min_s = _sanitize(int(ts_min), context="data_store.write_features_fallback:ts_min")
        ts_max_s = _sanitize(int(ts_max), context="data_store.write_features_fallback:ts_max")

        try:
            for feature in features:
                protocol_store.write_features(
                    feature_set_id=feature.feature_set_id,
                    instrument_id=feature.instrument_id,
                    features=feature.values,
                    ts_event=feature.ts_event,
                    ts_init=feature.ts_init,
                    publish_bus=False,
                )
        except Exception as exc:
            registry = getattr(self, "registry", None)
            if registry is not None:
                try:
                    self._emit_failed_event(
                        dataset_id=dataset_id,
                        instrument_id=instrument_id,
                        stage=Stage.FEATURE_COMPUTED,
                        source=source,
                        run_id=run_id_value,
                        ts_min=ts_min_s,
                        ts_max=ts_max_s,
                        count=0,
                        error=str(exc),
                    )
                except Exception:
                    logger.debug("Failed event emission skipped in fallback", exc_info=True)
            raise RuntimeError(f"Feature write failed: {exc}") from exc

        event = DataEvent(
            event_id=f"{run_id_value}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_features",
            source=source,
            run_id=run_id_value,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            record_count=len(features),
            status=EventStatus.SUCCESS.value,
        )

        registry = getattr(self, "registry", None)
        if registry is not None:
            try:
                emit_dataset_event(
                    registry,
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    stage=Stage.FEATURE_COMPUTED,
                    source=source,
                    run_id=run_id_value,
                    ts_min=ts_min_s,
                    ts_max=ts_max_s,
                    count=len(features),
                    status=EventStatus.SUCCESS,
                    dataset_type=dataset_id,
                    component=self.__class__.__name__,
                )
            except Exception:
                logger.warning(
                    "Failed to emit dataset event in write_features fallback",
                    exc_info=True,
                )

        return event

    def write_predictions(
        self,
        predictions: list[ModelPrediction],
        source: str = "inference",
        run_id: str | None = None,
    ) -> DataEvent:
        """Write predictions with validation and event emission."""
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
            return self._legacy_impl.validate_batch(
                dataset_id=dataset_id,
                data=data,
                strict_mode=strict_mode,
            )
        normalized = self._to_dataframe(data)
        return self._contract_enforcer.validate_batch(
            dataset_id=dataset_id,
            data=normalized,
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
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
            return self._legacy_impl._get_data_registry()
        return self.registry

    def get_health_status(self) -> dict[str, Any]:
        """Get health status from all components."""
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
            return self._legacy_impl.get_performance_metrics()
        return {
            "implementation": 1.0,  # 1.0 = component-based, 0.0 = legacy
        }

    def validate_configuration(self) -> list[str]:
        """Validate configuration and return list of errors."""
        if getattr(self, "_use_legacy", False):
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
        if getattr(self, "_use_legacy", False):
            logger.debug("Delegating emit_event to legacy DataStore implementation")
            return self._legacy_impl.emit_event(*args, **kwargs)

        if args:
            raise TypeError("emit_event accepts keyword arguments only in component mode")

        try:
            dataset_id = str(kwargs["dataset_id"])
            instrument_id = str(kwargs["instrument_id"])
            stage = kwargs.get("stage", Stage.DATA_INGESTED)
            source = kwargs.get("source", Source.LIVE)
            run_id = str(kwargs.get("run_id", "unknown"))
            ts_min = int(kwargs.get("ts_min", 0))
            ts_max = int(kwargs.get("ts_max", ts_min))
            count = int(kwargs.get("count", 0))
            status = kwargs.get("status", EventStatus.SUCCESS)
            error = kwargs.get("error")
            metadata = kwargs.get("metadata")
        except KeyError as exc:
            raise TypeError(f"Missing required argument for emit_event: {exc}") from exc

        stage_enum = self._coerce_stage(stage)
        try:
            emit_dataset_event(
                self.registry,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage_enum,
                source=to_source_enum(source),
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=status,
                error=error,
                metadata=metadata,
                dataset_type=dataset_id,
                component=self.__class__.__name__,
            )
        except Exception:
            logger.warning("emit_event registry emission failed", exc_info=True)

        publisher = getattr(self, "publisher", None)
        if not getattr(self, "_enable_publishing", False) or publisher is None:
            return

        try:
            try:
                from ml.config.bus import MessageBusConfig as _MBC

                bus_cfg = _MBC.from_env()
                topic_scheme = str(bus_cfg.scheme)
                topic_prefix = str(bus_cfg.topic_prefix)
            except Exception:
                topic_scheme = self._topic_scheme
                topic_prefix = self._topic_prefix

            topic = build_topic_for_stage(
                stage_enum,
                instrument_id,
                scheme=topic_scheme,
                prefix=topic_prefix,
            )
            payload = build_bus_payload(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage_enum,
                source=to_source_enum(source),
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=status,
                metadata=metadata,
            )
            publisher.publish(topic, payload)
        except Exception:
            logger.warning("emit_event bus publish skipped", exc_info=True)

    def emit_dataset_event(self, *args: Any, **kwargs: Any) -> None:
        """Emit dataset event (legacy compatibility)."""
        if getattr(self, "_use_legacy", False):
            logger.debug("Delegating emit_dataset_event to legacy DataStore implementation")
            return self._legacy_impl.emit_dataset_event(*args, **kwargs)
        if args:
            raise TypeError("emit_dataset_event accepts keyword arguments only")

        try:
            dataset_id = str(kwargs["dataset_id"])
            instrument_id = str(kwargs["instrument_id"])
            stage = kwargs.get("stage", Stage.DATA_INGESTED)
            source = kwargs.get("source", Source.LIVE)
            run_id = str(kwargs["run_id"])
            ts_min = int(kwargs["ts_min"])
            ts_max = int(kwargs["ts_max"])
            count = int(kwargs["count"])
            status = kwargs.get("status", EventStatus.SUCCESS)
        except KeyError as exc:
            raise TypeError(f"Missing required argument for emit_dataset_event: {exc}") from exc

        stage_enum = self._coerce_stage(stage)
        source_enum = to_source_enum(source)
        status_enum = status if isinstance(status, EventStatus) else EventStatus(str(status))

        correlation_id = make_correlation_id(
            run_id=run_id,
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
        )

        publisher = cast(MessagePublisherProtocol | None, getattr(self, "publisher", None))
        if not getattr(self, "_enable_publishing", False) or publisher is None:
            logger.debug(
                "emit_dataset_event publishing skipped enable=%s publisher_present=%s",
                getattr(self, "_enable_publishing", False),
                publisher is not None,
            )
            return

        try:
            try:
                from ml.config.bus import MessageBusConfig as _MBC

                bus_cfg = _MBC.from_env()
                topic_scheme = str(bus_cfg.scheme)
                topic_prefix = str(bus_cfg.topic_prefix)
            except Exception:
                topic_scheme = self._topic_scheme
                topic_prefix = self._topic_prefix

            topic = build_topic_for_stage(
                stage_enum,
                instrument_id,
                scheme=topic_scheme,
                prefix=topic_prefix,
            )
            payload = build_bus_payload(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage_enum,
                source=source_enum,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=status_enum,
                metadata={"correlation_id": correlation_id},
            )
            publisher.publish(topic, payload)
        except Exception:
            logger.warning("emit_dataset_event bus publish skipped", exc_info=True)

    def _get_manifest(self, dataset_id: str) -> DatasetManifest:
        """Retrieve dataset manifest with caching and migration detection."""
        if getattr(self, "_use_legacy", False):
            legacy_impl = cast("DataStoreLegacy", getattr(self, "_legacy_impl"))
            return legacy_impl._get_manifest(dataset_id)

        contract_enforcer_obj = getattr(self, "_contract_enforcer", None)
        if contract_enforcer_obj is not None and hasattr(contract_enforcer_obj, "get_manifest"):
            from ml.stores.contract_enforcer import ContractEnforcerProtocol as _ContractEnforcerProtocol

            contract_enforcer = cast(_ContractEnforcerProtocol, contract_enforcer_obj)
            manifest = contract_enforcer.get_manifest(dataset_id)
            self._manifest_cache[dataset_id] = manifest
            return manifest

        if dataset_id not in self._manifest_cache:
            manifest = self.registry.get_manifest(dataset_id)
            self._manifest_cache[dataset_id] = manifest

            if (
                self.allow_schema_migration
                and dataset_id in self._schema_migration_state
            ):
                previous = self._schema_migration_state[dataset_id].get("version")
                if previous and previous != manifest.version:
                    logger.info(
                        "Schema version change detected for %s: %s -> %s",
                        dataset_id,
                        previous,
                        manifest.version,
                    )
                    self._start_migration_window(dataset_id, manifest)

        return self._manifest_cache[dataset_id]

    def _get_contract(self, dataset_id: str) -> DataContract:
        """Retrieve data contract with caching."""
        if getattr(self, "_use_legacy", False):
            legacy_impl = cast("DataStoreLegacy", getattr(self, "_legacy_impl"))
            return legacy_impl._get_contract(dataset_id)

        contract_enforcer_obj = getattr(self, "_contract_enforcer", None)
        if contract_enforcer_obj is not None and hasattr(contract_enforcer_obj, "get_contract"):
            from ml.stores.contract_enforcer import ContractEnforcerProtocol as _ContractEnforcerProtocol

            contract_enforcer = cast(_ContractEnforcerProtocol, contract_enforcer_obj)
            contract = contract_enforcer.get_contract(dataset_id)
            self._contract_cache[dataset_id] = contract
            return contract

        if dataset_id not in self._contract_cache:
            contract = self.registry.get_contract(dataset_id)
            self._contract_cache[dataset_id] = contract
            logger.debug(
                "Loaded contract for %s: version=%s, mode=%s, rules=%d",
                dataset_id,
                contract.version,
                contract.enforcement_mode,
                len(contract.validation_rules),
            )
        return self._contract_cache[dataset_id]

    def _compute_schema_hash(self, data_frame: DataFrameLike, manifest: DatasetManifest) -> str:
        """Compute a schema hash for the provided data frame."""
        if getattr(self, "_use_legacy", False):
            legacy_impl = cast("DataStoreLegacy", getattr(self, "_legacy_impl"))
            return legacy_impl._compute_schema_hash(data_frame, manifest)

        contract_enforcer_obj = getattr(self, "_contract_enforcer", None)
        helper: Callable[[DataFrameLike, DatasetManifest], str] | None = None
        if contract_enforcer_obj is not None:
            helper = getattr(contract_enforcer_obj, "_compute_schema_hash", None)
        if callable(helper):
            return helper(data_frame, manifest)

        frame_any = cast(Any, data_frame)
        if not hasattr(frame_any, "columns"):
            return manifest.schema_hash

        actual_schema: dict[str, str] = {}
        for column in frame_any.columns:
            if column in manifest.schema:
                actual_schema[column] = manifest.schema[column]
            else:
                dtype = str(frame_any[column].dtype)
                lowered = dtype.lower()
                if "int" in lowered:
                    actual_schema[column] = "int64"
                elif "float" in lowered:
                    actual_schema[column] = "float64"
                elif "bool" in lowered:
                    actual_schema[column] = "bool"
                else:
                    actual_schema[column] = "str"

        for manifest_column, manifest_dtype in manifest.schema.items():
            actual_schema.setdefault(manifest_column, manifest_dtype)

        return compute_dataset_schema_hash(
            schema=actual_schema,
            primary_keys=manifest.primary_keys,
            ts_field=manifest.ts_field,
            seq_field=manifest.seq_field,
            pipeline_signature=manifest.pipeline_signature,
        )

    def _is_in_migration_window(self, dataset_id: str) -> bool:
        """Return True if dataset is within the migration window."""
        if getattr(self, "_use_legacy", False):
            legacy_impl = cast("DataStoreLegacy", getattr(self, "_legacy_impl"))
            return legacy_impl._is_in_migration_window(dataset_id)

        contract_enforcer_obj = getattr(self, "_contract_enforcer", None)
        helper = None
        if contract_enforcer_obj is not None:
            helper = getattr(contract_enforcer_obj, "_is_in_migration_window", None)
        if callable(helper):
            return bool(helper(dataset_id))

        if not self.allow_schema_migration:
            return False
        if dataset_id not in self._schema_migration_state:
            return False

        migration_info = self._schema_migration_state.get(dataset_id, {})
        start_time = migration_info.get("start_time", 0)
        window_ns = int(self.schema_migration_window_hours * 3600 * 1_000_000_000)
        if start_time <= 0:
            return False

        current_ns = sanitize_timestamp_ns(
            int(time.time_ns()),
            context="data_store._is_in_migration_window:now",
        )
        if current_ns - start_time < window_ns:
            return True

        self._schema_migration_state.pop(dataset_id, None)
        logger.info("Schema migration window expired for %s", dataset_id)
        return False

    def _start_migration_window(self, dataset_id: str, manifest: DatasetManifest) -> None:
        """Begin a migration window for the given dataset."""
        if getattr(self, "_use_legacy", False):
            legacy_impl = cast("DataStoreLegacy", getattr(self, "_legacy_impl"))
            legacy_impl._start_migration_window(dataset_id, manifest)
            return

        contract_enforcer_obj = getattr(self, "_contract_enforcer", None)
        helper = None
        if contract_enforcer_obj is not None:
            helper = getattr(contract_enforcer_obj, "_start_migration_window", None)
        if callable(helper):
            helper(dataset_id, manifest)
            return

        self._schema_migration_state[dataset_id] = {
            "start_time": time.time_ns(),
            "version": manifest.version,
            "schema_hash": manifest.schema_hash,
        }
        logger.info(
            "Started schema migration window for %s (version %s, %d hours)",
            dataset_id,
            manifest.version,
            self.schema_migration_window_hours,
        )
        if self.schema_migration_window_hours and self.schema_migration_window_hours > 0:
            try:
                schema_mismatch_counter.labels(
                    dataset=dataset_id,
                    mismatch_type="migration_started",
                ).inc()
            except Exception:
                logger.debug("Schema mismatch counter increment skipped", exc_info=True)

    def _coerce_stage(self, stage: Stage | str) -> Stage:
        """Coerce legacy stage identifiers to canonical Stage enums."""
        try:
            return to_stage_enum(stage)
        except Exception:
            value = str(stage).strip()
            alias = value.lower()
            if alias in {
                "feature_engineering",
                "features_engineered",
                "features_computed",
                "feature_computed",
            }:
                return Stage.FEATURE_COMPUTED
            if alias in {"model_inference", "prediction", "predictions_emitted", "prediction_emitted"}:
                return Stage.PREDICTION_EMITTED
            if alias in {"data_ingested", "ingested", "ingest", "data_ingest"}:
                return Stage.DATA_INGESTED
            if alias in {"catalog_written", "catalog_write", "catalog"}:
                return Stage.CATALOG_WRITTEN
            if alias in {"signal_emitted", "signal_generation", "signal_generated", "emit_signal"}:
                return Stage.SIGNAL_EMITTED
            return Stage.DATA_INGESTED

    def _to_dataframe(self, data: DataFrameLike | list[dict[str, Any]]) -> DataFrameLike:
        """Normalize input data to a DataFrame when possible."""
        if hasattr(data, "columns"):
            return cast(DataFrameLike, data)
        try:
            import pandas as _pd

            return cast(DataFrameLike, _pd.DataFrame(data))
        except Exception as exc:
            self.logger.debug(
                "data_store.to_dataframe_failed dataset_id=%s error=%s",
                getattr(self._config, "dataset_id", "unknown"),
                exc,
                exc_info=True,
            )
            raise TypeError("Unable to coerce data to DataFrame") from exc

    # ------------------------------------------------------------------
    # Legacy compatibility helpers (used by historical unit tests)
    # ------------------------------------------------------------------

    def _df_to_predictions(
        self,
        data_frame: DataFrameLike | list[dict[str, Any]],
    ) -> list[ModelPrediction]:
        """
        Backwards-compatible alias for the original private helper.

        Historical tests import :meth:`_df_to_predictions`; retain the behaviour by
        delegating to the active implementation.
        """
        frame: DataFrameLike
        if isinstance(data_frame, list):
            frame = self._to_dataframe(data_frame)
        else:
            frame = data_frame
        legacy_impl = getattr(self, "_legacy_impl", None)
        if getattr(self, "_use_legacy", False) and hasattr(legacy_impl, "_df_to_predictions"):
            legacy_helper = getattr(legacy_impl, "_df_to_predictions")
            return cast(list[ModelPrediction], legacy_helper(frame))
        return self._data_writer._data_frame_to_predictions(frame)

    def _df_to_signals(
        self,
        data_frame: DataFrameLike | list[dict[str, Any]],
    ) -> list[StrategySignal]:
        """
        Backwards-compatible alias for the original private helper.

        Ensures legacy unit tests importing :meth:`_df_to_signals` continue to work.
        """
        frame: DataFrameLike
        if isinstance(data_frame, list):
            frame = self._to_dataframe(data_frame)
        else:
            frame = data_frame
        legacy_impl = getattr(self, "_legacy_impl", None)
        if getattr(self, "_use_legacy", False) and hasattr(legacy_impl, "_df_to_signals"):
            legacy_helper = getattr(legacy_impl, "_df_to_signals")
            return cast(list[StrategySignal], legacy_helper(frame))
        return self._data_writer._data_frame_to_signals(frame)

    def _emit_partial_event(
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
        reason: str,
    ) -> None:
        """Emit a partial dataset event using centralized helper."""
        try:
            try:
                source_enum = to_source_enum(source)
            except Exception:
                source_enum = Source.LIVE
            emit_dataset_event(
                self.registry,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=self._coerce_stage(stage),
                source=source_enum,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=EventStatus.PARTIAL,
                metadata={"reason": reason},
                dataset_type=dataset_id,
                component=self.__class__.__name__,
            )
        except Exception:
            logger.warning("Failed to emit partial event", exc_info=True)

    def _emit_failed_event(
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
        error: str,
    ) -> None:
        """Emit a failed dataset event using centralized helper."""
        try:
            try:
                source_enum = to_source_enum(source)
            except Exception:
                source_enum = Source.LIVE
            emit_dataset_event(
                self.registry,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=self._coerce_stage(stage),
                source=source_enum,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=EventStatus.FAILED,
                error=error,
                dataset_type=dataset_id,
                component=self.__class__.__name__,
            )
        except Exception:
            logger.warning("Failed to emit failed event", exc_info=True)

    def read_range(
        self,
        dataset_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> Any:
        """
        Read a dataset range while preserving legacy routing semantics.

        When the component-based implementation is active, this method routes
        requests to the appropriate underlying store based on the dataset type.

        Args:
            dataset_id: Registry dataset identifier (e.g., ``predictions_model``).
            instrument_id: Instrument identifier used for lookups.
            start_ns: Inclusive start timestamp in nanoseconds.
            end_ns: Exclusive end timestamp in nanoseconds.

        Returns:
            Any: Store-specific payload (DataFrame, list, tuple, ...).

        Raises:
            ValueError: If the requested time range is invalid.
            NotImplementedError: When the dataset type does not support reads.
        """
        if getattr(self, "_use_legacy", False):
            return self._legacy_impl.read_range(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )

        if start_ns >= end_ns:
            raise ValueError(f"Invalid time range: start={start_ns} >= end={end_ns}")

        manifest = self._get_manifest(dataset_id)
        dataset_type: DatasetType
        raw_type = manifest.dataset_type
        if isinstance(raw_type, DatasetType):
            dataset_type = raw_type
        else:
            try:
                dataset_type = DatasetType(str(raw_type))
            except ValueError as exc:
                msg = f"Unsupported dataset type '{raw_type}' for read_range"
                raise NotImplementedError(msg) from exc

        if dataset_type == DatasetType.FEATURES:
            feature_store_obj = getattr(self, "feature_store", None)
            if feature_store_obj is not None:
                start_dt = datetime.fromtimestamp(start_ns / 1_000_000_000)
                end_dt = datetime.fromtimestamp(end_ns / 1_000_000_000)
                if hasattr(feature_store_obj, "get_training_data"):
                    return feature_store_obj.get_training_data(
                        instrument_id=instrument_id,
                        start=start_dt,
                        end=end_dt,
                    )
                if hasattr(feature_store_obj, "read_range"):
                    return feature_store_obj.read_range(
                        start_ns=start_ns,
                        end_ns=end_ns,
                        instrument_id=instrument_id,
                    )
            return self._data_reader.read_feature_range(
                start_ns=start_ns,
                end_ns=end_ns,
                instrument_id=instrument_id,
            )

        if dataset_type == DatasetType.PREDICTIONS:
            model_id = manifest.metadata.get("model_id") if manifest.metadata else None
            if not model_id:
                model_id = dataset_id.replace("predictions_", "", 1)
            model_store_obj = getattr(self, "model_store", None)
            if getattr(self, "_use_legacy", False) and model_store_obj is not None:
                return model_store_obj.read_predictions(
                    model_id=model_id,
                    instrument_id=instrument_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
            predictions = self._data_reader.read_predictions_range(
                model_id=model_id,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )
            return predictions

        if dataset_type == DatasetType.SIGNALS:
            strategy_id = manifest.metadata.get("strategy_id") if manifest.metadata else None
            if not strategy_id:
                strategy_id = dataset_id.replace("signals_", "", 1)
            strategy_store_obj = getattr(self, "strategy_store", None)
            if getattr(self, "_use_legacy", False) and strategy_store_obj is not None:
                return strategy_store_obj.read_signals(
                    strategy_id=strategy_id,
                    instrument_id=instrument_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
            signals = self._data_reader.read_signals_range(
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )
            return signals

        if self.raw_reader is not None:
            return self.raw_reader.read_range(
                dataset_type=dataset_type,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )

        raise NotImplementedError(f"Read not implemented for dataset type {dataset_type}")

    def _begin_transaction(self) -> AbstractContextManager[Any]:
        """Begin database transaction (legacy compatibility)."""
        if getattr(self, "_use_legacy", False):
            return self._legacy_impl._begin_transaction()
        # Not implemented in component-based version
        from contextlib import nullcontext

        return nullcontext()
