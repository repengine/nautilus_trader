#!/usr/bin/env python3

"""
Data Store facade providing typed read/write operations with contract validation.

This module provides a unified interface for all data operations in the ML pipeline,
integrating with the DataRegistry for manifest/contract retrieval, validating data
against contracts before writing, emitting events, and updating watermarks.

The DataStore acts as a facade over existing FeatureStore, ModelStore, and StrategyStore
while adding contract validation, event emission, and watermark tracking.

"""

from __future__ import annotations

import logging
import os
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from ml._imports import HAS_POLARS
from ml._imports import HAS_PROMETHEUS
from ml.common.correlation import make_correlation_id
from ml.common.event_emitter import emit_dataset_event_and_watermark
from ml.common.events_util import build_bus_payload
from ml.common.events_util import to_source_enum
from ml.common.events_util import to_source_str
from ml.common.events_util import to_stage_enum
from ml.common.events_util import to_status_enum
from ml.common.message_bus import BusPublisherMixin
from ml.common.message_bus import MessagePublisherProtocol
from ml.common.message_topics import build_topic_for_stage
from ml.common.protocols import MLComponentMixin
from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.data.dataset_manifest_defaults import build_auto_dataset_manifest
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.registry.utils import compute_dataset_schema_hash
from ml.registry.utils import get_default_registry_path
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.data_processor import DataProcessor
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
from ml.stores.strategy_store import StrategyStore
from ml.stores.validation_types import DataEvent
from ml.stores.validation_types import QualityReport
from ml.stores.validation_types import ValidationViolation


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

if TYPE_CHECKING:
    import pandas as pd
else:  # pragma: no cover - pandas optional in some environments
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - runtime optional dependency
        pd = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

# ========================================================================
# Prometheus Metrics
# ========================================================================


class _CounterLike(Protocol):
    def labels(self, **kwargs: object) -> _CounterLike: ...

    def inc(self, *args: object, **kwargs: object) -> None: ...


class _HistogramLike(Protocol):
    def labels(self, **kwargs: object) -> _HistogramLike: ...

    def observe(self, *args: object, **kwargs: object) -> None: ...


class _NoOpMetric:
    def labels(self, **_: object) -> _NoOpMetric:
        return self

    def inc(self, *_: object, **__: object) -> None:
        return None

    def observe(self, *_: object, **__: object) -> None:
        return None


# Declare metric variables once; assign real or no-op implementations below
validation_violations_counter: Any = _NoOpMetric()
validation_duration_histogram: Any = _NoOpMetric()
schema_mismatch_counter: Any = _NoOpMetric()
write_rejection_counter: Any = _NoOpMetric()
quality_score_histogram: Any = _NoOpMetric()

try:
    from ml.common.metrics import quality_score_histogram as _qsh
    from ml.common.metrics import schema_mismatch_counter as _smc
    from ml.common.metrics import validation_duration_histogram as _vd
    from ml.common.metrics import validation_violations_counter as _vvc
    from ml.common.metrics import write_rejection_counter as _wrc

    quality_score_histogram = _qsh
    schema_mismatch_counter = _smc
    validation_duration_histogram = _vd
    validation_violations_counter = _vvc
    write_rejection_counter = _wrc
except Exception:
    # Keep no-ops assigned
    logger.debug("Metrics import failed; using no-op counters/histograms", exc_info=True)


# ========================================================================
# DataStore Implementation
# ========================================================================


class DataStore(_MLComponentBase, _BusPublisherBase, _DataRegistryBase):
    """
    Typed read/write facade with contract validation and event emission.

    This class provides a unified interface for all data operations in the ML pipeline,
    ensuring data quality through contract validation, tracking all operations through
    events, and maintaining watermarks for data freshness tracking.

    Thread-safe for concurrent operations.

    Examples
    --------
    >>> # Initialize with registry and connection
    >>> store = DataStore(
    ...     registry=data_registry,
    ...     connection_string="postgresql://postgres@localhost/nautilus",
    ...     feature_store=feature_store,
    ...     model_store=model_store,
    ...     strategy_store=strategy_store
    ... )

    >>> # Write ingestion data with validation
    >>> event = store.write_ingestion(
    ...     dataset_id="bars_eurusd_1m",
    ...     records=bar_data,
    ...     source="historical",
    ...     run_id="run_123"
    ... )

    >>> # Validate data batch
    >>> report = store.validate_batch("bars_eurusd_1m", data_frame)
    >>> if report.quality_score < 0.95:
    ...     logger.warning(f"Data quality below threshold: {report.quality_score}")

    """

    # Declare bus-related attributes for type checkers
    _enable_publishing: bool = False
    _topic_scheme: str = "domain_op"
    _topic_prefix: str = "events.ml"
    _publish_mode: str = "batch"

    def __init__(
        self,
        connection_string: str,
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
    ) -> None:
        """
        Initialize DataStore with registry and underlying stores.

        Parameters
        ----------
        registry : DataRegistry
            Data registry for manifest/contract retrieval
        connection_string : str
            PostgreSQL connection string
        feature_store : FeatureStore | None
            Feature store instance (created if not provided)
        model_store : ModelStore | None
            Model store instance (created if not provided)
        strategy_store : StrategyStore | None
            Strategy store instance (created if not provided)
        fail_on_validation_error : bool
            If True, fail writes on validation errors. If False, log warnings.
        batch_size : int
            Batch size for write operations
        allow_schema_migration : bool
            If True, allow dual-write during schema migration window
        schema_migration_window_hours : int
            Hours to allow dual-write during schema migration

        """
        # Explicit attribute annotation for protocol conformance
        self.registry: RegistryProtocol
        # Expose connection_string for DataRegistryMixin
        self.connection_string = connection_string

        # Optional circuit breaker for orchestration contexts (propagated to stores)
        self._circuit_breaker: CircuitBreakerProtocol | None = circuit_breaker

        # Raw IO adapters (optional)
        self._raw_writer = raw_writer
        self._raw_reader = raw_reader

        # Allow registry to be optional for convenience in tests/integration
        if registry is None:
            reg_obj: RegistryProtocol | None = self._get_data_registry()
            if reg_obj is None:
                # Very defensive fallback to JSON-backed registry
                try:
                    from ml.registry.data_registry import DataRegistry
                    from ml.registry.persistence import BackendType
                    from ml.registry.persistence import PersistenceConfig

                    default_registry_dir = get_default_registry_path()
                    try:
                        default_registry_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        logger.debug(
                            "Creating default registry dir failed: %s",
                            exc,
                            exc_info=True,
                        )
                    reg_obj = DataRegistry(
                        registry_path=default_registry_dir,
                        persistence_config=PersistenceConfig(
                            backend=BackendType.JSON,
                            json_path=default_registry_dir,
                        ),
                    )
                except Exception:
                    reg_obj = None
            if reg_obj is None:
                raise RuntimeError("Failed to initialize DataRegistry")
            self.registry = reg_obj
            # Keep mixin cache in sync
            self._data_registry = self.registry
        else:
            self.registry = registry
            self._data_registry = registry
        self.fail_on_validation_error = fail_on_validation_error
        self.batch_size = batch_size
        self.allow_schema_migration = allow_schema_migration
        self.schema_migration_window_hours = schema_migration_window_hours

        # Topic scheme/prefix (env-driven defaults). Publishing is disabled by
        # default to avoid accidental PII/cardinality leaks. For backwards-
        # compatibility with existing unit tests, implicitly enable when a
        # non-None publisher is provided unless explicitly disabled.
        self._init_bus_publishing(
            enable_publishing=(enable_publishing or (publisher is not None)),
            publisher=publisher,
            publish_mode="batch",
        )

        # Initialize underlying stores
        self.feature_store = feature_store or FeatureStore(connection_string)
        self.model_store = model_store or ModelStore(connection_string)
        self.strategy_store = strategy_store or StrategyStore(connection_string)
        self._earnings_store: EarningsStoreProtocol = (
            earnings_store if earnings_store is not None else self._create_earnings_store(connection_string)
        )

        # Propagate circuit breaker to created stores if provided
        if self._circuit_breaker is not None:
            circuit_breaker_name = type(self._circuit_breaker).__name__
            for store_name, store in (
                ("feature_store", self.feature_store),
                ("model_store", self.model_store),
                ("strategy_store", self.strategy_store),
            ):
                if store is None:
                    continue
                try:
                    setattr(store, "_circuit_breaker", self._circuit_breaker)
                except Exception:
                    logger.debug(
                        "Failed to propagate circuit breaker to %s",
                        store_name,
                        extra={
                            "store": store_name,
                            "circuit_breaker": circuit_breaker_name,
                        },
                        exc_info=True,
                    )
        # Optional message bus publisher (no-op by default if None)
        self.publisher: MessagePublisherProtocol | None = publisher

        # Initialize data processor for validation
        self.data_processor = data_processor or DataProcessor(connection_string)

        # Cache for manifests and contracts
        self._manifest_cache: dict[str, DatasetManifest] = {}
        self._contract_cache: dict[str, DataContract] = {}
        self._schema_migration_state: dict[str, dict[str, Any]] = {}

        logger.info(
            "Initialized DataStore with fail_on_validation=%s, batch_size=%d, allow_migration=%s",
            fail_on_validation_error,
            batch_size,
            allow_schema_migration,
        )

    def _create_earnings_store(self, connection_string: str) -> EarningsStoreProtocol:
        """
        Initialize earnings store with progressive fallback.
        """
        try:
            store = EarningsStore(connection_string)
            logger.info("Initialized EarningsStore via DataStore facade")
            return store
        except Exception as exc:
            logger.warning("EarningsStore initialization failed; attempting fallback: %s", exc)
            file_store = self._try_file_earnings_store()
            if file_store is not None:
                return file_store
            logger.warning("File-backed earnings fallback unavailable; using DummyEarningsStore")
            self._record_fallback_metric(level="dummy")
            return DummyEarningsStore()

    def _try_file_earnings_store(self) -> EarningsStoreProtocol | None:
        """Attempt to initialize the file-backed earnings store fallback."""
        if not HAS_POLARS:
            logger.debug("Skipping file earnings fallback because polars is not available")
            return None
        file_root_str = os.getenv("ML_FILE_STORE_PATH")
        file_root = Path(file_root_str) if file_root_str else Path.home() / ".nautilus" / "ml" / "file_store"
        earnings_path = file_root / "earnings"
        try:
            store = FileEarningsStore(base_path=earnings_path)
        except Exception as file_exc:  # pragma: no cover - IO/path specific failures
            logger.debug("FileEarningsStore initialization failed: %s", file_exc, exc_info=True)
            return None
        logger.info("Initialized FileEarningsStore fallback at %s", earnings_path)
        self._record_fallback_metric(level="file")
        return store

    @staticmethod
    def _record_fallback_metric(*, level: str) -> None:
        """Increment the fallback activation counter when available."""
        try:  # pragma: no cover - metrics optional
            from ml.common.metrics_manager import MetricsManager as _MM

            _MM.default().inc(
                "ml_fallback_activations_total",
                "Fallback activations",
                labels={"component": "data_store", "level": level},
                labelnames=("component", "level"),
            )
        except Exception:
            logger.debug("Failed to record fallback activation metric", exc_info=True)

    # ---------------------------------------------------------------------
    # Typed, minimal read facades (cold-path only; for actors/services)
    # ---------------------------------------------------------------------

    def get_features_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
    ) -> dict[str, float] | None:
        """
        Return latest feature values at or before the given timestamp.

        Notes
        -----
        This is a thin façade over FeatureStore.get_latest_at_or_before.

        """
        return self.feature_store.get_latest_at_or_before(instrument_id, int(ts_event))

    def get_latest_prediction_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        model_id: str | None = None,
    ) -> PredictionRecord | None:
        """
        Return latest prediction at or before ts_event (optionally filtered by
        model_id).

        Returns
        -------
        PredictionRecord | None
            Minimal typed record or None when not found.

        """
        # Lazy imports to keep import-time overhead minimal
        from sqlalchemy import and_ as _and
        from sqlalchemy import desc as _desc
        from sqlalchemy import select as _select

        table = getattr(self.model_store, "model_predictions_table", None)
        engine = getattr(self.model_store, "engine", None)
        if table is None or engine is None:
            return None

        where = [table.c.instrument_id == instrument_id, table.c.ts_event <= int(ts_event)]
        if model_id is not None:
            where.append(table.c.model_id == model_id)

        stmt = (
            _select(
                table.c.model_id,
                table.c.ts_event,
                table.c.prediction,
                table.c.confidence,
            )
            .where(_and(*where))
            .order_by(_desc(table.c.ts_event))
            .limit(1)
        )

        with engine.connect() as conn:
            row = conn.execute(stmt).fetchone()
        if row is None:
            return None
        return PredictionRecord(
            model_id=str(row[0]),
            ts_event=int(row[1]),
            prediction=float(row[2]) if row[2] is not None else 0.0,
            confidence=float(row[3]) if row[3] is not None else 0.0,
        )

    def get_latest_signal_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        strategy_id: str | None = None,
    ) -> SignalRecord | None:
        """
        Return latest strategy signal at or before ts_event (optionally by strategy_id).

        Returns
        -------
        SignalRecord | None
            Minimal typed record or None when not found.

        """
        from sqlalchemy import and_ as _and
        from sqlalchemy import desc as _desc
        from sqlalchemy import select as _select

        table = getattr(self.strategy_store, "strategy_signals_table", None)
        engine = getattr(self.strategy_store, "engine", None)
        if table is None or engine is None:
            return None

        where = [table.c.instrument_id == instrument_id, table.c.ts_event <= int(ts_event)]
        if strategy_id is not None:
            where.append(table.c.strategy_id == strategy_id)

        stmt = (
            _select(
                table.c.strategy_id,
                table.c.ts_event,
                table.c.signal_type,
                table.c.strength,
            )
            .where(_and(*where))
            .order_by(_desc(table.c.ts_event))
            .limit(1)
        )

        with engine.connect() as conn:
            row = conn.execute(stmt).fetchone()
        if row is None:
            return None
        return SignalRecord(
            strategy_id=str(row[0]),
            ts_event=int(row[1]),
            signal_type=str(row[2]),
            strength=float(row[3]) if row[3] is not None else 0.0,
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
        Return earnings actuals visible at the specified timestamp.

        Records are filtered point-in-time and truncated to ``limit`` entries.
        """
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize_ts

        if limit <= 0:
            return []

        as_of_ts = _sanitize_ts(int(ts_event), context="data_store.get_earnings_actuals_at_or_before:ts_event")
        records = self._earnings_store.get_actuals(
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
        Return the latest consensus estimate for the specified period at ``ts_event``.
        """
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize_ts

        as_of_ts = _sanitize_ts(int(ts_event), context="data_store.get_earnings_estimate_at_or_before:ts_event")
        return self._earnings_store.get_estimates(
            ticker=ticker,
            period_end=period_end,
            as_of_ts=as_of_ts,
        )

    # ---------------------------------------------------------------------
    # Protocol surface wrappers (visible to mypy; delegate to mixins at runtime)
    # ---------------------------------------------------------------------
    def _init_bus_publishing(
        self,
        *,
        enable_publishing: bool,
        publisher: MessagePublisherProtocol | None,
        publish_mode: Literal["batch", "row", "both"] = "batch",
    ) -> None:
        """
        Initialize message bus publishing behavior.

        Delegates to BusPublisherMixin at runtime. Present here to satisfy Protocol-
        based type checking when TYPE_CHECKING is true.

        """
        # Delegate explicitly to the runtime mixin implementation
        from typing import cast as _cast

        from ml.common.message_bus import BusPublisherMixin as _BPM

        _BPM._init_bus_publishing(
            _cast("BusPublisherMixin", self),
            enable_publishing=enable_publishing,
            publisher=publisher,
            publish_mode=publish_mode,
        )

    def _get_data_registry(self) -> RegistryProtocol | None:
        """
        Return a DataRegistry instance with progressive fallback.

        Delegates to DataRegistryMixin at runtime.

        """
        from typing import cast as _cast

        from ml.stores.mixins import DataRegistryMixin as _DRM

        return _DRM._get_data_registry(_cast("DataRegistryMixin", self))

    def get_health_status(self) -> dict[str, Any]:
        """
        Lightweight health snapshot for diagnostics.
        """
        from typing import cast as _cast

        from ml.common.protocols import MLComponentMixin as _MM

        return _MM.get_health_status(_cast("MLComponentMixin", self))

    def get_performance_metrics(self) -> dict[str, float]:
        """
        Lightweight performance metrics for diagnostics.
        """
        from typing import cast as _cast

        from ml.common.protocols import MLComponentMixin as _MM

        return _MM.get_performance_metrics(_cast("MLComponentMixin", self))

    def validate_configuration(self) -> list[str]:
        """
        Validate configuration and return any issues.
        """
        from typing import cast as _cast

        from ml.common.protocols import MLComponentMixin as _MM

        return _MM.validate_configuration(_cast("MLComponentMixin", self))

    # =========================================================================
    # Write Operations
    # =========================================================================

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
        Emit a dataset processing event via the centralized helper.

        This method is a thin wrapper around emit_dataset_event() that maintains
        backwards compatibility while ensuring consistent correlation_id attachment
        and metrics labeling.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        instrument_id : str
            Instrument identifier.
        stage : Stage | str
            Processing stage.
        source : Source | str
            Event source (live/historical/backfill).
        run_id : str
            Unique identifier for this processing run.
        ts_min : int
            Minimum timestamp (ns) for covered data.
        ts_max : int
            Maximum timestamp (ns) for covered data.
        count : int
            Number of records processed.
        status : str
            Status string (EventStatus.value: "success", "failed", or "partial").
        error : str | None
            Error message if status is failed.
        metadata : dict[str, Any] | None
            Additional metadata to attach to the event.

        """
        # Normalize inputs (be tolerant of unknown sources by defaulting to 'live')
        stage_enum = to_stage_enum(stage)
        source_enum = to_source_enum(source)
        status_enum = to_status_enum(status)

        # Build correlation_id and merged metadata once
        corr_id = make_correlation_id(
            run_id=run_id,
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
        )
        event_metadata: dict[str, Any] = {"correlation_id": corr_id}
        if metadata:
            event_metadata.update({k: v for k, v in metadata.items() if k != "correlation_id"})

        # Emit directly to registry (robust against tests that monkeypatch the helper)
        try:
            self.registry.emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage_enum,
                source=source_enum,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=status_enum,
                error=error,
                metadata=event_metadata,
            )
        except TypeError:
            # Backwards-compatible registries may not accept metadata
            self.registry.emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage_enum,
                source=source_enum,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=status_enum,
                error=error,
            )

        # Optionally publish to message bus using selected topic scheme (respects _enable_publishing flag)
        if self._enable_publishing and self.publisher is not None:
            # event_metadata already contains correlation_id and merged labels

            topic = build_topic_for_stage(
                stage_enum,
                instrument_id,
                scheme=self._topic_scheme,
                prefix=self._topic_prefix,
            )
            payload = build_bus_payload(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage_enum.value,
                source=source_enum.value,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=status_enum.value,
                metadata=event_metadata,
            )
            try:
                self.publisher.publish(topic, payload)
            except Exception:
                logger.exception("Message bus publish failed for topic %s", topic)

    # Backwards-compatible alias used by tests
    def emit_dataset_event(
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
    ) -> None:
        """
        Alias for emit_event with a reduced parameter set used by unit tests.

        Parameters mirror `emit_event` but omit optional error/metadata.

        """
        # Lightweight path for tests: avoid registry round-trip and focus on bus publish
        stage_enum = to_stage_enum(stage)
        source_enum = to_source_enum(source)
        status_enum = to_status_enum(status)

        # Deterministic correlation id for payload metadata
        corr_id = make_correlation_id(
            run_id=run_id,
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=count,
        )

        if self._enable_publishing and self.publisher is not None:
            topic = build_topic_for_stage(
                stage_enum,
                instrument_id,
                scheme=self._topic_scheme,
                prefix=self._topic_prefix,
            )
            payload = build_bus_payload(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage_enum.value,
                source=source_enum.value,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=status_enum.value,
                metadata={"correlation_id": corr_id},
            )
            try:
                self.publisher.publish(topic, payload)
            except Exception:
                logger.exception("Message bus publish failed for topic %s", topic)

    def preflight_check(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],
        strict: bool = True,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Perform preflight schema validation before processing data.

        This method checks that the data conforms to the expected schema
        without actually writing anything. It validates column names,
        data types, required fields, and schema hash compatibility.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        data : DataFrame | list
            Data to validate
        strict : bool
            If True, require exact schema match. If False, allow subset.

        Returns
        -------
        tuple[bool, str | None, dict[str, Any]]
            (success, error_message, validation_details)

        Examples
        --------
        >>> success, error, details = store.preflight_check("bars_eurusd_1m", data_frame)
        >>> if not success:
        ...     print(f"Preflight check failed: {error}")
        ...     print(f"Details: {details}")

        """
        try:
            # Get manifest and contract
            manifest = self._get_manifest(dataset_id)

            # Convert to DataFrame for validation
            data_frame_obj = self._to_dataframe(data)
            data_frame = cast(DataFrameLike, data_frame_obj)
            data_frame_any = cast(Any, data_frame)

            validation_details: dict[str, Any] = {
                "dataset_id": dataset_id,
                "expected_schema_hash": manifest.schema_hash,
                "checks_performed": [],
                "warnings": [],
            }

            # Empty data shortcut: accept empty DataFrames as valid shape
            try:
                if hasattr(data_frame_any, "__len__") and len(data_frame_any) == 0:
                    validation_details["empty_data"] = True
                    validation_details["preflight_passed"] = True
                    return True, None, validation_details
            except Exception as exc:
                # Continue with normal checks if len() is not supported
                logger.debug(
                    "len() check failed during preflight (ignored): %s",
                    exc,
                    exc_info=True,
                )

            # Check 1: Required columns present
            if hasattr(data_frame, "columns"):
                actual_columns = set(data_frame.columns)
                expected_columns = set(manifest.schema.keys())

                missing_columns = expected_columns - actual_columns
                extra_columns = actual_columns - expected_columns

                validation_details["actual_columns"] = list(actual_columns)
                validation_details["expected_columns"] = list(expected_columns)
                validation_details["checks_performed"].append("column_presence")

                if missing_columns:
                    error_msg = f"Missing required columns: {missing_columns}"
                    validation_details["missing_columns"] = list(missing_columns)
                    if strict or any(col in manifest.primary_keys for col in missing_columns):
                        return False, error_msg, validation_details
                    else:
                        validation_details["warnings"].append(error_msg)

                if extra_columns and strict:
                    error_msg = f"Unexpected columns: {extra_columns}"
                    validation_details["extra_columns"] = list(extra_columns)
                    # Treat as schema hash mismatch for metrics purposes
                    if HAS_PROMETHEUS:
                        schema_mismatch_counter.labels(
                            dataset=dataset_id,
                            mismatch_type="hash_mismatch",
                        ).inc()
                    return False, error_msg, validation_details
                elif extra_columns:
                    validation_details["warnings"].append(
                        f"Extra columns will be ignored: {extra_columns}",
                    )

            # Check 2: Data types compatibility
            type_mismatches: list[dict[str, str]] = []
            if hasattr(data_frame, "columns"):
                validation_details["checks_performed"].append("type_compatibility")
                for col_name, expected_type in manifest.schema.items():
                    if col_name in data_frame.columns:
                        actual_type = str(data_frame[col_name].dtype)
                        if not self._types_compatible(actual_type, expected_type):
                            type_mismatches.append(
                                {
                                    "column": col_name,
                                    "expected": expected_type,
                                    "actual": actual_type,
                                },
                            )

            if type_mismatches:
                validation_details["type_mismatches"] = type_mismatches
                if strict:
                    error_msg = f"Type mismatches found: {type_mismatches}"
                    return False, error_msg, validation_details
                else:
                    validation_details["warnings"].append(
                        f"Type coercion will be attempted for {len(type_mismatches)} columns",
                    )

            # Check 3: Schema hash compatibility
            actual_schema_hash = self._compute_schema_hash(data_frame, manifest)
            validation_details["actual_schema_hash"] = actual_schema_hash
            validation_details["checks_performed"].append("schema_hash")

            if actual_schema_hash != manifest.schema_hash:
                # Check if we're in a migration window
                if self._is_in_migration_window(dataset_id):
                    validation_details["migration_mode"] = True
                    validation_details["warnings"].append(
                        "Schema migration in progress - dual-write enabled",
                    )
                else:
                    error_msg = (
                        f"Schema hash mismatch. Expected: {manifest.schema_hash}, "
                        f"Got: {actual_schema_hash}. Version bump required."
                    )
                    # Record schema mismatch metric
                    if HAS_PROMETHEUS:
                        schema_mismatch_counter.labels(
                            dataset=dataset_id,
                            mismatch_type="hash_mismatch",
                        ).inc()

                    if strict:
                        return False, error_msg, validation_details
                    else:
                        validation_details["warnings"].append(error_msg)

            # Check 4: Primary key fields present and not null
            if manifest.primary_keys:
                validation_details["checks_performed"].append("primary_keys")
                for pk_field in manifest.primary_keys:
                    if hasattr(data_frame_any, "columns") and pk_field in data_frame_any.columns:
                        # Handle both Polars and pandas
                        if hasattr(data_frame_any[pk_field], "is_null"):
                            # Polars
                            null_count = data_frame_any[pk_field].is_null().sum()
                        elif hasattr(data_frame_any[pk_field], "isna"):
                            # pandas
                            null_count = data_frame_any[pk_field].isna().sum()
                        else:
                            null_count = 0

                        if null_count > 0:
                            error_msg = (
                                f"Primary key field '{pk_field}' contains {null_count} null values"
                            )
                            return False, error_msg, validation_details

            # Check 5: Required fields (from constraints)
            if manifest.constraints and "nullability" in manifest.constraints:
                validation_details["checks_performed"].append("required_fields")
                for field, nullable in manifest.constraints["nullability"].items():
                    if not nullable and hasattr(data_frame, "columns") and field in data_frame.columns:
                        # Handle both Polars and pandas
                        if hasattr(data_frame[field], "is_null"):
                            # Polars
                            null_count = data_frame[field].is_null().sum()
                        elif hasattr(data_frame[field], "isna"):
                            # pandas
                            null_count = data_frame[field].isna().sum()
                        else:
                            null_count = 0

                        if null_count > 0:
                            error_msg = (
                                f"Required field '{field}' contains {null_count} null values"
                            )
                            return False, error_msg, validation_details

            validation_details["preflight_passed"] = True
            return True, None, validation_details

        except Exception:
            error_msg = "Preflight check failed"
            logger.error(
                error_msg,
                exc_info=True,
            )
            return False, error_msg, {}

    def _record_observability_stage_boundary(
        self,
        *,
        stage: str,
        instrument_id: str,
        ts_stage_start: int,
        ts_stage_end: int,
        row_count: int = 1,
    ) -> None:
        """
        Record observability data via centralized helper (cold path only).
        """
        from ml.common.observability_utils import record_stage_boundary as _rec

        obs_service = getattr(self, "_observability_service", None)
        _rec(
            obs_service,
            component="data_store",
            instrument_id=instrument_id,
            stage=stage,
            ts_stage_start=ts_stage_start,
            ts_stage_end=ts_stage_end,
            row_count=row_count,
        )

    def write_ingestion(
        self,
        dataset_id: str,
        records: list[dict[str, Any]] | DataFrameLike,  # Accept DataFrame or list
        source: str,
        run_id: str,
        instrument_id: str | None = None,
    ) -> DataEvent:
        """
        Write ingestion data with contract validation and event emission.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        records : list[dict[str, Any]] | DataFrame
            Data records to write
        source : str
            Data source (live, historical, backfill)
        run_id : str
            Processing run identifier
        instrument_id : str | None
            Instrument identifier (extracted from data if not provided)

        Returns
        -------
        DataEvent
            Event tracking the write operation

        Raises
        ------
        ValueError
            If dataset not found or validation fails (when fail_on_validation_error=True)
        RuntimeError
            If write operation fails

        Examples
        --------
        >>> event = store.write_ingestion(
        ...     dataset_id="bars_eurusd_1m",
        ...     records=bar_data,
        ...     source="historical",
        ...     run_id="run_20240101_120000"
        ... )
        >>> print(f"Wrote {event.record_count} records")

        """
        start_time = time.perf_counter()
        ts_stage_start = time.time_ns()  # For observability tracking

        # Get manifest and contract
        manifest = self._get_manifest(dataset_id)
        contract = self._get_contract(dataset_id)

        # Perform preflight schema check
        preflight_passed, preflight_error, preflight_details = self.preflight_check(
            dataset_id,
            records,
            strict=self.fail_on_validation_error,
        )

        if not preflight_passed:
            # Record write rejection metric
            if HAS_PROMETHEUS:
                write_rejection_counter.labels(
                    dataset_id=dataset_id,
                    reason="preflight_failed",
                ).inc()

            raise ValueError(
                f"Preflight check failed for {dataset_id}: {preflight_error}. "
                f"Details: {preflight_details}",
            )

        # Log warnings from preflight check
        if preflight_details.get("warnings"):
            for warning in preflight_details["warnings"]:
                logger.warning("Preflight warning for %s: %s", dataset_id, warning)

        # Convert to DataFrame if needed
        data_frame_obj = self._to_dataframe(records)
        data_frame = cast(DataFrameLike, data_frame_obj)

        extra_metadata: dict[str, object] = {}
        if pd is not None:
            data_frame_for_meta: pd.DataFrame | None
            if isinstance(data_frame, pd.DataFrame):
                data_frame_for_meta = data_frame
            elif hasattr(data_frame, "to_pandas") and callable(getattr(data_frame, "to_pandas")):
                try:
                    data_frame_for_meta = data_frame.to_pandas()
                except Exception:  # pragma: no cover - defensive conversion
                    data_frame_for_meta = None
            else:
                data_frame_for_meta = None
            if data_frame_for_meta is not None:
                extra_metadata = self._extract_ingestion_metadata_from_dataframe(data_frame_for_meta)

        # Extract instrument_id if not provided
        if instrument_id is None:
            if hasattr(data_frame, "columns") and "instrument_id" in data_frame.columns:
                # Handle both Polars and pandas
                col = data_frame["instrument_id"]
                if hasattr(col, "iloc"):
                    # pandas Series
                    instrument_id = str(col.iloc[0])
                else:
                    # Polars Series or other
                    instrument_id = str(col[0])
            else:
                instrument_id = "UNKNOWN"

        # Validate data against contract based on enforcement mode
        # Only use strict_mode if contract enforcement is "strict"
        use_strict = contract.enforcement_mode == "strict"
        quality_report = self.validate_batch(dataset_id, data_frame, strict_mode=use_strict)

        # Check validation results with fail-closed approach
        if quality_report.quality_score < 1.0:
            violations_str = self._format_violations(quality_report.violations)

            # Count critical violations (FAIL severity)
            critical_violations = [
                v for v in quality_report.violations if v.severity == QualityFlag.FAIL
            ]

            # Fail-closed: Block any data with critical violations (unless monitor_only)
            if critical_violations and contract.enforcement_mode != "monitor_only":
                # Record write rejection metric
                if HAS_PROMETHEUS:
                    write_rejection_counter.labels(
                        dataset_id=dataset_id,
                        reason="validation_failed",
                    ).inc()

                raise ValueError(
                    f"Data validation failed for {dataset_id} (fail-closed). "
                    f"Quality score: {quality_report.quality_score:.2f}. "
                    f"Critical violations: {len(critical_violations)}. "
                    f"Details: {violations_str}",
                )

            # For non-critical violations, check enforcement mode
            if self.fail_on_validation_error and contract.enforcement_mode == "strict":
                # Record write rejection metric
                if HAS_PROMETHEUS:
                    write_rejection_counter.labels(
                        dataset_id=dataset_id,
                        reason="strict_mode_violation",
                    ).inc()

                raise ValueError(
                    f"Data validation failed for {dataset_id} (strict mode). "
                    f"Quality score: {quality_report.quality_score:.2f}. "
                    f"Violations: {violations_str}",
                )
            elif contract.enforcement_mode == "lenient":
                logger.warning(
                    "Data validation warnings for %s (lenient mode): %s",
                    dataset_id,
                    violations_str,
                )
            else:  # monitor_only
                logger.info(
                    "Data validation issues for %s (monitor-only): %s",
                    dataset_id,
                    violations_str,
                )

        # Extract timestamp range
        ts_field = manifest.ts_field
        ts_min = int(cast(Any, data_frame)[ts_field].min())
        ts_max = int(cast(Any, data_frame)[ts_field].max())

        # Determine appropriate store and stage based on dataset type
        stage = self._get_stage_for_dataset_type(manifest.dataset_type)

        try:
            # Route to appropriate store based on dataset type
            if manifest.dataset_type == DatasetType.FEATURES:
                # Convert to FeatureData format and write
                feature_data = self._data_frame_to_feature_data(data_frame, instrument_id)
                # Note: FeatureStore doesn't have a batch method, write individually
                for feature in feature_data:
                    self.feature_store.write_features(
                        feature_set_id=feature.feature_set_id,
                        instrument_id=feature.instrument_id,
                        features=feature.values,
                        ts_event=feature.ts_event,
                        ts_init=feature.ts_init,
                    )

            elif manifest.dataset_type == DatasetType.PREDICTIONS:
                # Convert to ModelPrediction format and write
                predictions = self._data_frame_to_predictions(data_frame)
                try:
                    self.model_store.write_batch(predictions, emit_events=False, publish_bus=False)
                except TypeError:
                    self.model_store.write_batch(predictions)

            elif manifest.dataset_type == DatasetType.SIGNALS:
                # Convert to StrategySignal format and write
                signals = self._data_frame_to_signals(data_frame)
                self.strategy_store.write_batch(signals, emit_events=False, publish_bus=False)

            else:
                # Raw dataset types: delegate to optional writer if configured
                if self._raw_writer is not None:
                    try:
                        written = self._raw_writer.write(
                            dataset_type=manifest.dataset_type,
                            data=data_frame,
                        )
                        if written <= 0:
                            logger.warning(
                                "Raw writer reported 0 records written for %s",
                                dataset_id,
                            )
                            # Emit PARTIAL without watermark to avoid false coverage
                            self._emit_partial_event(
                                dataset_id=dataset_id,
                                instrument_id=instrument_id,
                                stage=stage,
                                source=source,
                                run_id=run_id,
                                ts_min=ts_min,
                                ts_max=ts_max,
                                count=len(data_frame),
                                reason="no_records_written",
                            )
                            return DataEvent(
                                event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
                                dataset_id=dataset_id,
                                instrument_id=instrument_id,
                                operation="write_ingestion",
                                source=source,
                                run_id=run_id,
                                ts_min=ts_min,
                                ts_max=ts_max,
                                record_count=len(data_frame),
                                status=EventStatus.PARTIAL.value,
                                metadata={"no_write": True},
                            )
                    except Exception as exc:  # best-effort; keep off hot path
                        logger.error(
                            "Raw writer failed for %s",
                            dataset_id,
                            exc_info=True,
                        )
                        self._emit_failed_event(
                            dataset_id=dataset_id,
                            instrument_id=instrument_id,
                            stage=stage,
                            source=source,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            count=len(data_frame),
                            error=str(exc),
                        )
                        return DataEvent(
                            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
                            dataset_id=dataset_id,
                            instrument_id=instrument_id,
                            operation="write_ingestion",
                            source=source,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            record_count=len(data_frame),
                            status=EventStatus.FAILED.value,
                            error_message=str(exc),
                        )
                else:
                    # No raw writer configured. Use fail_on_validation_error to decide behavior:
                    # - True: proceed with SUCCESS (best-effort ingestion semantics for pipelines/tests)
                    # - False: emit PARTIAL without watermark to avoid false coverage
                    if self.fail_on_validation_error:
                        logger.info(
                            "Raw writer not configured; proceeding with success for %s (best-effort)",
                            dataset_id,
                        )
                        # Emit SUCCESS event and update watermark for visibility
                        self._emit_success_event_and_update(
                            dataset_id=dataset_id,
                            instrument_id=instrument_id,
                            stage=stage,
                            source=source,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            count=len(data_frame),
                        )
                        return DataEvent(
                            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
                            dataset_id=dataset_id,
                            instrument_id=instrument_id,
                            operation="write_ingestion",
                            source=source,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            record_count=len(data_frame),
                            status=EventStatus.SUCCESS.value,
                            metadata={"no_write": True, **extra_metadata},
                        )
                    else:
                        logger.warning(
                            "Raw writer not configured; skipping persistence for %s",
                            dataset_id,
                        )
                        # Emit PARTIAL event without watermark; annotate metadata
                        self._emit_partial_event(
                            dataset_id=dataset_id,
                            instrument_id=instrument_id,
                            stage=stage,
                            source=source,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            count=len(data_frame),
                            reason="raw_writer_missing",
                        )
                        return DataEvent(
                            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
                            dataset_id=dataset_id,
                            instrument_id=instrument_id,
                            operation="write_ingestion",
                            source=source,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            record_count=len(data_frame),
                            status=EventStatus.PARTIAL.value,
                            metadata={"no_write": True, **extra_metadata},
                        )

            # Create event
            from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

            ts_min_s = _sanitize(int(ts_min), context="data_store.write_ingestion:ts_min")
            ts_max_s = _sanitize(int(ts_max), context="data_store.write_ingestion:ts_max")

            event = DataEvent(
                event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                operation="write_ingestion",
                source=source,
                run_id=run_id,
                ts_min=ts_min_s,
                ts_max=ts_max_s,
                record_count=len(data_frame),
                status=EventStatus.SUCCESS.value,
                metadata={
                    "quality_score": quality_report.quality_score,
                    "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                    **extra_metadata,
                },
            )

            # Centralized event + watermark emission via helper (best-effort)
            try:
                stage_enum = to_stage_enum(stage)
                src_enum = to_source_enum(source)
                emit_dataset_event_and_watermark(
                    self.registry,
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    stage=stage_enum,
                    source=src_enum,
                    run_id=run_id,
                    ts_min=ts_min_s,
                    ts_max=ts_max_s,
                    count=len(data_frame),
                    status=EventStatus.SUCCESS,
                    dataset_type=str(
                        (
                            manifest.dataset_type.value
                            if hasattr(manifest.dataset_type, "value")
                            else manifest.dataset_type
                        ),
                    ),
                    component=self.__class__.__name__,
                    metadata=event.metadata,
                )
                # Publish to message bus on success (non-blocking best-effort)
                if self._enable_publishing and self.publisher is not None:
                    try:
                        src_norm = to_source_str(src_enum)
                        # Deterministic correlation id
                        correlation_id = make_correlation_id(
                            run_id=run_id,
                            dataset_id=dataset_id,
                            instrument_id=instrument_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            count=len(data_frame),
                        )
                        topic = build_topic_for_stage(
                            stage_enum,
                            instrument_id,
                            scheme=self._topic_scheme,
                            prefix=self._topic_prefix,
                        )
                        payload: dict[str, Any] = {
                            "dataset_id": dataset_id,
                            "instrument_id": instrument_id,
                            "stage": stage_enum.value,
                            "source": src_norm,
                            "run_id": run_id,
                            "ts_min": ts_min_s,
                            "ts_max": ts_max_s,
                            "count": len(data_frame),
                            "status": EventStatus.SUCCESS.value,
                            "metadata": {"correlation_id": correlation_id},
                        }
                        self.publisher.publish(topic, payload)
                    except Exception:
                        logger.exception("Message bus publish failed for dataset %s", dataset_id)
            except Exception:
                logger.warning("Failed to emit dataset event/watermark via helper", exc_info=True)

            logger.info(
                "Successfully wrote %d records to %s (quality=%.2f)",
                len(data_frame),
                dataset_id,
                quality_report.quality_score,
            )

            # Record observability data (off hot path - background processing only)
            ts_stage_end = time.time_ns()
            self._record_observability_stage_boundary(
                stage="data_ingestion",
                instrument_id=str(instrument_id or "unknown"),
                ts_stage_start=ts_stage_start,
                ts_stage_end=ts_stage_end,
                row_count=len(data_frame),
            )

            return event

        except Exception as exc:
            # Create failure event
            event = DataEvent(
                event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                operation="write_ingestion",
                source=source,
                run_id=run_id,
                ts_min=ts_min if "ts_min" in locals() else 0,
                ts_max=ts_max if "ts_max" in locals() else 0,
                record_count=0,
                status=EventStatus.FAILED.value,
                error_message=str(exc),
            )

            # Emit failure event via façade
            self.emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage,
                source=source,
                run_id=run_id,
                ts_min=0,
                ts_max=0,
                count=0,
                status=EventStatus.FAILED.value,
                error=str(exc),
            )

            logger.error(
                "Failed to write data to %s",
                dataset_id,
                exc_info=True,
            )
            raise RuntimeError(f"Write operation failed: {exc}") from exc

    @staticmethod
    def _extract_ingestion_metadata_from_dataframe(data_frame: pd.DataFrame) -> dict[str, object]:
        if pd is None:
            return {}
        metadata: dict[str, object] = {}
        if data_frame.empty:
            return metadata

        if "source_dataset" in data_frame.columns:
            values = (
                data_frame["source_dataset"].dropna().astype(str).unique().tolist()
            )
            normalized = [value for value in values if value]
            if normalized:
                metadata["source_datasets"] = sorted(dict.fromkeys(normalized))

        return metadata

    def write_features(
        self,
        instrument_id: str,
        features: list[FeatureData],
        source: str = "computed",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write features with validation and event emission.

        Wraps FeatureStore.store_features_batch with contract validation and event tracking.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        features : list[FeatureData]
            Feature data to store
        source : str
            Data source (default: "computed")
        run_id : str | None
            Processing run identifier (generated if not provided)

        Returns
        -------
        DataEvent
            Event tracking the write operation

        Examples
        --------
        >>> event = store.write_features(
        ...     instrument_id="EUR/USD",
        ...     features=feature_list,
        ...     source="realtime"
        ... )

        """
        run_id = run_id or f"features_{time.time_ns()}"
        dataset_id = "features"

        # Register dataset if not exists
        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.FEATURES,
            instrument_id=instrument_id,
        )

        # Validate features
        for feature_data in features:
            if feature_data.instrument_id != instrument_id:
                raise ValueError(
                    f"Instrument mismatch: expected {instrument_id}, "
                    f"got {feature_data.instrument_id}",
                )

        stage = Stage.FEATURE_COMPUTED

        try:
            for feature in features:
                self.feature_store.write_features(
                    feature_set_id=feature.feature_set_id,
                    instrument_id=feature.instrument_id,
                    features=feature.values,
                    ts_event=feature.ts_event,
                    ts_init=feature.ts_init,
                    publish_bus=False,
                )
        except Exception as exc:
            logger.exception("Feature store write failed", exc_info=True)
            self.emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage.value,
                source=source,
                run_id=run_id,
                ts_min=0,
                ts_max=0,
                count=0,
                status=EventStatus.FAILED.value,
                error=str(exc),
            )
            raise RuntimeError(f"Feature write failed: {exc}") from exc

        # Calculate timestamp range
        ts_min = min(f.ts_event for f in features)
        ts_max = max(f.ts_event for f in features)

        # Create event and emit/update registry watermarks
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        ts_min_s = _sanitize(int(ts_min), context="data_store.write_features:ts_min")
        ts_max_s = _sanitize(int(ts_max), context="data_store.write_features:ts_max")
        event = DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_features",
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            record_count=len(features),
            status=EventStatus.SUCCESS.value,
        )
        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage.value,
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            count=len(features),
        )

        logger.debug("Wrote %d features for %s", len(features), instrument_id)
        return event

    def write_predictions(
        self,
        predictions: list[ModelPrediction],
        source: str = "inference",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write model predictions with validation and event emission.

        Wraps ModelStore.store_predictions_batch with contract validation and event tracking.

        Parameters
        ----------
        predictions : list[ModelPrediction]
            Model predictions to store
        source : str
            Data source (default: "inference")
        run_id : str | None
            Processing run identifier (generated if not provided)

        Returns
        -------
        DataEvent
            Event tracking the write operation

        Examples
        --------
        >>> event = store.write_predictions(
        ...     predictions=prediction_list,
        ...     source="realtime"
        ... )

        """
        if not predictions:
            raise ValueError("No predictions to write")

        run_id = run_id or f"predictions_{time.time_ns()}"

        # Group by instrument for event emission
        instrument_id = predictions[0].instrument_id
        model_id = predictions[0].model_id
        dataset_id = "predictions"

        # Register dataset if not exists
        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.PREDICTIONS,
            instrument_id=instrument_id,
        )

        # Store predictions
        try:
            self.model_store.write_batch(predictions, emit_events=False, publish_bus=False)
        except TypeError:
            self.model_store.write_batch(predictions)

        # Calculate timestamp range
        ts_min = min(p.ts_event for p in predictions)
        ts_max = max(p.ts_event for p in predictions)

        # Create event and emit/update registry watermarks
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        ts_min_s = _sanitize(int(ts_min), context="data_store.write_predictions:ts_min")
        ts_max_s = _sanitize(int(ts_max), context="data_store.write_predictions:ts_max")
        event = DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_predictions",
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            record_count=len(predictions),
            status=EventStatus.SUCCESS.value,
            metadata={"model_id": model_id},
        )
        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=Stage.PREDICTION_EMITTED.value,
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            count=len(predictions),
        )

        logger.debug(
            "Wrote %d predictions for %s from model %s",
            len(predictions),
            instrument_id,
            model_id,
        )
        return event

    def write_signals(
        self,
        signals: list[StrategySignal],
        source: str = "strategy",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write strategy signals with validation and event emission.

        Wraps StrategyStore.store_signals_batch with contract validation and event tracking.

        Parameters
        ----------
        signals : list[StrategySignal]
            Strategy signals to store
        source : str
            Data source (default: "strategy")
        run_id : str | None
            Processing run identifier (generated if not provided)

        Returns
        -------
        DataEvent
            Event tracking the write operation

        Examples
        --------
        >>> event = store.write_signals(
        ...     signals=signal_list,
        ...     source="realtime"
        ... )

        """
        if not signals:
            raise ValueError("No signals to write")

        run_id = run_id or f"signals_{time.time_ns()}"

        # Group by instrument for event emission
        instrument_id = signals[0].instrument_id
        strategy_id = signals[0].strategy_id
        dataset_id = "signals"

        # Register dataset if not exists
        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.SIGNALS,
            instrument_id=instrument_id,
        )

        # Store signals
        self.strategy_store.write_batch(signals, emit_events=False, publish_bus=False)

        # Calculate timestamp range
        ts_min = min(s.ts_event for s in signals)
        ts_max = max(s.ts_event for s in signals)

        # Create event and emit/update registry watermarks
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        ts_min_s = _sanitize(int(ts_min), context="data_store.write_signals:ts_min")
        ts_max_s = _sanitize(int(ts_max), context="data_store.write_signals:ts_max")
        event = DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_signals",
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            record_count=len(signals),
            status=EventStatus.SUCCESS.value,
            metadata={"strategy_id": strategy_id},
        )
        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=Stage.SIGNAL_EMITTED.value,
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            count=len(signals),
        )

        logger.debug(
            "Wrote %d signals for %s from strategy %s",
            len(signals),
            instrument_id,
            strategy_id,
        )
        return event

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
        """
        Persist an earnings actual record through the facade with contract validation.
        """
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize_ts

        dataset_id = EARNINGS_ACTUALS_DATASET_ID
        run_id_local = run_id or f"earnings_actual_{time.time_ns()}"
        ts_event_s = _sanitize_ts(int(ts_event), context="data_store.write_earnings_actual:ts_event")
        ts_init_s = _sanitize_ts(int(ts_init), context="data_store.write_earnings_actual:ts_init")

        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.EARNINGS_ACTUALS,
            instrument_id=ticker,
        )

        record: dict[str, Any] = {
            "ticker": ticker,
            "period_end": period_end,
            "filing_date": filing_date,
            "ts_event": ts_event_s,
            "ts_init": ts_init_s,
            "eps_basic": eps_basic,
            "eps_diluted": eps_diluted,
            "revenue": revenue,
            "net_income": net_income,
            "operating_income": operating_income,
            "shares_outstanding": shares_outstanding,
            "filing_type": filing_type,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
        }
        record_with_source = dict(record)
        record_with_source["data_source"] = "EDGAR"

        contract = self._get_contract(dataset_id)
        use_strict = contract.enforcement_mode == "strict"
        quality_report = self.validate_batch(dataset_id, [record], strict_mode=use_strict)
        self._enforce_quality_report(
            dataset_id=dataset_id,
            contract=contract,
            quality_report=quality_report,
        )

        try:
            self._earnings_store.write_actuals(
                ticker=ticker,
                period_end=period_end,
                filing_date=filing_date,
                eps_diluted=eps_diluted,
                revenue=revenue,
                ts_event=ts_event_s,
                ts_init=ts_init_s,
                eps_basic=eps_basic,
                net_income=net_income,
                operating_income=operating_income,
                shares_outstanding=shares_outstanding,
                filing_type=filing_type,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
            )
        except Exception as exc:  # pragma: no cover - database failure path
            logger.exception("Earnings actual write failed for %s", ticker)
            raise RuntimeError(f"Earnings actual write failed: {exc}") from exc

        raw_writer_status = "not_configured"
        if self._raw_writer is not None:
            raw_writer_status = "ok"
            try:
                self._raw_writer.write(
                    dataset_type=DatasetType.EARNINGS_ACTUALS,
                    data=[record_with_source],
                )
            except Exception as exc:  # pragma: no cover - mirror failures
                raw_writer_status = "failed"
                logger.warning(
                    "Raw earnings mirror write failed for %s: %s",
                    ticker,
                    exc,
                    exc_info=True,
                )

        event = DataEvent(
            event_id=f"{run_id_local}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=ticker,
            operation="write_earnings_actual",
            source=source,
            run_id=run_id_local,
            ts_min=ts_event_s,
            ts_max=ts_event_s,
            record_count=1,
            status=EventStatus.SUCCESS.value,
            metadata={
                "quality_score": quality_report.quality_score,
                "raw_writer_status": raw_writer_status,
            },
        )

        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=ticker,
            stage=Stage.DATA_INGESTED.value,
            source=source,
            run_id=run_id_local,
            ts_min=ts_event_s,
            ts_max=ts_event_s,
            count=1,
        )

        return event

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
        """
        Persist an earnings estimate record through the facade with contract validation.
        """
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize_ts

        dataset_id = EARNINGS_ESTIMATES_DATASET_ID
        run_id_local = run_id or f"earnings_estimate_{time.time_ns()}"
        ts_event_s = _sanitize_ts(int(ts_event), context="data_store.write_earnings_estimate:ts_event")
        ts_init_s = _sanitize_ts(int(ts_init), context="data_store.write_earnings_estimate:ts_init")

        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.EARNINGS_ESTIMATES,
            instrument_id=ticker,
        )

        record: dict[str, Any] = {
            "ticker": ticker,
            "estimate_date": estimate_date,
            "period_end": period_end,
            "ts_event": ts_event_s,
            "ts_init": ts_init_s,
            "eps_consensus": eps_consensus,
            "revenue_consensus": revenue_consensus,
            "num_analysts": num_analysts,
        }
        record_with_source = dict(record)
        record_with_source["data_source"] = "YAHOO"
        contract = self._get_contract(dataset_id)
        use_strict = contract.enforcement_mode == "strict"
        quality_report = self.validate_batch(dataset_id, [record], strict_mode=use_strict)
        self._enforce_quality_report(
            dataset_id=dataset_id,
            contract=contract,
            quality_report=quality_report,
        )

        try:
            self._earnings_store.write_estimates(
                ticker=ticker,
                estimate_date=estimate_date,
                period_end=period_end,
                eps_consensus=eps_consensus,
                ts_event=ts_event_s,
                ts_init=ts_init_s,
                revenue_consensus=revenue_consensus,
                num_analysts=num_analysts,
            )
        except Exception as exc:  # pragma: no cover - database failure path
            logger.exception("Earnings estimate write failed for %s", ticker)
            raise RuntimeError(f"Earnings estimate write failed: {exc}") from exc

        raw_writer_status = "not_configured"
        if self._raw_writer is not None:
            raw_writer_status = "ok"
            try:
                self._raw_writer.write(
                    dataset_type=DatasetType.EARNINGS_ESTIMATES,
                    data=[record_with_source],
                )
            except Exception as exc:  # pragma: no cover - mirror failures
                raw_writer_status = "failed"
                logger.warning(
                    "Raw earnings mirror write failed for %s estimates: %s",
                    ticker,
                    exc,
                    exc_info=True,
                )

        event = DataEvent(
            event_id=f"{run_id_local}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=ticker,
            operation="write_earnings_estimate",
            source=source,
            run_id=run_id_local,
            ts_min=ts_event_s,
            ts_max=ts_event_s,
            record_count=1,
            status=EventStatus.SUCCESS.value,
            metadata={
                "quality_score": quality_report.quality_score,
                "raw_writer_status": raw_writer_status,
            },
        )

        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=ticker,
            stage=Stage.DATA_INGESTED.value,
            source=source,
            run_id=run_id_local,
            ts_min=ts_event_s,
            ts_max=ts_event_s,
            count=1,
        )

        return event

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _emit_success_event_and_update(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: str,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        completeness_pct: float = 100.0,
    ) -> None:
        """
        Emit a success event and update registry watermark for the dataset.
        """
        try:
            # Build correlation id for observability; used for bus payload only
            correlation_id = make_correlation_id(
                run_id=run_id,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
            )
            source_norm = to_source_str(source)
            stage_enum = to_stage_enum(stage)
            src_enum = to_source_enum(source_norm)

            emit_dataset_event_and_watermark(
                self.registry,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage_enum,
                source=src_enum,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=EventStatus.SUCCESS,
                dataset_type=dataset_id,
                component=self.__class__.__name__,
            )

            # Optionally publish to message bus using selected topic mapping (respects _enable_publishing flag)
            if self._enable_publishing and self.publisher is not None:
                try:
                    topic = build_topic_for_stage(
                        stage_enum,
                        instrument_id,
                        scheme=self._topic_scheme,
                        prefix=self._topic_prefix,
                    )
                except Exception:
                    topic = build_topic_for_stage(
                        Stage.CATALOG_WRITTEN,
                        instrument_id,
                        scheme=self._topic_scheme,
                        prefix=self._topic_prefix,
                    )
                payload = build_bus_payload(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    stage=stage_enum.value,
                    source=source_norm,
                    run_id=run_id,
                    ts_min=ts_min,
                    ts_max=ts_max,
                    count=count,
                    status=EventStatus.SUCCESS,
                    metadata={"correlation_id": correlation_id},
                )
                try:
                    self.publisher.publish(topic, payload)
                except Exception:
                    logger.exception("Message bus publish failed for topic %s", topic)
        except Exception:
            logger.warning("Failed to emit event/update watermark via helper", exc_info=True)

    # =========================================================================
    # Read Operations
    # =========================================================================

    def read_range(
        self,
        dataset_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> object:  # Returns DataFrame or list depending on dataset
        """
        Read data for a specific time range.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        DataFrame | list
            Data for the specified range

        Raises
        ------
        ValueError
            If dataset not found or invalid time range

        Examples
        --------
        >>> data_frame = store.read_range(
        ...     dataset_id="features",
        ...     instrument_id="EUR/USD",
        ...     start_ns=1234567890000000000,
        ...     end_ns=1234567900000000000
        ... )

        """
        manifest = self._get_manifest(dataset_id)

        # Validate time range
        if start_ns >= end_ns:
            raise ValueError(f"Invalid time range: start={start_ns} >= end={end_ns}")

        # Route to appropriate store based on dataset type
        if manifest.dataset_type == DatasetType.FEATURES:
            # Convert nanoseconds to datetime for FeatureStore
            from datetime import datetime

            start_dt = datetime.fromtimestamp(start_ns / 1e9)
            end_dt = datetime.fromtimestamp(end_ns / 1e9)
            # Use feature store's get_training_data method
            return self.feature_store.get_training_data(
                instrument_id=instrument_id,
                start=start_dt,
                end=end_dt,
            )

        elif manifest.dataset_type == DatasetType.PREDICTIONS:
            # Use model store's read_predictions method
            # Extract model_id from dataset_id
            model_id = dataset_id.replace("predictions_", "")
            return self.model_store.read_predictions(
                model_id=model_id,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )

        elif manifest.dataset_type == DatasetType.SIGNALS:
            # Use strategy store's read_signals method
            # Extract strategy_id from dataset_id
            strategy_id = dataset_id.replace("signals_", "")
            return self.strategy_store.read_signals(
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )

        else:
            # Raw datasets: delegate to optional reader when available
            if self._raw_reader is not None:
                return self._raw_reader.read_range(
                    dataset_type=manifest.dataset_type,
                    instrument_id=instrument_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
            raise NotImplementedError(
                f"Read not implemented for dataset type {manifest.dataset_type}",
            )

    # =========================================================================
    # Validation Operations
    # =========================================================================

    def validate_batch(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],  # DataFrame or list
        strict_mode: bool = False,
    ) -> QualityReport:
        """
        Validate a batch of data against the dataset's contract.

        Performs comprehensive contract validation including type checking,
        null validation, range validation, uniqueness constraints,
        monotonicity checks, and lateness validation.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        data : DataFrame | list
            Data to validate
        strict_mode : bool
            If True, apply stricter validation rules

        Returns
        -------
        QualityReport
            Validation results with quality score and violations

        Examples
        --------
        >>> report = store.validate_batch("bars_eurusd_1m", data_frame)
        >>> print(f"Quality score: {report.quality_score:.2%}")
        >>> if report.violations:
        ...     for violation in report.violations:
        ...         print(f"  - {violation.description}")

        """
        start_time = time.perf_counter()

        # Get manifest and contract
        manifest = self._get_manifest(dataset_id)
        contract = self._get_contract(dataset_id)

        # Convert to DataFrame for validation
        data_frame_obj = self._to_dataframe(data)
        data_frame = cast(DataFrameLike, data_frame_obj)

        # Track violations
        violations: list[ValidationViolation] = []
        failed_record_indices = set()  # Track unique failed record indices
        validation_metadata = {
            "rules_evaluated": len(contract.validation_rules),
            "strict_mode": strict_mode,
        }

        # Apply each validation rule with enhanced checks
        for rule in contract.validation_rules:
            # Skip WARN-only rules in strict mode if configured
            if strict_mode and rule.severity == QualityFlag.WARN:
                # In strict mode, treat warnings as failures
                rule = ValidationRule(
                    rule_type=rule.rule_type,
                    field_name=rule.field_name,
                    parameters=rule.parameters,
                    severity=QualityFlag.FAIL,
                    description=rule.description,
                )

            violation = self._apply_validation_rule(rule, data_frame, manifest)
            if violation:
                violations.append(violation)
                # For simplicity, assume each violation affects unique records up to the total
                # In a real implementation, we'd track actual record indices
                if violation.severity == QualityFlag.FAIL:
                    # Add violation count but cap at total records to avoid overcounting
                    for i in range(min(violation.violation_count, len(data_frame))):
                        failed_record_indices.add(i)

                # Record violation metric
                if HAS_PROMETHEUS:
                    validation_violations_counter.labels(
                        dataset_id=dataset_id,
                        rule_type=str(violation.rule_type.value),
                        severity=str(violation.severity.value),
                    ).inc(violation.violation_count)

        # Calculate quality score
        total_records = len(data_frame)
        failed_records = len(failed_record_indices)
        passed_records = total_records - failed_records
        quality_score = passed_records / total_records if total_records > 0 else 0.0

        # Check quality thresholds
        if contract.quality_thresholds:
            # Calculate metrics
            from ml.common.dataframe_utils import total_nulls as _total_nulls

            data_frame_any = cast(Any, data_frame)
            null_count_total: int = _total_nulls(data_frame_any)

            base_count = (total_records * len(cast(Any, data_frame).columns)) if total_records > 0 else 0
            null_rate = float(null_count_total) / float(base_count) if base_count else 0.0

            if "null_rate" in contract.quality_thresholds:
                if null_rate > contract.quality_thresholds["null_rate"]:
                    violations.append(
                        ValidationViolation(
                            rule_type=ValidationRuleType.NULLABILITY,
                            field_name="*",
                            severity=QualityFlag.WARN,
                            violation_count=int(null_rate * total_records),
                            sample_values=[],
                            description=f"Null rate {null_rate:.2%} exceeds threshold",
                        ),
                    )

        validation_time_ms = (time.perf_counter() - start_time) * 1000

        # Record validation metrics
        if HAS_PROMETHEUS:
            validation_duration_histogram.labels(
                dataset_id=dataset_id,
            ).observe(validation_time_ms / 1000.0)

            quality_score_histogram.labels(
                dataset_id=dataset_id,
            ).observe(quality_score)

        report = QualityReport(
            dataset_id=dataset_id,
            total_records=total_records,
            passed_records=passed_records,
            failed_records=failed_records,
            quality_score=quality_score,
            violations=violations,
            validation_time_ms=validation_time_ms,
            metadata={
                "contract_version": contract.version,
                "enforcement_mode": contract.enforcement_mode,
                "strict_mode": strict_mode,
                **validation_metadata,
            },
        )

        logger.debug(
            "Validated %d records for %s: quality=%.2f, violations=%d",
            total_records,
            dataset_id,
            quality_score,
            len(violations),
        )

        return report

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_manifest(self, dataset_id: str) -> DatasetManifest:
        """
        Get dataset manifest with caching and version check.
        """
        if dataset_id not in self._manifest_cache:
            manifest = self.registry.get_manifest(dataset_id)
            self._manifest_cache[dataset_id] = manifest

            # Check for schema version changes
            if dataset_id in self._schema_migration_state:
                old_version = self._schema_migration_state[dataset_id].get("version")
                if old_version and old_version != manifest.version:
                    logger.info(
                        "Schema version change detected for %s: %s -> %s",
                        dataset_id,
                        old_version,
                        manifest.version,
                    )
                    # Start migration window if configured
                    if self.allow_schema_migration:
                        self._start_migration_window(dataset_id, manifest)

        return self._manifest_cache[dataset_id]

    def _get_contract(self, dataset_id: str) -> DataContract:
        """
        Get data contract with caching and version check.
        """
        if dataset_id not in self._contract_cache:
            contract = self.registry.get_contract(dataset_id)
            self._contract_cache[dataset_id] = contract

            # Log contract version for tracking
            logger.debug(
                "Loaded contract for %s: version=%s, mode=%s, rules=%d",
                dataset_id,
                contract.version,
                contract.enforcement_mode,
                len(contract.validation_rules),
            )

        return self._contract_cache[dataset_id]

    def _to_dataframe(
        self,
        data: DataFrameLike | list[dict[str, Any]],
    ) -> DataFrameLike | list[dict[str, Any]]:
        """
        Convert various data formats to DataFrame-like or pass-through list.
        """
        # Import here to avoid circular dependency
        from ml._imports import HAS_POLARS
        from ml._imports import pl

        if not HAS_POLARS:
            # If Polars not available, work with raw data
            if isinstance(data, list):
                return data
            return data

        # If already a DataFrame, return as is
        if hasattr(data, "columns"):
            return data

        # Convert list of dicts to DataFrame
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # Cast to DataFrameLike for strict typing compliance
            if HAS_POLARS and pl is not None:
                return cast(DataFrameLike, pl.DataFrame(data))
            return data

        # Return as is for other formats
        return data

    def _get_stage_for_dataset_type(self, dataset_type: DatasetType) -> str:
        """
        Map dataset type to processing stage.
        """
        stage_map: dict[DatasetType, str] = {
            DatasetType.BARS: str(Stage.CATALOG_WRITTEN.value),
            DatasetType.TRADES: str(Stage.CATALOG_WRITTEN.value),
            DatasetType.QUOTES: str(Stage.CATALOG_WRITTEN.value),
            DatasetType.MBP1: str(Stage.CATALOG_WRITTEN.value),
            DatasetType.TBBO: str(Stage.CATALOG_WRITTEN.value),
            DatasetType.FEATURES: str(Stage.FEATURE_COMPUTED.value),
            DatasetType.PREDICTIONS: str(Stage.PREDICTION_EMITTED.value),
            DatasetType.SIGNALS: str(Stage.SIGNAL_EMITTED.value),
        }
        return stage_map.get(dataset_type, str(Stage.DATA_INGESTED.value))

    def _coerce_stage_alias(self, stage: Stage | str) -> Stage:
        """
        Normalize loosely defined stage identifiers to canonical ``Stage`` enums.
        """
        try:
            return to_stage_enum(stage)
        except Exception:
            alias = str(stage).strip().lower()
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

    # ------------------------------ event helpers ------------------------------
    def _emit_partial_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: str,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        reason: str,
    ) -> None:
        """
        Emit a partial processing event using centralized helper.
        """
        try:
            stage_enum = self._coerce_stage_alias(stage)
            source_enum = to_source_enum(source)

            # Use centralized helper to ensure correlation_id is attached consistently
            from ml.common.event_emitter import emit_dataset_event as _emit

            _emit(
                self.registry,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage_enum,
                source=source_enum,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=EventStatus.PARTIAL,
                error=None,
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
        stage: str,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        error: str,
    ) -> None:
        """
        Emit a failed processing event using centralized helper.
        """
        try:
            stage_enum = self._coerce_stage_alias(stage)
            source_enum = to_source_enum(source)

            # Use centralized helper to ensure correlation_id is attached consistently
            from ml.common.event_emitter import emit_dataset_event as _emit

            _emit(
                self.registry,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage_enum,
                source=source_enum,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=EventStatus.FAILED,
                error=error,
                metadata=None,
                dataset_type=dataset_id,
                component=self.__class__.__name__,
            )
        except Exception:
            logger.warning("Failed to emit failed event", exc_info=True)

    def _apply_validation_rule(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        """
        Apply a single validation rule to data.
        """
        try:
            if rule.rule_type == ValidationRuleType.TYPE_CHECK:
                return self._validate_types(rule, data_frame, manifest)
            elif rule.rule_type == ValidationRuleType.RANGE:
                return self._validate_range(rule, data_frame)
            elif rule.rule_type == ValidationRuleType.UNIQUENESS:
                return self._validate_uniqueness(rule, data_frame)
            elif rule.rule_type == ValidationRuleType.MONOTONICITY:
                return self._validate_monotonicity(rule, data_frame)
            elif rule.rule_type == ValidationRuleType.NULLABILITY:
                return self._validate_nullability(rule, data_frame)
            elif rule.rule_type == ValidationRuleType.LATENESS:
                return self._validate_lateness(rule, data_frame, manifest)
            elif rule.rule_type == ValidationRuleType.REGEX:
                return self._validate_regex(rule, data_frame)
            else:
                logger.warning("Unknown validation rule type: %s", rule.rule_type)
                return None
        except Exception as exc:
            logger.error(
                "Error applying validation rule %s",
                rule.rule_type,
                exc_info=True,
            )
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=rule.field_name,
                severity=QualityFlag.WARN,
                violation_count=1,
                sample_values=[],
                description=f"Validation error: {exc}",
            )

    def _validate_types(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        data_frame_any = cast(Any, data_frame)
        """
        Validate data types match schema.
        """
        violations = 0
        sample_values = []

        # Check each column in schema
        for col_name, expected_type in manifest.schema.items():
            if hasattr(data_frame_any, "columns") and col_name in data_frame_any.columns:
                # Get actual type
                actual_type = str(data_frame_any[col_name].dtype)

                # Simple type checking (would be more sophisticated in production)
                if not self._types_compatible(actual_type, expected_type):
                    violations += 1
                    sample_values.append(f"{col_name}: {actual_type} != {expected_type}")

        if violations > 0:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=rule.field_name,
                severity=rule.severity,
                violation_count=violations,
                sample_values=sample_values[:5],
                description=f"Type mismatches found in {violations} columns",
            )

        return None

    def _validate_regex(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> ValidationViolation | None:
        """
        Validate a column against a regex pattern.
        """
        import re as _re

        data_frame_any = cast(Any, data_frame)
        field_name = rule.field_name
        params = rule.parameters
        pattern = str(params.get("pattern", ""))
        if not pattern or not hasattr(data_frame_any, "columns") or field_name not in data_frame_any.columns:
            return None

        try:
            regex = _re.compile(pattern)
        except Exception:
            # Treat invalid regex as configuration issue (warn)
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=field_name,
                severity=QualityFlag.WARN,
                violation_count=1,
                sample_values=[pattern],
                description="Invalid regex pattern",
            )

        col = data_frame_any[field_name]
        violations = 0
        samples: list[str] = []

        # Polars series path
        if hasattr(col, "__module__") and "polars" in str(col.__module__):
            try:
                mismatches = [not bool(regex.match(str(v))) for v in col.to_list()]
                violations = int(sum(1 for b in mismatches if b))
                if violations:
                    # collect first 5 mismatches
                    for v, bad in zip(col.to_list(), mismatches):
                        if bad:
                            samples.append(str(v))
                        if len(samples) >= 5:
                            break
            except Exception:
                violations = 0
        else:
            # pandas/iterable path
            try:
                # Convert to iterable of python values
                if hasattr(col, "tolist"):
                    values_iter = cast(Any, col).tolist()
                else:
                    values_iter = list(cast(Any, col))
                for v in values_iter:
                    if not bool(regex.match(str(v))):
                        violations += 1
                        if len(samples) < 5:
                            samples.append(str(v))
            except Exception:
                violations = 0

        if violations > 0:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=field_name,
                severity=rule.severity,
                violation_count=violations,
                sample_values=samples,
                description=f"{field_name} failed regex pattern",
            )
        return None

    def _validate_range(self, rule: ValidationRule, data_frame: object) -> ValidationViolation | None:
        """
        Validate values are within specified range.
        """
        data_frame_any = cast(Any, data_frame)
        field_name = rule.field_name
        params = rule.parameters

        if not hasattr(data_frame_any, "columns") or field_name not in data_frame_any.columns:
            return None

        violations = 0
        sample_values = []

        # Get column values
        col = data_frame_any[field_name]

        # Check min
        if "min" in params:
            min_val = params["min"]

            # Handle both Polars and pandas
            if hasattr(col, "__module__") and "polars" in str(col.__module__):
                # Polars Series
                below_min = col < min_val
                violations += below_min.sum()
                if violations > 0:
                    # Get sample of violating values
                    violating = col.filter(below_min)
                    sample_values.extend(violating.head(5).to_list())
            else:
                # pandas or raw data
                try:
                    below_min = col < min_val
                    if hasattr(below_min, "sum"):
                        violations += below_min.sum()
                        if violations > 0:
                            # Get sample of violating values
                            violating = col[below_min]
                            if hasattr(violating, "head"):
                                sample_values.extend(violating.head(5).to_list())
                except Exception:
                    logger.debug("Failed to compute min-bound violations sample", exc_info=True)

        # Check max
        if "max" in params:
            max_val = params["max"]

            # Handle both Polars and pandas
            if hasattr(col, "__module__") and "polars" in str(col.__module__):
                # Polars Series
                above_max = col > max_val
                violations += above_max.sum()
                if violations > 0 and len(sample_values) < 5:
                    # Get sample of violating values
                    violating = col.filter(above_max)
                    sample_values.extend(violating.head(5 - len(sample_values)).to_list())
            else:
                # pandas or raw data
                try:
                    above_max = col > max_val
                    if hasattr(above_max, "sum"):
                        violations += above_max.sum()
                        if violations > 0 and len(sample_values) < 5:
                            # Get sample of violating values
                            violating = col[above_max]
                            if hasattr(violating, "head"):
                                sample_values.extend(
                                    violating.head(5 - len(sample_values)).to_list(),
                                )
                except Exception:
                    logger.debug("Failed to compute max-bound violations sample", exc_info=True)

        if violations > 0:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=field_name,
                severity=rule.severity,
                violation_count=violations,
                sample_values=sample_values[:5],
                description=f"{field_name} values out of range [{params.get('min', '-inf')}, {params.get('max', 'inf')}]",
            )

        return None

    def _validate_uniqueness(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> ValidationViolation | None:
        """
        Validate uniqueness constraints.
        """
        field_name = rule.field_name
        data_frame_any = cast(Any, data_frame)

        if not hasattr(data_frame_any, "columns"):
            return None

        # Handle composite keys
        if "," in field_name:
            key_fields = [f.strip() for f in field_name.split(",")]
            if all(f in data_frame_any.columns for f in key_fields):
                # Check for duplicates on composite key
                if hasattr(data_frame_any, "is_duplicated"):
                    # Polars
                    duplicates = data_frame_any.select(key_fields).is_duplicated()
                    duplicate_count = duplicates.sum()
                elif hasattr(data_frame_any, "duplicated"):
                    # pandas
                    duplicates = data_frame_any.duplicated(subset=key_fields)
                    duplicate_count = duplicates.sum()
                else:
                    duplicate_count = 0

                if duplicate_count > 0:
                    return ValidationViolation(
                        rule_type=rule.rule_type,
                        field_name=field_name,
                        severity=rule.severity,
                        violation_count=duplicate_count,
                        sample_values=[],
                        description=f"Duplicate values found for composite key {field_name}",
                    )
        else:
            # Single field uniqueness
            if field_name in data_frame_any.columns:
                if hasattr(data_frame_any, "is_duplicated"):
                    # Polars
                    duplicates = data_frame_any.select([field_name]).is_duplicated()
                    duplicate_count = duplicates.sum()
                    sample_values = []
                    if duplicate_count > 0:
                        # Get sample duplicate values
                        duplicate_vals = data_frame_any[field_name].filter(duplicates)
                        sample_values = duplicate_vals.head(5).to_list()
                elif hasattr(data_frame_any, "duplicated"):
                    # pandas
                    duplicates = data_frame_any.duplicated(subset=[field_name])
                    duplicate_count = duplicates.sum()
                    sample_values = []
                    if duplicate_count > 0:
                        # Get sample duplicate values
                        duplicate_vals = data_frame_any[field_name][duplicates]
                        if hasattr(duplicate_vals, "head"):
                            sample_values = duplicate_vals.head(5).to_list()
                else:
                    duplicate_count = 0
                    sample_values = []

                if duplicate_count > 0:
                    return ValidationViolation(
                        rule_type=rule.rule_type,
                        field_name=field_name,
                        severity=rule.severity,
                        violation_count=duplicate_count,
                        sample_values=sample_values,
                        description=f"Duplicate values found in {field_name}",
                    )

        return None

    def _validate_monotonicity(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> ValidationViolation | None:
        """
        Validate monotonic sequences (e.g., timestamps).
        """
        data_frame_any = cast(Any, data_frame)

        field_name = rule.field_name
        params = rule.parameters

        if not hasattr(data_frame_any, "columns") or field_name not in data_frame_any.columns:
            return None

        col = data_frame_any[field_name]
        direction = params.get("direction", "increasing")
        strict = params.get("strict", True)

        violations = 0

        # Handle both Polars and pandas
        if hasattr(col, "__module__") and "polars" in str(col.__module__):
            # Polars Series
            diffs = col.diff()

            if direction == "increasing":
                if strict:
                    # Check for strictly increasing (diff must be > 0)
                    # First element of diff is null, so we drop it
                    violations = (diffs.drop_nulls() <= 0).sum()
                else:
                    # Check for non-decreasing (diff must be >= 0)
                    violations = (diffs.drop_nulls() < 0).sum()
            else:  # decreasing
                if strict:
                    # Check for strictly decreasing (diff must be < 0)
                    violations = (diffs.drop_nulls() >= 0).sum()
                else:
                    # Check for non-increasing (diff must be <= 0)
                    violations = (diffs.drop_nulls() > 0).sum()
        else:
            # pandas or raw data
            if hasattr(col, "diff"):
                diffs = col.diff()

                if direction == "increasing":
                    if strict:
                        # Check for strictly increasing
                        # dropna() to remove the first NaN
                        violations = (cast(Any, diffs.dropna()) <= 0).sum()
                    else:
                        # Check for non-decreasing
                        violations = (cast(Any, diffs.dropna()) < 0).sum()
                else:  # decreasing
                    if strict:
                        # Check for strictly decreasing
                        violations = (cast(Any, diffs.dropna()) >= 0).sum()
                    else:
                        # Check for non-increasing
                        violations = (cast(Any, diffs.dropna()) > 0).sum()

        if violations > 0:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=field_name,
                severity=rule.severity,
                violation_count=violations,
                sample_values=[],
                description=f"{field_name} is not {'strictly ' if strict else ''}{direction}",
            )

        return None

    def _validate_nullability(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> ValidationViolation | None:
        """
        Validate null value constraints.
        """
        data_frame_any = cast(Any, data_frame)
        field_name = rule.field_name
        params = rule.parameters

        if not hasattr(data_frame_any, "columns"):
            return None

        nullable = params.get("nullable", True)

        if not nullable:
            if field_name == "*":
                # Check all fields
                from ml.common.dataframe_utils import column_nulls as _col_nulls
                from ml.common.dataframe_utils import total_nulls as _total

                total_nulls = _total(data_frame_any)
                fields_with_nulls = [col for col in data_frame_any.columns if _col_nulls(data_frame_any, col) > 0]

                if total_nulls > 0:
                    return ValidationViolation(
                        rule_type=rule.rule_type,
                        field_name=field_name,
                        severity=rule.severity,
                        violation_count=total_nulls,
                        sample_values=fields_with_nulls[:5],
                        description=f"Null values found in {len(fields_with_nulls)} fields",
                    )
            else:
                # Check specific field
                if field_name in data_frame_any.columns:
                    from ml.common.dataframe_utils import column_nulls as _col_nulls

                    null_count = _col_nulls(data_frame_any, field_name)

                    if null_count > 0:
                        return ValidationViolation(
                            rule_type=rule.rule_type,
                            field_name=field_name,
                            severity=rule.severity,
                            violation_count=null_count,
                            sample_values=[],
                            description=f"{field_name} contains {null_count} null values",
                        )

        return None

    def _validate_lateness(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        """
        Validate data freshness/lateness.
        """
        data_frame_any = cast(Any, data_frame)
        params = rule.parameters
        max_lateness_ns = params.get("max_lateness_ns", 300_000_000_000)  # Default 5 minutes

        ts_field = manifest.ts_field
        if not hasattr(data_frame_any, "columns") or ts_field not in data_frame_any.columns:
            return None

        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        current_ns = _sanitize(int(time.time_ns()), context="data_store._validate_lateness:now")
        latest_ts = _sanitize(
            int(cast(Any, data_frame)[ts_field].max()),
            context="data_store._validate_lateness:latest",
        )

        lateness_ns = current_ns - latest_ts

        if lateness_ns > max_lateness_ns:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=ts_field,
                severity=rule.severity,
                violation_count=1,
                sample_values=[f"Lateness: {lateness_ns / 1e9:.1f} seconds"],
                description=f"Data is {lateness_ns / 1e9:.1f} seconds late",
            )

        return None

    def _types_compatible(self, actual: str, expected: str) -> bool:
        """
        Check if actual type is compatible with expected type.
        """
        # Simple type compatibility check
        type_map = {
            "int64": ["int", "int64", "i8", "Int64"],
            "float64": ["float", "float64", "f8", "Float64"],
            "str": ["str", "string", "object", "Utf8"],
            "bool": ["bool", "boolean", "Boolean"],
        }

        for expected_base, compatible_types in type_map.items():
            if expected in compatible_types:
                return any(t in actual.lower() for t in compatible_types)

        # Default to string comparison
        return actual.lower() == expected.lower()

    def _format_violations(self, violations: list[ValidationViolation]) -> str:
        """
        Format violations for logging.
        """
        if not violations:
            return "None"

        parts = []
        for v in violations[:3]:  # Show first 3 violations
            parts.append(f"{v.field_name}: {v.description} ({v.violation_count} records)")

        if len(violations) > 3:
            parts.append(f"... and {len(violations) - 3} more")

        return "; ".join(parts)

    def _enforce_quality_report(
        self,
        *,
        dataset_id: str,
        contract: DataContract,
        quality_report: QualityReport,
    ) -> None:
        """
        Apply contract enforcement logic for point-in-time writes.
        """
        if quality_report.quality_score >= 1.0:
            return

        violations_str = self._format_violations(quality_report.violations)
        critical = [v for v in quality_report.violations if v.severity == QualityFlag.FAIL]

        if critical and contract.enforcement_mode != "monitor_only":
            if HAS_PROMETHEUS:
                write_rejection_counter.labels(
                    dataset_id=dataset_id,
                    reason="validation_failed",
                ).inc()
            raise ValueError(
                f"Data validation failed for {dataset_id} (fail-closed). "
                f"Quality score: {quality_report.quality_score:.2f}. "
                f"Critical violations: {len(critical)}. "
                f"Details: {violations_str}",
            )

        if self.fail_on_validation_error and contract.enforcement_mode == "strict":
            if HAS_PROMETHEUS:
                write_rejection_counter.labels(
                    dataset_id=dataset_id,
                    reason="strict_mode_violation",
                ).inc()
            raise ValueError(
                f"Data validation failed for {dataset_id} (strict mode). "
                f"Quality score: {quality_report.quality_score:.2f}. "
                f"Violations: {violations_str}",
            )

        if contract.enforcement_mode == "lenient":
            logger.warning(
                "Data validation warnings for %s (lenient mode): %s",
                dataset_id,
                violations_str,
            )
        else:  # monitor_only or other advisory modes
            logger.info(
                "Data validation issues for %s (monitor-only): %s",
                dataset_id,
                violations_str,
            )

    def _data_frame_to_feature_data(self, data_frame: DataFrameLike, instrument_id: str) -> list[FeatureData]:
        """
        Convert DataFrame to list of FeatureData.
        """
        features = []

        # Generate a default feature_set_id if not present
        feature_set_id = f"features_{instrument_id.lower().replace('/', '_')}"

        # Handle both Polars and pandas-like DataFrames
        if hasattr(data_frame, "iter_rows"):
            # Polars DataFrame
            data_frame_polars = cast(Any, data_frame)
            for row in data_frame_polars.iter_rows(named=True):
                features.append(
                    FeatureData(
                        feature_set_id=feature_set_id,
                        instrument_id=instrument_id,
                        values={
                            str(k): v
                            for k, v in row.items()
                            if k not in ["instrument_id", "ts_event", "ts_init"]
                        },
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        elif hasattr(data_frame, "iterrows"):
            # pandas DataFrame
            for _, row in data_frame.iterrows():
                features.append(
                    FeatureData(
                        feature_set_id=feature_set_id,
                        instrument_id=instrument_id,
                        values={
                            str(k): v
                            for k, v in row.items()
                            if k not in ["instrument_id", "ts_event", "ts_init"]
                        },
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        else:
            # Fallback for list of dicts
            for row in data_frame:
                if isinstance(row, dict):
                    features.append(
                        FeatureData(
                            feature_set_id=feature_set_id,
                            instrument_id=instrument_id,
                            values={
                                str(k): v
                                for k, v in row.items()
                                if k not in ["instrument_id", "ts_event", "ts_init"]
                            },
                            _ts_event=int(row["ts_event"]),
                            _ts_init=int(row.get("ts_init") or row["ts_event"]),
                        ),
                    )

        return features

    def _data_frame_to_predictions(self, data_frame: DataFrameLike | list[dict[str, Any]]) -> list[ModelPrediction]:
        """
        Convert DataFrame to list of ModelPrediction.
        """
        predictions = []

        # Handle both Polars and pandas-like DataFrames
        if hasattr(data_frame, "iter_rows"):
            # Polars DataFrame
            data_frame_polars = cast(Any, data_frame)
            for row in data_frame_polars.iter_rows(named=True):
                predictions.append(
                    ModelPrediction(
                        model_id=row["model_id"],
                        instrument_id=row["instrument_id"],
                        prediction=row["prediction"],
                        confidence=row.get("confidence", 0.0),
                        features_used=row.get("features_used", {}),
                        inference_time_ms=row.get("inference_time_ms", 0.0),
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        elif hasattr(data_frame, "iterrows"):
            # pandas DataFrame
            for _, row in data_frame.iterrows():
                predictions.append(
                    ModelPrediction(
                        model_id=row["model_id"],
                        instrument_id=row["instrument_id"],
                        prediction=row["prediction"],
                        confidence=row.get("confidence", 0.0),
                        features_used=row.get("features_used", {}),
                        inference_time_ms=row.get("inference_time_ms", 0.0),
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        else:
            # Fallback for list of dicts
            for row in data_frame:
                if isinstance(row, dict):
                    predictions.append(
                        ModelPrediction(
                            model_id=row["model_id"],
                            instrument_id=row["instrument_id"],
                            prediction=row["prediction"],
                            confidence=row.get("confidence", 0.0),
                            features_used=row.get("features_used", {}),
                            inference_time_ms=row.get("inference_time_ms", 0.0),
                            _ts_event=int(row["ts_event"]),
                            _ts_init=int(row.get("ts_init") or row["ts_event"]),
                        ),
                    )

        return predictions

    def _data_frame_to_signals(self, data_frame: DataFrameLike | list[dict[str, Any]]) -> list[StrategySignal]:
        """
        Convert DataFrame to list of StrategySignal.
        """
        signals = []

        # Handle both Polars and pandas-like DataFrames
        if hasattr(data_frame, "iter_rows"):
            # Polars DataFrame
            data_frame_polars = cast(Any, data_frame)
            for row in data_frame_polars.iter_rows(named=True):
                signals.append(
                    StrategySignal(
                        strategy_id=row["strategy_id"],
                        instrument_id=row["instrument_id"],
                        signal_type=row["signal_type"],
                        strength=float(row.get("strength", row.get("signal_value", 0.0))),
                        model_predictions=row.get("model_predictions", {}),
                        risk_metrics=row.get("risk_metrics", {}),
                        execution_params=row.get("execution_params", {}),
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        elif hasattr(data_frame, "iterrows"):
            # pandas DataFrame
            for _, row in data_frame.iterrows():
                signals.append(
                    StrategySignal(
                        strategy_id=row["strategy_id"],
                        instrument_id=row["instrument_id"],
                        signal_type=row["signal_type"],
                        strength=float(row.get("strength", row.get("signal_value", 0.0))),
                        model_predictions=row.get("model_predictions", {}),
                        risk_metrics=row.get("risk_metrics", {}),
                        execution_params=row.get("execution_params", {}),
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        else:
            # Fallback for list of dicts
            for row in data_frame:
                if isinstance(row, dict):
                    signals.append(
                        StrategySignal(
                            strategy_id=row["strategy_id"],
                            instrument_id=row["instrument_id"],
                            signal_type=row["signal_type"],
                            strength=float(
                                row.get("strength", row.get("signal_value", 0.0)) or 0.0,
                            ),
                            model_predictions=row.get("model_predictions", {}),
                            risk_metrics=row.get("risk_metrics", {}),
                            execution_params=row.get("execution_params", {}),
                            _ts_event=int(row["ts_event"]),
                            _ts_init=int(row.get("ts_init") or row["ts_event"]),
                        ),
                    )

        return signals

    # ------------------------------------------------------------------
    # Legacy compatibility helpers (used by older unit tests)
    # ------------------------------------------------------------------

    def _df_to_predictions(
        self,
        data_frame: DataFrameLike | list[dict[str, Any]],
    ) -> list[ModelPrediction]:
        """
        Backwards-compatible alias for `_data_frame_to_predictions`.

        Historically the unit tests referenced `_df_to_predictions`; retain the typed
        wrapper so existing imports continue to work.
        """
        return self._data_frame_to_predictions(data_frame)

    def _df_to_signals(
        self,
        data_frame: DataFrameLike | list[dict[str, Any]],
    ) -> list[StrategySignal]:
        """
        Backwards-compatible alias for `_data_frame_to_signals`.
        """
        return self._data_frame_to_signals(data_frame)

    def _ensure_dataset_registered(
        self,
        dataset_id: str,
        dataset_type: DatasetType,
        instrument_id: str,
    ) -> None:
        """
        Ensure dataset is registered in the registry.
        """
        try:
            self.registry.get_manifest(dataset_id)
        except ValueError:
            location = f"ml_{dataset_type.value}"
            manifest = build_auto_dataset_manifest(
                dataset_id=dataset_id,
                dataset_type=dataset_type,
                location=location,
                storage_kind=StorageKind.POSTGRES,
                pipeline_signature="data_store_auto",
                retention_days=365,
                metadata={
                    "auto_registered": True,
                    "instrument_id": instrument_id,
                    "storage_table": location,
                    "source": "data_store",
                },
            )

            self.registry.register_dataset(manifest)
            logger.info("Auto-registered dataset %s", dataset_id)

    def _compute_schema_hash(self, data_frame: DataFrameLike, manifest: DatasetManifest) -> str:
        """
        Compute schema hash for the actual data.
        """
        data_frame_any = cast(Any, data_frame)
        if not hasattr(data_frame_any, "columns"):
            # For non-DataFrame data, use manifest hash
            return manifest.schema_hash

        # Build schema dict from actual data
        actual_schema: dict[str, str] = {}
        for col in data_frame_any.columns:
            if col in manifest.schema:
                actual_schema[col] = manifest.schema[col]
            else:
                dtype = str(data_frame_any[col].dtype)
                lower = dtype.lower()
                if "int" in lower:
                    actual_schema[col] = "int64"
                elif "float" in lower:
                    actual_schema[col] = "float64"
                elif "bool" in lower:
                    actual_schema[col] = "bool"
                else:
                    actual_schema[col] = "str"

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
        """
        Check if dataset is in schema migration window.
        """
        if not self.allow_schema_migration:
            return False

        if dataset_id not in self._schema_migration_state:
            return False

        migration_info = self._schema_migration_state[dataset_id]
        migration_start = migration_info.get("start_time", 0)
        window_ns = self.schema_migration_window_hours * 3600 * 1e9

        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize2

        current_time = _sanitize2(
            int(time.time_ns()),
            context="data_store._is_in_migration_window:now",
        )
        if current_time - migration_start < window_ns:
            return True
        else:
            # Migration window expired, clear state
            del self._schema_migration_state[dataset_id]
            logger.info("Schema migration window expired for %s", dataset_id)
            return False

    def _start_migration_window(self, dataset_id: str, manifest: DatasetManifest) -> None:
        """
        Start a schema migration window for dual-write.
        """
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

        # Record migration start metric
        if HAS_PROMETHEUS:
            schema_mismatch_counter.labels(
                dataset=dataset_id,
                mismatch_type="migration_started",
            ).inc()

    def _update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: str | Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        """
        Internal hook to update the registry watermark (patchable in tests).
        """
        src_enum = to_source_enum(source)
        self.registry.update_watermark(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            source=src_enum,
            last_success_ns=last_success_ns,
            count=count,
            completeness_pct=completeness_pct,
        )

    def _begin_transaction(
        self,
    ) -> AbstractContextManager[object]:  # pragma: no cover (test hook for patching)
        """
        Return a no-op context manager for tests to patch.
        """
        from contextlib import nullcontext

        return nullcontext()
