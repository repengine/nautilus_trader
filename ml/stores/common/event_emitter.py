#!/usr/bin/env python3

"""
Event emitter component for DataStore.

Extracted from DataStore (Phase 2.4.4). Provides event emission and message
bus integration for dataset processing operations.

ALL methods are COLD path (async operations acceptable, non-blocking).

"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ml._imports import HAS_PROMETHEUS
from ml.common.correlation import make_correlation_id
from ml.common.message_topics import build_topic_for_stage
from ml.common.metrics_bootstrap import get_counter
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage


if TYPE_CHECKING:
    from ml.common.message_bus import MessagePublisherProtocol as MessagePublisher
    from ml.registry.protocols import RegistryProtocol

logger = logging.getLogger(__name__)


# Get metrics via bootstrap (returns dummy metrics if Prometheus unavailable)
event_emission_counter = get_counter(
    "ml_datastore_event_emissions_total",
    "Total number of events emitted by DataStore",
    labelnames=["event_type", "status"],
)
event_emission_errors = get_counter(
    "ml_datastore_event_emission_errors_total",
    "Total number of event emission errors",
    labelnames=["event_type", "error_type"],
)


# =========================================================================
# EventEmitterComponent
# =========================================================================


class EventEmitterComponent:
    """
    Event emission and message bus integration for DataStore.

    Extracted from DataStore (Phase 2.4.4).
    All methods are COLD path (async operations acceptable, non-blocking).

    Provides:
    - Generic event emission to registry
    - Dataset-specific event emission
    - Partial success event emission
    - Failure event emission
    - Message bus publishing (non-blocking)

    CRITICAL: All event emission must be NON-BLOCKING. Wrap in try/except
    to ensure failures don't crash data operations.

    Example
    -------
    >>> from ml.stores.common.event_emitter import EventEmitterComponent
    >>> emitter = EventEmitterComponent(
    ...     registry=registry,
    ...     publisher=publisher,
    ...     enable_publishing=True,
    ...     topic_scheme="hierarchical",
    ...     topic_prefix="ml",
    ... )
    >>> emitter.emit_dataset_event(
    ...     dataset_id="bars_eurusd_1m",
    ...     status=EventStatus.SUCCESS,
    ...     metadata={"quality_score": 1.0},
    ... )

    """

    def __init__(
        self,
        registry: RegistryProtocol,
        publisher: MessagePublisher | None = None,
        *,
        enable_publishing: bool = True,
        topic_scheme: str = "hierarchical",
        topic_prefix: str = "ml",
    ) -> None:
        """
        Initialize event emitter with registry and message bus dependencies.

        Args:
            registry: Data registry for event persistence
            publisher: Optional message bus publisher
            enable_publishing: If True, publish events to message bus
            topic_scheme: Topic naming scheme (hierarchical or flat)
            topic_prefix: Topic prefix for message bus

        """
        self._registry = registry
        self._publisher = publisher
        self._enable_publishing = enable_publishing
        self._topic_scheme = topic_scheme
        self._topic_prefix = topic_prefix

    # =========================================================================
    # Public API - All COLD PATH (non-blocking)
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
        Emit a dataset processing event via centralized registry.

        EXTRACTED FROM: ml/stores/data_store.py:783
        COLD PATH: Event emission is async, non-blocking

        This method emits events to the data registry for tracking dataset
        operations and optionally publishes to the message bus for downstream
        consumers.

        CRITICAL: Non-blocking - failures are logged but don't raise.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        stage : Stage | str
            Processing stage (DATA_INGESTED, FEATURES_COMPUTED, etc.)
        source : Source | str
            Event source (LIVE, HISTORICAL, BACKFILL)
        run_id : str
            Unique identifier for this processing run
        ts_min : int
            Minimum timestamp (ns) for covered data
        ts_max : int
            Maximum timestamp (ns) for covered data
        count : int
            Number of records processed
        status : str
            Status string (EventStatus.value: "success", "failed", "partial")
        error : str | None
            Error message if status is failed
        metadata : dict[str, Any] | None
            Additional metadata to attach to the event

        Examples
        --------
        >>> emitter.emit_event(
        ...     dataset_id="bars_eurusd_1m",
        ...     instrument_id="EURUSD.SIM",
        ...     stage=Stage.DATA_INGESTED,
        ...     source=Source.HISTORICAL,
        ...     run_id="run_20240101_120000",
        ...     ts_min=1699999900000000000,
        ...     ts_max=1699999990000000000,
        ...     count=100,
        ...     status="success",
        ... )

        """
        # Normalize inputs (be tolerant of unknown sources by defaulting to LIVE)
        try:
            stage_enum = stage if isinstance(stage, Stage) else Stage[str(stage)]
        except (KeyError, ValueError):
            stage_enum = Stage.DATA_INGESTED
        try:
            source_enum = Source(source) if not isinstance(source, Source) else source
        except Exception:
            source_enum = Source.LIVE
        status_enum = EventStatus(status) if not isinstance(status, EventStatus) else status

        # Build correlation_id and merged metadata
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

        # Emit to registry (robust against tests that monkeypatch the helper)
        try:
            self._registry.emit_event(
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

            # Record success metric
            if HAS_PROMETHEUS:
                event_emission_counter.labels(
                    event_type="emit_event",
                    status=status_enum.value,
                ).inc()

        except TypeError:
            # Backwards-compatible registries may not accept metadata
            try:
                self._registry.emit_event(
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

                # Record success metric
                if HAS_PROMETHEUS:
                    event_emission_counter.labels(
                        event_type="emit_event",
                        status=status_enum.value,
                    ).inc()

            except Exception as exc:
                # Non-blocking: log and continue
                logger.warning(
                    "Registry event emission failed for %s: %s",
                    dataset_id,
                    exc,
                    exc_info=True,
                )
                if HAS_PROMETHEUS:
                    event_emission_errors.labels(
                        event_type="emit_event",
                        error_type="registry_error",
                    ).inc()

        except Exception as exc:
            # Non-blocking: log and continue
            logger.warning(
                "Registry event emission failed for %s: %s",
                dataset_id,
                exc,
                exc_info=True,
            )
            if HAS_PROMETHEUS:
                event_emission_errors.labels(
                    event_type="emit_event",
                    error_type="registry_error",
                ).inc()

        # Optionally publish to message bus (respects _enable_publishing flag)
        if self._enable_publishing and self._publisher is not None:
            try:
                from ml.common.events_util import build_bus_payload

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
                self._publisher.publish(topic, payload)

                logger.debug(
                    "Published event to topic %s: %s/%s (count=%d, status=%s)",
                    topic,
                    dataset_id,
                    instrument_id,
                    count,
                    status_enum.value,
                )

            except Exception as exc:
                # Non-blocking: log and continue
                logger.warning(
                    "Message bus publish failed for %s: %s",
                    dataset_id,
                    exc,
                    exc_info=True,
                )
                if HAS_PROMETHEUS:
                    event_emission_errors.labels(
                        event_type="emit_event",
                        error_type="bus_publish_error",
                    ).inc()

    def emit_dataset_event(
        self,
        *,
        dataset_id: str,
        status: EventStatus | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit a dataset-specific event with simplified parameter set.

        EXTRACTED FROM: ml/stores/data_store.py:910
        COLD PATH: Event emission is async, non-blocking

        This is a lightweight wrapper around emit_event() for scenarios where
        only dataset_id, status, and metadata are needed. Used primarily by
        unit tests and simple event tracking.

        CRITICAL: Non-blocking - failures are logged but don't raise.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        status : EventStatus | str
            Event status (SUCCESS, FAILED, PARTIAL)
        metadata : dict[str, Any] | None
            Additional event metadata

        Examples
        --------
        >>> emitter.emit_dataset_event(
        ...     dataset_id="bars_eurusd_1m",
        ...     status=EventStatus.SUCCESS,
        ...     metadata={"quality_score": 1.0},
        ... )

        """
        # Normalize status
        status_enum = status if isinstance(status, EventStatus) else EventStatus(status)

        # Build minimal event with defaults
        current_ns = time.time_ns()
        run_id = f"dataset_event_{dataset_id}_{current_ns}"

        # Use emit_event with default values
        try:
            self.emit_event(
                dataset_id=dataset_id,
                instrument_id="UNKNOWN",
                stage=Stage.DATA_INGESTED,
                source=Source.LIVE,
                run_id=run_id,
                ts_min=current_ns,
                ts_max=current_ns,
                count=0,
                status=status_enum.value,
                error=None,
                metadata=metadata,
            )

            # Record success metric
            if HAS_PROMETHEUS:
                event_emission_counter.labels(
                    event_type="emit_dataset_event",
                    status=status_enum.value,
                ).inc()

        except Exception as exc:
            # Non-blocking: log and continue
            logger.warning(
                "Dataset event emission failed for %s: %s",
                dataset_id,
                exc,
                exc_info=True,
            )
            if HAS_PROMETHEUS:
                event_emission_errors.labels(
                    event_type="emit_dataset_event",
                    error_type="emission_error",
                ).inc()

    def _emit_partial_event(
        self,
        *,
        operation: str,
        details: dict[str, Any],
    ) -> None:
        """
        Emit a partial success event for incomplete operations.

        EXTRACTED FROM: ml/stores/data_store.py:2661
        COLD PATH: Event emission is async, non-blocking

        Used when an operation partially succeeds (e.g., some records written,
        others failed validation). Emits event with PARTIAL status and includes
        details about what succeeded/failed.

        CRITICAL: Non-blocking - failures are logged but don't raise.

        Parameters
        ----------
        operation : str
            Operation name (write_ingestion, write_features, etc.)
        details : dict[str, Any]
            Details about the partial success (records_written, records_failed, etc.)

        Examples
        --------
        >>> emitter._emit_partial_event(
        ...     operation="write_ingestion",
        ...     details={
        ...         "dataset_id": "bars_eurusd_1m",
        ...         "records_written": 90,
        ...         "records_failed": 10,
        ...         "reason": "validation_failed",
        ...     },
        ... )

        """
        try:
            # Extract required fields from details
            dataset_id = details.get("dataset_id", "UNKNOWN")
            instrument_id = details.get("instrument_id", "UNKNOWN")
            run_id = details.get("run_id", f"partial_{operation}_{time.time_ns()}")
            ts_min = details.get("ts_min", time.time_ns())
            ts_max = details.get("ts_max", time.time_ns())
            count = details.get("records_written", 0)

            # Emit event with PARTIAL status
            self.emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=Stage.DATA_INGESTED,
                source=Source.LIVE,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=EventStatus.PARTIAL.value,
                error=None,
                metadata={
                    "operation": operation,
                    "reason": details.get("reason", "partial_success"),
                    **details,
                },
            )

            # Record metric
            if HAS_PROMETHEUS:
                event_emission_counter.labels(
                    event_type="_emit_partial_event",
                    status="partial",
                ).inc()

            logger.debug(
                "Emitted partial event for %s: %s",
                operation,
                details.get("reason", "partial_success"),
            )

        except Exception as exc:
            # Non-blocking: log and continue
            logger.warning(
                "Partial event emission failed for %s: %s",
                operation,
                exc,
                exc_info=True,
            )
            if HAS_PROMETHEUS:
                event_emission_errors.labels(
                    event_type="_emit_partial_event",
                    error_type="emission_error",
                ).inc()

    def _emit_failed_event(
        self,
        *,
        operation: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """
        Emit a failure event for operations that completely failed.

        EXTRACTED FROM: ml/stores/data_store.py:2704
        COLD PATH: Event emission is async, non-blocking

        Used when an operation completely fails (e.g., database connection lost,
        validation failed before any writes). Emits event with FAILED status and
        includes error details.

        CRITICAL: Non-blocking - failures are logged but don't raise.

        Parameters
        ----------
        operation : str
            Operation name (write_ingestion, write_features, etc.)
        error : Exception
            Exception that caused the failure
        context : dict[str, Any]
            Context about the failed operation (dataset_id, instrument_id, etc.)

        Examples
        --------
        >>> try:
        ...     writer.write_ingestion(...)
        ... except Exception as exc:
        ...     emitter._emit_failed_event(
        ...         operation="write_ingestion",
        ...         error=exc,
        ...         context={
        ...             "dataset_id": "bars_eurusd_1m",
        ...             "instrument_id": "EURUSD.SIM",
        ...             "run_id": "run_20240101_120000",
        ...         },
        ...     )

        """
        try:
            # Extract required fields from context
            dataset_id = context.get("dataset_id", "UNKNOWN")
            instrument_id = context.get("instrument_id", "UNKNOWN")
            run_id = context.get("run_id", f"failed_{operation}_{time.time_ns()}")
            ts_min = context.get("ts_min", time.time_ns())
            ts_max = context.get("ts_max", time.time_ns())
            count = context.get("count", 0)

            # Format error message (handle nested exceptions)
            error_message = str(error)
            if hasattr(error, "__cause__") and error.__cause__ is not None:
                error_message = f"{error_message} (caused by: {error.__cause__})"

            # Emit event with FAILED status
            self.emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=Stage.DATA_INGESTED,
                source=Source.LIVE,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=EventStatus.FAILED.value,
                error=error_message,
                metadata={
                    "operation": operation,
                    "error_type": type(error).__name__,
                    **context,
                },
            )

            # Record metric
            if HAS_PROMETHEUS:
                event_emission_counter.labels(
                    event_type="_emit_failed_event",
                    status="failed",
                ).inc()

            logger.debug(
                "Emitted failed event for %s: %s",
                operation,
                error_message,
            )

        except Exception as exc:
            # Non-blocking: log and continue
            logger.warning(
                "Failed event emission failed for %s: %s",
                operation,
                exc,
                exc_info=True,
            )
            if HAS_PROMETHEUS:
                event_emission_errors.labels(
                    event_type="_emit_failed_event",
                    error_type="emission_error",
                ).inc()
