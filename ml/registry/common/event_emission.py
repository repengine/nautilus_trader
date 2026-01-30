#!/usr/bin/env python3

"""
EventEmissionComponent - Handles data processing event emission.

This component is extracted from the DataRegistry god class following the
established TDD decomposition pattern. It handles:
- Event emission and recording
- Event trimming (JSON backend)
- PostgreSQL event persistence via SQL functions

Thread-safety: Uses the persistence component's lock for thread-safe operations.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.common.sql_utils import set_instrumentation_search_path
from ml.registry.persistence import BackendType


if TYPE_CHECKING:
    from ml.registry.common.data_persistence import DataPersistenceComponent


logger = logging.getLogger(__name__)


class EventEmissionComponent:
    """
    Handles data processing event emission.

    This component manages the emission of data processing events,
    recording them either in JSON files or PostgreSQL database.

    Attributes
    ----------
    persistence : DataPersistenceComponent
        The persistence component for storage operations.

    Thread Safety
    -------------
    All operations use the persistence component's lock for thread-safe access.

    Example
    -------
    >>> from ml.registry.common.data_persistence import DataPersistenceComponent
    >>> persistence = DataPersistenceComponent(config, path)
    >>> emitter = EventEmissionComponent(persistence)
    >>> emitter.emit_event(
    ...     dataset_id="my_dataset",
    ...     instrument_id="EUR/USD",
    ...     stage=Stage.CATALOG_WRITTEN,
    ...     source=Source.HISTORICAL,
    ...     run_id="run_123",
    ...     ts_min=1000000000,
    ...     ts_max=2000000000,
    ...     count=100,
    ...     status=EventStatus.SUCCESS,
    ... )
    """

    def __init__(self, persistence: DataPersistenceComponent) -> None:
        """
        Initialize event emission component.

        Parameters
        ----------
        persistence : DataPersistenceComponent
            The persistence component for storage operations.
        """
        self._persistence = persistence

    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """
        Emit a data processing event.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        instrument_id : str
            Instrument identifier.
        stage : Stage
            Processing stage (INGESTED, CATALOG_WRITTEN, FEATURE_COMPUTED, etc.).
        source : Source
            Data source (live, historical, backfill).
        run_id : str
            Unique identifier for this processing run.
        ts_min : int
            Minimum timestamp in nanoseconds.
        ts_max : int
            Maximum timestamp in nanoseconds.
        count : int
            Number of records processed.
        status : EventStatus
            Processing status (success, failed, partial).
        error : str | None
            Error message if status is failed.
        metadata : dict[str, object] | None
            Additional metadata for the event.

        Raises
        ------
        RuntimeError
            If database session cannot be obtained (PostgreSQL backend).

        Example
        -------
        >>> emitter.emit_event(
        ...     dataset_id="bars_eurusd_1m",
        ...     instrument_id="EUR/USD",
        ...     stage=Stage.CATALOG_WRITTEN,
        ...     source=Source.HISTORICAL,
        ...     run_id="run_123",
        ...     ts_min=1234567890000000000,
        ...     ts_max=1234567900000000000,
        ...     count=1000,
        ...     status=EventStatus.SUCCESS,
        ... )
        """
        with self._persistence._lock:
            # Anchor event time within the provided window (start of period)
            ts_event = ts_min

            # Normalize enums to their persisted string values
            stage_val = stage.value
            source_val = source.value
            status_val = status.value

            event = {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "stage": stage_val,
                "source": source_val,
                "run_id": run_id,
                "ts_min": ts_min,
                "ts_max": ts_max,
                "ts_event": ts_event,
                "count": count,
                "status": status_val,
                "error": error,
                "created_at": time.time(),
                "metadata": metadata or {},
            }

            if self._persistence.backend == BackendType.JSON:
                self._emit_event_json(event)
            elif self._persistence.backend == BackendType.POSTGRES:
                self._emit_event_postgres(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    stage_val=stage_val,
                    source_val=source_val,
                    run_id=run_id,
                    ts_min=ts_min,
                    ts_max=ts_max,
                    ts_event=ts_event,
                    count=count,
                    status_val=status_val,
                    error=error,
                    metadata=metadata,
                )

            logger.debug(
                "Emitted event: dataset=%s, instrument=%s, stage=%s, status=%s, count=%d",
                dataset_id,
                instrument_id,
                stage_val,
                status_val,
                count,
            )

    def _emit_event_json(self, event: dict[str, Any]) -> None:
        """
        Emit event to JSON backend.

        Parameters
        ----------
        event : dict[str, Any]
            Event data to store.
        """
        self._persistence._events.append(event)

        # Trim old events to prevent unbounded growth (keep last 10000)
        if len(self._persistence._events) > 10000:
            self._persistence._events = self._persistence._events[-10000:]

        # Tests expect persistence immediately after emit_event
        self._persistence._save_registry(immediate=True)

    def _emit_event_postgres(
        self,
        dataset_id: str,
        instrument_id: str,
        stage_val: str,
        source_val: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        ts_event: int,
        count: int,
        status_val: str,
        error: str | None,
        metadata: dict[str, object] | None,
    ) -> None:
        """
        Emit event to PostgreSQL backend.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        instrument_id : str
            Instrument identifier.
        stage_val : str
            Stage value as string.
        source_val : str
            Source value as string.
        run_id : str
            Run identifier.
        ts_min : int
            Minimum timestamp.
        ts_max : int
            Maximum timestamp.
        ts_event : int
            Event timestamp.
        count : int
            Record count.
        status_val : str
            Status value as string.
        error : str | None
            Error message.
        metadata : dict[str, object] | None
            Additional metadata.

        Raises
        ------
        RuntimeError
            If database session cannot be obtained.
        """
        session = self._persistence.persistence.get_session()
        if session is None:
            raise RuntimeError("Failed to get database session")

        try:
            set_instrumentation_search_path(session)
            # Prefer extended function with metadata if available; else fallback
            try:
                query_ext = text(
                    """
                    SELECT emit_data_event_ext(
                        :dataset_id, :instrument_id, :stage, :source, :run_id,
                        :ts_min, :ts_max, :count, :status, :error, :metadata
                    )
                """
                )
                session.execute(
                    query_ext,
                    {
                        "dataset_id": dataset_id,
                        "instrument_id": instrument_id,
                        "stage": stage_val,
                        "source": source_val,
                        "run_id": run_id,
                        "ts_min": ts_min,
                        "ts_max": ts_max,
                        "count": count,
                        "status": status_val,
                        "error": error,
                        "metadata": json.dumps(metadata or {}),
                    },
                )
            except Exception:
                # Clear the failed transaction before fallback
                session.rollback()
                set_instrumentation_search_path(session)
                query = text(
                    """
                    SELECT emit_data_event(
                        :dataset_id, :instrument_id, :stage, :source, :run_id,
                        :ts_min, :ts_max, :count, :status, :error
                    )
                """
                )
                session.execute(
                    query,
                    {
                        "dataset_id": dataset_id,
                        "instrument_id": instrument_id,
                        "stage": stage_val,
                        "source": source_val,
                        "run_id": run_id,
                        "ts_min": ts_min,
                        "ts_max": ts_max,
                        "count": count,
                        "status": status_val,
                        "error": error,
                    },
                )
            session.commit()

        except Exception:
            # Fallback: if SQL function path fails, attempt direct insert
            session.rollback()
            try:
                set_instrumentation_search_path(session)
                fallback_insert = text(
                    """
                    INSERT INTO ml_data_events (
                        dataset_id, instrument_id, stage, source, run_id,
                        ts_min, ts_max, ts_event, count, status, error, created_at
                    ) VALUES (
                        :dataset_id, :instrument_id, :stage, :source, :run_id,
                        :ts_min, :ts_max, :ts_event, :count, :status, :error, NOW()
                    )
                    """
                )
                session.execute(
                    fallback_insert,
                    {
                        "dataset_id": dataset_id,
                        "instrument_id": instrument_id,
                        "stage": stage_val,
                        "source": source_val,
                        "run_id": run_id,
                        "ts_min": ts_min,
                        "ts_max": ts_max,
                        "ts_event": ts_event,
                        "count": count,
                        "status": status_val,
                        "error": error,
                    },
                )
                session.commit()
            except Exception as e2:
                session.rollback()
                logger.error("Failed to emit event: %s", e2, exc_info=True)
                raise
            finally:
                session.close()
