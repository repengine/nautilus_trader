#!/usr/bin/env python3

"""
Event management component for DataRegistry.

This module handles dataset lifecycle event emission including creation, updates,
deletions, and processing events.

"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import text

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage


if TYPE_CHECKING:
    from ml.registry.persistence import PersistenceManager


logger = logging.getLogger(__name__)


class EventManagerProtocol(Protocol):
    """
    Protocol for event management operations.

    This protocol defines the interface for dataset lifecycle event emission.

    """

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
        error: str | None,
        metadata: dict[str, object] | None,
        persistence: PersistenceManager,
    ) -> None:
        """
        Emit a data processing event.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        stage : Stage
            Processing stage enum
        source : Source
            Data source enum
        run_id : str
            Unique identifier for this processing run
        ts_min : int
            Minimum timestamp in nanoseconds
        ts_max : int
            Maximum timestamp in nanoseconds
        count : int
            Number of records processed
        status : EventStatus
            Processing status enum
        error : str | None
            Error message if status is failed
        metadata : dict[str, object] | None
            Additional event metadata
        persistence : PersistenceManager
            Persistence manager for backend operations

        """
        ...


class EventManager:
    """
    Manages dataset lifecycle event operations.

    This component handles event emission for dataset lifecycle operations including
    creation, updates, processing events, and deletions. All operations are coordinated
    by the parent DataRegistry for thread safety.

    Examples
    --------
    >>> manager = EventManager()
    >>> from ml.config.events import Stage, Source, EventStatus
    >>> manager.emit_event(
    ...     dataset_id="bars_eurusd_1m",
    ...     instrument_id="EUR/USD",
    ...     stage=Stage.CATALOG_WRITTEN,
    ...     source=Source.HISTORICAL,
    ...     run_id="run_123",
    ...     ts_min=1234567890000000000,
    ...     ts_max=1234567900000000000,
    ...     count=1000,
    ...     status=EventStatus.SUCCESS,
    ...     error=None,
    ...     metadata=None,
    ...     persistence=persistence,
    ... )

    """

    def __init__(self) -> None:
        """Initialize event manager with empty storage."""
        self._events: list[dict[str, Any]] = []

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
        persistence: PersistenceManager | None = None,
    ) -> None:
        """
        Emit a data processing event.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        stage : Stage
            Processing stage enum
        source : Source
            Data source enum
        run_id : str
            Unique identifier for this processing run
        ts_min : int
            Minimum timestamp in nanoseconds
        ts_max : int
            Maximum timestamp in nanoseconds
        count : int
            Number of records processed
        status : EventStatus
            Processing status enum
        error : str | None
            Error message if status is failed
        metadata : dict[str, object] | None
            Additional event metadata
        persistence : PersistenceManager | None
            Persistence manager for backend operations

        Examples
        --------
        >>> from ml.config.events import Stage, Source, EventStatus
        >>> manager.emit_event(
        ...     dataset_id="bars_eurusd_1m",
        ...     instrument_id="EUR/USD",
        ...     stage=Stage.CATALOG_WRITTEN,
        ...     source=Source.HISTORICAL,
        ...     run_id="run_123",
        ...     ts_min=1234567890000000000,
        ...     ts_max=1234567900000000000,
        ...     count=1000,
        ...     status=EventStatus.SUCCESS,
        ...     error=None,
        ...     metadata=None,
        ...     persistence=persistence,
        ... )

        """
        if persistence is None:
            return

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

        if persistence.config.backend.value == "json":
            self._events.append(event)

            # Trim old events to prevent unbounded growth (keep last 10000)
            if len(self._events) > 10000:
                self._events = self._events[-10000:]

        elif persistence.config.backend.value == "postgres":
            session = persistence.get_session()
            if session is None:
                raise RuntimeError("Failed to get database session")

            try:
                # Prefer extended function with metadata if available; else fallback
                try:
                    query_ext = text(
                        """
                        SELECT emit_data_event_ext(
                            :dataset_id, :instrument_id, :stage, :source, :run_id,
                            :ts_min, :ts_max, :count, :status, :error, :metadata
                        )
                    """,
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
                    query = text(
                        """
                        SELECT emit_data_event(
                            :dataset_id, :instrument_id, :stage, :source, :run_id,
                            :ts_min, :ts_max, :count, :status, :error
                        )
                    """,
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
                    fallback_insert = text(
                        """
                        INSERT INTO ml_data_events (
                            dataset_id, instrument_id, stage, source, run_id,
                            ts_min, ts_max, ts_event, count, status, error, created_at
                        ) VALUES (
                            :dataset_id, :instrument_id, :stage, :source, :run_id,
                            :ts_min, :ts_max, :ts_event, :count, :status, :error, NOW()
                        )
                        """,
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
                    logger.error("Failed to emit event: %s", e2)
                    raise
                finally:
                    session.close()

        logger.debug(
            "Emitted event: dataset=%s, instrument=%s, stage=%s, status=%s, count=%d",
            dataset_id,
            instrument_id,
            stage_val,
            status_val,
            count,
        )


__all__ = [
    "EventManager",
    "EventManagerProtocol",
]
