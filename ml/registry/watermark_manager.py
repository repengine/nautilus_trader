#!/usr/bin/env python3

"""
Watermark management component for DataRegistry.

This module handles dataset watermark operations for tracking data processing progress
and supporting incremental data processing.

"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, overload

from sqlalchemy import text

from ml.config.events import Source


if TYPE_CHECKING:
    from ml.registry.persistence import PersistenceManager


logger = logging.getLogger(__name__)


# Watermark dataclass for tracking data processing progress
@dataclass(frozen=True)
class Watermark:
    """
    Watermark tracking data processing progress for a dataset.

    Attributes
    ----------
    dataset_id : str
        Dataset identifier
    instrument_id : str
        Instrument identifier
    source : str
        Data source ('live', 'historical', 'backfill')
    last_success_ns : int
        Last successful processing timestamp in nanoseconds
    last_attempt_ns : int
        Last attempted processing timestamp in nanoseconds
    last_count : int
        Count from last successful processing
    completeness_pct : float
        Percentage of expected data received (0-100)
    updated_at : float
        Unix timestamp of last update

    """

    dataset_id: str
    instrument_id: str
    source: str
    last_success_ns: int
    last_attempt_ns: int
    last_count: int
    completeness_pct: float
    updated_at: float


class WatermarkManagerProtocol(Protocol):
    """
    Protocol for watermark management operations.

    This protocol defines the interface for watermark tracking and retrieval.

    """

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
        persistence: PersistenceManager,
    ) -> None:
        """
        Update watermark for a dataset/instrument/source combination.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        source : Source
            Data source enum
        last_success_ns : int
            Last successful processing timestamp in nanoseconds
        count : int
            Count from last successful processing
        completeness_pct : float
            Percentage of expected data received (0-100)
        persistence : PersistenceManager
            Persistence manager for backend operations

        """
        ...

    @overload
    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        persistence: PersistenceManager,
    ) -> Watermark | None: ...

    @overload
    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: str,
        persistence: PersistenceManager,
    ) -> Watermark | None: ...

    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source | str,
        persistence: PersistenceManager,
    ) -> Watermark | None:
        """
        Get watermark for a dataset/instrument/source combination.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        source : Source | str
            Data source as enum or string
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        Watermark | None
            The watermark if exists, None otherwise

        """
        ...

    def iter_watermarks(
        self,
        dataset_id: str | None,
        instrument_id: str | None,
        source: Source | str | None,
        limit: int | None,
        persistence: PersistenceManager,
    ) -> Iterator[Watermark]:
        """
        Yield watermarks optionally filtered by dataset, instrument, or source.

        Parameters
        ----------
        dataset_id : str | None
            Optional dataset ID filter
        instrument_id : str | None
            Optional instrument ID filter
        source : Source | str | None
            Optional source filter
        limit : int | None
            Optional limit on number of records returned
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        Iterator[Watermark]
            Iterator of watermarks

        """
        ...


class WatermarkManager:
    """
    Manages dataset watermark operations.

    This component handles high-water mark tracking for incremental data processing,
    including timestamp range management and watermark persistence. All operations
    are coordinated by the parent DataRegistry for thread safety.

    Examples
    --------
    >>> manager = WatermarkManager()
    >>> manager.update_watermark(
    ...     dataset_id="bars_eurusd_1m",
    ...     instrument_id="EUR/USD",
    ...     source=Source.LIVE,
    ...     last_success_ns=1234567900000000000,
    ...     count=1000,
    ...     completeness_pct=98.5,
    ...     persistence=persistence,
    ... )

    """

    def __init__(self) -> None:
        """Initialize watermark manager with empty cache."""
        self._watermarks: dict[str, Watermark] = {}

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
        persistence: PersistenceManager,
    ) -> None:
        """
        Update watermark for a dataset/instrument/source combination.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        source : Source
            Data source enum
        last_success_ns : int
            Last successful processing timestamp in nanoseconds
        count : int
            Count from last successful processing
        completeness_pct : float
            Percentage of expected data received (0-100)
        persistence : PersistenceManager
            Persistence manager for backend operations

        Examples
        --------
        >>> from ml.config.events import Source
        >>> manager.update_watermark(
        ...     dataset_id="bars_eurusd_1m",
        ...     instrument_id="EUR/USD",
        ...     source=Source.LIVE,
        ...     last_success_ns=1234567900000000000,
        ...     count=1000,
        ...     completeness_pct=98.5,
        ...     persistence=persistence,
        ... )

        """
        source_val = source.value
        watermark_key = f"{dataset_id}:{instrument_id}:{source_val}"

        watermark = Watermark(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            source=source_val,
            last_success_ns=last_success_ns,
            last_attempt_ns=last_success_ns,
            last_count=count,
            completeness_pct=completeness_pct,
            updated_at=time.time(),
        )

        if persistence.config.backend.value == "json":
            self._watermarks[watermark_key] = watermark

        elif persistence.config.backend.value == "postgres":
            session = persistence.get_session()
            if session is None:
                raise RuntimeError("Failed to get database session")

            try:
                # Use the SQL function to update watermark
                query = text(
                    """
                    SELECT update_watermark(
                        :dataset_id, :instrument_id, :source,
                        :last_success_ns, :count, :completeness_pct
                    )
                """,
                )

                session.execute(
                    query,
                    {
                        "dataset_id": dataset_id,
                        "instrument_id": instrument_id,
                        "source": source_val,
                        "last_success_ns": last_success_ns,
                        "count": count,
                        "completeness_pct": completeness_pct,
                    },
                )
                session.commit()

                # Update cache
                self._watermarks[watermark_key] = watermark

            except Exception as e:
                session.rollback()
                logger.error("Failed to update watermark: %s", e)
                raise
            finally:
                session.close()

        logger.debug(
            "Updated watermark: dataset=%s, instrument=%s, source=%s, completeness=%.1f%%",
            dataset_id,
            instrument_id,
            source_val,
            completeness_pct,
        )

    @overload
    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        persistence: PersistenceManager,
    ) -> Watermark | None: ...

    @overload
    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: str,
        persistence: PersistenceManager,
    ) -> Watermark | None: ...

    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source | str,
        persistence: PersistenceManager,
    ) -> Watermark | None:
        """
        Get watermark for a dataset/instrument/source combination.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        source : Source | str
            Data source as enum (preferred) or persisted string
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        Watermark | None
            The watermark if exists, None otherwise

        Examples
        --------
        >>> watermark = manager.get_watermark("bars_eurusd_1m", "EUR/USD", "live", persistence)
        >>> watermark = manager.get_watermark("bars_eurusd_1m", "EUR/USD", Source.LIVE, persistence)
        >>> if watermark:
        ...     print(f"Last success: {watermark.last_success_ns}")
        ...     print(f"Completeness: {watermark.completeness_pct}%")

        """
        source_val = source.value if isinstance(source, Source) else str(source)
        watermark_key = f"{dataset_id}:{instrument_id}:{source_val}"

        if persistence.config.backend.value == "json":
            return self._watermarks.get(watermark_key)

        elif persistence.config.backend.value == "postgres":
            # Check cache first
            if watermark_key in self._watermarks:
                return self._watermarks[watermark_key]

            session = persistence.get_session()
            if session is None:
                raise RuntimeError("Failed to get database session")

            try:
                query = text(
                    """
                    SELECT dataset_id, instrument_id, source,
                           last_success_ns, last_attempt_ns, last_count,
                           completeness_pct, EXTRACT(EPOCH FROM updated_at) as updated_at
                    FROM ml_data_watermarks
                    WHERE dataset_id = :dataset_id
                      AND instrument_id = :instrument_id
                      AND source = :source
                """,
                )

                result = session.execute(
                    query,
                    {
                        "dataset_id": dataset_id,
                        "instrument_id": instrument_id,
                        "source": source_val,
                    },
                ).fetchone()

                if result is None:
                    return None

                watermark = self._watermark_from_row(result)

                # Cache for future use
                self._watermarks[watermark_key] = watermark

                return watermark

            finally:
                session.close()

        return None

    def iter_watermarks(
        self,
        dataset_id: str | None = None,
        instrument_id: str | None = None,
        source: Source | str | None = None,
        limit: int | None = None,
        persistence: PersistenceManager | None = None,
    ) -> Iterator[Watermark]:
        """
        Yield watermarks optionally filtered by dataset, instrument, or source.

        Parameters
        ----------
        dataset_id : str | None
            Optional dataset ID filter
        instrument_id : str | None
            Optional instrument ID filter
        source : Source | str | None
            Optional source filter
        limit : int | None
            Optional limit on number of records returned
        persistence : PersistenceManager | None
            Persistence manager for backend operations

        Yields
        ------
        Watermark
            Watermark record

        Examples
        --------
        >>> for watermark in manager.iter_watermarks(dataset_id="bars_eurusd_1m", persistence=persistence):
        ...     print(f"Instrument: {watermark.instrument_id}, Completeness: {watermark.completeness_pct}%")

        """
        if persistence is None:
            return

        limit_value = None if limit is None else max(0, int(limit))
        source_value = (
            source.value
            if isinstance(source, Source)
            else (str(source) if source is not None else None)
        )

        records: list[Watermark] = []

        if persistence.config.backend.value == "json":
            candidates = list(self._watermarks.values())
            if dataset_id is not None:
                candidates = [wm for wm in candidates if wm.dataset_id == dataset_id]
            if instrument_id is not None:
                candidates = [wm for wm in candidates if wm.instrument_id == instrument_id]
            if source_value is not None:
                candidates = [wm for wm in candidates if wm.source == source_value]

            candidates.sort(key=lambda wm: wm.updated_at, reverse=True)

            if limit_value is not None:
                candidates = candidates[:limit_value]

            records = candidates
        else:
            session = persistence.get_session()
            if session is None:
                raise RuntimeError("Failed to get database session")

            try:
                conditions: list[str] = []
                params: dict[str, Any] = {}
                if dataset_id is not None:
                    conditions.append("dataset_id = :dataset_id")
                    params["dataset_id"] = dataset_id
                if instrument_id is not None:
                    conditions.append("instrument_id = :instrument_id")
                    params["instrument_id"] = instrument_id
                if source_value is not None:
                    conditions.append("source = :source")
                    params["source"] = source_value

                where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
                limit_clause = ""
                if limit_value is not None:
                    limit_clause = " LIMIT :limit"
                    params["limit"] = limit_value

                query = text(
                    """
                    SELECT dataset_id, instrument_id, source,
                           last_success_ns, last_attempt_ns, last_count,
                           completeness_pct, EXTRACT(EPOCH FROM updated_at) AS updated_at
                    FROM ml_data_watermarks
                    """
                    + where_clause
                    + " ORDER BY updated_at DESC"
                    + limit_clause,
                )

                rows = session.execute(query, params).fetchall()
            finally:
                session.close()

            records = []
            for row in rows:
                watermark = self._watermark_from_row(row)
                key = f"{watermark.dataset_id}:{watermark.instrument_id}:{watermark.source}"
                self._watermarks[key] = watermark
                records.append(watermark)

        yield from records

    @staticmethod
    def _watermark_from_row(row: Any) -> Watermark:
        """
        Convert a database row to a watermark instance.

        Parameters
        ----------
        row : Any
            Database row

        Returns
        -------
        Watermark
            Watermark object

        """
        data = dict(getattr(row, "_mapping", row))
        return Watermark(
            dataset_id=str(data.get("dataset_id", "")),
            instrument_id=str(data.get("instrument_id", "")),
            source=str(data.get("source", "")),
            last_success_ns=int(data.get("last_success_ns", 0)),
            last_attempt_ns=int(data.get("last_attempt_ns", 0)),
            last_count=int(data.get("last_count", 0)),
            completeness_pct=float(data.get("completeness_pct", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )

    def _dict_to_watermark(self, data: dict[str, Any]) -> Watermark:
        """
        Convert dictionary to Watermark.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary representation of watermark

        Returns
        -------
        Watermark
            Watermark object

        """
        return Watermark(**data)

    def _watermark_to_dict(self, watermark: Watermark) -> dict[str, Any]:
        """
        Convert Watermark to dictionary.

        Parameters
        ----------
        watermark : Watermark
            Watermark to convert

        Returns
        -------
        dict[str, Any]
            Dictionary representation of watermark

        """
        return {
            "dataset_id": watermark.dataset_id,
            "instrument_id": watermark.instrument_id,
            "source": watermark.source,
            "last_success_ns": watermark.last_success_ns,
            "last_attempt_ns": watermark.last_attempt_ns,
            "last_count": watermark.last_count,
            "completeness_pct": watermark.completeness_pct,
            "updated_at": watermark.updated_at,
        }


__all__ = [
    "Watermark",
    "WatermarkManager",
    "WatermarkManagerProtocol",
]
