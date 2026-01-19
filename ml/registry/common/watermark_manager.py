#!/usr/bin/env python3

"""
WatermarkManagerComponent - Handles data processing watermark tracking.

This component is extracted from the DataRegistry god class following the
established TDD decomposition pattern. It handles:
- Watermark updates
- Watermark retrieval
- Watermark iteration with filtering

Thread-safety: Uses the persistence component's lock for thread-safe operations.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, overload

from sqlalchemy import text

from ml.config.events import Source
from ml.registry.persistence import BackendType
from ml.registry.watermark import Watermark


if TYPE_CHECKING:
    from ml.registry.common.data_persistence import DataPersistenceComponent


logger = logging.getLogger(__name__)


class WatermarkManagerComponent:
    """
    Handles data processing watermark tracking.

    This component manages the update, retrieval, and iteration of
    watermarks for dataset/instrument/source combinations.

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
    >>> manager = WatermarkManagerComponent(persistence)
    >>> manager.update_watermark(
    ...     dataset_id="my_dataset",
    ...     instrument_id="EUR/USD",
    ...     source=Source.LIVE,
    ...     last_success_ns=1000000000,
    ...     count=100,
    ...     completeness_pct=98.5,
    ... )
    """

    def __init__(self, persistence: DataPersistenceComponent) -> None:
        """
        Initialize watermark manager component.

        Parameters
        ----------
        persistence : DataPersistenceComponent
            The persistence component for storage operations.
        """
        self._persistence = persistence

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        """
        Update watermark for a dataset/instrument/source combination.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        instrument_id : str
            Instrument identifier.
        source : Source
            Data source enum (LIVE, HISTORICAL, BACKFILL).
        last_success_ns : int
            Last successful processing timestamp in nanoseconds.
        count : int
            Count from last successful processing.
        completeness_pct : float
            Percentage of expected data received (0-100).

        Raises
        ------
        RuntimeError
            If database session cannot be obtained (PostgreSQL backend).

        Example
        -------
        >>> manager.update_watermark(
        ...     dataset_id="bars_eurusd_1m",
        ...     instrument_id="EUR/USD",
        ...     source=Source.LIVE,
        ...     last_success_ns=1234567900000000000,
        ...     count=1000,
        ...     completeness_pct=98.5,
        ... )
        """
        with self._persistence._lock:
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

            if self._persistence.backend == BackendType.JSON:
                self._persistence._watermarks[watermark_key] = watermark
                # Persist immediately to satisfy conformance tests
                self._persistence._save_registry(immediate=True)

            elif self._persistence.backend == BackendType.POSTGRES:
                session = self._persistence.persistence.get_session()
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
                    """
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
                    self._persistence._watermarks[watermark_key] = watermark

                except Exception as e:
                    session.rollback()
                    logger.error("Failed to update watermark: %s", e, exc_info=True)
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
    ) -> Watermark | None: ...

    @overload
    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: str,
    ) -> Watermark | None: ...

    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source | str,
    ) -> Watermark | None:
        """
        Get watermark for a dataset/instrument/source combination.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        instrument_id : str
            Instrument identifier.
        source : Source | str
            Data source as enum (preferred) or persisted string.

        Returns
        -------
        Watermark | None
            The watermark if exists, None otherwise.

        Raises
        ------
        RuntimeError
            If database session cannot be obtained (PostgreSQL backend).

        Example
        -------
        >>> watermark = manager.get_watermark("my_dataset", "EUR/USD", Source.LIVE)
        >>> if watermark:
        ...     print(f"Last success: {watermark.last_success_ns}")
        """
        with self._persistence._lock:
            source_val = source.value if isinstance(source, Source) else str(source)
            watermark_key = f"{dataset_id}:{instrument_id}:{source_val}"

            if self._persistence.backend == BackendType.JSON:
                return self._persistence._watermarks.get(watermark_key)

            elif self._persistence.backend == BackendType.POSTGRES:
                # Check cache first
                if watermark_key in self._persistence._watermarks:
                    return self._persistence._watermarks[watermark_key]

                session = self._persistence.persistence.get_session()
                if session is None:
                    raise RuntimeError("Failed to get database session")

                try:
                    query = text(  # nosec B608: dynamic WHERE clause uses bound params only
                        """
                        SELECT dataset_id, instrument_id, source,
                               last_success_ns, last_attempt_ns, last_count,
                               completeness_pct, EXTRACT(EPOCH FROM updated_at) as updated_at
                        FROM ml_data_watermarks
                        WHERE dataset_id = :dataset_id
                          AND instrument_id = :instrument_id
                          AND source = :source
                    """
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

                    watermark = self._persistence._watermark_from_row(result)

                    # Cache for future use
                    self._persistence._watermarks[watermark_key] = watermark

                    return watermark

                finally:
                    session.close()

        return None

    def iter_watermarks(
        self,
        *,
        dataset_id: str | None = None,
        instrument_id: str | None = None,
        source: Source | str | None = None,
        limit: int | None = None,
    ) -> Iterator[Watermark]:
        """
        Yield watermarks optionally filtered by dataset, instrument, or source.

        Parameters
        ----------
        dataset_id : str | None
            Filter by dataset ID.
        instrument_id : str | None
            Filter by instrument ID.
        source : Source | str | None
            Filter by data source.
        limit : int | None
            Maximum number of watermarks to return.

        Yields
        ------
        Watermark
            Matching watermarks sorted by updated_at descending.

        Raises
        ------
        RuntimeError
            If database session cannot be obtained (PostgreSQL backend).
        """
        limit_value = None if limit is None else max(0, int(limit))
        source_value = (
            source.value
            if isinstance(source, Source)
            else (str(source) if source is not None else None)
        )

        with self._persistence._lock:
            records: list[Watermark] = []

            if self._persistence.backend == BackendType.JSON:
                candidates = list(self._persistence._watermarks.values())
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
                session = self._persistence.persistence.get_session()
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
                        + limit_clause
                    )

                    rows = session.execute(query, params).fetchall()
                finally:
                    session.close()

                records = []
                for row in rows:
                    watermark = self._persistence._watermark_from_row(row)
                    key = f"{watermark.dataset_id}:{watermark.instrument_id}:{watermark.source}"
                    self._persistence._watermarks[key] = watermark
                    records.append(watermark)

        yield from records
