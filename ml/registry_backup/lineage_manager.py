#!/usr/bin/env python3

"""
Lineage management component for DataRegistry.

This module handles dataset lineage tracking operations including parent-child
relationships and lineage graph traversal.

"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.registry.dataclasses import DatasetLineageRecord


if TYPE_CHECKING:
    from ml.registry.persistence import PersistenceManager


logger = logging.getLogger(__name__)


class LineageManagerProtocol(Protocol):
    """
    Protocol for lineage management operations.

    This protocol defines the interface for dataset lineage tracking and traversal.

    """

    def link_lineage(
        self,
        child_dataset_id: str,
        parent_ids: list[str],
        transform_id: str,
        ts_range: dict[str, int],
        params: dict[str, Any],
        persistence: PersistenceManager,
    ) -> None:
        """
        Link dataset lineage relationships.

        Parameters
        ----------
        child_dataset_id : str
            Child dataset ID
        parent_ids : list[str]
            List of parent dataset IDs
        transform_id : str
            Unique identifier for this transformation
        ts_range : dict[str, int]
            Time range with 'start_ns' and 'end_ns' keys
        params : dict[str, Any]
            Transformation parameters used
        persistence : PersistenceManager
            Persistence manager for backend operations

        """
        ...

    def iter_lineage(
        self,
        child: str | None,
        parent: str | None,
        limit: int | None,
        persistence: PersistenceManager,
    ) -> Iterator[DatasetLineageRecord]:
        """
        Yield lineage records filtered by optional child or parent identifiers.

        Parameters
        ----------
        child : str | None
            Optional child dataset ID filter
        parent : str | None
            Optional parent dataset ID filter
        limit : int | None
            Optional limit on number of records returned
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        Iterator[DatasetLineageRecord]
            Iterator of lineage records

        """
        ...


class LineageManager:
    """
    Manages dataset lineage tracking operations.

    This component handles parent-child relationships between datasets, enabling
    lineage graph traversal and dependency tracking. All operations are coordinated
    by the parent DataRegistry for thread safety.

    Examples
    --------
    >>> manager = LineageManager()
    >>> manager.link_lineage(
    ...     child_dataset_id="features_microstructure",
    ...     parent_ids=["bars_eurusd_1m"],
    ...     transform_id="feature_pipeline_v1",
    ...     ts_range={"start_ns": 1234567890000000000, "end_ns": 1234567900000000000},
    ...     params={"lookback_bars": 20},
    ...     persistence=persistence,
    ... )

    """

    def __init__(self) -> None:
        """Initialize lineage manager with empty storage."""
        self._lineage: list[dict[str, Any]] = []
        self._lineage_table_cache: dict[int, Table] = {}

    def _get_lineage_table(self, session: Session) -> Table:
        """Reflect and cache the lineage table for the active engine."""
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Failed to resolve database bind for lineage queries")

        cache_key = id(bind)
        table = self._lineage_table_cache.get(cache_key)
        if table is None:
            metadata = MetaData()
            table = Table("ml_data_lineage", metadata, autoload_with=bind)
            self._lineage_table_cache[cache_key] = table
        return table

    def link_lineage(
        self,
        child_dataset_id: str,
        parent_ids: list[str],
        transform_id: str,
        ts_range: dict[str, int],
        params: dict[str, Any],
        persistence: PersistenceManager,
    ) -> None:
        """
        Link dataset lineage relationships.

        Parameters
        ----------
        child_dataset_id : str
            Child dataset ID
        parent_ids : list[str]
            List of parent dataset IDs
        transform_id : str
            Unique identifier for this transformation
        ts_range : dict[str, int]
            Time range with 'start_ns' and 'end_ns' keys
        params : dict[str, Any]
            Transformation parameters used
        persistence : PersistenceManager
            Persistence manager for backend operations

        Examples
        --------
        >>> manager.link_lineage(
        ...     child_dataset_id="features_microstructure",
        ...     parent_ids=["bars_eurusd_1m", "quotes_eurusd"],
        ...     transform_id="feature_pipeline_v1",
        ...     ts_range={"start_ns": 1234567890000000000, "end_ns": 1234567900000000000},
        ...     params={"lookback_bars": 20, "include_imbalance": True},
        ...     persistence=persistence,
        ... )

        """
        for parent_id in parent_ids:
            lineage_entry = {
                "transform_id": transform_id,
                "child_dataset_id": child_dataset_id,
                "parent_dataset_id": parent_id,
                "ts_range": ts_range,
                "parameters": params,
                "created_at": time.time(),
            }

            if persistence.config.backend.value == "json":
                self._lineage.append(lineage_entry)

                # Trim old lineage entries to prevent unbounded growth (keep last 5000)
                if len(self._lineage) > 5000:
                    self._lineage = self._lineage[-5000:]

            elif persistence.config.backend.value == "postgres":
                session = persistence.get_session()
                if session is None:
                    raise RuntimeError("Failed to get database session")

                try:
                    table = self._get_lineage_table(session)
                    insert_stmt = table.insert().values(
                        transform_id=transform_id,
                        child_dataset_id=child_dataset_id,
                        parent_dataset_id=parent_id,
                        ts_range=json.dumps(ts_range),
                        parameters=json.dumps(params),
                    )

                    session.execute(insert_stmt)

                except Exception as e:
                    session.rollback()
                    logger.error("Failed to link lineage: %s", e)
                    raise

                # Commit after all parents are linked
                if parent_id == parent_ids[-1]:
                    session.commit()
                    session.close()

        logger.info(
            "Linked lineage: child=%s, parents=%s, transform=%s",
            child_dataset_id,
            parent_ids,
            transform_id,
        )

    def iter_lineage(
        self,
        child: str | None = None,
        parent: str | None = None,
        limit: int | None = None,
        persistence: PersistenceManager | None = None,
    ) -> Iterator[DatasetLineageRecord]:
        """
        Yield lineage records filtered by optional child or parent identifiers.

        Parameters
        ----------
        child : str | None
            Optional child dataset ID filter
        parent : str | None
            Optional parent dataset ID filter
        limit : int | None
            Optional limit on number of records returned
        persistence : PersistenceManager | None
            Persistence manager for backend operations

        Yields
        ------
        DatasetLineageRecord
            Lineage record

        Examples
        --------
        >>> for record in manager.iter_lineage(child="features_microstructure", persistence=persistence):
        ...     print(f"Parent: {record.parent_dataset_id}")

        """
        if persistence is None:
            return

        limit_value = None if limit is None else max(0, int(limit))

        records: list[DatasetLineageRecord] = []

        if persistence.config.backend.value == "json":
            entries = list(self._lineage)
            entries.sort(key=lambda entry: float(entry.get("created_at", 0.0)), reverse=True)

            for entry in entries:
                if child is not None and entry.get("child_dataset_id") != child:
                    continue
                if parent is not None and entry.get("parent_dataset_id") != parent:
                    continue

                ts_range = entry.get("ts_range") or {}
                if isinstance(ts_range, str):
                    try:
                        ts_range = json.loads(ts_range)
                    except json.JSONDecodeError:
                        ts_range = {}

                parameters = entry.get("parameters") or {}
                if isinstance(parameters, str):
                    try:
                        parameters = json.loads(parameters)
                    except json.JSONDecodeError:
                        parameters = {}

                record = DatasetLineageRecord(
                    transform_id=str(entry.get("transform_id", "")),
                    child_dataset_id=str(entry.get("child_dataset_id", "")),
                    parent_dataset_id=str(entry.get("parent_dataset_id", "")),
                    ts_range={str(k): int(v) for k, v in dict(ts_range).items()},
                    parameters={str(k): v for k, v in dict(parameters).items()},
                    created_at=float(entry.get("created_at", 0.0)),
                )
                records.append(record)

                if limit_value is not None and len(records) >= limit_value:
                    break
        else:
            session = persistence.get_session()
            if session is None:
                raise RuntimeError("Failed to get database session")

            try:
                table = self._get_lineage_table(session)
                stmt = select(
                    table.c.transform_id,
                    table.c.child_dataset_id,
                    table.c.parent_dataset_id,
                    table.c.ts_range,
                    table.c.parameters,
                    func.extract("epoch", table.c.created_at).label("created_at"),
                )

                if child is not None:
                    stmt = stmt.where(table.c.child_dataset_id == child)
                if parent is not None:
                    stmt = stmt.where(table.c.parent_dataset_id == parent)

                stmt = stmt.order_by(table.c.created_at.desc())
                if limit_value is not None:
                    stmt = stmt.limit(limit_value)

                rows = session.execute(stmt).all()
            finally:
                session.close()

            records = [self._lineage_from_row(row) for row in rows]

        yield from records

    @staticmethod
    def _lineage_from_row(row: Any) -> DatasetLineageRecord:
        """
        Convert a database row to a dataset lineage record.

        Parameters
        ----------
        row : Any
            Database row

        Returns
        -------
        DatasetLineageRecord
            Dataset lineage record object

        """
        data = dict(getattr(row, "_mapping", row))

        ts_range_raw = data.get("ts_range") or {}
        if isinstance(ts_range_raw, str):
            try:
                ts_range = json.loads(ts_range_raw)
            except json.JSONDecodeError:
                ts_range = {}
        else:
            ts_range = dict(ts_range_raw)

        params_raw = data.get("parameters") or {}
        if isinstance(params_raw, str):
            try:
                parameters = json.loads(params_raw)
            except json.JSONDecodeError:
                parameters = {}
        else:
            parameters = dict(params_raw)

        created_at_raw = data.get("created_at", 0.0)
        created_at = float(created_at_raw) if created_at_raw is not None else 0.0

        return DatasetLineageRecord(
            transform_id=str(data.get("transform_id", "")),
            child_dataset_id=str(data.get("child_dataset_id", "")),
            parent_dataset_id=str(data.get("parent_dataset_id", "")),
            ts_range={str(k): int(v) for k, v in ts_range.items()},
            parameters={str(k): v for k, v in parameters.items()},
            created_at=created_at,
        )


__all__ = [
    "LineageManager",
    "LineageManagerProtocol",
]
