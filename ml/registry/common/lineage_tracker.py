#!/usr/bin/env python3

"""
LineageTrackerComponent - Handles dataset lineage relationship management.

This component is extracted from the DataRegistry god class following the
established TDD decomposition pattern. It handles:
- Lineage linking
- Lineage iteration
- Pipeline signature get/set

Thread-safety: Uses the persistence component's lock for thread-safe operations.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from ml.registry.dataclasses import DatasetLineageRecord
from ml.registry.persistence import BackendType


if TYPE_CHECKING:
    from ml.registry.common.data_persistence import DataPersistenceComponent


logger = logging.getLogger(__name__)


class LineageTrackerComponent:
    """
    Handles dataset lineage relationship management.

    This component manages the linking and tracking of dataset lineage
    relationships, including pipeline signatures.

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
    >>> tracker = LineageTrackerComponent(persistence)
    >>> tracker.link_lineage(
    ...     child_dataset_id="features",
    ...     parent_ids=["bars_eurusd"],
    ...     transform_id="feature_pipeline_v1",
    ...     ts_range={"start_ns": 1000, "end_ns": 2000},
    ...     params={"lookback": 20},
    ... )
    """

    def __init__(self, persistence: DataPersistenceComponent) -> None:
        """
        Initialize lineage tracker component.

        Parameters
        ----------
        persistence : DataPersistenceComponent
            The persistence component for storage operations.
        """
        self._persistence = persistence

    def link_lineage(
        self,
        child_dataset_id: str,
        parent_ids: list[str],
        transform_id: str,
        ts_range: dict[str, int],
        params: dict[str, Any],
    ) -> None:
        """
        Link dataset lineage relationships.

        Parameters
        ----------
        child_dataset_id : str
            Child dataset ID.
        parent_ids : list[str]
            List of parent dataset IDs.
        transform_id : str
            Unique identifier for this transformation.
        ts_range : dict[str, int]
            Time range with 'start_ns' and 'end_ns' keys.
        params : dict[str, Any]
            Transformation parameters used.

        Raises
        ------
        RuntimeError
            If database session cannot be obtained (PostgreSQL backend).

        Example
        -------
        >>> tracker.link_lineage(
        ...     child_dataset_id="features_microstructure",
        ...     parent_ids=["bars_eurusd_1m", "quotes_eurusd"],
        ...     transform_id="feature_pipeline_v1",
        ...     ts_range={"start_ns": 1234567890000000000, "end_ns": 1234567900000000000},
        ...     params={"lookback_bars": 20, "include_imbalance": True}
        ... )
        """
        with self._persistence._lock:
            for parent_id in parent_ids:
                lineage_entry = {
                    "transform_id": transform_id,
                    "child_dataset_id": child_dataset_id,
                    "parent_dataset_id": parent_id,
                    "ts_range": ts_range,
                    "parameters": params,
                    "created_at": time.time(),
                }

                if self._persistence.backend == BackendType.JSON:
                    self._persistence._lineage.append(lineage_entry)

                    # Trim old lineage entries to prevent unbounded growth (keep last 5000)
                    if len(self._persistence._lineage) > 5000:
                        self._persistence._lineage = self._persistence._lineage[-5000:]

                    self._persistence._save_registry()

                elif self._persistence.backend == BackendType.POSTGRES:
                    session = self._persistence.persistence.get_session()
                    if session is None:
                        raise RuntimeError("Failed to get database session")

                    try:
                        query = text(
                            """
                            INSERT INTO ml_data_lineage
                            (transform_id, child_dataset_id, parent_dataset_id, ts_range, parameters)
                            VALUES
                            (:transform_id, :child_dataset_id, :parent_dataset_id, :ts_range, :parameters)
                        """
                        )

                        session.execute(
                            query,
                            {
                                "transform_id": transform_id,
                                "child_dataset_id": child_dataset_id,
                                "parent_dataset_id": parent_id,
                                "ts_range": json.dumps(ts_range),
                                "parameters": json.dumps(params),
                            },
                        )

                    except Exception as e:
                        session.rollback()
                        logger.error("Failed to link lineage: %s", e, exc_info=True)
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
        *,
        child: str | None = None,
        parent: str | None = None,
        limit: int | None = None,
    ) -> Iterator[DatasetLineageRecord]:
        """
        Yield lineage records filtered by optional child or parent identifiers.

        Parameters
        ----------
        child : str | None
            Filter by child dataset ID.
        parent : str | None
            Filter by parent dataset ID.
        limit : int | None
            Maximum number of records to return.

        Yields
        ------
        DatasetLineageRecord
            Matching lineage records sorted by created_at descending.

        Raises
        ------
        RuntimeError
            If database session cannot be obtained (PostgreSQL backend).
        """
        limit_value = None if limit is None else max(0, int(limit))

        with self._persistence._lock:
            records: list[DatasetLineageRecord] = []

            if self._persistence.backend == BackendType.JSON:
                entries = list(self._persistence._lineage)
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
                session = self._persistence.persistence.get_session()
                if session is None:
                    raise RuntimeError("Failed to get database session")

                try:
                    conditions: list[str] = []
                    params: dict[str, Any] = {}
                    if child is not None:
                        conditions.append("child_dataset_id = :child")
                        params["child"] = child
                    if parent is not None:
                        conditions.append("parent_dataset_id = :parent")
                        params["parent"] = parent

                    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
                    limit_clause = ""
                    if limit_value is not None:
                        limit_clause = " LIMIT :limit"
                        params["limit"] = limit_value

                    query = text(
                        """
                        SELECT transform_id, child_dataset_id, parent_dataset_id,
                               ts_range, parameters,
                               EXTRACT(EPOCH FROM created_at) AS created_at
                        FROM ml_data_lineage
                        """
                        + where_clause
                        + " ORDER BY created_at DESC"
                        + limit_clause
                    )

                    rows = session.execute(query, params).fetchall()
                finally:
                    session.close()

                records = [self._persistence._lineage_from_row(row) for row in rows]

        yield from records

    def get_pipeline_signature(self, dataset_id: str) -> str | None:
        """
        Get the pipeline signature for a dataset.

        Retrieves the pipeline signature stored in the dataset manifest's metadata.
        Returns None if the dataset doesn't exist or has no stored signature.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.

        Returns
        -------
        str | None
            Pipeline signature (SHA256 hex digest) or None if not found.

        Example
        -------
        >>> signature = tracker.get_pipeline_signature("my_dataset")
        >>> if signature:
        ...     print(f"Stored signature: {signature[:16]}...")
        """
        with self._persistence._lock:
            try:
                if dataset_id not in self._persistence._manifests:
                    return None

                manifest = self._persistence._manifests[dataset_id]

                # Check metadata field for pipeline_signature
                if manifest.metadata and "pipeline_signature" in manifest.metadata:
                    sig = manifest.metadata["pipeline_signature"]
                    if isinstance(sig, str):
                        return sig
                # Also check the pipeline_signature field directly
                if hasattr(manifest, "pipeline_signature") and manifest.pipeline_signature:
                    return manifest.pipeline_signature
                return None
            except KeyError:
                # Dataset doesn't exist
                logger.debug("Dataset %s not found, no signature available", dataset_id)
                return None
            except Exception as e:
                logger.warning(
                    "Failed to get pipeline signature for dataset_id=%s: %s",
                    dataset_id,
                    e,
                    exc_info=True,
                )
                return None

    def set_pipeline_signature(self, dataset_id: str, signature: str) -> None:
        """
        Set the pipeline signature for a dataset.

        Stores the pipeline signature in the dataset manifest's metadata.
        This allows tracking which pipeline configuration was used to create
        the dataset, enabling validation of pipeline consistency across runs.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        signature : str
            Pipeline signature (SHA256 hex digest).

        Raises
        ------
        ValueError
            If signature is empty or dataset doesn't exist.

        Example
        -------
        >>> tracker.set_pipeline_signature("my_dataset", "abc123...")
        """
        if not signature:
            raise ValueError("Pipeline signature cannot be empty")

        with self._persistence._lock:
            if dataset_id not in self._persistence._manifests:
                raise ValueError(f"Dataset '{dataset_id}' not found")

            manifest = self._persistence._manifests[dataset_id]

            # Update metadata with signature
            metadata = dict(manifest.metadata) if manifest.metadata else {}
            metadata["pipeline_signature"] = signature

            # Create updated manifest with new metadata
            manifest_dict = self._persistence._manifest_to_dict(manifest)
            manifest_dict["metadata"] = metadata

            from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

            manifest_dict["last_modified"] = _sanitize(
                int(time.time_ns()),
                context="registry.set_pipeline_signature:last_modified",
            )

            updated_manifest = self._persistence._dict_to_manifest(manifest_dict)
            self._persistence._manifests[dataset_id] = updated_manifest
            self._persistence._save_registry()

            logger.info(
                "Set pipeline signature for dataset_id=%s, signature=%s...",
                dataset_id,
                signature[:16],
            )
