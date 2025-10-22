#!/usr/bin/env python3

"""
Manifest management component for DataRegistry.

This module handles dataset manifest CRUD operations (create, read, update, delete)
with support for both JSON and PostgreSQL backends.

"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ml.common.timestamps import sanitize_timestamp_ns
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind


if TYPE_CHECKING:
    from ml.registry.persistence import PersistenceManager


logger = logging.getLogger(__name__)


class ManifestManagerProtocol(Protocol):
    """
    Protocol for manifest management operations.

    This protocol defines the interface for dataset manifest CRUD operations.

    """

    def register_manifest(
        self,
        manifest: DatasetManifest,
        persistence: PersistenceManager,
    ) -> str:
        """
        Register a new dataset manifest.

        Parameters
        ----------
        manifest : DatasetManifest
            Dataset manifest to register
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        str
            Dataset ID of the registered dataset

        Raises
        ------
        ValueError
            If dataset with same ID already exists

        """
        ...

    def get_manifest(
        self,
        dataset_id: str,
        persistence: PersistenceManager,
    ) -> DatasetManifest:
        """
        Get a dataset manifest by ID.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to retrieve
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        DatasetManifest
            The dataset manifest

        Raises
        ------
        ValueError
            If dataset doesn't exist

        """
        ...

    def update_manifest(
        self,
        dataset_id: str,
        changes: dict[str, Any],
        persistence: PersistenceManager,
    ) -> None:
        """
        Update an existing dataset manifest.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to update
        changes : dict[str, Any]
            Dictionary of fields to update
        persistence : PersistenceManager
            Persistence manager for backend operations

        Raises
        ------
        ValueError
            If dataset doesn't exist or changes are invalid

        """
        ...

    def list_manifests(
        self,
        persistence: PersistenceManager,
    ) -> list[DatasetManifest]:
        """
        Return all dataset manifests known to the registry.

        Parameters
        ----------
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        list[DatasetManifest]
            List of all dataset manifests

        """
        ...


class ManifestManager:
    """
    Manages dataset manifest lifecycle operations.

    This component handles CRUD operations for dataset manifests with support
    for both JSON and PostgreSQL backends. All operations are coordinated by
    the parent DataRegistry for thread safety.

    Examples
    --------
    >>> manager = ManifestManager()
    >>> dataset_id = manager.register_manifest(manifest, persistence)
    >>> manifest = manager.get_manifest(dataset_id, persistence)

    """

    def __init__(self) -> None:
        """Initialize manifest manager with empty cache."""
        self._manifests: dict[str, DatasetManifest] = {}
        self._registry_table_cache: dict[int, Table] = {}

    def _get_registry_table(self, session: Session) -> Table:
        """
        Lazily reflect the dataset registry table for the current session bind.
        """
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Failed to resolve database bind for manifest queries")

        cache_key = id(bind)
        table = self._registry_table_cache.get(cache_key)
        if table is None:
            metadata = MetaData()
            table = Table("ml_dataset_registry", metadata, autoload_with=bind)
            self._registry_table_cache[cache_key] = table
        return table

    def register_manifest(
        self,
        manifest: DatasetManifest,
        persistence: PersistenceManager,
    ) -> str:
        """
        Register a new dataset manifest.

        Parameters
        ----------
        manifest : DatasetManifest
            Dataset manifest to register
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        str
            Dataset ID of the registered dataset

        Raises
        ------
        ValueError
            If dataset with same ID already exists

        Examples
        --------
        >>> manifest = DatasetManifest(
        ...     dataset_id="bars_eurusd_1m",
        ...     dataset_type=DatasetType.BARS,
        ...     # ... other fields
        ... )
        >>> dataset_id = manager.register_manifest(manifest, persistence)

        """
        if manifest.dataset_id in self._manifests:
            raise ValueError(f"Dataset '{manifest.dataset_id}' already exists")

        # Store in appropriate backend
        if persistence.config.backend.value == "json":
            self._manifests[manifest.dataset_id] = manifest
        elif persistence.config.backend.value == "postgres":
            session = persistence.get_session()
            if session is None:
                raise RuntimeError("Failed to get database session")

            existing_manifest: DatasetManifest | None = None
            try:
                table = self._get_registry_table(session)

                dataset_type_db = (
                    manifest.dataset_type.name
                    if isinstance(manifest.dataset_type, DatasetType)
                    else str(manifest.dataset_type).split(".")[-1]
                ).upper()

                insert_values: dict[str, Any] = {
                    "dataset_id": manifest.dataset_id,
                    "name": manifest.metadata.get("name", manifest.dataset_id),
                    "version": manifest.version,
                    "dataset_type": dataset_type_db,
                    "storage_kind": (
                        manifest.storage_kind.value
                        if isinstance(manifest.storage_kind, StorageKind)
                        else str(manifest.storage_kind).lower()
                    ),
                    "location": manifest.location,
                    "partitioning": manifest.partitioning or {},
                    "retention_days": manifest.retention_days,
                    "schema": manifest.schema,
                    "schema_hash": manifest.schema_hash,
                    "constraints": manifest.constraints or {},
                    "parents": manifest.lineage or [],
                    "pipeline_signature": manifest.pipeline_signature,
                    "metadata": manifest.metadata,
                }

                session.execute(table.insert().values(**insert_values))
                session.commit()

                # Cache locally
                self._manifests[manifest.dataset_id] = manifest

            except IntegrityError:
                session.rollback()
                logger.info(
                    "Dataset '%s' already exists; hydrating manifest",
                    manifest.dataset_id,
                )
                existing_manifest = self.get_manifest(manifest.dataset_id, persistence)
            except Exception as e:
                session.rollback()
                logger.error("Failed to register dataset: %s", e)
                raise
            finally:
                session.close()

            if existing_manifest is not None:
                self._manifests[existing_manifest.dataset_id] = existing_manifest
                return existing_manifest.dataset_id

        logger.info("Registered dataset '%s' version %s", manifest.dataset_id, manifest.version)
        return manifest.dataset_id

    def get_manifest(
        self,
        dataset_id: str,
        persistence: PersistenceManager,
    ) -> DatasetManifest:
        """
        Get a dataset manifest by ID.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to retrieve
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        DatasetManifest
            The dataset manifest

        Raises
        ------
        ValueError
            If dataset doesn't exist

        Examples
        --------
        >>> manifest = manager.get_manifest("bars_eurusd_1m", persistence)
        >>> print(manifest.dataset_type)
        DatasetType.BARS

        """
        if persistence.config.backend.value == "json":
            if dataset_id not in self._manifests:
                raise ValueError(f"Dataset '{dataset_id}' not found")
            return self._manifests[dataset_id]

        elif persistence.config.backend.value == "postgres":
            # Check cache first
            if dataset_id in self._manifests:
                return self._manifests[dataset_id]

            session = persistence.get_session()
            if session is None:
                raise RuntimeError("Failed to get database session")

            try:
                table = self._get_registry_table(session)
                stmt = (
                    select(
                        table.c.dataset_id,
                        table.c.dataset_type,
                        table.c.storage_kind,
                        table.c.location,
                        table.c.partitioning,
                        table.c.retention_days,
                        table.c.schema,
                        table.c.schema_hash,
                        table.c.constraints,
                        table.c.parents.label("lineage"),
                        table.c.pipeline_signature,
                        table.c.version,
                        (func.extract("epoch", table.c.created_at) * 1_000_000_000).label("created_at"),
                        (func.extract("epoch", table.c.last_modified) * 1_000_000_000).label("last_modified"),
                        table.c.metadata,
                    )
                    .where(table.c.dataset_id == dataset_id)
                    .limit(1)
                )

                result = session.execute(stmt).fetchone()
                if result is None:
                    raise ValueError(f"Dataset '{dataset_id}' not found")

                # Convert to manifest
                manifest = self._manifest_from_row(result)

                # Cache for future use
                self._manifests[dataset_id] = manifest

                return manifest

            finally:
                session.close()

        raise ValueError(f"Dataset '{dataset_id}' not found")

    def update_manifest(
        self,
        dataset_id: str,
        changes: dict[str, Any],
        persistence: PersistenceManager,
    ) -> None:
        """
        Update an existing dataset manifest.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to update
        changes : dict[str, Any]
            Dictionary of fields to update
        persistence : PersistenceManager
            Persistence manager for backend operations

        Raises
        ------
        ValueError
            If dataset doesn't exist or changes are invalid

        Examples
        --------
        >>> manager.update_manifest(
        ...     "bars_eurusd_1m",
        ...     {"retention_days": 180, "version": "1.1.0"},
        ...     persistence,
        ... )

        """
        if persistence.config.backend.value == "json":
            if dataset_id not in self._manifests:
                raise ValueError(f"Dataset '{dataset_id}' not found")

            manifest = self._manifests[dataset_id]

            # Create new manifest with updates
            manifest_dict = self._manifest_to_dict(manifest)
            manifest_dict.update(changes)
            manifest_dict["last_modified"] = sanitize_timestamp_ns(
                int(time.time_ns()),
                context="manifest_manager.update_manifest:json.last_modified",
            )

            # Convert back to manifest object
            updated_manifest = self._dict_to_manifest(manifest_dict)
            self._manifests[dataset_id] = updated_manifest

        elif persistence.config.backend.value == "postgres":
            session = persistence.get_session()
            if session is None:
                raise RuntimeError("Failed to get database session")

            try:
                table = self._get_registry_table(session)

                column_aliases = {"lineage": "parents"}
                json_defaults: dict[str, Any] = {
                    "partitioning": {},
                    "schema": {},
                    "constraints": {},
                    "metadata": {},
                    "parents": [],
                }
                allowed_fields = {
                    "name",
                    "version",
                    "dataset_type",
                    "storage_kind",
                    "location",
                    "partitioning",
                    "retention_days",
                    "schema",
                    "schema_hash",
                    "constraints",
                    "parents",
                    "pipeline_signature",
                    "metadata",
                }

                db_updates: dict[str, Any] = {}
                cache_updates: dict[str, Any] = {}

                for field, raw_value in changes.items():
                    column = column_aliases.get(field, field)
                    if column not in allowed_fields:
                        continue

                    value = raw_value
                    cache_key = "lineage" if column == "parents" else field

                    if column == "dataset_type":
                        if isinstance(value, DatasetType):
                            db_value = value.name.upper()
                            cache_value = value.value
                        else:
                            db_value = str(value).split(".")[-1].upper()
                            cache_value = str(value).lower()
                    elif column == "storage_kind":
                        if isinstance(value, StorageKind):
                            db_value = value.value
                            cache_value = value.value
                        else:
                            db_value = str(value).lower()
                            cache_value = db_value
                    elif column in json_defaults:
                        if value is None:
                            db_value = json_defaults[column]
                        else:
                            db_value = value
                        cache_value = db_value
                    else:
                        db_value = value
                        cache_value = value

                    db_updates[column] = db_value
                    cache_updates[cache_key] = cache_value

                if not db_updates:
                    raise ValueError("No valid fields to update")

                update_stmt = (
                    table.update()
                    .where(table.c.dataset_id == dataset_id)
                    .values(**db_updates, last_modified=func.now())
                )

                result = session.execute(update_stmt)
                # Check if any rows were affected
                row_count = getattr(result, "rowcount", None)
                if row_count is not None and row_count == 0:
                    raise ValueError(f"Dataset '{dataset_id}' not found")

                session.commit()

                # Update cache if present
                if dataset_id in self._manifests:
                    manifest_dict = self._manifest_to_dict(self._manifests[dataset_id])
                    manifest_dict.update(cache_updates)
                    manifest_dict["last_modified"] = sanitize_timestamp_ns(
                        int(time.time_ns()),
                        context="manifest_manager.update_manifest:pg.cache.last_modified",
                    )
                    self._manifests[dataset_id] = self._dict_to_manifest(manifest_dict)

            except Exception as e:
                session.rollback()
                logger.error("Failed to update dataset: %s", e)
                raise
            finally:
                session.close()

        logger.info("Updated dataset '%s' with changes: %s", dataset_id, list(changes.keys()))

    def list_manifests(
        self,
        persistence: PersistenceManager,
    ) -> list[DatasetManifest]:
        """
        Return all dataset manifests known to the registry.

        Parameters
        ----------
        persistence : PersistenceManager
            Persistence manager for backend operations

        Returns
        -------
        list[DatasetManifest]
            List of all dataset manifests

        Examples
        --------
        >>> manifests = manager.list_manifests(persistence)
        >>> for manifest in manifests:
        ...     print(manifest.dataset_id)

        """
        if persistence.config.backend.value == "json":
            return [self._manifests[dataset_id] for dataset_id in sorted(self._manifests)]

        session = persistence.get_session()
        if session is None:
            return list(self._manifests.values())

        try:
            table = self._get_registry_table(session)
            stmt = (
                select(
                    table.c.dataset_id,
                    table.c.dataset_type,
                    table.c.storage_kind,
                    table.c.location,
                    table.c.partitioning,
                    table.c.retention_days,
                    table.c.schema,
                    table.c.schema_hash,
                    table.c.constraints,
                    table.c.parents.label("lineage"),
                    table.c.pipeline_signature,
                    table.c.version,
                    (func.extract("epoch", table.c.created_at) * 1_000_000_000).label("created_at"),
                    (func.extract("epoch", table.c.last_modified) * 1_000_000_000).label("last_modified"),
                    table.c.metadata,
                )
                .order_by(table.c.dataset_id)
            )
            rows = session.execute(stmt).all()
        finally:
            session.close()

        manifests: list[DatasetManifest] = []
        for row in rows:
            manifest = self._manifest_from_row(row)
            self._manifests[manifest.dataset_id] = manifest
            manifests.append(manifest)

        return manifests

    def _manifest_from_row(self, row: Any) -> DatasetManifest:
        """
        Convert a database row to a dataset manifest.

        Parameters
        ----------
        row : Any
            Database row

        Returns
        -------
        DatasetManifest
            Dataset manifest object

        """
        manifest_data = dict(getattr(row, "_mapping", row))

        def _ensure_json(value: Any) -> Any:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        metadata = _ensure_json(manifest_data.get("metadata") or {}) or {}
        manifest_data["metadata"] = metadata
        manifest_data["partitioning"] = _ensure_json(manifest_data.get("partitioning") or {}) or {}
        manifest_data["constraints"] = _ensure_json(manifest_data.get("constraints") or {}) or {}
        manifest_data["schema"] = _ensure_json(manifest_data.get("schema") or {}) or {}
        manifest_data["lineage"] = _ensure_json(manifest_data.get("lineage") or []) or []

        manifest_data["seq_field"] = metadata.get("seq_field")
        manifest_data["ts_field"] = metadata.get(
            "ts_field",
            manifest_data.get("ts_field", "ts_event"),
        )
        manifest_data["primary_keys"] = metadata.get(
            "primary_keys",
            manifest_data.get("primary_keys", ["instrument_id", "ts_event"]),
        )

        return self._dict_to_manifest(manifest_data)

    def _dict_to_manifest(self, data: dict[str, Any]) -> DatasetManifest:
        """
        Convert dictionary to DatasetManifest.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary representation of manifest

        Returns
        -------
        DatasetManifest
            Dataset manifest object

        """
        # Convert string enum values back to enum types (case-insensitive)
        dataset_type_val = data.get("dataset_type")
        if isinstance(dataset_type_val, str):
            dataset_type_val = dataset_type_val.lower()
        data["dataset_type"] = DatasetType(dataset_type_val)

        storage_kind_val = data.get("storage_kind")
        if isinstance(storage_kind_val, str):
            storage_kind_val = storage_kind_val.lower()
        data["storage_kind"] = StorageKind(storage_kind_val)
        return DatasetManifest(**data)

    def _manifest_to_dict(self, manifest: DatasetManifest) -> dict[str, Any]:
        """
        Convert DatasetManifest to dictionary.

        Parameters
        ----------
        manifest : DatasetManifest
            Dataset manifest to convert

        Returns
        -------
        dict[str, Any]
            Dictionary representation of manifest

        """
        data = {
            "dataset_id": manifest.dataset_id,
            "dataset_type": manifest.dataset_type.value,
            "storage_kind": manifest.storage_kind.value,
            "location": manifest.location,
            "partitioning": manifest.partitioning,
            "retention_days": manifest.retention_days,
            "schema": manifest.schema,
            "ts_field": manifest.ts_field,
            "seq_field": manifest.seq_field,
            "primary_keys": manifest.primary_keys,
            "schema_hash": manifest.schema_hash,
            "constraints": manifest.constraints,
            "lineage": manifest.lineage,
            "pipeline_signature": manifest.pipeline_signature,
            "version": manifest.version,
            "created_at": manifest.created_at,
            "last_modified": manifest.last_modified,
            "metadata": manifest.metadata,
        }
        return data


__all__ = [
    "ManifestManager",
    "ManifestManagerProtocol",
]
