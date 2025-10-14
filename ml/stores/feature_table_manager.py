"""
Feature table management for FeatureStore.

This module handles database schema setup, table reflection/creation, and table-level
operations like clearing features.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import BIGINT
from sqlalchemy import BOOLEAN
from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import Index
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.engine import Engine


if TYPE_CHECKING:
    from sqlalchemy import MetaData


logger = logging.getLogger(__name__)


class FeatureTableManagerProtocol(Protocol):
    """
    Protocol for feature table management operations.
    """

    def setup_tables(self) -> Table:
        """
        Set up or reflect feature storage table.
        """
        ...

    def clear_features(
        self,
        instrument_id: str | None = None,
        feature_version: str | None = None,
    ) -> None:
        """
        Clear stored features.
        """
        ...

    def get_feature_values_table(self) -> Table:
        """
        Get the feature values table.
        """
        ...


class FeatureTableManager:
    """
    Handles feature table schema and management.

    Responsibilities:
    - Table creation and reflection
    - Schema management
    - Feature deletion

    """

    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize feature table manager.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine for database operations
        metadata : MetaData
            SQLAlchemy metadata container
        logger : logging.Logger | None
            Logger for operations (default: creates module logger)

        """
        self._engine = engine
        self._metadata = metadata
        self._logger = logger or logging.getLogger(__name__)
        self._feature_values_table: Table | None = None

    def setup_tables(self) -> Table:
        """
        Reflect (preferred) or create a compatible ml_feature_values table.

        The canonical schema is created by migrations (partitioned by ts_event):
        - feature_set_id VARCHAR(255)
        - instrument_id VARCHAR(100)
        - ts_event BIGINT
        - ts_init BIGINT
        - values JSONB
        - is_live BOOLEAN
        - source VARCHAR(50)
        - created_at TIMESTAMPTZ

        Primary key (id, ts_event) where id is BIGSERIAL.

        Returns
        -------
        Table
            The feature values table

        """
        from ml.stores.table_factory import get_schema_name

        schema_name = get_schema_name(self._engine)

        try:
            # Prefer reflecting the migrated table. Avoid opportunistic DDL here to
            # prevent lock contention with concurrent writers in tests/integration;
            # migrations and test fixtures are responsible for indexes/partitions.
            self._feature_values_table = Table(
                "ml_feature_values",
                self._metadata,
                autoload_with=self._engine,
                schema=schema_name,
            )
            self._logger.debug("Reflected ml_feature_values table from database")
        except Exception as e:
            # Fallback: create a non-partitioned compatible table for tests/dev
            self._logger.warning(
                "Failed to reflect ml_feature_values table, creating fallback: %s",
                e,
            )
            from sqlalchemy import Integer

            self._feature_values_table = Table(
                "ml_feature_values",
                self._metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("feature_set_id", String(255), nullable=False),
                Column("instrument_id", String(100), nullable=False),
                Column("ts_event", BIGINT, nullable=False),
                Column("ts_init", BIGINT, nullable=False),
                Column("values", JSON, nullable=False),
                Column("is_live", BOOLEAN, default=False),
                Column("source", String(50)),
                Column("created_at", BIGINT),
                Index(
                    "idx_ml_feature_values_lookup",
                    "feature_set_id",
                    "instrument_id",
                    "ts_event",
                ),
                Index(
                    "uq_ml_feature_values_key_dev",
                    "feature_set_id",
                    "instrument_id",
                    "ts_event",
                    unique=True,
                ),
                Index("idx_ml_feature_values_live", "is_live"),
                schema=schema_name,
            )
            self._metadata.create_all(self._engine)
            self._logger.info("Created fallback ml_feature_values table")

        return self._feature_values_table

    def get_feature_values_table(self) -> Table:
        """
        Get the feature values table.

        Returns
        -------
        Table
            The feature values table

        Raises
        ------
        RuntimeError
            If table has not been set up via setup_tables()

        """
        if self._feature_values_table is None:
            raise RuntimeError(
                "Feature values table not initialized. Call setup_tables() first.",
            )
        return self._feature_values_table

    def clear_features(
        self,
        instrument_id: str | None = None,
        feature_version: str | None = None,
    ) -> None:
        """
        Clear stored features.

        Parameters
        ----------
        instrument_id : str, optional
            Clear only for specific instrument.
        feature_version : str, optional
            Clear only specific version.

        """
        table = self.get_feature_values_table()

        with self._engine.begin() as conn:
            delete_stmt = table.delete()

            if instrument_id:
                delete_stmt = delete_stmt.where(
                    table.c.instrument_id == instrument_id,
                )

            if feature_version:
                delete_stmt = delete_stmt.where(
                    table.c.feature_version == feature_version,
                )

            result = conn.execute(delete_stmt)
            rows_deleted = result.rowcount if hasattr(result, "rowcount") else 0
            self._logger.info(
                "Cleared %d feature rows (instrument_id=%s, feature_version=%s)",
                rows_deleted,
                instrument_id,
                feature_version,
            )
