#!/usr/bin/env python3

"""
Feature schema component for FeatureStore.

Extracted from FeatureStore (Phase 3.7.4). Provides schema and configuration
logic including table setup/reflection, feature naming, feature set ID derivation,
config hashing, and timestamp normalization.

This component is COLD PATH - called during initialization and setup.

"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from sqlalchemy import BIGINT
from sqlalchemy import BOOLEAN
from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table


if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from ml.features.engineering import FeatureConfig
    from ml.features.pipeline import PipelineRunner
    from ml.features.pipeline import PipelineSpec


logger = logging.getLogger(__name__)


# =========================================================================
# Protocols
# =========================================================================


@runtime_checkable
class FeatureSchemaProtocol(Protocol):
    """
    Protocol for feature schema operations.

    Defines the interface for table setup, feature naming, feature set ID
    derivation, config hashing, and timestamp normalization.

    """

    def setup_tables(self) -> Table:
        """
        Reflect (preferred) or create a compatible ml_feature_values table.

        Returns:
            The SQLAlchemy Table object for ml_feature_values

        """
        ...

    def get_feature_set_id(self) -> str:
        """
        Derive a stable feature_set_id for storage.

        Prefer pipeline signature; otherwise use config hash prefix.

        Returns:
            Feature set identifier string (format: "fs_{hash[:12]}")

        """
        ...

    def get_feature_names(self) -> list[str]:
        """
        Get OFFLINE feature names from pipeline or config.

        Returns:
            List of feature name strings

        """
        ...

    def get_feature_names_online(self) -> list[str]:
        """
        Get ONLINE (hot-path) feature names with L1_ONLY gating.

        Returns:
            List of online feature name strings

        """
        ...

    def compute_config_hash(self) -> str:
        """
        Compute hash of feature configuration for versioning.

        Returns:
            SHA256 hash string (first 16 characters)

        """
        ...

    def normalize_ts_ns(self, ts_value: int) -> tuple[int, bool]:
        """
        Normalize a timestamp to nanoseconds.

        Args:
            ts_value: Timestamp value to normalize

        Returns:
            Tuple of (normalized_value, was_normalized)

        """
        ...


# =========================================================================
# Configuration
# =========================================================================


@dataclass(frozen=True)
class FeatureSchemaConfig:
    """
    Configuration for FeatureSchemaComponent.

    Attributes
    ----------
    table_name : str
        Name of the feature values table (default: "ml_feature_values")
    schema_name : str | None
        Database schema name (None uses default from engine)
    use_partitioned_table : bool
        Whether to expect partitioned table from migrations (default: True)

    """

    table_name: str = "ml_feature_values"
    schema_name: str | None = None
    use_partitioned_table: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not self.table_name:
            raise ValueError("table_name cannot be empty")


# =========================================================================
# Component Implementation
# =========================================================================


@dataclass
class FeatureSchemaComponent:
    """
    Feature schema operations for FeatureStore.

    Extracted from FeatureStore (Phase 3.7.4).

    Provides:
    - setup_tables() - Table setup/reflection
    - get_feature_set_id() - Derive stable feature set ID
    - get_feature_names() - Get offline feature names
    - get_feature_names_online() - Get online feature names (L1_ONLY)
    - compute_config_hash() - Compute SHA256 hash of config
    - normalize_ts_ns() - Timestamp normalization

    Example
    -------
    >>> from ml.stores.common.feature_schema import FeatureSchemaComponent
    >>> schema = FeatureSchemaComponent(
    ...     engine=engine,
    ...     feature_config=feature_config,
    ... )
    >>> table = schema.setup_tables()
    >>> feature_set_id = schema.get_feature_set_id()
    >>> feature_names = schema.get_feature_names()

    """

    engine: Engine
    feature_config: FeatureConfig | None = None
    pipeline_spec: PipelineSpec | None = None
    pipeline_runner_offline: PipelineRunner | None = None
    pipeline_runner_online: PipelineRunner | None = None
    pipeline_hash: str = ""
    config: FeatureSchemaConfig = field(default_factory=FeatureSchemaConfig)
    metadata: MetaData = field(default_factory=MetaData)
    _feature_engineer: Any = field(default=None)
    _feature_values_table: Table | None = field(default=None)

    def __post_init__(self) -> None:
        """Initialize derived attributes."""
        # Compute pipeline hash if not provided and pipeline runner exists
        if not self.pipeline_hash and self.pipeline_runner_offline is not None:
            self.pipeline_hash = self.pipeline_runner_offline.compute_signature()
        elif not self.pipeline_hash:
            # Use a mutable object reference to store the computed hash
            object.__setattr__(self, "pipeline_hash", self.compute_config_hash())

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

        Returns:
            The SQLAlchemy Table object for ml_feature_values

        Example
        -------
        >>> table = schema.setup_tables()
        >>> assert table.name == "ml_feature_values"

        """
        schema_name = self._get_schema_name()

        try:
            # Prefer reflecting the migrated table. Avoid opportunistic DDL here to
            # prevent lock contention with concurrent writers in tests/integration;
            # migrations and test fixtures are responsible for indexes/partitions.
            table = Table(
                self.config.table_name,
                self.metadata,
                autoload_with=self.engine,
                schema=schema_name,
            )
            self._feature_values_table = table
            return table
        except Exception:
            logger.debug(
                "Failed to reflect table %s, creating fallback",
                self.config.table_name,
                exc_info=True,
            )
            # Fallback: create a non-partitioned compatible table for tests/dev
            table = self._create_fallback_table(schema_name)
            self._feature_values_table = table
            return table

    def _get_schema_name(self) -> str | None:
        """
        Get the appropriate schema name for the database.

        Returns:
            Schema name string or None for default

        """
        if self.config.schema_name is not None:
            return self.config.schema_name

        # Use table_factory helper if available
        try:
            from ml.stores.table_factory import get_schema_name
            return get_schema_name(self.engine)
        except Exception:
            logger.debug(
                "Could not get schema name from table_factory",
                exc_info=True,
            )
            # Default to public for PostgreSQL, None for others
            if self.engine.dialect.name == "postgresql":
                return "public"
            return None

    def _create_fallback_table(self, schema_name: str | None) -> Table:
        """
        Create a non-partitioned fallback table for tests/dev.

        Args:
            schema_name: Database schema name or None

        Returns:
            The created Table object

        """
        table = Table(
            self.config.table_name,
            self.metadata,
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
        self.metadata.create_all(self.engine)
        return table

    def get_feature_set_id(self) -> str:
        """
        Derive a stable feature_set_id for storage.

        Prefer pipeline signature; otherwise use config hash prefix.

        Returns:
            Feature set identifier string (format: "fs_{hash[:12]}")

        Example
        -------
        >>> feature_set_id = schema.get_feature_set_id()
        >>> assert feature_set_id.startswith("fs_")
        >>> assert len(feature_set_id) == 15  # "fs_" + 12 char hash

        """
        if self.pipeline_hash:
            return f"fs_{self.pipeline_hash[:12]}"
        return f"fs_{self.compute_config_hash()[:12]}"

    def get_feature_names(self) -> list[str]:
        """
        Get OFFLINE feature names from pipeline or config.

        Uses pipeline_runner_offline if available, otherwise falls back
        to FeatureEngineer.get_feature_names().

        Returns:
            List of feature name strings

        Example
        -------
        >>> names = schema.get_feature_names()
        >>> assert "close_return" in names

        """
        if self.pipeline_runner_offline is not None:
            return list(self.pipeline_runner_offline.compute_feature_names())

        # Fallback to FeatureEngineer
        if self._feature_engineer is not None:
            return cast(list[str], self._feature_engineer.get_feature_names())

        # Last resort: create FeatureEngineer from config
        if self.feature_config is not None:
            try:
                from ml.features.engineering import FeatureEngineer
                engineer = FeatureEngineer(self.feature_config)
                return engineer.get_feature_names()
            except Exception:
                logger.debug(
                    "Failed to get feature names from FeatureEngineer",
                    exc_info=True,
                )
        return []

        return []

    def get_feature_names_online(self) -> list[str]:
        """
        Get ONLINE (hot-path) feature names from pipeline or config with L1_ONLY gating.

        Uses pipeline_runner_online if available, otherwise creates a
        PipelineRunner with L1_ONLY data requirements from FeatureEngineer config.

        Returns:
            List of online feature name strings

        Example
        -------
        >>> online_names = schema.get_feature_names_online()
        >>> # Online names may be a subset of offline names
        >>> assert len(online_names) <= len(schema.get_feature_names())

        """
        if self.pipeline_runner_online is not None:
            return self.pipeline_runner_online.compute_feature_names()

        # Derive from FeatureEngineer configuration if no pipeline_spec provided
        if self._feature_engineer is not None:
            try:
                from ml.features.pipeline import PipelineRunner as _PR
                from ml.registry.base import DataRequirements as _DR

                spec = self._feature_engineer.build_pipeline_spec_from_config()
                return _PR(spec, allowable=_DR.L1_ONLY).compute_feature_names()
            except Exception:
                logger.debug(
                    "Failed to get online feature names from FeatureEngineer",
                    exc_info=True,
                )

        # Last resort: try to create from feature_config
        if self.feature_config is not None:
            try:
                from ml.features.engineering import FeatureEngineer
                from ml.features.pipeline import PipelineRunner as _PR
                from ml.registry.base import DataRequirements as _DR

                engineer = FeatureEngineer(self.feature_config)
                spec = engineer.build_pipeline_spec_from_config()
                return _PR(spec, allowable=_DR.L1_ONLY).compute_feature_names()
            except Exception:
                logger.debug(
                    "Failed to get online feature names from config",
                    exc_info=True,
                )

        return []

    def compute_config_hash(self) -> str:
        """
        Compute hash of feature configuration for versioning.

        Handles both dict-like and dataclass config objects.
        Returns the first 16 characters of the SHA256 hash.

        Returns:
            SHA256 hash string (first 16 characters)

        Example
        -------
        >>> hash1 = schema.compute_config_hash()
        >>> hash2 = schema.compute_config_hash()
        >>> assert hash1 == hash2  # Deterministic
        >>> assert len(hash1) == 16

        """
        if self.feature_config is None:
            # Return a default hash for empty config
            return hashlib.sha256(b"{}").hexdigest()[:16]

        # Handle both dict-like and dataclass objects
        if hasattr(self.feature_config, "__dict__"):
            config_dict = self.feature_config.__dict__
        else:
            # For frozen dataclasses, convert to dict
            try:
                import msgspec
                config_dict = msgspec.to_builtins(self.feature_config)
            except Exception:
                logger.debug(
                    "Failed to convert config to dict with msgspec",
                    exc_info=True,
                )
                config_dict = {}

        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @staticmethod
    def normalize_ts_ns(ts_value: int) -> tuple[int, bool]:
        """
        Normalize a timestamp to nanoseconds.

        Delegates to the centralized timestamp normalization utility.

        Args:
            ts_value: Timestamp value to normalize

        Returns:
            Tuple of (normalized_value, was_normalized) where was_normalized
            is True if the value was converted from a smaller unit.

        Example
        -------
        >>> # Milliseconds to nanoseconds
        >>> norm, changed = FeatureSchemaComponent.normalize_ts_ns(1700000000000)
        >>> assert changed is True
        >>> assert norm == 1700000000000000000
        >>> # Already nanoseconds
        >>> norm, changed = FeatureSchemaComponent.normalize_ts_ns(1700000000000000000)
        >>> assert changed is False

        """
        from ml.common.timestamps import normalize_timestamp_ns
        return normalize_timestamp_ns(ts_value)

    def set_feature_engineer(self, engineer: Any) -> None:
        """
        Set the FeatureEngineer instance for feature name retrieval.

        Args:
            engineer: FeatureEngineer instance

        """
        self._feature_engineer = engineer

    @property
    def feature_values_table(self) -> Table | None:
        """
        Get the feature values table if it has been set up.

        Returns:
            The Table object or None if setup_tables() hasn't been called

        """
        return self._feature_values_table


__all__ = [
    "FeatureSchemaComponent",
    "FeatureSchemaConfig",
    "FeatureSchemaProtocol",
]
