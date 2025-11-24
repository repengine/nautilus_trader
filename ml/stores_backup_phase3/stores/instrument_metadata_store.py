"""
Instrument Metadata Store for ML pipeline integration with Nautilus Trader.

This module provides storage and retrieval of temporal instrument metadata for
factor-based portfolio construction. Metadata includes duration buckets, issuer types,
and liquidity tiers that evolve over time.

Key features:
- Protocol-first design (InstrumentMetadataStoreProtocol)
- Progressive fallback (PostgreSQL → DummyInstrumentMetadataStore)
- Temporal versioning with ts_event and ts_init
- Efficient point-in-time queries
- Integration with PartitionManager for monthly partitioning

Performance:
- Cold path only (no hot path usage)
- Point-in-time queries: O(log n) via indexed lookups
- Factor-based filtering: O(n) scanned, indexed for common patterns

"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Final

from sqlalchemy import BIGINT
from sqlalchemy import SMALLINT
from sqlalchemy import Column
from sqlalchemy import Index
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from ml.common.db_utils import get_or_create_engine
from ml.stores.mixins import HealthMixin


if TYPE_CHECKING:
    from collections.abc import Mapping


logger = logging.getLogger(__name__)


# Module constants
SCHEMA: Final[str] = "ml"
TABLE_NAME: Final[str] = "instrument_metadata"


class InstrumentMetadataStore(HealthMixin):
    """
    PostgreSQL-backed instrument metadata store.

    Stores temporal instrument metadata with factor mappings for portfolio construction.
    Implements InstrumentMetadataStoreProtocol for structural typing.

    Examples
    --------
    >>> store = InstrumentMetadataStore("postgresql://user:pass@localhost/db")
    >>> store.write_metadata(
    ...     instrument_id="US10Y.BOND",
    ...     duration_bucket=2,  # Long duration
    ...     issuer_type=0,      # Sovereign
    ...     liquidity_tier=1,   # High liquidity
    ...     ts_event=time.time_ns(),
    ...     ts_init=time.time_ns(),
    ... )
    >>> metadata = store.get_metadata("US10Y.BOND")
    >>> assert metadata["duration_bucket"] == 2

    """

    def __init__(
        self,
        connection_string: str,
        schema: str = SCHEMA,
        table_name: str = TABLE_NAME,
    ) -> None:
        """
        Initialize the instrument metadata store.

        Parameters
        ----------
        connection_string : str
            PostgreSQL connection string
        schema : str, default="ml"
            Database schema name
        table_name : str, default="instrument_metadata"
            Table name

        """
        self.connection_string = connection_string
        self.schema = schema
        self.table_name = table_name

        # Initialize database engine via EngineManager (Pattern 1)
        self.engine: Engine = get_or_create_engine(connection_string)

        # Define table schema
        self.metadata_obj = MetaData(schema=schema)
        self.table = self._define_table()

        logger.info(
            "Initialized InstrumentMetadataStore",
            extra={"schema": schema, "table": table_name},
        )

    def _define_table(self) -> Table:
        """Define the instrument_metadata table schema."""
        table = Table(
            self.table_name,
            self.metadata_obj,
            Column("instrument_id", Text, nullable=False),
            Column("ts_event", BIGINT, nullable=False),
            Column("ts_init", BIGINT, nullable=False),
            Column("duration_bucket", SMALLINT, nullable=False),
            Column("issuer_type", SMALLINT, nullable=False),
            Column("liquidity_tier", SMALLINT, nullable=False),
            Column("region", Text, nullable=True),
            Column("sector", Text, nullable=True),
            Column("rating", Text, nullable=True),
            Column("valid_from_ns", BIGINT, nullable=False),
            Column("valid_until_ns", BIGINT, nullable=True),
            Column("created_at_ns", BIGINT, nullable=False),
            Column("updated_at_ns", BIGINT, nullable=False),
            Index(
                f"idx_{self.table_name}_ts_event",
                "ts_event",
                postgresql_using="BRIN",
            ),
            Index(
                f"idx_{self.table_name}_instrument_ts",
                "instrument_id",
                "ts_event",
            ),
            Index(
                f"idx_{self.table_name}_validity",
                "instrument_id",
                "valid_from_ns",
                "valid_until_ns",
                postgresql_where=Column("valid_until_ns").is_(None),
            ),
            schema=self.schema,
        )
        return table

    def write_metadata(
        self,
        instrument_id: str,
        duration_bucket: int,
        issuer_type: int,
        liquidity_tier: int,
        ts_event: int,
        ts_init: int,
        region: str | None = None,
        sector: str | None = None,
        rating: str | None = None,
        valid_from_ns: int | None = None,
        valid_until_ns: int | None = None,
    ) -> None:
        """
        Write instrument metadata to the store.

        Parameters
        ----------
        instrument_id : str
            Nautilus InstrumentId (e.g., "US10Y.BOND")
        duration_bucket : int
            Duration classification: 0=Short, 1=Medium, 2=Long
        issuer_type : int
            Issuer classification: 0=SOVEREIGN, 1=QUASI_SOVEREIGN, 2=CORPORATE_IG, 3=CORPORATE_HY
        liquidity_tier : int
            Liquidity classification: 1=High, 2=Medium, 3=Low
        ts_event : int
            Event timestamp in nanoseconds
        ts_init : int
            Initialization timestamp in nanoseconds
        region : str | None
            Geographic region
        sector : str | None
            Market sector
        rating : str | None
            Credit rating
        valid_from_ns : int | None
            Start of validity period (defaults to ts_event)
        valid_until_ns : int | None
            End of validity period (None = currently valid)

        """
        # Validate inputs
        if not instrument_id:
            raise ValueError("instrument_id cannot be empty")
        if duration_bucket not in {0, 1, 2}:
            raise ValueError(f"Invalid duration_bucket: {duration_bucket}. Must be 0, 1, or 2")
        if issuer_type not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid issuer_type: {issuer_type}. Must be 0, 1, 2, or 3")
        if liquidity_tier not in {1, 2, 3}:
            raise ValueError(f"Invalid liquidity_tier: {liquidity_tier}. Must be 1, 2, or 3")

        # Set defaults
        if valid_from_ns is None:
            valid_from_ns = ts_event

        current_time_ns = time.time_ns()

        # Build record
        record = {
            "instrument_id": instrument_id,
            "ts_event": ts_event,
            "ts_init": ts_init,
            "duration_bucket": duration_bucket,
            "issuer_type": issuer_type,
            "liquidity_tier": liquidity_tier,
            "region": region,
            "sector": sector,
            "rating": rating,
            "valid_from_ns": valid_from_ns,
            "valid_until_ns": valid_until_ns,
            "created_at_ns": current_time_ns,
            "updated_at_ns": current_time_ns,
        }

        # Upsert record (insert or update on conflict)
        stmt = insert(self.table).values(**record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["instrument_id", "ts_event"],
            set_={
                "duration_bucket": stmt.excluded.duration_bucket,
                "issuer_type": stmt.excluded.issuer_type,
                "liquidity_tier": stmt.excluded.liquidity_tier,
                "region": stmt.excluded.region,
                "sector": stmt.excluded.sector,
                "rating": stmt.excluded.rating,
                "valid_from_ns": stmt.excluded.valid_from_ns,
                "valid_until_ns": stmt.excluded.valid_until_ns,
                "updated_at_ns": current_time_ns,
            },
        )

        with self.engine.begin() as conn:
            conn.execute(stmt)

        logger.debug(
            "Wrote instrument metadata",
            extra={
                "instrument_id": instrument_id,
                "duration_bucket": duration_bucket,
                "issuer_type": issuer_type,
                "liquidity_tier": liquidity_tier,
            },
        )

    def get_metadata(
        self,
        instrument_id: str,
        ts_event: int | None = None,
    ) -> Mapping[str, Any] | None:
        """
        Get metadata for an instrument at a specific point in time.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int | None
            Query timestamp in nanoseconds (None = get current/latest)

        Returns
        -------
        Mapping[str, Any] | None
            Metadata dictionary or None if not found

        """
        if ts_event is None:
            # Get currently valid metadata
            stmt = (
                select(self.table)
                .where(self.table.c.instrument_id == instrument_id)
                .where(self.table.c.valid_until_ns.is_(None))
                .order_by(self.table.c.ts_event.desc())
                .limit(1)
            )
        else:
            # Get metadata valid at specific time
            stmt = (
                select(self.table)
                .where(self.table.c.instrument_id == instrument_id)
                .where(self.table.c.ts_event <= ts_event)
                .where(
                    (self.table.c.valid_until_ns.is_(None))
                    | (self.table.c.valid_until_ns > ts_event),
                )
                .order_by(self.table.c.ts_event.desc())
                .limit(1)
            )

        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            row = result.fetchone()

        if row is None:
            return None

        return dict(row._mapping)

    def get_instruments_by_factors(
        self,
        duration_bucket: int | None = None,
        issuer_type: int | None = None,
        liquidity_tier: int | None = None,
        ts_event: int | None = None,
    ) -> list[str]:
        """
        Get instruments matching factor criteria.

        Parameters
        ----------
        duration_bucket : int | None
            Filter by duration bucket
        issuer_type : int | None
            Filter by issuer type
        liquidity_tier : int | None
            Filter by liquidity tier
        ts_event : int | None
            Query timestamp (None = current)

        Returns
        -------
        list[str]
            List of matching instrument IDs

        """
        # Build WHERE clause
        conditions = []

        if duration_bucket is not None:
            conditions.append(self.table.c.duration_bucket == duration_bucket)
        if issuer_type is not None:
            conditions.append(self.table.c.issuer_type == issuer_type)
        if liquidity_tier is not None:
            conditions.append(self.table.c.liquidity_tier == liquidity_tier)

        if ts_event is None:
            # Get currently valid instruments
            conditions.append(self.table.c.valid_until_ns.is_(None))
        else:
            # Get instruments valid at specific time
            conditions.append(self.table.c.ts_event <= ts_event)
            conditions.append(
                (self.table.c.valid_until_ns.is_(None))
                | (self.table.c.valid_until_ns > ts_event),
            )

        stmt = select(self.table.c.instrument_id).distinct()
        for condition in conditions:
            stmt = stmt.where(condition)

        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            rows = result.fetchall()

        return [row[0] for row in rows]

    def flush(self) -> None:
        """Flush any pending writes to persistent storage."""
        # No-op for PostgreSQL (writes are immediately persisted)

    def get_health_status(self) -> dict[str, Any]:
        """
        Get component health status.

        Returns
        -------
        dict[str, Any]
            Health status information

        """
        try:
            with self.engine.connect() as conn:
                # Simple query to check database connectivity
                result = conn.execute(select(self.table).limit(1))
                result.fetchall()

            return {
                "status": "healthy",
                "component": "InstrumentMetadataStore",
                "schema": self.schema,
                "table": self.table_name,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "InstrumentMetadataStore",
                "error": str(e),
            }


class DummyInstrumentMetadataStore:
    """
    In-memory fallback implementation of InstrumentMetadataStoreProtocol.

    Used when PostgreSQL is unavailable. Provides same interface but with no persistence.
    Implements progressive fallback (Pattern 4).

    Examples
    --------
    >>> store = DummyInstrumentMetadataStore()
    >>> store.write_metadata(
    ...     instrument_id="US10Y.BOND",
    ...     duration_bucket=2,
    ...     issuer_type=0,
    ...     liquidity_tier=1,
    ...     ts_event=time.time_ns(),
    ...     ts_init=time.time_ns(),
    ... )
    >>> metadata = store.get_metadata("US10Y.BOND")
    >>> assert metadata is not None

    """

    def __init__(self) -> None:
        """Initialize the dummy store with in-memory storage."""
        self._metadata: dict[str, list[dict[str, Any]]] = {}
        logger.warning(
            "Initialized DummyInstrumentMetadataStore - no persistence available",
        )

    def write_metadata(
        self,
        instrument_id: str,
        duration_bucket: int,
        issuer_type: int,
        liquidity_tier: int,
        ts_event: int,
        ts_init: int,
        region: str | None = None,
        sector: str | None = None,
        rating: str | None = None,
        valid_from_ns: int | None = None,
        valid_until_ns: int | None = None,
    ) -> None:
        """Write metadata to in-memory storage."""
        if valid_from_ns is None:
            valid_from_ns = ts_event

        record = {
            "instrument_id": instrument_id,
            "ts_event": ts_event,
            "ts_init": ts_init,
            "duration_bucket": duration_bucket,
            "issuer_type": issuer_type,
            "liquidity_tier": liquidity_tier,
            "region": region,
            "sector": sector,
            "rating": rating,
            "valid_from_ns": valid_from_ns,
            "valid_until_ns": valid_until_ns,
            "created_at_ns": time.time_ns(),
            "updated_at_ns": time.time_ns(),
        }

        if instrument_id not in self._metadata:
            self._metadata[instrument_id] = []

        self._metadata[instrument_id].append(record)

    def get_metadata(
        self,
        instrument_id: str,
        ts_event: int | None = None,
    ) -> Mapping[str, Any] | None:
        """Get metadata from in-memory storage."""
        if instrument_id not in self._metadata:
            return None

        records = self._metadata[instrument_id]

        if ts_event is None:
            # Get currently valid (latest with valid_until_ns=None)
            valid_records = [r for r in records if r["valid_until_ns"] is None]
            if not valid_records:
                return None
            return max(valid_records, key=lambda r: r["ts_event"])
        else:
            # Get metadata valid at specific time
            valid_records = [
                r
                for r in records
                if r["ts_event"] <= ts_event
                and (r["valid_until_ns"] is None or r["valid_until_ns"] > ts_event)
            ]
            if not valid_records:
                return None
            return max(valid_records, key=lambda r: r["ts_event"])

    def get_instruments_by_factors(
        self,
        duration_bucket: int | None = None,
        issuer_type: int | None = None,
        liquidity_tier: int | None = None,
        ts_event: int | None = None,
    ) -> list[str]:
        """Get instruments matching factor criteria from in-memory storage."""
        matching_instruments = set()

        for instrument_id, records in self._metadata.items():
            # Get the metadata for this instrument at the query time
            metadata = self.get_metadata(instrument_id, ts_event)
            if metadata is None:
                continue

            # Check if it matches the factor criteria
            if duration_bucket is not None and metadata["duration_bucket"] != duration_bucket:
                continue
            if issuer_type is not None and metadata["issuer_type"] != issuer_type:
                continue
            if liquidity_tier is not None and metadata["liquidity_tier"] != liquidity_tier:
                continue

            matching_instruments.add(instrument_id)

        return sorted(matching_instruments)

    def flush(self) -> None:
        """Flush (no-op for in-memory store)."""

    def get_health_status(self) -> dict[str, Any]:
        """Get component health status."""
        return {
            "status": "degraded",
            "component": "DummyInstrumentMetadataStore",
            "persistence": "in-memory-only",
            "instruments_cached": len(self._metadata),
        }


__all__ = [
    "DummyInstrumentMetadataStore",
    "InstrumentMetadataStore",
]
