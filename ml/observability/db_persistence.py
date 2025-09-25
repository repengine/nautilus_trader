"""
Observability DB persistor (off hot-path).

Provides a minimal adapter to persist observability DataFrames to a relational database
using SQLAlchemy engines provisioned by EngineManager. Intended for background tasks; do
not call from hot loops.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field

import pandas as pd
from sqlalchemy import BIGINT
from sqlalchemy import FLOAT
from sqlalchemy import INTEGER
from sqlalchemy import NVARCHAR
from sqlalchemy import Column
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager


@dataclass(slots=True)
class ObservabilityDBPersistor:
    """
    Persist observability tables to a SQL database.

    Parameters
    ----------
    connection_string : str
        SQLAlchemy database URL (e.g., postgresql:// or sqlite:///path.db).

    """

    connection_string: str
    # Initialized in __post_init__ / _ensure_tables
    engine: Engine = field(init=False)
    metadata: MetaData = field(init=False)
    latency_table: Table = field(init=False)
    metrics_table: Table = field(init=False)
    correlation_table: Table = field(init=False)
    health_table: Table = field(init=False)

    def __post_init__(self) -> None:
        self.engine: Engine = EngineManager.get_engine(self.connection_string)
        self.metadata = MetaData()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """
        Create observability tables if they don't exist.
        """
        # Define schemas explicitly for consistency across backends
        self.latency_table = Table(
            "obs_latency_watermarks",
            self.metadata,
            Column("correlation_id", String(64), nullable=False),
            Column("instrument_id", String(100), nullable=False),
            Column("pipeline_stage", String(64), nullable=False),
            Column("ts_stage_start", BIGINT, nullable=False),
            Column("ts_stage_end", BIGINT, nullable=False),
            Column("stage_latency_ns", BIGINT, nullable=False),
            Column("cumulative_latency_ns", BIGINT, nullable=False),
        )
        self.metrics_table = Table(
            "obs_metrics",
            self.metadata,
            Column("metric_name", String(128), nullable=False),
            Column("metric_type", String(32), nullable=False),
            Column("value", FLOAT, nullable=False),
            Column("timestamp", BIGINT, nullable=False),
            Column("labels", NVARCHAR(4096)),
        )
        self.correlation_table = Table(
            "obs_event_correlation",
            self.metadata,
            Column("correlation_id", String(64), nullable=False),
            Column("event_id", String(64), nullable=False),
            Column("parent_event_id", String(64)),
            Column("instrument_id", String(100), nullable=False),
            Column("domain", String(32), nullable=False),
            Column("lineage_depth", INTEGER, nullable=False),
            Column("ts_event", BIGINT, nullable=False),
            Column("propagation_path", NVARCHAR(4096)),
        )
        self.health_table = Table(
            "obs_health_scores",
            self.metadata,
            Column("component_id", String(64), nullable=False),
            Column("health_score", FLOAT, nullable=False),
            Column("subsystem_scores", NVARCHAR(4096)),
            Column("timestamp", BIGINT, nullable=False),
            Column("measurement_window_ms", INTEGER, nullable=False),
            Column("alert_threshold", FLOAT, nullable=False),
        )

        # Create if not exists
        self.metadata.create_all(self.engine)

    def persist(self, tables: Mapping[str, pd.DataFrame | None]) -> dict[str, int]:
        """
        Persist non-empty DataFrames to their corresponding tables.

        Supported keys: latency, metrics, correlation, health.
        Returns mapping of table name to row count inserted.

        """
        written: dict[str, int] = {}
        with self.engine.begin() as conn:
            if (df := tables.get("latency")) is not None and not df.empty:
                df.to_sql(
                    "obs_latency_watermarks",
                    conn,
                    if_exists="append",
                    index=False,
                    method="multi",
                )
                written["latency"] = len(df)
            if (df := tables.get("metrics")) is not None and not df.empty:
                df.to_sql("obs_metrics", conn, if_exists="append", index=False, method="multi")
                written["metrics"] = len(df)
            if (df := tables.get("correlation")) is not None and not df.empty:
                df.to_sql(
                    "obs_event_correlation",
                    conn,
                    if_exists="append",
                    index=False,
                    method="multi",
                )
                written["correlation"] = len(df)
            if (df := tables.get("health")) is not None and not df.empty:
                df.to_sql(
                    "obs_health_scores",
                    conn,
                    if_exists="append",
                    index=False,
                    method="multi",
                )
                written["health"] = len(df)
        return written

    def apply_retention(self, *, retention_days: int) -> dict[str, int]:
        """
        Delete rows older than the given retention window from all tables.

        Parameters
        ----------
        retention_days : int
            Number of days to retain. Rows older than ``now - retention_days``
            based on the appropriate time column per table are removed.

        Returns
        -------
        dict[str, int]
            Mapping of physical table name to number of rows deleted.

        Notes
        -----
        - Time columns (ns) used per table:
            - obs_latency_watermarks: ts_stage_end
            - obs_metrics: timestamp
            - obs_event_correlation: ts_event
            - obs_health_scores: timestamp

        """
        import time

        from sqlalchemy import bindparam
        from sqlalchemy.sql import column as _column
        from sqlalchemy.sql import delete as _delete
        from sqlalchemy.sql import table as _table

        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        # Compute cutoff in nanoseconds
        now_ns = _sanitize(
            int(time.time_ns()),
            context="observability.apply_retention:now",
        )
        cutoff_ns = _sanitize(
            int(now_ns - int(retention_days) * 24 * 60 * 60 * 1_000_000_000),
            context="observability.apply_retention:cutoff",
        )

        retention_tables: dict[str, str] = {
            "obs_latency_watermarks": "ts_stage_end",
            "obs_metrics": "timestamp",
            "obs_event_correlation": "ts_event",
            "obs_health_scores": "timestamp",
        }

        deleted: dict[str, int] = {}
        with self.engine.begin() as conn:
            for table_name, ts_column in retention_tables.items():
                try:
                    stmt = _delete(
                        _table(table_name, _column(ts_column)),
                    ).where(_column(ts_column) < bindparam("cutoff"))
                    result = conn.execute(stmt, {"cutoff": int(cutoff_ns)})
                    # SQLAlchemy 2.0: result.rowcount may be -1 depending on backend; coerce to int >= 0
                    count = int(result.rowcount or 0)
                    deleted[table_name] = count
                except Exception:
                    # If table doesn't exist yet or backend-specific behavior, record zero deletions
                    deleted.setdefault(table_name, 0)
        return deleted
