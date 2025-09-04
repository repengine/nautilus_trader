"""
Observability DB persistor (off hot-path).

Provides a minimal adapter to persist observability DataFrames to a relational
database using SQLAlchemy engines provisioned by EngineManager. Intended for
background tasks; do not call from hot loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd
from sqlalchemy import BIGINT, FLOAT, INTEGER, JSON, NVARCHAR, Column, MetaData, String, Table
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
        """Create observability tables if they don't exist."""
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
                df.to_sql("obs_latency_watermarks", conn, if_exists="append", index=False, method="multi")
                written["latency"] = int(len(df))
            if (df := tables.get("metrics")) is not None and not df.empty:
                df.to_sql("obs_metrics", conn, if_exists="append", index=False, method="multi")
                written["metrics"] = int(len(df))
            if (df := tables.get("correlation")) is not None and not df.empty:
                df.to_sql("obs_event_correlation", conn, if_exists="append", index=False, method="multi")
                written["correlation"] = int(len(df))
            if (df := tables.get("health")) is not None and not df.empty:
                df.to_sql("obs_health_scores", conn, if_exists="append", index=False, method="multi")
                written["health"] = int(len(df))
        return written
