from __future__ import annotations

from datetime import UTC
from datetime import datetime
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager
from ml.observability.migrations import ensure_monthly_partitions


pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.usefixtures("cloned_test_database"),
    pytest.mark.usefixtures(
        "isolated_prometheus_registry",
        "mock_tracing_backend",
        "isolated_orchestrator_env",
    ),
]


def test_partition_creation_on_empty_table(
    cloned_test_database: str,
) -> None:
    eng: Engine = EngineManager.get_engine(cloned_test_database)
    table = "obs_part_test_latency"
    ts_col = "ts_stage_end"
    with eng.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        # Create non-partitioned empty base table
        ddl = (
            f"CREATE TABLE {table} ("
            "correlation_id TEXT, instrument_id TEXT, pipeline_stage TEXT, "
            f"ts_stage_start BIGINT, {ts_col} BIGINT NOT NULL, "
            "stage_latency_ns BIGINT, cumulative_latency_ns BIGINT)"
        )
        conn.execute(text(ddl))

    ensure_monthly_partitions(eng, table, ts_col)

    with eng.begin() as conn:
        # Table should be partitioned
        is_partitioned = conn.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM pg_partitioned_table pt JOIN pg_class c ON pt.partrelid=c.oid WHERE c.relname=:t)",
            ),
            {"t": table},
        ).scalar_one()
        assert bool(is_partitioned)

        # At least one monthly partition exists
        part_like = conn.execute(
            text("SELECT COUNT(1) FROM pg_class WHERE relname LIKE :p"),
            {"p": f"{table}_%"},
        ).scalar_one()
        assert int(part_like) >= 1


def test_skip_partition_when_not_empty(
    cloned_test_database: str,
) -> None:
    eng: Engine = EngineManager.get_engine(cloned_test_database)
    table = "obs_part_test_metrics"
    ts_col = "timestamp"
    with eng.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        conn.execute(text(f"CREATE TABLE {table} ({ts_col} BIGINT NOT NULL)"))
        # Insert a row to make it non-empty
        conn.execute(text(f"INSERT INTO {table} ({ts_col}) VALUES (1)"))

    ensure_monthly_partitions(eng, table, ts_col)

    with eng.begin() as conn:
        is_partitioned = conn.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM pg_partitioned_table pt JOIN pg_class c ON pt.partrelid=c.oid WHERE c.relname=:t)",
            ),
            {"t": table},
        ).scalar_one()
        # Should not partition non-empty table
        assert not bool(is_partitioned)
