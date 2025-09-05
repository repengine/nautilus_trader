from __future__ import annotations

import os
from datetime import UTC
from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.observability.migrations import ensure_monthly_partitions

# Ensure DDL operations run without parallel interference (PostgreSQL locks can deadlock under xdist)
pytestmark = pytest.mark.serial


@pytest.mark.skipif(
    os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/nautilus_test"
    ).startswith("sqlite"),
    reason="PostgreSQL not available",
)
def test_partition_creation_on_empty_table() -> None:
    url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/nautilus_test")
    eng: Engine = create_engine(url)
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


@pytest.mark.skipif(
    os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/nautilus_test"
    ).startswith("sqlite"),
    reason="PostgreSQL not available",
)
def test_skip_partition_when_not_empty() -> None:
    url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/nautilus_test")
    eng: Engine = create_engine(url)
    table = "obs_part_test_metrics"
    ts_col = "timestamp"
    with eng.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        conn.execute(text(f"CREATE TABLE {table} ({ts_col} BIGINT NOT NULL)"))
        # Insert a row to make it non-empty
        conn.execute(text(f"INSERT INTO {table} ({ts_col}) VALUES (1)"))  # noqa: S608

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
