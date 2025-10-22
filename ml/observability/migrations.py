from __future__ import annotations

from datetime import UTC
from datetime import datetime
from typing import Final

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.engine import Engine


OBS_TABLES: Final[dict[str, str]] = {
    "obs_latency_watermarks": "ts_stage_end",
    "obs_metrics": "timestamp",
    "obs_event_correlation": "ts_event",
    "obs_health_scores": "timestamp",
}


def _create_index(
    conn: Connection,
    index_name: str,
    table_name: str,
    columns: tuple[str, ...],
    *,
    using: str | None = None,
) -> None:
    """
    Create an index if it does not exist using safe identifier quoting.
    """
    preparer = conn.dialect.identifier_preparer
    columns_expr = ", ".join(preparer.quote(col) for col in columns)
    using_clause = ""
    if using is not None:
        allowed = {"BRIN", "BTREE", "HASH", "GIN", "GIST"}
        using_upper = using.upper()
        if using_upper not in allowed:
            raise ValueError(f"Unsupported index method '{using}'")
        using_clause = f" USING {using_upper}"

    conn.execute(
        text(
            """
            DO $$
            BEGIN
                EXECUTE format(
                    'CREATE INDEX IF NOT EXISTS %I ON %I%s (%s)',
                    :index_name,
                    :table_name,
                    :using_clause,
                    :columns_expr
                );
            END $$;
            """,
        ),
        {
            "index_name": index_name,
            "table_name": table_name,
            "using_clause": using_clause,
            "columns_expr": columns_expr,
        },
    )


def _create_partitioned_parent(conn: Connection, table_name: str, ts_col: str) -> None:
    """Create a partitioned parent table if it does not already exist."""
    conn.execute(
        text(
            f"CREATE TABLE IF NOT EXISTS {table_name} ({ts_col} BIGINT NOT NULL) PARTITION BY RANGE ({ts_col})",
        ),
    )


def _drop_table_cascade(conn: Connection, table_name: str) -> None:
    """
    Drop a table (if it exists) using CASCADE semantics.
    """
    conn.execute(
        text(
            """
            DO $$
            BEGIN
                EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', :table_name);
            END $$;
            """,
        ),
        {"table_name": table_name},
    )


def _create_partition(
    conn: Connection,
    parent_name: str,
    partition_name: str,
    start_ns: int,
    end_ns: int,
) -> None:
    """
    Create a range partition for the provided bounds if it does not exist.
    """
    conn.execute(
        text(
            """
            DO $$
            BEGIN
                EXECUTE format(
                    'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%s) TO (%s)',
                    :partition_name,
                    :parent_name,
                    :start_bound,
                    :end_bound
                );
            END $$;
            """,
        ),
        {
            "partition_name": partition_name,
            "parent_name": parent_name,
            "start_bound": str(start_ns),
            "end_bound": str(end_ns),
        },
    )


def apply_observability_indices(engine: Engine) -> None:
    """
    Apply BRIN and composite indices for observability tables when on PostgreSQL.

    Safe to run multiple times. Partitioning support can be added in a future migration;
    this function focuses on low-cost indexing improvements.

    """
    if engine.dialect.name != "postgresql":  # no-op on other backends
        return

    with engine.begin() as conn:
        for table, ts_col in OBS_TABLES.items():
            _create_index(conn, f"{table}_{ts_col}_brin", table, (ts_col,), using="BRIN")

        _create_index(
            conn,
            "obs_event_correlation_instrument_ts_idx",
            "obs_event_correlation",
            ("instrument_id", "ts_event"),
        )
        _create_index(
            conn,
            "obs_metrics_name_ts_idx",
            "obs_metrics",
            ("metric_name", "timestamp"),
        )


__all__ = ["apply_observability_indices"]


def _month_bounds(dt: datetime) -> tuple[datetime, datetime]:
    year = dt.year
    month = dt.month
    start = datetime(year, month, 1, tzinfo=UTC)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(year, month + 1, 1, tzinfo=UTC)
    return start, end


def ensure_monthly_partitions(engine: Engine, table: str, ts_col: str) -> None:
    """
    Ensure `table` is partitioned monthly on `ts_col` and that partitions exist for the
    current and next month.

    If the existing table is not partitioned and is empty, it will be dropped and
    recreated as a partitioned parent to avoid data migration complexity. No-op on non-
    PostgreSQL backends.

    """
    if engine.dialect.name != "postgresql":  # pragma: no cover - backend-specific
        return

    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT to_regclass(:t) IS NOT NULL"),
            {"t": table},
        ).scalar_one()

        is_partitioned = False
        if exists:
            is_partitioned = bool(
                conn.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM pg_partitioned_table pt
                            JOIN pg_class c ON pt.partrelid = c.oid
                            WHERE c.relname = :t
                        )
                        """,
                    ),
                    {"t": table},
                ).scalar_one(),
            )
            if not is_partitioned:
                preparer = conn.dialect.identifier_preparer
                table_ident = preparer.quote(table)
                count_stmt = text(f"SELECT COUNT(1) FROM {table_ident}")
                rowcount_val = int(conn.execute(count_stmt).scalar_one())
                if rowcount_val == 0:
                    _drop_table_cascade(conn, table)
                    _create_partitioned_parent(conn, table, ts_col)
                    is_partitioned = True
        else:
            _create_partitioned_parent(conn, table, ts_col)
            is_partitioned = True

        if not is_partitioned:
            return

        # Ensure partitions for current and next month
        now = datetime.now(UTC)
        for dt in (
            now,
            datetime(now.year + (now.month // 12), ((now.month % 12) + 1), 1, tzinfo=UTC),
        ):
            start, end = _month_bounds(dt)
            part_name = f"{table}_{start.strftime('%Y_%m')}"
            from ml.common.timestamps import sanitize_timestamp_ns

            start_ns = sanitize_timestamp_ns(
                int(start.timestamp() * 1_000_000_000),
                context=f"obs.ensure_monthly_partitions.{table}.start",
            )
            end_ns = sanitize_timestamp_ns(
                int(end.timestamp() * 1_000_000_000),
                context=f"obs.ensure_monthly_partitions.{table}.end",
            )
            _create_partition(conn, table, part_name, start_ns, end_ns)
            _create_index(conn, f"{part_name}_{ts_col}_brin", part_name, (ts_col,), using="BRIN")


def apply_observability_monthly_partitions(engine: Engine) -> None:
    """
    Apply monthly partitioning to all observability tables when possible.
    """
    if engine.dialect.name != "postgresql":  # pragma: no cover
        return
    for table, ts_col in OBS_TABLES.items():
        ensure_monthly_partitions(engine, table, ts_col)
