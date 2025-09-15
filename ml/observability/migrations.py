from __future__ import annotations

from datetime import UTC
from datetime import datetime
from typing import Final

from sqlalchemy import text
from sqlalchemy.engine import Engine


OBS_TABLES: Final[dict[str, str]] = {
    "obs_latency_watermarks": "ts_stage_end",
    "obs_metrics": "timestamp",
    "obs_event_correlation": "ts_event",
    "obs_health_scores": "timestamp",
}


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
            # BRIN index on timestamp columns
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {table}_{ts_col}_brin "
                    f"ON {table} USING BRIN ({ts_col});",
                ),
            )
        # Composite indices for common lookups
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS obs_event_correlation_instrument_ts_idx "
                "ON obs_event_correlation (instrument_id, ts_event);",
            ),
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS obs_metrics_name_ts_idx "
                "ON obs_metrics (metric_name, timestamp);",
            ),
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
        # Determine existence and row count
        exists = conn.execute(
            text("SELECT to_regclass(:t) IS NOT NULL"),
            {"t": table},
        ).scalar_one()

        def _create_parent() -> None:
            # Create as partitioned by RANGE on timestamp column
            conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {table} (LIKE {table}_template INCLUDING ALL)",
                ),
            )
            # Fallback if template not present: define minimal schema with ts column
            conn.execute(
                text(
                    f"DO $$ BEGIN\n"
                    f"IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table}') THEN\n"
                    f"EXECUTE 'CREATE TABLE {table} ({ts_col} BIGINT NOT NULL) PARTITION BY RANGE ({ts_col})';\n"
                    f"END IF;\nEND $$;",
                ),
            )

        is_partitioned = False
        if exists:
            is_partitioned = bool(
                conn.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM pg_partitioned_table pt JOIN pg_class c ON pt.partrelid=c.oid WHERE c.relname=:t)",
                    ),
                    {"t": table},
                ).scalar_one(),
            )
            if not is_partitioned:
                rowcount_val = int(conn.execute(text(f"SELECT COUNT(1) FROM {table}")).scalar_one())
                if rowcount_val == 0:
                    conn.execute(text(f"DROP TABLE {table} CASCADE"))
                    # Create partitioned parent
                    conn.execute(
                        text(
                            f"CREATE TABLE {table} ({ts_col} BIGINT NOT NULL) PARTITION BY RANGE ({ts_col})",
                        ),
                    )
                    is_partitioned = True
        else:
            # Fresh create
            conn.execute(
                text(
                    f"CREATE TABLE {table} ({ts_col} BIGINT NOT NULL) PARTITION BY RANGE ({ts_col})",
                ),
            )
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
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS "
                    f"{part_name} PARTITION OF {table} FOR VALUES FROM (:start) TO (:end)",
                ),
                {"start": start_ns, "end": end_ns},
            )
            # BRIN index on partition child
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {part_name}_{ts_col}_brin ON {part_name} USING BRIN ({ts_col})",
                ),
            )


def apply_observability_monthly_partitions(engine: Engine) -> None:
    """
    Apply monthly partitioning to all observability tables when possible.
    """
    if engine.dialect.name != "postgresql":  # pragma: no cover
        return
    for table, ts_col in OBS_TABLES.items():
        ensure_monthly_partitions(engine, table, ts_col)
