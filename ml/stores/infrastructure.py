"""
Infrastructure helpers for ML stores: partitions and DB preflight checks.

Consolidates:
- partition_manager.PartitionManager (+ run_partition_maintenance)
- db_preflight.check_db_prereqs
"""

from __future__ import annotations

import logging
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from ml.core.db_engine import EngineManager


if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


logger = logging.getLogger(__name__)


class PartitionManager:
    """
    Manages PostgreSQL table partitions for ML stores.
    """

    def __init__(
        self,
        connection_string: str,
        tables: list[str] | None = None,
        months_ahead: int = 3,
        retention_months: int = 24,
        logger: logging.Logger | None = None,
    ):
        self.engine: Engine = EngineManager.get_engine(connection_string)
        self.tables = tables or [
            "ml_feature_values",
            "ml_model_predictions",
            "ml_strategy_signals",
        ]
        self.months_ahead = months_ahead
        self.retention_months = retention_months
        self.logger = logger or logging.getLogger(__name__)

    def create_future_partitions(self) -> int:
        created_count = 0
        current_date = datetime.now().date()
        with self.engine.begin() as conn:
            for table_name in self.tables:
                result = conn.execute(
                    text(
                        """
                        SELECT MAX(tablename) as last_partition
                        FROM pg_tables
                        WHERE schemaname = 'public'
                        AND tablename LIKE :pattern
                        ORDER BY tablename DESC
                    """,
                    ),
                    {"pattern": f"{table_name}_%"},
                )
                last_partition = result.scalar()
                if last_partition:
                    try:
                        date_str = last_partition.split("_")[-2:]
                        year = int(date_str[0])
                        month = int(date_str[1])
                        last_date = datetime(year, month, 1).date()
                    except (ValueError, IndexError):
                        last_date = datetime(current_date.year, current_date.month, 1).date()
                else:
                    last_date = datetime(current_date.year, current_date.month, 1).date()
                target_date = current_date + timedelta(days=self.months_ahead * 30)
                while last_date <= target_date:
                    partition_name = f"{table_name}_{last_date.year:04d}_{last_date.month:02d}"
                    exists_result = conn.execute(
                        text(
                            """
                            SELECT EXISTS (
                                SELECT 1 FROM pg_tables
                                WHERE schemaname = 'public'
                                AND tablename = :name
                            )
                        """,
                        ),
                        {"name": partition_name},
                    )
                    if not exists_result.scalar():
                        start_dt = datetime.combine(last_date.replace(day=1), time.min)
                        from ml.common.timestamps import sanitize_timestamp_ns

                        start_ns = sanitize_timestamp_ns(
                            int(start_dt.timestamp() * 1e9),
                            context="infrastructure.create_future_partitions.start",
                        )
                        if last_date.month == 12:
                            end_date_d: date = last_date.replace(
                                year=last_date.year + 1,
                                month=1,
                                day=1,
                            )
                        else:
                            end_date_d = last_date.replace(month=last_date.month + 1, day=1)
                        end_dt = datetime.combine(end_date_d, time.min)
                        end_ns = sanitize_timestamp_ns(
                            int(end_dt.timestamp() * 1e9),
                            context="infrastructure.create_future_partitions.end",
                        )
                        conn.execute(
                            text(
                                f"""
                                CREATE TABLE IF NOT EXISTS {partition_name}
                                PARTITION OF {table_name}
                                FOR VALUES FROM ({start_ns}) TO ({end_ns})
                                """,
                            ),
                        )
                        self.logger.info(f"Created partition {partition_name}")
                        created_count += 1
                    if last_date.month == 12:
                        last_date = last_date.replace(year=last_date.year + 1, month=1)
                    else:
                        last_date = last_date.replace(month=last_date.month + 1)
        return created_count

    def cleanup_old_partitions(self) -> int:
        removed_count = 0
        cutoff_date = datetime.now().date() - timedelta(days=self.retention_months * 30)
        with self.engine.begin() as conn:
            for table_name in self.tables:
                result = conn.execute(
                    text(
                        """
                        SELECT tablename
                        FROM pg_tables
                        WHERE schemaname = 'public'
                        AND tablename LIKE :pattern
                        """,
                    ),
                    {"pattern": f"{table_name}_%"},
                )
                for row in result:
                    partition_name = row[0]
                    try:
                        date_str = partition_name.split("_")[-2:]
                        year = int(date_str[0])
                        month = int(date_str[1])
                        partition_date = datetime(year, month, 1).date()
                        if partition_date < cutoff_date:
                            conn.execute(text(f"DROP TABLE IF EXISTS {partition_name} CASCADE"))
                            self.logger.info(f"Dropped old partition {partition_name}")
                            removed_count += 1
                    except (ValueError, IndexError):
                        continue
        return removed_count

    def get_partition_stats(self) -> dict[str, list[dict[str, str | int]]]:
        stats: dict[str, list[dict[str, str | int]]] = {}
        with self.engine.connect() as conn:
            for table_name in self.tables:
                result = conn.execute(
                    text(
                        """
                        SELECT
                            tablename,
                            pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size,
                            pg_total_relation_size(tablename::regclass) as size_bytes
                        FROM pg_tables
                        WHERE schemaname = 'public'
                        AND tablename LIKE :pattern
                        ORDER BY tablename
                        """,
                    ),
                    {"pattern": f"{table_name}_%"},
                )
                partitions: list[dict[str, str | int]] = []
                for row in result:
                    partitions.append({"name": row[0], "size": row[1], "size_bytes": row[2]})
                stats[table_name] = partitions
        return stats

    def ensure_current_partition(self, table_name: str) -> bool:
        current_date = datetime.now().date()
        partition_name = f"{table_name}_{current_date.year:04d}_{current_date.month:02d}"
        with self.engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM pg_tables
                        WHERE schemaname = 'public'
                        AND tablename = :name
                    )
                    """,
                ),
                {"name": partition_name},
            )
            if not result.scalar():
                start_dt = datetime.combine(current_date.replace(day=1), time.min)
                from ml.common.timestamps import sanitize_timestamp_ns

                start_ns = sanitize_timestamp_ns(
                    int(start_dt.timestamp() * 1e9),
                    context="infrastructure.ensure_current_partition.start",
                )
                if current_date.month == 12:
                    end_date_d = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    end_date_d = current_date.replace(month=current_date.month + 1, day=1)
                end_dt = datetime.combine(end_date_d, time.min)
                end_ns = sanitize_timestamp_ns(
                    int(end_dt.timestamp() * 1e9),
                    context="infrastructure.ensure_current_partition.end",
                )
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS {partition_name}
                        PARTITION OF {table_name}
                        FOR VALUES FROM ({start_ns}) TO ({end_ns})
                        """,
                    ),
                )
                self.logger.info(f"Created current partition {partition_name}")
                return True
        return False

    def create_test_partitions(
        self,
        start_year: int = 2023,
        start_month: int = 1,
        end_year: int = 2026,
        end_month: int = 12,
    ) -> int:
        created_count = 0
        with self.engine.begin() as conn:
            for table_name in self.tables:
                current_year = start_year
                current_month = start_month
                while (current_year < end_year) or (
                    current_year == end_year and current_month <= end_month
                ):
                    partition_name = f"{table_name}_{current_year:04d}_{current_month:02d}"
                    exists_result = conn.execute(
                        text(
                            """
                            SELECT EXISTS (
                                SELECT 1 FROM pg_tables
                                WHERE schemaname = 'public'
                                AND tablename = :name
                            )
                            """,
                        ),
                        {"name": partition_name},
                    )
                    if not exists_result.scalar():
                        start_date_d = datetime(current_year, current_month, 1)
                        from ml.common.timestamps import sanitize_timestamp_ns

                        start_ns = sanitize_timestamp_ns(
                            int(start_date_d.timestamp() * 1e9),
                            context="infrastructure.create_test_partitions.start",
                        )
                        if current_month == 12:
                            end_date = datetime(current_year + 1, 1, 1)
                        else:
                            end_date = datetime(current_year, current_month + 1, 1)
                        end_ns = sanitize_timestamp_ns(
                            int(end_date.timestamp() * 1e9),
                            context="infrastructure.create_test_partitions.end",
                        )
                        try:
                            conn.execute(
                                text(
                                    f"""
                                    CREATE TABLE IF NOT EXISTS {partition_name}
                                    PARTITION OF {table_name}
                                    FOR VALUES FROM ({start_ns}) TO ({end_ns})
                                    """,
                                ),
                            )
                            self.logger.info(f"Created test partition {partition_name}")
                            created_count += 1
                        except Exception as e:
                            if "already exists" not in str(e) and "overlap" not in str(e):
                                self.logger.warning(
                                    f"Failed to create partition {partition_name}: {e}",
                                )
                    if current_month == 12:
                        current_month = 1
                        current_year += 1
                    else:
                        current_month += 1
        self.logger.info(f"Created {created_count} test partitions")
        return created_count

    def run_maintenance(self) -> dict[str, int | str]:
        self.logger.info("Starting partition maintenance")
        for table in self.tables:
            self.ensure_current_partition(table)
        created = self.create_future_partitions()
        removed = self.cleanup_old_partitions()
        self.logger.info("Partition maintenance complete: created=%s, removed=%s", created, removed)
        return {"created": created, "removed": removed, "timestamp": datetime.now().isoformat()}


def run_partition_maintenance(
    connection_string: str,
    months_ahead: int = 3,
    retention_months: int = 24,
) -> None:
    manager = PartitionManager(
        connection_string=connection_string,
        months_ahead=months_ahead,
        retention_months=retention_months,
    )
    manager.run_maintenance()


# =============================================================================
# DB preflight (from db_preflight.py)
# =============================================================================


REQUIRED_FUNCTIONS = [
    "emit_data_event",
    "update_watermark",
]

PARTITIONED_TABLES = [
    "ml_feature_values",
    "ml_model_predictions",
    "ml_strategy_signals",
]


def _ensure_helper_functions(conn: Any) -> None:
    try:
        exists = conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_proc WHERE proname='create_monthly_partitions')"),
        ).scalar()
        if not bool(exists):
            conn.execute(
                text(
                    """
CREATE OR REPLACE FUNCTION create_monthly_partitions(
    table_name TEXT,
    start_date DATE,
    num_months INTEGER
)
RETURNS VOID AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_ns BIGINT;
    end_ns BIGINT;
BEGIN
    FOR i IN 0..num_months-1 LOOP
        partition_date := start_date + (i || ' months')::INTERVAL;
        partition_name := table_name || '_' || TO_CHAR(partition_date, 'YYYY_MM');
        start_ns := EXTRACT(EPOCH FROM partition_date) * 1000000000;
        end_ns := EXTRACT(EPOCH FROM partition_date + '1 month'::INTERVAL) * 1000000000;
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
            partition_name, table_name, start_ns, end_ns
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;
                    """,
                ),
            )
    except Exception as exc:
        logger.debug("Helper function ensure failed (ignored): %s", exc, exc_info=True)


def check_db_prereqs(connection_string: str) -> dict[str, bool | str]:
    engine = EngineManager.get_engine(connection_string)
    summary: dict[str, bool | str] = {"ok": True}
    try:
        with engine.begin() as conn:
            _ensure_helper_functions(conn)
            for fn in REQUIRED_FUNCTIONS:
                exists = conn.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM pg_proc WHERE proname = :fn
                        )
                        """,
                    ),
                    {"fn": fn},
                ).scalar()
                key = f"fn:{fn}"
                summary[key] = bool(exists)
                if not exists:
                    summary["ok"] = False
                    logger.warning("DB preflight: missing function %s", fn)

            today = date.today()
            part_suffix = f"_{today.year:04d}_{today.month:02d}"
            missing_any = False
            for table in PARTITIONED_TABLES:
                partition_name = table + part_suffix
                exists = conn.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM pg_tables
                            WHERE schemaname = 'public' AND tablename = :name
                        )
                        """,
                    ),
                    {"name": partition_name},
                ).scalar()
                key = f"partition:{partition_name}"
                summary[key] = bool(exists)
                if not exists:
                    summary["ok"] = False
                    logger.warning("DB preflight: missing partition %s", partition_name)
                    missing_any = True

            if missing_any:
                # Prefer PartitionManager as the single source of truth for partition creation
                try:
                    pm = PartitionManager(connection_string)
                    # Ensure current month exists for each table explicitly
                    for table in PARTITIONED_TABLES:
                        try:
                            created = pm.ensure_current_partition(table)
                            if created:
                                pname = f"{table}_{today.year:04d}_{today.month:02d}"
                                summary[f"partition:{pname}"] = True
                        except Exception as ensure_exc:
                            # Continue to future creation attempts even if single-table ensure fails
                            logger.debug(
                                "Ensure current partition failed for %s: %s",
                                table,
                                ensure_exc,
                                exc_info=True,
                            )
                    # Create a few months ahead for smooth operation
                    pm.create_future_partitions()

                    # Recheck current-month partitions for accuracy
                    today2 = date.today()
                    suffix2 = f"_{today2.year:04d}_{today2.month:02d}"
                    for table in PARTITIONED_TABLES:
                        pname = table + suffix2
                        ok2 = conn.execute(
                            text(
                                "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename=:n)",
                            ),
                            {"n": pname},
                        ).scalar()
                        summary[f"partition:{pname}"] = bool(ok2)
                        if not ok2:
                            summary["ok"] = False
                except Exception as exc:
                    logger.warning("Partition remediation via PartitionManager failed: %s", exc)

    except Exception as e:
        logger.error("DB preflight failed: %s", e)
        summary["ok"] = False
        summary["error"] = str(e)
    return summary


__all__ = [
    "PartitionManager",
    "check_db_prereqs",
    "run_partition_maintenance",
]
