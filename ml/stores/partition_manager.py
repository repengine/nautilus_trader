"""
Partition management utilities for ML stores.

This module provides automatic partition creation and maintenance for time-series tables
in the ML pipeline.

"""

from __future__ import annotations

import logging
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from typing import TYPE_CHECKING

from sqlalchemy import text

from ml.core.db_engine import EngineManager


if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


class PartitionManager:
    """
    Manages PostgreSQL table partitions for ML stores.

    Provides automatic partition creation, cleanup, and monitoring for time-partitioned
    tables.

    """

    def __init__(
        self,
        connection_string: str,
        tables: list[str] | None = None,
        months_ahead: int = 3,
        retention_months: int = 24,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize partition manager.

        Parameters
        ----------
        connection_string : str
            PostgreSQL connection string
        tables : list[str] | None
            Tables to manage (defaults to all ML tables)
        months_ahead : int
            Number of months to create partitions in advance
        retention_months : int
            Number of months to retain old partitions
        logger : logging.Logger | None
            Optional logger instance

        """
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
        """
        Create partitions for future months.

        Returns
        -------
        int
            Number of partitions created

        """
        created_count = 0
        current_date = datetime.now().date()

        with self.engine.connect() as conn:
            for table_name in self.tables:
                # Find last existing partition
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
                    # Extract date from partition name (format: table_YYYY_MM)
                    try:
                        date_str = last_partition.split("_")[-2:]
                        year = int(date_str[0])
                        month = int(date_str[1])
                        last_date = datetime(year, month, 1).date()
                    except (ValueError, IndexError):
                        # If parsing fails, start from current month
                        last_date = datetime(current_date.year, current_date.month, 1).date()
                else:
                    # No partitions exist, start from current month
                    last_date = datetime(current_date.year, current_date.month, 1).date()

                # Create partitions up to months_ahead in the future
                target_date = current_date + timedelta(days=self.months_ahead * 30)

                while last_date <= target_date:
                    partition_name = f"{table_name}_{last_date.year:04d}_{last_date.month:02d}"

                    # Check if partition exists
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
                        # Calculate nanosecond timestamps (date -> datetime at midnight)
                        start_dt = datetime.combine(last_date.replace(day=1), time.min)
                        start_ns = int(start_dt.timestamp() * 1e9)

                        # Calculate end of month
                        if last_date.month == 12:
                            end_date_d: date = last_date.replace(
                                year=last_date.year + 1,
                                month=1,
                                day=1,
                            )
                        else:
                            end_date_d = last_date.replace(month=last_date.month + 1, day=1)
                        end_dt = datetime.combine(end_date_d, time.min)
                        end_ns = int(end_dt.timestamp() * 1e9)

                        # Create partition
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
                        conn.commit()

                    # Move to next month
                    if last_date.month == 12:
                        last_date = last_date.replace(year=last_date.year + 1, month=1)
                    else:
                        last_date = last_date.replace(month=last_date.month + 1)

        return created_count

    def cleanup_old_partitions(self) -> int:
        """
        Remove partitions older than retention period.

        Returns
        -------
        int
            Number of partitions removed

        """
        removed_count = 0
        cutoff_date = datetime.now().date() - timedelta(days=self.retention_months * 30)

        with self.engine.connect() as conn:
            for table_name in self.tables:
                # Find old partitions
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
                        # Extract date from partition name
                        date_str = partition_name.split("_")[-2:]
                        year = int(date_str[0])
                        month = int(date_str[1])
                        partition_date = datetime(year, month, 1).date()

                        if partition_date < cutoff_date:
                            # Drop old partition
                            conn.execute(
                                text(f"DROP TABLE IF EXISTS {partition_name} CASCADE"),
                            )
                            self.logger.info(f"Dropped old partition {partition_name}")
                            removed_count += 1
                            conn.commit()
                    except (ValueError, IndexError):
                        # Skip if date parsing fails
                        continue

        return removed_count

    def get_partition_stats(self) -> dict[str, list[dict[str, str | int]]]:
        """
        Get statistics about existing partitions.

        Returns
        -------
        dict[str, list[dict[str, str]]]
            Partition information by table

        """
        stats = {}

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
                    partitions.append(
                        {
                            "name": row[0],
                            "size": row[1],
                            "size_bytes": row[2],
                        },
                    )

                stats[table_name] = partitions

        return stats

    def ensure_current_partition(self, table_name: str) -> bool:
        """
        Ensure partition exists for current month.

        Parameters
        ----------
        table_name : str
            Table name to check

        Returns
        -------
        bool
            True if partition was created, False if already existed

        """
        current_date = datetime.now().date()
        partition_name = f"{table_name}_{current_date.year:04d}_{current_date.month:02d}"

        with self.engine.connect() as conn:
            # Check if partition exists
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
                # Create partition for current month
                start_dt = datetime.combine(current_date.replace(day=1), time.min)
                start_ns = int(start_dt.timestamp() * 1e9)

                # Calculate end of month
                if current_date.month == 12:
                    end_date_d = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    end_date_d = current_date.replace(month=current_date.month + 1, day=1)
                end_dt = datetime.combine(end_date_d, time.min)
                end_ns = int(end_dt.timestamp() * 1e9)

                conn.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS {partition_name}
                        PARTITION OF {table_name}
                        FOR VALUES FROM ({start_ns}) TO ({end_ns})
                    """,
                    ),
                )
                conn.commit()

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
        """
        Create partitions for a specific date range (used for testing).

        Parameters
        ----------
        start_year : int
            Starting year for partitions
        start_month : int
            Starting month for partitions
        end_year : int
            Ending year for partitions
        end_month : int
            Ending month for partitions

        Returns
        -------
        int
            Number of partitions created

        """
        created_count = 0

        with self.engine.connect() as conn:
            for table_name in self.tables:
                # Iterate through the date range
                current_year = start_year
                current_month = start_month

                while (current_year < end_year) or (
                    current_year == end_year and current_month <= end_month
                ):
                    partition_name = f"{table_name}_{current_year:04d}_{current_month:02d}"

                    # Check if partition exists
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
                        # Calculate nanosecond timestamps
                        start_date = datetime(current_year, current_month, 1)
                        start_ns = int(start_date.timestamp() * 1e9)

                        # Calculate end of month
                        if current_month == 12:
                            end_date = datetime(current_year + 1, 1, 1)
                        else:
                            end_date = datetime(current_year, current_month + 1, 1)
                        end_ns = int(end_date.timestamp() * 1e9)

                        try:
                            # Create partition
                            conn.execute(
                                text(
                                    f"""
                                    CREATE TABLE IF NOT EXISTS {partition_name}
                                    PARTITION OF {table_name}
                                    FOR VALUES FROM ({start_ns}) TO ({end_ns})
                                """,
                                ),
                            )
                            conn.commit()
                            self.logger.info(f"Created test partition {partition_name}")
                            created_count += 1
                        except Exception as e:
                            # Ignore duplicate partition errors
                            if "already exists" not in str(e) and "overlap" not in str(e):
                                self.logger.warning(
                                    f"Failed to create partition {partition_name}: {e}",
                                )
                            conn.rollback()

                    # Move to next month
                    if current_month == 12:
                        current_month = 1
                        current_year += 1
                    else:
                        current_month += 1

        self.logger.info(f"Created {created_count} test partitions")
        return created_count

    def run_maintenance(self) -> dict[str, int | str]:
        """
        Run full partition maintenance.

        Creates future partitions and cleans up old ones.

        Returns
        -------
        dict[str, int]
            Maintenance statistics

        """
        self.logger.info("Starting partition maintenance")

        # Ensure current partitions exist
        for table in self.tables:
            self.ensure_current_partition(table)

        # Create future partitions
        created = self.create_future_partitions()

        # Cleanup old partitions
        removed = self.cleanup_old_partitions()

        self.logger.info(
            f"Partition maintenance complete: created={created}, removed={removed}",
        )

        return {
            "created": created,
            "removed": removed,
            "timestamp": datetime.now().isoformat(),
        }


# Standalone function for use in schedulers
def run_partition_maintenance(
    connection_string: str,
    months_ahead: int = 3,
    retention_months: int = 24,
) -> None:
    """
    Run partition maintenance as a standalone task.

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string
    months_ahead : int
        Number of months to create partitions in advance
    retention_months : int
        Number of months to retain old partitions

    """
    manager = PartitionManager(
        connection_string=connection_string,
        months_ahead=months_ahead,
        retention_months=retention_months,
    )
    manager.run_maintenance()
