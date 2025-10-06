"""Utility helpers for ML database interactions."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from collections.abc import Sequence
from datetime import date
from typing import Any, Final

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager


LOGGER = logging.getLogger(__name__)

__all__ = sorted(
    [
        "STORE_PARTITIONED_TABLES",
        "ensure_default_partition",
        "ensure_monthly_partitions",
        "ensure_partition_tables_ready",
        "get_default_pool_config",
        "get_or_create_engine",
    ]
)

STORE_PARTITIONED_TABLES: Final[tuple[str, ...]] = (
    "ml_feature_values",
    "ml_model_predictions",
    "ml_strategy_signals",
)

_VALID_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def get_default_pool_config() -> dict[str, Any]:
    """Return the default database pool configuration."""
    return {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    }


def get_or_create_engine(
    connection_string: str,
    *,
    pool_size: int | None = None,
    max_overflow: int | None = None,
    pool_pre_ping: bool = True,
    pool_recycle: int = 3600,
    **kwargs: Any,
) -> Engine:
    """Create or retrieve a SQLAlchemy engine with standardized settings."""
    if not connection_string:
        msg = "connection_string cannot be empty"
        raise ValueError(msg)

    defaults = get_default_pool_config()
    final_pool_size = defaults["pool_size"] if pool_size is None else pool_size
    final_max_overflow = defaults["max_overflow"] if max_overflow is None else max_overflow

    try:
        engine = EngineManager.get_engine(
            connection_string,
            pool_size=final_pool_size,
            max_overflow=final_max_overflow,
            pool_pre_ping=pool_pre_ping,
            pool_recycle=pool_recycle,
            **kwargs,
        )
    except Exception as exc:  # pragma: no cover - defensive log
        LOGGER.error("Failed to create database engine: %s", exc)
        raise RuntimeError(f"Database engine creation failed: {exc}") from exc

    host_hint = connection_string.split("@")[-1] if "@" in connection_string else connection_string
    LOGGER.debug(
        "Created engine for %s with pool_size=%d overflow=%d",
        host_hint,
        final_pool_size,
        final_max_overflow,
    )
    return engine


def ensure_default_partition(
    engine: Engine,
    table_name: str,
    *,
    schema: str = "public",
) -> None:
    """Ensure the DEFAULT partition exists for a partitioned table."""
    safe_schema = _validate_identifier(schema)
    safe_table = _validate_identifier(table_name)
    sql = text(
        f"CREATE TABLE IF NOT EXISTS {safe_schema}.{safe_table}_default "
        f"PARTITION OF {safe_schema}.{safe_table} DEFAULT"
    )

    with engine.begin() as conn:
        conn.execute(text(f"SET LOCAL search_path TO {safe_schema}, pg_catalog"))
        conn.execute(sql)


def ensure_monthly_partitions(
    engine: Engine,
    table_name: str,
    *,
    schema: str = "public",
    start_date: date | None = None,
    months_ahead: int = 6,
) -> None:
    """Ensure partitions exist for each month starting at *start_date*."""
    if months_ahead < 0:
        msg = "months_ahead must be non-negative"
        raise ValueError(msg)

    safe_table = _validate_identifier(table_name)
    safe_schema = _validate_identifier(schema)
    start = start_date or _first_day_of_current_month()

    stmt = text(
        "SELECT create_monthly_partitions(:table_name, :start_date, :months)"
    )

    with engine.begin() as conn:
        conn.execute(text(f"SET LOCAL search_path TO {safe_schema}, pg_catalog"))
        try:
            conn.execute(
                stmt,
                {
                    "table_name": safe_table,
                    "start_date": start,
                    "months": months_ahead,
                },
            )
        except Exception as exc:  # pragma: no cover - helper may be absent
            LOGGER.debug(
                "create_monthly_partitions unavailable for %s.%s: %s",
                safe_schema,
                safe_table,
                exc,
            )


def ensure_partition_tables_ready(
    engine: Engine,
    table_names: Sequence[str],
    *,
    schema: str = "public",
    start_date: date | None = None,
    months_ahead: int = 6,
) -> None:
    """Ensure default partitions and near-term monthly partitions exist."""
    if months_ahead < 0:
        msg = "months_ahead must be non-negative"
        raise ValueError(msg)

    safe_schema = _validate_identifier(schema)
    tables: Iterable[str] = tuple(_validate_identifier(name) for name in table_names)
    start = start_date or _first_day_of_current_month()

    with engine.begin() as conn:
        conn.execute(text(f"SET LOCAL search_path TO {safe_schema}, pg_catalog"))
        for table in tables:
            conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {table}_default PARTITION OF {table} DEFAULT"
                )
            )
            try:
                conn.execute(
                    text(
                        "SELECT create_monthly_partitions(:table_name, :start_date, :months)"
                    ),
                    {
                        "table_name": table,
                        "start_date": start,
                        "months": months_ahead,
                    },
                )
            except Exception as exc:  # pragma: no cover - helper may be absent
                LOGGER.debug(
                    "create_monthly_partitions unavailable for %s.%s: %s",
                    safe_schema,
                    table,
                    exc,
                )


def _validate_identifier(identifier: str) -> str:
    if not _VALID_IDENTIFIER.match(identifier):
        msg = f"Invalid SQL identifier: {identifier!r}"
        raise ValueError(msg)
    return identifier


def _first_day_of_current_month() -> date:
    today = date.today()
    return today.replace(day=1)
