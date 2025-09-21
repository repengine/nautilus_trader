"""
Common helpers for cold-path, per-minute cache modules.

This module centralizes small, repeated utilities used by l2_cache and micro_cache to
keep those modules focused on their domain-specific logic.

All utilities here are cold-path only.

"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl as pl_runtime


if TYPE_CHECKING:  # Typed-only aliases
    import polars as _pl

# Local runtime alias to avoid Optional[Module] union typing at use sites
PL: Any = cast(Any, pl_runtime)


def ensure_polars() -> None:
    """
    Ensure Polars is available before use (cold-path).

    Raises a helpful error via the project import shim when Polars is missing.

    """
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])  # pragma: no cover


def iter_days(start: datetime, end: datetime) -> Iterator[date]:
    """
    Iterate over UTC dates covering the half-open interval [start, end).

    The iteration is day-aligned in UTC and includes the day containing
    ``start``; if ``end`` is not at 00:00, the day containing ``end`` is
    also included to ensure full coverage before post-filtering.

    """
    cur = datetime(start.year, start.month, start.day, tzinfo=UTC)
    stop = datetime(end.year, end.month, end.day, tzinfo=UTC)
    if end.time() != datetime.min.time():
        stop = stop + timedelta(days=1)
    while cur < stop:
        yield cur.date()
        cur += timedelta(days=1)


def day_partition_path(base: Path, symbol: str, day: date) -> Path:
    """
    Build a year/month/day partitioned parquet path under ``base``.

    Layout: ``<base>/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet``

    """
    y = f"year={day.year:04d}"
    m = f"month={day.month:02d}"
    d = f"day={day.day:02d}"
    return base / symbol / y / m / f"{d}.parquet"


def filter_df_by_ns_range(
    df: _pl.DataFrame,
    start: datetime,
    end: datetime,
) -> _pl.DataFrame:
    """
    Filter a Polars frame to [start, end) by ``timestamp`` and sort.

    Accepts either integer ns timestamps or pl.Datetime("ns", "UTC"). Ensures the column
    is cast to pl.Datetime("ns", "UTC") for consistency.

    """
    ensure_polars()
    if df.is_empty():
        return df
    if df["timestamp"].dtype != PL.Datetime:
        df = df.with_columns(PL.col("timestamp").cast(PL.Datetime("ns", "UTC")))
    start_ns = int(start.timestamp() * 1_000_000_000)
    end_ns = int(end.timestamp() * 1_000_000_000)
    return df.filter(
        (PL.col("timestamp").cast(PL.Int64) >= start_ns)
        & (PL.col("timestamp").cast(PL.Int64) < end_ns),
    ).sort("timestamp")
