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
from ml.common import resolve_symbol_data_dir_candidates
from ml.common import resolve_symbol_data_dir_exact


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


def resolve_cache_symbol_dir(base_dir: Path, symbol: str) -> str:
    """
    Resolve the directory name used for cache partitions (write path).

    Prefers full instrument identifiers (e.g., ``SPY.XNAS``) to align parquet
    partitions with coverage entities.

    Args:
        base_dir: Cache root directory for symbol partitions.
        symbol: Instrument identifier (with or without venue suffix).

    Returns:
        Normalized cache directory name for writes.
    """
    normalized = resolve_cache_write_symbol_dir(symbol)
    if not normalized:
        return ""
    resolved = resolve_symbol_data_dir_exact(base_dir, normalized)
    return resolved.name if resolved is not None else normalized


def resolve_cache_write_symbol_dir(symbol: str) -> str:
    """
    Normalize a symbol to the cache write directory name.

    Args:
        symbol: Instrument identifier (with or without venue suffix).

    Returns:
        Normalized cache directory name.
    """
    normalized = symbol.strip().upper()
    return normalized


def resolve_cache_read_symbol_dirs(base_dir: Path, symbol: str) -> tuple[str, ...]:
    """
    Resolve cache directory names to read, preferring full then base symbols.

    Args:
        base_dir: Cache root directory for symbol partitions.
        symbol: Instrument identifier (with or without venue suffix).

    Returns:
        Tuple of directory names to search, ordered by preference.
    """
    normalized = resolve_cache_write_symbol_dir(symbol)
    if not normalized:
        return ()
    candidates = resolve_symbol_data_dir_candidates(base_dir, normalized)
    if candidates:
        return tuple(path.name for path in candidates)
    head, _, _tail = normalized.partition(".")
    base = head or normalized
    if base != normalized:
        return (normalized, base)
    return (normalized,)


def resolve_cache_partition_path(
    base_dir: Path,
    symbol: str,
    day: date,
) -> tuple[Path | None, bool]:
    """
    Resolve the existing cache partition path for ``symbol`` and ``day``.

    Returns a tuple of (path, is_write_path) where ``is_write_path`` indicates
    whether the resolved path matches the preferred write directory.

    Args:
        base_dir: Cache root directory for symbol partitions.
        symbol: Instrument identifier (with or without venue suffix).
        day: UTC day of the cache partition.

    Returns:
        Tuple of the resolved cache path (or ``None``) and whether it matches
        the write directory.
    """
    write_symbol = resolve_cache_write_symbol_dir(symbol)
    if write_symbol:
        write_path = day_partition_path(base_dir, write_symbol, day)
        if write_path.exists():
            return write_path, True
    for cache_symbol in resolve_cache_read_symbol_dirs(base_dir, symbol):
        if cache_symbol == write_symbol:
            continue
        candidate = day_partition_path(base_dir, cache_symbol, day)
        if candidate.exists():
            return candidate, False
    return None, False


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
    if df["timestamp"].dtype != PL.Datetime("ns", "UTC"):
        df = df.with_columns(PL.col("timestamp").cast(PL.Datetime("ns", "UTC")))
    start_ns = int(start.timestamp() * 1_000_000_000)
    end_ns = int(end.timestamp() * 1_000_000_000)
    return df.filter(
        (PL.col("timestamp").cast(PL.Int64) >= start_ns)
        & (PL.col("timestamp").cast(PL.Int64) < end_ns),
    ).sort("timestamp")
