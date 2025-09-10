"""
L2 per-minute feature cache utilities.

This module provides a caching layer for L2 (MBP-10) per-minute aggregates.
It allows building training datasets without re-aggregating large raw depth
files repeatedly, by persisting day-partitioned, per-minute features under a
stable location.

Directory layout (default):

    data/features/l2_minute/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet

Example:
    >>> from pathlib import Path
    >>> from datetime import datetime, timezone
    >>> cache = L2MinuteCache(Path("data/features/l2_minute"))
    >>> start = datetime(2025, 8, 11, tzinfo=timezone.utc)
    >>> end = datetime(2025, 8, 18, tzinfo=timezone.utc)
    >>> df = cache.get_range(
    ...     symbol="SPY",
    ...     start=start,
    ...     end=end,
    ...     raw_base_dir=Path("data/tier1"),
    ... )
    >>> df.columns  # doctest: +SKIP
    ['timestamp', 'midprice', 'spread_bps', 'microprice_bps', 'depth_imbalance_top1', ...]

All timestamps are UTC ns. When a requested day is not cached, the cache
computes the per-minute aggregates for that day using the L2Aggregator and
persists the result before returning the merged range.

"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.features.l2_aggregate import L2Aggregator


def _ensure_polars() -> None:
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])  # pragma: no cover


def _utc_day_bounds(dt: datetime) -> tuple[datetime, datetime]:
    """
    Return [start, end) bounds for the UTC day containing ``dt``.

    Args:
        dt: A timezone-aware datetime.

    Returns:
        Tuple of (day_start, day_end) datetimes in UTC.

    """
    day = datetime(dt.year, dt.month, dt.day, tzinfo=UTC)
    return day, day + timedelta(days=1)


def _iter_days(start: datetime, end: datetime) -> Iterator[date]:
    """
    Iterate over UTC days in [start, end).

    Args:
        start: Inclusive start datetime (UTC recommended).
        end: Exclusive end datetime (UTC recommended).

    Yields:
        Each date object spanning the interval.

    """
    cur = datetime(start.year, start.month, start.day, tzinfo=UTC)
    stop = datetime(end.year, end.month, end.day, tzinfo=UTC)
    if end.time() != datetime.min.time():
        stop = stop + timedelta(days=1)
    while cur < stop:
        yield cur.date()
        cur += timedelta(days=1)


@dataclass(slots=True)
class L2MinuteCache:
    """
    Cache for per-minute L2 aggregates.

    Attributes:
        cache_dir: Base directory for cached files.

    The cache directory layout is partitioned by year, month, and day to
    enable efficient incremental updates and reads.

    """

    cache_dir: Path

    def path_for(self, symbol: str, day: date) -> Path:
        """
        Return the cache path for ``symbol`` and ``day``.

        Args:
            symbol: Instrument symbol (e.g., "SPY").
            day: UTC date for the partition.

        Returns:
            Absolute path to the cache parquet file for that day.

        """
        y = f"year={day.year:04d}"
        m = f"month={day.month:02d}"
        d = f"day={day.day:02d}"
        return self.cache_dir / symbol / y / m / f"{d}.parquet"

    def ensure_day(
        self,
        symbol: str,
        day: date,
        raw_base_dir: Path,
    ) -> Path:
        """
        Ensure the cache for a specific ``symbol`` and ``day`` exists.

        If the parquet file is missing, compute the per-minute aggregates for
        that day from raw L2 and persist to the cache.

        Args:
            symbol: Instrument symbol.
            day: UTC date partition.
            raw_base_dir: Path to the raw tier1 data root (e.g., ``data/tier1``).

        Returns:
            Path to the cached parquet file.

        """
        _ensure_polars()
        out = self.path_for(symbol, day)
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        # Compute from raw for the day
        start_dt = datetime(day.year, day.month, day.day, tzinfo=UTC)
        end_dt = start_dt + timedelta(days=1)
        agg = L2Aggregator(raw_base_dir)
        df = agg.compute_for_symbol(symbol, start=start_dt, end=end_dt)
        if df.is_empty():
            # Create empty schema with timestamp to preserve partition
            df = pl.DataFrame({"timestamp": []})
        df.write_parquet(str(out))
        return out

    def get_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        raw_base_dir: Path,
    ) -> pl.DataFrame:
        """
        Get cached per-minute L2 features for ``symbol`` in [start, end).

        Missing day partitions are computed and cached on-demand.

        Args:
            symbol: Instrument symbol.
            start: Inclusive start datetime (UTC recommended).
            end: Exclusive end datetime (UTC recommended).
            raw_base_dir: Raw tier1 data root for fallback aggregation.

        Returns:
            Polars DataFrame with columns ``timestamp`` and L2 feature columns,
            sorted by timestamp and filtered to [start, end).

        """
        _ensure_polars()
        parts: list[pl.DataFrame] = []
        for day in _iter_days(start, end):
            p = self.ensure_day(symbol=symbol, day=day, raw_base_dir=raw_base_dir)
            if p.exists():
                parts.append(pl.read_parquet(str(p)))
        if not parts:
            return pl.DataFrame({"timestamp": []})
        df = pl.concat(parts, how="vertical")
        if df.is_empty():
            return df
        # Filter to exact [start, end) and sort
        if df["timestamp"].dtype != pl.Datetime:
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
        start_ns = int(start.timestamp() * 1_000_000_000)
        end_ns = int(end.timestamp() * 1_000_000_000)
        df = df.filter(
            (pl.col("timestamp").cast(pl.Int64) >= start_ns)
            & (pl.col("timestamp").cast(pl.Int64) < end_ns),
        ).sort("timestamp")
        return df
