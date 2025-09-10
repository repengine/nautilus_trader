"""
Microstructure per-minute feature cache utilities.

This module provides a caching layer for L1/L0-derived per-minute microstructure
features to avoid recomputation. It mirrors :mod:`ml.data.l2_cache` semantics.

Directory layout (default):

    data/features/micro_minute/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet

Example:
    >>> from pathlib import Path
    >>> from datetime import datetime, timezone
    >>> cache = MicroMinuteCache(Path("data/features/micro_minute"))
    >>> start = datetime(2025, 8, 11, tzinfo=timezone.utc)
    >>> end = datetime(2025, 8, 18, tzinfo=timezone.utc)
    >>> df = cache.get_range(
    ...     symbol="SPY",
    ...     start=start,
    ...     end=end,
    ...     raw_base_dir=Path("data/tier1"),
    ... )
    >>> df.columns  # doctest: +SKIP
    ['timestamp', '... micro features ...']

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
from ml.features.micro_aggregate import MicrostructureAggregator


def _ensure_polars() -> None:
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])  # pragma: no cover


def _iter_days(start: datetime, end: datetime) -> Iterator[date]:
    cur = datetime(start.year, start.month, start.day, tzinfo=UTC)
    stop = datetime(end.year, end.month, end.day, tzinfo=UTC)
    if end.time() != datetime.min.time():
        stop = stop + timedelta(days=1)
    while cur < stop:
        yield cur.date()
        cur += timedelta(days=1)


@dataclass(slots=True)
class MicroMinuteCache:
    """
    Cache for per-minute microstructure features.

    Attributes:
        cache_dir: Base directory for cached files.

    """

    cache_dir: Path

    def path_for(self, symbol: str, day: date) -> Path:
        y = f"year={day.year:04d}"
        m = f"month={day.month:02d}"
        d = f"day={day.day:02d}"
        return self.cache_dir / symbol / y / m / f"{d}.parquet"

    def ensure_day(self, symbol: str, day: date, raw_base_dir: Path) -> Path:
        _ensure_polars()
        out = self.path_for(symbol, day)
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        # Compute full micro features and slice to the day for now.
        # NOTE: MicrostructureAggregator may compute a broad range; this cache
        # still avoids repeated runs for subsequent builds.
        agg = MicrostructureAggregator(raw_base_dir)
        df = agg.compute_for_symbol(symbol)
        if df.is_empty():
            df = pl.DataFrame({"timestamp": []})
        elif df["timestamp"].dtype != pl.Datetime:
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
        # Filter to exact day window
        start_dt = datetime(day.year, day.month, day.day, tzinfo=UTC)
        end_dt = start_dt + timedelta(days=1)
        start_ns = int(start_dt.timestamp() * 1_000_000_000)
        end_ns = int(end_dt.timestamp() * 1_000_000_000)
        df = df.filter(
            (pl.col("timestamp").cast(pl.Int64) >= start_ns)
            & (pl.col("timestamp").cast(pl.Int64) < end_ns),
        ).sort("timestamp")
        df.write_parquet(str(out))
        return out

    def get_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        raw_base_dir: Path,
    ) -> pl.DataFrame:
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
