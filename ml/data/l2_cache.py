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

import logging
import shutil
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ml._imports import pl as pl_runtime
from ml.data.cache_common import day_partition_path
from ml.data.cache_common import ensure_polars
from ml.data.cache_common import filter_df_by_ns_range
from ml.data.cache_common import iter_days
from ml.data.cache_common import resolve_cache_partition_path
from ml.data.cache_common import resolve_cache_write_symbol_dir
from ml.features.l2_aggregate import L2_MINUTE_COLUMNS
from ml.features.l2_aggregate import L2Aggregator


if TYPE_CHECKING:
    import polars as _pl

PL = cast(Any, pl_runtime)
logger = logging.getLogger(__name__)


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

    @staticmethod
    def _coerce_partition_columns(df: _pl.DataFrame) -> _pl.DataFrame:
        missing = [name for name in L2_MINUTE_COLUMNS if name not in df.columns]
        if missing:
            df = df.with_columns([PL.lit(0.0).alias(name) for name in missing if name != "timestamp"])
            if "timestamp" in missing:
                df = df.with_columns(
                    [
                        PL.lit(None)
                        .cast(PL.Datetime("ns", "UTC"))
                        .alias("timestamp"),
                    ],
                )
        return df.select(list(L2_MINUTE_COLUMNS))

    def path_for(self, symbol: str, day: date) -> Path:
        """
        Return the cache path for ``symbol`` and ``day``.

        Args:
            symbol: Instrument symbol (e.g., "SPY").
            day: UTC date for the partition.

        Returns:
            Absolute path to the cache parquet file for that day.

        """
        cache_symbol = resolve_cache_write_symbol_dir(symbol)
        return day_partition_path(self.cache_dir, cache_symbol, day)

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
        ensure_polars()
        existing, is_write = resolve_cache_partition_path(self.cache_dir, symbol, day)
        if existing is not None and self._is_valid_partition(existing, symbol, day):
            if is_write:
                return existing
            promoted = self._promote_partition(existing, self.path_for(symbol, day))
            return promoted
        out = self.path_for(symbol, day)
        invalid_existing = None
        if existing is not None and not self._is_valid_partition(existing, symbol, day):
            invalid_existing = existing
        if invalid_existing is not None:
            invalid_existing.unlink(missing_ok=True)
        # Compute from raw for the day
        start_dt = datetime(day.year, day.month, day.day, tzinfo=UTC)
        end_dt = start_dt + timedelta(days=1)
        agg = L2Aggregator(raw_base_dir)
        df = agg.compute_for_symbol(symbol, start=start_dt, end=end_dt)
        if df.is_empty():
            logger.debug(
                "l2_cache.partition_empty",
                extra={"symbol": symbol, "day": day.isoformat()},
            )
            return out
        else:
            df = self._coerce_partition_columns(df)
        if df.is_empty():
            logger.debug(
                "l2_cache.partition_empty",
                extra={"symbol": symbol, "day": day.isoformat()},
            )
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(out))
        return out

    def _is_valid_partition(self, path: Path, symbol: str, day: date) -> bool:
        try:
            existing = cast("_pl.DataFrame", PL.read_parquet(str(path)))
        except Exception:
            logger.debug(
                "l2_cache.read_existing_failed",
                exc_info=True,
                extra={"symbol": symbol, "day": day.isoformat(), "path": str(path)},
            )
            return False
        expected = set(L2_MINUTE_COLUMNS)
        return expected.issubset(set(existing.columns))

    def _promote_partition(self, source: Path, dest: Path) -> Path:
        if dest.exists() or source == dest:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, dest)
        except Exception:
            logger.debug(
                "l2_cache.promote_failed",
                exc_info=True,
                extra={"source": str(source), "dest": str(dest)},
            )
            return source
        return dest

    def get_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        raw_base_dir: Path,
        allow_compute: bool = True,
    ) -> _pl.DataFrame:
        """
        Get cached per-minute L2 features for ``symbol`` in [start, end).

        Missing day partitions are computed and cached on-demand when
        ``allow_compute`` is True.

        Args:
            symbol: Instrument symbol.
            start: Inclusive start datetime (UTC recommended).
            end: Exclusive end datetime (UTC recommended).
            raw_base_dir: Raw tier1 data root for fallback aggregation.
            allow_compute: When False, only read existing cache partitions and
                skip aggregation of missing days.

        Returns:
            Polars DataFrame with columns ``timestamp`` and L2 feature columns,
            sorted by timestamp and filtered to [start, end).

        """
        ensure_polars()
        parts: list[_pl.DataFrame] = []
        for day in iter_days(start, end):
            if allow_compute:
                p = self.ensure_day(symbol=symbol, day=day, raw_base_dir=raw_base_dir)
                if p.exists():
                    part = cast("_pl.DataFrame", PL.read_parquet(str(p)))
                    parts.append(self._coerce_partition_columns(part))
                continue
            existing, _ = resolve_cache_partition_path(self.cache_dir, symbol, day)
            if existing is None:
                continue
            if not self._is_valid_partition(existing, symbol, day):
                logger.debug(
                    "l2_cache.invalid_partition",
                    extra={"symbol": symbol, "day": day.isoformat(), "path": str(existing)},
                )
                continue
            part = cast("_pl.DataFrame", PL.read_parquet(str(existing)))
            parts.append(self._coerce_partition_columns(part))
        if not parts:
            return cast("_pl.DataFrame", PL.DataFrame({"timestamp": []}))
        df = PL.concat(parts, how="vertical")
        if df.is_empty():
            return cast("_pl.DataFrame", df)
        # Filter to exact [start, end) and sort
        return filter_df_by_ns_range(df, start=start, end=end)
