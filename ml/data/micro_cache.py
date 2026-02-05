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

import logging
import shutil
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from ml._imports import pl
from ml.data.cache_common import day_partition_path
from ml.data.cache_common import ensure_polars
from ml.data.cache_common import filter_df_by_ns_range
from ml.data.cache_common import iter_days
from ml.data.cache_common import resolve_cache_partition_path
from ml.data.cache_common import resolve_cache_write_symbol_dir
from ml.features.micro_aggregate import MICRO_COLUMNS
from ml.features.micro_aggregate import MicrostructureAggregator
from ml.ml_types import PolarsDF


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MicroMinuteCache:
    """
    Cache for per-minute microstructure features.

    Attributes:
        cache_dir: Base directory for cached files.

    """

    cache_dir: Path

    def path_for(self, symbol: str, day: date) -> Path:
        cache_symbol = resolve_cache_write_symbol_dir(symbol)
        return day_partition_path(self.cache_dir, cache_symbol, day)

    def is_valid_partition(self, path: Path) -> bool:
        """
        Return True when the cache partition contains real microstructure data.

        Args:
            path: Path to a cached parquet partition.

        Returns:
            True when the partition contains at least one row and expected columns.
        """
        ensure_polars()
        _pl = pl
        assert _pl is not None
        try:
            df = _pl.read_parquet(str(path), n_rows=1)
        except Exception:
            logger.debug(
                "micro_cache.read_existing_failed",
                exc_info=True,
                extra={"path": str(path)},
            )
            return False
        if df.is_empty():
            return False
        required = {"timestamp", *MICRO_COLUMNS}
        return required.issubset(set(df.columns))

    def ensure_day(self, symbol: str, day: date, raw_base_dir: Path) -> Path:
        ensure_polars()
        _pl = pl
        assert _pl is not None
        existing, is_write = resolve_cache_partition_path(self.cache_dir, symbol, day)
        invalid_existing = None
        if existing is not None:
            if not self.is_valid_partition(existing):
                invalid_existing = existing
            else:
                if is_write:
                    return existing
                promoted = self._promote_partition(existing, self.path_for(symbol, day))
                return promoted
        out = self.path_for(symbol, day)
        # Compute full micro features and slice to the day for now.
        # NOTE: MicrostructureAggregator may compute a broad range; this cache
        # still avoids repeated runs for subsequent builds.
        agg = MicrostructureAggregator(raw_base_dir)
        start_dt = datetime(day.year, day.month, day.day, tzinfo=UTC)
        end_dt = start_dt + timedelta(days=1)
        df = agg.compute_for_symbol(symbol, start=start_dt, end=end_dt)
        if df.is_empty():
            if invalid_existing is not None:
                invalid_existing.unlink(missing_ok=True)
            logger.debug(
                "micro_cache.partition_empty",
                extra={"symbol": symbol, "day": day.isoformat()},
            )
            return out
        elif df["timestamp"].dtype != _pl.Datetime("ns", "UTC"):
            df = df.with_columns(_pl.col("timestamp").cast(_pl.Datetime("ns", "UTC")))
        # Filter to exact day window
        start_ns = int(start_dt.timestamp() * 1_000_000_000)
        end_ns = int(end_dt.timestamp() * 1_000_000_000)
        df = df.filter(
            (_pl.col("timestamp").cast(_pl.Int64) >= start_ns)
            & (_pl.col("timestamp").cast(_pl.Int64) < end_ns),
        ).sort("timestamp")
        if df.is_empty():
            if invalid_existing is not None:
                invalid_existing.unlink(missing_ok=True)
            logger.debug(
                "micro_cache.partition_empty",
                extra={"symbol": symbol, "day": day.isoformat()},
            )
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(out))
        return out

    def _promote_partition(self, source: Path, dest: Path) -> Path:
        if dest.exists() or source == dest:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, dest)
        except Exception:
            logger.debug(
                "micro_cache.promote_failed",
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
    ) -> PolarsDF:
        """
        Return cached per-minute microstructure features for ``symbol`` in [start, end).

        Args:
            symbol: Instrument symbol (e.g., "SPY").
            start: Inclusive start datetime (UTC recommended).
            end: Exclusive end datetime (UTC recommended).
            raw_base_dir: Raw tier1 data root used for cache backfills.
            allow_compute: When False, read only existing cache partitions
                without aggregating from raw data.

        Returns:
            Polars DataFrame with microstructure columns filtered to [start, end).
        """
        ensure_polars()
        _pl = pl
        assert _pl is not None
        parts: list[PolarsDF] = []
        for day in iter_days(start, end):
            if allow_compute:
                p = self.ensure_day(symbol=symbol, day=day, raw_base_dir=raw_base_dir)
                if p.exists():
                    parts.append(_pl.read_parquet(str(p)))
                continue
            existing, _ = resolve_cache_partition_path(self.cache_dir, symbol, day)
            if existing is None:
                continue
            if not self.is_valid_partition(existing):
                logger.debug(
                    "micro_cache.invalid_partition",
                    extra={"symbol": symbol, "day": day.isoformat(), "path": str(existing)},
                )
                continue
            parts.append(_pl.read_parquet(str(existing)))
        if not parts:
            from typing import cast as _cast

            return _cast(PolarsDF, _pl.DataFrame({"timestamp": []}))
        df = _pl.concat(parts, how="vertical")
        if df.is_empty():
            from typing import cast as _cast

            return _cast(PolarsDF, df)
        # Filter to exact [start, end) and sort
        return filter_df_by_ns_range(df, start=start, end=end)
