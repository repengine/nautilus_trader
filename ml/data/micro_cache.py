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

from dataclasses import dataclass
from typing import cast as _cast
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from ml._imports import pl
from ml.ml_types import PolarsDF
from ml.features.micro_aggregate import MicrostructureAggregator
from ml.data.cache_common import day_partition_path
from ml.data.cache_common import ensure_polars
from ml.data.cache_common import filter_df_by_ns_range
from ml.data.cache_common import iter_days



@dataclass(slots=True)
class MicroMinuteCache:
    """
    Cache for per-minute microstructure features.

    Attributes:
        cache_dir: Base directory for cached files.

    """

    cache_dir: Path

    def path_for(self, symbol: str, day: date) -> Path:
        return day_partition_path(self.cache_dir, symbol, day)

    def ensure_day(self, symbol: str, day: date, raw_base_dir: Path) -> Path:
        ensure_polars()
        _pl = pl
        assert _pl is not None
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
            df = _pl.DataFrame({"timestamp": []})
        elif df["timestamp"].dtype != _pl.Datetime:
            df = df.with_columns(_pl.col("timestamp").cast(_pl.Datetime("ns", "UTC")))
        # Filter to exact day window
        start_dt = datetime(day.year, day.month, day.day, tzinfo=UTC)
        end_dt = start_dt + timedelta(days=1)
        start_ns = int(start_dt.timestamp() * 1_000_000_000)
        end_ns = int(end_dt.timestamp() * 1_000_000_000)
        df = df.filter(
            (_pl.col("timestamp").cast(_pl.Int64) >= start_ns)
            & (_pl.col("timestamp").cast(_pl.Int64) < end_ns),
        ).sort("timestamp")
        df.write_parquet(str(out))
        return out

    def get_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        raw_base_dir: Path,
    ) -> PolarsDF:
        ensure_polars()
        _pl = pl
        assert _pl is not None
        parts: list[PolarsDF] = []
        for day in iter_days(start, end):
            p = self.ensure_day(symbol=symbol, day=day, raw_base_dir=raw_base_dir)
            if p.exists():
                parts.append(_pl.read_parquet(str(p)))
        if not parts:
            from typing import cast as _cast
            return _cast(PolarsDF, _pl.DataFrame({"timestamp": []}))
        df = _pl.concat(parts, how="vertical")
        if df.is_empty():
            from typing import cast as _cast
            return _cast(PolarsDF, df)
        # Filter to exact [start, end) and sort
        return filter_df_by_ns_range(df, start=start, end=end)
