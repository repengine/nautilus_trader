"""
Cache-first join helpers for microstructure and L2 feature families.

These helpers centralize cache policy handling, timestamp alignment, and
null-filling for micro/L2 joins to keep dataset builders DRY and consistent.
"""

from __future__ import annotations

import logging
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ml._imports import pd as pd_runtime
from ml._imports import pl as pl_runtime
from ml.config.feature_cache import FeatureCachePolicy
from ml.config.feature_cache import normalize_feature_cache_policy
from ml.data.common.time_series_windowing import TimeSeriesWindowingComponent
from ml.data.l2_cache import L2MinuteCache
from ml.data.micro_cache import MicroMinuteCache
from ml.features.l2_aggregate import L2_MINUTE_COLUMNS
from ml.features.l2_aggregate import L2Aggregator
from ml.features.micro_aggregate import MICRO_COLUMNS
from ml.features.micro_aggregate import MicrostructureAggregator


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl
else:  # pragma: no cover - typing fallback
    _pd = Any
    _pl = Any


PL: Any = cast(Any, pl_runtime)
PD: Any = cast(Any, pd_runtime)
logger = logging.getLogger(__name__)


def join_micro_cache_polars(
    dataset: _pl.DataFrame,
    *,
    symbol: str,
    raw_base_dir: Path,
    cache_dir: Path,
    policy: FeatureCachePolicy,
) -> _pl.DataFrame:
    """
    Join microstructure features onto a Polars dataset using cache policy.

    Args:
        dataset: Polars dataset with a ``timestamp`` column.
        symbol: Instrument symbol for cache lookup.
        raw_base_dir: Root directory for raw micro data (used for live aggregation).
        cache_dir: Root directory for cached micro features.
        policy: Cache policy token controlling cache vs live aggregation.

    Returns:
        Dataset with microstructure features joined (if available).
    """
    if dataset.is_empty() or "timestamp" not in dataset.columns:
        return dataset

    policy_token = normalize_feature_cache_policy(
        policy,
        label="micro_cache_policy",
    )
    bounds = _resolve_bounds_polars(dataset)
    if bounds is None:
        return dataset
    start, end = bounds

    try:
        micro = _load_micro_polars(
            symbol=symbol,
            start=start,
            end=end,
            raw_base_dir=raw_base_dir,
            cache_dir=cache_dir,
            policy=policy_token,
        )
        if micro.is_empty():
            micro = _empty_micro_frame()
        dataset = _ensure_timestamp_polars(dataset)
        micro = _ensure_timestamp_polars(micro)
        before_cols = set(dataset.columns)
        dataset = dataset.join(micro, on="timestamp", how="left")
        return _fill_new_numeric_polars(dataset, before_cols)
    except Exception:
        logger.debug(
            "Microstructure cache join failed for %s",
            symbol,
            exc_info=True,
        )
        return dataset


def join_l2_cache_polars(
    dataset: _pl.DataFrame,
    *,
    symbol: str,
    raw_base_dir: Path,
    cache_dir: Path,
    policy: FeatureCachePolicy,
) -> _pl.DataFrame:
    """
    Join L2 features onto a Polars dataset using cache policy.

    Args:
        dataset: Polars dataset with a ``timestamp`` column.
        symbol: Instrument symbol for cache lookup.
        raw_base_dir: Root directory for raw L2 data (used for live aggregation).
        cache_dir: Root directory for cached L2 features.
        policy: Cache policy token controlling cache vs live aggregation.

    Returns:
        Dataset with L2 features joined (if available).

    Notes:
        When ``policy`` is ``"cache_only"``, this join never triggers raw
        aggregation. This keeps L2 features gated while the Databento
        subscription is inactive.
    """
    if dataset.is_empty() or "timestamp" not in dataset.columns:
        return dataset

    policy_token = normalize_feature_cache_policy(
        policy,
        label="l2_cache_policy",
    )
    bounds = _resolve_bounds_polars(dataset)
    if bounds is None:
        return dataset
    start, end = bounds

    try:
        l2 = _load_l2_polars(
            symbol=symbol,
            start=start,
            end=end,
            raw_base_dir=raw_base_dir,
            cache_dir=cache_dir,
            policy=policy_token,
        )
        if l2.is_empty():
            l2 = _empty_l2_frame()
        dataset = _ensure_timestamp_polars(dataset)
        l2 = _ensure_timestamp_polars(l2)
        before_cols = set(dataset.columns)
        dataset = dataset.join(l2, on="timestamp", how="left")
        return _fill_new_numeric_polars(dataset, before_cols)
    except Exception:
        logger.debug(
            "L2 cache join failed for %s",
            symbol,
            exc_info=True,
        )
        return dataset


def join_micro_cache_pandas(
    dataset: _pd.DataFrame,
    *,
    symbol: str,
    raw_base_dir: Path,
    cache_dir: Path,
    policy: FeatureCachePolicy,
) -> _pd.DataFrame:
    """
    Join microstructure features onto a Pandas dataset using cache policy.

    Args:
        dataset: Pandas dataset with a ``timestamp`` column.
        symbol: Instrument symbol for cache lookup.
        raw_base_dir: Root directory for raw micro data (used for live aggregation).
        cache_dir: Root directory for cached micro features.
        policy: Cache policy token controlling cache vs live aggregation.

    Returns:
        Dataset with microstructure features joined (if available).
    """
    if dataset.empty or "timestamp" not in dataset.columns:
        return dataset
    if PD is None:
        return dataset

    policy_token = normalize_feature_cache_policy(
        policy,
        label="micro_cache_policy",
    )
    bounds = _resolve_bounds_pandas(dataset)
    if bounds is None:
        return dataset
    start, end = bounds

    try:
        micro = _load_micro_polars(
            symbol=symbol,
            start=start,
            end=end,
            raw_base_dir=raw_base_dir,
            cache_dir=cache_dir,
            policy=policy_token,
        )
        if micro.is_empty():
            micro = _empty_micro_frame()
        micro_pd = micro.to_pandas()
        dataset = _ensure_timestamp_pandas(dataset)
        micro_pd = _ensure_timestamp_pandas(micro_pd)
        before_cols = set(dataset.columns)
        dataset = dataset.merge(micro_pd, on="timestamp", how="left")
        return _fill_new_numeric_pandas(dataset, before_cols)
    except Exception:
        logger.debug(
            "Microstructure cache join failed for %s",
            symbol,
            exc_info=True,
        )
        return dataset


def join_l2_cache_pandas(
    dataset: _pd.DataFrame,
    *,
    symbol: str,
    raw_base_dir: Path,
    cache_dir: Path,
    policy: FeatureCachePolicy,
) -> _pd.DataFrame:
    """
    Join L2 features onto a Pandas dataset using cache policy.

    Args:
        dataset: Pandas dataset with a ``timestamp`` column.
        symbol: Instrument symbol for cache lookup.
        raw_base_dir: Root directory for raw L2 data (used for live aggregation).
        cache_dir: Root directory for cached L2 features.
        policy: Cache policy token controlling cache vs live aggregation.

    Returns:
        Dataset with L2 features joined (if available).

    Notes:
        When ``policy`` is ``"cache_only"``, this join never triggers raw
        aggregation. This keeps L2 features gated while the Databento
        subscription is inactive.
    """
    if dataset.empty or "timestamp" not in dataset.columns:
        return dataset
    if PD is None:
        return dataset

    policy_token = normalize_feature_cache_policy(
        policy,
        label="l2_cache_policy",
    )
    bounds = _resolve_bounds_pandas(dataset)
    if bounds is None:
        return dataset
    start, end = bounds

    try:
        l2 = _load_l2_polars(
            symbol=symbol,
            start=start,
            end=end,
            raw_base_dir=raw_base_dir,
            cache_dir=cache_dir,
            policy=policy_token,
        )
        if l2.is_empty():
            l2 = _empty_l2_frame()
        l2_pd = l2.to_pandas()
        dataset = _ensure_timestamp_pandas(dataset)
        l2_pd = _ensure_timestamp_pandas(l2_pd)
        before_cols = set(dataset.columns)
        dataset = dataset.merge(l2_pd, on="timestamp", how="left")
        return _fill_new_numeric_pandas(dataset, before_cols)
    except Exception:
        logger.debug(
            "L2 cache join failed for %s",
            symbol,
            exc_info=True,
        )
        return dataset


def _resolve_bounds_polars(
    dataset: _pl.DataFrame,
) -> tuple[datetime, datetime] | None:
    min_ns, max_ns = TimeSeriesWindowingComponent.frame_time_bounds(dataset)
    if min_ns is None or max_ns is None:
        return None
    start = datetime.fromtimestamp(min_ns / 1_000_000_000, tz=UTC)
    end = datetime.fromtimestamp(max_ns / 1_000_000_000, tz=UTC) + timedelta(microseconds=1)
    return start, end


def _resolve_bounds_pandas(
    dataset: _pd.DataFrame,
) -> tuple[datetime, datetime] | None:
    if PD is None:
        return None
    ts = PD.to_datetime(dataset["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        return None
    min_ts = ts.min()
    max_ts = ts.max()
    if min_ts is PD.NaT or max_ts is PD.NaT:
        return None
    start = min_ts.to_pydatetime()
    end = max_ts.to_pydatetime() + timedelta(microseconds=1)
    return start, end


def _ensure_timestamp_polars(df: _pl.DataFrame) -> _pl.DataFrame:
    if df.is_empty():
        return df
    if df["timestamp"].dtype != PL.Datetime("ns", "UTC"):
        return df.with_columns(PL.col("timestamp").cast(PL.Datetime("ns", "UTC")))
    return df


def _ensure_timestamp_pandas(df: _pd.DataFrame) -> _pd.DataFrame:
    if PD is None:
        return df
    df = df.copy()
    df["timestamp"] = PD.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _empty_micro_frame() -> _pl.DataFrame:
    columns: dict[str, _pl.Series] = {
        "timestamp": PL.Series("timestamp", [], dtype=PL.Datetime("ns", "UTC")),
    }
    for name in MICRO_COLUMNS:
        columns[name] = PL.Series(name, [], dtype=PL.Float64)
    return cast("_pl.DataFrame", PL.DataFrame(columns))


def _empty_l2_frame() -> _pl.DataFrame:
    columns: dict[str, _pl.Series] = {
        "timestamp": PL.Series("timestamp", [], dtype=PL.Datetime("ns", "UTC")),
    }
    for name in L2_MINUTE_COLUMNS:
        if name == "timestamp":
            continue
        columns[name] = PL.Series(name, [], dtype=PL.Float64)
    return cast("_pl.DataFrame", PL.DataFrame(columns))


def _fill_new_numeric_polars(
    df: _pl.DataFrame,
    before_cols: set[str],
) -> _pl.DataFrame:
    new_cols = [c for c in df.columns if c not in before_cols]
    fills = []
    for col in new_cols:
        try:
            if df.schema[col].is_numeric():
                fills.append(PL.col(col).fill_null(0))
        except Exception:
            continue
    if fills:
        df = df.with_columns(fills)
    return df


def _fill_new_numeric_pandas(
    df: _pd.DataFrame,
    before_cols: set[str],
) -> _pd.DataFrame:
    if PD is None:
        return df
    new_cols = [c for c in df.columns if c not in before_cols]
    numeric_cols = [
        c for c in new_cols if PD.api.types.is_numeric_dtype(df[c])
    ]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def _load_micro_polars(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    raw_base_dir: Path,
    cache_dir: Path,
    policy: FeatureCachePolicy,
) -> _pl.DataFrame:
    if policy == "live_only":
        agg = MicrostructureAggregator(raw_base_dir)
        return agg.compute_for_symbol(symbol, start=start, end=end)
    cache = MicroMinuteCache(cache_dir)
    micro = cache.get_range(
        symbol=symbol,
        start=start,
        end=end,
        raw_base_dir=raw_base_dir,
        allow_compute=policy != "cache_only",
    )
    if micro.is_empty() and policy == "cache_first":
        agg = MicrostructureAggregator(raw_base_dir)
        micro = agg.compute_for_symbol(symbol, start=start, end=end)
    return micro


def _load_l2_polars(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    raw_base_dir: Path,
    cache_dir: Path,
    policy: FeatureCachePolicy,
) -> _pl.DataFrame:
    if policy == "live_only":
        agg = L2Aggregator(raw_base_dir)
        return agg.compute_for_symbol(symbol, start=start, end=end)
    cache = L2MinuteCache(cache_dir)
    l2 = cache.get_range(
        symbol=symbol,
        start=start,
        end=end,
        raw_base_dir=raw_base_dir,
        allow_compute=policy != "cache_only",
    )
    if l2.is_empty() and policy == "cache_first":
        agg = L2Aggregator(raw_base_dir)
        l2 = agg.compute_for_symbol(symbol, start=start, end=end)
    return l2


__all__ = [
    "join_l2_cache_pandas",
    "join_l2_cache_polars",
    "join_micro_cache_pandas",
    "join_micro_cache_polars",
]
