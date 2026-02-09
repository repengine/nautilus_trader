"""
Cache hydration helpers for microstructure and L2 per-minute features.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import cast

from ml._imports import pl
from ml.common.timestamps import sanitize_timestamp_ns
from ml.config.dataset_ids import L2_MINUTE_DATASET_ID
from ml.config.dataset_ids import MICRO_MINUTE_DATASET_ID
from ml.config.events import Source
from ml.data.cache_common import ensure_polars
from ml.data.cache_common import filter_df_by_ns_range
from ml.data.cache_common import resolve_cache_partition_path
from ml.data.l2_cache import L2MinuteCache
from ml.data.micro_cache import MicroMinuteCache
from ml.features.l2_aggregate import L2_MINUTE_COLUMNS
from ml.features.micro_aggregate import MICRO_COLUMNS
from ml.features.micro_aggregate import MicrostructureAggregator
from ml.ml_types import PolarsDF
from ml.stores.protocols import DataStoreFacadeProtocol


logger = logging.getLogger(__name__)


def _require_polars_module() -> ModuleType:
    ensure_polars()
    module: ModuleType | None = pl
    if module is None:
        from ml._imports import pl as _pl

        return cast(ModuleType, _pl)
    return module


@dataclass(frozen=True)
class SymbolHydrationResult:
    """Per-symbol hydration summary."""

    symbol: str
    requested_partitions: int
    written_partitions: int
    skipped_partitions: int
    empty_partitions: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CacheHydrationResult:
    """Aggregate hydration summary."""

    label: str
    results: tuple[SymbolHydrationResult, ...]

    @property
    def total_requested(self) -> int:
        return sum(result.requested_partitions for result in self.results)

    @property
    def total_written(self) -> int:
        return sum(result.written_partitions for result in self.results)

    @property
    def total_skipped(self) -> int:
        return sum(result.skipped_partitions for result in self.results)

    @property
    def total_empty(self) -> int:
        return sum(result.empty_partitions for result in self.results)

    @property
    def failed(self) -> tuple[SymbolHydrationResult, ...]:
        return tuple(result for result in self.results if result.errors)


@dataclass(frozen=True)
class MicroCacheHydrationConfig:
    """Configuration for microstructure cache hydration."""

    symbols: tuple[str, ...]
    start_date: date
    end_date: date
    raw_base_dir: Path = field(default_factory=lambda: Path("data/tier1"))
    cache_dir: Path = field(default_factory=lambda: Path("data/features/micro_minute"))
    max_workers: int = 4
    force_rebuild: bool = False


@dataclass(frozen=True)
class L2CacheHydrationConfig:
    """Configuration for L2 cache hydration."""

    symbols: tuple[str, ...]
    start_date: date
    end_date: date
    raw_base_dir: Path = field(default_factory=lambda: Path("data/tier1"))
    cache_dir: Path = field(default_factory=lambda: Path("data/features/l2_minute"))
    max_workers: int = 4
    force_rebuild: bool = False


def hydrate_micro_caches(config: MicroCacheHydrationConfig) -> CacheHydrationResult:
    """Hydrate microstructure caches for a date range."""
    symbols = _normalize_symbols(config.symbols)
    if not symbols:
        raise ValueError("No symbols provided for micro cache hydration")
    days = _date_range(config.start_date, config.end_date)
    requested = len(days)
    cache = MicroMinuteCache(Path(config.cache_dir))
    raw_base_dir = Path(config.raw_base_dir)
    logger.info(
        "Hydrating micro caches",
        extra={
            "symbols": len(symbols),
            "days": requested,
            "force": config.force_rebuild,
            "cache_dir": str(config.cache_dir),
            "raw_base_dir": str(config.raw_base_dir),
        },
    )

    def worker(symbol: str) -> SymbolHydrationResult:
        return _hydrate_micro_symbol(
            cache=cache,
            symbol=symbol,
            raw_base_dir=raw_base_dir,
            days=days,
            force=config.force_rebuild,
        )

    results = tuple(
        _run_parallel(symbols, worker, config.max_workers, requested),
    )
    return CacheHydrationResult(label="micro", results=results)


def hydrate_l2_caches(config: L2CacheHydrationConfig) -> CacheHydrationResult:
    """Hydrate L2 caches for a date range."""
    symbols = _normalize_symbols(config.symbols)
    if not symbols:
        raise ValueError("No symbols provided for L2 cache hydration")
    days = _date_range(config.start_date, config.end_date)
    requested = len(days)
    cache = L2MinuteCache(Path(config.cache_dir))
    raw_base_dir = Path(config.raw_base_dir)
    logger.info(
        "Hydrating L2 caches",
        extra={
            "symbols": len(symbols),
            "days": requested,
            "force": config.force_rebuild,
            "cache_dir": str(config.cache_dir),
            "raw_base_dir": str(config.raw_base_dir),
        },
    )

    def worker(symbol: str) -> SymbolHydrationResult:
        return _hydrate_l2_symbol(
            cache=cache,
            symbol=symbol,
            raw_base_dir=raw_base_dir,
            days=days,
            force=config.force_rebuild,
        )

    results = tuple(
        _run_parallel(symbols, worker, config.max_workers, requested),
    )
    return CacheHydrationResult(label="l2", results=results)


def ingest_micro_cache_partitions(
    *,
    data_store: DataStoreFacadeProtocol,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    cache_dir: Path,
    run_id: str = "micro_cache_ingest",
) -> None:
    """Persist hydrated micro caches into the SQL feature dataset."""
    _pl = _require_polars_module()
    ts_init_ns = sanitize_timestamp_ns(time.time_ns(), context="micro_cache_ingest")
    normalized_instruments = _normalize_instrument_ids(symbols)
    if not normalized_instruments:
        return
    cache_root = Path(cache_dir)
    for instrument_id in normalized_instruments:
        frames: list[PolarsDF] = []
        for day in _date_range(start_date, end_date):
            partition_path, _ = resolve_cache_partition_path(cache_root, instrument_id, day)
            if partition_path is None:
                continue
            try:
                df = _pl.read_parquet(str(partition_path))
            except Exception:
                logger.warning(
                    "micro_cache.parquet_read_failed",
                    exc_info=True,
                    extra={"symbol": instrument_id, "path": str(partition_path)},
                )
                continue
            if df.is_empty():
                continue
            prepared = (
                df.with_columns(
                    [
                        _pl.lit(instrument_id).alias("instrument_id"),
                        _pl.col("timestamp").cast(_pl.Datetime("ns")).cast(_pl.Int64).alias("timestamp_ns"),
                    ],
                )
                .with_columns(
                    [
                        _pl.col("timestamp_ns").alias("timestamp"),
                        _pl.col("timestamp_ns").alias("ts_event"),
                        _pl.lit(ts_init_ns).alias("ts_init"),
                    ],
                )
                .drop("timestamp_ns")
                .select(
                    [
                        "instrument_id",
                        "timestamp",
                        "ts_event",
                        "ts_init",
                        *MICRO_COLUMNS,
                    ],
                )
            )
            frames.append(prepared)
        if not frames:
            continue
        combined = _pl.concat(frames, how="vertical")
        if combined.is_empty():
            continue
        data_store.write_ingestion(
            dataset_id=MICRO_MINUTE_DATASET_ID,
            records=combined,
            source=Source.HISTORICAL.value,
            run_id=f"{run_id}_{instrument_id}",
            instrument_id=instrument_id,
        )


def ingest_l2_cache_partitions(
    *,
    data_store: DataStoreFacadeProtocol,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    cache_dir: Path,
    run_id: str = "l2_cache_ingest",
) -> None:
    """Persist hydrated L2 caches into the SQL feature dataset."""
    _pl = _require_polars_module()
    ts_init_ns = sanitize_timestamp_ns(time.time_ns(), context="l2_cache_ingest")
    normalized_instruments = _normalize_instrument_ids(symbols)
    if not normalized_instruments:
        return
    cache_root = Path(cache_dir)
    l2_columns = [col for col in L2_MINUTE_COLUMNS if col != "timestamp"]
    for instrument_id in normalized_instruments:
        frames: list[PolarsDF] = []
        for day in _date_range(start_date, end_date):
            partition_path, _ = resolve_cache_partition_path(cache_root, instrument_id, day)
            if partition_path is None:
                continue
            try:
                df = _pl.read_parquet(str(partition_path))
            except Exception:
                logger.warning(
                    "l2_cache.parquet_read_failed",
                    exc_info=True,
                    extra={"symbol": instrument_id, "path": str(partition_path)},
                )
                continue
            if df.is_empty():
                continue
            prepared = (
                df.with_columns(
                    [
                        _pl.lit(instrument_id).alias("instrument_id"),
                        _pl.col("timestamp").cast(_pl.Datetime("ns")).cast(_pl.Int64).alias("timestamp_ns"),
                    ],
                )
                .with_columns(
                    [
                        _pl.col("timestamp_ns").alias("timestamp"),
                        _pl.col("timestamp_ns").alias("ts_event"),
                        _pl.lit(ts_init_ns).alias("ts_init"),
                    ],
                )
                .drop("timestamp_ns")
                .select(
                    [
                        "instrument_id",
                        "timestamp",
                        "ts_event",
                        "ts_init",
                        *l2_columns,
                    ],
                )
            )
            frames.append(prepared)
        if not frames:
            continue
        combined = _pl.concat(frames, how="vertical")
        if combined.is_empty():
            continue
        data_store.write_ingestion(
            dataset_id=L2_MINUTE_DATASET_ID,
            records=combined,
            source=Source.HISTORICAL.value,
            run_id=f"{run_id}_{instrument_id}",
            instrument_id=instrument_id,
        )


def _run_parallel(
    symbols: Sequence[str],
    worker: Callable[[str], SymbolHydrationResult],
    max_workers: int,
    requested_partitions: int,
) -> Iterable[SymbolHydrationResult]:
    bound_workers = max(1, min(max_workers, len(symbols)))
    if bound_workers == 1:
        for symbol in symbols:
            yield _execute_worker(worker, symbol, requested_partitions)
        return
    with ThreadPoolExecutor(max_workers=bound_workers) as executor:
        futures: dict[Future[SymbolHydrationResult], str] = {
            executor.submit(worker, symbol): symbol for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                yield future.result()
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Hydration failed for %s", symbol)
                yield SymbolHydrationResult(
                    symbol=symbol,
                    requested_partitions=requested_partitions,
                    written_partitions=0,
                    skipped_partitions=requested_partitions,
                    empty_partitions=0,
                    errors=(str(exc),),
                )


def _execute_worker(
    worker: Callable[[str], SymbolHydrationResult],
    symbol: str,
    requested_partitions: int,
) -> SymbolHydrationResult:
    try:
        return worker(symbol)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Hydration failed for %s", symbol)
        return SymbolHydrationResult(
            symbol=symbol,
            requested_partitions=requested_partitions,
            written_partitions=0,
            skipped_partitions=requested_partitions,
            empty_partitions=0,
            errors=(str(exc),),
        )


def _hydrate_micro_symbol(
    cache: MicroMinuteCache,
    symbol: str,
    raw_base_dir: Path,
    days: tuple[date, ...],
    *,
    force: bool,
) -> SymbolHydrationResult:
    requested = len(days)
    ensure_polars()
    _pl = pl
    if _pl is None:
        raise RuntimeError("Polars runtime unavailable for micro cache hydration")
    missing_days: list[date] = []
    for day in days:
        path = cache.path_for(symbol, day)
        if path.exists() and not force and cache.is_valid_partition(path):
            continue
        missing_days.append(day)
    if not missing_days:
        return SymbolHydrationResult(
            symbol=symbol,
            requested_partitions=requested,
            written_partitions=0,
            skipped_partitions=requested,
            empty_partitions=0,
        )

    agg = MicrostructureAggregator(raw_base_dir)
    window_start = datetime(days[0].year, days[0].month, days[0].day, tzinfo=UTC)
    window_end = datetime(days[-1].year, days[-1].month, days[-1].day, tzinfo=UTC) + timedelta(
        days=1,
    )
    frame = agg.compute_for_symbol(symbol, start=window_start, end=window_end)
    if not frame.is_empty():
        if frame["timestamp"].dtype != _pl.Datetime:
            frame = frame.with_columns(_pl.col("timestamp").cast(_pl.Datetime("ns", "UTC")))
        frame = filter_df_by_ns_range(frame, start=window_start, end=window_end)

    written = 0
    empty = 0
    for day in missing_days:
        partition = _slice_day(frame, day)
        path = cache.path_for(symbol, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        if partition.is_empty():
            empty += 1
            if path.exists():
                path.unlink(missing_ok=True)
            continue
        partition.write_parquet(str(path))
        written += 1
    skipped = requested - written
    return SymbolHydrationResult(
        symbol=symbol,
        requested_partitions=requested,
        written_partitions=written,
        skipped_partitions=skipped,
        empty_partitions=empty,
    )


def _hydrate_l2_symbol(
    cache: L2MinuteCache,
    symbol: str,
    raw_base_dir: Path,
    days: tuple[date, ...],
    *,
    force: bool,
) -> SymbolHydrationResult:
    requested = len(days)
    written = 0
    empty = 0
    _pl = _require_polars_module()
    for day in days:
        path = cache.path_for(symbol, day)
        if force and path.exists():
            path.unlink()
        before_mtime = path.stat().st_mtime if path.exists() else None
        cache.ensure_day(symbol, day, raw_base_dir)
        if not path.exists():
            continue
        after_mtime = path.stat().st_mtime
        wrote_partition = before_mtime is None or after_mtime != before_mtime
        if wrote_partition:
            written += 1
            if _is_parquet_empty(path, _pl):
                empty += 1
    skipped = requested - written
    return SymbolHydrationResult(
        symbol=symbol,
        requested_partitions=requested,
        written_partitions=written,
        skipped_partitions=skipped,
        empty_partitions=empty,
    )


def _is_parquet_empty(path: Path, module: ModuleType) -> bool:
    df = module.read_parquet(str(path), n_rows=1, columns=["timestamp"])
    return bool(df.height == 0)


def _slice_day(frame: PolarsDF, day: date) -> PolarsDF:
    if frame.is_empty():
        return frame
    _pl = _require_polars_module()
    start_dt = datetime(day.year, day.month, day.day, tzinfo=UTC)
    end_dt = start_dt + timedelta(days=1)
    return frame.filter(
        (_pl.col("timestamp") >= start_dt) & (_pl.col("timestamp") < end_dt),
    ).sort("timestamp")


def _date_range(start: date, end: date) -> tuple[date, ...]:
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _normalize_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in symbols:
        trimmed = token.strip().upper()
        if not trimmed:
            continue
        normalized = trimmed
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _normalize_instrument_ids(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for token in symbols:
        trimmed = token.strip().upper()
        if not trimmed:
            continue
        if trimmed in seen:
            continue
        seen.add(trimmed)
        normalized.append(trimmed)
    return tuple(normalized)


__all__ = [
    "CacheHydrationResult",
    "L2CacheHydrationConfig",
    "MicroCacheHydrationConfig",
    "SymbolHydrationResult",
    "hydrate_l2_caches",
    "hydrate_micro_caches",
    "ingest_l2_cache_partitions",
    "ingest_micro_cache_partitions",
]
