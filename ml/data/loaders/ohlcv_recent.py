"""
Utilities for backfilling recent OHLCV bars from Databento.

This module centralizes the logic previously embedded in the
``ml.cli.backfill_ohlcv_recent`` script.  The functions defined here are
explicitly typed and safe to reuse from tests, tasks, or orchestrators.

The helpers work purely on the cold path.  They discover symbols, clamp the
requested window using the configured :class:`DatabentoCoveragePolicy`, fetch
bars through the ingestion service, and merge the data into the local tiered
directory structure.

"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import numpy as np

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl as _pl
from ml.data.ingest.policy import DatabentoCoveragePolicy


if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from pandas import DataFrame as PandasDataFrame
    from polars import DataFrame as PolarsDataFrame

    from ml.data.ingest.service import DatabentoIngestionService
else:  # pragma: no cover - typing fallback
    PandasDataFrame = Any  # type: ignore[assignment]
    PolarsDataFrame = Any  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET: Final[str] = "EQUS.MINI"
DEFAULT_SCHEMA: Final[str] = "ohlcv-1m"


if not HAS_POLARS:  # pragma: no cover - import guard mirrors CLI behaviour
    check_ml_dependencies(["polars"])
assert _pl is not None  # narrow Optional at runtime
PL = _pl


class FetchSymbolDataFn(Protocol):
    """
    Callable protocol for fetching symbol data from Databento.
    """

    def __call__(
        self,
        *,
        service: DatabentoIngestionService,
        dataset: str,
        schema: str,
        symbol: str,
        start: datetime,
        end: datetime,
        reason: str,
    ) -> PandasDataFrame: ...


class SymbolBackfillStatus(str, Enum):
    """
    Enumeration of per-symbol backfill outcomes.
    """

    SUCCESS = "success"
    EMPTY = "empty"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(slots=True, frozen=True)
class SymbolBackfillSummary:
    """
    Details of the ingestion result for an individual symbol.
    """

    symbol: str
    status: SymbolBackfillStatus
    requested_start: datetime | None
    requested_end: datetime | None
    rows_downloaded: int = 0
    message: str | None = None


@dataclass(slots=True, frozen=True)
class OhlcvRecentBackfillConfig:
    """
    Configuration describing the recent OHLCV backfill operation.
    """

    data_dir: Path
    symbols: Sequence[str] | None = None
    tier: int | None = None
    start: datetime | None = None
    end: datetime | None = None
    lookback_days: int = 14
    dataset: str = DEFAULT_DATASET
    schema: str = DEFAULT_SCHEMA


@dataclass(slots=True, frozen=True)
class OhlcvRecentBackfillResult:
    """
    Aggregated result for an OHLCV backfill run.
    """

    summaries: tuple[SymbolBackfillSummary, ...]
    dataset: str
    schema: str

    @property
    def successful_symbols(self) -> tuple[str, ...]:
        """
        Return symbols which produced at least one row.
        """
        return tuple(
            summary.symbol
            for summary in self.summaries
            if summary.status is SymbolBackfillStatus.SUCCESS
        )

    @property
    def skipped_symbols(self) -> tuple[str, ...]:
        """
        Return symbols skipped by policy or empty windows.
        """
        return tuple(
            summary.symbol
            for summary in self.summaries
            if summary.status is SymbolBackfillStatus.SKIPPED
        )


def _discover_symbols(data_dir: Path) -> list[str]:
    symbols: list[str] = []
    if not data_dir.exists():
        return symbols
    for entry in data_dir.iterdir():
        if entry.is_dir() and entry.name.isupper():
            symbols.append(entry.name)
    symbols.sort()
    return symbols


def _symbols_from_universe_file(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except Exception:  # pragma: no cover - defensive guard
        return []

    if isinstance(payload, dict):
        direct = payload.get("symbols")
        if isinstance(direct, list) and all(isinstance(item, str) for item in direct):
            return sorted({symbol.upper() for symbol in direct})
        collected: set[str] = set()
        for value in payload.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        symbol = item.get("symbol")
                        if isinstance(symbol, str) and symbol:
                            collected.add(symbol.upper())
        if collected:
            return sorted(collected)
    return []


def _resolve_symbols(config: OhlcvRecentBackfillConfig) -> tuple[str, ...]:
    if config.symbols:
        return tuple(dict.fromkeys(symbol.upper() for symbol in config.symbols))
    if config.tier in (1, 2, 3):
        universe_path = Path(f"ml/config/universe_tier{config.tier}.json")
        if universe_path.exists():
            symbols = _symbols_from_universe_file(universe_path)
            if symbols:
                return tuple(symbols)
    return tuple(_discover_symbols(config.data_dir))


def _last_bar_timestamp(base: Path, symbol: str) -> datetime | None:
    l0 = base / symbol / "l0" / f"{symbol}_ohlcv.parquet"
    frames: list[PolarsDataFrame] = []
    if l0.exists():
        try:
            df = PL.read_parquet(str(l0))
            if not df.is_empty():
                column = "timestamp"
                if column not in df.columns:
                    column = "ts_event" if "ts_event" in df.columns else column
                if column in df.columns:
                    frames.append(df.select(column).rename({column: "timestamp"}))
        except Exception as exc:  # pragma: no cover - logging only
            LOGGER.debug("Failed to read l0 parquet for %s", symbol, exc_info=exc)
    for candidate in ("ohlcv-1m_historical.parquet", "ohlcv-1m_recent.parquet"):
        path = base / symbol / candidate
        if path.exists():
            try:
                df = PL.read_parquet(str(path)).select(
                    [PL.col("timestamp").alias("timestamp")],
                )
                if not df.is_empty():
                    frames.append(df)
            except Exception as exc:  # pragma: no cover - logging only
                LOGGER.debug(
                    "Failed to read fallback parquet for %s (%s)",
                    symbol,
                    candidate,
                    exc_info=exc,
                )
    if not frames:
        return None
    concatenated = PL.concat(frames, how="vertical").drop_nulls()
    if concatenated.is_empty():
        return None
    ts_value: object = concatenated.select(PL.col("timestamp").max())[0, 0]
    if hasattr(ts_value, "to_pydatetime"):
        return cast("datetime", ts_value.to_pydatetime())
    if isinstance(ts_value, np.datetime64):
        nanos = int(ts_value.astype("int64"))
        return datetime.fromtimestamp(nanos / 1_000_000_000)
    if isinstance(ts_value, (int, float)):
        return datetime.fromtimestamp(float(ts_value))
    raise TypeError(f"Unsupported timestamp type {type(ts_value)!r}")


def _merge_save(base: Path, symbol: str, df_new: PandasDataFrame) -> None:
    out_dir = base / symbol / "l0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_ohlcv.parquet"

    if hasattr(df_new, "to_dict"):
        new_pl = PL.from_pandas(df_new, include_index=True)
    else:  # pragma: no cover - defensive
        new_pl = PL.DataFrame(df_new)

    time_aliases = ("timestamp", "ts_event", "ts", "time", "index")
    if "timestamp" not in new_pl.columns:
        for alias in time_aliases:
            if alias in new_pl.columns:
                new_pl = new_pl.rename({alias: "timestamp"})
                break
    if "timestamp" in new_pl.columns:
        try:
            new_pl = new_pl.with_columns(PL.col("timestamp").cast(PL.Datetime("ns")))
        except Exception as exc:  # pragma: no cover - logging only
            LOGGER.debug("Failed casting timestamp for %s", symbol, exc_info=exc)
    keep_columns = [
        col
        for col in ("timestamp", "open", "high", "low", "close", "volume")
        if col in new_pl.columns
    ]
    new_pl = new_pl.select(keep_columns).sort("timestamp")
    if out_path.exists():
        try:
            existing = PL.read_parquet(str(out_path)).select(keep_columns)
            merged = (
                PL.concat([existing, new_pl], how="vertical")
                .unique(subset=["timestamp"])
                .sort("timestamp")
            )
        except Exception:  # pragma: no cover - fallback path
            merged = new_pl
    else:
        merged = new_pl
    merged.write_parquet(str(out_path))


def backfill_recent_ohlcv(
    config: OhlcvRecentBackfillConfig,
    *,
    service: DatabentoIngestionService | None = None,
    policy: DatabentoCoveragePolicy | None = None,
    fetch_fn: FetchSymbolDataFn | None = None,
) -> OhlcvRecentBackfillResult:
    """
    Backfill recent OHLCV bars for the requested symbols.

    Parameters
    ----------
    config:
        Configuration describing the data directory, symbols, and time window.
    service:
        Optional Databento ingestion service used to fetch the bars. When
        omitted, the service is resolved via :func:`ml.data.ingest.api.ensure_service`.
    policy:
        Optional coverage policy. When ``None`` the policy is resolved from the
        environment via :func:`DatabentoCoveragePolicy.from_env`.
    fetch_fn:
        Optional override for :func:`ml.data.ingest.api.fetch_symbol_data` to
        facilitate testing.

    """
    from ml.data.ingest.api import ensure_service  # local import to avoid cycles
    from ml.data.ingest.api import fetch_symbol_data  # local import to avoid cycles

    active_policy = policy or DatabentoCoveragePolicy.from_env()
    fetch = fetch_fn or fetch_symbol_data
    resolved_service = service or ensure_service()

    symbols = _resolve_symbols(config)
    if not symbols:
        LOGGER.info("No symbols discovered for OHLCV backfill in %s", config.data_dir)
        return OhlcvRecentBackfillResult(tuple(), config.dataset, config.schema)

    requested_end = config.end or datetime.now()
    requested_start = config.start or (requested_end - timedelta(days=max(config.lookback_days, 1)))

    summaries: list[SymbolBackfillSummary] = []

    for symbol in symbols:
        allowed = active_policy.filter_symbols([symbol])
        if not allowed:
            summaries.append(
                SymbolBackfillSummary(
                    symbol=symbol,
                    status=SymbolBackfillStatus.SKIPPED,
                    requested_start=requested_start,
                    requested_end=requested_end,
                    rows_downloaded=0,
                    message="filtered-by-policy",
                ),
            )
            continue

        start_dt = requested_start
        if config.start is None:
            last_ts = _last_bar_timestamp(config.data_dir, symbol)
            if last_ts is not None:
                start_dt = max(last_ts + timedelta(minutes=1), requested_start)

        clamped_start, clamped_end = active_policy.clamp_range(
            start_dt,
            requested_end,
            dataset=config.dataset,
            schema=config.schema,
        )
        if clamped_start >= clamped_end:
            summaries.append(
                SymbolBackfillSummary(
                    symbol=symbol,
                    status=SymbolBackfillStatus.SKIPPED,
                    requested_start=clamped_start,
                    requested_end=clamped_end,
                    rows_downloaded=0,
                    message="empty-window",
                ),
            )
            continue

        try:
            frame = fetch(
                service=resolved_service,
                dataset=config.dataset,
                schema=config.schema,
                symbol=symbol,
                start=clamped_start,
                end=clamped_end,
                reason="backfill_ohlcv_recent",
            )
        except Exception as exc:  # pragma: no cover - fetch failures
            LOGGER.warning("Failed fetching %s: %s", symbol, exc)
            summaries.append(
                SymbolBackfillSummary(
                    symbol=symbol,
                    status=SymbolBackfillStatus.ERROR,
                    requested_start=clamped_start,
                    requested_end=clamped_end,
                    rows_downloaded=0,
                    message=str(exc),
                ),
            )
            continue

        if frame is None or getattr(frame, "empty", False):
            summaries.append(
                SymbolBackfillSummary(
                    symbol=symbol,
                    status=SymbolBackfillStatus.EMPTY,
                    requested_start=clamped_start,
                    requested_end=clamped_end,
                    rows_downloaded=0,
                    message="no-rows",
                ),
            )
            continue

        rows = int(getattr(frame, "shape", (0, 0))[0])
        _merge_save(config.data_dir, symbol, frame)
        LOGGER.info(
            "Backfilled %s rows=%d start=%s end=%s",
            symbol,
            rows,
            clamped_start,
            clamped_end,
        )
        summaries.append(
            SymbolBackfillSummary(
                symbol=symbol,
                status=SymbolBackfillStatus.SUCCESS,
                requested_start=clamped_start,
                requested_end=clamped_end,
                rows_downloaded=rows,
                message=None,
            ),
        )

    return OhlcvRecentBackfillResult(tuple(summaries), config.dataset, config.schema)


BackfillRecentOhlcvTaskConfig = OhlcvRecentBackfillConfig


__all__ = [
    "BackfillRecentOhlcvTaskConfig",
    "OhlcvRecentBackfillConfig",
    "OhlcvRecentBackfillResult",
    "SymbolBackfillStatus",
    "SymbolBackfillSummary",
    "backfill_recent_ohlcv",
]
