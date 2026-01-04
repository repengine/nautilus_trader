#!/usr/bin/env python3
"""
ALFRED (vintage FRED) data loader.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from ml._imports import check_ml_dependencies
from ml._imports import fredapi as _fredapi
from ml._imports import pd as _pd
from ml._imports import pl
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.ml_types import PolarsDF


logger = logging.getLogger(__name__)


if pl is None:
    check_ml_dependencies(["polars"])  # pragma: no cover - defensive
    import importlib as _importlib

    pl = _importlib.import_module("polars")

if _pd is None:
    check_ml_dependencies(["pandas"])  # pragma: no cover - defensive
    import importlib as _importlib_pd

    _pd = _importlib_pd.import_module("pandas")

POLARS = cast(Any, pl)
PANDAS = cast(Any, _pd)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

if _fredapi is None:
    # Defer hard failure until instantiation time
    _fredapi = None


class _FredVintageClient(Protocol):
    """
    Protocol for the subset of fredapi.Fred used by the loader.
    """

    def get_series_all_releases(
        self,
        series_id: str,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.DataFrame:  # pragma: no cover - Protocol
        ...

    def get_series(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pd.Series:  # pragma: no cover - Protocol
        ...


def _ensure_polars_frame(obj: pd.DataFrame) -> PolarsDF:
    """
    Convert pandas DataFrame to Polars with normalized schema.
    """
    polars_frame = POLARS.from_pandas(obj, include_index=False)
    return cast(PolarsDF, polars_frame)


@dataclass(slots=True, frozen=True)
class ALFREDConfig:
    """
    Configuration for :class:`ALFREDDataLoader`.
    """

    series_ids: tuple[str, ...] = field(default_factory=tuple)
    out_dir: Path = Path("data/fred/vintages")
    start_date: str | None = None
    end_date: str | None = None
    api_key: str | None = None
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    window_days: int = 0
    fallback_to_fred_series: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.series_ids:
            msg = "ALFREDConfig.series_ids must contain at least one series"
            raise ValueError(msg)


class ALFREDDataLoader:
    """
    Fetch and persist ALFRED (vintage FRED) releases.
    """

    def __init__(
        self,
        config: ALFREDConfig,
        fred_client: _FredVintageClient | None = None,
    ) -> None:
        if _fredapi is None and fred_client is None:
            check_ml_dependencies(["fredapi"])
        if _pd is None:
            check_ml_dependencies(["pandas"])
        self._config = config
        self._client = fred_client or self._default_client()
        self._fetch_counter = get_counter(
            "nautilus_ml_alfred_fetch_total",
            "Total ALFRED series fetch attempts",
            ["series"],
        )
        self._fetch_error_counter = get_counter(
            "nautilus_ml_alfred_fetch_errors_total",
            "ALFRED series fetch errors",
            ["series"],
        )
        self._fetch_duration_hist = get_histogram(
            "nautilus_ml_alfred_fetch_seconds",
            "ALFRED series fetch duration",
            ["series"],
        )

    def refresh(self) -> dict[str, dict[str, int]]:
        """
        Fetch all configured series and persist vintages.
        """
        stats: dict[str, dict[str, int]] = {}
        for series in self._config.series_ids:
            stats[series] = self._refresh_series(series)
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _default_client(self) -> _FredVintageClient:
        api_key = self._config.api_key or os.getenv("FRED_API_KEY")
        if api_key is None:
            msg = "FRED_API_KEY must be provided via ALFREDConfig.api_key or environment"
            raise ValueError(msg)
        assert _fredapi is not None  # guarded in __init__
        return cast(_FredVintageClient, _fredapi.Fred(api_key=api_key))

    def _refresh_series(self, series_id: str) -> dict[str, int]:
        logger.info("Fetching ALFRED vintages for %s", series_id)
        releases = 0
        rows = 0
        attempt = 0
        while True:
            try:
                attempt += 1
                self._fetch_counter.labels(series=series_id).inc()
                start = time.perf_counter()
                pandas_frame = self._fetch_series_windowed(series_id)
                duration = time.perf_counter() - start
                self._fetch_duration_hist.labels(series=series_id).observe(duration)
                break
            except Exception as exc:  # pragma: no cover - retry path
                self._fetch_error_counter.labels(series=series_id).inc()
                if attempt > self._config.max_retries:
                    logger.exception("Failed to fetch ALFRED series %s", series_id)
                    raise
                logger.warning(
                    "Retrying ALFRED fetch for %s after error: %s",
                    series_id,
                    exc,
                    exc_info=True,
                )
                time.sleep(self._config.retry_delay_seconds)

        if pandas_frame.empty:
            logger.info("No ALFRED rows returned for %s", series_id)
            series_dir = self._config.out_dir / series_id
            series_dir.mkdir(parents=True, exist_ok=True)
            # Write empty calendar to maintain parity with downstream loading logic
            empty = POLARS.DataFrame(
                {
                    "series_id": POLARS.Series([], dtype=POLARS.Utf8),
                    "observation_ts": POLARS.Series([], dtype=POLARS.Datetime("ns")),
                    "value": POLARS.Series([], dtype=POLARS.Float64),
                    "release_ts": POLARS.Series([], dtype=POLARS.Datetime("ns")),
                    "release_end_ts": POLARS.Series([], dtype=POLARS.Datetime("ns")),
                },
            )
            empty.write_parquet(series_dir / "release_calendar.parquet")
            return {"releases": releases, "rows": rows}

        assert _pd is not None
        pandas_frame = pandas_frame.copy()
        pandas_frame["realtime_start"] = (
            PANDAS.to_datetime(pandas_frame["realtime_start"], utc=True)
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
        if "realtime_end" in pandas_frame.columns:
            pandas_frame["realtime_end"] = PANDAS.to_datetime(
                pandas_frame["realtime_end"],
                utc=True,
                errors="coerce",
            )
            if pandas_frame["realtime_end"].notna().any():
                pandas_frame["realtime_end"] = (
                    PANDAS.to_datetime(pandas_frame["realtime_end"], utc=True, errors="coerce")
                    .dt.tz_convert("UTC")
                    .dt.tz_localize(None)
                )
        else:
            pandas_frame["realtime_end"] = PANDAS.NaT
        pandas_frame["date"] = (
            PANDAS.to_datetime(pandas_frame["date"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        )
        pandas_frame["value"] = PANDAS.to_numeric(pandas_frame["value"], errors="coerce")
        pandas_frame = pandas_frame.dropna(subset=["value", "date", "realtime_start"])

        if pandas_frame.empty:
            logger.warning("All ALFRED rows dropped for %s after cleaning", series_id)
            return {"releases": releases, "rows": rows}

        polars_df = _ensure_polars_frame(pandas_frame)
        polars_df = polars_df.rename(
            {
                "date": "observation_ts",
                "realtime_start": "release_ts",
                "realtime_end": "release_end_ts",
            },
        ).with_columns(
            [
                POLARS.lit(series_id).alias("series_id"),
                POLARS.col("observation_ts").cast(POLARS.Datetime("ns")),
                POLARS.col("release_ts").cast(POLARS.Datetime("ns")),
                POLARS.col("release_end_ts").cast(POLARS.Datetime("ns")),
                POLARS.col("value").cast(POLARS.Float64),
            ],
        )

        series_dir = self._config.out_dir / series_id
        series_dir.mkdir(parents=True, exist_ok=True)

        calendar = polars_df.select(
            [
                "series_id",
                "observation_ts",
                "value",
                "release_ts",
                "release_end_ts",
            ],
        ).sort(["observation_ts", "release_ts"])
        calendar.write_parquet(series_dir / "release_calendar.parquet")

        for partition in calendar.partition_by("release_ts", maintain_order=True):
            if partition.is_empty():
                continue
            release_ts_value = partition.get_column("release_ts").item(0)
            if not isinstance(release_ts_value, datetime):
                continue
            release_key = release_ts_value.strftime("%Y%m%d")
            partition.write_parquet(series_dir / f"{release_key}.parquet")
            releases += 1
            rows += partition.height

        logger.info(
            "Stored %d ALFRED releases for %s (%d rows)",
            releases,
            series_id,
            rows,
        )
        return {"releases": releases, "rows": rows}

    def _fetch_series_windowed(self, series_id: str) -> pd.DataFrame:
        import pandas as pd

        start_raw = self._config.start_date
        end_raw = self._config.end_date
        window_days = max(int(self._config.window_days or 0), 0)
        fallback_enabled = series_id in self._config.fallback_to_fred_series
        if not start_raw and not end_raw:
            try:
                return self._client.get_series_all_releases(series_id)
            except Exception:
                if fallback_enabled:
                    return self._fetch_fred_series(series_id)
                raise

        def _parse_day(value: str | None) -> datetime | None:
            if value is None:
                return None
            dt = datetime.strptime(value, "%Y-%m-%d")
            return dt.replace(tzinfo=UTC)

        end_dt = _parse_day(end_raw) or datetime.now(tz=UTC)
        start_dt = _parse_day(start_raw) or (end_dt - timedelta(days=365))

        frames: list[pd.DataFrame] = []
        current_start = start_dt
        if window_days == 0:
            window_days = max((end_dt - start_dt).days, 1)
        delta = timedelta(days=window_days)
        while current_start <= end_dt:
            current_end = min(current_start + delta, end_dt)
            try:
                frame = self._client.get_series_all_releases(
                    series_id,
                    realtime_start=current_start.strftime("%Y-%m-%d"),
                    realtime_end=current_end.strftime("%Y-%m-%d"),
                )
            except Exception:
                if fallback_enabled:
                    return self._fetch_fred_series(series_id)
                raise
            if not frame.empty:
                frames.append(frame)
            current_start = current_end + timedelta(days=1)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["realtime_start", "date"], keep="last")
        return merged

    def _fetch_fred_series(self, series_id: str) -> pd.DataFrame:
        import pandas as pd

        try:
            series = self._client.get_series(
                series_id,
                observation_start=self._config.start_date,
                observation_end=self._config.end_date,
            )
        except Exception:
            logger.warning(
                "ALFRED fallback to FRED series failed for %s",
                series_id,
                exc_info=True,
            )
            return pd.DataFrame()

        if series is None or series.empty:
            return pd.DataFrame()

        frame = series.rename("value").reset_index()
        frame = frame.rename(columns={frame.columns[0]: "date"})
        frame["realtime_start"] = frame["date"]
        frame["realtime_end"] = frame["date"]
        return frame[["realtime_start", "realtime_end", "date", "value"]]


__all__ = ["ALFREDConfig", "ALFREDDataLoader"]
