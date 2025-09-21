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
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

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
    ) -> Any:  # pragma: no cover - Protocol
        ...


def _ensure_polars_frame(obj: Any) -> PolarsDF:
    """
    Convert pandas DataFrame to Polars with normalized schema.
    """
    df = POLARS.from_pandas(obj, include_index=False)
    return cast(PolarsDF, df)


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
                df = self._client.get_series_all_releases(
                    series_id,
                    realtime_start=self._config.start_date,
                    realtime_end=self._config.end_date,
                )
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
                )
                time.sleep(self._config.retry_delay_seconds)

        if df.empty:
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
        df = df.copy()  # Avoid mutating caller-owned frame
        df["realtime_start"] = (
            PANDAS.to_datetime(df["realtime_start"], utc=True)
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
        df["realtime_end"] = PANDAS.to_datetime(df["realtime_end"], utc=True, errors="coerce")
        if df["realtime_end"].notna().any():
            df["realtime_end"] = (
                PANDAS.to_datetime(df["realtime_end"], utc=True, errors="coerce")
                .dt.tz_convert("UTC")
                .dt.tz_localize(None)
            )
        df["date"] = (
            PANDAS.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        )
        df["value"] = PANDAS.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value", "date", "realtime_start"])

        if df.empty:
            logger.warning("All ALFRED rows dropped for %s after cleaning", series_id)
            return {"releases": releases, "rows": rows}

        polars_df = _ensure_polars_frame(df)
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


__all__ = ["ALFREDConfig", "ALFREDDataLoader"]
