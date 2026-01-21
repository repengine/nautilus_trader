"""
Utilities for ingesting scheduled events into normalized datasets.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.common.timestamps import sanitize_timestamp_ns
from ml.config.dataset_ids import EVENTS_CALENDAR_DATASET_ID
from ml.config.events import Source
from ml.config.ingestion_windows import WatermarkWindowConfig
from ml.config.ingestion_windows import events_window_defaults
from ml.ml_types import PolarsDF
from ml.stores.protocols import DataStoreFacadeProtocol


if pl is None:
    check_ml_dependencies(["polars"])  # pragma: no cover - ensure availability when used
    import importlib as _importlib

    pl = _importlib.import_module("polars")
POLARS = cast(Any, pl)
logger = logging.getLogger(__name__)


DEFAULT_EVENT_TYPES: tuple[str, ...] = (
    "fed_meeting",
    "economic_release",
    "earnings",
    "options_expiry",
    "holiday",
)


def _normalize_datetime(dt: datetime) -> datetime:
    """
    Return a timezone-naive UTC datetime for consistent persistence.
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(UTC).replace(tzinfo=None)


def _iter_months(start: datetime, end: datetime) -> Iterable[tuple[int, int]]:
    """
    Yield (year, month) pairs covering the inclusive range.
    """
    year = start.year
    month = start.month
    while year < end.year or (year == end.year and month <= end.month):
        yield year, month
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1


def _third_friday(year: int, month: int) -> datetime:
    """
    Compute the third Friday for a given month/year at 16:00 UTC.
    """
    dt = datetime(year, month, 1, 16, 0, tzinfo=UTC)
    friday_count = 0
    while True:
        if dt.weekday() == 4:  # Friday
            friday_count += 1
            if friday_count == 3:
                return dt
        dt += timedelta(days=1)


def _quarterly_months() -> tuple[int, ...]:
    """
    Return canonical earnings months (Jan/Apr/Jul/Oct).
    """
    return (1, 4, 7, 10)


@dataclass(slots=True, frozen=True)
class EventIngestionConfig:
    """
    Configuration for :class:`EventIngestionUtility`.

    Attributes
    ----------
    start : datetime
        Inclusive start datetime for event ingestion.
    end : datetime
        Inclusive end datetime for event ingestion.
    out_dir : Path
        Output directory for event artifacts.
    write_parquet : bool
        Whether to persist events to parquet on disk.
    alfred_vintage_dir : Path | None
        Optional ALFRED vintages directory to load macro release events.
    economic_series : tuple[str, ...]
        Macro series identifiers for economic stub events.
    economic_stub_path : Path | None
        Optional custom stub data for economic events.
    corporate_source_path : Path | None
        Optional corporate events file path.
    calendar_code : str
        Exchange calendar code for holiday schedules.
    include_options_expiry : bool
        Whether to include options expiry events.
    watermark_config : WatermarkWindowConfig
        Window configuration used to derive incremental ranges from watermarks.
    """

    start: datetime
    end: datetime
    out_dir: Path = Path("data/features/events")
    write_parquet: bool = True
    alfred_vintage_dir: Path | None = None
    economic_series: tuple[str, ...] = ("CPI",)
    economic_stub_path: Path | None = None
    corporate_source_path: Path | None = None
    calendar_code: str = "XNYS"
    include_options_expiry: bool = True
    watermark_config: WatermarkWindowConfig = field(default_factory=events_window_defaults)

    def __post_init__(self) -> None:
        if self.end <= self.start:
            msg = "Event ingestion end must be after start"
            raise ValueError(msg)


class EventIngestionUtility:
    """
    Ingest scheduled events into a normalized Polars dataset.
    """

    def __init__(
        self,
        config: EventIngestionConfig,
        *,
        data_store: DataStoreFacadeProtocol | None = None,
        ingest_run_id: str | None = None,
    ) -> None:
        if pl is None:
            check_ml_dependencies(["polars"])
        self._cfg = config
        self._start = _normalize_datetime(config.start)
        self._end = _normalize_datetime(config.end)
        self._latest_frame: PolarsDF | None = None
        self._data_store = data_store
        self._ingest_run_id = ingest_run_id or "event_ingestion"

    def ingest(self) -> Path:
        """
        Collect events and persist to ``out_dir/events.parquet``.
        """
        events_df = self._collect_events()
        self._latest_frame = events_df.clone()
        out_dir = self._cfg.out_dir.expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "events.parquet"
        if self._cfg.write_parquet:
            events_df.write_parquet(target)
        if self._data_store is not None:
            self._ingest_sql(events_df)
        return target

    def latest_frame(self) -> PolarsDF | None:
        """Return the most recent events frame produced by :meth:`ingest`."""
        return self._latest_frame

    # ------------------------------------------------------------------
    # Event collection helpers
    # ------------------------------------------------------------------
    def _collect_events(self) -> PolarsDF:
        _pl = POLARS
        events: list[dict[str, Any]] = []
        events.extend(self._generate_fomc_events())
        events.extend(self._generate_options_expiry_events())
        events.extend(self._generate_quarterly_earnings_stub())
        events.extend(self._load_economic_stub())
        events.extend(self._load_alfred_vintages())
        events.extend(self._generate_holiday_events())
        events.extend(self._load_corporate_events())

        if not events:
            schema = {
                "event_timestamp": _pl.Datetime("ns"),
                "event_type": _pl.Utf8,
                "name": _pl.Utf8,
                "instrument_id": _pl.Utf8,
                "importance": _pl.Utf8,
                "source": _pl.Utf8,
                "metadata": _pl.Utf8,
            }
            return cast(PolarsDF, _pl.DataFrame(schema=schema))

        df = _pl.DataFrame(events)
        df = df.with_columns(
            [
                _pl.col("event_timestamp").cast(_pl.Datetime("ns")),
                _pl.col("instrument_id").fill_null(""),
                _pl.col("importance").fill_null("MEDIUM"),
                _pl.col("source").fill_null("ingestion"),
            ],
        )
        df = df.sort(["event_timestamp", "event_type", "name"])
        return cast(PolarsDF, df)

    def _ingest_sql(self, frame: PolarsDF) -> None:
        if frame.is_empty():
            return
        _pl = POLARS
        ts_init = sanitize_timestamp_ns(time.time_ns(), context="events_ingest")
        prepared = (
            frame.with_columns(
                [
                    _pl.col("event_timestamp")
                    .cast(_pl.Datetime("ns"))
                    .cast(_pl.Int64)
                    .alias("event_timestamp"),
                    _pl.col("event_timestamp")
                    .cast(_pl.Datetime("ns"))
                    .cast(_pl.Int64)
                    .alias("ts_event"),
                    _pl.lit(ts_init).alias("ts_init"),
                ],
            )
            .with_columns(
                [
                    _pl.col("instrument_id").fill_null("").cast(_pl.Utf8),
                    _pl.col("metadata").fill_null("").cast(_pl.Utf8),
                ],
            )
            .select(
                [
                    "event_timestamp",
                    "event_type",
                    "name",
                    "instrument_id",
                    "importance",
                    "source",
                    "metadata",
                    "ts_event",
                    "ts_init",
                ],
            )
        )
        try:
            self._data_store.write_ingestion(  # type: ignore[union-attr]
                dataset_id=EVENTS_CALENDAR_DATASET_ID,
                records=prepared,
                source=Source.HISTORICAL.value,
                run_id=self._ingest_run_id,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "event_ingestion.sql_ingest_failed",
                exc_info=True,
            )

    def _generate_fomc_events(self) -> list[dict[str, Any]]:
        fomc_dates = [
            datetime(2024, 1, 31, 19, 0, tzinfo=UTC),
            datetime(2024, 3, 20, 18, 0, tzinfo=UTC),
            datetime(2024, 5, 1, 18, 0, tzinfo=UTC),
            datetime(2024, 6, 12, 18, 0, tzinfo=UTC),
            datetime(2024, 7, 31, 18, 0, tzinfo=UTC),
            datetime(2024, 9, 18, 18, 0, tzinfo=UTC),
            datetime(2024, 11, 7, 18, 0, tzinfo=UTC),
            datetime(2024, 12, 18, 18, 0, tzinfo=UTC),
        ]
        results: list[dict[str, Any]] = []
        for dt in fomc_dates:
            dt_norm = _normalize_datetime(dt)
            if self._start <= dt_norm <= self._end:
                results.append(
                    {
                        "event_timestamp": dt_norm,
                        "event_type": "fed_meeting",
                        "name": "Federal Funds Rate Decision",
                        "instrument_id": None,
                        "importance": "HIGH",
                        "source": "federal_reserve",
                        "metadata": json.dumps({"series": "FOMC"}),
                    },
                )
        return results

    def _generate_options_expiry_events(self) -> list[dict[str, Any]]:
        if not self._cfg.include_options_expiry:
            return []
        results: list[dict[str, Any]] = []
        for year, month in _iter_months(self._start, self._end):
            expiry = _third_friday(year, month)
            dt_norm = _normalize_datetime(expiry)
            if self._start <= dt_norm <= self._end:
                metadata = {"triple_witching": month in (3, 6, 9, 12)}
                name = "Triple Witching" if metadata["triple_witching"] else "Options Expiry"
                results.append(
                    {
                        "event_timestamp": dt_norm,
                        "event_type": "options_expiry",
                        "name": name,
                        "instrument_id": None,
                        "importance": "MEDIUM",
                        "source": "exchange",
                        "metadata": json.dumps(metadata),
                    },
                )
        return results

    def _generate_quarterly_earnings_stub(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for year in range(self._start.year - 1, self._end.year + 1):
            for quarter_month in _quarterly_months():
                # Earnings reported roughly 20th of quarter month at 21:30 UTC (after close)
                try:
                    ts = datetime(year, quarter_month, 20, 21, 30, tzinfo=UTC)
                except ValueError:
                    continue
                ts_norm = _normalize_datetime(ts)
                if not (self._start <= ts_norm <= self._end):
                    continue
                quarter = f"Q{((quarter_month - 1) // 3) + 1}"
                results.append(
                    {
                        "event_timestamp": ts_norm,
                        "event_type": "earnings",
                        "name": f"Earnings Season {quarter}",
                        "instrument_id": None,
                        "importance": "MEDIUM",
                        "source": "stub",
                        "metadata": json.dumps({"quarter": quarter, "year": year}),
                    },
                )
        return results

    def _load_economic_stub(self) -> list[dict[str, Any]]:
        if self._cfg.economic_stub_path is None:
            return []
        _pl = pl
        if _pl is None:
            raise RuntimeError("Polars runtime unavailable for economic stub loading")
        df = _pl.read_csv(str(self._cfg.economic_stub_path))
        records: list[dict[str, Any]] = []
        for row in df.iter_rows(named=True):
            ts = _normalize_datetime(datetime.fromisoformat(str(row["timestamp"])))
            if not (self._start <= ts <= self._end):
                continue
            records.append(
                {
                    "event_timestamp": ts,
                    "event_type": str(row.get("event_type", "economic_release")),
                    "name": str(row.get("name", "Economic Release")),
                    "instrument_id": row.get("instrument_id"),
                    "importance": str(row.get("importance", "MEDIUM")),
                    "source": str(row.get("source", "stub")),
                    "metadata": json.dumps(
                        {
                            k: row[k]
                            for k in row.keys()
                            if k
                            not in {
                                "timestamp",
                                "event_type",
                                "name",
                                "instrument_id",
                                "importance",
                                "source",
                            }
                        },
                    ),
                },
            )
        return records

    def _load_alfred_vintages(self) -> list[dict[str, Any]]:
        if self._cfg.alfred_vintage_dir is None:
            return []
        _pl = pl
        if _pl is None:
            raise RuntimeError("Polars runtime unavailable for ALFRED vintage loading")
        results: list[dict[str, Any]] = []
        for series in self._cfg.economic_series:
            cal_path = self._cfg.alfred_vintage_dir / series / "release_calendar.parquet"
            if not cal_path.exists():
                continue
            df = _pl.read_parquet(str(cal_path))
            if df.is_empty():
                continue
            # Each row corresponds to an observation; treat release_ts as event timestamp
            df = df.drop_nulls("release_ts")
            for row in df.iter_rows(named=True):
                release_ts = row["release_ts"]
                if release_ts is None:
                    continue
                release_dt = _normalize_datetime(release_ts)
                if not (self._start <= release_dt <= self._end):
                    continue
                metadata = {
                    "series_id": series,
                    "observation_ts": str(row.get("observation_ts")),
                }
                results.append(
                    {
                        "event_timestamp": release_dt,
                        "event_type": "economic_release",
                        "name": f"{series} Release",
                        "instrument_id": None,
                        "importance": "MEDIUM",
                        "source": "alfred",
                        "metadata": json.dumps(metadata),
                    },
                )
        return results

    def _load_corporate_events(self) -> list[dict[str, Any]]:
        if self._cfg.corporate_source_path is None:
            return []
        _pl = pl
        if _pl is None:
            raise RuntimeError("Polars runtime unavailable for corporate event loading")
        df = _pl.read_csv(str(self._cfg.corporate_source_path))
        results: list[dict[str, Any]] = []
        for row in df.iter_rows(named=True):
            ts = _normalize_datetime(datetime.fromisoformat(str(row["timestamp"])))
            if not (self._start <= ts <= self._end):
                continue
            results.append(
                {
                    "event_timestamp": ts,
                    "event_type": str(row.get("event_type", "earnings")),
                    "name": str(row.get("name", "Corporate Event")),
                    "instrument_id": row.get("instrument_id"),
                    "importance": str(row.get("importance", "MEDIUM")),
                    "source": str(row.get("source", "corporate")),
                    "metadata": json.dumps(
                        {
                            k: row[k]
                            for k in row.keys()
                            if k
                            not in {
                                "timestamp",
                                "event_type",
                                "name",
                                "instrument_id",
                                "importance",
                                "source",
                            }
                        },
                    ),
                },
            )
        return results

    def _generate_holiday_events(self) -> list[dict[str, Any]]:
        holidays: list[datetime] = []
        years = range(self._start.year, self._end.year + 1)
        for year in years:
            holidays.extend(
                [
                    datetime(year, 1, 1, 0, 0, tzinfo=UTC),  # New Year's Day
                    datetime(year, 7, 4, 0, 0, tzinfo=UTC),  # Independence Day
                    datetime(year, 12, 25, 0, 0, tzinfo=UTC),  # Christmas
                ],
            )
            # Thanksgiving: fourth Thursday of November
            november = datetime(year, 11, 1, tzinfo=UTC)
            thursday_count = 0
            day = november
            while day.month == 11:
                if day.weekday() == 3:
                    thursday_count += 1
                    if thursday_count == 4:
                        holidays.append(day)
                        break
                day += timedelta(days=1)

        results: list[dict[str, Any]] = []
        for dt in holidays:
            dt_norm = _normalize_datetime(dt)
            if self._start <= dt_norm <= self._end:
                results.append(
                    {
                        "event_timestamp": dt_norm,
                        "event_type": "holiday",
                        "name": "Exchange Holiday",
                        "instrument_id": None,
                        "importance": "LOW",
                        "source": "calendar_stub",
                        "metadata": json.dumps({"calendar": self._cfg.calendar_code}),
                    },
                )
        return results
