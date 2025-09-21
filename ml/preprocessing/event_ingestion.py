"""
Utilities for ingesting scheduled events into normalized datasets.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.ml_types import PolarsDF


if pl is None:
    check_ml_dependencies(["polars"])  # pragma: no cover - ensure availability when used
    import importlib as _importlib

    pl = _importlib.import_module("polars")
POLARS = cast(Any, pl)


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
    """

    start: datetime
    end: datetime
    out_dir: Path = Path("data/events")
    alfred_vintage_dir: Path | None = None
    economic_series: tuple[str, ...] = ("CPI",)
    economic_stub_path: Path | None = None
    corporate_source_path: Path | None = None
    calendar_code: str = "XNYS"
    include_options_expiry: bool = True

    def __post_init__(self) -> None:
        if self.end <= self.start:
            msg = "Event ingestion end must be after start"
            raise ValueError(msg)


class EventIngestionUtility:
    """
    Ingest scheduled events into a normalized Polars dataset.
    """

    def __init__(self, config: EventIngestionConfig) -> None:
        if pl is None:
            check_ml_dependencies(["polars"])
        self._cfg = config
        self._start = _normalize_datetime(config.start)
        self._end = _normalize_datetime(config.end)

    def ingest(self) -> Path:
        """
        Collect events and persist to ``out_dir/events.parquet``.
        """
        events_df = self._collect_events()
        out_dir = self._cfg.out_dir.expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "events.parquet"
        events_df.write_parquet(target)
        return target

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
        assert _pl is not None
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
        assert _pl is not None
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
        assert _pl is not None
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
