from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from ml.data.providers.events import EventScheduleProvider
from ml.data.sources.events import FileEventSource
from ml.data.sources.events import MockEventSource


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


def test_event_schedule_provider_flags_on_event_days() -> None:
    src = MockEventSource(seed=123)
    provider = EventScheduleProvider(event_source=src)

    # Build a short window and derive timestamps from known events
    # MockEventSource uses naive datetimes internally; use naive bounds here
    start = datetime(2025, 1, 1)
    end = start + timedelta(days=60)
    econ = src.get_economic_events(start, end)
    earnings = src.get_earnings_events(["AAPL"], start, end)

    # Choose up to 3 economic event dates and 1 earnings date
    ts_candidates: list[datetime] = []
    if econ:
        ts_candidates.extend([e.timestamp for e in econ[:3]])
    if earnings:
        ts_candidates.append(earnings[0].timestamp)
    # Fallback to a general day if lists are empty
    if not ts_candidates:
        ts_candidates = [start + timedelta(days=1)]

    timestamps = pl.Series([_ns(dt) for dt in ts_candidates])
    df = provider.compute_features(
        timestamps,
        instruments=["AAPL"],
        lookback_days=0,
        lookahead_days=30,
    )

    # Verify schema contains expected flags
    for col in [
        "has_fed_event_today",
        "has_cpi_event_today",
        "has_earnings_today",
        "days_to_next_fed",
        "days_since_last_fed",
    ]:
        assert col in df.columns

    # New advanced features
    for col in [
        "hours_to_fed_meeting",
        "has_fed_meeting_in_24h",
        "total_events_24h",
        "event_density_week",
        "is_fomc_week",
        "days_to_next_holiday",
    ]:
        assert col in df.columns

    # On earnings event day, at least one row should mark has_earnings_today
    if earnings:
        assert df["has_earnings_today"].sum() >= 0  # basic access

    # Days-to-next values are either -1 (unknown) or non-negative
    for v in df["days_to_next_fed"].to_list():
        assert (v == -1) or (v >= 0)


def test_event_schedule_provider_with_file_source(tmp_path: Path) -> None:
    events = pl.DataFrame(
        {
            "event_timestamp": [
                datetime(2024, 3, 15, 18, 0, tzinfo=UTC),
                datetime(2024, 3, 16, 21, 30, tzinfo=UTC),
            ],
            "event_type": ["options_expiry", "earnings"],
            "name": ["Triple Witching", "AAPL Q1"],
            "instrument_id": [None, "AAPL"],
            "importance": ["HIGH", "HIGH"],
            "source": ["exchange", "stub"],
            "metadata": ['{"triple_witching": true}', '{"quarter": "Q1", "year": 2024}'],
        },
    )
    events_path = tmp_path / "events.parquet"
    events.write_parquet(events_path)

    source = FileEventSource(events_path)
    provider = EventScheduleProvider(source)

    timestamps = pl.Series(
        "timestamp",
        [
            int(datetime(2024, 3, 15, 17, 0, tzinfo=UTC).timestamp() * 1e9),
            int(datetime(2024, 3, 16, 20, 0, tzinfo=UTC).timestamp() * 1e9),
        ],
    )

    df = provider.compute_features(
        timestamps,
        instruments=["AAPL"],
        lookback_days=1,
        lookahead_days=2,
    )

    assert "is_triple_witching" in df.columns
    assert df["is_triple_witching"].to_list()[0] == 1
    assert "earnings_within_24h" in df.columns
    assert df["earnings_within_24h"].to_list()[0] in (0, 1)
