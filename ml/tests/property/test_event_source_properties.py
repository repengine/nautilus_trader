from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml.data.sources.events import MockEventSource


@settings(deadline=None, max_examples=25)
@given(
    # Constrain to a 365-day window to keep generation bounded and fast
    base=st.datetimes(min_value=datetime(2023, 1, 1), max_value=datetime(2025, 12, 31)),
    days=st.integers(min_value=1, max_value=30),
)
def test_economic_events_within_range_and_sorted(base: datetime, days: int) -> None:
    """
    Economic events are within the requested window and sorted by timestamp.
    """
    start = base.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=days)

    src = MockEventSource()
    events = src.get_economic_events(start, end)

    # Timestamps are sorted and within range
    assert events == sorted(events, key=lambda e: e.timestamp)
    for ev in events:
        assert start <= ev.timestamp <= end
        assert ev.importance in {"HIGH", "MEDIUM", "LOW"}


@settings(deadline=None, max_examples=25)
@given(
    base=st.datetimes(min_value=datetime(2023, 1, 1), max_value=datetime(2025, 12, 31)),
    days=st.integers(min_value=1, max_value=120),
    instruments=st.lists(st.sampled_from(["AAPL.NASDAQ", "MSFT.NASDAQ", "EUR/USD.SIM", "BTC/USDT.BINANCE"]), min_size=1, max_size=4, unique=True),
)
def test_earnings_events_within_range_and_instruments(
    base: datetime,
    days: int,
    instruments: list[str],
) -> None:
    """
    Earnings events returned are within range and tied to requested instruments.
    """
    start = base.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=days)

    src = MockEventSource()
    events = src.get_earnings_events(instruments=instruments, start=start, end=end)

    assert events == sorted(events, key=lambda e: e.timestamp)
    for ev in events:
        assert start <= ev.timestamp <= end
        assert ev.instrument_id in instruments

