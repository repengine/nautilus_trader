from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import polars as pl

from ml.data.providers.events import EventScheduleProvider
from ml.data.sources.events import MockEventSource


def test_event_provider_basic_features() -> None:
    provider = EventScheduleProvider(MockEventSource(seed=123))
    base = datetime(2025, 1, 1, tzinfo=UTC)
    datetimes = [base + timedelta(days=i) for i in range(10)]
    ts = pl.Series("timestamp", datetimes).cast(pl.Datetime("ns", "UTC")).cast(pl.Int64)
    ev = provider.compute_features(ts, instruments=["SPY"])
    assert not ev.is_empty()
    cols = set(ev.columns)
    expected = {
        "timestamp",
        "has_fed_event_today",
        "has_cpi_event_today",
        "has_earnings_today",
        "days_to_next_fed",
        "days_to_next_cpi",
        "days_to_next_earnings",
        "days_since_last_fed",
        "days_since_last_cpi",
        "days_since_last_earnings",
        "event_importance_score",
        "event_clustering_score",
    }
    assert expected.issubset(cols)
