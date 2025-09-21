from __future__ import annotations

from datetime import datetime, timezone, UTC

import polars as pl

from ml.data.providers.calendar import MarketCalendarProvider
from ml.data.sources.calendar import MockCalendarSource


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


def test_market_calendar_provider_basic_schema_and_props() -> None:
    provider = MarketCalendarProvider(calendar_source=MockCalendarSource())
    # Build a small set of timestamps across a weekday and weekend
    ts_list = [
        _ns(datetime(2025, 8, 15, 12, 0, tzinfo=UTC)),  # Friday
        _ns(datetime(2025, 8, 16, 12, 0, tzinfo=UTC)),  # Saturday
    ]
    timestamps = pl.Series(ts_list)
    df = provider.compute_features(timestamps, exchange="NYSE")

    # Columns of interest exist
    for col in [
        "timestamp",
        "is_trading_day",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]:
        assert col in df.columns

    # Check unit-circle property for encodings (≈1)
    import math

    for row in df.iter_rows(named=True):
        mag_hour = row["hour_sin"] ** 2 + row["hour_cos"] ** 2
        mag_dow = row["dow_sin"] ** 2 + row["dow_cos"] ** 2
        mag_month = row["month_sin"] ** 2 + row["month_cos"] ** 2
        assert math.isclose(mag_hour, 1.0, rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(mag_dow, 1.0, rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(mag_month, 1.0, rel_tol=1e-6, abs_tol=1e-6)

    # Weekend row should be marked appropriately
    weekend_row = df.filter(pl.col("timestamp") == ts_list[1]).row(0, named=True)
    assert weekend_row["is_weekend"] is True
    assert weekend_row["is_trading_day"] is False
