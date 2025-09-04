from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import polars as pl

import ml.data.fred_join as fred_mod
from ml.data.fred_join import join_fred_asof


def test_join_fred_asof_polars_basic(monkeypatch) -> None:
    # Market data every minute
    base = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    ts = [base + timedelta(minutes=i) for i in range(6)]
    left = pl.DataFrame({"timestamp": ts, "close": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]})
    # FRED releases at T0 and T+3min
    fred_ml = pl.DataFrame(
        {
            "timestamp": [base, base + timedelta(minutes=3)],
            "series_id": ["VIXCLS", "VIXCLS"],
            "value": [10.0, 12.0],
        },
    )

    # Monkeypatch loader to return our fred data
    monkeypatch.setattr(fred_mod, "_load_fred_ml_pl", lambda fred_path=None: fred_ml)
    joined = join_fred_asof(left, lag_days=0)
    assert "VIXCLS" in joined.columns
    # After 9:30 VIX=10 until 9:32, then VIX=12 from 9:33
    vals = joined.select("VIXCLS").to_series().to_list()
    assert vals[0] == 10.0
    assert vals[3] == 12.0
