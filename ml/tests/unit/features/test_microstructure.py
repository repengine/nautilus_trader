from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import polars as pl

from ml.features.micro_aggregate import MICRO_COLUMNS
from ml.features.micro_aggregate import aggregate_microstructure_minute_pl


def test_aggregate_microstructure_minute_pl_basic() -> None:
    # Two minutes of synthetic quotes and trades
    base = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    ts = [base, base + timedelta(seconds=30), base + timedelta(minutes=1)]
    quotes = pl.DataFrame(
        {
            "ts_event": ts,
            "bid_px_00": [100.0, 100.2, 100.5],
            "ask_px_00": [100.1, 100.3, 100.6],
            "bid_sz_00": [500, 600, 700],
            "ask_sz_00": [400, 650, 650],
        },
    )
    trades = pl.DataFrame(
        {
            "ts_event": ts,
            "price": [100.05, 100.25, 100.55],
            "size": [100, 200, 150],
            "side": ["BUY", "SELL", "BUY"],
        },
    )

    out = aggregate_microstructure_minute_pl(quotes, trades)
    assert out.shape[0] >= 2
    for col in MICRO_COLUMNS:
        assert col in out.columns
    # Midprice roughly between bid/ask
    from typing import Any, cast

    assert float(cast(Any, out["midprice"].drop_nulls().min())) >= 100.0
    assert float(cast(Any, out["midprice"].drop_nulls().max())) <= 100.6
