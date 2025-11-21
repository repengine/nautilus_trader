from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import polars as pl

from ml.features.l2_aggregate import L2_MINUTE_COLUMNS
from ml.features.l2_aggregate import TOPKS
from ml.features.l2_aggregate import aggregate_l2_minute_pl


def _make_l2_df() -> pl.DataFrame:
    base = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    ts = [base + timedelta(seconds=i) for i in range(90)]  # 1.5 minutes
    # Construct synthetic book: prices increasing on ask, decreasing on bid, sizes constant
    from typing import Any

    data: dict[str, list[Any]] = {"ts_event": list(ts)}
    for i in range(10):
        data[f"bid_px_{i:02d}"] = [100.0 - 0.01 * i] * len(ts)
        data[f"ask_px_{i:02d}"] = [100.02 + 0.01 * i] * len(ts)
        data[f"bid_sz_{i:02d}"] = [1000] * len(ts)
        data[f"ask_sz_{i:02d}"] = [1000] * len(ts)
    return pl.DataFrame(data)


def _make_unsorted_l2_df() -> pl.DataFrame:
    base = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    ts = [base + timedelta(minutes=1), base, base + timedelta(seconds=30)]
    data: dict[str, list[float]] = {"ts_event": ts}
    for i in range(10):
        data[f"bid_px_{i:02d}"] = [100.0 - 0.01 * i] * len(ts)
        data[f"ask_px_{i:02d}"] = [100.02 + 0.01 * i] * len(ts)
        data[f"bid_sz_{i:02d}"] = [1000.0] * len(ts)
        data[f"ask_sz_{i:02d}"] = [1000.0] * len(ts)
    return pl.DataFrame(data)


def test_aggregate_l2_minute_pl_basic() -> None:
    df = _make_l2_df()
    out = aggregate_l2_minute_pl(df)
    assert out.shape[0] >= 1
    cols = set(out.columns)
    assert {"timestamp", "midprice", "spread_bps", "microprice_bps"}.issubset(cols)
    for k in TOPKS:
        assert f"depth_imbalance_top{k}" in cols
        assert f"dwp_bps_top{k}" in cols
        assert f"bid_slope_top{k}" in cols
        assert f"ask_slope_top{k}" in cols
    # Symmetric sizes -> depth imbalance near 0
    for k in TOPKS:
        assert abs(out[f"depth_imbalance_top{k}"][0]) < 1e-6


def test_aggregate_l2_minute_pl_sorts_rows() -> None:
    df = _make_unsorted_l2_df()
    out = aggregate_l2_minute_pl(df)
    timestamps = [ts for ts in out["timestamp"].to_list()]
    assert timestamps == sorted(timestamps)


def test_aggregate_l2_minute_pl_empty_schema() -> None:
    schema = [("ts_event", pl.Datetime("ns", "UTC"))]
    for i in range(10):
        schema.extend(
            [
                (f"bid_px_{i:02d}", pl.Float64),
                (f"ask_px_{i:02d}", pl.Float64),
                (f"bid_sz_{i:02d}", pl.Float64),
                (f"ask_sz_{i:02d}", pl.Float64),
            ],
        )
    empty = pl.DataFrame(schema=schema)
    out = aggregate_l2_minute_pl(empty)
    assert out.is_empty()
    assert tuple(out.columns) == L2_MINUTE_COLUMNS
