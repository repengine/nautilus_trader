from __future__ import annotations

# ruff: noqa: I001

from datetime import UTC, datetime, timedelta

import polars as pl

from ml.features.l2_aggregate import TOPKS, aggregate_l2_minute_pl


def _make_l2_df(scale: float) -> pl.DataFrame:
    base = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    ts = [base + timedelta(seconds=i) for i in range(60)]
    data = {"ts_event": ts}
    for i in range(10):
        data[f"bid_px_{i:02d}"] = [100.0 - 0.01 * i] * len(ts)
        data[f"ask_px_{i:02d}"] = [100.02 + 0.01 * i] * len(ts)
        data[f"bid_sz_{i:02d}"] = [1000 * scale] * len(ts)
        data[f"ask_sz_{i:02d}"] = [1000 * scale] * len(ts)
    return pl.DataFrame(data)


def test_l2_invariance_under_size_scaling() -> None:
    out1 = aggregate_l2_minute_pl(_make_l2_df(1.0))
    out2 = aggregate_l2_minute_pl(_make_l2_df(10.0))
    # Compare per-minute row 0
    r1 = out1.row(0, named=True)
    r2 = out2.row(0, named=True)
    for k in TOPKS:
        assert abs(r1[f"depth_imbalance_top{k}"] - r2[f"depth_imbalance_top{k}"]) < 1e-9
        assert abs(r1[f"dwp_bps_top{k}"] - r2[f"dwp_bps_top{k}"]) < 1e-9
        assert abs(r1[f"bid_slope_top{k}"] - r2[f"bid_slope_top{k}"]) < 1e-9
        assert abs(r1[f"ask_slope_top{k}"] - r2[f"ask_slope_top{k}"]) < 1e-9
