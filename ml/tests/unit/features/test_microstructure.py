from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import polars as pl

from ml.features.micro_aggregate import MICRO_COLUMNS
from ml.features.micro_aggregate import MicrostructureAggregator
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


def test_aggregate_microstructure_minute_pl_sorts_inputs() -> None:
    base = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    ts = [base + timedelta(minutes=1), base, base + timedelta(seconds=30)]
    quotes = pl.DataFrame(
        {
            "ts_event": ts,
            "bid_px_00": [100.5, 100.0, 100.2],
            "ask_px_00": [100.6, 100.1, 100.3],
            "bid_sz_00": [700, 500, 600],
            "ask_sz_00": [650, 400, 650],
        },
    )
    trades = pl.DataFrame(
        {
            "ts_event": ts,
            "price": [100.55, 100.05, 100.25],
            "size": [150, 100, 200],
            "side": ["SELL", "BUY", "SELL"],
        },
    )

    out = aggregate_microstructure_minute_pl(quotes, trades)
    timestamps = [ts for ts in out["timestamp"].drop_nulls().to_list()]
    assert timestamps == sorted(timestamps)


def test_aggregate_microstructure_drops_null_timestamps() -> None:
    base = datetime(2025, 1, 1, 9, 30, tzinfo=UTC)
    ts = [base, None, base + timedelta(minutes=1)]
    quotes = pl.DataFrame(
        {
            "ts_event": ts,
            "bid_px_00": [100.0, 100.5, 100.8],
            "ask_px_00": [100.1, 100.6, 100.9],
            "bid_sz_00": [500, 0, 600],
            "ask_sz_00": [400, 0, 650],
        },
    )
    trades = pl.DataFrame(
        {
            "ts_event": ts,
            "price": [100.05, 100.65, 100.85],
            "size": [100, 0, 200],
            "side": ["BUY", "SELL", "BUY"],
        },
    )
    out = aggregate_microstructure_minute_pl(quotes, trades)
    assert not out.is_empty()


def test_micro_aggregator_resolves_symbol_with_venue(tmp_path: Path) -> None:
    base_dir = tmp_path / "tier1"
    symbol_dir = base_dir / "SPY.XNAS" / "l1"
    symbol_dir.mkdir(parents=True)
    _write_l1_file(symbol_dir / "SPY.XNAS_bbo.parquet", mid=100.0)
    _write_trade_file(symbol_dir / "SPY.XNAS_trades.parquet")

    agg = MicrostructureAggregator(base_dir)
    df = agg.compute_for_symbol("SPY")
    assert not df.is_empty()
    assert "midprice" in df.columns


def test_micro_aggregator_prefers_latest_file(tmp_path: Path) -> None:
    base_dir = tmp_path / "tier1"
    l1_dir = base_dir / "SPY" / "l1"
    l1_dir.mkdir(parents=True)
    dated = l1_dir / "SPY_bbo_2024-01-01_2024-01-05.parquet"
    latest = l1_dir / "SPY_bbo.parquet"
    _write_l1_file(dated, mid=10.0)
    import time
    time.sleep(0.01)  # Ensure filesystem timestamp difference
    _write_l1_file(latest, mid=50.0)
    
    # Explicitly set mtime to ensure latest is newer (in case of low resolution fs)
    import os
    stat = dated.stat()
    os.utime(latest, (stat.st_atime + 10, stat.st_mtime + 10))
    
    _write_trade_file(l1_dir / "SPY_trades.parquet")

    agg = MicrostructureAggregator(base_dir)
    df = agg.compute_for_symbol("SPY")
    assert float(df["midprice"].drop_nulls().mean()) > 20.0


def _write_l1_file(path: Path, mid: float) -> None:
    ts = [
        datetime(2025, 8, 4, 9, 30, tzinfo=UTC),
        datetime(2025, 8, 4, 9, 31, tzinfo=UTC),
    ]
    df = pl.DataFrame(
        {
            "ts_event": ts,
            "bid_px_00": [mid - 0.05, mid - 0.04],
            "ask_px_00": [mid + 0.05, mid + 0.04],
            "bid_sz_00": [100, 110],
            "ask_sz_00": [90, 95],
        },
    )
    df.write_parquet(path)


def _write_trade_file(path: Path) -> None:
    ts = [
        datetime(2025, 8, 4, 9, 30, tzinfo=UTC),
        datetime(2025, 8, 4, 9, 31, tzinfo=UTC),
    ]
    trades = pl.DataFrame(
        {
            "ts_event": ts,
            "price": [100.0, 100.1],
            "size": [50, 75],
            "side": ["BUY", "SELL"],
        },
    )
    trades.write_parquet(path)
