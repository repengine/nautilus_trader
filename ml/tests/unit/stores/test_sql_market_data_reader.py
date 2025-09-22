#!/usr/bin/env python3

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

from ml.core.db_engine import EngineManager
from ml.stores.providers import SqlMarketDataReader
from ml.registry.dataclasses import DatasetType


@pytest.mark.usefixtures("monkeypatch")
def test_sql_market_data_reader_returns_polars(tmp_path: Path) -> None:
    pl = pytest.importorskip("polars")

    db_path = tmp_path / "reader.db"
    conn_str = f"sqlite:///{db_path}"
    engine = EngineManager.get_engine(conn_str)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    instrument_id TEXT NOT NULL,
                    ts_event BIGINT NOT NULL,
                    ts_init BIGINT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    bid REAL,
                    ask REAL,
                    bid_size REAL,
                    ask_size REAL,
                    last REAL,
                    trade_count INTEGER,
                    vwap REAL,
                    PRIMARY KEY (instrument_id, ts_event)
                )
                """,
            ),
        )
        start_dt = datetime(2024, 1, 1, tzinfo=UTC)
        rows = [
            (
                "SPY.NYSE",
                int(start_dt.timestamp() * 1_000_000_000),
                int(start_dt.timestamp() * 1_000_000_000),
                100.0,
                101.0,
                99.5,
                100.5,
                1000.0,
            ),
            (
                "SPY.NYSE",
                int((start_dt + timedelta(minutes=1)).timestamp() * 1_000_000_000),
                int((start_dt + timedelta(minutes=1)).timestamp() * 1_000_000_000),
                100.5,
                101.5,
                99.8,
                101.0,
                1100.0,
            ),
        ]
        conn.execute(
            text(
                """
                INSERT OR REPLACE INTO market_data(
                    instrument_id,
                    ts_event,
                    ts_init,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    bid,
                    ask,
                    bid_size,
                    ask_size,
                    last,
                    trade_count,
                    vwap
                ) VALUES (
                    :instrument_id,
                    :ts_event,
                    :ts_init,
                    :open,
                    :high,
                    :low,
                    :close,
                    :volume,
                    :bid,
                    :ask,
                    :bid_size,
                    :ask_size,
                    :last,
                    :trade_count,
                    :vwap
                )
                """,
            ),
            [
                {
                    "instrument_id": row[0],
                    "ts_event": row[1],
                    "ts_init": row[2],
                    "open": row[3],
                    "high": row[4],
                    "low": row[5],
                    "close": row[6],
                    "volume": row[7],
                    "bid": None,
                    "ask": None,
                    "bid_size": None,
                    "ask_size": None,
                    "last": None,
                    "trade_count": None,
                    "vwap": None,
                }
                for row in rows
            ],
        )

    reader = SqlMarketDataReader(connection_string=conn_str)
    frame = reader.read_range(
        dataset_type=DatasetType.BARS,
        instrument_id="SPY.NYSE",
        start_ns=rows[0][1],
        end_ns=rows[-1][1] + 1,
    )

    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 2
    assert frame["timestamp"].to_list() == [row[1] for row in rows]
    assert frame["open"].to_list() == [100.0, 100.5]
