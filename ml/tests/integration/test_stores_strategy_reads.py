"""
DB-backed read/stat tests for StrategyStore.

These tests write a handful of rows via the public write APIs, flush, and then
exercise read_signals/get_latest/get_statistics/get_signal_distribution.
"""

from __future__ import annotations

import time

import pytest

from ml.stores.strategy_store import StrategyStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.usefixtures("clean_postgres_db_module"),
]


def test_strategy_store_reads_and_stats(test_database) -> None:
    """
    StrategyStore read_signals/get_latest/get_statistics/distribution behave as expected.
    """
    store = StrategyStore(connection_string=test_database.connection_string)

    base_ts = int(time.time_ns())
    strategy_id = "stratA"
    instrument_id = "EUR/USD"

    # Insert a small set of signals (BUY, SELL, HOLD)
    rows = [
        ("BUY", 0.8, base_ts + 1),
        ("SELL", 0.3, base_ts + 2),
        ("HOLD", 0.5, base_ts + 3),
    ]
    for sig_type, strength, ts in rows:
        store.write_signal(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            signal_type=sig_type,
            strength=strength,
            model_predictions={"m": strength},
            risk_metrics={"risk_score": 0.1},
            execution_params={"th": 0.1},
            ts_event=ts,
        )

    # Persist and emit registry events
    store.flush()

    # get_latest returns most recent first, limited
    latest = store.get_latest(instrument_id=instrument_id, limit=2)
    assert len(latest.index) == 2
    # Ensure ordering is descending by ts_event
    assert int(latest.iloc[0]["ts_event"]) == rows[-1][2]
    assert int(latest.iloc[1]["ts_event"]) == rows[-2][2]
    assert latest.iloc[0]["strategy_id"] == strategy_id

    # read_signals returns within window and ordered by ts_event asc
    df = store.read_signals(
        strategy_id=strategy_id,
        instrument_id=instrument_id,
        start_ns=rows[0][2],
        end_ns=rows[-1][2] + 1,
    )
    assert len(df.index) == 3
    assert list(df["signal_type"]) == [r[0] for r in rows]

    # get_statistics aggregates counts and min/max
    stats = store.get_statistics(start_ns=rows[0][2], end_ns=rows[-1][2] + 1)
    assert stats["total_signals"] == 3
    assert stats["unique_strategies"] == 1
    assert stats["unique_instruments"] == 1
    assert stats["buy_signals"] == 1
    assert stats["sell_signals"] == 1
    assert stats["hold_signals"] == 1
    assert int(stats["min_timestamp_ns"]) == rows[0][2]
    assert int(stats["max_timestamp_ns"]) == rows[-1][2]
    # avg_strength is a float and within range
    assert isinstance(stats["avg_strength"], float)
    assert 0.0 <= stats["avg_strength"] <= 1.0

    # get_signal_distribution with and without filter
    dist_all = store.get_signal_distribution()
    assert dist_all.get("BUY", 0) == 1
    assert dist_all.get("SELL", 0) == 1
    assert dist_all.get("HOLD", 0) == 1

    dist_strat = store.get_signal_distribution(strategy_id=strategy_id)
    assert dist_strat == dist_all

