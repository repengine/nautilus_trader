"""
Integration test for StrategyStore performance aggregation helpers.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from ml.stores.strategy_store import StrategyStore


pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.serial,
    pytest.mark.usefixtures("clean_postgres_db_module"),
]


def test_strategy_performance_update_and_read(test_database: Any) -> None:
    store = StrategyStore(connection_string=test_database.connection_string)

    strategy_id = "strat_perf"
    instrument_id = "EUR/USD"
    base = int(time.time_ns())
    rows = [
        ("BUY", 0.2, base + 1),
        ("SELL", 0.6, base + 2),
        ("HOLD", 0.4, base + 3),
    ]
    for sig, strength, ts in rows:
        store.write_signal(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            signal_type=sig,
            strength=strength,
            model_predictions={"m": strength},
            risk_metrics={"risk_score": strength},
            execution_params={},
            ts_event=ts,
        )
    store.flush()

    # Aggregate and check
    store.update_performance_metrics(
        strategy_id=strategy_id,
        period_start=base,
        period_end=base + 10,
    )
    perf = store.get_strategy_performance(strategy_id=strategy_id, start_ns=base, end_ns=base + 10)

    assert perf["signal_count"] == 3
    assert perf["buy_count"] == 1
    assert perf["sell_count"] == 1
    assert perf["hold_count"] == 1
    assert 0.0 <= perf["avg_strength"] <= 1.0
    assert 0.0 <= perf["std_strength"]
