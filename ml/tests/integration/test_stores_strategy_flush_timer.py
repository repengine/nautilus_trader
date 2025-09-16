from __future__ import annotations

import time

import pytest

from ml.stores.strategy_store import StrategyStore


class FakeClock:
    def __init__(self, ns: int) -> None:
        self._ns = ns

    def timestamp_ns(self) -> int:
        return self._ns

    def advance_ms(self, ms: int) -> None:
        self._ns += int(ms * 1_000_000)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.usefixtures("clean_postgres_db_module"),
]


def test_flush_by_time_trigger(clean_postgres_db, postgres_connection: str) -> None:  # type: ignore[override]
    now = time.time_ns()
    clock = FakeClock(now)
    store = StrategyStore(
        connection_string=postgres_connection,
        batch_size=1000,
        flush_interval_seconds=0.01,
        clock=clock,
    )

    # Seed a first record and flush to initialize last_flush_ns
    store.write_signal(
        strategy_id="S",
        instrument_id="SPY",
        signal_type="BUY",
        strength=0.5,
        model_predictions={},
        risk_metrics={},
        execution_params={},
        ts_event=now,
        is_live=False,
    )
    store.flush()

    # Advance time beyond flush interval; next write should flush due to time
    clock.advance_ms(100)

    store.write_signal(
        strategy_id="S",
        instrument_id="SPY",
        signal_type="SELL",
        strength=0.4,
        model_predictions={},
        risk_metrics={},
        execution_params={},
        ts_event=now + 1,
        is_live=False,
    )

    # Ensure persisted by reading the range
    df = store.read_signals("S", "SPY", now - 1000, now + 10_000)
    assert len(df) >= 2
