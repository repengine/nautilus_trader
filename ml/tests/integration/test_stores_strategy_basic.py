from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

import pytest

from ml.stores.strategy_store import StrategyStore
from nautilus_trader.model.identifiers import InstrumentId

pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.serial,
    pytest.mark.usefixtures("cloned_test_database"),
    pytest.mark.usefixtures(
        "isolated_prometheus_registry",
        "mock_tracing_backend",
        "isolated_orchestrator_env",
    ),
]



def test_strategy_store_write_and_read(
    cloned_test_database: str,
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
) -> None:
    store = StrategyStore(
        connection_string=cloned_test_database,
        batch_size=2,
        flush_interval_seconds=0.01,
    )

    ts_event, _ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id).split(".")[0]  # Get "EUR/USD" from "EUR/USD.SIM"

    # Two buffered writes should trigger flush by batch size
    store.write_signal(
        strategy_id="strat-A",
        instrument_id=instrument_id_str,
        signal_type="BUY",
        strength=0.8,
        model_predictions={"model_v1": 0.81},
        risk_metrics={"var": 0.01},
        execution_params={"sl": 0.5, "tp": 1.0},
        ts_event=ts_event,
        is_live=False,
    )
    store.write_signal(
        strategy_id="strat-A",
        instrument_id=instrument_id_str,
        signal_type="HOLD",
        strength=0.1,
        model_predictions={},
        risk_metrics={},
        execution_params={},
        ts_event=ts_event + 1,
        is_live=False,
    )

    # Ensure flush occurred and buffer is empty
    store.flush()

    # Read back in a wide range
    start_ns = ts_event - 1000
    end_ns = ts_event + 10_000
    df = store.read_signals("strat-A", instrument_id_str, start_ns, end_ns)
    assert len(df) == 2
    # Assert monotone ordering and first/last types
    assert list(df["signal_type"]) == ["BUY", "HOLD"]
