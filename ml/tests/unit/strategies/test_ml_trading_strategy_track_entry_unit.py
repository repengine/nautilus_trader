"""
Track entry mapping coverage for MLTradingStrategy when execute_trades=True.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ml.actors.base import MLSignal
from ml.strategies.ml_strategy import MLTradingStrategy
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue


def _sig(pred: float, conf: float = 0.8) -> MLSignal:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return MLSignal(instrument_id=inst, model_id="m1", prediction=pred, confidence=conf, ts_event=1, ts_init=1)


def test_track_trade_entry_called_when_execute_trades_true() -> None:
    strat = SimpleNamespace()
    # Enable actual track
    strat.track_performance = True
    calls: list[dict[str, Any]] = []
    strat._track_trade_entry = lambda model_id, signal, order_id: calls.append(  # type: ignore[attr-defined]
        {"model_id": model_id, "order_id": order_id}
    )

    # Configuration to allow placing
    strat._config = SimpleNamespace(execute_trades=True)  # type: ignore[attr-defined]
    strat.log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)  # type: ignore[attr-defined]
    strat._active_positions = 0  # type: ignore[attr-defined]
    strat._pending_orders = 0  # type: ignore[attr-defined]
    strat._get_current_position = lambda: None  # type: ignore[attr-defined]
    # Size calculation returns a dummy value
    strat._calculate_position_size = lambda: 1  # type: ignore[attr-defined]
    # Bind enter method and place returns dummy order id
    strat._enter_position = MLTradingStrategy._enter_position.__get__(strat, MLTradingStrategy)  # type: ignore[attr-defined]
    strat._place_market_order = lambda side, qty, **kw: "OID"  # type: ignore[attr-defined]
    # Stub persistence
    strat._persist_strategy_decision = lambda **kw: None  # type: ignore[attr-defined]

    # Execute BUY path
    MLTradingStrategy._process_ml_signal(strat, _sig(0.9))  # type: ignore[misc]
    # Expect tracking call with model id and order id
    assert calls and calls[-1]["model_id"] == "m1" and calls[-1]["order_id"] == "OID"
