"""
Track entry mapping coverage for MLTradingStrategy when execute_trades=True.
"""

from __future__ import annotations

from typing import Any, cast

from ml.actors.base import MLSignal
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.tests.utils.stubs import LoggerStub
from ml.tests.utils.stubs import build_ml_trading_strategy_stub
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


def _sig(pred: float, conf: float = 0.8) -> MLSignal:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return MLSignal(
        instrument_id=inst,
        model_id="m1",
        prediction=pred,
        confidence=conf,
        metadata={"decision_metadata": {"version": "v1"}},
        ts_event=1,
        ts_init=1,
    )


def test_track_trade_entry_called_when_execute_trades_true() -> None:
    strat = build_ml_trading_strategy_stub(execute_trades=True)
    strat.log = LoggerStub()
    strat.track_performance = True

    calls: list[tuple[str, str]] = []

    def _track(model_id: str, signal: MLSignal, order_id: object) -> None:
        del signal
        calls.append((model_id, str(order_id)))

    def _place_market_order(*args: object, **kwargs: object) -> str:
        del args, kwargs
        return "OID"

    cast(Any, strat)._track_trade_entry = _track
    cast(Any, strat)._pending_orders = 0
    cast(Any, strat)._calculate_position_size = lambda: 1
    cast(Any, strat)._enter_position = MLTradingStrategy._enter_position.__get__(
        strat,
        MLTradingStrategy,
    )
    cast(Any, strat)._place_market_order = _place_market_order
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    # Execute BUY path
    MLTradingStrategy._process_ml_signal(strat, _sig(0.9))
    # Expect tracking call with model id and order id
    assert calls and calls[-1] == ("m1", "OID")
