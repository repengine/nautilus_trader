"""
Sizing rejection persistence tests for MLTradingStrategy.
"""

from __future__ import annotations

from typing import Any, cast

from ml.actors.base import MLSignal
from ml.tests.utils.stubs import LoggerStub
from ml.tests.utils.stubs import build_ml_trading_strategy_stub
from nautilus_trader.model.enums import OrderSide
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
        ts_event=1,
        ts_init=1,
    )


def test_sizing_reject_persists_hold_with_reason() -> None:
    strat = build_ml_trading_strategy_stub(execute_trades=True)
    strat.log = LoggerStub()
    strat._config.persist_hold_on_sizing_reject = True
    captured: dict[str, object] = {}

    def _persist_strategy_decision(
        *,
        signal: MLSignal,
        decision_type: str,
        position_size: object,
        risk_metrics: object,
        execution_params: object,
        persist_hold: bool = False,
    ) -> None:
        del signal, decision_type, position_size, risk_metrics
        captured["execution_params"] = execution_params
        captured["persist_hold"] = persist_hold

    cast(Any, strat).size_and_validate = lambda _signal: None
    cast(Any, strat)._get_sizing_reject_reason = lambda: "risk_rejected"
    cast(Any, strat)._persist_strategy_decision = _persist_strategy_decision

    from ml.strategies.ml_strategy import MLTradingStrategy

    MLTradingStrategy._enter_position(strat, OrderSide.BUY, _sig(0.9))

    assert captured.get("persist_hold") is True
    exec_params = cast(dict[str, object], captured.get("execution_params"))
    assert exec_params["reason"] == "sizing_rejected"
    assert exec_params["sizing_reject_reason"] == "risk_rejected"
    assert exec_params["intended_action"] == "enter"
