"""
Additional branch coverage tests for MLTradingStrategy.

Covers reversal and hold branches using a dummy self instance and stubs for
methods/attributes.

"""

from __future__ import annotations

from typing import Any, Callable, cast

from ml.actors.base import MLSignal
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.tests.utils.stubs import LoggerStub
from ml.tests.utils.stubs import StrategyDecisionRecorder
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
        ts_event=1,
        ts_init=1,
    )


def test_reversal_branch_dry_run() -> None:
    class _Pos:
        def __init__(self, side_name: str) -> None:
            self.side = type("_Side", (), {"name": side_name})()
            self.quantity = object()

    strat = build_ml_trading_strategy_stub()
    strat.log = LoggerStub()
    cast(Any, strat)._active_positions = 1

    def _current() -> _Pos:
        return _Pos("LONG")

    cast(Any, strat)._get_current_position = _current
    cast(Any, strat)._should_reverse_position = MLTradingStrategy._should_reverse_position.__get__(
        strat,
        MLTradingStrategy,
    )
    cast(Any, strat)._reverse_position = MLTradingStrategy._reverse_position.__get__(
        strat,
        MLTradingStrategy,
    )
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    # Signal indicates SELL vs current LONG -> reversal path
    MLTradingStrategy._process_ml_signal(strat, _sig(0.0))
    assert strat._dry_run_trades == 1


def test_hold_branch_persists_decision() -> None:
    class _Pos:
        def __init__(self, side_name: str) -> None:
            self.side = type("_Side", (), {"name": side_name})()

    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(decision_recorder=recorder)
    strat.log = LoggerStub()
    cast(Any, strat)._active_positions = 1

    cast(Any, strat)._get_current_position = lambda: _Pos("LONG")

    def _no_reverse(current: object, target: object) -> bool:
        del current, target
        return False

    cast(Any, strat)._should_reverse_position = cast(Callable[[object, object], bool], _no_reverse)
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    MLTradingStrategy._process_ml_signal(strat, _sig(0.9))
    assert recorder.records and recorder.records[-1].decision_type == "HOLD"
