"""
Smoke test for MLTradingStrategy decision flow in dry-run mode.

Bypasses Nautilus base initialization by constructing an instance via object.__new__ and
stubbing required methods/attributes. Verifies that _process_ml_signal records a
decision and increments dry-run counters.

"""

from __future__ import annotations

from typing import Any, cast

from ml.actors.base import MLSignal
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.tests.utils.stubs import LoggerStub
from ml.tests.utils.stubs import StrategyDecisionRecorder
from ml.tests.utils.stubs import build_ml_trading_strategy_stub
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue


def _build_signal(pred: float, conf: float) -> MLSignal:
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


def test_ml_trading_strategy_process_signal_dry_run() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(decision_recorder=recorder)
    strat.log = LoggerStub()
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    # Process a bullish signal (>0.5) by calling the unbound method
    sig = _build_signal(0.9, 0.8)
    MLTradingStrategy._process_ml_signal(strat, sig)

    # One dry run trade and one persisted decision
    assert strat._dry_run_trades == 1
    assert len(recorder.records) == 1
    assert recorder.records[0].decision_type in {"BUY", "SELL", "HOLD"}
