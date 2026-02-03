"""
Short-entry policy tests for MLTradingStrategy.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, cast

from ml.actors.base import MLSignal
from ml.config.base import AccountMode
from ml.config.base import ShortEntryPolicy
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.tests.utils.stubs import LoggerStub
from ml.tests.utils.stubs import StrategyDecisionRecorder
from ml.tests.utils.stubs import build_ml_trading_strategy_stub
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Quantity


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


def test_short_entry_blocked_when_exit_only_and_no_position() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(execute_trades=True, decision_recorder=recorder)
    strat.log = LoggerStub()
    strat._config.account_mode = AccountMode.CASH
    strat._config.short_entry_policy = ShortEntryPolicy.EXIT_ONLY
    cast(Any, strat)._get_current_position = lambda: None
    entered: list[OrderSide] = []

    def _enter(side: OrderSide, _signal: MLSignal) -> None:
        entered.append(side)

    cast(Any, strat)._enter_position = _enter
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    MLTradingStrategy._process_ml_signal(strat, _sig(0.1))

    assert not entered
    assert recorder.records
    record = recorder.records[-1]
    assert record.decision_type == "HOLD"
    exec_params = record.execution_params
    assert exec_params["action"] == "hold"
    assert exec_params["reason"] == "short_entry_blocked"
    assert exec_params["short_entry_policy"] == ShortEntryPolicy.EXIT_ONLY.value


def test_short_entry_blocked_persists_hold_when_configured() -> None:
    strat = build_ml_trading_strategy_stub(execute_trades=True)
    strat.log = LoggerStub()
    strat._config.account_mode = AccountMode.CASH
    strat._config.short_entry_policy = ShortEntryPolicy.EXIT_ONLY
    strat._config.persist_hold_on_short_entry_block = True
    strat._config.persist_all_signals = False
    cast(Any, strat)._get_current_position = lambda: None

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
        del signal, decision_type, position_size, risk_metrics, execution_params
        captured["persist_hold"] = persist_hold

    cast(Any, strat)._persist_strategy_decision = _persist_strategy_decision
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    MLTradingStrategy._process_ml_signal(strat, _sig(0.1))

    assert captured.get("persist_hold") is True


def test_short_entry_allowed_when_margin_and_policy_unset() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(execute_trades=True, decision_recorder=recorder)
    strat.log = LoggerStub()
    strat._config.account_mode = AccountMode.MARGIN
    strat._config.short_entry_policy = None
    cast(Any, strat)._get_current_position = lambda: None
    entered: list[OrderSide] = []

    def _enter(side: OrderSide, _signal: MLSignal) -> None:
        entered.append(side)

    cast(Any, strat)._enter_position = _enter
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    MLTradingStrategy._process_ml_signal(strat, _sig(0.1))

    assert entered == [OrderSide.SELL]


def test_short_entry_exit_only_exits_long_without_reverse() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(execute_trades=True, decision_recorder=recorder)
    strat.log = LoggerStub()
    strat._config.account_mode = AccountMode.CASH
    strat._config.short_entry_policy = ShortEntryPolicy.DENY
    strat.clock = SimpleNamespace(timestamp_ns=lambda: 2_000_000_000)
    strat._config.serialize_order_intents = False

    class _Pos:
        def __init__(self) -> None:
            self.side = type("_Side", (), {"name": "LONG"})()
            self.quantity = Quantity.from_str("1.0")

    cast(Any, strat)._get_current_position = lambda: _Pos()

    def _should_reverse(current: object, target: object) -> bool:
        del current, target
        return True

    cast(Any, strat)._should_reverse_position = cast(Callable[[object, object], bool], _should_reverse)
    reversed_calls: list[str] = []

    def _reverse(_current: object, _side: object, _signal: MLSignal) -> None:
        reversed_calls.append("reverse")

    cast(Any, strat)._reverse_position = _reverse
    orders: list[tuple[OrderSide, Quantity, bool]] = []

    def _place_market_order(
        side: OrderSide,
        quantity: Quantity,
        reduce_only: bool = False,
    ) -> None:
        orders.append((side, quantity, reduce_only))

    cast(Any, strat)._place_market_order = _place_market_order
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    MLTradingStrategy._process_ml_signal(strat, _sig(0.1))

    assert not reversed_calls
    assert orders
    assert orders[-1][0] == OrderSide.SELL
    assert orders[-1][2] is True
    exec_params = recorder.records[-1].execution_params
    assert exec_params["action"] == "exit"
    assert exec_params["short_entry_policy"] == ShortEntryPolicy.DENY.value
