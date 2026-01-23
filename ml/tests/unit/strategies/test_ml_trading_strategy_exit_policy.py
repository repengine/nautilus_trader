"""
Exit policy tests for MLTradingStrategy.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, cast

from ml.actors.base import MLSignal
from ml.config.base import ExitPolicyConfig
from ml.config.base import ModelExitConfig
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.tests.utils.stubs import LoggerStub
from ml.tests.utils.stubs import StrategyDecisionRecorder
from ml.tests.utils.stubs import build_ml_trading_strategy_stub
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Quantity


class _PositionStub:
    def __init__(
        self,
        side_name: str,
        *,
        entry_price: float,
        quantity: Quantity,
        ts_opened: int,
    ) -> None:
        self.side = SimpleNamespace(name=side_name)
        self.avg_px_open = entry_price
        self.quantity = quantity
        self.ts_opened = ts_opened


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


def test_exit_policy_stop_loss_when_price_breaches_places_reduce_only_exit() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(execute_trades=True, decision_recorder=recorder)
    strat.log = LoggerStub()
    strat.clock = SimpleNamespace(timestamp_ns=lambda: 1_100_000_000)
    strat._config.exit_policy_config = ExitPolicyConfig(
        stop_loss_pct=0.02,
        take_profit_pct=0.0,
        max_holding_ms=None,
    )
    strat._config.serialize_order_intents = False

    position = _PositionStub(
        "LONG",
        entry_price=100.0,
        quantity=Quantity.from_str("1.0"),
        ts_opened=1_000_000_000,
    )
    cast(Any, strat)._get_current_position = lambda: position
    cast(Any, strat)._resolve_market_price = lambda instrument_id: 97.0
    cast(Any, strat)._should_reverse_position = lambda current, target: False
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    orders: list[tuple[OrderSide, Quantity, bool]] = []

    def _place_market_order(
        side: OrderSide,
        quantity: Quantity,
        reduce_only: bool = False,
    ) -> None:
        orders.append((side, quantity, reduce_only))

    cast(Any, strat)._place_market_order = _place_market_order

    MLTradingStrategy._process_ml_signal(strat, _sig(0.9))

    assert orders
    assert orders[-1][0] == OrderSide.SELL
    assert orders[-1][2] is True
    assert recorder.records
    exec_params = recorder.records[-1].execution_params
    assert exec_params["action"] == "exit"
    exit_payload = cast(dict[str, object], exec_params["exit"])
    assert exit_payload["reason"] == "stop_loss"
    assert exit_payload["trigger_price"] == 97.0
    assert exit_payload["time_in_trade_ns"] == 100_000_000


def test_exit_policy_take_profit_when_target_reached_places_reduce_only_exit() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(execute_trades=True, decision_recorder=recorder)
    strat.log = LoggerStub()
    strat.clock = SimpleNamespace(timestamp_ns=lambda: 1_500_000_000)
    strat._config.exit_policy_config = ExitPolicyConfig(
        stop_loss_pct=0.0,
        take_profit_pct=0.04,
        max_holding_ms=None,
    )
    strat._config.serialize_order_intents = False

    position = _PositionStub(
        "LONG",
        entry_price=100.0,
        quantity=Quantity.from_str("2.0"),
        ts_opened=1_000_000_000,
    )
    cast(Any, strat)._get_current_position = lambda: position
    cast(Any, strat)._resolve_market_price = lambda instrument_id: 105.0
    cast(Any, strat)._should_reverse_position = lambda current, target: False
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    orders: list[tuple[OrderSide, Quantity, bool]] = []

    def _place_market_order(
        side: OrderSide,
        quantity: Quantity,
        reduce_only: bool = False,
    ) -> None:
        orders.append((side, quantity, reduce_only))

    cast(Any, strat)._place_market_order = _place_market_order

    MLTradingStrategy._process_ml_signal(strat, _sig(0.9))

    assert orders
    assert orders[-1][0] == OrderSide.SELL
    assert orders[-1][2] is True
    exec_params = recorder.records[-1].execution_params
    exit_payload = cast(dict[str, object], exec_params["exit"])
    assert exit_payload["reason"] == "take_profit"
    assert exit_payload["trigger_price"] == 105.0
    assert exit_payload["time_in_trade_ns"] == 500_000_000


def test_exit_policy_timeout_when_time_exceeded_places_reduce_only_exit() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(execute_trades=True, decision_recorder=recorder)
    strat.log = LoggerStub()
    strat.clock = SimpleNamespace(timestamp_ns=lambda: 1_020_000_000)
    strat._config.exit_policy_config = ExitPolicyConfig(
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        max_holding_ms=10,
    )
    strat._config.serialize_order_intents = False

    position = _PositionStub(
        "SHORT",
        entry_price=100.0,
        quantity=Quantity.from_str("1.0"),
        ts_opened=1_000_000_000,
    )
    cast(Any, strat)._get_current_position = lambda: position
    cast(Any, strat)._resolve_market_price = lambda instrument_id: 100.5
    cast(Any, strat)._should_reverse_position = lambda current, target: False
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    orders: list[tuple[OrderSide, Quantity, bool]] = []

    def _place_market_order(
        side: OrderSide,
        quantity: Quantity,
        reduce_only: bool = False,
    ) -> None:
        orders.append((side, quantity, reduce_only))

    cast(Any, strat)._place_market_order = _place_market_order

    MLTradingStrategy._process_ml_signal(strat, _sig(0.1))

    assert orders
    assert orders[-1][0] == OrderSide.BUY
    assert orders[-1][2] is True
    exec_params = recorder.records[-1].execution_params
    exit_payload = cast(dict[str, object], exec_params["exit"])
    assert exit_payload["reason"] == "timeout"
    assert exit_payload["trigger_price"] == 100.5
    assert exit_payload["time_in_trade_ns"] == 20_000_000


def test_exit_policy_reverse_when_signal_opposes_persists_exit_metadata() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(execute_trades=False, decision_recorder=recorder)
    strat.log = LoggerStub()
    strat.clock = SimpleNamespace(timestamp_ns=lambda: 1_100_000_000)
    strat._config.exit_policy_config = ExitPolicyConfig(
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        max_holding_ms=None,
    )
    strat._config.serialize_order_intents = False

    position = _PositionStub(
        "LONG",
        entry_price=100.0,
        quantity=Quantity.from_str("1.0"),
        ts_opened=1_000_000_000,
    )
    cast(Any, strat)._get_current_position = lambda: position
    cast(Any, strat)._resolve_market_price = lambda instrument_id: 101.0

    def _should_reverse(current: object, target: object) -> bool:
        del current, target
        return True

    cast(Any, strat)._should_reverse_position = cast(Callable[[object, object], bool], _should_reverse)
    cast(Any, strat)._reverse_position = lambda current, side, signal: None
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    MLTradingStrategy._process_ml_signal(strat, _sig(0.0))

    assert recorder.records
    exec_params = recorder.records[-1].execution_params
    assert exec_params["action"] == "reverse"
    exit_payload = cast(dict[str, object], exec_params["exit"])
    assert exit_payload["reason"] == "reverse"
    assert exit_payload["trigger_price"] == 101.0
    assert exit_payload["time_in_trade_ns"] == 100_000_000


def test_exit_policy_precedes_model_exit_when_both_trigger() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(execute_trades=True, decision_recorder=recorder)
    strat.log = LoggerStub()
    strat.clock = SimpleNamespace(timestamp_ns=lambda: 1_100_000_000)
    strat._config.exit_policy_config = ExitPolicyConfig(
        stop_loss_pct=0.02,
        take_profit_pct=0.0,
        max_holding_ms=None,
    )
    strat._config.model_exit_config = ModelExitConfig(exit_on_flip=True, reverse_on_flip=False)
    strat._config.serialize_order_intents = False

    position = _PositionStub(
        "LONG",
        entry_price=100.0,
        quantity=Quantity.from_str("1.0"),
        ts_opened=1_000_000_000,
    )
    cast(Any, strat)._get_current_position = lambda: position
    cast(Any, strat)._resolve_market_price = lambda instrument_id: 97.0
    cast(Any, strat)._should_reverse_position = lambda current, target: True
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    orders: list[tuple[OrderSide, Quantity, bool]] = []

    def _place_market_order(
        side: OrderSide,
        quantity: Quantity,
        reduce_only: bool = False,
    ) -> None:
        orders.append((side, quantity, reduce_only))

    cast(Any, strat)._place_market_order = _place_market_order

    MLTradingStrategy._process_ml_signal(strat, _sig(0.1))

    assert orders
    exec_params = recorder.records[-1].execution_params
    assert exec_params["action"] == "exit"
    exit_payload = cast(dict[str, object], exec_params["exit"])
    assert exit_payload["reason"] == "stop_loss"


def test_model_exit_precedes_reversal_when_configured() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(execute_trades=True, decision_recorder=recorder)
    strat.log = LoggerStub()
    strat.clock = SimpleNamespace(timestamp_ns=lambda: 1_100_000_000)
    strat._config.exit_policy_config = ExitPolicyConfig(
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        max_holding_ms=None,
    )
    strat._config.model_exit_config = ModelExitConfig(exit_on_flip=True, reverse_on_flip=False)
    strat._config.serialize_order_intents = False

    position = _PositionStub(
        "LONG",
        entry_price=100.0,
        quantity=Quantity.from_str("1.0"),
        ts_opened=1_000_000_000,
    )
    cast(Any, strat)._get_current_position = lambda: position
    cast(Any, strat)._resolve_market_price = lambda instrument_id: 101.0
    cast(Any, strat)._should_reverse_position = lambda current, target: True
    cast(Any, strat).target_side_from_prediction = (
        MLTradingStrategy.target_side_from_prediction.__get__(strat, MLTradingStrategy)
    )

    orders: list[tuple[OrderSide, Quantity, bool]] = []

    def _place_market_order(
        side: OrderSide,
        quantity: Quantity,
        reduce_only: bool = False,
    ) -> None:
        orders.append((side, quantity, reduce_only))

    cast(Any, strat)._place_market_order = _place_market_order

    MLTradingStrategy._process_ml_signal(strat, _sig(0.1))

    assert orders
    exec_params = recorder.records[-1].execution_params
    assert exec_params["action"] == "exit"
    exit_payload = cast(dict[str, object], exec_params["exit"])
    assert exit_payload["reason"] == "model_flip"
