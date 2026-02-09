"""
Additional branch coverage tests for MLTradingStrategy.

Covers reversal and hold branches using a dummy self instance and stubs for
methods/attributes.

"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, cast

from ml.actors.base import MLSignal
from ml.config.base import ExitHorizonConfig
from ml.config.base import ModelExitConfig
from ml.config.base import ShortEntryPolicy
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.strategies.ml_strategy import MultiModelMLStrategy
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


def test_short_entry_blocked_persists_hold_decision() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(decision_recorder=recorder)
    strat.log = LoggerStub()
    cast(Any, strat)._config.short_entry_policy = ShortEntryPolicy.DENY

    MLTradingStrategy._process_ml_signal(strat, _sig(0.0))

    record = recorder.records[-1]
    assert record.decision_type == "HOLD"
    assert record.execution_params["reason"] == "short_entry_blocked"
    assert record.execution_params["short_entry_policy"] == ShortEntryPolicy.DENY.value


def test_resolve_model_exit_config_applies_horizon_min_hold_clamp() -> None:
    strat = build_ml_trading_strategy_stub()
    cast(Any, strat)._config.exit_horizon_config = ExitHorizonConfig(
        enabled=True,
        min_hold_multiplier=2.0,
        min_hold_min_ms=50,
        min_hold_max_ms=400,
        apply_to_model_exit=True,
    )
    cast(Any, strat)._config.model_exit_config = ModelExitConfig(
        exit_on_flip=True,
        reverse_on_flip=False,
        exit_prediction_band=0.0,
        min_hold_ms=None,
    )

    resolved = MLTradingStrategy._resolve_model_exit_config(strat, horizon_ms=500)

    assert resolved is not None
    assert resolved.min_hold_ms == 400


def test_timestamp_ns_falls_back_when_clock_raises() -> None:
    class _BrokenClock:
        def timestamp_ns(self) -> int:
            raise RuntimeError("boom")

    strat = build_ml_trading_strategy_stub()
    strat.log = LoggerStub()
    cast(Any, strat).clock = _BrokenClock()

    resolved = MLTradingStrategy._timestamp_ns(strat)

    assert resolved > 0
    assert any(
        level == "debug" and args and args[0] == "ml_strategy.clock_timestamp_failed"
        for level, args, _kwargs in strat.log.records
    )


def test_derive_horizon_max_holding_returns_none_when_disabled() -> None:
    strat = build_ml_trading_strategy_stub()
    cast(Any, strat)._config.exit_horizon_config = SimpleNamespace(
        enabled=False,
        apply_to_exit_policy=True,
        max_holding_multiplier=3.0,
    )

    derived = MLTradingStrategy._derive_horizon_max_holding_ms(strat, 500)

    assert derived is None


def test_derive_horizon_max_holding_returns_none_when_multiplier_non_positive() -> None:
    strat = build_ml_trading_strategy_stub()
    cast(Any, strat)._config.exit_horizon_config = SimpleNamespace(
        enabled=True,
        apply_to_exit_policy=True,
        max_holding_multiplier=0.0,
    )

    derived = MLTradingStrategy._derive_horizon_max_holding_ms(strat, 500)

    assert derived is None


def test_derive_horizon_min_hold_clamps_to_minimum() -> None:
    strat = build_ml_trading_strategy_stub()
    cast(Any, strat)._config.exit_horizon_config = ExitHorizonConfig(
        enabled=True,
        min_hold_multiplier=0.01,
        min_hold_min_ms=50,
        min_hold_max_ms=500,
        apply_to_model_exit=True,
    )

    derived = MLTradingStrategy._derive_horizon_min_hold_ms(strat, 500)

    assert derived == 50


def test_resolve_model_exit_config_keeps_existing_min_hold() -> None:
    strat = build_ml_trading_strategy_stub()
    existing = ModelExitConfig(
        exit_on_flip=True,
        reverse_on_flip=False,
        min_hold_ms=1234,
    )
    cast(Any, strat)._config.model_exit_config = existing

    resolved = MLTradingStrategy._resolve_model_exit_config(strat, horizon_ms=60_000)

    assert resolved is existing


def test_position_entry_price_skips_invalid_values_and_selects_first_positive() -> None:
    strat = build_ml_trading_strategy_stub()
    position = SimpleNamespace(
        avg_px_open="not-a-float",
        avg_px=0.0,
        avg_price=101.25,
        entry_price=99.0,
    )

    entry = MLTradingStrategy._position_entry_price(strat, position)

    assert entry == 101.25


def test_exit_side_for_position_maps_long_short_and_unknown() -> None:
    strat = build_ml_trading_strategy_stub()
    long_position = SimpleNamespace(side=SimpleNamespace(name="LONG"))
    short_position = SimpleNamespace(side=SimpleNamespace(name="SHORT"))
    flat_position = SimpleNamespace(side=SimpleNamespace(name="FLAT"))

    assert MLTradingStrategy._exit_side_for_position(strat, long_position) is OrderSide.SELL
    assert MLTradingStrategy._exit_side_for_position(strat, short_position) is OrderSide.BUY
    assert MLTradingStrategy._exit_side_for_position(strat, flat_position) is None


def test_update_returns_from_signal_respects_toggle_and_logs_failures() -> None:
    strat = build_ml_trading_strategy_stub()
    strat.log = LoggerStub()

    class _Updater:
        def __init__(self) -> None:
            self.calls: int = 0

        def should_update_from_signal(self) -> bool:
            return False

        def update_from_signal(self, signal: MLSignal, *, cache: object, reference_ts: int) -> None:
            del signal, cache, reference_ts
            self.calls += 1

    updater = _Updater()
    cast(Any, strat)._returns_updater = updater
    cast(Any, strat).cache = object()
    MLTradingStrategy._update_returns_from_signal(strat, _sig(0.6))
    assert updater.calls == 0

    class _FailingUpdater(_Updater):
        def should_update_from_signal(self) -> bool:
            return True

        def update_from_signal(self, signal: MLSignal, *, cache: object, reference_ts: int) -> None:
            del signal, cache, reference_ts
            raise RuntimeError("boom")

    cast(Any, strat)._returns_updater = _FailingUpdater()
    MLTradingStrategy._update_returns_from_signal(strat, _sig(0.6))
    assert any(
        level == "debug" and args and args[0] == "ml_strategy.returns_update_failed"
        for level, args, _kwargs in strat.log.records
    )


def test_enter_position_short_policy_and_dry_run_paths() -> None:
    strat = build_ml_trading_strategy_stub(execute_trades=False)
    strat.log = LoggerStub()
    cast(Any, strat)._resolve_short_entry_policy = lambda: ShortEntryPolicy.DENY
    cast(Any, strat)._should_block_entry_orders = lambda: False
    cast(Any, strat).size_and_validate = lambda _signal: Quantity.from_str("2.0")

    # SELL should be blocked by short-entry policy before sizing.
    MLTradingStrategy._enter_position(strat, OrderSide.SELL, _sig(0.1))
    assert strat._active_positions == 0

    # BUY should follow dry-run path and increment active position count.
    cast(Any, strat)._resolve_short_entry_policy = lambda: ShortEntryPolicy.ALLOW
    MLTradingStrategy._enter_position(strat, OrderSide.BUY, _sig(0.9))
    assert strat._active_positions == 1


def test_should_reverse_position_falls_back_when_hook_raises() -> None:
    strat = build_ml_trading_strategy_stub()
    strat.log = LoggerStub()
    cast(Any, strat).should_reverse = lambda _current, _target: (_ for _ in ()).throw(
        RuntimeError("hook failed"),
    )

    long_position = SimpleNamespace(side=SimpleNamespace(name="LONG"))
    short_position = SimpleNamespace(side=SimpleNamespace(name="SHORT"))
    assert MLTradingStrategy._should_reverse_position(strat, long_position, OrderSide.SELL) is True
    assert MLTradingStrategy._should_reverse_position(strat, short_position, OrderSide.BUY) is True
    assert any(
        level == "debug" and args and "heuristic" in str(args[0])
        for level, args, _kwargs in strat.log.records
    )


def test_hold_without_position_persists_neutral_band_decision() -> None:
    recorder = StrategyDecisionRecorder()
    strat = build_ml_trading_strategy_stub(decision_recorder=recorder)
    strat.log = LoggerStub()
    cast(Any, strat)._active_positions = 0
    cast(Any, strat)._get_current_position = lambda: None
    cast(Any, strat).decision_from_prediction = lambda _prediction: "HOLD"
    cast(Any, strat).target_side_from_prediction = lambda _prediction, _threshold=0.5: OrderSide.BUY

    MLTradingStrategy._process_ml_signal(strat, _sig(0.5))

    assert recorder.records
    record = recorder.records[-1]
    assert record.decision_type == "HOLD"
    assert record.execution_params["action"] == "hold"
    assert record.execution_params["reason"] == "neutral_band"


def test_enter_position_blocks_when_intent_limits_disallow_entry() -> None:
    strat = build_ml_trading_strategy_stub(execute_trades=True)
    strat.log = LoggerStub()
    cast(Any, strat)._resolve_short_entry_policy = lambda: ShortEntryPolicy.ALLOW
    cast(Any, strat)._should_block_entry_orders = lambda: True

    MLTradingStrategy._enter_position(strat, OrderSide.BUY, _sig(0.9))

    assert strat._active_positions == 0
    assert any(
        level == "info" and args and "Intent entry suppressed" in str(args[0])
        for level, args, _kwargs in strat.log.records
    )


def test_enter_position_handles_market_fallback_and_submission_failure() -> None:
    strat = build_ml_trading_strategy_stub(execute_trades=True)
    errors: list[tuple[tuple[object, ...], dict[str, object]]] = []
    strat.log = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: errors.append((args, kwargs)),
    )
    cast(Any, strat)._resolve_short_entry_policy = lambda: ShortEntryPolicy.ALLOW
    cast(Any, strat)._should_block_entry_orders = lambda: False
    cast(Any, strat).size_and_validate = lambda _signal: Quantity.from_str("1.0")

    MLTradingStrategy._enter_position(strat, OrderSide.BUY, _sig(0.9))
    assert errors
    assert strat._active_positions == 0

    cast(Any, strat)._place_market_order = lambda _side, _quantity: "OID-1"
    MLTradingStrategy._enter_position(strat, OrderSide.BUY, _sig(0.9))
    assert strat._active_positions == 1


def test_reverse_position_covers_dry_run_and_guard_paths() -> None:
    current_position = SimpleNamespace(
        side=SimpleNamespace(name="LONG"),
        quantity=Quantity.from_str("1.0"),
    )
    strat = build_ml_trading_strategy_stub(execute_trades=False)
    strat.log = LoggerStub()
    cast(Any, strat)._calculate_position_size = lambda: Quantity.from_str("2.0")
    cast(Any, strat)._place_market_order = lambda _side, _qty, reduce_only=False: reduce_only

    MLTradingStrategy._reverse_position(strat, current_position, OrderSide.SELL, _sig(0.1))
    assert any(
        level == "info" and args and "Would close" in str(args[0])
        for level, args, _kwargs in strat.log.records
    )

    strat._config.execute_trades = True
    cast(Any, strat)._should_block_entry_orders = lambda: True
    MLTradingStrategy._reverse_position(strat, current_position, OrderSide.SELL, _sig(0.1))


def test_reverse_position_handles_sizing_and_submission_failures() -> None:
    current_position = SimpleNamespace(
        side=SimpleNamespace(name="LONG"),
        quantity=Quantity.from_str("1.0"),
    )
    warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []
    errors: list[tuple[tuple[object, ...], dict[str, object]]] = []
    strat = build_ml_trading_strategy_stub(execute_trades=True)
    strat.log = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: warnings.append((args, kwargs)),
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: errors.append((args, kwargs)),
    )
    cast(Any, strat)._should_block_entry_orders = lambda: False
    cast(Any, strat)._place_market_order = lambda _side, _qty, reduce_only=False: None

    cast(Any, strat).size_and_validate = lambda _signal: None
    MLTradingStrategy._reverse_position(strat, current_position, OrderSide.SELL, _sig(0.1))
    assert warnings

    cast(Any, strat).size_and_validate = lambda _signal: Quantity.from_str("2.0")
    MLTradingStrategy._reverse_position(strat, current_position, OrderSide.SELL, _sig(0.1))
    assert errors


def test_reverse_position_tracks_trade_entry_when_enabled() -> None:
    current_position = SimpleNamespace(
        side=SimpleNamespace(name="LONG"),
        quantity=Quantity.from_str("1.0"),
    )
    tracked: list[tuple[str, object, object]] = []
    strat = build_ml_trading_strategy_stub(execute_trades=True)
    strat.log = LoggerStub()
    strat.track_performance = True
    cast(Any, strat)._should_block_entry_orders = lambda: False
    cast(Any, strat).size_and_validate = lambda _signal: Quantity.from_str("2.0")
    cast(Any, strat)._place_market_order = lambda _side, _qty, reduce_only=False: reduce_only
    cast(Any, strat)._submit_smart_order = lambda _side, _qty, _signal: "OID-2"
    cast(Any, strat)._track_trade_entry = (
        lambda model_id, signal, order_id: tracked.append((model_id, signal, order_id))
    )

    signal = _sig(0.1)
    MLTradingStrategy._reverse_position(strat, current_position, OrderSide.SELL, signal)

    assert tracked
    assert tracked[-1][0] == "m1"
    assert tracked[-1][2] == "OID-2"


def test_on_order_filled_tracks_pnl_and_cleans_mapping(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "ml.strategies.base.BaseMLStrategy.on_order_filled",
        lambda self, event: None,
    )
    monkeypatch.setattr(MLTradingStrategy, "log", LoggerStub(), raising=False)
    strategy = cast(MLTradingStrategy, MLTradingStrategy.__new__(MLTradingStrategy))
    strategy.track_performance = True
    updates: list[tuple[str, float]] = []
    cast(Any, strategy)._update_model_performance = (
        lambda model_id, pnl: updates.append((model_id, pnl))
    )
    cast(Any, strategy)._order_to_model = {
        "oid-sell": {"model_id": "m-sell"},
        "oid-buy": {"model_id": "m-buy"},
    }

    sell_event = SimpleNamespace(
        client_order_id="oid-sell",
        order_side=SimpleNamespace(name="SELL"),
        last_px=SimpleNamespace(as_double=lambda: 105.0),
        avg_px=SimpleNamespace(as_double=lambda: 100.0),
    )
    buy_event = SimpleNamespace(
        client_order_id="oid-buy",
        order_side=SimpleNamespace(name="BUY"),
        last_px=SimpleNamespace(as_double=lambda: 95.0),
        avg_px=SimpleNamespace(as_double=lambda: 100.0),
    )

    MLTradingStrategy.on_order_filled(strategy, sell_event)
    MLTradingStrategy.on_order_filled(strategy, buy_event)

    assert updates == [("m-sell", 5.0), ("m-buy", 5.0)]
    assert strategy._order_to_model == {}


def test_multimodel_init_and_dynamic_weight_calculation(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        MLTradingStrategy,
        "__init__",
        lambda self, config, stores=None: setattr(self, "_model_performance", {}),
    )
    config = SimpleNamespace(use_dynamic_weights=True)

    strategy = MultiModelMLStrategy(config)
    assert strategy.track_performance is True
    assert strategy.use_dynamic_weights is True
    assert strategy._get_dynamic_model_weights() == {}

    strategy._model_performance = {
        "model-a": {"accuracy": 0.8, "total_profit": 120.0, "total_trades": 3},
        "model-b": {"accuracy": 0.6, "total_profit": -20.0, "total_trades": 2},
    }
    weights = strategy._get_dynamic_model_weights()

    assert set(weights) == {"model-a", "model-b"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_multimodel_aggregate_updates_weights_before_parent_call(monkeypatch: Any) -> None:
    forwarded: list[MLSignal] = []
    monkeypatch.setattr(
        MLTradingStrategy,
        "_aggregate_signal",
        lambda self, signal: forwarded.append(signal),
    )
    strategy = cast(MultiModelMLStrategy, MultiModelMLStrategy.__new__(MultiModelMLStrategy))
    strategy.use_dynamic_weights = True
    strategy.model_weights = {}
    cast(Any, strategy)._get_dynamic_model_weights = lambda: {"model-a": 0.75, "model-b": 0.25}
    signal = _sig(0.9)

    MultiModelMLStrategy._aggregate_signal(strategy, signal)

    assert strategy.model_weights == {"model-a": 0.75, "model-b": 0.25}
    assert forwarded == [signal]
