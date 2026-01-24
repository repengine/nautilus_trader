from __future__ import annotations

from ml.actors.base import MLSignal
from ml.config.base import ExecutionValidationMode
from ml.config.base import LimitPriceConfig, LimitPriceSource
from ml.strategies.execution import ExecutionConfig, OrderExecutor
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity


class _Inst:
    def __init__(self, iid: InstrumentId) -> None:
        self.id = iid
        self.venue = iid.venue
        self.size_precision = 6
        self.price_precision = 5

    def make_price(self, value: float) -> Price:
        rounded = round(float(value), int(self.price_precision))
        return Price(rounded, int(self.price_precision))


def _mk_instrument(sym: str = "EUR/USD.SIM") -> _Inst:
    return _Inst(InstrumentId.from_str(sym))


def test_executor_prefers_aggressive_limit_when_high_confidence_and_tight_spread() -> None:
    cfg = ExecutionConfig(
        market_order_threshold=0.9,
        limit_order_threshold=0.7,
        prefer_maker_orders=True,
        prefer_maker_spread_bps=5,
        use_time_in_force_ioc=True,
    )
    ex = OrderExecutor(cfg)
    sig = MLSignal(
        instrument_id=_mk_instrument().id,
        model_id="M",
        prediction=0.95,
        confidence=0.95,
        ts_event=1,
        ts_init=1,
    )
    order = ex.create_order(
        side=OrderSide.BUY,
        quantity=Quantity.from_str("10"),
        signal=sig,
        market_state={"bid": 1.0, "ask": 1.0001, "spread_bps": 1.0},
        instrument=_mk_instrument(),
    )
    # High confidence would choose market, but tight spread + maker preference downgrades to aggressive limit
    assert order is not None
    assert getattr(order, "price", None) is not None
    assert order.time_in_force in (TimeInForce.IOC, TimeInForce.GTC)
    # TTL plan should be present (IOC -> attempts 0)
    plan = ex.get_last_ttl_plan()
    assert plan is not None
    assert plan["order_type"] == "limit_aggressive"
    assert plan["attempts"] == 0


def test_executor_prefers_passive_limit_and_records_ttl_plan() -> None:
    cfg = ExecutionConfig(
        limit_order_threshold=0.8,
        min_confidence=0.5,
        prefer_maker_orders=True,
        prefer_maker_spread_bps=10,
        use_time_in_force_ioc=False,
        ttl_max_attempts=4,
        ttl_replace_cadence_seconds=3,
    )
    ex = OrderExecutor(cfg)
    sig = MLSignal(
        instrument_id=_mk_instrument().id,
        model_id="M",
        prediction=0.7,
        confidence=0.7,
        ts_event=2,
        ts_init=2,
    )
    order = ex.create_order(
        side=OrderSide.SELL,
        quantity=Quantity.from_str("5"),
        signal=sig,
        market_state={"bid": 2.0, "ask": 2.0004, "spread_bps": 2.0},
        instrument=_mk_instrument(),
    )
    assert order is not None
    assert getattr(order, "price", None) is not None
    assert order.time_in_force == TimeInForce.GTC
    assert getattr(order, "post_only", False)
    plan = ex.get_last_ttl_plan()
    assert plan is not None
    assert plan["order_type"] == "limit_passive"
    assert plan["attempts"] == 4
    assert plan["cadence_seconds"] == 3.0


def test_executor_uses_passive_limit_when_spread_wide_and_confidence_medium() -> None:
    cfg = ExecutionConfig(
        market_order_threshold=0.9,
        limit_order_threshold=0.7,
        max_spread_bps=20,
        prefer_maker_orders=True,
        use_time_in_force_ioc=False,
    )
    ex = OrderExecutor(cfg)
    sig = MLSignal(
        instrument_id=_mk_instrument().id,
        model_id="M",
        prediction=0.8,
        confidence=0.8,
        ts_event=3,
        ts_init=3,
    )
    order = ex.create_order(
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1"),
        signal=sig,
        market_state={"bid": 1.0, "ask": 1.01, "spread_bps": 100.0},
        instrument=_mk_instrument(),
    )
    assert order is not None
    assert getattr(order, "price", None) is not None
    assert order.time_in_force == TimeInForce.GTC
    assert getattr(order, "post_only", False) is True


def test_executor_validation_mode_market_forces_market_orders() -> None:
    cfg = ExecutionConfig(
        min_confidence=0.1,
        validation_mode=ExecutionValidationMode.MARKET,
    )
    ex = OrderExecutor(cfg)
    sig = MLSignal(
        instrument_id=_mk_instrument().id,
        model_id="M",
        prediction=0.9,
        confidence=0.9,
        ts_event=10,
        ts_init=10,
    )
    order = ex.create_order(
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1"),
        signal=sig,
        market_state={"bid": 1.0, "ask": 1.01, "spread_bps": 100.0},
        instrument=_mk_instrument(),
    )
    assert order is not None
    assert order.order_type == OrderType.MARKET


def test_executor_validation_mode_cross_bbo_crosses_spread() -> None:
    cfg = ExecutionConfig(
        market_order_threshold=0.95,
        limit_order_threshold=0.5,
        min_confidence=0.1,
        aggressive_offset_bps=2,
        validation_mode=ExecutionValidationMode.CROSS_BBO,
    )
    ex = OrderExecutor(cfg)
    sig = MLSignal(
        instrument_id=_mk_instrument().id,
        model_id="M",
        prediction=0.6,
        confidence=0.6,
        ts_event=11,
        ts_init=11,
    )
    market_state = {"bid": 100.0, "ask": 100.1, "spread_bps": 10.0}
    buy_order = ex.create_order(
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1"),
        signal=sig,
        market_state=market_state,
        instrument=_mk_instrument(),
    )
    assert buy_order is not None
    assert float(buy_order.price.as_double()) >= market_state["ask"]

    sell_order = ex.create_order(
        side=OrderSide.SELL,
        quantity=Quantity.from_str("1"),
        signal=sig,
        market_state=market_state,
        instrument=_mk_instrument(),
    )
    assert sell_order is not None
    assert float(sell_order.price.as_double()) <= market_state["bid"]


def test_executor_validation_mode_disabled_does_not_cross_bbo() -> None:
    cfg = ExecutionConfig(
        market_order_threshold=0.95,
        limit_order_threshold=0.5,
        min_confidence=0.1,
        aggressive_offset_bps=2,
        validation_mode=ExecutionValidationMode.DISABLED,
    )
    ex = OrderExecutor(cfg)
    sig = MLSignal(
        instrument_id=_mk_instrument().id,
        model_id="M",
        prediction=0.6,
        confidence=0.6,
        ts_event=12,
        ts_init=12,
    )
    market_state = {"bid": 100.0, "ask": 100.1, "spread_bps": 10.0}
    buy_order = ex.create_order(
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1"),
        signal=sig,
        market_state=market_state,
        instrument=_mk_instrument(),
    )
    assert buy_order is not None
    assert float(buy_order.price.as_double()) < market_state["ask"]

    sell_order = ex.create_order(
        side=OrderSide.SELL,
        quantity=Quantity.from_str("1"),
        signal=sig,
        market_state=market_state,
        instrument=_mk_instrument(),
    )
    assert sell_order is not None
    assert float(sell_order.price.as_double()) > market_state["bid"]


def test_executor_uses_last_trade_fallback_when_quotes_missing() -> None:
    cfg = ExecutionConfig(
        limit_order_threshold=0.7,
        min_confidence=0.5,
        market_order_threshold=0.95,
        use_time_in_force_ioc=False,
        limit_price_config=LimitPriceConfig(
            source_priority=[
                LimitPriceSource.LAST_TRADE,
                LimitPriceSource.CACHE_LAST,
            ],
        ),
    )
    ex = OrderExecutor(cfg)
    sig = MLSignal(
        instrument_id=_mk_instrument().id,
        model_id="M",
        prediction=0.6,
        confidence=0.8,
        ts_event=4,
        ts_init=4,
    )
    order = ex.create_order(
        side=OrderSide.BUY,
        quantity=Quantity.from_str("2"),
        signal=sig,
        market_state={
            "bid": 0.0,
            "ask": 0.0,
            "spread_bps": 0.0,
            "last_trade": 100.0,
            "cache_last": 0.0,
        },
        instrument=_mk_instrument(),
    )
    assert order is not None
    expected = round(100.0 * (1 - (cfg.passive_offset_bps / 10_000)), 5)
    assert float(order.price.as_double()) == expected
    assert ex.get_last_limit_price_source() == "last_trade"


def test_executor_uses_cached_price_when_trade_missing() -> None:
    cfg = ExecutionConfig(
        limit_order_threshold=0.7,
        min_confidence=0.5,
        market_order_threshold=0.95,
        use_time_in_force_ioc=False,
        limit_price_config=LimitPriceConfig(
            source_priority=[
                LimitPriceSource.LAST_TRADE,
                LimitPriceSource.CACHE_LAST,
            ],
        ),
    )
    ex = OrderExecutor(cfg)
    sig = MLSignal(
        instrument_id=_mk_instrument().id,
        model_id="M",
        prediction=0.6,
        confidence=0.8,
        ts_event=5,
        ts_init=5,
    )
    order = ex.create_order(
        side=OrderSide.SELL,
        quantity=Quantity.from_str("2"),
        signal=sig,
        market_state={
            "bid": 0.0,
            "ask": 0.0,
            "spread_bps": 0.0,
            "last_trade": 0.0,
            "cache_last": 50.0,
        },
        instrument=_mk_instrument(),
    )
    assert order is not None
    expected = round(50.0 * (1 + (cfg.passive_offset_bps / 10_000)), 5)
    assert float(order.price.as_double()) == expected
    assert ex.get_last_limit_price_source() == "cache_last"


def test_executor_sets_limit_price_precision_when_trailing_zeros() -> None:
    cfg = ExecutionConfig(
        market_order_threshold=0.9,
        limit_order_threshold=0.5,
        min_confidence=0.1,
        aggressive_offset_bps=0,
        passive_offset_bps=0,
        use_time_in_force_ioc=True,
    )
    ex = OrderExecutor(cfg)
    inst = _mk_instrument()
    inst.price_precision = 6
    sig = MLSignal(
        instrument_id=inst.id,
        model_id="M",
        prediction=0.6,
        confidence=0.6,
        ts_event=6,
        ts_init=6,
    )
    order = ex.create_order(
        side=OrderSide.BUY,
        quantity=Quantity.from_str("2"),
        signal=sig,
        market_state={"bid": 1.23450, "ask": 1.23456, "spread_bps": 1.0},
        instrument=inst,
    )
    assert order is not None
    assert order.price is not None
    assert order.price.precision == inst.price_precision
