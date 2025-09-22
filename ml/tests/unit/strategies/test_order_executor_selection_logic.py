from __future__ import annotations

from ml.actors.base import MLSignal
from ml.strategies.execution import ExecutionConfig, OrderExecutor
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity


class _Inst:
    def __init__(self, iid: InstrumentId) -> None:
        self.id = iid
        self.venue = iid.venue
        self.size_precision = 6
        self.price_precision = 5


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
