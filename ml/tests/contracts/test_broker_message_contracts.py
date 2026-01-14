#!/usr/bin/env python3
"""
Contract tests for broker-facing order intent behavior.

These tests validate that order submission produces schema-correct intents, preserves
idempotent client order IDs, and exposes retry plans for resting orders.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ml.actors.base import MLSignal
from ml.strategies.common.order_submission import OrderSubmissionComponent
from ml.strategies.execution import ExecutionConfig, OrderExecutor
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId, StrategyId, TraderId
from nautilus_trader.model.objects import Quantity


pytestmark = [
    pytest.mark.contracts,
    pytest.mark.parallel_safe,
]


@dataclass(slots=True)
class _FixedClock:
    """
    Deterministic clock for timestamp assertions.
    """

    ts_init: int

    def timestamp_ns(self) -> int:
        """
        Return a fixed timestamp.
        """
        return self.ts_init


class _Cache:
    """
    Minimal cache stub for order submission paths.
    """

    def __init__(self) -> None:
        """
        Initialize cache stub.
        """
        self._counter = 0

    def client_order_id(self) -> ClientOrderId:
        """
        Return a deterministic client order ID.
        """
        self._counter += 1
        return ClientOrderId(f"O-{self._counter:06d}")

    def quote_tick(self, instrument_id: Any) -> None:
        """
        Return the latest quote tick.
        """
        del instrument_id
        return None


@dataclass(slots=True)
class _Instrument:
    """
    Minimal instrument stub for order executor.
    """

    id: InstrumentId
    price_precision: int = 5
    size_precision: int = 6

    @property
    def venue(self) -> Any:
        """
        Expose venue attribute expected by Nautilus orders.
        """
        return self.id.venue


def test_order_intent_schema_and_idempotent_id() -> None:
    """
    Market order intents must include required fields and stable client IDs.
    """
    submitted: list[Any] = []
    clock = _FixedClock(ts_init=123456789)
    component = OrderSubmissionComponent(
        strategy_id="STRAT-TEST",
        cache=_Cache(),
        submit_order_callback=submitted.append,
        trader_id="TRADER-TEST",
        clock=clock,
    )

    instrument_id = InstrumentId.from_str("EUR/USD.SIM")
    order_id = component.place_market_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
    )

    assert order_id is not None
    assert len(submitted) == 1
    order = submitted[0]

    assert str(order.client_order_id) == str(order_id)
    assert order.instrument_id == instrument_id
    assert str(order.strategy_id) == "STRAT-TEST"
    assert str(order.trader_id) == "TRADER-TEST"
    side_value = getattr(order, "order_side", getattr(order, "side", None))
    assert side_value == OrderSide.BUY
    assert order.time_in_force == TimeInForce.GTC
    assert float(order.quantity.as_double()) == 1.0
    assert getattr(order, "ts_init", None) == clock.timestamp_ns()


def test_order_intent_retry_plan_for_resting_limit() -> None:
    """
    Smart orders should expose a retry plan when using resting limits.
    """
    config = ExecutionConfig(
        market_order_threshold=0.95,
        limit_order_threshold=0.8,
        min_confidence=0.5,
        prefer_maker_orders=True,
        use_time_in_force_ioc=False,
        ttl_max_attempts=3,
        ttl_replace_cadence_seconds=4,
    )
    executor = OrderExecutor(config)
    instrument = _Instrument(id=InstrumentId.from_str("EUR/USD.SIM"))
    signal = MLSignal(
        instrument_id=instrument.id,
        model_id="MODEL-TEST",
        prediction=0.7,
        confidence=0.7,
        ts_event=1,
        ts_init=1,
    )
    order = executor.create_order(
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
        signal=signal,
        market_state={"bid": 100.0, "ask": 100.1, "spread_bps": 10.0},
        instrument=instrument,
        trader_id=TraderId("TRADER-TEST"),
        strategy_id=StrategyId("STRAT-TEST"),
        client_order_id=ClientOrderId("O-000001"),
    )

    assert order is not None
    plan = executor.get_last_ttl_plan()
    assert plan is not None
    assert plan["order_type"] == "limit_passive"
    assert plan["attempts"] == config.ttl_max_attempts
    assert plan["cadence_seconds"] == float(config.ttl_replace_cadence_seconds)
