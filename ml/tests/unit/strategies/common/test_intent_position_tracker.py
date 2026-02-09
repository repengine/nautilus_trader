"""
Unit tests for order intent position tracking.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from typing import cast

from ml.strategies.common.intent_positions import IntentPosition
from ml.strategies.common.intent_positions import OrderIntentPositionTracker
from ml.tests.utils.stubs import LoggerStub
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity


def test_intent_position_tracker_tracks_entry_and_exit() -> None:
    tracker = OrderIntentPositionTracker()
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("2.0"),
        reduce_only=False,
        ts_init=10,
    )

    position = tracker.get_position(instrument_id)
    assert position is not None
    assert position.side.name == "LONG"
    assert position.quantity.as_decimal() == Decimal("2.0")
    assert tracker.active_positions == 1

    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.SELL,
        quantity=Quantity.from_str("1.0"),
        reduce_only=True,
        ts_init=20,
    )

    position = tracker.get_position(instrument_id)
    assert position is not None
    assert position.quantity.as_decimal() == Decimal("1.0")
    assert tracker.active_positions == 1

    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.SELL,
        quantity=Quantity.from_str("2.0"),
        reduce_only=True,
        ts_init=30,
    )

    assert tracker.get_position(instrument_id) is None
    assert tracker.active_positions == 0


def test_intent_position_tracker_ignores_reduce_only_increase() -> None:
    tracker = OrderIntentPositionTracker()
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
        reduce_only=False,
        ts_init=10,
    )

    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
        reduce_only=True,
        ts_init=20,
    )

    position = tracker.get_position(instrument_id)
    assert position is not None
    assert position.quantity.as_decimal() == Decimal("1.0")
    assert tracker.active_positions == 1


def test_intent_position_tracker_tracks_short_positions() -> None:
    tracker = OrderIntentPositionTracker()
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.SELL,
        quantity=Quantity.from_str("3.0"),
        reduce_only=False,
        ts_init=10,
    )

    position = tracker.get_position(instrument_id)
    assert position is not None
    assert position.side.name == "SHORT"
    assert position.quantity.as_decimal() == Decimal("3.0")


def test_intent_position_signed_quantity_and_flat_side() -> None:
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")
    position = IntentPosition(instrument_id=instrument_id, net_qty=Decimal("0"))

    assert position.side.name == "FLAT"
    assert position.quantity.as_decimal() == Decimal("0")
    assert position.signed_decimal_qty() == Decimal("0")
    assert position.is_open is False


def test_intent_position_tracker_ignores_invalid_orders() -> None:
    tracker = OrderIntentPositionTracker(log=LoggerStub())
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    class _BrokenQuantity:
        def as_decimal(self) -> Decimal:
            raise RuntimeError("boom")

    class _ZeroQuantity:
        def as_decimal(self) -> Decimal:
            return Decimal("0")

    tracker.record_order(
        instrument_id=cast(Any, None),
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
        reduce_only=False,
    )
    tracker.record_order(
        instrument_id=instrument_id,
        side=cast(Any, "UNSUPPORTED"),
        quantity=Quantity.from_str("1.0"),
        reduce_only=False,
    )
    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=cast(Any, None),
        reduce_only=False,
    )
    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=cast(Any, _BrokenQuantity()),
        reduce_only=False,
    )
    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=cast(Any, _ZeroQuantity()),
        reduce_only=False,
    )

    assert tracker.active_positions == 0
    assert tracker.get_position(instrument_id) is None


def test_intent_position_tracker_resets_open_timestamp_on_side_flip() -> None:
    tracker = OrderIntentPositionTracker()
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
        reduce_only=False,
        ts_init=10,
        entry_price=1.11,
    )
    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
        reduce_only=False,
        ts_init=20,
        entry_price=1.22,
    )
    position = tracker.get_position(instrument_id)
    assert position is not None
    assert position.ts_opened == 10
    assert position.entry_price == 1.11

    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.SELL,
        quantity=Quantity.from_str("3.0"),
        reduce_only=False,
        ts_init=30,
        entry_price=2.22,
    )
    position = tracker.get_position(instrument_id)
    assert position is not None
    assert position.side.name == "SHORT"
    assert position.ts_opened == 30
    assert position.entry_price == 1.11

    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
        reduce_only=False,
        ts_init=40,
    )
    assert tracker.get_position(instrument_id) is None


def test_intent_position_tracker_reduce_only_short_paths() -> None:
    log = LoggerStub()
    tracker = OrderIntentPositionTracker(log=log)
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("1.0"),
        reduce_only=True,
        ts_init=1,
    )

    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.SELL,
        quantity=Quantity.from_str("3.0"),
        reduce_only=False,
        ts_init=2,
    )
    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.SELL,
        quantity=Quantity.from_str("1.0"),
        reduce_only=True,
        ts_init=3,
    )
    tracker.record_order(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.from_str("5.0"),
        reduce_only=True,
        ts_init=4,
    )

    assert tracker.get_position(instrument_id) is None
    assert any(
        level == "debug" and args and args[0] == "ml_strategy.intent_reduce_only_no_position"
        for level, args, _kwargs in log.records
    )
    assert any(
        level == "debug" and args and args[0] == "ml_strategy.intent_reduce_only_increases_short"
        for level, args, _kwargs in log.records
    )
