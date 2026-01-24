"""
Unit tests for order intent position tracking.
"""

from __future__ import annotations

from decimal import Decimal

from ml.strategies.common.intent_positions import OrderIntentPositionTracker
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
