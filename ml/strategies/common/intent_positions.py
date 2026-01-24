"""
Synthetic positions tracker for order intent serialization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from decimal import Decimal
from typing import Any

from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity

from ml.strategies.common.decision_persistence import _SafeLogger
from ml.strategies.common.positions import PositionViewProtocol
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import PositionSide


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IntentPosition(PositionViewProtocol):
    """
    Lightweight synthetic position derived from order intents.
    """

    instrument_id: InstrumentId
    net_qty: Decimal
    ts_opened: int | None = None
    entry_price: float | None = None
    is_open: bool = field(init=False)

    def __post_init__(self) -> None:
        """
        Populate the derived open/closed flag.
        """
        object.__setattr__(self, "is_open", self.net_qty != Decimal("0"))

    @property
    def side(self) -> PositionSide:
        """
        Resolve position side from the signed quantity.
        """
        if self.net_qty > 0:
            return PositionSide.LONG
        if self.net_qty < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    @property
    def quantity(self) -> Quantity:
        """
        Resolve absolute position quantity for strategy logic.
        """
        return Quantity.from_str(str(abs(self.net_qty)))

    def signed_decimal_qty(self) -> Decimal:
        """
        Return signed quantity for sizing and risk calculations.
        """
        return self.net_qty


class OrderIntentPositionTracker:
    """
    Track synthetic positions when order intents are serialized.
    """

    def __init__(self, *, log: Any | None = None) -> None:
        self._positions: dict[InstrumentId, IntentPosition] = {}
        self._log = _SafeLogger(log if log is not None else logger)

    @property
    def active_positions(self) -> int:
        """
        Count open synthetic positions.
        """
        return sum(1 for position in self._positions.values() if position.is_open)

    def get_position(self, instrument_id: InstrumentId) -> IntentPosition | None:
        """
        Return the synthetic position for an instrument if open.
        """
        position = self._positions.get(instrument_id)
        if position is None or not position.is_open:
            return None
        return position

    def record_order(
        self,
        *,
        instrument_id: InstrumentId,
        side: OrderSide,
        quantity: Quantity,
        reduce_only: bool,
        ts_init: int | None = None,
        entry_price: float | None = None,
    ) -> None:
        """
        Update synthetic positions based on an order intent.
        """
        if instrument_id is None:
            return
        if side not in {OrderSide.BUY, OrderSide.SELL}:
            return
        if quantity is None:
            return

        try:
            qty = quantity.as_decimal()
        except Exception as exc:
            self._log.debug(
                "ml_strategy.intent_position_quantity_unavailable",
                exc_info=True,
                error=str(exc),
            )
            return
        if qty <= 0:
            return

        signed_delta = qty if side is OrderSide.BUY else -qty
        if reduce_only:
            self._apply_reduce_only(
                instrument_id=instrument_id,
                signed_delta=signed_delta,
            )
            return
        self._apply_entry(
            instrument_id=instrument_id,
            signed_delta=signed_delta,
            ts_init=ts_init,
            entry_price=entry_price,
        )

    def _apply_entry(
        self,
        *,
        instrument_id: InstrumentId,
        signed_delta: Decimal,
        ts_init: int | None,
        entry_price: float | None,
    ) -> None:
        current = self._positions.get(instrument_id)
        current_qty = current.net_qty if current is not None else Decimal("0")
        new_qty = current_qty + signed_delta
        if new_qty == Decimal("0"):
            self._positions.pop(instrument_id, None)
            return

        ts_opened = current.ts_opened if current is not None else None
        if current is None or current_qty == Decimal("0"):
            ts_opened = ts_init
        else:
            if current_qty > 0 and new_qty < 0:
                ts_opened = ts_init
            elif current_qty < 0 and new_qty > 0:
                ts_opened = ts_init

        resolved_entry = current.entry_price if current is not None else None
        if resolved_entry is None:
            resolved_entry = entry_price

        self._positions[instrument_id] = IntentPosition(
            instrument_id=instrument_id,
            net_qty=new_qty,
            ts_opened=ts_opened,
            entry_price=resolved_entry,
        )

    def _apply_reduce_only(
        self,
        *,
        instrument_id: InstrumentId,
        signed_delta: Decimal,
    ) -> None:
        current = self._positions.get(instrument_id)
        if current is None or current.net_qty == Decimal("0"):
            self._log.debug(
                "ml_strategy.intent_reduce_only_no_position",
                instrument_id=str(instrument_id),
            )
            return

        current_qty = current.net_qty
        if current_qty > 0 and signed_delta > 0:
            self._log.debug(
                "ml_strategy.intent_reduce_only_increases_long",
                instrument_id=str(instrument_id),
            )
            return
        if current_qty < 0 and signed_delta < 0:
            self._log.debug(
                "ml_strategy.intent_reduce_only_increases_short",
                instrument_id=str(instrument_id),
            )
            return

        new_qty = current_qty + signed_delta
        if current_qty > 0:
            new_qty = max(Decimal("0"), new_qty)
        else:
            new_qty = min(Decimal("0"), new_qty)

        if new_qty == Decimal("0"):
            self._positions.pop(instrument_id, None)
            return

        self._positions[instrument_id] = IntentPosition(
            instrument_id=instrument_id,
            net_qty=new_qty,
            ts_opened=current.ts_opened,
            entry_price=current.entry_price,
        )


__all__ = [
    "IntentPosition",
    "OrderIntentPositionTracker",
]
