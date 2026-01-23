"""
Tests for price-aware exposure calculations in RiskManager.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from ml.config.base import ExposurePriceConfig
from ml.config.base import ExposurePriceSource
from ml.strategies.risk import RiskConfig
from ml.strategies.risk import RiskManager
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


@dataclass
class DummyPosition:
    instrument_id: InstrumentId
    quantity: Quantity | None = None
    avg_px_open: float | None = None
    multiplier: float | None = None
    is_open: bool = True

    def signed_decimal_qty(self) -> Decimal:
        if self.quantity is None:
            return Decimal("0")
        return Decimal(str(self.quantity.as_double()))


@dataclass
class DummyPrice:
    value: float

    def as_double(self) -> float:
        return self.value


@dataclass
class DummyQuote:
    bid_price: DummyPrice
    ask_price: DummyPrice


class DummyPriceProvider:
    def __init__(
        self,
        *,
        quote_tick: DummyQuote | None = None,
        last_price: DummyPrice | None = None,
    ) -> None:
        self._quote_tick = quote_tick
        self._last_price = last_price

    def quote_tick(self, instrument_id: InstrumentId) -> DummyQuote | None:
        return self._quote_tick

    def price(self, instrument_id: InstrumentId, price_type: Any) -> DummyPrice | None:
        return self._last_price


def test_resolve_position_value_prefers_quote_mid_when_available(
    isolated_prometheus_registry: Any,
) -> None:
    position = DummyPosition(
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        quantity=Quantity.from_str("2"),
        avg_px_open=100.0,
        multiplier=2.0,
    )
    provider = DummyPriceProvider(
        quote_tick=DummyQuote(
            bid_price=DummyPrice(101.0),
            ask_price=DummyPrice(103.0),
        ),
    )
    rm = RiskManager(
        RiskConfig(
            exposure_price_config=ExposurePriceConfig(
                source_priority=[
                    ExposurePriceSource.QUOTE_MID,
                    ExposurePriceSource.POSITION_AVG,
                ],
            ),
        ),
        market_price_provider=provider,
    )

    value = rm._resolve_position_value(position)

    assert value == pytest.approx(408.0)
    degraded = isolated_prometheus_registry.registry.get_sample_value(
        "ml_positions_exposure_degraded_total",
        labels={"reason": "price_missing"},
    )
    assert degraded is None


def test_resolve_position_value_falls_back_to_position_avg_when_quote_missing(
    isolated_prometheus_registry: Any,
) -> None:
    position = DummyPosition(
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        quantity=Quantity.from_str("2"),
        avg_px_open=99.0,
        multiplier=2.0,
    )
    provider = DummyPriceProvider(quote_tick=None)
    rm = RiskManager(
        RiskConfig(
            exposure_price_config=ExposurePriceConfig(
                source_priority=[
                    ExposurePriceSource.QUOTE_MID,
                    ExposurePriceSource.POSITION_AVG,
                ],
            ),
        ),
        market_price_provider=provider,
    )

    value = rm._resolve_position_value(position)

    assert value == pytest.approx(396.0)
    degraded = isolated_prometheus_registry.registry.get_sample_value(
        "ml_positions_exposure_degraded_total",
        labels={"reason": "price_missing"},
    )
    assert degraded is None


def test_resolve_position_value_falls_back_to_cache_last_when_position_price_missing(
    isolated_prometheus_registry: Any,
) -> None:
    position = DummyPosition(
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        quantity=Quantity.from_str("3"),
        multiplier=1.0,
    )
    provider = DummyPriceProvider(last_price=DummyPrice(110.0))
    rm = RiskManager(
        RiskConfig(
            exposure_price_config=ExposurePriceConfig(
                source_priority=[
                    ExposurePriceSource.QUOTE_MID,
                    ExposurePriceSource.POSITION_AVG,
                    ExposurePriceSource.CACHE_LAST,
                ],
            ),
        ),
        market_price_provider=provider,
    )

    value = rm._resolve_position_value(position)

    assert value == pytest.approx(330.0)
    degraded = isolated_prometheus_registry.registry.get_sample_value(
        "ml_positions_exposure_degraded_total",
        labels={"reason": "price_missing"},
    )
    assert degraded is None


def test_resolve_position_value_uses_notional_when_price_available(
    isolated_prometheus_registry: Any,
) -> None:
    rm = RiskManager()
    position = DummyPosition(
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        quantity=Quantity.from_str("2"),
        avg_px_open=100.0,
        multiplier=2.0,
    )

    value = rm._resolve_position_value(position)

    assert value == pytest.approx(400.0)
    degraded = isolated_prometheus_registry.registry.get_sample_value(
        "ml_positions_exposure_degraded_total",
        labels={"reason": "price_missing"},
    )
    assert degraded is None


def test_resolve_position_value_emits_degraded_metric_when_price_missing(
    isolated_prometheus_registry: Any,
) -> None:
    rm = RiskManager()
    position = DummyPosition(
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        quantity=Quantity.from_str("3"),
    )

    value = rm._resolve_position_value(position)

    assert value == pytest.approx(3.0)
    degraded = isolated_prometheus_registry.registry.get_sample_value(
        "ml_positions_exposure_degraded_total",
        labels={"reason": "price_missing"},
    )
    assert degraded == 1.0
