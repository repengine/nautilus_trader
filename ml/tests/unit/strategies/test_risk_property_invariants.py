"""
Property tests for risk management invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import pytest

from ml.strategies.risk import RiskConfig
from ml.strategies.risk import RiskManager
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity


@dataclass
class _DummyQuantity:
    value: float

    def as_double(self) -> float:
        return self.value


@dataclass
class _DummyPosition:
    instrument_id: InstrumentId
    quantity: _DummyQuantity
    is_open: bool = True


class _DummyPortfolio:
    def __init__(self, positions: list[_DummyPosition]) -> None:
        self._positions = positions

    def positions(self) -> list[_DummyPosition]:
        return list(self._positions)


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


@settings(max_examples=50)
@given(
    balance=st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    loss_pct=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
)
def test_daily_loss_halts_when_over_limit_property(balance: float, loss_pct: float) -> None:
    limit = 0.05
    assume(loss_pct >= limit)
    config = RiskConfig(daily_loss_limit_pct=limit)
    manager = RiskManager(config)
    manager._current_equity = balance
    manager.update_daily_pnl(-balance * loss_pct)

    assert manager.is_trading_halted() is True


@settings(max_examples=50)
@given(
    peak=st.floats(min_value=1_000.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
    drawdown_pct=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False),
    size_value=st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
)
def test_drawdown_adjustment_never_increases_position_property(
    peak: float,
    drawdown_pct: float,
    size_value: float,
) -> None:
    config = RiskConfig(drawdown_reduction_factor=0.5)
    manager = RiskManager(config)
    manager._peak_equity = peak
    manager._current_equity = peak * (1 - drawdown_pct)
    size = Quantity.from_str(str(size_value))

    adjusted = manager._apply_drawdown_adjustment(size)
    adjusted_value = float(adjusted.as_double())

    assert adjusted_value <= size_value or math.isclose(
        adjusted_value,
        size_value,
        rel_tol=1e-12,
        abs_tol=1e-9,
    )
    assert adjusted_value >= size_value * 0.3


@settings(max_examples=50)
@given(
    balance=st.floats(min_value=1_000.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
    limit=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    existing=st.floats(min_value=0.0, max_value=200_000.0, allow_nan=False, allow_infinity=False),
    new_value=st.floats(min_value=0.0, max_value=200_000.0, allow_nan=False, allow_infinity=False),
)
def test_exposure_rejects_when_over_limit_property(
    balance: float,
    limit: float,
    existing: float,
    new_value: float,
) -> None:
    assume(balance > 0)
    assume(existing + new_value > balance * limit)
    config = RiskConfig(max_total_exposure=limit, max_position_pct=1.0)
    manager = RiskManager(config)
    instrument = InstrumentId.from_str("AAA.SIM")
    positions = [_DummyPosition(instrument_id=instrument, quantity=_DummyQuantity(existing))]
    portfolio = _DummyPortfolio(positions)

    assert manager._check_portfolio_exposure(new_value, balance, portfolio) is False
