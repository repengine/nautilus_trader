"""
Tests for staged risk actions in RiskManager.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ml.strategies.risk import RiskAction
from ml.strategies.risk import RiskConfig
from ml.strategies.risk import RiskLiquidationConfig
from ml.strategies.risk import RiskManager
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


def test_get_risk_action_liquidates_on_daily_loss_threshold() -> None:
    """Ensure daily loss liquidation triggers when configured."""
    config = RiskConfig(
        liquidation_config=RiskLiquidationConfig(
            enabled=True,
            daily_loss_limit_pct=0.1,
        ),
    )
    manager = RiskManager(config)
    manager._current_equity = 1000.0
    manager._peak_equity = 1000.0
    manager._daily_pnl = -200.0

    decision = manager.get_risk_action()

    assert decision.action is RiskAction.LIQUIDATE
    assert decision.reason == "daily_loss_liquidate"
    assert manager.is_trading_halted() is True


def test_get_risk_action_liquidates_on_drawdown_threshold() -> None:
    """Ensure drawdown liquidation triggers when configured."""
    config = RiskConfig(
        liquidation_config=RiskLiquidationConfig(
            enabled=True,
            drawdown_limit_pct=0.2,
        ),
    )
    manager = RiskManager(config)
    manager._peak_equity = 1000.0
    manager._current_equity = 700.0
    manager._daily_pnl = 0.0

    decision = manager.get_risk_action()

    assert decision.action is RiskAction.LIQUIDATE
    assert decision.reason == "drawdown_liquidate"
    assert manager.is_trading_halted() is True


def test_get_risk_action_halts_on_cooldown() -> None:
    """Ensure liquidation cooldown halts repeat actions."""
    config = RiskConfig(
        liquidation_config=RiskLiquidationConfig(
            enabled=True,
            daily_loss_limit_pct=0.1,
            cooldown_ms=60_000,
        ),
    )
    manager = RiskManager(config)
    manager._current_equity = 1000.0
    manager._peak_equity = 1000.0
    manager._daily_pnl = -200.0

    decision = manager.get_risk_action(ts_event=1_000_000_000)

    assert decision.action is RiskAction.LIQUIDATE

    cooldown_decision = manager.get_risk_action(ts_event=1_000_000_000 + 1_000_000)

    assert cooldown_decision.action is RiskAction.HALT
    assert cooldown_decision.reason == "liquidation_cooldown"


def test_get_risk_action_returns_halt_when_already_halted() -> None:
    manager = RiskManager()
    manager._trading_halted = True
    manager._halt_reason = "manual_halt"
    manager._halt_detail = "manual"

    decision = manager.get_risk_action()

    assert decision.action is RiskAction.HALT
    assert decision.reason == "manual_halt"
    assert decision.detail == "manual"


@dataclass
class _Price:
    value: float

    def as_double(self) -> float:
        return self.value


@dataclass
class _Quote:
    bid_price: _Price
    ask_price: _Price


@dataclass
class _Position:
    instrument_id: InstrumentId
    quantity: Quantity
    avg_px_open: float
    is_open: bool = True
    multiplier: float = 1.0
    side: object = type("_LongSide", (), {"name": "LONG"})()


class _Account:
    class _Balance:
        def as_double(self) -> float:
            return 1000.0

    def balance_total(self) -> _Balance:
        return self._Balance()


class _Portfolio:
    def __init__(self, positions: list[_Position]) -> None:
        self._positions = positions

    def positions(self) -> list[_Position]:
        return list(self._positions)

    def account(self, _venue: object) -> _Account:
        return _Account()


class _PriceProvider:
    def __init__(self, quote: _Quote) -> None:
        self._quote = quote

    def quote_tick(self, instrument_id: InstrumentId) -> _Quote:
        del instrument_id
        return self._quote


def test_get_risk_action_liquidates_on_unrealized_loss_threshold() -> None:
    config = RiskConfig(
        liquidation_config=RiskLiquidationConfig(
            enabled=True,
            unrealized_loss_limit_pct=0.01,
        ),
    )
    manager = RiskManager(
        config,
        market_price_provider=_PriceProvider(
            _Quote(
                bid_price=_Price(89.0),
                ask_price=_Price(91.0),
            ),
        ),
    )
    manager._current_equity = 1000.0
    manager._peak_equity = 1000.0
    position = _Position(
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        quantity=Quantity.from_str("2"),
        avg_px_open=100.0,
    )

    decision = manager.get_risk_action(portfolio=_Portfolio([position]))

    assert decision.action is RiskAction.LIQUIDATE
    assert decision.reason == "unrealized_loss_liquidate"
    assert manager.is_trading_halted() is True


def test_risk_config_rejects_disabled_liquidation_thresholds() -> None:
    with pytest.raises(ValueError, match=r"liquidation_config\.enabled must be True"):
        RiskConfig(
            liquidation_config=RiskLiquidationConfig(
                enabled=False,
                daily_loss_limit_pct=0.1,
            ),
        )
