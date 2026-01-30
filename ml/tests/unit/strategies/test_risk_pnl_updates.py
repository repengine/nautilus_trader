from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ml.strategies.risk import RiskConfig
from ml.strategies.risk import RiskManager
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


def test_update_daily_pnl_resets_using_event_timestamp() -> None:
    rm = RiskManager()
    future_reset = datetime.now() + timedelta(hours=1)
    rm._daily_reset_time = future_reset
    rm._daily_pnl = 10.0
    rm._trades_today = 2
    rm._losses_today = 1

    event_time = future_reset + timedelta(minutes=1)
    ts_event = int(event_time.timestamp() * 1_000_000_000)

    rm.update_daily_pnl(-5.0, ts_event=ts_event)

    assert rm._daily_pnl == -5.0
    assert rm._trades_today == 1
    assert rm._losses_today == 1


def test_check_position_initializes_equity_from_balance() -> None:
    class DummyBalance:
        def __init__(self, value: float) -> None:
            self._value = value

        def as_double(self) -> float:
            return self._value

    class DummyAccount:
        def __init__(self, balance: float) -> None:
            self._balance = balance

        def balance_total(self) -> DummyBalance:
            return DummyBalance(self._balance)

    class DummyPortfolio:
        def __init__(self, balance: float) -> None:
            self._balance = balance

        def account(self, _venue: object) -> DummyAccount:
            return DummyAccount(self._balance)

        def positions(self) -> list[object]:
            return []

    rm = RiskManager(
        RiskConfig(
            max_position_pct=1.0,
            max_loss_per_trade_pct=1.0,
            daily_loss_limit_pct=0.5,
            max_drawdown_pct=0.5,
        ),
    )
    portfolio = DummyPortfolio(100_000.0)
    instrument = InstrumentId.from_str("AAA.SIM")
    proposed = Quantity.from_str("1000")

    approved = rm.check_position(proposed, instrument, portfolio)

    assert approved is not None
    assert rm._equity_initialized is True
    assert rm._current_equity == 100_000.0
    assert rm._peak_equity == 100_000.0

    rm.update_daily_pnl(-1_000.0)

    assert rm._current_drawdown_pct() == pytest.approx(0.01)
    assert rm.is_trading_halted() is False
