from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ml.strategies.risk import RiskConfig, RiskManager
from ml.strategies.sizing import CompositeSizer, SizingConfig


# ==========================
# Helpers / dummies
# ==========================


@dataclass
class _DummyQuantity:
    _val: float

    @classmethod
    def from_str(cls, s: str) -> _DummyQuantity:  # type: ignore[override]
        return cls(float(s))

    def as_double(self) -> float:  # noqa: D401
        return self._val


@dataclass
class _DummyBalance:
    val: float

    def as_double(self) -> float:  # noqa: D401
        return self.val


@dataclass
class _DummyAccount:
    bal: float

    def balance_total(self) -> _DummyBalance:  # noqa: D401
        return _DummyBalance(self.bal)


@dataclass
class _DummyPosition:
    instrument_id: str
    quantity: _DummyQuantity
    is_open: bool = True


@dataclass
class _DummyPortfolio:
    venue: str = "SIM"
    _positions: list[_DummyPosition] = None  # type: ignore[assignment]
    _account_balance: float = 10000.0

    def __post_init__(self) -> None:
        if self._positions is None:
            self._positions = []

    def positions(self) -> list[_DummyPosition]:  # noqa: D401
        return list(self._positions)

    def account(self, _venue: Any) -> Any:  # noqa: D401
        return _DummyAccount(self._account_balance)


@dataclass
class _DummySignal:
    confidence: float
    instrument_id: str = "TEST/SIM"
    prediction: float = 1.0


# Monkeypatch Quantities to strategy code path for unit tests
import ml.strategies.sizing as _sizing_mod  # noqa: E402


def _patch_quantity(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Not required when Nautilus Quantity is available; placeholder kept for clarity.
    return None


# ==========================
# Tests
# ==========================


def test_composite_sizer_bounds_and_monotonic(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _patch_quantity(monkeypatch)

    cfg = SizingConfig(
        kelly_fraction=0.25,
        target_volatility=0.10,
        max_position_pct=0.15,
        min_position_pct=0.01,
        confidence_scaling=True,
        performance_scaling=False,
        lookback_periods=20,
    )
    sizer = CompositeSizer(cfg)

    # Fill volatility buffer to stabilize output
    for _ in range(cfg.lookback_periods):
        sizer.update_market_data(0.001)

    account = _DummyAccount(bal=10000.0)

    # Lower confidence signal
    sig_low = _DummySignal(confidence=0.5)
    q_low = sizer.calculate(sig_low, account, current_positions=[])
    assert q_low is not None
    val_low = float(q_low.as_double())

    # Higher confidence signal should not size smaller
    sig_high = _DummySignal(confidence=0.8)
    q_high = sizer.calculate(sig_high, account, current_positions=[])
    assert q_high is not None
    val_high = float(q_high.as_double())

    assert val_high >= val_low

    # Bounds check: value is between min and max pct of balance
    min_val = account.balance_total().as_double() * cfg.min_position_pct
    max_val = account.balance_total().as_double() * cfg.max_position_pct
    assert min_val <= val_high <= max_val


def test_risk_manager_daily_circuit_breaker_and_correlation(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Configure stricter limits to trigger checks easily
    rcfg = RiskConfig(
        daily_loss_limit_pct=0.06,
        max_position_pct=0.15,
        max_total_exposure=1.0,
        stop_loss_pct=0.02,
        max_loss_per_trade_pct=0.02,
        correlation_threshold=0.7,
        max_correlated_positions=1,
    )
    rm = RiskManager(rcfg)

    portfolio = _DummyPortfolio(_account_balance=10000.0)

    # Seed equity to avoid zero division
    rm._current_equity = 10000.0  # type: ignore[attr-defined]

    # Exceed daily loss limit
    rm.update_daily_pnl(-800.0)  # 8% loss
    assert rm.check_daily_limits() is False

    # Proposed size should be rejected when trading halted
    from nautilus_trader.model.identifiers import InstrumentId  # type: ignore
    inst = InstrumentId.from_str("TEST.SIM")
    from nautilus_trader.model.objects import Quantity  # type: ignore
    proposed = Quantity.from_str("1000")
    assert rm.check_position(proposed, inst, portfolio) is None

    # Correlation limit enforcement: one open position correlated > threshold
    rm._trading_halted = False  # type: ignore[attr-defined]
    portfolio._positions.append(_DummyPosition(inst, _DummyQuantity(1000.0), is_open=True))
    # Inject correlation > threshold for same inst (self treated as 1.0 by guard)
    # Add another instrument correlated to inst
    inst2 = InstrumentId.from_str("TEST2.SIM")
    portfolio._positions.append(_DummyPosition(inst2, _DummyQuantity(500.0), is_open=True))
    # Patch RiskManager internal correlation lookup to return high correlation
    rm._position_correlations[(inst, inst2)] = 0.9  # type: ignore[index]
    rm._position_correlations[(inst2, inst)] = 0.9  # type: ignore[index]

    # Proposed new trade on inst should be rejected due to correlation limit
    assert rm._check_correlation_limits(inst, portfolio) is False
