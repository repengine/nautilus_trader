from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import time
from typing import Any

from ml.actors.base import MLSignal
from ml.config.base import PositionsSource
from ml.strategies.common.correlation import CorrelationSnapshot
from ml.strategies.common.positions import PositionsHealthStatus
from ml.strategies.common.positions import PositionsSnapshot
from ml.strategies.risk import RiskConfig, RiskManager
from ml.strategies.sizing import CompositeSizer, SizingConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from pytest import MonkeyPatch


# ==========================
# Helpers / dummies
# ==========================


@dataclass
class _DummyQuantity:
    _val: float

    @classmethod
    def from_str(cls, s: str) -> _DummyQuantity:
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
    _positions: list[_DummyPosition] = field(default_factory=list)
    _account_balance: float = 10000.0

    def positions(self) -> list[_DummyPosition]:  # noqa: D401
        return list(self._positions)

    def account(self, _venue: Any) -> Any:  # noqa: D401
        return _DummyAccount(self._account_balance)


@dataclass(frozen=True)
class _DummyPositionView:
    instrument_id: InstrumentId
    is_open: bool = True

    def signed_decimal_qty(self) -> Decimal:
        return Decimal("1.0")


class _DummyPositionsProvider:
    def __init__(self, positions: list[_DummyPositionView]) -> None:
        self._snapshot = PositionsSnapshot(
            positions=positions,
            source=PositionsSource.CACHE_OPEN,
        )

    def get_positions_snapshot(
        self,
        *,
        instrument_id: InstrumentId | None = None,
        require_full_list: bool = False,
    ) -> PositionsSnapshot:
        del instrument_id, require_full_list
        return self._snapshot

    def check_positions_ready(
        self,
        *,
        instrument_id: InstrumentId | None = None,
        require_full_list: bool = False,
        require_positions: bool = False,
    ) -> PositionsHealthStatus:
        del instrument_id, require_full_list, require_positions
        return PositionsHealthStatus(
            ready=True,
            degraded=False,
            source=self._snapshot.source,
            reason=None,
            positions_count=len(self._snapshot.positions),
        )


class _DummyCorrelationProvider:
    def __init__(self, snapshot: CorrelationSnapshot | None) -> None:
        self._snapshot = snapshot

    def get_correlation_snapshot(
        self,
        inst1: InstrumentId,
        inst2: InstrumentId,
    ) -> CorrelationSnapshot | None:
        del inst1, inst2
        return self._snapshot


# Monkeypatch Quantities to strategy code path for unit tests
def _patch_quantity(monkeypatch: MonkeyPatch) -> None:
    del monkeypatch


# ==========================
# Tests
# ==========================


def test_composite_sizer_bounds_and_monotonic(monkeypatch: MonkeyPatch) -> None:
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
    sig_low = MLSignal(
        instrument_id=InstrumentId.from_str("TEST.SIM"),
        model_id="model",
        prediction=1.0,
        confidence=0.5,
        ts_event=1,
        ts_init=1,
    )
    q_low = sizer.calculate(sig_low, account, current_positions=[])
    assert q_low is not None
    val_low = float(q_low.as_double())

    # Higher confidence signal should not size smaller
    sig_high = MLSignal(
        instrument_id=InstrumentId.from_str("TEST.SIM"),
        model_id="model",
        prediction=1.0,
        confidence=0.8,
        ts_event=2,
        ts_init=2,
    )
    q_high = sizer.calculate(sig_high, account, current_positions=[])
    assert q_high is not None
    val_high = float(q_high.as_double())

    assert val_high >= val_low

    # Bounds check: value is between min and max pct of balance
    min_val = account.balance_total().as_double() * cfg.min_position_pct
    max_val = account.balance_total().as_double() * cfg.max_position_pct
    assert min_val <= val_high <= max_val


def test_risk_manager_daily_circuit_breaker_and_correlation(monkeypatch: MonkeyPatch) -> None:
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
    setattr(rm, "_current_equity", 10000.0)

    # Exceed daily loss limit
    rm.update_daily_pnl(-800.0)  # 8% loss
    assert rm.check_daily_limits() is False

    # Proposed size should be rejected when trading halted
    inst = InstrumentId.from_str("TEST.SIM")
    proposed = Quantity.from_str("1000")
    assert rm.check_position(proposed, inst, portfolio) is None

    # Correlation limit enforcement: one open position correlated > threshold
    setattr(rm, "_trading_halted", False)
    portfolio._positions.append(_DummyPosition(inst, _DummyQuantity(1000.0), is_open=True))
    # Inject correlation > threshold for same inst (self treated as 1.0 by guard)
    # Add another instrument correlated to inst
    inst2 = InstrumentId.from_str("TEST2.SIM")
    portfolio._positions.append(_DummyPosition(inst2, _DummyQuantity(500.0), is_open=True))
    rm.set_correlation_provider(
        _DummyCorrelationProvider(
            CorrelationSnapshot(value=0.9, ts_event=time.time_ns(), source="test"),
        ),
    )

    # Proposed new trade on inst should be rejected due to correlation limit
    assert rm._check_correlation_limits(inst, portfolio) is False


def test_risk_manager_uses_positions_provider_snapshot() -> None:
    inst = InstrumentId.from_str("AAA.SIM")
    inst2 = InstrumentId.from_str("BBB.SIM")
    provider = _DummyPositionsProvider([_DummyPositionView(inst2)])
    rm = RiskManager(
        RiskConfig(correlation_threshold=0.0, max_correlated_positions=1),
        positions_provider=provider,
    )
    rm.set_correlation_provider(
        _DummyCorrelationProvider(
            CorrelationSnapshot(value=0.1, ts_event=time.time_ns(), source="test"),
        ),
    )
    portfolio = _DummyPortfolio()

    assert rm._check_correlation_limits(inst, portfolio) is False
