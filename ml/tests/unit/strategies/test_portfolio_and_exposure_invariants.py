from __future__ import annotations

from dataclasses import dataclass

from ml.strategies.portfolio import PortfolioConfig, PortfolioManager
from ml.strategies.risk import RiskConfig, RiskManager


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
class _DummyPosition:
    instrument_id: object
    quantity: _DummyQuantity
    is_open: bool = True


@dataclass
class _DummyAccount:
    bal: float

    def balance_total(self):  # type: ignore[no-untyped-def]
        class _B:
            def __init__(self, v: float) -> None:
                self._v = v

            def as_double(self) -> float:  # noqa: D401
                return self._v

        return _B(self.bal)


@dataclass
class _DummyPortfolio:
    _positions: list[_DummyPosition]
    _balance: float = 10000.0

    def positions(self) -> list[_DummyPosition]:  # noqa: D401
        return list(self._positions)

    def account(self, _venue):  # type: ignore[no-untyped-def]
        return _DummyAccount(self._balance)


@dataclass
class _Signal:
    instrument_id: object
    confidence: float


def test_exposure_limit_rejects_when_exceeding_max() -> None:
    # Setup risk manager with 50% max exposure
    rcfg = RiskConfig(max_total_exposure=0.5, max_position_pct=1.0)
    rm = RiskManager(rcfg)

    from nautilus_trader.model.identifiers import InstrumentId

    inst = InstrumentId.from_str("AAA.SIM")
    # Existing open position worth 4000
    pos = _DummyPosition(inst, _DummyQuantity(4000.0), is_open=True)
    portfolio = _DummyPortfolio([pos], _balance=10000.0)

    # Propose new position 2000 -> total exposure 6000/10000 = 60% > 50%
    from nautilus_trader.model.objects import Quantity

    proposed = Quantity.from_str("2000")
    assert rm._check_portfolio_exposure(2000.0, 10000.0, portfolio) is False  # type: ignore[attr-defined]

    # Via public check_position path should reject
    result = rm.check_position(proposed, inst, portfolio)
    assert result is None


def test_portfolio_correlation_adjustment_scales_down_group() -> None:
    from nautilus_trader.model.identifiers import InstrumentId

    inst1 = InstrumentId.from_str("AAA.SIM")
    inst2 = InstrumentId.from_str("BBB.SIM")

    signals = [_Signal(inst1, 0.8), _Signal(inst2, 0.8)]
    capital = 10_000.0

    # Manager without correlation adjustment → equal allocation 5k each
    cfg_no_adj = PortfolioConfig(
        allocation_method="equal",
        use_correlation_adjustment=False,
        max_correlated_weight=0.4,
        max_position_weight=1.0,
    )
    pm_no_adj = PortfolioManager(cfg_no_adj)
    alloc_no_adj = pm_no_adj.allocate_signals(signals, capital)
    assert alloc_no_adj[inst1] == alloc_no_adj[inst2] == capital / 2

    # Manager with correlation adjustment; set high correlation between instruments
    cfg_adj = PortfolioConfig(
        allocation_method="equal",
        use_correlation_adjustment=True,
        max_correlated_weight=0.4,  # at most 40% total for correlated group
        correlation_threshold=0.5,
        max_position_weight=1.0,
    )
    pm_adj = PortfolioManager(cfg_adj)
    # Seed correlation matrix to 0.9 between inst1 and inst2
    idx1 = pm_adj._get_instrument_index(inst1)  # type: ignore[attr-defined]
    idx2 = pm_adj._get_instrument_index(inst2)  # type: ignore[attr-defined]
    pm_adj._correlation_matrix[idx1, idx2] = 0.9  # type: ignore[attr-defined]
    pm_adj._correlation_matrix[idx2, idx1] = 0.9  # type: ignore[attr-defined]

    alloc_adj = pm_adj.allocate_signals(signals, capital)
    # Expect total group allocation scaled to 40% of capital, split evenly → 2k each
    assert round(alloc_adj[inst1], 5) == round(capital * 0.2, 5)
    assert round(alloc_adj[inst2], 5) == round(capital * 0.2, 5)
    assert round(alloc_adj[inst1] + alloc_adj[inst2], 5) == round(capital * 0.4, 5)
