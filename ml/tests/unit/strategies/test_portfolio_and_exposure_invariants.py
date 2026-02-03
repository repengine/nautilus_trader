from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ml.actors.base import MLSignal
from ml.strategies.portfolio import PortfolioConfig, PortfolioManager
from ml.strategies.risk import RiskConfig, RiskManager
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity


# ==========================
# Helpers / dummies
# ==========================


@dataclass
class _DummyBalance:
    value: float

    def as_double(self) -> float:
        return self.value


@dataclass
class _DummyPosition:
    instrument_id: object
    quantity: Quantity
    is_open: bool = True


@dataclass
class _DummyAccount:
    bal: float

    def balance_total(self) -> _DummyBalance:
        return _DummyBalance(self.bal)


@dataclass
class _DummyPortfolio:
    _positions: list[_DummyPosition]
    _balance: float = 10000.0

    def positions(self) -> list[_DummyPosition]:  # noqa: D401
        return list(self._positions)

    def account(self, _venue: object) -> _DummyAccount:
        return _DummyAccount(self._balance)


def test_exposure_limit_rejects_when_exceeding_max() -> None:
    # Setup risk manager with 50% max exposure
    rcfg = RiskConfig(max_total_exposure=0.5, max_position_pct=1.0)
    rm = RiskManager(rcfg)
    inst = InstrumentId.from_str("AAA.SIM")
    # Existing open position worth 4000
    pos = _DummyPosition(inst, Quantity.from_str("4000"), is_open=True)
    portfolio = _DummyPortfolio([pos], _balance=10000.0)

    # Propose new position 2000 -> total exposure 6000/10000 = 60% > 50%
    proposed = Quantity.from_str("2000")
    assert rm._check_portfolio_exposure(2000.0, 10000.0, portfolio) is False

    # Via public check_position path should reject
    result = rm.check_position(proposed, inst, portfolio)
    assert result is None


def test_portfolio_correlation_adjustment_scales_down_group() -> None:
    inst1 = InstrumentId.from_str("AAA.SIM")
    inst2 = InstrumentId.from_str("BBB.SIM")

    ts_base = 1_700_000_000_000_000_000
    signals = [
        MLSignal(
            instrument_id=inst1,
            model_id="model",
            prediction=0.0,
            confidence=0.8,
            metadata={"decision_metadata": {"version": "v1"}},
            ts_event=ts_base,
        ),
        MLSignal(
            instrument_id=inst2,
            model_id="model",
            prediction=0.0,
            confidence=0.8,
            metadata={"decision_metadata": {"version": "v1"}},
            ts_event=ts_base,
        ),
    ]
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
    returns = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pm_adj.update_correlation(inst1, inst2, returns, returns)

    alloc_adj = pm_adj.allocate_signals(signals, capital)
    # Expect total group allocation scaled to 40% of capital, split evenly → 2k each
    assert round(alloc_adj[inst1], 5) == round(capital * 0.2, 5)
    assert round(alloc_adj[inst2], 5) == round(capital * 0.2, 5)
    assert round(alloc_adj[inst1] + alloc_adj[inst2], 5) == round(capital * 0.4, 5)
