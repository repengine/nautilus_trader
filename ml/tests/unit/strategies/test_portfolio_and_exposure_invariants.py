from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from ml.actors.base import MLSignal
from ml.strategies.portfolio import (
    PortfolioBatchingConfig,
    PortfolioConfig,
    PortfolioManager,
)
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


def test_portfolio_batching_config_validation_guards() -> None:
    with pytest.raises(ValueError, match="window_ms must be non-negative"):
        PortfolioBatchingConfig(window_ms=-1)
    with pytest.raises(ValueError, match="min_batch_size must be >= 1"):
        PortfolioBatchingConfig(min_batch_size=0)
    with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
        PortfolioBatchingConfig(max_batch_size=0)
    with pytest.raises(ValueError, match="min_batch_size must be <= max_batch_size"):
        PortfolioBatchingConfig(min_batch_size=3, max_batch_size=2)


def test_portfolio_unknown_method_falls_back_to_equal_and_caps_positions() -> None:
    inst1 = InstrumentId.from_str("AAA.SIM")
    inst2 = InstrumentId.from_str("BBB.SIM")
    inst3 = InstrumentId.from_str("CCC.SIM")
    signals = [
        MLSignal(
            instrument_id=inst1,
            model_id="m",
            prediction=0.9,
            confidence=0.9,
            metadata={"decision_metadata": {"version": "v1"}},
            ts_event=1,
        ),
        MLSignal(
            instrument_id=inst2,
            model_id="m",
            prediction=0.9,
            confidence=0.9,
            metadata={"decision_metadata": {"version": "v1"}},
            ts_event=1,
        ),
        MLSignal(
            instrument_id=inst3,
            model_id="m",
            prediction=0.9,
            confidence=0.9,
            metadata={"decision_metadata": {"version": "v1"}},
            ts_event=1,
        ),
    ]
    pm = PortfolioManager(
        PortfolioConfig(
            allocation_method="unknown",
            use_correlation_adjustment=False,
            min_position_weight=0.01,
            max_position_weight=0.25,
            max_positions=3,
        ),
    )

    allocations = pm.allocate_signals(signals, available_capital=10_000.0)

    assert allocations[inst1] == 2500.0
    assert allocations[inst2] == 2500.0
    assert allocations[inst3] == 2500.0


def test_portfolio_kelly_falls_back_to_equal_without_positive_sharpe() -> None:
    inst1 = InstrumentId.from_str("AAA.SIM")
    inst2 = InstrumentId.from_str("BBB.SIM")
    signals = [
        MLSignal(
            instrument_id=inst1,
            model_id="m",
            prediction=0.8,
            confidence=0.8,
            metadata={"decision_metadata": {"version": "v1"}},
            ts_event=1,
        ),
        MLSignal(
            instrument_id=inst2,
            model_id="m",
            prediction=0.8,
            confidence=0.8,
            metadata={"decision_metadata": {"version": "v1"}},
            ts_event=1,
        ),
    ]
    pm = PortfolioManager(
        PortfolioConfig(
            allocation_method="kelly",
            use_correlation_adjustment=False,
            max_positions=2,
            min_position_weight=0.01,
            max_position_weight=1.0,
        ),
    )

    allocations = pm.allocate_signals(signals, available_capital=2_000.0)

    assert allocations[inst1] == 1_000.0
    assert allocations[inst2] == 1_000.0


def test_portfolio_set_annualization_factor_and_rebalance_threshold() -> None:
    inst = InstrumentId.from_str("AAA.SIM")
    pm = PortfolioManager(PortfolioConfig(rebalance_threshold=0.05))
    pm._target_weights = {inst: 0.8}
    pm._current_weights = {inst: 0.7}
    pm._last_rebalance_time = 0.0

    pm.set_annualization_factor(0.0)
    assert pm._annualization_factor is None

    pm.set_annualization_factor(252.0)
    assert pm._annualization_factor == 252.0
    assert pm.should_rebalance() is True


def test_portfolio_apply_limits_drops_small_allocations_and_caps_large() -> None:
    inst_small = InstrumentId.from_str("AAA.SIM")
    inst_mid = InstrumentId.from_str("BBB.SIM")
    inst_large = InstrumentId.from_str("CCC.SIM")
    pm = PortfolioManager(
        PortfolioConfig(
            min_position_weight=0.10,
            max_position_weight=0.20,
        ),
    )

    limited = pm._apply_limits(
        allocations={
            inst_small: 50.0,
            inst_mid: 150.0,
            inst_large: 500.0,
        },
        capital=1_000.0,
    )

    assert inst_small not in limited
    assert limited[inst_mid] == 150.0
    assert limited[inst_large] == 200.0


def test_portfolio_correlation_adjustment_keeps_group_when_within_limit() -> None:
    inst1 = InstrumentId.from_str("AAA.SIM")
    inst2 = InstrumentId.from_str("BBB.SIM")
    pm = PortfolioManager(
        PortfolioConfig(
            use_correlation_adjustment=True,
            correlation_threshold=0.5,
            max_correlated_weight=1.0,
        ),
    )
    returns = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pm.update_correlation(inst1, inst2, returns, returns)

    allocations = {inst1: 600.0, inst2: 400.0}
    adjusted = pm._adjust_for_correlation(allocations)

    assert adjusted == allocations


def test_portfolio_should_not_rebalance_within_minimum_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = InstrumentId.from_str("AAA.SIM")
    pm = PortfolioManager(PortfolioConfig(rebalance_threshold=0.01))
    pm._target_weights = {inst: 0.9}
    pm._current_weights = {inst: 0.1}
    pm._last_rebalance_time = 9_900.0
    monkeypatch.setattr("ml.strategies.portfolio.time.time", lambda: 10_000.0)

    assert pm.should_rebalance() is False


def test_portfolio_correlation_matrix_snapshot_and_max_correlation() -> None:
    inst1 = InstrumentId.from_str("AAA.SIM")
    inst2 = InstrumentId.from_str("BBB.SIM")
    pm = PortfolioManager(PortfolioConfig())

    returns_one = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    returns_two = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pm.update_correlation(inst1, inst2, returns_one, returns_two, ts_event=123)

    matrix = pm.get_correlation_matrix([inst1, inst2])
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] == pytest.approx(1.0)

    snapshot = pm.get_correlation_snapshot(inst1, inst2)
    assert snapshot.ts_event == 123
    assert snapshot.source == "portfolio_manager"

    pm._current_weights = {inst1: 0.6, inst2: 0.4}
    assert pm._get_max_correlation() >= 0.0


def test_portfolio_metrics_include_performance_and_default_sharpe() -> None:
    inst = InstrumentId.from_str("CCC.SIM")
    pm = PortfolioManager(PortfolioConfig(correlation_lookback=16))
    for value in (0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01, 0.03, -0.01, 0.02):
        pm.update_returns(inst, value)

    pm.update_performance(inst, pnl=50.0)
    pm._current_weights = {inst: 1.0}
    metrics = pm.get_portfolio_metrics()

    assert metrics["total_pnl"] == pytest.approx(50.0)
    assert metrics["n_positions"] == pytest.approx(1.0)
    assert metrics["concentration_hhi"] == pytest.approx(1.0)
