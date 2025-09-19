from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.strategies.portfolio import PortfolioConfig, PortfolioManager


@dataclass
class _Sig:
    instrument_id: object
    confidence: float = 0.8


def test_risk_parity_with_correlation_adjustment() -> None:
    from nautilus_trader.model.identifiers import InstrumentId

    a = InstrumentId.from_str("AAA.SIM")
    b = InstrumentId.from_str("BBB.SIM")
    c = InstrumentId.from_str("CCC.SIM")

    cfg = PortfolioConfig(
        allocation_method="risk_parity",
        use_correlation_adjustment=True,
        max_correlated_weight=0.5,  # cap correlated group at 50%
        min_position_weight=0.05,
        max_position_weight=1.0,
        correlation_threshold=0.6,
        correlation_lookback=30,
    )

    pm = PortfolioManager(cfg)

    # Seed volatilities via returns: A low vol, B high vol, C medium (vary values to avoid zero std)
    for r in (0.001 + 0.0001 * np.sin(np.linspace(0, 2 * np.pi, 30))):
        pm.update_returns(a, float(r))
    for r in (0.006 + 0.001 * np.sin(np.linspace(0, 2 * np.pi, 30))):
        pm.update_returns(b, float(r))
    for r in (0.0025 + 0.0005 * np.sin(np.linspace(0, 2 * np.pi, 30))):
        pm.update_returns(c, float(r))

    # Make A and B highly correlated; C uncorrelated
    idx_a = pm._get_instrument_index(a)  # type: ignore[attr-defined]
    idx_b = pm._get_instrument_index(b)  # type: ignore[attr-defined]
    pm._correlation_matrix[idx_a, idx_b] = 0.9  # type: ignore[attr-defined]
    pm._correlation_matrix[idx_b, idx_a] = 0.9  # type: ignore[attr-defined]

    capital = 10_000.0
    sigs = [_Sig(a), _Sig(b), _Sig(c)]

    alloc = pm.allocate_signals(sigs, capital)

    # Risk parity: lower vol → larger weight; within (A,B) group, A gets more than B
    assert alloc[a] > alloc[b]

    # Correlated group (A,B) total should be capped at ~50% of capital
    group_total = alloc[a] + alloc[b]
    assert group_total <= capital * 0.5 + 1e-6
    # Remaining allocation is not re-assigned in current implementation
