from __future__ import annotations

import numpy as np

from ml.actors.base import MLSignal
from ml.strategies.portfolio import PortfolioConfig, PortfolioManager


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
        annualization_factor=252.0,
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
    returns = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pm.update_correlation(a, b, returns, returns)

    capital = 10_000.0
    base_ts = 1_700_000_000_000_000_000
    sigs = [
        MLSignal(instrument_id=a, model_id="model", prediction=0.0, confidence=0.8, ts_event=base_ts),
        MLSignal(instrument_id=b, model_id="model", prediction=0.0, confidence=0.8, ts_event=base_ts),
        MLSignal(instrument_id=c, model_id="model", prediction=0.0, confidence=0.8, ts_event=base_ts),
    ]

    alloc = pm.allocate_signals(sigs, capital)

    # Risk parity: lower vol → larger weight; within (A,B) group, A gets more than B
    assert alloc[a] > alloc[b]

    # Correlated group (A,B) total should be capped at ~50% of capital
    group_total = alloc[a] + alloc[b]
    assert group_total <= capital * 0.5 + 1e-6
    # Remaining allocation is not re-assigned in current implementation
