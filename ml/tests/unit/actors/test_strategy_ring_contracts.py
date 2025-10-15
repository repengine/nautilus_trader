"""
Strategy ring metadata contract and momentum invariants.

Contracts covered:
- Context includes required ring keys and semantics across wraps
- MomentumStrategy matches telescoping slope: (last-first)/(lookback-1)
- Confidence gate is inclusive (>=); momentum gate is strict (>)
"""

from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np
import numpy.typing as npt
import pytest
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.objects import Price, Quantity

from ml.actors.signal import MomentumStrategy


def _make_bar(bt: BarType, close: float) -> Bar:
    def p(v: float) -> str:
        return f"{v:.9f}"

    ts = 1
    return Bar(
        bar_type=bt,
        open=Price.from_str(p(close)),
        high=Price.from_str(p(close + 0.0001)),
        low=Price.from_str(p(close - 0.0001)),
        close=Price.from_str(p(close)),
        volume=Quantity.from_int(1000),
        ts_event=ts,
        ts_init=ts,
    )


def _compute_momentum_from_ring(ring: npt.NDArray[np.float32], idx: int, cnt: int, look: int) -> float:
    cap = int(ring.shape[0])
    first_idx = (idx - look) % cap
    last_idx = (idx - 1) % cap
    first_val = float(ring[first_idx])
    last_val = float(ring[last_idx])
    denom = max(1, look - 1)
    return (last_val - first_val) / denom


@pytest.mark.unit
def test_momentum_ring_contract_wrap_and_thresholds() -> None:
    bt = BarType.from_str("EUR/USD.SIM-1-MINUTE-MID-INTERNAL")
    bar = _make_bar(bt, close=1.1000)

    look = 4
    strat = MomentumStrategy(lookback=look, threshold=0.6, momentum_threshold=0.01)

    # Build ring with wrap: capacity 6, fill with 7 values
    cap = 6
    ring = np.zeros(cap, dtype=np.float32)
    values = [0.10, 0.12, 0.11, 0.14, 0.18, 0.19, 0.21]
    idx = 0
    cnt = 0
    for v in values:
        ring[idx] = float(v)
        idx = (idx + 1) % cap
        cnt = min(cnt + 1, cap)

    assert cnt == cap
    assert idx == (len(values) % cap)

    context: MutableMapping[str, Any] = {
        "_prediction_ring": ring,
        "_prediction_ring_index": int(idx),
        "_prediction_ring_count": int(cnt),
        "prediction_history": [0.1, 0.2, 0.3],
        "confidence_history": [0.6, 0.7, 0.8],
        "timestamp_ns": 1,
        "model_id": "test",
        "log_predictions": False,
        "market_regime": "normal",
    }

    # Compute reference momentum from ring
    mom = _compute_momentum_from_ring(ring, idx, cnt, look)
    # Confidence exactly at threshold should pass; momentum must be strictly > threshold
    pred = float(values[-1])
    features = np.zeros(1, dtype=np.float32)
    signal = strat.generate_signal(bar, prediction=pred, confidence=0.6, features=features, context=context)
    if abs(mom) > strat.momentum_threshold:
        assert signal is not None
    else:
        assert signal is None

    # Boundary checks
    # Momentum equal to threshold should NOT signal
    strat2 = MomentumStrategy(lookback=look, threshold=0.6, momentum_threshold=abs(mom))
    sig2 = strat2.generate_signal(bar, prediction=pred, confidence=0.6, features=features, context=context)
    assert sig2 is None
    # Confidence strictly below threshold should NOT signal
    sig3 = strat.generate_signal(bar, prediction=pred, confidence=0.59, features=features, context=context)
    assert sig3 is None
    # Confidence equal to threshold should be allowed if momentum passes
    if abs(mom) > strat.momentum_threshold:
        sig4 = strat.generate_signal(bar, prediction=pred, confidence=0.6, features=features, context=context)
        assert sig4 is not None


@pytest.mark.unit
def test_momentum_fallback_to_history_when_ring_missing() -> None:
    bt = BarType.from_str("EUR/USD.SIM-1-MINUTE-MID-INTERNAL")
    bar = _make_bar(bt, close=1.1000)
    look = 3
    strat = MomentumStrategy(lookback=look, threshold=0.5, momentum_threshold=0.0)

    history = [0.10, 0.12, 0.14]
    features = np.zeros(1, dtype=np.float32)
    context: MutableMapping[str, Any] = {
        # No ring keys
        "prediction_history": history,
        "confidence_history": [0.5, 0.5, 0.5],
        "timestamp_ns": 1,
        "model_id": "test",
        "log_predictions": False,
        "market_regime": "normal",
    }

    # Momentum from history
    hist_mom = float(np.mean(np.diff(history)))
    # Since momentum_threshold=0.0 and confidence=0.5 >= threshold, signal should be emitted
    sig = strat.generate_signal(bar, prediction=history[-1], confidence=0.5, features=features, context=context)
    assert sig is not None
    # Check that prediction is adjusted using momentum rule (prediction * (1 + momentum))
    assert pytest.approx(sig.prediction, rel=1e-6) == history[-1] * (1.0 + hist_mom)
