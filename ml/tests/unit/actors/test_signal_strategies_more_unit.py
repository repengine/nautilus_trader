"""
Additional strategy tests for Extremes and Momentum strategies.

Functional outcomes only: return MLSignal for extreme/momentum cases
under deterministic contexts.

"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.actors.base import MLSignal
from ml.actors.signal import ExtremesStrategy
from ml.actors.signal import MomentumStrategy
from ml.tests.utils.stubs import make_stub_bar
from nautilus_trader.model.identifiers import InstrumentId


def _stub_bar(instrument_id: InstrumentId) -> object:
    return make_stub_bar(instrument_id)


def test_extremes_strategy_detects_top_extreme(default_instrument_id: InstrumentId) -> None:
    strat = ExtremesStrategy(top_pct=0.1, threshold=0.5, window_size=10)
    # Prefill strategy ring buffer in context to avoid warmup early-exit
    preds = np.linspace(0.0, 0.9, 10, dtype=np.float32)
    ctx: dict[str, Any] = {
        "timestamp_ns": 1,
        "model_id": "m1",
        "_pred_ring": preds.copy(),
        "_pred_scratch": np.empty(10, dtype=np.float32),
        "_pred_ring_filled": 10,
        "_pred_ring_idx": 0,
    }
    sig = strat.generate_signal(
        bar=_stub_bar(default_instrument_id),
        prediction=0.95,
        confidence=0.9,
        features=np.zeros(1, dtype=np.float32),
        context=ctx,
    )
    assert isinstance(sig, MLSignal)


def test_momentum_strategy_requires_slope(default_instrument_id: InstrumentId) -> None:
    strat = MomentumStrategy(lookback=5, threshold=0.5, momentum_threshold=0.01)
    # Increasing predictions -> positive momentum
    preds = [0.1, 0.12, 0.15, 0.18, 0.22]
    ctx: dict[str, Any] = {"prediction_history": preds, "timestamp_ns": 1, "model_id": "m1"}
    sig = strat.generate_signal(
        bar=_stub_bar(default_instrument_id),
        prediction=0.3,
        confidence=0.9,
        features=np.zeros(1, dtype=np.float32),
        context=ctx,
    )
    assert isinstance(sig, MLSignal)
