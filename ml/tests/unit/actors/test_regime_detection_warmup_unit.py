"""
Unit tests for regime detection warm-up policy.

Contracts:
- Before min_count (3), regime = "unknown".
- After min_count, increasing volatility proxy increases average volatility.
- Regression: count-based averaging differs from zero-padded; assert count-based behavior.
"""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import pytest
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.objects import Price, Quantity

from ml.actors.signal import MLSignalActor
from ml.tests.builders import MLConfigBuilder


def _bar(bt: BarType, close: float, ts: int) -> Bar:
    def p(v: float) -> str:
        return f"{v:.9f}"

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


@pytest.mark.unit
def test_regime_detection_count_based_warmup_and_monotonicity(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure dummy mode and no metrics server
    monkeypatch.setenv("ML_ALLOW_DUMMY", "1")
    monkeypatch.setenv("ML_DISABLE_METRICS_SERVER", "1")

    cfg = MLConfigBuilder.signal_config(
        adaptive_window=10,  # Larger than warm-up threshold
        enable_regime_detection=True,
    )
    actor = MLSignalActor(cfg)

    bt = cfg.bar_type
    # Feed two bars -> count = 2 (<3) => regime unknown
    for i in range(2):
        actor._update_prediction_history(prediction=0.0, confidence=0.0, bar=_bar(bt, 1.1000 + i * 0.01, ts=i + 1))
    actor._detect_market_regime(_bar(bt, 1.1200, ts=99))
    assert actor._market_regime == "unknown"

    # Third bar -> now count = 3, compute average volatility over valid prefix only
    # Use large jumps so avg over count-based window is high
    actor._update_prediction_history(prediction=0.0, confidence=0.0, bar=_bar(bt, 1.1400, ts=3))
    # Sanity: internal window count is 3 and window contains non-zero in first 3
    n = int(actor._window_count)
    assert n == 3
    avg_valid = float(np.mean(actor._volatility_window[:n]))
    avg_zero_padded = float(np.mean(actor._volatility_window))
    assert avg_valid > avg_zero_padded  # Count-based average should be greater

    actor._detect_market_regime(_bar(bt, 1.1500, ts=4))
    # With large average volatility, regime should be high_volatility (count-based)
    assert actor._market_regime == "high_volatility"

    # Monotonicity: increasing volatility increases average over valid prefix
    prev_avg = avg_valid
    actor._update_prediction_history(prediction=0.0, confidence=0.0, bar=_bar(bt, 1.2000, ts=5))
    new_avg = float(np.mean(actor._volatility_window[: int(actor._window_count)]))
    assert new_avg >= prev_avg
