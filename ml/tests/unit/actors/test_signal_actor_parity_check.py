from __future__ import annotations

import time
from typing import Deque, cast

import numpy as np
import numpy.typing as npt
import pytest

from ml.actors.signal import MLSignalActor, MLSignalActorConfig
from collections import deque


def _bar_dict(ts_ns: int) -> dict[str, str | int]:
    # Minimal OHLCV bar dict values (strings for price fields as required by Bar.from_dict)
    return {
        "bar_type": "SPY.XNAS-1-MINUTE-LAST-EXTERNAL",
        "open": "100.0",
        "high": "101.0",
        "low": "99.5",
        "close": "100.5",
        "volume": "1000",
        "ts_event": ts_ns,
        "ts_init": ts_ns,
    }


@pytest.mark.serial
def test_signal_actor_parity_smoke_check_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Configure actor to enable parity smoke-check with a short window.
    cfg = MLSignalActorConfig(
        model_path="models/dummy.onnx",
        model_id="dummy",
        bar_type="SPY.XNAS-1-MINUTE-LAST-EXTERNAL",
        instrument_id="SPY.XNAS",
        warm_up_period=0,
        use_dummy_stores=True,
        actor_id="ParityActor-001",
    )
    actor = MLSignalActor(cfg)
    # Enable parity smoke-check on the actor (test-only fields)
    setattr(actor, "_parity_enabled", True)
    setattr(actor, "_parity_window", 5)
    setattr(actor, "_parity_tolerance", 1e-6)

    # Stub prediction to avoid requiring a real model
    def _predict_stub(features: npt.NDArray[np.float32]) -> tuple[float, float]:
        return float(np.mean(features)), 0.9

    monkeypatch.setattr(actor, "_predict", _predict_stub)

    # Monkeypatch compute to force parity vectors, and pre-populate recent buffers
    def _compute_stub(_bar: object) -> npt.NDArray[np.float32]:
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    setattr(actor, "_compute_features", _compute_stub)
    # Pre-fill recent buffers
    recent_bars = cast(deque[object], getattr(actor, "_recent_bars"))
    recent_features = cast(deque[npt.NDArray[np.float32]], getattr(actor, "_recent_features"))
    recent_bars.clear()
    recent_features.clear()
    for _ in range(5):
        recent_bars.append(object())
        recent_features.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    # Run parity check directly
    getattr(actor, "_run_parity_smoke_check")()

    assert getattr(actor, "_parity_checked", False) is True
