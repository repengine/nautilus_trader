"""
Unit tests for ML signal generation strategies (no model/DB dependencies).

We validate basic behavior of ThresholdSignalStrategy to ensure that
hot-path logic produces an MLSignal when the confidence threshold is
met, and returns None otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ml.actors.base import MLSignal
from ml.actors.signal import ThresholdSignalStrategy
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


def _stub_bar() -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    bar_type = SimpleNamespace(instrument_id=inst)
    return SimpleNamespace(bar_type=bar_type, ts_event=1)


def test_threshold_strategy_generates_signal_when_confident() -> None:
    strat = ThresholdSignalStrategy(threshold=0.7)
    bar = _stub_bar()
    features = np.array([0.1, 0.2], dtype=np.float32)
    ctx = {"model_id": "m1", "timestamp_ns": 1}

    sig = strat.generate_signal(bar=bar, prediction=0.9, confidence=0.75, features=features, context=ctx)
    assert isinstance(sig, MLSignal)
    assert sig.model_id == "m1"
    assert sig.prediction == 0.9


def test_threshold_strategy_suppresses_signal_when_not_confident() -> None:
    strat = ThresholdSignalStrategy(threshold=0.8)
    bar = _stub_bar()
    features = np.array([0.1, 0.2], dtype=np.float32)
    ctx = {"model_id": "m1", "timestamp_ns": 1}

    sig = strat.generate_signal(bar=bar, prediction=0.9, confidence=0.5, features=features, context=ctx)
    assert sig is None

