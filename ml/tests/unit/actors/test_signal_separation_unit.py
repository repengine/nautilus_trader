"""
Signal separation gating test using MLSignalActor._try_generate_signal.

Verifies that min_signal_separation_bars prevents publishing too frequently.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ml.actors.signal import MLSignalActor, ThresholdSignalStrategy
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue


def _stub_bar() -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return SimpleNamespace(bar_type=SimpleNamespace(instrument_id=inst), ts_event=1)


def test_signal_separation_gates_publishing() -> None:
    class _D:  # dummy self
        pass

    actor = _D()
    actor._signal_strategy = ThresholdSignalStrategy(threshold=0.0)
    actor._last_signal_bar = 1
    actor._bars_processed = 2
    actor._model_id = "m1"
    actor._signal_config = SimpleNamespace(min_signal_separation_bars=2, signal_strategy="threshold")
    actor._config = SimpleNamespace(log_predictions=False)
    actor.id = SimpleNamespace(value="actor-1")
    actor._prediction_history = []
    actor._confidence_history = []
    actor._adaptive_threshold = 0.0
    actor._market_regime = None
    actor._feature_set_id = None
    actor.clock = SimpleNamespace(timestamp_ns=lambda: 123)
    actor._performance_monitor = None
    actor._signals_generated_metric = None
    actor.log = SimpleNamespace(debug=lambda *a, **k: None)

    published: list[Any] = []
    actor._publish_signal = lambda sig: published.append(sig)

    # Not enough separation: no publish
    MLSignalActor._try_generate_signal(actor, _stub_bar(), 0.1, 0.9, np.zeros(1, dtype=np.float32))  # type: ignore[misc]
    assert not published

    # Advance bars to meet separation
    actor._bars_processed = 3
    MLSignalActor._try_generate_signal(actor, _stub_bar(), 0.1, 0.9, np.zeros(1, dtype=np.float32))  # type: ignore[misc]
    assert published

