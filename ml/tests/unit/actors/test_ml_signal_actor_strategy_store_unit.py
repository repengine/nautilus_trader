"""
Ensure MLSignalActor._try_generate_signal persists to StrategyStore when present.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ml.actors.signal import MLSignalActor
from ml.actors.signal import ThresholdSignalStrategy
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


class _StubStrategyStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def write_signal(self, **kwargs: Any) -> None:
        """Capture persisted decision"""
        self.calls.append(kwargs)


def _stub_bar() -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return SimpleNamespace(bar_type=SimpleNamespace(instrument_id=inst), ts_event=1)


def test_signal_persists_to_strategy_store() -> None:
    class _D:
        pass

    actor = _D()
    actor._signal_strategy = ThresholdSignalStrategy(threshold=0.0)
    actor._last_signal_bar = -1
    actor._bars_processed = 0
    actor._model_id = "m1"
    actor._signal_config = SimpleNamespace(min_signal_separation_bars=0, signal_strategy="threshold")
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
    # Stub strategy store to capture writes
    store = _StubStrategyStore()
    actor._strategy_store = store
    # No external publish needed here
    actor._publish_signal = lambda s: None

    MLSignalActor._try_generate_signal(actor, _stub_bar(), 0.1, 0.9, np.zeros(2, dtype=np.float32))  # type: ignore[misc]
    assert store.calls, "Expected a strategy store write"
    call = store.calls[-1]
    assert call["strategy_id"]
    assert call["instrument_id"].endswith("SIM")
    assert call["signal_type"] in ("buy", "sell")

