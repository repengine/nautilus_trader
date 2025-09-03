"""
Metrics path coverage for MLSignalActor._try_generate_signal.

Stubs a counter with labels().inc() and verifies it is invoked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ml.actors.signal import MLSignalActor, ThresholdSignalStrategy
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue


class _DummyCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def labels(self, **labels: Any) -> "_DummyCounter":  # noqa: D401
        """Return self for chained inc()"""
        self.calls.append((tuple(sorted(labels.keys())), "labels"))
        return self

    def inc(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        """Record inc call"""
        self.calls.append(((), "inc"))


def _stub_bar() -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return SimpleNamespace(bar_type=SimpleNamespace(instrument_id=inst), ts_event=1)


def test_signals_generated_metric_incremented() -> None:
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
    actor.log = SimpleNamespace(debug=lambda *a, **k: None)
    # Do not set _strategy_store to avoid persistence call in this test
    # Stub metric counter
    counter = _DummyCounter()
    actor._signals_generated_metric = counter
    # Publish suppressed
    actor._publish_signal = lambda s: None

    MLSignalActor._try_generate_signal(actor, _stub_bar(), 0.1, 0.9, np.zeros(1, dtype=np.float32))  # type: ignore[misc]
    # Expect metric called
    assert any(tag == "inc" for _, tag in counter.calls)
