"""
Smoke test for MLSignalActor hot path using stubs (no Nautilus runtime).

We bypass heavy initialization by constructing the object without calling
the base initializer and invoking the internal signal generation helper
directly with a stubbed bar and strategy.
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


def _stub_bar() -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    bar_type = SimpleNamespace(instrument_id=inst)
    # Only the attributes used by strategies are required
    return SimpleNamespace(bar_type=bar_type, ts_event=1)


def test_ml_signal_actor_try_generate_signal_smoke() -> None:
    # Dummy self object implementing attributes used by _try_generate_signal
    class _Dummy:
        pass

    actor = _Dummy()
    actor._signal_strategy = ThresholdSignalStrategy(threshold=0.7)
    actor._last_signal_bar = -999
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

    class _Clock:
        def timestamp_ns(self) -> int:
            return 123

    actor.clock = _Clock()
    actor._performance_monitor = None
    actor._signals_generated_metric = None

    published: list[Any] = []

    def _publish_signal(sig: Any) -> None:
        published.append(sig)

    actor._publish_signal = _publish_signal
    actor.log = SimpleNamespace(debug=lambda *a, **k: None)

    # Call the unbound method with dummy self
    features = np.array([0.1, 0.2], dtype=np.float32)
    MLSignalActor._try_generate_signal(  # type: ignore[misc]
        actor,  # self
        bar=_stub_bar(),
        prediction=0.9,
        confidence=0.8,
        features=features,
    )

    assert len(published) == 1
    sig = published[0]
    assert float(sig.prediction) == 0.9
    assert sig.model_id == "m1"
