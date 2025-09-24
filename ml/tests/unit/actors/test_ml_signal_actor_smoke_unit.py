"""
Smoke test for MLSignalActor hot path using stubs (no Nautilus runtime).

We bypass heavy initialization by constructing the object without calling the base
initializer and invoking the internal signal generation helper directly with a stubbed
bar and strategy.

"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from ml.actors.signal import MLSignalActor
from ml.actors.signal import ThresholdSignalStrategy
from ml.tests.utils.stubs import SignalActorHarness
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


def _stub_bar(instrument_id: InstrumentId) -> object:
    bar_type = SimpleNamespace(instrument_id=instrument_id)
    # Only the attributes used by strategies are required
    return SimpleNamespace(bar_type=bar_type, ts_event=1)


def test_ml_signal_actor_try_generate_signal_smoke(default_instrument_id: InstrumentId) -> None:
    published: list[Any] = []

    actor = SignalActorHarness(
        _signal_strategy=ThresholdSignalStrategy(threshold=0.7),
        _signal_config=SimpleNamespace(
            min_signal_separation_bars=0,
            signal_strategy="threshold",
        ),
        _config=SimpleNamespace(log_predictions=False),
        id=SimpleNamespace(value="actor-1"),
        _model_id="m1",
        _last_signal_bar=-999,
        _bars_processed=0,
        clock=SimpleNamespace(timestamp_ns=lambda: 123),
        _publish_signal=lambda sig: published.append(sig),
        log=SimpleNamespace(debug=lambda *a, **k: None),
    )

    actor_proxy = cast(MLSignalActor, actor)

    # Call the unbound method with dummy self
    features = np.array([0.1, 0.2], dtype=np.float32)
    MLSignalActor._try_generate_signal(
        actor_proxy,
        bar=_stub_bar(default_instrument_id),
        prediction=0.9,
        confidence=0.8,
        features=features,
    )

    assert len(published) == 1
    sig = published[0]
    assert float(sig.prediction) == 0.9
    assert sig.model_id == "m1"
