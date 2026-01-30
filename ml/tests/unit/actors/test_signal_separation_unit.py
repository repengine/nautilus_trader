"""
Signal separation gating test using MLSignalActor._try_generate_signal.

Verifies that min_signal_separation_bars prevents publishing too frequently.

"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from ml.actors.signal import MLSignalActor
from ml.actors.signal import ThresholdSignalStrategy
from ml.tests.utils.stubs import make_stub_bar
from ml.tests.utils.stubs import SignalActorHarness
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


def _stub_bar() -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return make_stub_bar(inst)


def test_signal_separation_gates_publishing() -> None:
    published: list[Any] = []

    actor = SignalActorHarness(
        _signal_strategy=ThresholdSignalStrategy(threshold=0.0),
        _signal_config=SimpleNamespace(
            min_signal_separation_bars=2,
            signal_strategy="threshold",
        ),
        _config=SimpleNamespace(log_predictions=False),
        id=SimpleNamespace(value="actor-1"),
        _model_id="m1",
        _last_signal_bar=1,
        _bars_processed=2,
        clock=SimpleNamespace(timestamp_ns=lambda: 123),
        log=SimpleNamespace(debug=lambda *a, **k: None),
        _prediction_window=np.zeros(1, dtype=np.float32),
        _publish_signal=lambda sig: published.append(sig),
    )
    actor_proxy = cast(MLSignalActor, actor)

    # Not enough separation: no publish
    MLSignalActor._try_generate_signal(actor_proxy, _stub_bar(), 0.1, 0.9, np.zeros(1, dtype=np.float32))
    assert not published

    # Advance bars to meet separation
    actor._bars_processed = 3
    MLSignalActor._try_generate_signal(actor_proxy, _stub_bar(), 0.1, 0.9, np.zeros(1, dtype=np.float32))
    assert published
