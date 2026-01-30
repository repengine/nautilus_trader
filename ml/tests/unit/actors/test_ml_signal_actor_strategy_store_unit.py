"""
Ensure MLSignalActor._try_generate_signal persists to StrategyStore when present.
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


class _StubStrategyStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def write_signal(self, **kwargs: Any) -> None:
        """
        Capture persisted decision.
        """
        self.calls.append(kwargs)


def _stub_bar() -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return make_stub_bar(inst)


def test_signal_persists_to_strategy_store() -> None:
    store = _StubStrategyStore()

    actor = SignalActorHarness(
        _signal_strategy=ThresholdSignalStrategy(threshold=0.0),
        _signal_config=SimpleNamespace(
            min_signal_separation_bars=0,
            signal_strategy="threshold",
        ),
        _config=SimpleNamespace(log_predictions=False),
        id=SimpleNamespace(value="actor-1"),
        _model_id="m1",
        _last_signal_bar=-1,
        _bars_processed=0,
        clock=SimpleNamespace(timestamp_ns=lambda: 123),
        _signals_generated_metric=None,
        log=SimpleNamespace(debug=lambda *a, **k: None),
        _strategy_store=store,
        _publish_signal=lambda s: None,
    )

    actor_proxy = cast(MLSignalActor, actor)

    MLSignalActor._try_generate_signal(actor_proxy, _stub_bar(), 0.1, 0.9, np.zeros(2, dtype=np.float32))
    assert store.calls, "Expected a strategy store write"
    call = store.calls[-1]
    assert call["strategy_id"]
    assert call["instrument_id"].endswith("SIM")
    assert call["signal_type"] in ("buy", "sell")
