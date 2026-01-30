"""
Metrics path coverage for MLSignalActor._try_generate_signal.

Stubs a counter with labels().inc() and verifies it is invoked.

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


class _DummyCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def labels(self, **labels: Any) -> _DummyCounter:
        """
        Return self for chained inc()
        """
        self.calls.append((tuple(sorted(labels.keys())), "labels"))
        return self

    def inc(self, *args: Any, **kwargs: Any) -> None:
        """
        Record inc call.
        """
        self.calls.append(((), "inc"))


def _stub_bar() -> object:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return make_stub_bar(inst)


def test_signals_generated_metric_incremented() -> None:
    counter = _DummyCounter()

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
        _performance_monitor=None,
        _signals_generated_metric=counter,
        log=SimpleNamespace(debug=lambda *a, **k: None),
        _publish_signal=lambda s: None,
    )

    actor_proxy = cast(MLSignalActor, actor)

    MLSignalActor._try_generate_signal(actor_proxy, _stub_bar(), 0.1, 0.9, np.zeros(1, dtype=np.float32))
    # Expect metric called
    assert any(tag == "inc" for _, tag in counter.calls)
