from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any

import numpy as np

from ml.actors.base import MLSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.common.message_bus import MessagePublisherProtocol
from ml.tests.builders import MLConfigBuilder
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


@contextmanager
def env(vars: dict[str, str]) -> None:
    old = {k: os.environ.get(k) for k in vars}
    try:
        os.environ.update(vars)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


def test_actor_side_domain_event_bridge_publishes(
    monkeypatch,
    tmp_path,
    default_instrument_id,
    default_bar_type,
) -> None:
    # Arrange: env enables actor-side bridge with stage-first scheme
    pub = CapturePublisher()

    def _fake_factory(_cfg: Any) -> MessagePublisherProtocol:
        return pub

    with env(
        {
            "ML_BUS_FROM_ACTOR": "1",
            "ML_BUS_ENABLE": "1",
            "ML_BUS_SCHEME": "stage_first",
        },
    ):
        # Patch the publisher factory used by the actor bus helper
        monkeypatch.setattr("ml.actors.ml_domain_events.publisher_from_config", _fake_factory)

        cfg = MLConfigBuilder.signal_config(
            model_path=str(tmp_path / "model.onnx"),
            model_id="demo",
            bar_type=default_bar_type,
            instrument_id=default_instrument_id,
        )
        actor = MLSignalActor(cfg)
        # Avoid base publish path side-effects (actor is unregistered in unit tests).
        # Patch via the actor's actual MRO to stay stable if other tests reload/import-scrub modules.
        base_cls = next(
            cls for cls in type(actor).__mro__ if cls.__name__ == "BaseMLInferenceActor"
        )
        monkeypatch.setattr(base_cls, "_publish_signal", lambda self, s: None)

        # Act: publish a signal; bridge enqueues event in background
        sig = MLSignal(
            instrument_id=default_instrument_id,
            model_id="demo",
            prediction=0.9,
            confidence=0.9,
            features=np.array([0.0], dtype=np.float32),
            ts_event=123,
            ts_init=123,
        )
        actor._publish_signal(sig)

        # Allow background thread to drain
        time.sleep(0.05)

        # Stop and drain bridge
        bridge = getattr(actor, "_actor_bus_bridge", None)
        if bridge is not None:
            bridge.stop(drain=True, timeout=1.0)

    # Assert: publisher received a stage-first SIGNAL_EMITTED topic
    assert pub.calls, "actor-side publisher should be called"
    topic, payload = pub.calls[-1]
    assert topic.startswith("events.ml.SIGNAL_EMITTED."), topic
    assert payload.get("dataset_id") == "signals"
    assert payload.get("stage") == "SIGNAL_EMITTED"
