from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from ml.actors.signal import MLSignalActor
from ml.common.message_bus import MessagePublisherProtocol
from ml.tests.builders import MLConfigBuilder


@contextmanager
def env(vars: dict[str, str]) -> Iterator[None]:
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


@pytest.mark.contracts
def test_actor_bus_disables_store_publishers(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, default_instrument_id, default_bar_type) -> None:  # noqa: ANN001 - fixture types
    pub = CapturePublisher()

    def _fake_factory(_cfg: Any) -> MessagePublisherProtocol:  # noqa: ANN001
        return pub

    with env({"ML_BUS_FROM_ACTOR": "1", "ML_BUS_ENABLE": "1", "ML_BUS_SCHEME": "stage_first"}):
        monkeypatch.setattr("ml.actors.ml_domain_events.publisher_from_config", _fake_factory)

        cfg = MLConfigBuilder.signal_config(
            model_path=str(tmp_path / "model.onnx"),
            model_id="demo",
            bar_type=default_bar_type,
            instrument_id=default_instrument_id,
        )
        actor = MLSignalActor(cfg)

        # All store-level publishers should be disabled to avoid double-publish
        stores = [
            getattr(actor, "_feature_store", None),
            getattr(actor, "_model_store", None),
            getattr(actor, "_strategy_store", None),
            getattr(actor, "_data_store", None),
        ]
        for st in stores:
            if st is None:
                continue
            if hasattr(st, "publisher"):
                assert getattr(st, "publisher") is None
            if hasattr(st, "_enable_publishing"):
                assert bool(getattr(st, "_enable_publishing")) is False
