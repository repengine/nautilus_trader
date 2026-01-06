from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, ContextManager
from unittest.mock import MagicMock

import pytest

from ml.common.message_bus import MessagePublisherProtocol
from ml.stores.feature_store_facade import FeatureStore


PatchEngineManager = Callable[..., ContextManager[MagicMock]]


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


@pytest.mark.contracts
def test_feature_store_honors_env_topic_scheme_and_prefix(
    monkeypatch: pytest.MonkeyPatch,
    patch_engine_manager: PatchEngineManager,
) -> None:
    # Avoid real DB interactions
    monkeypatch.setattr("ml.stores.feature_store_facade.FeatureStore._setup_tables", lambda self: None)
    monkeypatch.setattr(
        "ml.stores.feature_store_facade.FeatureStore._execute_write",
        lambda self, row: None,
    )

    pub = CapturePublisher()
    with env({"ML_BUS_SCHEME": "stage_first", "ML_BUS_TOPIC_PREFIX": "custom.prefix"}):
        with patch_engine_manager():
            store = FeatureStore(
                connection_string="postgresql://ignored",  # ignored due to monkeypatch
                enable_publishing=True,
                publisher=pub,
                publish_mode="batch",
            )
            # write_features with explicit args should publish a batch summary when publish_mode includes "batch"
            store.write_features(
                feature_set_id="fs",
                instrument_id="EUR/USD",
                features={"x": 1.0},
                ts_event=123,
            )

            assert pub.calls, "Expected a publish call"
            topic, payload = pub.calls[-1]
            assert topic.startswith("custom.prefix.FEATURE_COMPUTED."), topic
            assert payload["stage"] == "FEATURE_COMPUTED"
            assert payload["status"] in {"success", "partial", "failed"}
