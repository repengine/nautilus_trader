from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from ml.common.message_bus import MessagePublisherProtocol
from ml.config.events import Stage
from ml.stores.data_store import DataStore


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


class StubRegistry:
    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: str,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: str,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        return None

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: str,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        return None

    # Protocol methods unused in this test
    def get_manifest(self, dataset_id: str):  # type: ignore[override]
        raise NotImplementedError

    def get_contract(self, dataset_id: str):  # type: ignore[override]
        raise NotImplementedError

    def register_dataset(self, manifest):  # type: ignore[override]
        raise NotImplementedError


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


def test_data_store_stage_first_topics(tmp_path) -> None:
    pub = CapturePublisher()
    with env({"ML_BUS_SCHEME": "stage_first", "ML_BUS_ENABLE": "1"}):
        store = DataStore(
            connection_string=f"sqlite:///{tmp_path}/ds.db",
            registry=StubRegistry(),
            publisher=pub,
            enable_publishing=True,
        )
        store.emit_event(
            dataset_id="features",
            instrument_id="EURUSD.SIM",
            stage=Stage.FEATURE_COMPUTED,
            source="historical",
            run_id="r1",
            ts_min=1,
            ts_max=2,
            count=1,
        )
        assert pub.calls, "Publisher should be called when enabled"
        topic, payload = pub.calls[-1]
        assert topic.startswith("events.ml.FEATURE_COMPUTED."), topic
        assert payload["stage"] == Stage.FEATURE_COMPUTED.value
