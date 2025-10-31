from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ml.config.bus import MessageBusConfig
from ml.config.streaming_pipeline import StreamingPersistenceConfig
from ml.consumers.streaming_training_worker import StreamingTrainingPersistenceWorker
from ml.tests.fixtures.streaming_events import build_streaming_test_payloads


class _DummyRedis:
    def __init__(self, batches: list[list[tuple[str, list[tuple[str, dict[str, str]]]]]]) -> None:
        self._batches = list(batches)

    @classmethod
    def from_url(cls, url: str, decode_responses: bool = False) -> _DummyRedis:  # noqa: ARG003
        module = sys.modules["redis"]
        batches = getattr(module, "_batches", [])
        return cls(batches)

    def xread(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        return self._batches.pop(0) if self._batches else []


@pytest.mark.integration
@pytest.mark.redis
def test_streaming_persistence_worker_persists_redis_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dummy_module = ModuleType("redis")
    dummy_module.Redis = _DummyRedis
    dummy_module._batches = []
    monkeypatch.setitem(sys.modules, "redis", dummy_module)
    monkeypatch.setattr("ml._imports.redis", None, raising=False)
    monkeypatch.setattr("ml._imports.HAS_REDIS", True, raising=False)
    monkeypatch.setattr(
        "ml.consumers.redis_streams_consumer.IdempotentConsumer.process",
        lambda self, payload: True,
    )

    payloads = build_streaming_test_payloads(parquet_path=tmp_path / "dataset.parquet")
    plan_payload = json.dumps(payloads.plan_message())
    result_payload = json.dumps(payloads.result_message())
    heartbeat_payload = json.dumps(payloads.heartbeat_message())

    batches = [
        [
            (
                "ml-events",
                [
                    ("1-0", {"topic": "events.ml.DATASET_PLANNED.dataset", "payload": plan_payload}),
                    (
                        "2-0",
                        {"topic": "events.ml.MODEL_TRAINING_COMPLETED.dataset", "payload": result_payload},
                    ),
                    (
                        "3-0",
                        {"topic": "events.ml.WORKER_HEARTBEAT.dataset", "payload": heartbeat_payload},
                    ),
                ],
            ),
        ],
    ]

    dummy_module._batches = batches  # type: ignore[attr-defined]

    config = StreamingPersistenceConfig(state_path=str(tmp_path / "state.json"))
    bus_config = MessageBusConfig(
        enabled=True,
        backend="redis",
        redis_url="redis://redis:6379/0",
        redis_stream="ml-events",
    )
    worker = StreamingTrainingPersistenceWorker(config=config, message_bus_config=bus_config)

    processed = worker.poll_once()
    assert processed == 3
    snapshot = worker.service.snapshot()
    assert payloads.plan_event.plan_id in snapshot["plans"]
    assert payloads.plan_event.plan_id in snapshot["results"]
    assert snapshot["heartbeats"]
    assert Path(worker.config.state_path).exists()
    assert worker.service.state_store.get_stream_cursor() == "3-0"
