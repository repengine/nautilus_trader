from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pytest

from ml.config.bus import MessageBusConfig
from ml.config.events import EventStatus
from ml.config.streaming_pipeline import StreamingPersistenceConfig
from ml.consumers.streaming_training import InMemoryStreamingTrainingStateStore
from ml.consumers.streaming_training_worker import StreamingTrainingPersistenceWorker
from ml.training.event_driven.payloads import (
    build_heartbeat_message,
    build_plan_message,
    build_result_message,
)
from ml.training.event_driven.services import (
    DatasetPlanEvent,
    TrainingHeartbeatEvent,
    TrainingResultEvent,
)
from ml.training.teacher.streaming_loader import (
    StreamingLimitSummary,
    TFTShardIndex,
    TFTStreamingConfig,
    TFTStreamingMetadata,
    TFTStreamingSummary,
)
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry, StreamingRunTelemetry


class _DummyConsumer:
    def __init__(self, handler: Any, events: deque[tuple[str, dict[str, Any]]]) -> None:
        self._handler = handler
        self._events = events
        self.polls: int = 0
        self._last_entry_id: str | None = None
        self._cursor_counter = 0

    def poll_once(
        self,
        *,
        count: int,
        block_ms: int,
        last_id: str = "$",
    ) -> int:
        del count, block_ms, last_id
        if not self._events:
            return 0
        topic, payload = self._events.popleft()
        self._handler(topic, payload)
        self.polls += 1
        self._cursor_counter += 1
        self._last_entry_id = f"{self._cursor_counter}-0"
        return 1

    @property
    def last_entry_id(self) -> str | None:
        return self._last_entry_id


def _plan_event(tmp_path: Path) -> DatasetPlanEvent:
    metadata = TFTStreamingMetadata(
        shard_indices=(
            TFTShardIndex(
                shard_id="s0",
                instrument_id="AAPL",
                row_start=0,
                row_end=4,
                row_count=5,
                time_start=1,
                time_end=5,
            ),
        ),
        numeric_stats={},
        categorical_vocab={},
        instrument_row_counts={"AAPL": 5},
    )
    summary = TFTStreamingSummary(total_shards=1, total_rows=5, max_shard_rows=5)
    limits = StreamingLimitSummary(skipped_shards=0, skipped_rows=0, skipped_sequences=0)
    streaming_cfg = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=("feature",),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=16,
        drop_last=False,
        shuffle_shards=False,
        seed=7,
        num_workers=0,
        max_total_rows=100,
        max_total_sequences=200,
        max_shards=3,
    )
    return DatasetPlanEvent(
        plan_id="plan-worker",
        dataset_id="dataset",
        parquet_path=tmp_path / "dataset.parquet",
        metadata=metadata,
        metadata_summary=summary,
        limits=limits,
        streaming_config=streaming_cfg,
        caps={"max_total_rows": 100},
        status=EventStatus.SUCCESS,
    )


def _result_event(plan: DatasetPlanEvent) -> TrainingResultEvent:
    limits = StreamingLimitSummary()
    telemetry = StreamingRunTelemetry(
        metadata_summary=plan.metadata_summary,
        caps=plan.caps,
        train=StreamingLoaderTelemetry.from_metadata(
            "train",
            plan.metadata,
            limits,
            plan.streaming_config,
        ),
        validation=StreamingLoaderTelemetry.from_metadata(
            "validation",
            plan.metadata,
            limits,
            plan.streaming_config,
        ),
        max_gpu_memory_mb=384.0,
    )
    return TrainingResultEvent(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        model_id="model",
        telemetry=telemetry,
        artifact_paths={"logits": "/tmp/logits.npz"},
        metrics={"roc_auc": 0.6},
        status=EventStatus.SUCCESS,
    )


def _heartbeat_event(plan: DatasetPlanEvent) -> TrainingHeartbeatEvent:
    return TrainingHeartbeatEvent(
        worker_id="worker",
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        progress_pct=80.0,
        rss_mb=256.0,
        shards_processed=2,
    )


def test_streaming_persistence_worker_processes_events(tmp_path: Path) -> None:
    plan = _plan_event(tmp_path)
    result = _result_event(plan)
    heartbeat = _heartbeat_event(plan)
    messages = deque(
        [
            ("events.ml.DATASET_PLANNED.dataset", build_plan_message(plan).as_dict()),
            (
                "events.ml.MODEL_TRAINING_COMPLETED.dataset",
                build_result_message(result).as_dict(),
            ),
            (
                "events.ml.WORKER_HEARTBEAT.dataset",
                build_heartbeat_message(heartbeat, dataset_id=plan.dataset_id).as_dict(),
            ),
        ],
    )

    store = InMemoryStreamingTrainingStateStore()
    config = StreamingPersistenceConfig(
        state_path=str(tmp_path / "state.json"),
        batch_size=4,
        block_ms=0,
        poll_interval_seconds=0.0,
    )
    bus_config = MessageBusConfig(
        enabled=True,
        backend="redis",
        redis_url="redis://localhost:6379/0",
        redis_stream="ml-events",
    )

    def factory(service: Any, _config: MessageBusConfig) -> _DummyConsumer:
        return _DummyConsumer(handler=service.handle, events=messages)

    worker = StreamingTrainingPersistenceWorker(
        config=config,
        message_bus_config=bus_config,
        state_store=store,
        consumer_factory=factory,
    )

    assert worker.poll_once() == 1
    assert worker.poll_once() == 1
    assert worker.poll_once() == 1
    assert worker.poll_once() == 0

    snapshot = store.snapshot()
    assert plan.plan_id in snapshot["plans"]
    assert plan.plan_id in snapshot["results"]
    assert any(
        hb["plan_id"] == plan.plan_id for hb in snapshot["heartbeats"].values()
    )
    result_payload = snapshot["results"][plan.plan_id]
    resources_payload = result_payload["telemetry"]["resources"]
    assert resources_payload["max_gpu_memory_mb"] == 384.0


def test_streaming_persistence_worker_disabled(tmp_path: Path) -> None:
    config = StreamingPersistenceConfig(
        enabled=False,
        state_path=str(tmp_path / "state.json"),
    )
    worker = StreamingTrainingPersistenceWorker(
        config=config,
        message_bus_config=MessageBusConfig(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379/0",
            redis_stream="ml-events",
        ),
    )

    assert worker.poll_once() == 0
    worker.run_forever()  # Should exit immediately without raising


def test_streaming_persistence_worker_initializes_observability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records: list[tuple[str, float, dict[str, Any]]] = []

    class DummyObservabilityService:
        def add_metric(
            self,
            *,
            metric_name: str,
            metric_type: str,
            value: float,
            timestamp: int,
            labels: dict[str, Any] | None = None,
        ) -> None:
            records.append((metric_name, value, dict(labels or {})))

    monkeypatch.setattr(
        "ml.observability.service.ObservabilityService",
        DummyObservabilityService,
    )
    config = StreamingPersistenceConfig(state_path=str(tmp_path / "state.json"))
    worker = StreamingTrainingPersistenceWorker(
        config=config,
        message_bus_config=MessageBusConfig(enabled=False, backend="noop"),
        state_store=InMemoryStreamingTrainingStateStore(),
    )

    service = worker.service
    assert service is not None
    sink = worker.observability
    assert sink is not None
    sink.add_metric(
        metric_name="test_metric",
        metric_type="gauge",
        value=2.0,
        timestamp=123,
        labels={"kind": "demo"},
    )
    assert len(records) == 1
    assert records[0][0] == "test_metric"
    assert records[0][2]["kind"] == "demo"


def test_streaming_persistence_worker_persists_stream_cursor(tmp_path: Path) -> None:
    plan = _plan_event(tmp_path)
    messages = deque(
        [
            ("events.ml.DATASET_PLANNED.dataset", build_plan_message(plan).as_dict()),
        ],
    )
    store = InMemoryStreamingTrainingStateStore()
    store.update_stream_cursor("7-0")
    config = StreamingPersistenceConfig(
        state_path=str(tmp_path / "state.json"),
        batch_size=1,
        block_ms=0,
        poll_interval_seconds=0.0,
    )
    bus_config = MessageBusConfig(
        enabled=True,
        backend="redis",
        redis_url="redis://localhost:6379/0",
        redis_stream="ml-events",
    )

    class _CursorAwareConsumer:
        def __init__(self, handler: Any) -> None:
            self._handler = handler
            self._last_entry_id: str | None = None
            self.received_last_ids: list[str] = []
            self._cursor_counter = 10

        def poll_once(self, *, count: int, block_ms: int, last_id: str = "$") -> int:  # noqa: ARG002
            self.received_last_ids.append(last_id)
            if not messages:
                return 0
            topic, payload = messages.popleft()
            self._handler(topic, payload)
            self._cursor_counter += 1
            self._last_entry_id = f"{self._cursor_counter}-0"
            return 1

        @property
        def last_entry_id(self) -> str | None:
            return self._last_entry_id

    constructed: _CursorAwareConsumer | None = None

    def factory(service: Any, _config: MessageBusConfig) -> _CursorAwareConsumer:
        nonlocal constructed
        constructed = _CursorAwareConsumer(service.handle)
        return constructed

    worker = StreamingTrainingPersistenceWorker(
        config=config,
        message_bus_config=bus_config,
        state_store=store,
        consumer_factory=factory,
    )

    assert worker.poll_once() == 1
    assert constructed is not None
    assert constructed.received_last_ids[0] == "7-0"
    assert store.get_stream_cursor() == constructed.last_entry_id
