from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ml.common.in_memory_bus import InMemoryPublisher
from ml.common.message_topics import build_topic_for_stage
from ml.config.events import EventStatus, Source, Stage
from ml.config.streaming_pipeline import (
    DatasetServiceConfig,
    StreamingWorkerConfig,
    TrainingOrchestratorConfig,
)
from ml.training.event_driven.orchestrator import (
    InMemoryOrchestratorBus,
    InMemoryStreamingOrchestrator,
    PublisherOrchestratorBus,
)
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.event_driven.services import DatasetPlanner
from ml.training.event_driven.services import TrainingHeartbeatEvent
from ml.training.event_driven.services import TrainingResultEvent, TrainingWorker
from ml.training.teacher.streaming_loader import (
    StreamingLimitSummary,
    TFTStreamingConfig,
    TFTStreamingMetadata,
    TFTStreamingSummary,
    TFTShardIndex,
)
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry, StreamingRunTelemetry


def _make_streaming_config() -> TFTStreamingConfig:
    return TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=None,
        num_workers=0,
    )


def _telemetry() -> StreamingRunTelemetry:
    summary = TFTStreamingSummary(total_shards=1, total_rows=4, max_shard_rows=4)
    train = StreamingLoaderTelemetry(
        loader="train",
        total_shards=1,
        selected_shards=1,
        skipped_shards=0,
        total_rows=4,
        selected_rows=4,
        skipped_rows=0,
        total_sequences=2,
        selected_sequences=2,
        skipped_sequences=0,
    )
    val = StreamingLoaderTelemetry(
        loader="validation",
        total_shards=1,
        selected_shards=1,
        skipped_shards=0,
        total_rows=2,
        selected_rows=2,
        skipped_rows=0,
        total_sequences=1,
        selected_sequences=1,
        skipped_sequences=0,
    )
    return StreamingRunTelemetry(
        metadata_summary=summary,
        caps={"max_shards": 1},
        train=train,
        validation=val,
    )


def _request() -> DatasetPlanRequest:
    return DatasetPlanRequest(
        dataset_id="ds",
        streaming_config=_make_streaming_config(),
        feature_names=("feature",),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature",),
        parquet_path=Path("."),  # unused for fixed planner
    )


class _FixedPlanner(DatasetPlanner):
    def __init__(self) -> None:
        metadata = TFTStreamingMetadata(
            shard_indices=(
                TFTShardIndex(
                    shard_id="s0",
                    instrument_id="AAPL",
                    row_start=0,
                    row_end=3,
                    row_count=4,
                    time_start=0,
                    time_end=3,
                ),
            ),
            numeric_stats={},
            categorical_vocab={},
            instrument_row_counts={"AAPL": 4},
        )
        summary = TFTStreamingSummary(total_shards=1, total_rows=4, max_shard_rows=4)
        limits = StreamingLimitSummary(skipped_shards=0, skipped_rows=0, skipped_sequences=0)
        self._event = DatasetPlanEvent(
            plan_id="plan-1",
            dataset_id="ds",
            parquet_path=Path("."),
            metadata=metadata,
            metadata_summary=summary,
            limits=limits,
            streaming_config=_make_streaming_config(),
            caps={"max_shards": 1},
            status=EventStatus.SUCCESS,
        )
        super().__init__(DatasetServiceConfig(parquet_root="."))

    def plan(self, request: DatasetPlanRequest) -> DatasetPlanEvent:
        return self._event


class _StubWorker(TrainingWorker):
    def __init__(self) -> None:
        super().__init__(StreamingWorkerConfig())

    def run(self, plan: DatasetPlanEvent) -> TrainingResultEvent:
        return TrainingResultEvent(
            plan_id=plan.plan_id,
            dataset_id=plan.dataset_id,
            model_id="stub",
            telemetry=_telemetry(),
            artifact_paths={"logits": "/tmp/logits.npz"},
            metrics={"roc_auc": 0.5},
            status=EventStatus.SUCCESS,
        )


def test_orchestrator_enqueue_publishes_plan() -> None:
    planner = _FixedPlanner()
    bus = InMemoryOrchestratorBus()
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=60,
        ),
        planner,
        bus,
    )
    plan = orchestrator.enqueue_training(_request())
    published_plan = bus.plan_events[0]
    assert plan.plan_id == published_plan.event.plan_id
    assert published_plan.topic == build_topic_for_stage(Stage.DATASET_PLANNED, plan.dataset_id)

    heartbeat = TrainingHeartbeatEvent(
        worker_id="w1",
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        progress_pct=50.0,
        rss_mb=128.0,
        shards_processed=1,
        timestamp=datetime.utcnow(),
    )
    orchestrator.handle_heartbeat(heartbeat)
    published_hb = bus.heartbeats[-1]
    assert published_hb.event.plan_id == plan.plan_id
    assert published_hb.topic == build_topic_for_stage(Stage.WORKER_HEARTBEAT, plan.dataset_id)

    result = TrainingResultEvent(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        model_id="teacher",
        telemetry=_telemetry(),
        artifact_paths={"logits": "/tmp/logits.npz"},
        metrics={"roc_auc": 0.6},
        status=EventStatus.SUCCESS,
    )
    orchestrator.mark_result(result)
    published_result = bus.result_events[-1]
    assert published_result.event.plan_id == plan.plan_id
    assert published_result.topic == build_topic_for_stage(Stage.MODEL_TRAINING_COMPLETED, plan.dataset_id)


def test_orchestrator_detects_expired_plan() -> None:
    planner = _FixedPlanner()
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=1,
        ),
        planner,
    )
    plan = orchestrator.enqueue_training(_request())
    heartbeat = TrainingHeartbeatEvent(
        worker_id="w1",
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        progress_pct=0.0,
        rss_mb=0.0,
        shards_processed=0,
        timestamp=datetime.utcnow() - timedelta(seconds=5),
    )
    orchestrator.handle_heartbeat(heartbeat)
    expired = orchestrator.expired_plans()
    assert expired == []
    state = orchestrator._plans[plan.plan_id]
    state.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
    expired = orchestrator.expired_plans()
    assert expired and expired[0].plan_id == plan.plan_id


def test_orchestrator_executes_registered_worker(tmp_path: Path) -> None:
    planner = _FixedPlanner()
    worker = _StubWorker()
    bus = InMemoryOrchestratorBus()
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=60,
        ),
        planner,
        bus,
        worker=worker,
    )
    plan = orchestrator.enqueue_training(_request())
    assert plan.plan_id not in orchestrator.inflight_plan_ids()
    assert bus.result_events[-1].event.plan_id == plan.plan_id


def test_orchestrator_publishes_via_message_bus() -> None:
    publisher = InMemoryPublisher()
    captured: list[tuple[str, dict[str, Any]]] = []
    publisher.subscribe("#", lambda topic, payload: captured.append((topic, payload)))

    planner = _FixedPlanner()
    bus = PublisherOrchestratorBus(publisher, default_source=Source.LIVE)
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=60,
        ),
        planner,
        bus,
    )
    plan_request = _request()
    plan = orchestrator.enqueue_training(plan_request)
    assert captured, "expected plan publication"
    plan_topic, plan_payload = captured[0]
    assert plan_topic == build_topic_for_stage(Stage.DATASET_PLANNED, plan.dataset_id)
    assert plan_payload["payload_type"] == "streaming_plan"
    assert plan_payload["dataset_id"] == plan.dataset_id
    assert plan_payload["plan_id"] == plan.plan_id

    heartbeat = TrainingHeartbeatEvent(
        worker_id="w1",
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        progress_pct=12.5,
        rss_mb=256.0,
        shards_processed=1,
        timestamp=datetime.utcnow(),
    )
    orchestrator.handle_heartbeat(heartbeat)
    assert len(captured) >= 2
    hb_topic, hb_payload = captured[-1]
    assert hb_topic == build_topic_for_stage(Stage.WORKER_HEARTBEAT, plan.dataset_id)
    assert hb_payload["payload_type"] == "streaming_heartbeat"
    assert hb_payload["payload"]["worker_id"] == "w1"

    result = TrainingResultEvent(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        model_id="teacher",
        telemetry=_telemetry(),
        artifact_paths={"logits": "/tmp/logits.npz"},
        metrics={"roc_auc": 0.6},
        status=EventStatus.SUCCESS,
    )
    orchestrator.handle_result(result)
    assert len(captured) >= 3
    result_topic, result_payload = captured[-1]
    assert result_topic == build_topic_for_stage(Stage.MODEL_TRAINING_COMPLETED, plan.dataset_id)
    assert result_payload["payload_type"] == "streaming_result"
    assert result_payload["payload"]["metrics"]["roc_auc"] == 0.6
    assert not bus.failed_events


def test_orchestrator_persists_plan_state(tmp_path: Path) -> None:
    planner = _FixedPlanner()
    state_path = tmp_path / "plans.json"
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=60,
            enable_state_persistence=True,
        ),
        planner,
        state_path=state_path,
    )
    plan = orchestrator.enqueue_training(_request())
    assert state_path.exists()
    snapshot = json.loads(state_path.read_text(encoding="utf-8"))
    assert plan.plan_id in snapshot
    assert snapshot[plan.plan_id]["dataset_id"] == plan.dataset_id


def test_orchestrator_enforces_dataset_backpressure(tmp_path: Path) -> None:
    planner = _FixedPlanner()
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=60,
            dataset_retry_limit=1,
            enable_state_persistence=True,
        ),
        planner,
        state_path=tmp_path / "plans.json",
    )
    orchestrator.enqueue_training(_request())
    with pytest.raises(RuntimeError):
        orchestrator.enqueue_training(_request())


def test_orchestrator_restores_plan_state(tmp_path: Path) -> None:
    planner = _FixedPlanner()
    state_path = tmp_path / "plans.json"
    cfg = TrainingOrchestratorConfig(
        command_topic="",
        result_topic="",
        heartbeat_topic="",
        worker_timeout_seconds=60,
        dataset_retry_limit=1,
        enable_state_persistence=True,
    )
    orchestrator = InMemoryStreamingOrchestrator(cfg, planner, state_path=state_path)
    orchestrator.enqueue_training(_request())

    orchestrator_reloaded = InMemoryStreamingOrchestrator(cfg, planner, state_path=state_path)
    assert orchestrator_reloaded.inflight_plan_ids()
    with pytest.raises(RuntimeError):
        orchestrator_reloaded.enqueue_training(_request())


def test_orchestrator_marks_saturation_and_resets() -> None:
    planner = _FixedPlanner()
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=30,
            saturation_heartbeat_limit=1,
            enable_state_persistence=False,
        ),
        planner,
    )
    plan = orchestrator.enqueue_training(_request())
    heartbeat = TrainingHeartbeatEvent(
        worker_id="w1",
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        progress_pct=10.0,
        rss_mb=64.0,
        shards_processed=1,
        timestamp=datetime.utcnow(),
    )
    orchestrator.handle_heartbeat(heartbeat)
    orchestrator.handle_heartbeat(heartbeat)
    assert orchestrator.saturated_plan_ids() == (plan.plan_id,)

    progress_heartbeat = TrainingHeartbeatEvent(
        worker_id="w1",
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        progress_pct=20.0,
        rss_mb=64.0,
        shards_processed=2,
        timestamp=datetime.utcnow(),
    )
    orchestrator.handle_heartbeat(progress_heartbeat)
    assert orchestrator.saturated_plan_ids() == ()


def test_orchestrator_resume_plan_resets_state(tmp_path: Path) -> None:
    planner = _FixedPlanner()
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=60,
            enable_state_persistence=True,
        ),
        planner,
        state_path=tmp_path / "plans.json",
    )
    plan = orchestrator.enqueue_training(_request())
    state = orchestrator._plans[plan.plan_id]
    state.completed = True
    state.retry_attempts = 3
    state.saturated = True
    orchestrator._persist_plan_states()
    resumed = orchestrator.resume_plan(plan.plan_id)
    assert resumed is True
    resumed_state = orchestrator._plans[plan.plan_id]
    assert resumed_state.completed is False
    assert resumed_state.retry_attempts == 0
    assert resumed_state.saturated is False


def test_orchestrator_clear_backlog_removes_completed(tmp_path: Path) -> None:
    planner = _FixedPlanner()
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=60,
            enable_state_persistence=True,
        ),
        planner,
        state_path=tmp_path / "plans.json",
    )
    first = orchestrator.enqueue_training(_request())
    orchestrator._plans[first.plan_id].completed = True
    template = orchestrator._plans[first.plan_id]
    template_event = template.event
    assert template_event is not None
    second_event = DatasetPlanEvent(
        plan_id="plan-second",
        dataset_id="ds2",
        parquet_path=template_event.parquet_path,
        metadata=template_event.metadata,
        metadata_summary=template_event.metadata_summary,
        limits=template_event.limits,
        streaming_config=template_event.streaming_config,
        caps=template_event.caps,
        status=template_event.status,
    )
    plan_state_cls = type(template)
    orchestrator._plans[second_event.plan_id] = plan_state_cls(
        event=second_event,
        dataset_id=second_event.dataset_id,
    )
    removed = orchestrator.clear_backlog()
    assert removed == (first.plan_id,)
    assert second_event.plan_id in orchestrator._plans


def test_orchestrator_retry_window_delays_requeue() -> None:
    planner = _FixedPlanner()
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=5,
            retry_window_seconds=4,
            dataset_retry_limit=3,
            enable_state_persistence=False,
        ),
        planner,
    )
    plan = orchestrator.enqueue_training(_request())
    state = orchestrator._plans[plan.plan_id]
    state.last_heartbeat = datetime.utcnow() - timedelta(seconds=10)
    expired = orchestrator.expired_plans()
    assert expired == []
    assert state.next_retry_at is not None
    # Still within retry window
    expired = orchestrator.expired_plans()
    assert expired == []
    state.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
    expired = orchestrator.expired_plans()
    assert expired and expired[0].plan_id == plan.plan_id


def test_orchestrator_retry_limit_removes_plan() -> None:
    planner = _FixedPlanner()
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=5,
            retry_window_seconds=1,
            dataset_retry_limit=1,
            enable_state_persistence=False,
        ),
        planner,
    )
    plan = orchestrator.enqueue_training(_request())
    state = orchestrator._plans[plan.plan_id]
    state.last_heartbeat = datetime.utcnow() - timedelta(seconds=10)
    state.next_retry_at = datetime.utcnow() - timedelta(seconds=2)
    expired = orchestrator.expired_plans()
    assert expired and expired[0].plan_id == plan.plan_id
    # Next timeout should remove plan instead of retrying again
    state = orchestrator._plans.get(plan.plan_id)
    if state is not None:
        state.last_heartbeat = datetime.utcnow() - timedelta(seconds=10)
        state.next_retry_at = datetime.utcnow() - timedelta(seconds=2)
    expired = orchestrator.expired_plans()
    assert expired == []
    assert plan.plan_id not in orchestrator._plans
