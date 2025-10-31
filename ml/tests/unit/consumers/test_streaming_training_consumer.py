from __future__ import annotations

from datetime import datetime
from pathlib import Path

from typing import Any

from ml.config.events import EventStatus, Source
from ml.common.in_memory_bus import InMemoryPublisher
from ml.consumers.streaming_training import (
    FileBackedStreamingTrainingStateStore,
    InMemoryStreamingTrainingStateStore,
    StreamingHeartbeatRecord,
    StreamingPlanRecord,
    StreamingResultRecord,
    StreamingTrainingConsumer,
    attach_streaming_training_monitor,
)
from ml.training.event_driven.payloads import build_heartbeat_message, build_plan_message, build_result_message
from ml.training.event_driven.services import DatasetPlanEvent, TrainingHeartbeatEvent, TrainingResultEvent
from ml.training.teacher.streaming_loader import (
    StreamingLimitSummary,
    TFTShardIndex,
    TFTStreamingConfig,
    TFTStreamingMetadata,
    TFTStreamingSummary,
)
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry, StreamingRunTelemetry


def _plan_event() -> DatasetPlanEvent:
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
        plan_id="plan-abc",
        dataset_id="dataset",
        parquet_path=Path("/data/dataset"),
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
    )
    return TrainingResultEvent(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        model_id="tft-model",
        telemetry=telemetry,
        artifact_paths={"logits": "/tmp/logits.npz"},
        metrics={"roc_auc": 0.5},
        status=EventStatus.SUCCESS,
    )


def _heartbeat_event(plan: DatasetPlanEvent) -> TrainingHeartbeatEvent:
    return TrainingHeartbeatEvent(
        worker_id="worker-1",
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        progress_pct=42.0,
        rss_mb=256.0,
        shards_processed=3,
        timestamp=datetime.utcnow(),
    )


def test_streaming_training_consumer_tracks_state() -> None:
    plan = _plan_event()
    result = _result_event(plan)
    heartbeat = _heartbeat_event(plan)

    consumer = StreamingTrainingConsumer()
    plan_message = build_plan_message(plan, source=Source.HISTORICAL).as_dict()
    result_message = build_result_message(result, source=Source.HISTORICAL).as_dict()
    heartbeat_message = build_heartbeat_message(heartbeat, dataset_id=plan.dataset_id, source=Source.HISTORICAL).as_dict()

    consumer.handle("events.ml.DATASET_PLANNED.dataset", plan_message)
    stored_plan = consumer.state_store.get_plan(plan.plan_id)
    assert stored_plan is not None
    assert stored_plan.plan_id == plan.plan_id

    # Duplicate delivery should be ignored
    consumer.handle("events.ml.DATASET_PLANNED.dataset", plan_message)
    assert consumer.state_store.outstanding_plan_ids() == (plan.plan_id,)

    consumer.handle("events.ml.MODEL_TRAINING_COMPLETED.dataset", result_message)
    stored_result = consumer.state_store.get_result(plan.plan_id)
    assert stored_result is not None
    assert stored_result.metrics == result.metrics
    updated_plan = consumer.state_store.get_plan(plan.plan_id)
    assert updated_plan is not None
    assert updated_plan.status == result.status
    assert consumer.state_store.outstanding_plan_ids() == ()

    consumer.handle("events.ml.WORKER_HEARTBEAT.dataset", heartbeat_message)
    stored_heartbeat = consumer.state_store.latest_heartbeat(heartbeat.worker_id)
    assert stored_heartbeat is not None
    assert stored_heartbeat.plan_id == heartbeat.plan_id


def test_streaming_training_consumer_ignores_unknown_payload() -> None:
    store = InMemoryStreamingTrainingStateStore()
    consumer = StreamingTrainingConsumer(store)
    consumer.handle("events.ml.UNKNOWN", {"payload_type": "mystery", "correlation_id": "cid-1"})
    assert consumer.state_store.outstanding_plan_ids() == ()


class _ObservabilityRecorder:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, dict[str, Any]]] = []

    def add_metric(
        self,
        *,
        metric_name: str,
        metric_type: str,
        value: float,
        timestamp: int,
        labels: dict[str, Any] | None = None,
    ) -> None:
        self.metrics.append((metric_name, value, dict(labels or {})))


def test_streaming_training_consumer_reports_metrics() -> None:
    plan = _plan_event()
    result = _result_event(plan)
    heartbeat = _heartbeat_event(plan)
    observability = _ObservabilityRecorder()
    consumer = StreamingTrainingConsumer(observability=observability)

    consumer.handle("events.ml.DATASET_PLANNED.dataset", build_plan_message(plan, source=Source.HISTORICAL).as_dict())
    consumer.handle(
        "events.ml.MODEL_TRAINING_COMPLETED.dataset",
        build_result_message(result, source=Source.HISTORICAL).as_dict(),
    )
    consumer.handle(
        "events.ml.WORKER_HEARTBEAT.dataset",
        build_heartbeat_message(heartbeat, dataset_id=plan.dataset_id, source=Source.HISTORICAL).as_dict(),
    )

    metric_names = [name for name, _, _ in observability.metrics]
    assert "ml_tft_streaming_training_backlog" in metric_names
    assert "ml_tft_streaming_worker_progress_pct" in metric_names
    assert "ml_tft_streaming_worker_rss_mb" in metric_names
    assert "ml_tft_streaming_workers_active" in metric_names
    assert "ml_tft_streaming_validation_metric" in metric_names
    progress_labels = next(labels for name, _, labels in observability.metrics if name == "ml_tft_streaming_worker_progress_pct")
    rss_labels = next(labels for name, _, labels in observability.metrics if name == "ml_tft_streaming_worker_rss_mb")
    active_labels = next(labels for name, _, labels in observability.metrics if name == "ml_tft_streaming_workers_active")
    validation_labels = next(labels for name, _, labels in observability.metrics if name == "ml_tft_streaming_validation_metric")
    assert progress_labels["dataset_id"] == plan.dataset_id
    assert rss_labels["dataset_id"] == plan.dataset_id
    assert active_labels["dataset_id"] == plan.dataset_id
    assert validation_labels["dataset_id"] == plan.dataset_id
    assert validation_labels["plan_id"] == plan.plan_id
    assert validation_labels["metric"] == "roc_auc"


def test_streaming_training_state_store_snapshot_roundtrip() -> None:
    plan = _plan_event()
    result = _result_event(plan)
    heartbeat = _heartbeat_event(plan)

    store = InMemoryStreamingTrainingStateStore()
    plan_record = StreamingPlanRecord(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        status=plan.status,
        created_at=plan.created_at,
        caps=plan.caps,
        limits={
            "skipped_shards": plan.limits.skipped_shards,
            "skipped_rows": plan.limits.skipped_rows,
            "skipped_sequences": plan.limits.skipped_sequences,
        },
        metadata_summary={
            "total_shards": plan.metadata_summary.total_shards,
            "total_rows": plan.metadata_summary.total_rows,
            "max_shard_rows": plan.metadata_summary.max_shard_rows,
        },
        streaming_config={
            "batch_size": plan.streaming_config.batch_size,
            "max_shards": plan.streaming_config.max_shards,
        },
        correlation_id="plan-corr",
        topic="events.ml.DATASET_PLANNED.dataset",
    )
    store.record_plan(plan_record)

    result_record = StreamingResultRecord(
        plan_id=result.plan_id,
        dataset_id=result.dataset_id,
        status=result.status,
        completed_at=result.completed_at,
        model_id=result.model_id,
        metrics=result.metrics,
        artifact_paths=result.artifact_paths,
        telemetry=result.telemetry.as_dict(),
        correlation_id="result-corr",
        topic="events.ml.MODEL_TRAINING_COMPLETED.dataset",
    )
    store.record_result(result_record)

    heartbeat_record = StreamingHeartbeatRecord(
        plan_id=heartbeat.plan_id or plan.plan_id,
        dataset_id=heartbeat.dataset_id or plan.dataset_id,
        status=EventStatus.PARTIAL,
        worker_id=heartbeat.worker_id,
        progress_pct=heartbeat.progress_pct,
        rss_mb=heartbeat.rss_mb,
        shards_processed=heartbeat.shards_processed,
        timestamp=heartbeat.timestamp,
        correlation_id="heartbeat-corr",
        topic="events.ml.WORKER_HEARTBEAT.dataset",
    )
    store.record_heartbeat(heartbeat_record)

    snapshot = store.snapshot()
    assert snapshot["stream_cursor"] is None
    restored = InMemoryStreamingTrainingStateStore()
    restored.restore(snapshot)

    assert restored.get_plan(plan.plan_id) is not None
    assert restored.get_result(plan.plan_id) is not None
    assert restored.latest_heartbeat(heartbeat.worker_id) is not None
    restored.update_stream_cursor("5-1")
    assert restored.snapshot()["stream_cursor"] == "5-1"


def test_file_backed_state_store_persists_snapshot(tmp_path: Path) -> None:
    plan = _plan_event()
    result = _result_event(plan)
    heartbeat = _heartbeat_event(plan)
    state_path = tmp_path / "streaming_state.json"

    store = FileBackedStreamingTrainingStateStore(state_path)
    consumer = StreamingTrainingConsumer(state_store=store)

    consumer.handle("events.ml.DATASET_PLANNED.dataset", build_plan_message(plan, source=Source.HISTORICAL).as_dict())
    consumer.handle(
        "events.ml.MODEL_TRAINING_COMPLETED.dataset",
        build_result_message(result, source=Source.HISTORICAL).as_dict(),
    )
    consumer.handle(
        "events.ml.WORKER_HEARTBEAT.dataset",
        build_heartbeat_message(heartbeat, dataset_id=plan.dataset_id, source=Source.HISTORICAL).as_dict(),
    )

    assert state_path.exists()

    restored = FileBackedStreamingTrainingStateStore(state_path)
    assert restored.get_plan(plan.plan_id) is not None
    assert restored.get_result(plan.plan_id) is not None
    assert restored.latest_heartbeat(heartbeat.worker_id) is not None


def test_attach_streaming_training_monitor_subscribes_and_persists(tmp_path: Path) -> None:
    plan = _plan_event()
    result = _result_event(plan)
    heartbeat = _heartbeat_event(plan)
    state_path = tmp_path / "monitor_state.json"
    observability = _ObservabilityRecorder()
    bus = InMemoryPublisher()

    consumer = attach_streaming_training_monitor(
        bus,
        state_path=state_path,
        observability=observability,
    )

    assert isinstance(consumer, StreamingTrainingConsumer)

    bus.publish("events.ml.DATASET_PLANNED.dataset", build_plan_message(plan, source=Source.HISTORICAL).as_dict())
    bus.publish(
        "events.ml.MODEL_TRAINING_COMPLETED.dataset",
        build_result_message(result, source=Source.HISTORICAL).as_dict(),
    )
    bus.publish(
        "events.ml.WORKER_HEARTBEAT.dataset",
        build_heartbeat_message(heartbeat, dataset_id=plan.dataset_id, source=Source.HISTORICAL).as_dict(),
    )

    restored = FileBackedStreamingTrainingStateStore(state_path)
    assert restored.get_result(plan.plan_id) is not None
    assert any(name == "ml_tft_streaming_training_backlog" for name, _, _ in observability.metrics)
