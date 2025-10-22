from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.config.bus import MessageBusConfig
from ml.consumers.streaming_training_service import StreamingTrainingPersistenceService
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
        plan_id="plan-service",
        dataset_id="dataset",
        parquet_path=tmp_path / "dataset.parquet",
        metadata=metadata,
        metadata_summary=summary,
        limits=limits,
        streaming_config=streaming_cfg,
        caps={"max_total_rows": 100},
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
        max_gpu_memory_mb=256.0,
    )
    return TrainingResultEvent(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        model_id="model",
        telemetry=telemetry,
        artifact_paths={"logits": "/tmp/logits.npz"},
        metrics={"roc_auc": 0.6},
        status=plan.status,
    )


def _heartbeat_event(plan: DatasetPlanEvent) -> TrainingHeartbeatEvent:
    return TrainingHeartbeatEvent(
        worker_id="worker-service",
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        progress_pct=75.0,
        rss_mb=128.0,
        shards_processed=2,
    )


def test_streaming_training_persistence_service_handles_events(tmp_path: Path) -> None:
    service = StreamingTrainingPersistenceService.create(state_path=tmp_path / "state.json")
    plan = _plan_event(tmp_path)
    result = _result_event(plan)
    heartbeat = _heartbeat_event(plan)

    service.handle("events.ml.DATASET_PLANNED.dataset", build_plan_message(plan).as_dict())
    service.handle(
        "events.ml.MODEL_TRAINING_COMPLETED.dataset",
        build_result_message(result).as_dict(),
    )
    service.handle(
        "events.ml.WORKER_HEARTBEAT.dataset",
        build_heartbeat_message(heartbeat, dataset_id=plan.dataset_id).as_dict(),
    )

    snapshot = service.snapshot()
    assert plan.plan_id in snapshot["plans"]
    assert plan.plan_id in snapshot["results"]
    assert Path(service.state_path).exists()
    result_payload = snapshot["results"][plan.plan_id]
    telemetry_payload = result_payload["telemetry"]
    assert telemetry_payload["resources"]["max_gpu_memory_mb"] == 256.0


def test_streaming_training_persistence_service_creates_stream_consumer(tmp_path: Path) -> None:
    service = StreamingTrainingPersistenceService.create(state_path=tmp_path / "state.json")
    cfg = MessageBusConfig(
        enabled=True,
        backend="redis",
        redis_url="redis://localhost:6379/0",
        redis_stream="ml-events",
    )
    consumer = service.create_stream_consumer(cfg)

    handler: Any = consumer._handler  # type: ignore[attr-defined]
    handler(
        "events.ml.DATASET_PLANNED.dataset",
        {"payload_type": "streaming_plan", "correlation_id": "cid1"},
    )
    snapshot = service.snapshot()
    assert isinstance(snapshot, dict)
