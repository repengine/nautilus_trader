from __future__ import annotations

from pathlib import Path

from flask import Flask

from ml.config.events import Source
from ml.dashboard.app import create_app
from ml.dashboard.config import DashboardConfig
from ml.dashboard.service import DashboardService
from ml.training.event_driven.payloads import build_heartbeat_message
from ml.training.event_driven.payloads import build_plan_message
from ml.training.event_driven.payloads import build_result_message
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import TrainingHeartbeatEvent
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.teacher.streaming_loader import StreamingLimitSummary
from ml.training.teacher.streaming_loader import TFTShardIndex
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingMetadata
from ml.training.teacher.streaming_loader import TFTStreamingSummary
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry


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
        plan_id="plan-test",
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
        max_gpu_memory_mb=512.0,
    )
    return TrainingResultEvent(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        model_id="model",
        telemetry=telemetry,
        artifact_paths={"logits": "/tmp/logits.npz"},
        metrics={"roc_auc": 0.55},
        status=plan.status,
    )


def _heartbeat_event(plan: DatasetPlanEvent) -> TrainingHeartbeatEvent:
    return TrainingHeartbeatEvent(
        worker_id="worker-1",
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        progress_pct=50.0,
        rss_mb=256.0,
        shards_processed=3,
    )


def test_dashboard_streaming_monitor_tracks_events(tmp_path: Path) -> None:
    cfg = DashboardConfig(
        streaming_state_path=tmp_path / "state.json",
        auth_tokens=(),
        grafana_embed_enabled=False,
    )
    svc = DashboardService.from_config(cfg)

    initial_state = svc.get_streaming_training_state()
    assert initial_state["enabled"] is True
    assert initial_state["plans"] == {}

    plan = _plan_event(tmp_path)
    result = _result_event(plan)
    heartbeat = _heartbeat_event(plan)

    svc._process_streaming_event(
        "events.ml.DATASET_PLANNED.dataset",
        build_plan_message(plan, source=Source.HISTORICAL).as_dict(),
    )
    svc._process_streaming_event(
        "events.ml.MODEL_TRAINING_COMPLETED.dataset",
        build_result_message(result, source=Source.HISTORICAL).as_dict(),
    )
    svc._process_streaming_event(
        "events.ml.WORKER_HEARTBEAT.dataset",
        build_heartbeat_message(
            heartbeat,
            dataset_id=plan.dataset_id,
            source=Source.HISTORICAL,
        ).as_dict(),
    )

    state = svc.get_streaming_training_state()
    assert state["enabled"] is True
    assert plan.plan_id in state["plans"]
    assert plan.plan_id in state["results"]
    assert state["datasets"][plan.dataset_id] == [plan.plan_id]
    assert state["outstanding_plan_ids"] == []
    assert Path(cfg.streaming_state_path).exists()
    result_payload = state["results"][plan.plan_id]
    telemetry_payload = result_payload["telemetry"]
    assert telemetry_payload["resources"]["max_gpu_memory_mb"] == 512.0
    latest_result = state["dataset_details"][plan.dataset_id]["latest_result"]
    assert latest_result is not None
    assert latest_result["resources"]["max_gpu_memory_mb"] == 512.0


def test_streaming_state_endpoint(tmp_path: Path) -> None:
    cfg = DashboardConfig(
        streaming_state_path=tmp_path / "state.json",
        auth_tokens=(),
        grafana_embed_enabled=False,
    )
    app: Flask = create_app(cfg)
    client = app.test_client()
    response = client.get("/api/training/streaming/state")
    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload, dict)
    assert "plans" in payload
    assert "results" in payload
