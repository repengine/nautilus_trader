from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from prometheus_client import REGISTRY

from ml._imports import HAS_PANDAS, HAS_TORCH, check_ml_dependencies, pd
from ml.common.metrics_bootstrap import HAS_METRICS_BACKEND
from ml.config.events import EventStatus, Source
from ml.config.streaming_pipeline import (
    DatasetServiceConfig,
    StreamingWorkerConfig,
    TrainingOrchestratorConfig,
)
from ml.consumers.streaming_training_service import StreamingTrainingPersistenceService
from ml.training.event_driven.dataset_service import StreamingDatasetPlanner
from ml.training.event_driven.payloads import build_plan_message, build_result_message
from ml.training.event_driven.orchestrator import (
    InMemoryOrchestratorBus,
    InMemoryStreamingOrchestrator,
)
from ml.training.event_driven.services import (
    DatasetPlanEvent,
    DatasetPlanRequest,
    DatasetPlanner,
    TrainingHeartbeatEvent,
    TrainingResultEvent,
    TrainingWorker,
)
from ml.training.event_driven.worker import LightningStreamingWorker
from ml.training.teacher import streaming_loader as stream
from ml.training.teacher.streaming_loader import (
    StreamingLimitSummary,
    TFTShardIndex,
    TFTStreamingConfig,
    TFTStreamingMetadata,
    TFTStreamingSummary,
)
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry, StreamingRunTelemetry
from ml.training.teacher.tft_teacher import StreamingFitResult


class _StaticPlanner(DatasetPlanner):
    def __init__(self, plan_event: DatasetPlanEvent) -> None:
        super().__init__(DatasetServiceConfig(parquet_root="."))
        self._plan_event = plan_event

    def plan(self, request: DatasetPlanRequest) -> DatasetPlanEvent:
        assert request.dataset_id == self._plan_event.dataset_id
        return self._plan_event


def _plan_event(tmp_path: Path) -> DatasetPlanEvent:
    metadata = TFTStreamingMetadata(
        shard_indices=(
            TFTShardIndex(
                shard_id="shard-0",
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
        batch_size=4,
        drop_last=False,
        shuffle_shards=False,
        seed=7,
        num_workers=0,
        max_total_rows=100,
        max_total_sequences=200,
        max_shards=3,
    )
    return DatasetPlanEvent(
        plan_id="plan-integration",
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
        max_gpu_memory_mb=128.0,
    )
    return TrainingResultEvent(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        model_id="model-integration",
        telemetry=telemetry,
        artifact_paths={"logits": "/tmp/logits.npz"},
        metrics={"roc_auc": 0.5},
        status=EventStatus.SUCCESS,
    )


@pytest.mark.integration
def test_plan_worker_round_trip(tmp_path: Path) -> None:
    plan_event = _plan_event(tmp_path)
    planner = _StaticPlanner(plan_event)
    bus = InMemoryOrchestratorBus()

    class _Worker(TrainingWorker):
        def __init__(self) -> None:
            super().__init__(StreamingWorkerConfig())

        def run(self, plan: DatasetPlanEvent) -> TrainingResultEvent:
            return _result_event(plan)

    orchestrator = InMemoryStreamingOrchestrator(
        config=TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
        ),
        planner=planner,
        bus=bus,
        worker=_Worker(),
    )

    plan_request = DatasetPlanRequest(
        dataset_id=plan_event.dataset_id,
        streaming_config=plan_event.streaming_config,
        feature_names=("feature",),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature",),
        parquet_path=plan_event.parquet_path,
    )

    plan = orchestrator.enqueue_training(plan_request)
    assert plan.plan_id == plan_event.plan_id
    assert bus.plan_events
    assert bus.plan_events[0].topic.startswith("ml.data.planned")
    assert bus.plan_events[0].event.dataset_id == plan_event.dataset_id

    assert bus.result_events
    result = bus.result_events[-1]
    assert result.topic.startswith("ml.models.training_completed")
    assert result.event.plan_id == plan_event.plan_id
    assert result.event.telemetry.max_gpu_memory_mb == 128.0
    assert orchestrator.expired_plans() == []
    assert plan.plan_id not in orchestrator.inflight_plan_ids()

    heartbeat = TrainingHeartbeatEvent(
        worker_id="worker",
        plan_id=plan_event.plan_id,
        dataset_id=plan_event.dataset_id,
        progress_pct=100.0,
        rss_mb=64.0,
        shards_processed=1,
    )
    orchestrator.handle_heartbeat(heartbeat)
    assert bus.heartbeats
    assert bus.heartbeats[-1].event.worker_id == "worker"


@pytest.mark.integration
@pytest.mark.skipif(
    not (HAS_TORCH and HAS_PANDAS),
    reason="streaming pipeline requires torch and pandas",
)
def test_streaming_pipeline_records_gpu_telemetry(tmp_path: Path, monkeypatch: Any) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas dependency unavailable at runtime")

    dataset_dir = tmp_path / "full_tft_95"
    dataset_dir.mkdir()
    rows = 96
    instruments = np.tile(["AAPL", "MSFT", "NVDA"], rows // 3)
    frame = pd.DataFrame(
        {
            "time_index": np.arange(rows, dtype=np.int64),
            "instrument_id": instruments,
            "feature": np.linspace(0.0, 1.0, num=rows, dtype=np.float32),
            "macro_vintage_age_minutes": np.linspace(10.0, 75.0, num=rows, dtype=np.float32),
            "y": np.tile(np.array([0.0, 1.0], dtype=np.float32), rows // 2),
        },
    )
    dataset_path = dataset_dir / "dataset_with_vintage_age.parquet"
    frame.to_parquet(dataset_path, index=False)

    service_config = DatasetServiceConfig(
        parquet_root=str(dataset_dir),
        shard_row_budget=32,
        max_total_rows=120_000,
        max_total_sequences=90_000,
        max_shards=32,
    )
    streaming_config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=("feature",),
        time_varying_known_reals=("macro_vintage_age_minutes",),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=48,
        drop_last=False,
        shuffle_shards=False,
        seed=11,
        num_workers=0,
        max_total_rows=120_000,
        max_total_sequences=90_000,
        max_shards=32,
    )
    planner = StreamingDatasetPlanner(service_config)
    plan_request = DatasetPlanRequest(
        dataset_id="full_tft_95",
        streaming_config=streaming_config,
        feature_names=("feature", "macro_vintage_age_minutes"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "macro_vintage_age_minutes"),
        parquet_path=dataset_path,
    )
    plan_event = planner.plan(plan_request)
    assert plan_event.status.value == EventStatus.SUCCESS.value
    assert plan_event.metadata_summary.total_rows == rows

    peak_gpu_mb = 512.0

    class _FakeMonitor:
        def __init__(self, _interval: float) -> None:
            self._peak = peak_gpu_mb

        def start(self) -> None:  # pragma: no cover - simple stub
            return None

        def stop(self) -> None:  # pragma: no cover - simple stub
            return None

        def max_memory_mb(self) -> float:
            return self._peak

    monkeypatch.setattr("ml.training.event_driven.worker.GPUMemoryMonitor", _FakeMonitor)

    class _DeterministicTeacher:
        def __init__(self, cfg: TFTStreamingConfig) -> None:
            self._cfg = cfg

        def fit_streaming(
            self,
            parquet_path: Path,
            train_loader,
            val_loader,
            *,
            train_metadata,
            val_metadata,
            full_metadata,
            streaming_config,
        ) -> StreamingFitResult:
            del parquet_path, train_loader, val_loader, full_metadata, streaming_config
            train_sequences = max(1, stream.count_sequences(train_metadata, self._cfg))
            val_sequences = max(1, stream.count_sequences(val_metadata, self._cfg))
            z_train = np.linspace(-0.5, 0.5, num=train_sequences, dtype=np.float64)
            z_val = np.linspace(-1.0, 1.0, num=val_sequences, dtype=np.float64)
            labels = np.tile(np.array([0.0, 1.0], dtype=np.float64), val_sequences // 2 + 1)[
                :val_sequences
            ]
            return StreamingFitResult(z_train=z_train, z_val=z_val, y_val=labels)

    def _teacher_factory(_plan: DatasetPlanEvent, cfg: TFTStreamingConfig) -> _DeterministicTeacher:
        return _DeterministicTeacher(cfg)

    worker_config = StreamingWorkerConfig(
        max_total_rows=120_000,
        max_total_sequences=90_000,
        max_shards=32,
        max_runtime_seconds=7_200,
        train_fraction=0.8,
        gpu_memory_monitor_interval_seconds=0.05,
        model_id="tft-streaming-test",
    )
    worker = LightningStreamingWorker(
        worker_config,
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )

    result_event = worker.run(plan_event)
    assert result_event.telemetry.max_gpu_memory_mb == pytest.approx(peak_gpu_mb)
    artifact_path = Path(result_event.artifact_paths[worker_config.logits_artifact_key])
    assert artifact_path.exists()

    service = StreamingTrainingPersistenceService.create(state_path=tmp_path / "state.json")
    plan_message = build_plan_message(plan_event, source=Source.HISTORICAL).as_dict()
    service.handle("events.ml.DATASET_PLANNED.full_tft_95", plan_message)
    snapshot = service.snapshot()
    assert plan_event.plan_id in snapshot["plans"]

    result_message = build_result_message(result_event, source=Source.HISTORICAL).as_dict()
    service.handle("events.ml.MODEL_TRAINING_COMPLETED.full_tft_95", result_message)
    snapshot = service.snapshot()
    resources = snapshot["results"][plan_event.plan_id]["telemetry"]["resources"]
    assert resources["max_gpu_memory_mb"] == pytest.approx(peak_gpu_mb)
    assert service.state_store.outstanding_plan_ids() == ()

    if HAS_METRICS_BACKEND:
        metric_value = REGISTRY.get_sample_value(
            "ml_tft_streaming_worker_gpu_peak_mb",
            labels={"dataset_id": plan_event.dataset_id},
        )
        assert metric_value == pytest.approx(peak_gpu_mb)
