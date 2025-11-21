from __future__ import annotations

import signal
from pathlib import Path
from typing import Mapping

import pytest

from ml.cli.streaming_training_runner import DatasetSpecification
from ml.cli.streaming_training_runner import FeatureLayout
from ml.cli.streaming_training_runner import RunnerConfig
from ml.cli.streaming_training_runner import StreamingTrainingRunner
from ml.config.events import EventStatus
from ml.config.streaming_pipeline import AzureScheduledEventsConfig
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.config.streaming_pipeline import TrainingOrchestratorConfig
from ml.training.event_driven.azure_events import ScheduledEventNotice
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.teacher.streaming_loader import PhaseOneFeatureSignals
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingSummary
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry


class _StubOrchestrator:
    def __init__(self, inflight: tuple[str, ...]) -> None:
        self._inflight = inflight

    def inflight_plan_ids(self) -> tuple[str, ...]:
        return self._inflight


def _build_runner_config(tmp_path: Path, *, plan_interval: float = 120.0,
                         adaptive_backlog_threshold: int | None = None,
                         adaptive_gpu_threshold: float | None = None,
                         adaptive_cooldown: float = 120.0,
                         adaptive_multiplier: float = 2.0) -> RunnerConfig:
    streaming_cfg = TFTStreamingConfig(
        time_idx_col="timestamp",
        group_id_col="instrument_id",
        target_col="target",
        static_categoricals=(),
        static_reals=(),
        time_varying_known_reals=("known",),
        time_varying_unknown_reals=("unknown",),
        max_encoder_length=10,
        max_prediction_length=2,
        batch_size=4,
    )
    feature_layout = FeatureLayout(
        feature_names=("target",),
        numeric_columns=("target",),
        categorical_columns=(),
        feature_schema={"target": "float"},
        phase_one_signals=PhaseOneFeatureSignals(),
    )
    dataset_spec = DatasetSpecification(
        dataset_id="full_tft_95",
        dataset_dir=tmp_path,
        metadata={},
        report={},
        streaming_config=streaming_cfg,
        feature_layout=feature_layout,
        phase_one_signals=feature_layout.phase_one_signals,
    )
    planner_cfg = DatasetServiceConfig(parquet_root=str(tmp_path))
    worker_cfg = StreamingWorkerConfig()
    orchestrator_cfg = TrainingOrchestratorConfig(
        command_topic="cmd",
        result_topic="result",
        heartbeat_topic="hb",
        adaptive_backlog_threshold=adaptive_backlog_threshold,
        adaptive_gpu_threshold_mb=adaptive_gpu_threshold,
        adaptive_cooldown_seconds=adaptive_cooldown,
        adaptive_interval_multiplier=adaptive_multiplier,
    )
    return RunnerConfig(
        dataset=dataset_spec,
        planner=planner_cfg,
        worker=worker_cfg,
        orchestrator=orchestrator_cfg,
        state_path=tmp_path / "state.json",
        output_dir=tmp_path / "output",
        registry_root=tmp_path / "registry",
        logits_key="logits",
        plan_interval_seconds=plan_interval,
        max_plans=1,
        promotion_threshold=None,
        promotion_command=None,
        promotion_checks=(),
        pipeline_signature="sig",
        pipeline_version="1",
        persist_snapshot=False,
    )


def _make_result_event(max_gpu: float | None) -> TrainingResultEvent:
    summary = TFTStreamingSummary(total_shards=1, total_rows=10, max_shard_rows=10)
    telemetry = StreamingRunTelemetry(
        metadata_summary=summary,
        caps={},
        train=StreamingLoaderTelemetry(
            loader="train",
            total_shards=1,
            selected_shards=1,
            skipped_shards=0,
            total_rows=10,
            selected_rows=10,
            skipped_rows=0,
            total_sequences=5,
            selected_sequences=5,
            skipped_sequences=0,
        ),
        validation=StreamingLoaderTelemetry(
            loader="val",
            total_shards=1,
            selected_shards=1,
            skipped_shards=0,
            total_rows=10,
            selected_rows=10,
            skipped_rows=0,
            total_sequences=5,
            selected_sequences=5,
            skipped_sequences=0,
        ),
        max_gpu_memory_mb=max_gpu,
    )
    return TrainingResultEvent(
        plan_id="plan-001",
        dataset_id="full_tft_95",
        model_id="model",
        telemetry=telemetry,
        artifact_paths={"logits": "logits.npz"},
        metrics={"roc_auc": 0.5},
        status=EventStatus.SUCCESS,
    )


def test_runner_defer_when_backlog_threshold_reached(tmp_path: Path) -> None:
    config = _build_runner_config(tmp_path, adaptive_backlog_threshold=1)
    runner = StreamingTrainingRunner(config)
    orchestrator = _StubOrchestrator(("plan-1",))
    assert runner._should_defer_due_to_backlog(orchestrator) is True


def test_runner_interval_scales_with_gpu_peak(tmp_path: Path) -> None:
    config = _build_runner_config(
        tmp_path,
        plan_interval=60.0,
        adaptive_gpu_threshold=100.0,
        adaptive_cooldown=180.0,
        adaptive_multiplier=3.0,
    )
    runner = StreamingTrainingRunner(config)
    result = _make_result_event(max_gpu=128.0)
    interval = runner._next_plan_interval(result)
    # multiplier yields 180, cooldown also 180 -> expect 180 seconds
    assert interval == pytest.approx(180.0)


def test_runner_interval_unchanged_when_threshold_not_met(tmp_path: Path) -> None:
    config = _build_runner_config(
        tmp_path,
        plan_interval=45.0,
        adaptive_gpu_threshold=200.0,
    )
    runner = StreamingTrainingRunner(config)
    result = _make_result_event(max_gpu=128.0)
    assert runner._next_plan_interval(result) == pytest.approx(45.0)


def test_runner_interval_zero_when_base_zero(tmp_path: Path) -> None:
    config = _build_runner_config(tmp_path, plan_interval=0.0, adaptive_gpu_threshold=100.0)
    runner = StreamingTrainingRunner(config)
    result = _make_result_event(max_gpu=256.0)
    assert runner._next_plan_interval(result) == pytest.approx(0.0)


def test_runner_signal_requests_checkpoint(tmp_path: Path) -> None:
    config = _build_runner_config(tmp_path)
    runner = StreamingTrainingRunner(config)

    class _WorkerStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def save_checkpoint_now(self, *, reason: str, triggered_by_signal: bool) -> None:
            self.calls.append((reason, triggered_by_signal))

    worker = _WorkerStub()
    runner._active_worker = worker  # type: ignore[attr-defined]
    runner._handle_signal(signal.SIGTERM)
    assert runner._stop_requested is True
    assert worker.calls == [("signal:SIGTERM", True)]
    runner._active_worker = None


def test_runner_azure_notice_triggers_checkpoint(tmp_path: Path) -> None:
    config = _build_runner_config(tmp_path)
    config.scheduled_events = AzureScheduledEventsConfig(enabled=True)
    runner = StreamingTrainingRunner(config)

    class _WorkerStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def save_checkpoint_now(self, *, reason: str, triggered_by_signal: bool) -> None:
            self.calls.append((reason, triggered_by_signal))

    worker = _WorkerStub()
    runner._active_worker = worker  # type: ignore[attr-defined]
    notice = ScheduledEventNotice(
        event_id="evt-1234",
        event_type="Preempt",
        event_status="Scheduled",
        resources=("vm-instance",),
        not_before="2024-06-30T12:00:00Z",
    )
    runner._handle_scheduled_event_notice(notice)
    assert runner._stop_requested is True
    assert worker.calls == [("azure:preempt:evt-1234", True)]
    runner._active_worker = None
