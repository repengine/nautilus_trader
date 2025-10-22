from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml._imports import HAS_PANDAS, HAS_TORCH, check_ml_dependencies, pd
from ml.config.events import EventStatus
from ml.config.streaming_pipeline import (
    DatasetServiceConfig,
    StreamingWorkerConfig,
    TrainingOrchestratorConfig,
)
from ml.training.event_driven.dataset_service import StreamingDatasetPlanner
from ml.training.event_driven.orchestrator import InMemoryOrchestratorBus, InMemoryStreamingOrchestrator
from ml.training.event_driven.services import DatasetPlanEvent, DatasetPlanRequest
from ml.training.event_driven.worker import LightningStreamingWorker
from ml.training.teacher import streaming_loader as stream
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.tft_teacher import StreamingFitResult

try:  # Optional dependency gate for PyTorch Forecasting
    from pytorch_forecasting import TemporalFusionTransformer  # noqa: F401

    HAS_PYTORCH_FORECASTING = True
except Exception:  # pragma: no cover - dependency missing
    HAS_PYTORCH_FORECASTING = False


def _build_streaming_config() -> TFTStreamingConfig:
    return TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=("feature",),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=5,
        num_workers=0,
        max_total_rows=None,
        max_total_sequences=None,
        max_shards=None,
    )


def _planner_and_request(tmp_path: Path) -> tuple[StreamingDatasetPlanner, DatasetPlanRequest]:
    if not HAS_PANDAS:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas runtime dependency unavailable")
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas runtime dependency unavailable")

    dataset_path = tmp_path / "dataset.parquet"
    frame = pd.DataFrame(
        {
            "time_index": np.arange(12, dtype=np.int64),
            "instrument_id": ["AAPL"] * 6 + ["MSFT"] * 6,
            "feature": np.linspace(0.0, 1.0, num=12, dtype=np.float32),
            "y": np.tile(np.array([0.0, 1.0], dtype=np.float32), 6),
        },
    )
    frame.to_parquet(dataset_path, index=False)

    service_config = DatasetServiceConfig(
        parquet_root=str(tmp_path),
        shard_row_budget=6,
    )
    streaming_config = _build_streaming_config()
    request = DatasetPlanRequest(
        dataset_id="dataset.parquet",
        streaming_config=streaming_config,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature",),
    )
    planner = StreamingDatasetPlanner(service_config)
    return planner, request


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
        del parquet_path, train_loader, val_loader, full_metadata  # unused in stub
        train_sequences = max(1, stream.count_sequences(train_metadata, streaming_config))
        val_sequences = max(2, stream.count_sequences(val_metadata, streaming_config))
        z_train = np.linspace(-0.2, 0.2, num=train_sequences, dtype=np.float64)
        z_val = np.linspace(-1.0, 1.0, num=val_sequences, dtype=np.float64)
        labels = np.tile(np.array([0.0, 1.0], dtype=np.float64), val_sequences // 2 + 1)[:val_sequences]
        return StreamingFitResult(z_train=z_train, z_val=z_val, y_val=labels)


def _teacher_factory(_plan: DatasetPlanEvent, cfg: TFTStreamingConfig) -> _DeterministicTeacher:
    return _DeterministicTeacher(cfg)


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_lightning_worker_retries_and_succeeds(tmp_path: Path, monkeypatch: Any, caplog: pytest.LogCaptureFixture) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)
    failures: dict[str, int] = {"remaining": 1}

    class _FlakyTeacher(_DeterministicTeacher):
        def __init__(self, cfg: TFTStreamingConfig, state: dict[str, int]) -> None:
            super().__init__(cfg)
            self._state = state

        def fit_streaming(self, *args: Any, **kwargs: Any) -> StreamingFitResult:
            if self._state["remaining"] > 0:
                self._state["remaining"] -= 1
                raise RuntimeError("transient training error")
            return super().fit_streaming(*args, **kwargs)

    def flaky_factory(_plan: DatasetPlanEvent, cfg: TFTStreamingConfig) -> _FlakyTeacher:
        return _FlakyTeacher(cfg, failures)

    worker_config = StreamingWorkerConfig(
        max_total_rows=20,
        max_total_sequences=20,
        max_shards=4,
        train_fraction=0.6,
        model_id="tft-test-worker",
        max_retry_attempts=2,
        retry_backoff_seconds=0.0,
    )
    monkeypatch.setattr("ml.training.event_driven.worker.time.sleep", lambda _seconds: None)
    worker = LightningStreamingWorker(
        worker_config,
        output_dir=tmp_path / "artifacts",
        teacher_factory=flaky_factory,
    )

    caplog.set_level("WARNING")
    result = worker.run(plan)

    assert failures["remaining"] == 0
    assert result.status in {EventStatus.SUCCESS, EventStatus.PARTIAL}
    assert any("retrying after failure" in record.message for record in caplog.records)


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_lightning_worker_exhausts_retries(tmp_path: Path, monkeypatch: Any, caplog: pytest.LogCaptureFixture) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)

    class _FailingTeacher(_DeterministicTeacher):
        def fit_streaming(self, *args: Any, **kwargs: Any) -> StreamingFitResult:
            raise RuntimeError("irrecoverable failure")

    def failing_factory(_plan: DatasetPlanEvent, cfg: TFTStreamingConfig) -> _FailingTeacher:
        return _FailingTeacher(cfg)

    worker_config = StreamingWorkerConfig(
        max_total_rows=20,
        max_total_sequences=20,
        max_shards=4,
        train_fraction=0.6,
        model_id="tft-test-worker",
        max_retry_attempts=2,
        retry_backoff_seconds=0.0,
    )
    monkeypatch.setattr("ml.training.event_driven.worker.time.sleep", lambda _seconds: None)
    worker = LightningStreamingWorker(
        worker_config,
        output_dir=tmp_path / "artifacts",
        teacher_factory=failing_factory,
    )

    caplog.set_level("ERROR")
    with pytest.raises(RuntimeError):
        worker.run(plan)

    assert any("exhausted retry attempts" in record.message for record in caplog.records)


@pytest.mark.skipif(
    not (HAS_TORCH and HAS_PANDAS and HAS_PYTORCH_FORECASTING),
    reason="Streaming training requires pandas, torch, and pytorch-forecasting",
)
def test_lightning_worker_runs_with_real_teacher(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)
    worker_config = StreamingWorkerConfig(
        max_total_rows=20,
        max_total_sequences=20,
        max_shards=4,
        train_fraction=0.6,
        model_id="tft-real-worker",
        max_retry_attempts=1,
        retry_backoff_seconds=0.0,
    )
    worker = LightningStreamingWorker(worker_config, output_dir=tmp_path / "artifacts")

    result = worker.run(plan)

    artifact_key = worker_config.logits_artifact_key
    artifact_path = Path(result.artifact_paths[artifact_key])
    assert artifact_path.exists()
    assert result.status in {EventStatus.SUCCESS, EventStatus.PARTIAL}
    metric = result.metrics.get(worker_config.validation_metric)
    if metric is not None:
        assert 0.0 <= metric <= 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_lightning_worker_emits_artifact(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)

    worker_config = StreamingWorkerConfig(
        max_total_rows=20,
        max_total_sequences=20,
        max_shards=4,
        train_fraction=0.6,
        model_id="tft-test-worker",
    )
    worker = LightningStreamingWorker(
        worker_config,
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )

    result = worker.run(plan)

    artifact_key = worker_config.logits_artifact_key
    artifact_path = Path(result.artifact_paths[artifact_key])
    assert artifact_path.exists()
    assert result.status in {EventStatus.SUCCESS, EventStatus.PARTIAL}
    assert result.metrics.get(worker_config.validation_metric) is not None
    assert "pr_auc" in result.metrics
    assert "log_loss" in result.metrics
    assert "brier_score" in result.metrics
    assert "calibration_ece_20" in result.metrics
    assert result.telemetry.train.selected_shards > 0
    assert result.dataset_id == plan.dataset_id


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_lightning_worker_integration_with_orchestrator(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    worker_config = StreamingWorkerConfig(
        max_total_rows=20,
        max_total_sequences=20,
        max_shards=4,
        train_fraction=0.6,
        model_id="tft-test-worker",
    )
    worker = LightningStreamingWorker(
        worker_config,
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    bus = InMemoryOrchestratorBus()
    orchestrator = InMemoryStreamingOrchestrator(
        TrainingOrchestratorConfig(
            command_topic="",
            result_topic="",
            heartbeat_topic="",
            worker_timeout_seconds=90,
        ),
        planner,
        bus,
        worker=worker,
    )

    plan = orchestrator.enqueue_training(request)
    assert plan.plan_id not in orchestrator.inflight_plan_ids()
    assert bus.result_events
    latest_result = bus.result_events[-1].event
    assert latest_result.plan_id == plan.plan_id
    assert latest_result.status in {EventStatus.SUCCESS, EventStatus.PARTIAL}
    assert latest_result.metrics.get(worker_config.validation_metric) is not None
    assert latest_result.dataset_id == plan.dataset_id
