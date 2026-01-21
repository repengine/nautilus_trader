from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from ml._imports import HAS_PANDAS, HAS_POLARS, HAS_SKLEARN, HAS_TORCH, check_ml_dependencies, pd, pl
from ml.config.events import EventStatus
from ml.config.streaming_pipeline import (
    CurriculumGuardRule,
    CurriculumScheduleConfig,
    CurriculumStageConfig,
    DatasetServiceConfig,
    EnsembleMemberConfig,
    StreamingEnsembleConfig,
    StreamingWorkerConfig,
    TrainingOrchestratorConfig,
)
from ml.training.event_driven.dataset_service import StreamingDatasetPlanner
from ml.training.event_driven.orchestrator import InMemoryOrchestratorBus, InMemoryStreamingOrchestrator
from ml.training.event_driven.services import DatasetPlanEvent, DatasetPlanRequest
from ml.training.event_driven.worker import LightningStreamingWorker, StreamingCheckpointManager
from ml.training.teacher import streaming_loader as stream
from ml.training.teacher.streaming_loader import TFTStreamingConfig, TFTStreamingSummary
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry, StreamingRunTelemetry
from ml.training.teacher.tft_teacher import StreamingFitResult, StreamingRowMetadata, TFTTeacher

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


def _build_row_metadata(prefix: str, length: int) -> StreamingRowMetadata:
    instruments = np.array([f"{prefix}_instrument"] * length, dtype=np.str_)
    time_indices = np.arange(length, dtype=np.int64)
    row_ids = np.asarray(
        [f"{instrument}::{time_idx}" for instrument, time_idx in zip(instruments, time_indices, strict=False)],
        dtype=np.str_,
    )
    return StreamingRowMetadata(
        row_ids=row_ids,
        instrument_ids=instruments,
        time_indices=time_indices,
    )


def _reverse_metadata(meta: StreamingRowMetadata) -> StreamingRowMetadata:
    order = slice(None, None, -1)
    return StreamingRowMetadata(
        row_ids=meta.row_ids[order].copy(),
        instrument_ids=meta.instrument_ids[order].copy(),
        time_indices=meta.time_indices[order].copy(),
    )


def _save_peer_logits(
    path: Path,
    fit_result: StreamingFitResult,
    *,
    delta: float = 1.0,
    mutate_train: Callable[[StreamingRowMetadata], StreamingRowMetadata] | None = None,
    mutate_val: Callable[[StreamingRowMetadata], StreamingRowMetadata] | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "z_train": fit_result.z_train + delta,
        "z_val": fit_result.z_val + delta,
        "y_val": fit_result.y_val,
    }
    if fit_result.val_returns is not None:
        payload["val_returns"] = fit_result.val_returns
    train_rows = fit_result.train_rows
    val_rows = fit_result.val_rows
    if train_rows is not None:
        if mutate_train is not None:
            train_rows = mutate_train(train_rows)
        payload["train_row_ids"] = train_rows.row_ids
        payload["train_instrument_ids"] = train_rows.instrument_ids
        payload["train_time_indices"] = train_rows.time_indices
    if val_rows is not None:
        if mutate_val is not None:
            val_rows = mutate_val(val_rows)
        payload["val_row_ids"] = val_rows.row_ids
        payload["val_instrument_ids"] = val_rows.instrument_ids
        payload["val_time_indices"] = val_rows.time_indices
    np.savez_compressed(path, **payload)


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
        bootstrap_sample_rows: int | None = None,
        callbacks=None,
        checkpoint_path=None,
    ) -> StreamingFitResult:
        del (
            parquet_path,
            train_loader,
            val_loader,
            full_metadata,
            bootstrap_sample_rows,
            callbacks,
            checkpoint_path,
        )  # unused in stub
        train_sequences = max(1, stream.count_sequences(train_metadata, streaming_config))
        val_sequences = max(2, stream.count_sequences(val_metadata, streaming_config))
        z_train = np.linspace(-0.2, 0.2, num=train_sequences, dtype=np.float64)
        z_val = np.linspace(-1.0, 1.0, num=val_sequences, dtype=np.float64)
        labels = np.tile(np.array([0.0, 1.0], dtype=np.float64), val_sequences // 2 + 1)[:val_sequences]
        train_rows = _build_row_metadata("train", z_train.size)
        val_rows = _build_row_metadata("val", z_val.size)
        return StreamingFitResult(
            z_train=z_train,
            z_val=z_val,
            y_val=labels,
            train_rows=train_rows,
            val_rows=val_rows,
            val_returns=np.linspace(-0.001, 0.001, num=z_val.size, dtype=np.float64),
        )


def _teacher_factory(_plan: DatasetPlanEvent, cfg: TFTStreamingConfig) -> _DeterministicTeacher:
    return _DeterministicTeacher(cfg)


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_worker_rss_guard_reduces_num_workers(tmp_path: Path, monkeypatch: Any) -> None:
    worker_config = StreamingWorkerConfig(
        rss_guard_threshold_mb=1.0,
        rss_guard_max_workers=0,
    )
    worker = LightningStreamingWorker(
        worker_config,
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    streaming_cfg = replace(_build_streaming_config(), num_workers=4)
    monkeypatch.setattr("ml.training.event_driven.worker.current_rss_mb", lambda: 10.0)

    capped = worker._apply_worker_caps(streaming_cfg)

    assert capped.num_workers == 0


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
    validation_diag = result.telemetry.validation_returns
    if validation_diag is not None:
        assert validation_diag.fallback_join is False
        assert validation_diag.mismatch_count == 0


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
def test_lightning_worker_temperature_calibration_metrics(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)

    worker_config = StreamingWorkerConfig(
        enable_temperature_calibration=True,
        temperature_calibration_steps=5,
    )
    worker = LightningStreamingWorker(
        worker_config,
        output_dir=tmp_path / "calibration",
        teacher_factory=_teacher_factory,
    )

    result = worker.run(plan)

    assert "temperature_calibration_log_loss" in result.metrics
    assert "temperature_calibration_ece_20" in result.metrics
    assert "temperature_calibration_brier_score" in result.metrics
    assert "temperature_calibration_log_loss_delta" in result.metrics
    assert "temperature_calibration_ece_20_delta" in result.metrics
    assert "temperature_calibration_brier_score_delta" in result.metrics
    assert result.metrics["temperature_calibration_temperature"] > 0.0


@pytest.mark.skipif(not HAS_TORCH or not HAS_SKLEARN, reason="torch + scikit-learn dependencies required for platt calibration")
def test_lightning_worker_platt_calibration_metrics(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)

    worker_config = StreamingWorkerConfig(enable_platt_calibration=True)
    worker = LightningStreamingWorker(
        worker_config,
        output_dir=tmp_path / "platt",
        teacher_factory=_teacher_factory,
    )

    result = worker.run(plan)

    assert "platt_calibration_log_loss" in result.metrics
    assert "platt_calibration_ece_20" in result.metrics
    assert "platt_calibration_brier_score" in result.metrics
    assert "platt_calibration_log_loss_delta" in result.metrics
    assert "platt_calibration_ece_20_delta" in result.metrics
    assert "platt_calibration_brier_score_delta" in result.metrics


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_SKLEARN,
    reason="torch + scikit-learn dependencies required for isotonic calibration",
)
def test_lightning_worker_isotonic_calibration_metrics(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)

    worker_config = StreamingWorkerConfig(enable_isotonic_calibration=True)
    worker = LightningStreamingWorker(
        worker_config,
        output_dir=tmp_path / "isotonic",
        teacher_factory=_teacher_factory,
    )

    result = worker.run(plan)

    assert "isotonic_calibration_log_loss" in result.metrics
    assert "isotonic_calibration_ece_20" in result.metrics
    assert "isotonic_calibration_brier_score" in result.metrics
    assert "isotonic_calibration_log_loss_delta" in result.metrics
    assert "isotonic_calibration_ece_20_delta" in result.metrics
    assert "isotonic_calibration_brier_score_delta" in result.metrics


@pytest.mark.skipif(not HAS_POLARS, reason="polars dependency required for forward returns")
def test_worker_loads_validation_returns_from_parquet(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_forward.parquet"
    pl.DataFrame(
        {
            "instrument_id": ["IWM", "IWM", "QQQ"],
            "time_index": [0, 1, 0],
            "forward_return": [0.01, -0.02, 0.05],
            "y": [1.0, 0.0, 1.0],
        },
    ).write_parquet(dataset_path)

    val_rows = StreamingRowMetadata(
        row_ids=np.array(["IWM::0", "IWM::1"], dtype=np.str_),
        instrument_ids=np.array(["IWM", "IWM"], dtype=np.str_),
        time_indices=np.array([0, 1], dtype=np.int64),
    )
    fit_result = StreamingFitResult(
        z_train=np.array([0.0], dtype=np.float64),
        z_val=np.array([0.5, -0.5], dtype=np.float64),
        y_val=np.array([1.0, 0.0], dtype=np.float64),
        train_rows=None,
        val_rows=val_rows,
        val_returns=None,
    )

    worker_config = StreamingWorkerConfig(validation_return_column="forward_return")
    worker = LightningStreamingWorker(worker_config, output_dir=tmp_path / "returns")

    metadata = stream.TFTStreamingMetadata(
        shard_indices=(),
        numeric_stats={},
        categorical_vocab={},
        instrument_row_counts={},
    )
    plan = DatasetPlanEvent(
        plan_id="plan_returns",
        dataset_id="dataset_forward",
        parquet_path=dataset_path,
        metadata=metadata,
        metadata_summary=stream.TFTStreamingSummary(total_shards=0, total_rows=0, max_shard_rows=0),
        limits=stream.StreamingLimitSummary(),
        streaming_config=_build_streaming_config(),
        caps={},
    )

    enriched = worker._maybe_attach_validation_returns(plan, fit_result)
    assert enriched.val_returns is not None
    np.testing.assert_allclose(
        enriched.val_returns,
        np.array([0.01, -0.02], dtype=np.float64),
    )
    diag = getattr(worker, "_validation_returns_telemetry")
    assert diag is not None
    assert diag.fallback_join is False
    assert diag.mismatch_count == 0
    assert diag.missing_count == 0


@pytest.mark.skipif(not HAS_POLARS, reason="polars dependency required for forward returns")
def test_worker_loads_validation_returns_from_parquet_in_chunks(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_forward_chunked.parquet"
    pl.DataFrame(
        {
            "instrument_id": ["IWM", "IWM", "QQQ", "QQQ"],
            "time_index": [0, 1, 0, 1],
            "forward_return": [0.01, -0.02, 0.05, 0.03],
            "y": [1.0, 0.0, 1.0, 0.0],
        },
    ).write_parquet(dataset_path)

    val_rows = StreamingRowMetadata(
        row_ids=np.array(["IWM::0", "IWM::1", "QQQ::0", "QQQ::1"], dtype=np.str_),
        instrument_ids=np.array(["IWM", "IWM", "QQQ", "QQQ"], dtype=np.str_),
        time_indices=np.array([0, 1, 0, 1], dtype=np.int64),
    )
    fit_result = StreamingFitResult(
        z_train=np.array([0.0], dtype=np.float64),
        z_val=np.array([0.5, -0.5, 0.4, -0.4], dtype=np.float64),
        y_val=np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        train_rows=None,
        val_rows=val_rows,
        val_returns=None,
    )

    worker_config = StreamingWorkerConfig(
        validation_return_column="forward_return",
        validation_join_chunk_rows=2,
    )
    worker = LightningStreamingWorker(worker_config, output_dir=tmp_path / "returns_chunked")

    metadata = stream.TFTStreamingMetadata(
        shard_indices=(),
        numeric_stats={},
        categorical_vocab={},
        instrument_row_counts={},
    )
    plan = DatasetPlanEvent(
        plan_id="plan_returns_chunked",
        dataset_id="dataset_forward_chunked",
        parquet_path=dataset_path,
        metadata=metadata,
        metadata_summary=stream.TFTStreamingSummary(total_shards=0, total_rows=0, max_shard_rows=0),
        limits=stream.StreamingLimitSummary(),
        streaming_config=_build_streaming_config(),
        caps={},
    )

    enriched = worker._maybe_attach_validation_returns(plan, fit_result)
    assert enriched.val_returns is not None
    np.testing.assert_allclose(
        enriched.val_returns,
        np.array([0.01, -0.02, 0.05, 0.03], dtype=np.float64),
    )
    diag = getattr(worker, "_validation_returns_telemetry")
    assert diag is not None
    assert diag.fallback_join is False
    assert diag.mismatch_count == 0
    assert diag.missing_count == 0


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_worker_defers_when_validation_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan_event = planner.plan(request)
    worker_config = StreamingWorkerConfig()
    worker = LightningStreamingWorker(worker_config, output_dir=tmp_path / "deferred_artifacts")

    def _fake_execute_training_attempt(
        plan: DatasetPlanEvent,
        context: Any,
        *,
        callbacks: Any | None = None,
        checkpoint_path: Path | None = None,
    ) -> StreamingFitResult:
        del callbacks, checkpoint_path  # unused for deterministic stub
        return StreamingFitResult(
            z_train=np.array([0.0], dtype=np.float64),
            z_val=np.array([], dtype=np.float64),
            y_val=np.array([], dtype=np.float64),
            train_rows=StreamingRowMetadata.empty(),
            val_rows=StreamingRowMetadata.empty(),
            val_returns=np.array([], dtype=np.float64),
        )

    monkeypatch.setattr(
        worker,
        "_execute_training_attempt",
        _fake_execute_training_attempt,
        raising=True,
    )
    monkeypatch.setattr(
        worker,
        "_maybe_attach_validation_returns",
        lambda plan, result: result,
        raising=True,
    )
    monkeypatch.setattr(
        worker,
        "_apply_ensemble",
        lambda plan, result: (result, {}, None),
        raising=True,
    )

    result_event = worker.run(plan_event)
    assert result_event.status is EventStatus.DEFERRED
    assert result_event.metrics == {}
    telemetry_caps = result_event.telemetry.caps
    assert telemetry_caps["validation_failure_reason"] == "validation_data_empty"
    assert telemetry_caps["validation_failure_y_val_size"] == 0
    assert telemetry_caps["validation_failure_logit_size"] == 0


@pytest.mark.skipif(not HAS_POLARS, reason="polars dependency required for forward returns")
def test_worker_validation_returns_join_handles_instrument_mismatch(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset_path = tmp_path / "dataset_mismatch.parquet"
    pl.DataFrame(
        {
            "instrument_id": ["VNQ", "VTI"],
            "time_index": [10, 20],
            "forward_return": [0.12, -0.34],
            "y": [1.0, 0.0],
        },
    ).write_parquet(dataset_path)

    val_rows = StreamingRowMetadata(
        row_ids=np.array(["VNQI::10", "VNQI::20"], dtype=np.str_),
        instrument_ids=np.array(["VNQI", "VNQI"], dtype=np.str_),
        time_indices=np.array([10, 20], dtype=np.int64),
    )
    fit_result = StreamingFitResult(
        z_train=np.array([0.0], dtype=np.float64),
        z_val=np.array([0.0, 0.0], dtype=np.float64),
        y_val=np.array([1.0, 0.0], dtype=np.float64),
        train_rows=None,
        val_rows=val_rows,
        val_returns=None,
    )

    worker_config = StreamingWorkerConfig(validation_return_column="forward_return")
    worker = LightningStreamingWorker(worker_config, output_dir=tmp_path / "returns_mismatch")

    metadata = stream.TFTStreamingMetadata(
        shard_indices=(),
        numeric_stats={},
        categorical_vocab={},
        instrument_row_counts={},
    )
    plan = DatasetPlanEvent(
        plan_id="plan_returns_mismatch",
        dataset_id="dataset_forward",
        parquet_path=dataset_path,
        metadata=metadata,
        metadata_summary=stream.TFTStreamingSummary(total_shards=0, total_rows=0, max_shard_rows=0),
        limits=stream.StreamingLimitSummary(),
        streaming_config=_build_streaming_config(),
        caps={},
    )

    enriched = worker._maybe_attach_validation_returns(plan, fit_result)
    assert enriched.val_returns is not None
    np.testing.assert_allclose(
        enriched.val_returns,
        np.zeros(2, dtype=np.float64),
    )
    assert enriched.val_rows is not None
    assert list(enriched.val_rows.instrument_ids.tolist()) == ["VNQI", "VNQI"]
    assert list(enriched.val_rows.row_ids.tolist()) == ["VNQI::10", "VNQI::20"]
    diag = getattr(worker, "_validation_returns_telemetry")
    assert diag is not None
    assert diag.fallback_join is False
    assert diag.mismatch_count == 2
    assert diag.missing_count == 2
    assert any("retaining original instruments" in record.message for record in caplog.records)
    assert not any("validation_returns_instrument_mismatch" in record.message for record in caplog.records)


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
            enable_state_persistence=False,
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


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_curriculum_schedule_adjusts_train_fraction(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)
    curriculum = CurriculumScheduleConfig(
        enabled=True,
        stages=(CurriculumStageConfig(max_total_rows=20, train_fraction=0.5),),
        default_train_fraction=0.7,
    )
    worker_config = StreamingWorkerConfig(
        max_total_rows=20,
        max_total_sequences=20,
        max_shards=4,
        train_fraction=0.9,
        curriculum=curriculum,
    )
    worker = LightningStreamingWorker(
        worker_config,
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    context = worker._prepare_context(plan)
    assert context.train_fraction == pytest.approx(0.5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_curriculum_guard_and_amp_guard_annotations(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)
    guarded_plan = replace(
        plan,
        caps={**plan.caps, "recent_peak_gpu_mb": 2_500.0},
    )
    curriculum = CurriculumScheduleConfig(
        enabled=True,
        stages=(CurriculumStageConfig(max_total_rows=20, train_fraction=0.5, label="phase-a"),),
        default_train_fraction=0.8,
        guards=(
            CurriculumGuardRule(
                stage_label="phase-a",
                max_gpu_mb=2_400.0,
                fallback_train_fraction=0.65,
                reason="gpu guard",
            ),
        ),
    )
    worker_config = StreamingWorkerConfig(
        max_total_rows=20,
        max_total_sequences=20,
        max_shards=4,
        train_fraction=0.9,
        loss_pos_weight=2.5,
        enable_amp=True,
        amp_guard_threshold_mb=2_400.0,
        curriculum=curriculum,
    )
    worker = LightningStreamingWorker(
        worker_config,
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    context = worker._prepare_context(guarded_plan)
    assert context.train_fraction == pytest.approx(0.65)
    assert context.curriculum_stage_label == "phase-a"
    assert context.curriculum_guard_reason == "gpu guard"
    assert context.amp_enabled is False
    assert context.amp_guard_reason is not None
    telemetry_caps = context.telemetry.caps
    assert telemetry_caps["dataset_seed"] == 5
    assert telemetry_caps["worker_curriculum_stage"] == "phase-a"
    assert telemetry_caps["worker_amp_enabled"] is False
    assert telemetry_caps["worker_loss_name"] == "bce"
    assert telemetry_caps["worker_loss_pos_weight"] == pytest.approx(2.5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_prepare_context_uses_presplit_metadata(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)
    train_metadata, val_metadata = stream.split_metadata_by_row_fraction(
        plan.metadata,
        train_fraction=0.5,
    )
    override_plan = replace(
        plan,
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        caps={**plan.caps, "global_train_fraction": 0.4},
    )
    worker_config = StreamingWorkerConfig(
        train_fraction=0.9,
        max_total_rows=None,
        max_total_sequences=None,
        max_shards=None,
    )
    worker = LightningStreamingWorker(worker_config, output_dir=tmp_path / "artifacts")
    context = worker._prepare_context(override_plan)

    assert context.train_fraction == pytest.approx(0.4)
    assert {shard.shard_id for shard in context.train_metadata.shard_indices} == {
        shard.shard_id for shard in train_metadata.shard_indices
    }
    assert {shard.shard_id for shard in context.val_metadata.shard_indices} == {
        shard.shard_id for shard in val_metadata.shard_indices
    }


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_build_teacher_uses_loss_configuration(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)
    worker_config = StreamingWorkerConfig(loss_name="poisson", loss_pos_weight=None)
    worker = LightningStreamingWorker(worker_config, output_dir=tmp_path / "artifacts")
    context = worker._prepare_context(plan)
    teacher = worker._build_teacher(plan, context.worker_streaming_cfg, amp_enabled=False)
    assert isinstance(teacher, TFTTeacher)
    assert teacher.config.loss_name == "poisson"
    assert teacher.config.pos_weight is None

    weighted_config = StreamingWorkerConfig(loss_name="bce", loss_pos_weight=1.75)
    weighted_worker = LightningStreamingWorker(weighted_config, output_dir=tmp_path / "artifacts_weighted")
    weighted_context = weighted_worker._prepare_context(plan)
    weighted_teacher = weighted_worker._build_teacher(plan, weighted_context.worker_streaming_cfg, amp_enabled=False)
    assert isinstance(weighted_teacher, TFTTeacher)
    assert weighted_teacher.config.loss_name == "bce"
    assert weighted_teacher.config.pos_weight == pytest.approx(1.75)


def test_worker_seed_application(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    worker_config = StreamingWorkerConfig(worker_seed=123)
    worker = LightningStreamingWorker(worker_config, output_dir=tmp_path)
    recorded: dict[str, int] = {}

    def _record_random(value: int) -> None:
        recorded["random"] = value

    def _record_numpy(value: int) -> None:
        recorded["numpy"] = value

    monkeypatch.setattr("ml.training.event_driven.worker.random.seed", _record_random)
    monkeypatch.setattr("ml.training.event_driven.worker.np.random.seed", _record_numpy)
    worker._apply_worker_seed()
    assert recorded["random"] == 123
    assert recorded["numpy"] == 123


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_ensemble_blending_merges_logits(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)
    base_config = StreamingWorkerConfig(
        max_total_rows=20,
        max_total_sequences=20,
        max_shards=4,
    )
    base_worker = LightningStreamingWorker(
        base_config,
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    context = base_worker._prepare_context(plan)
    fit_result = base_worker._execute_training_attempt(plan, context)
    peer_path = tmp_path / "peer_logits.npz"
    _save_peer_logits(peer_path, fit_result, delta=1.0)
    ensemble_cfg = StreamingEnsembleConfig(
        enabled=True,
        blend_mode="weighted",
        normalize_weights=True,
        members=(EnsembleMemberConfig(artifact_path=str(peer_path), weight=1.0, required=True),),
    )
    ensemble_config = StreamingWorkerConfig(
        max_total_rows=20,
        max_total_sequences=20,
        max_shards=4,
        ensemble=ensemble_cfg,
    )
    ensemble_worker = LightningStreamingWorker(
        ensemble_config,
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    blended, metrics_info, ensemble_telemetry = ensemble_worker._apply_ensemble(plan, fit_result)
    assert metrics_info.get("ensemble_members_used") == pytest.approx(1.0)
    assert metrics_info.get("ensemble_members_misaligned", 0.0) == pytest.approx(0.0)
    assert blended.z_train.shape == fit_result.z_train.shape
    assert blended.z_val.shape == fit_result.z_val.shape
    assert np.allclose(
        blended.z_train[:2],
        np.mean([fit_result.z_train[:2], (fit_result.z_train + 1.0)[:2]], axis=0),
    )
    assert ensemble_telemetry.members_used == 1
    assert ensemble_telemetry.optional_members_skipped == 0
    assert ensemble_telemetry.misaligned_members == 0
    assert len(ensemble_telemetry.members) == 2
    assert ensemble_telemetry.members[0].artifact_path == "__primary__"
    assert ensemble_telemetry.members[1].used is True


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_ensemble_blending_when_weighted_normalization_enabled_returns_weighted_average(
    tmp_path: Path,
) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)
    base_worker = LightningStreamingWorker(
        StreamingWorkerConfig(max_total_rows=20, max_total_sequences=20, max_shards=4),
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    context = base_worker._prepare_context(plan)
    fit_result = base_worker._execute_training_attempt(plan, context)
    peer_one = tmp_path / "peer_weighted_one.npz"
    peer_two = tmp_path / "peer_weighted_two.npz"
    _save_peer_logits(peer_one, fit_result, delta=1.0)
    _save_peer_logits(peer_two, fit_result, delta=4.0)
    ensemble_cfg = StreamingEnsembleConfig(
        enabled=True,
        blend_mode="weighted",
        normalize_weights=True,
        members=(
            EnsembleMemberConfig(artifact_path=str(peer_one), weight=3.0, required=True),
            EnsembleMemberConfig(artifact_path=str(peer_two), weight=1.0, required=True),
        ),
    )
    ensemble_worker = LightningStreamingWorker(
        StreamingWorkerConfig(
            max_total_rows=20,
            max_total_sequences=20,
            max_shards=4,
            ensemble=ensemble_cfg,
        ),
        output_dir=tmp_path / "artifacts_weighted",
        teacher_factory=_teacher_factory,
    )
    blended, _, _ = ensemble_worker._apply_ensemble(plan, fit_result)
    expected_delta = 7.0 / 5.0
    assert np.allclose(blended.z_train[:2], fit_result.z_train[:2] + expected_delta)


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_optional_ensemble_member_misalignment_skips(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)
    worker = LightningStreamingWorker(
        StreamingWorkerConfig(max_total_rows=20, max_total_sequences=20, max_shards=4),
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    context = worker._prepare_context(plan)
    fit_result = worker._execute_training_attempt(plan, context)
    peer_path = tmp_path / "peer_mismatch.npz"
    _save_peer_logits(
        peer_path,
        fit_result,
        delta=0.5,
        mutate_val=_reverse_metadata,
    )
    ensemble_cfg = StreamingEnsembleConfig(
        enabled=True,
        members=(EnsembleMemberConfig(artifact_path=str(peer_path), weight=1.0, required=False),),
    )
    ensemble_worker = LightningStreamingWorker(
        StreamingWorkerConfig(
            max_total_rows=20,
            max_total_sequences=20,
            max_shards=4,
            ensemble=ensemble_cfg,
        ),
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    blended, metrics, ensemble_telemetry = ensemble_worker._apply_ensemble(plan, fit_result)
    assert np.array_equal(blended.z_train, fit_result.z_train)
    assert np.array_equal(blended.z_val, fit_result.z_val)
    assert metrics.get("ensemble_members_used") == pytest.approx(0.0)
    assert metrics.get("ensemble_optional_members_skipped") == pytest.approx(1.0)
    assert metrics.get("ensemble_members_misaligned") == pytest.approx(1.0)
    assert ensemble_telemetry.members_used == 0
    assert ensemble_telemetry.optional_members_skipped == 1
    assert ensemble_telemetry.misaligned_members == 1
    assert any(member.skipped_reason for member in ensemble_telemetry.members if member.artifact_path != "__primary__")


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_required_ensemble_member_misalignment_raises(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)
    worker = LightningStreamingWorker(
        StreamingWorkerConfig(max_total_rows=20, max_total_sequences=20, max_shards=4),
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    context = worker._prepare_context(plan)
    fit_result = worker._execute_training_attempt(plan, context)
    peer_path = tmp_path / "peer_mismatch_required.npz"
    _save_peer_logits(
        peer_path,
        fit_result,
        delta=0.25,
        mutate_val=_reverse_metadata,
    )
    ensemble_cfg = StreamingEnsembleConfig(
        enabled=True,
        members=(EnsembleMemberConfig(artifact_path=str(peer_path), weight=1.0, required=True),),
    )
    ensemble_worker = LightningStreamingWorker(
        StreamingWorkerConfig(
            max_total_rows=20,
            max_total_sequences=20,
            max_shards=4,
            ensemble=ensemble_cfg,
        ),
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    with pytest.raises(ValueError):
        ensemble_worker._apply_ensemble(plan, fit_result)


@pytest.mark.skipif(not HAS_TORCH, reason="torch dependency required for streaming worker")
def test_worker_emits_economic_and_stability_metrics(tmp_path: Path) -> None:
    planner, request = _planner_and_request(tmp_path)
    plan = planner.plan(request)
    worker = LightningStreamingWorker(
        StreamingWorkerConfig(
            max_total_rows=20,
            max_total_sequences=20,
            max_shards=4,
        ),
        output_dir=tmp_path / "artifacts",
        teacher_factory=_teacher_factory,
    )
    result = worker.run(plan)
    metrics = result.metrics
    assert "economic_slippage_adjusted_sharpe" in metrics
    assert "economic_hit_rate" in metrics
    assert "economic_turnover" in metrics
    assert "economic_max_drawdown" in metrics
    assert "stability_ks_statistic" in metrics
    telemetry = result.telemetry
    assert telemetry.economic is not None
    assert telemetry.economic.hit_rate is not None
    assert telemetry.stability is not None


class _TrainerStub:
    def __init__(self) -> None:
        self.global_step = 1
        self.current_epoch = 0
        self.callback_metrics = {"loss": 0.1}
        self.saved_paths: list[Path] = []

    def save_checkpoint(self, path: str) -> None:
        Path(path).write_text("checkpoint", encoding="utf-8")
        self.saved_paths.append(Path(path))


def test_checkpoint_manager_manual_save_and_resume(tmp_path: Path) -> None:
    manager = StreamingCheckpointManager(
        tmp_path / "manual",
        retention=2,
        interval_seconds=None,
        interval_steps=None,
    )
    assert manager.prepare_plan("plan-manual", "dataset-manual") is None
    trainer = _TrainerStub()
    manager.attach_trainer(trainer, object())
    saved_path = manager.request_manual_save(reason="manual-trigger", triggered_by_signal=True)
    assert saved_path is not None
    latest_ckpt = tmp_path / "manual" / "plan-manual_latest.ckpt"
    metadata_path = tmp_path / "manual" / "plan-manual_latest.json"
    assert latest_ckpt.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["trigger"] == "manual:signal"
    assert payload["reason"] == "manual-trigger"
    assert payload["global_step"] == 1
    manager.detach_trainer()

    resume_record = manager.prepare_plan("plan-manual", "dataset-manual")
    assert resume_record is not None
    assert resume_record.global_step == 1
    assert resume_record.path == latest_ckpt


def test_checkpoint_manager_interval_triggers_save(tmp_path: Path) -> None:
    manager = StreamingCheckpointManager(
        tmp_path / "interval",
        retention=2,
        interval_seconds=None,
        interval_steps=1,
    )
    manager.prepare_plan("plan-interval", "dataset-interval")
    trainer = _TrainerStub()
    manager.attach_trainer(trainer, object())
    manager.maybe_save_interval()
    latest_ckpt = tmp_path / "interval" / "plan-interval_latest.ckpt"
    assert latest_ckpt.exists()
    metadata_path = tmp_path / "interval" / "plan-interval_latest.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["trigger"] == "interval_steps"
    manager.detach_trainer()


def test_checkpoint_manager_uses_checkpoint_key(tmp_path: Path) -> None:
    manager = StreamingCheckpointManager(
        tmp_path / "global",
        retention=1,
        interval_seconds=None,
        interval_steps=None,
    )
    assert manager.prepare_plan(
        "plan-global",
        "dataset-global",
        checkpoint_key="global-run",
    ) is None
    trainer = _TrainerStub()
    manager.attach_trainer(trainer, object())
    manager.request_manual_save(reason="global-run", triggered_by_signal=False)
    manager.detach_trainer()

    latest_ckpt = tmp_path / "global" / "global-run_latest.ckpt"
    metadata_path = tmp_path / "global" / "global-run_latest.json"
    assert latest_ckpt.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_key"] == "global-run"

    resume_record = manager.prepare_plan(
        "plan-global",
        "dataset-global",
        checkpoint_key="global-run",
    )
    assert resume_record is not None
    assert resume_record.checkpoint_key == "global-run"
    assert resume_record.path == latest_ckpt


def test_worker_checkpoint_telemetry_block(tmp_path: Path) -> None:
    worker_config = StreamingWorkerConfig(checkpoint_dir=str(tmp_path / "telemetry"))
    worker = LightningStreamingWorker(worker_config, output_dir=tmp_path / "artifacts")
    manager = worker._checkpoint_manager
    assert manager is not None
    manager.prepare_plan("plan-telemetry", "dataset-telemetry")
    trainer = _TrainerStub()
    manager.attach_trainer(trainer, object())
    manager.request_manual_save(reason="telemetry", triggered_by_signal=False)
    manager.detach_trainer()
    resume_record = manager.prepare_plan("plan-telemetry", "dataset-telemetry")

    summary = TFTStreamingSummary(total_shards=1, total_rows=1, max_shard_rows=1)
    telemetry = StreamingRunTelemetry(
        metadata_summary=summary,
        caps={},
        train=StreamingLoaderTelemetry(
            loader="train",
            total_shards=1,
            selected_shards=1,
            skipped_shards=0,
            total_rows=1,
            selected_rows=1,
            skipped_rows=0,
            total_sequences=1,
            selected_sequences=1,
            skipped_sequences=0,
        ),
        validation=StreamingLoaderTelemetry(
            loader="validation",
            total_shards=1,
            selected_shards=1,
            skipped_shards=0,
            total_rows=1,
            selected_rows=1,
            skipped_rows=0,
            total_sequences=1,
            selected_sequences=1,
            skipped_sequences=0,
        ),
    )
    augmented = worker._attach_checkpoint_telemetry(
        telemetry,
        checkpoint_manager=manager,
        resume_record=resume_record,
        resume_applied=True,
    )
    assert augmented.checkpoint is not None
    assert augmented.checkpoint.resumed is True
    assert augmented.checkpoint.resume_checkpoint_path is not None
    assert augmented.checkpoint.latest_checkpoint_path is not None
