from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping

import pytest

import ml.training.event_driven.sweep as sweep_module
from ml._imports import HAS_OPTUNA
from ml.config.events import EventStatus
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.event_driven.sweep import FileTrialLogger
from ml.training.event_driven.sweep import LightningStreamingTrialRunner
from ml.training.event_driven.sweep import SweepSearchSpace
from ml.training.event_driven.sweep import SweepTrialOutcome
from ml.training.event_driven.sweep import SweepTrialParameters
from ml.training.event_driven.sweep import StreamingTrialRunner
from ml.training.event_driven.sweep import StreamingWorkerStudyRunner
from ml.training.event_driven.sweep import build_trial_runner
from ml.training.teacher.streaming_loader import TFTStreamingConfig


class _StubRunner(StreamingTrialRunner):
    def __init__(self) -> None:
        self._invocations = 0

    def run(self, params: SweepTrialParameters, *, trial_dir: Path) -> SweepTrialOutcome:
        self._invocations += 1
        trial_dir.mkdir(parents=True, exist_ok=True)
        objective = 0.6 + 0.05 * self._invocations
        metrics: Mapping[str, float] = {"roc_auc": objective, "pr_auc": objective - 0.05}
        artifacts = {"logits": str(trial_dir / "logits.npz")}
        return SweepTrialOutcome(
            objective=objective,
            metrics=metrics,
            artifacts=artifacts,
            status="success",
        )


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.summaries: list[dict[str, Any]] = []

    def record(
        self,
        trial_number: int,
        *,
        params: SweepTrialParameters,
        metrics: Mapping[str, float],
        objective: float | None,
        status: str,
        duration_seconds: float,
        extras: Mapping[str, Any] | None = None,
    ) -> None:
        self.records.append(
            {
                "trial_number": trial_number,
                "params": params,
                "metrics": dict(metrics),
                "objective": objective,
                "status": status,
                "duration_seconds": duration_seconds,
                "extras": dict(extras or {}),
            },
        )

    def finalize(self, study_summary: Mapping[str, Any]) -> None:
        self.summaries.append(dict(study_summary))


class _DummyTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.user_attrs: dict[str, Any] = {}

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = value


class _SamplingTrial:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def suggest_categorical(self, name: str, values: list[Any]) -> Any:
        self.calls.append((name, tuple(values)))
        return values[-1]

    def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
        self.calls.append((name, (low, high, log)))
        return (low + high) / 2.0


class _PlannerStub:
    def __init__(self) -> None:
        self.requests: list[DatasetPlanRequest] = []

    def plan(self, request: DatasetPlanRequest) -> object:
        self.requests.append(request)
        return {"plan_id": "plan-1"}


@dataclass(slots=True)
class _WorkerResult:
    metrics: dict[str, float]
    artifact_paths: dict[str, str]
    status: EventStatus
    plan_id: str = "plan-1"


class _WorkerStub:
    created_configs: list[StreamingWorkerConfig] = []
    received_plans: list[object] = []
    result: _WorkerResult = _WorkerResult(
        metrics={"roc_auc": 0.8},
        artifact_paths={"logits": "/tmp/logits.npz"},
        status=EventStatus.SUCCESS,
    )

    def __init__(self, config: StreamingWorkerConfig, *, output_dir: Path) -> None:
        self._config = config
        self._output_dir = output_dir
        type(self).created_configs.append(config)

    def run(self, plan_event: object) -> _WorkerResult:
        type(self).received_plans.append(plan_event)
        return type(self).result


def _streaming_config(*, batch_size: int = 16) -> TFTStreamingConfig:
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
        batch_size=batch_size,
        drop_last=False,
        shuffle_shards=True,
        seed=11,
        num_workers=0,
    )


def _dataset_request(*, batch_size: int = 16) -> DatasetPlanRequest:
    return DatasetPlanRequest(
        dataset_id="dataset",
        streaming_config=_streaming_config(batch_size=batch_size),
        feature_names=("feature",),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature",),
        parquet_path=Path("/tmp/dataset.parquet"),
    )


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna dependency required for sweep tests")
def test_study_runner_records_trials(tmp_path: Path) -> None:
    output_dir = tmp_path / "sweep"
    runner = _StubRunner()
    search_space = SweepSearchSpace(
        batch_sizes=(64,),
        hidden_sizes=(16,),
        lstm_layers=(1,),
        attention_head_sizes=(2,),
        dropouts=(0.1,),
        learning_rate_range=(1e-4, 1e-3),
        optimizers=("adam",),
        lr_schedulers=("reduce_on_plateau",),
        max_epochs=(1,),
    )
    logger = FileTrialLogger(output_dir)
    study_runner = StreamingWorkerStudyRunner(
        runner=runner,
        search_space=search_space,
        output_dir=output_dir,
        logger=logger,
        study_name="unit-test-sweep",
    )

    study = study_runner.run(max_trials=2, seed=123)

    assert study.best_value == pytest.approx(0.7)
    assert (output_dir / "trials" / "trial_000.json").exists()
    assert (output_dir / "trials" / "trial_001.json").exists()
    summary_path = output_dir / "study_summary.json"
    assert summary_path.exists()


def test_sweep_search_space_rejects_empty_sizes() -> None:
    with pytest.raises(ValueError, match="batch_sizes search space must not be empty"):
        SweepSearchSpace(batch_sizes=())

    with pytest.raises(ValueError, match="hidden_sizes search space must not be empty"):
        SweepSearchSpace(hidden_sizes=())

    with pytest.raises(ValueError, match="dropouts search space must not be empty"):
        SweepSearchSpace(dropouts=())


def test_sweep_search_space_rejects_invalid_learning_rate_range() -> None:
    with pytest.raises(ValueError, match="learning_rate_range must contain positive bounds"):
        SweepSearchSpace(learning_rate_range=(1e-3, 1e-3))


def test_file_trial_logger_persists_trial_and_summary_payloads(tmp_path: Path) -> None:
    logger = FileTrialLogger(tmp_path / "sweep")
    params = SweepTrialParameters(
        batch_size=64,
        hidden_size=16,
        lstm_layers=1,
        attention_head_size=2,
        dropout=0.1,
        learning_rate=1e-3,
        optimizer="adam",
        lr_scheduler="reduce_on_plateau",
        max_epochs=1,
    )

    logger.record(
        3,
        params=params,
        metrics={"roc_auc": 0.81},
        objective=0.81,
        status="success",
        duration_seconds=2.5,
        extras={"artifact": "logits.npz"},
    )
    logger.finalize({"best_value": 0.81})

    trial_payload = json.loads((tmp_path / "sweep" / "trials" / "trial_003.json").read_text())
    summary_payload = json.loads((tmp_path / "sweep" / "study_summary.json").read_text())
    assert trial_payload["trial_number"] == 3
    assert trial_payload["extras"]["artifact"] == "logits.npz"
    assert summary_payload["best_value"] == pytest.approx(0.81)


def test_lightning_trial_runner_objective_metric_resolution() -> None:
    assert LightningStreamingTrialRunner._objective_from_metrics({"roc_auc": 0.75, "pr_auc": 0.5}) == pytest.approx(0.75)
    assert LightningStreamingTrialRunner._objective_from_metrics({"pr_auc": 0.66}) == pytest.approx(0.66)
    with pytest.raises(RuntimeError, match="Objective metric not present"):
        LightningStreamingTrialRunner._objective_from_metrics({})


def test_lightning_trial_runner_updates_worker_config_and_penalizes_non_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _PlannerStub()
    request = _dataset_request(batch_size=8)
    worker_config = StreamingWorkerConfig()
    runner = LightningStreamingTrialRunner(
        planner=planner,
        dataset_request=request,
        worker_config=worker_config,
        output_root=tmp_path,
    )
    params = SweepTrialParameters(
        batch_size=96,
        hidden_size=32,
        lstm_layers=2,
        attention_head_size=4,
        dropout=0.15,
        learning_rate=8e-4,
        optimizer="adamw",
        lr_scheduler="cosine",
        max_epochs=2,
    )
    _WorkerStub.created_configs = []
    _WorkerStub.received_plans = []
    _WorkerStub.result = _WorkerResult(
        metrics={"roc_auc": 0.8},
        artifact_paths={"logits": "/tmp/logits.npz"},
        status=EventStatus.FAILED,
        plan_id="plan-1",
    )
    monkeypatch.setattr(sweep_module, "LightningStreamingWorker", _WorkerStub)

    outcome = runner.run(params, trial_dir=tmp_path / "trial_000")

    assert outcome.status == EventStatus.FAILED.value
    assert outcome.objective == pytest.approx(0.8 * 0.95)
    assert planner.requests
    assert planner.requests[0].streaming_config.batch_size == params.batch_size
    assert _WorkerStub.received_plans == [{"plan_id": "plan-1"}]
    worker_cfg = _WorkerStub.created_configs[0]
    assert worker_cfg.hidden_size == params.hidden_size
    assert worker_cfg.lstm_layers == params.lstm_layers
    assert worker_cfg.attention_head_size == params.attention_head_size
    assert worker_cfg.dropout == pytest.approx(params.dropout)
    assert worker_cfg.learning_rate == pytest.approx(params.learning_rate)
    assert worker_cfg.optimizer == params.optimizer
    assert worker_cfg.lr_scheduler == params.lr_scheduler
    assert worker_cfg.max_epochs == params.max_epochs


def test_streaming_worker_study_runner_objective_records_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _StubRunner()
    logger = _RecordingLogger()
    study_runner = StreamingWorkerStudyRunner(
        runner=runner,
        output_dir=tmp_path,
        logger=logger,
        study_name="unit-objective-success",
    )
    params = SweepTrialParameters(
        batch_size=64,
        hidden_size=16,
        lstm_layers=1,
        attention_head_size=2,
        dropout=0.1,
        learning_rate=5e-4,
        optimizer="adam",
        lr_scheduler="reduce_on_plateau",
        max_epochs=1,
    )

    def _sample_parameters(_trial: Any) -> SweepTrialParameters:
        return params

    trial = _DummyTrial(number=4)
    monkeypatch.setattr(study_runner, "_sample_parameters", _sample_parameters)

    objective = study_runner._objective(trial)

    assert objective == pytest.approx(0.65)
    assert len(logger.records) == 1
    assert trial.user_attrs["status"] == "success"
    assert "metrics" in trial.user_attrs
    assert "artifacts" in trial.user_attrs


def test_streaming_worker_study_runner_objective_records_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingRunner(StreamingTrialRunner):
        def run(self, params: SweepTrialParameters, *, trial_dir: Path) -> SweepTrialOutcome:
            raise RuntimeError(f"failed for {params.batch_size} at {trial_dir}")

    logger = _RecordingLogger()
    study_runner = StreamingWorkerStudyRunner(
        runner=_FailingRunner(),
        output_dir=tmp_path,
        logger=logger,
        study_name="unit-objective-failure",
    )
    params = SweepTrialParameters(
        batch_size=32,
        hidden_size=16,
        lstm_layers=1,
        attention_head_size=2,
        dropout=0.1,
        learning_rate=1e-3,
        optimizer="adam",
        lr_scheduler="reduce_on_plateau",
        max_epochs=1,
    )

    def _sample_parameters(_trial: Any) -> SweepTrialParameters:
        return params

    trial = _DummyTrial(number=7)
    monkeypatch.setattr(study_runner, "_sample_parameters", _sample_parameters)

    with pytest.raises(RuntimeError, match="failed for 32"):
        study_runner._objective(trial)

    assert len(logger.records) == 1
    assert logger.records[0]["status"] == "failed"
    assert trial.user_attrs["failure_reason"].startswith("failed for 32")


def test_streaming_worker_study_runner_samples_parameter_values() -> None:
    search_space = SweepSearchSpace(
        batch_sizes=(64, 96),
        hidden_sizes=(16, 32),
        lstm_layers=(1, 2),
        attention_head_sizes=(2, 4),
        dropouts=(0.1, 0.2),
        learning_rate_range=(1e-4, 1e-3),
        optimizers=("adam", "adamw"),
        lr_schedulers=("reduce_on_plateau", "cosine"),
        max_epochs=(1, 2),
    )
    study_runner = StreamingWorkerStudyRunner(
        runner=_StubRunner(),
        search_space=search_space,
        output_dir=Path("."),
    )
    trial = _SamplingTrial()

    params = study_runner._sample_parameters(trial)

    assert params.batch_size == 96
    assert params.hidden_size == 32
    assert params.lstm_layers == 2
    assert params.attention_head_size == 4
    assert params.dropout == pytest.approx(0.2)
    assert params.learning_rate == pytest.approx((1e-4 + 1e-3) / 2.0)
    assert params.optimizer == "adamw"
    assert params.lr_scheduler == "cosine"
    assert params.max_epochs == 2
    assert trial.calls


def test_streaming_worker_study_runner_build_summary_includes_best_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sweep_module, "HAS_OPTUNA", False)
    study_runner = StreamingWorkerStudyRunner(
        runner=_StubRunner(),
        output_dir=tmp_path,
        study_name="summary",
    )

    class _Direction:
        name = "MAXIMIZE"

    class _BestTrial:
        number = 5
        value = 0.91
        params = {"batch_size": 64}
        user_attrs = {"metrics": {"roc_auc": 0.91}, "status": "success"}

    class _Study:
        study_name = "summary"
        direction = _Direction()
        best_trial = _BestTrial()

    summary = study_runner._build_summary(_Study())

    assert summary["best_trial_number"] == 5
    assert summary["best_value"] == pytest.approx(0.91)
    assert summary["best_metrics"]["roc_auc"] == pytest.approx(0.91)
    assert summary["trials"] == []


def test_streaming_worker_study_runner_build_summary_ignores_dataframe_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sweep_module, "HAS_OPTUNA", True)
    monkeypatch.setattr(sweep_module, "optuna", object())
    study_runner = StreamingWorkerStudyRunner(
        runner=_StubRunner(),
        output_dir=tmp_path,
        study_name="summary-failure",
    )

    class _Direction:
        name = "MAXIMIZE"

    class _Study:
        study_name = "summary-failure"
        direction = _Direction()
        best_trial = None

        def trials_dataframe(self) -> object:
            raise RuntimeError("boom")

    summary = study_runner._build_summary(_Study())

    assert summary["best_trial_number"] is None
    assert summary["trials"] == []


def test_build_trial_runner_constructs_lightning_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DatasetPlannerStub:
        def __init__(self, config: DatasetServiceConfig) -> None:
            self.config = config

    monkeypatch.setattr(sweep_module, "StreamingDatasetPlanner", _DatasetPlannerStub)
    runner = build_trial_runner(
        dataset_service_config=DatasetServiceConfig(parquet_root=str(tmp_path)),
        dataset_request=_dataset_request(),
        worker_config=StreamingWorkerConfig(),
        output_root=tmp_path / "runs",
    )

    assert isinstance(runner, LightningStreamingTrialRunner)
    assert isinstance(runner._planner, _DatasetPlannerStub)
