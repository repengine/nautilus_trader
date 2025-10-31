"""Optuna-based hyperparameter sweeps for streaming workers."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Protocol

from msgspec.structs import replace as struct_replace

from ml._imports import HAS_OPTUNA
from ml._imports import check_ml_dependencies
from ml._imports import optuna
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.training.event_driven.dataset_service import StreamingDatasetPlanner
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.event_driven.worker import LightningStreamingWorker


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SweepTrialParameters:
    """Hyperparameters sampled for a streaming worker trial."""

    batch_size: int
    hidden_size: int
    lstm_layers: int
    attention_head_size: int
    dropout: float
    learning_rate: float
    optimizer: str
    lr_scheduler: str
    max_epochs: int


@dataclass(slots=True, frozen=True)
class SweepTrialOutcome:
    """Outcome produced by a sweep trial execution."""

    objective: float
    metrics: Mapping[str, float]
    artifacts: Mapping[str, str]
    status: str


@dataclass(slots=True, frozen=True)
class SweepSearchSpace:
    """Parameter ranges leveraged by the sweep runner."""

    batch_sizes: tuple[int, ...] = (64, 96, 128, 192)
    hidden_sizes: tuple[int, ...] = (16, 32, 64, 96)
    lstm_layers: tuple[int, ...] = (1, 2, 3)
    attention_head_sizes: tuple[int, ...] = (2, 4, 8)
    dropouts: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3)
    learning_rate_range: tuple[float, float] = (1e-4, 5e-3)
    optimizers: tuple[str, ...] = ("adam", "adamw")
    lr_schedulers: tuple[str, ...] = ("reduce_on_plateau", "onecycle", "cosine")
    max_epochs: tuple[int, ...] = (1, 2, 3)

    def __post_init__(self) -> None:
        if not self.batch_sizes:
            raise ValueError("batch_sizes search space must not be empty")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes search space must not be empty")
        if not self.dropouts:
            raise ValueError("dropouts search space must not be empty")
        lr_min, lr_max = self.learning_rate_range
        if lr_min <= 0 or lr_max <= 0 or lr_min >= lr_max:
            raise ValueError("learning_rate_range must contain positive bounds with min < max")


class TrialLogger(Protocol):
    """Protocol for persisting sweep trial metadata."""

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
        """Persist trial metadata."""

    def finalize(self, study_summary: Mapping[str, Any]) -> None:
        """Persist study-level metadata at completion."""


class FileTrialLogger(TrialLogger):
    """Persist trial metadata as JSON artefacts on disk."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._record_path = self._output_dir / "trials"
        self._record_path.mkdir(parents=True, exist_ok=True)
        self._summary_path = self._output_dir / "study_summary.json"

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
        payload: dict[str, Any] = {
            "trial_number": trial_number,
            "status": status,
            "objective": objective,
            "parameters": {
                "batch_size": params.batch_size,
                "hidden_size": params.hidden_size,
                "lstm_layers": params.lstm_layers,
                "attention_head_size": params.attention_head_size,
                "dropout": params.dropout,
                "learning_rate": params.learning_rate,
                "optimizer": params.optimizer,
                "lr_scheduler": params.lr_scheduler,
                "max_epochs": params.max_epochs,
            },
            "metrics": dict(metrics),
            "duration_seconds": duration_seconds,
        }
        if extras:
            payload["extras"] = dict(extras)
        trial_path = self._record_path / f"trial_{trial_number:03d}.json"
        trial_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def finalize(self, study_summary: Mapping[str, Any]) -> None:
        self._summary_path.write_text(
            json.dumps(dict(study_summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )


class StreamingTrialRunner(Protocol):
    """Execute a single trial with supplied parameters."""

    def run(self, params: SweepTrialParameters, *, trial_dir: Path) -> SweepTrialOutcome:
        """Execute the trial and return measured metrics."""


class LightningStreamingTrialRunner(StreamingTrialRunner):
    """Execute sweep trials via the Lightning streaming worker."""

    def __init__(
        self,
        *,
        planner: StreamingDatasetPlanner,
        dataset_request: DatasetPlanRequest,
        worker_config: StreamingWorkerConfig,
        output_root: Path,
    ) -> None:
        self._planner = planner
        self._base_request = dataset_request
        self._worker_config = worker_config
        self._output_root = output_root
        self._output_root.mkdir(parents=True, exist_ok=True)

    def run(self, params: SweepTrialParameters, *, trial_dir: Path) -> SweepTrialOutcome:
        streaming_cfg = dataclass_replace(
            self._base_request.streaming_config,
            batch_size=params.batch_size,
        )
        request = DatasetPlanRequest(
            dataset_id=self._base_request.dataset_id,
            streaming_config=streaming_cfg,
            feature_names=self._base_request.feature_names,
            categorical_columns=self._base_request.categorical_columns,
            numeric_columns=self._base_request.numeric_columns,
            phase_one_signals=self._base_request.phase_one_signals,
            parquet_path=self._base_request.parquet_path,
        )
        plan_event = self._planner.plan(request)

        worker_config = struct_replace(
            self._worker_config,
            max_epochs=params.max_epochs,
            hidden_size=params.hidden_size,
            lstm_layers=params.lstm_layers,
            attention_head_size=params.attention_head_size,
            dropout=params.dropout,
            learning_rate=params.learning_rate,
            optimizer=params.optimizer,
            lr_scheduler=params.lr_scheduler,
        )

        trial_dir.mkdir(parents=True, exist_ok=True)
        worker = LightningStreamingWorker(worker_config, output_dir=trial_dir)
        result = worker.run(plan_event)
        metrics = dict(result.metrics)
        artifacts = dict(result.artifact_paths)

        objective_metric = self._objective_from_metrics(metrics)
        if result.status.value.lower() != "success":
            logger.warning(
                "sweep trial produced non-success status",
                extra={"status": result.status.value, "plan_id": result.plan_id},
            )
            objective_metric *= 0.95

        return SweepTrialOutcome(
            objective=objective_metric,
            metrics=metrics,
            artifacts=artifacts,
            status=result.status.value,
        )

    @staticmethod
    def _objective_from_metrics(metrics: Mapping[str, float]) -> float:
        if "roc_auc" in metrics:
            return float(metrics["roc_auc"])
        if "pr_auc" in metrics:
            return float(metrics["pr_auc"])
        raise RuntimeError("Objective metric not present in worker metrics (expected roc_auc or pr_auc)")


class StreamingWorkerStudyRunner:
    """Coordinate an Optuna study for streaming worker hyperparameters."""

    def __init__(
        self,
        *,
        runner: StreamingTrialRunner,
        search_space: SweepSearchSpace | None = None,
        output_dir: Path,
        logger: TrialLogger | None = None,
        study_name: str = "streaming-worker-sweep",
        storage: str | None = None,
        direction: str = "maximize",
    ) -> None:
        self._runner = runner
        self._search_space = search_space or SweepSearchSpace()
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger or FileTrialLogger(self._output_dir)
        self._study_name = study_name
        self._storage = storage
        self._direction = direction

    def run(self, max_trials: int, *, seed: int | None = None) -> Any:
        if not HAS_OPTUNA:
            check_ml_dependencies(["optuna"])
        if optuna is None:  # pragma: no cover - defensive
            raise RuntimeError("optuna import guard failed")
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(
            direction=self._direction,
            sampler=sampler,
            study_name=self._study_name,
            storage=self._storage,
            load_if_exists=False,
        )
        study.optimize(self._objective, n_trials=max_trials)
        summary = self._build_summary(study)
        self._logger.finalize(summary)
        return study

    def _objective(self, trial: Any) -> float:
        params = self._sample_parameters(trial)
        trial_dir = self._output_dir / f"trial_{trial.number:03d}"
        start = time.perf_counter()
        try:
            outcome = self._runner.run(params, trial_dir=trial_dir)
        except Exception as exc:
            duration = time.perf_counter() - start
            logger.error(
                "sweep trial failed",
                extra={"trial_number": trial.number},
                exc_info=True,
            )
            self._logger.record(
                trial.number,
                params=params,
                metrics={},
                objective=None,
                status="failed",
                duration_seconds=duration,
                extras={"error": str(exc)},
            )
            trial.set_user_attr("failure_reason", str(exc))
            raise

        duration = time.perf_counter() - start
        self._logger.record(
            trial.number,
            params=params,
            metrics=outcome.metrics,
            objective=outcome.objective,
            status=outcome.status,
            duration_seconds=duration,
            extras={"artifacts": dict(outcome.artifacts)},
        )
        trial.set_user_attr("metrics", dict(outcome.metrics))
        trial.set_user_attr("status", outcome.status)
        trial.set_user_attr("artifacts", dict(outcome.artifacts))
        return outcome.objective

    def _sample_parameters(self, trial: Any) -> SweepTrialParameters:
        space = self._search_space
        batch_size = trial.suggest_categorical("batch_size", list(space.batch_sizes))
        hidden_size = trial.suggest_categorical("hidden_size", list(space.hidden_sizes))
        lstm_layers = trial.suggest_categorical("lstm_layers", list(space.lstm_layers))
        attention_heads = trial.suggest_categorical(
            "attention_head_size",
            list(space.attention_head_sizes),
        )
        dropout = float(trial.suggest_categorical("dropout", list(space.dropouts)))
        lr_min, lr_max = space.learning_rate_range
        learning_rate = trial.suggest_float(
            "learning_rate",
            lr_min,
            lr_max,
            log=True,
        )
        optimizer = trial.suggest_categorical("optimizer", list(space.optimizers))
        lr_scheduler = trial.suggest_categorical("lr_scheduler", list(space.lr_schedulers))
        max_epochs = trial.suggest_categorical("max_epochs", list(space.max_epochs))
        return SweepTrialParameters(
            batch_size=int(batch_size),
            hidden_size=int(hidden_size),
            lstm_layers=int(lstm_layers),
            attention_head_size=int(attention_heads),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            optimizer=str(optimizer),
            lr_scheduler=str(lr_scheduler),
            max_epochs=int(max_epochs),
        )

    def _build_summary(self, study: Any) -> Mapping[str, Any]:
        summary: dict[str, Any] = {
            "study_name": study.study_name,
            "direction": study.direction.name,
            "best_trial_number": study.best_trial.number if study.best_trial else None,
        }
        trials_payload: list[dict[str, Any]] = []
        if HAS_OPTUNA and optuna is not None:
            try:
                dataframe = study.trials_dataframe()
                trials_payload = json.loads(dataframe.to_json(orient="records"))
            except Exception:
                logger.debug("failed to collect trials dataframe for summary", exc_info=True)
        summary["trials"] = trials_payload
        if study.best_trial:
            summary["best_value"] = study.best_trial.value
            summary["best_params"] = dict(study.best_trial.params)
            summary["best_user_attrs"] = dict(study.best_trial.user_attrs)
            best_metrics = study.best_trial.user_attrs.get("metrics", {})
            if isinstance(best_metrics, Mapping):
                summary["best_metrics"] = dict(best_metrics)
        return summary


def build_trial_runner(
    *,
    dataset_service_config: DatasetServiceConfig,
    dataset_request: DatasetPlanRequest,
    worker_config: StreamingWorkerConfig,
    output_root: Path,
) -> LightningStreamingTrialRunner:
    """Convenience helper to construct a Lightning-backed trial runner."""
    planner = StreamingDatasetPlanner(dataset_service_config)
    return LightningStreamingTrialRunner(
        planner=planner,
        dataset_request=dataset_request,
        worker_config=worker_config,
        output_root=output_root,
    )


__all__ = sorted(
    [
        "LightningStreamingTrialRunner",
        "StreamingTrialRunner",
        "StreamingWorkerStudyRunner",
        "SweepSearchSpace",
        "SweepTrialOutcome",
        "SweepTrialParameters",
        "build_trial_runner",
    ],
)
