"""Lightning-backed streaming training worker for the event-driven pipeline."""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import time
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from ml import _imports as _ml_imports
from ml._imports import HAS_POLARS
from ml._imports import HAS_SKLEARN
from ml._imports import HAS_TORCH
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml._imports import sklearn
from ml.common.gpu_monitor import GPUMemoryMonitor
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram
from ml.config.events import EventStatus
from ml.config.streaming_pipeline import CurriculumGuardContext
from ml.config.streaming_pipeline import CurriculumResolution
from ml.config.streaming_pipeline import EnsembleMemberConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.evaluation.metrics import binary_logloss
from ml.evaluation.metrics import expected_calibration_error
from ml.evaluation.metrics import pr_auc
from ml.evaluation.metrics import roc_auc
from ml.training.event_driven.economic_metrics import compute_economic_and_stability_metrics
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.event_driven.services import TrainingWorker
from ml.training.teacher import streaming_loader as stream
from ml.training.teacher.streaming_loader import StreamingLimitSummary
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingMetadata
from ml.training.teacher.streaming_loader import TFTStreamingSummary
from ml.training.teacher.streaming_loader import summarize_metadata
from ml.training.teacher.streaming_telemetry import StreamingCheckpointTelemetry
from ml.training.teacher.streaming_telemetry import StreamingEconomicTelemetry
from ml.training.teacher.streaming_telemetry import StreamingEnsembleMemberTelemetry
from ml.training.teacher.streaming_telemetry import StreamingEnsembleTelemetry
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry
from ml.training.teacher.streaming_telemetry import StreamingStabilityTelemetry
from ml.training.teacher.streaming_telemetry import ValidationReturnsTelemetry
from ml.training.teacher.tft_teacher import StreamingFitResult
from ml.training.teacher.tft_teacher import StreamingRowMetadata
from ml.training.teacher.tft_teacher import TFTTeacher
from ml.training.teacher.tft_teacher import TFTTeacherConfig


logger = logging.getLogger(__name__)

torch = cast(Any, getattr(_ml_imports, "torch", None))


TeacherFactory = Callable[[DatasetPlanEvent, TFTStreamingConfig], TFTTeacher]

if TYPE_CHECKING:
    import polars as _polars

    PolarsDataFrame = _polars.DataFrame
    from pytorch_lightning.callbacks import Callback as _LightningCallbackBase
else:  # pragma: no cover - used for typing only
    PolarsDataFrame = Any
    try:  # pragma: no cover - optional Lightning dependency
        from lightning.pytorch.callbacks import Callback as _LightningCallbackBase  # type: ignore[assignment]
    except Exception:  # pragma: no cover
        try:
            from pytorch_lightning.callbacks import Callback as _LightningCallbackBase  # type: ignore[assignment]
        except Exception:  # pragma: no cover
            class _FallbackCallback:  # pragma: no cover - minimal stub when Lightning absent
                pass

            _LightningCallbackBase = _FallbackCallback


@dataclass(slots=True, frozen=True)
class _ValidationReturnsDiagnostics:
    fallback_join: bool
    mismatch_count: int
    missing_count: int


@dataclass(slots=True, frozen=True)
class _ValidationMetadataRealignment:
    metadata: StreamingRowMetadata
    corrected_count: int
    missing_count: int


@dataclass(slots=True, frozen=True)
class _ValidationReturnsOutcome:
    returns: npt.NDArray[np.float64]
    diagnostics: _ValidationReturnsDiagnostics


_TRAINING_RUNS_TOTAL = get_counter(
    "ml_tft_streaming_training_runs_total",
    "Total TFT streaming training runs grouped by status.",
    labelnames=("status",),
)
_TRAINING_DURATION_SECONDS = get_histogram(
    "ml_tft_streaming_training_duration_seconds",
    "Duration (seconds) of TFT streaming training runs grouped by status.",
    labelnames=("status",),
    buckets=(30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1200.0, 1800.0, 2400.0),
)
_TRAINING_RETRY_ATTEMPTS = get_counter(
    "ml_tft_streaming_training_retry_attempts_total",
    "Total retry attempts for TFT streaming training grouped by outcome.",
    labelnames=("outcome",),
)
_GPU_PEAK_MEMORY_MB = get_gauge(
    "ml_tft_streaming_worker_gpu_peak_mb",
    "Peak GPU memory usage observed during TFT streaming runs.",
    labelnames=("dataset_id",),
)
_CHECKPOINT_SAVES_TOTAL = get_counter(
    "ml_streaming_checkpoints_total",
    "Streaming checkpoint save attempts grouped by outcome and trigger.",
    labelnames=("outcome", "trigger"),
)
_CHECKPOINT_RESUMES_TOTAL = get_counter(
    "ml_streaming_checkpoint_resumes_total",
    "Streaming checkpoint resume attempts grouped by outcome.",
    labelnames=("outcome",),
)
_CHECKPOINT_SIGNAL_TOTAL = get_counter(
    "ml_streaming_checkpoint_evictions_total",
    "Streaming checkpoints triggered by eviction notices grouped by outcome.",
    labelnames=("outcome",),
)


class ValidationDatasetEmptyError(RuntimeError):
    """Raised when the validation dataset contains no usable rows."""

    def __init__(self, reason: str, *, details: Mapping[str, int | float] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details: dict[str, int | float] = {
            str(key): value for key, value in (details or {}).items()
        }

@dataclass(slots=True)
class _TrainingContext:
    worker_streaming_cfg: TFTStreamingConfig
    limited_metadata: TFTStreamingMetadata
    overall_limits: StreamingLimitSummary
    train_metadata: TFTStreamingMetadata
    train_limits: StreamingLimitSummary
    val_metadata: TFTStreamingMetadata
    val_limits: StreamingLimitSummary
    telemetry: StreamingRunTelemetry
    train_fraction: float
    curriculum_stage_label: str | None = None
    curriculum_guard_reason: str | None = None
    amp_enabled: bool = False
    amp_guard_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _LoadedMemberLogits:
    """Stored ensemble member logits plus optional row-alignment metadata."""

    z_train: npt.NDArray[np.float64]
    z_val: npt.NDArray[np.float64]
    train_rows: StreamingRowMetadata | None
    val_rows: StreamingRowMetadata | None


@dataclass(frozen=True, slots=True)
class _CheckpointRecord:
    """Persisted checkpoint metadata for resumable streaming training."""

    plan_id: str
    dataset_id: str
    path: Path
    epoch: int
    global_step: int
    saved_at: str
    reason: str | None
    trigger: str
    metrics: Mapping[str, float]


class StreamingCheckpointManager:
    """Coordinate periodic/manual checkpoint persistence for streaming workers."""

    def __init__(
        self,
        directory: Path,
        *,
        retention: int,
        interval_seconds: float | None,
        interval_steps: int | None,
    ) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._retention = max(1, int(retention))
        self._interval_seconds = float(interval_seconds) if interval_seconds is not None else None
        self._interval_steps = int(interval_steps) if interval_steps is not None else None
        self._active_plan_id: str | None = None
        self._active_dataset_id: str | None = None
        self._trainer: Any | None = None
        self._pl_module: Any | None = None
        self._resume_record: _CheckpointRecord | None = None
        self._latest_record: _CheckpointRecord | None = None
        self._last_checkpoint_time: float | None = None
        self._last_checkpoint_step: int | None = None
        self._pending_manual_requests: list[tuple[str, bool]] = []

    @property
    def resume_record(self) -> _CheckpointRecord | None:
        """Return checkpoint metadata discovered prior to training start."""
        return self._resume_record

    @property
    def latest_record(self) -> _CheckpointRecord | None:
        """Return metadata for the most recent checkpoint written during this plan."""
        return self._latest_record

    def prepare_plan(self, plan_id: str, dataset_id: str) -> _CheckpointRecord | None:
        """Reset manager state for a new plan and return any resumable checkpoint."""
        self._active_plan_id = plan_id
        self._active_dataset_id = dataset_id
        self._trainer = None
        self._pl_module = None
        self._last_checkpoint_time = None
        self._last_checkpoint_step = None
        self._pending_manual_requests.clear()
        self._resume_record = self._load_latest(plan_id)
        self._latest_record = self._resume_record
        return self._resume_record

    def refresh_latest(self) -> _CheckpointRecord | None:
        """Reload checkpoint metadata from disk for the active plan."""
        if self._active_plan_id is None:
            return None
        self._latest_record = self._load_latest(self._active_plan_id)
        return self._latest_record

    def create_callbacks(self) -> tuple[StreamingCheckpointCallback, ...]:
        """Return Lightning callbacks that attach the manager to trainer lifecycle."""
        return (StreamingCheckpointCallback(self),)

    def attach_trainer(self, trainer: Any, pl_module: Any) -> None:
        """Capture trainer context so manual saves can execute immediately."""
        self._trainer = trainer
        self._pl_module = pl_module
        self._process_pending_requests()

    def detach_trainer(self) -> None:
        """Release trainer references after fit completes."""
        self._trainer = None
        self._pl_module = None

    def maybe_save_interval(self) -> None:
        """Persist checkpoint when cadence thresholds are satisfied."""
        trainer = self._trainer
        if trainer is None or self._active_plan_id is None:
            return
        global_step = int(getattr(trainer, "global_step", 0))
        if global_step <= 0:
            self._process_pending_requests()
            return
        now = time.monotonic()
        trigger: str | None = None
        if self._interval_steps is not None:
            last_step = self._last_checkpoint_step
            if last_step is None or (global_step - last_step) >= self._interval_steps:
                trigger = "interval_steps"
        if trigger is None and self._interval_seconds is not None:
            last_time = self._last_checkpoint_time
            if last_time is None or (now - last_time) >= self._interval_seconds:
                trigger = "interval_seconds"
        if trigger is not None:
            self._perform_save(
                trainer,
                reason=f"{trigger} checkpoint",
                trigger=trigger,
                triggered_by_signal=False,
                global_step=global_step,
                timestamp=now,
            )
        self._process_pending_requests()

    def request_manual_save(self, reason: str, *, triggered_by_signal: bool) -> Path | None:
        """Persist a checkpoint immediately or schedule it once the trainer is ready."""
        if self._trainer is None or self._active_plan_id is None:
            self._pending_manual_requests.append((reason, triggered_by_signal))
            return None
        return self._perform_save(
            self._trainer,
            reason=reason,
            trigger="manual",
            triggered_by_signal=triggered_by_signal,
            force=True,
        )

    def _process_pending_requests(self) -> None:
        if not self._pending_manual_requests or self._trainer is None or self._active_plan_id is None:
            return
        requests = list(self._pending_manual_requests)
        self._pending_manual_requests.clear()
        for reason, signal_flag in requests:
            self._perform_save(
                self._trainer,
                reason=reason,
                trigger="manual",
                triggered_by_signal=signal_flag,
                force=True,
            )

    def _perform_save(
        self,
        trainer: Any,
        *,
        reason: str,
        trigger: str,
        triggered_by_signal: bool,
        force: bool = False,
        global_step: int | None = None,
        timestamp: float | None = None,
    ) -> Path | None:
        if self._active_plan_id is None or self._active_dataset_id is None:
            return None
        step_value = int(global_step if global_step is not None else getattr(trainer, "global_step", 0))
        if step_value <= 0 and not force:
            return None
        epoch_value = int(getattr(trainer, "current_epoch", 0))
        save_time = float(timestamp if timestamp is not None else time.monotonic())
        utc_now = datetime.utcnow().replace(microsecond=0)
        timestamp_label = utc_now.strftime("%Y%m%dT%H%M%SZ")
        unique_name = f"{self._active_plan_id}_{timestamp_label}_step{step_value}.ckpt"
        temp_path = self._directory / unique_name
        try:
            trainer.save_checkpoint(str(temp_path))
        except Exception:
            logger.error(
                "checkpoint_save_failed",
                extra={
                    "plan_id": self._active_plan_id,
                    "dataset_id": self._active_dataset_id,
                    "checkpoint_path": str(temp_path),
                    "trigger": trigger,
                    "reason": reason,
                    "signal": triggered_by_signal,
                },
                exc_info=True,
            )
            _CHECKPOINT_SAVES_TOTAL.labels(outcome="failed", trigger=trigger).inc()
            if triggered_by_signal:
                _CHECKPOINT_SIGNAL_TOTAL.labels(outcome="failed").inc()
            return None

        latest_path = self._directory / f"{self._active_plan_id}_latest.ckpt"
        if latest_path.exists():
            if self._retention >= 2:
                previous_path = self._directory / f"{self._active_plan_id}_previous.ckpt"
                try:
                    latest_path.replace(previous_path)
                except Exception:
                    logger.debug(
                        "checkpoint_previous_rotate_failed",
                        extra={
                            "plan_id": self._active_plan_id,
                            "previous_path": str(latest_path),
                            "target_path": str(previous_path),
                        },
                        exc_info=True,
                    )
            else:
                try:
                    latest_path.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.debug(
                        "checkpoint_latest_unlink_failed",
                        extra={"plan_id": self._active_plan_id, "path": str(latest_path)},
                        exc_info=True,
                    )
        try:
            temp_path.replace(latest_path)
        except Exception:
            logger.error(
                "checkpoint_promote_failed",
                extra={
                    "plan_id": self._active_plan_id,
                    "dataset_id": self._active_dataset_id,
                    "source": str(temp_path),
                    "target": str(latest_path),
                },
                exc_info=True,
            )
            _CHECKPOINT_SAVES_TOTAL.labels(outcome="failed", trigger=trigger).inc()
            if triggered_by_signal:
                _CHECKPOINT_SIGNAL_TOTAL.labels(outcome="failed").inc()
            return None

        if self._retention > 2:
            archive_name = self._directory / f"{self._active_plan_id}_{timestamp_label}_archive.ckpt"
            try:
                shutil.copy2(latest_path, archive_name)
            except Exception:
                logger.debug(
                    "checkpoint_archive_copy_failed",
                    extra={
                        "plan_id": self._active_plan_id,
                        "source": str(latest_path),
                        "archive": str(archive_name),
                    },
                    exc_info=True,
                )
            self._prune_archives(self._active_plan_id)

        self._last_checkpoint_time = save_time
        self._last_checkpoint_step = step_value
        metrics = self._extract_metrics(trainer)
        record = _CheckpointRecord(
            plan_id=self._active_plan_id,
            dataset_id=self._active_dataset_id,
            path=latest_path,
            epoch=epoch_value,
            global_step=step_value,
            saved_at=utc_now.isoformat() + "Z",
            reason=reason,
            trigger=trigger if not triggered_by_signal else f"{trigger}:signal",
            metrics=metrics,
        )
        self._latest_record = record
        self._write_metadata(record)
        _CHECKPOINT_SAVES_TOTAL.labels(outcome="success", trigger=trigger).inc()
        if triggered_by_signal:
            _CHECKPOINT_SIGNAL_TOTAL.labels(outcome="success").inc()
        logger.info(
            "checkpoint_saved",
            extra={
                "plan_id": record.plan_id,
                "dataset_id": record.dataset_id,
                "checkpoint_path": str(record.path),
                "epoch": record.epoch,
                "global_step": record.global_step,
                "trigger": record.trigger,
            },
        )
        return record.path

    def _prune_archives(self, plan_id: str) -> None:
        allowed_archives = max(0, self._retention - 2)
        archive_candidates = sorted(
            [
                path
                for path in self._directory.glob(f"{plan_id}_*_archive.ckpt")
                if path.is_file()
            ],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in archive_candidates[allowed_archives:]:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except Exception:
                logger.debug(
                    "checkpoint_archive_prune_failed",
                    extra={"plan_id": plan_id, "path": str(path)},
                    exc_info=True,
                )

    def _write_metadata(self, record: _CheckpointRecord) -> None:
        payload = {
            "plan_id": record.plan_id,
            "dataset_id": record.dataset_id,
            "checkpoint_path": str(record.path),
            "epoch": record.epoch,
            "global_step": record.global_step,
            "saved_at": record.saved_at,
            "reason": record.reason,
            "trigger": record.trigger,
            "metrics": dict(record.metrics),
        }
        metadata_path = self._directory / f"{record.plan_id}_latest.json"
        try:
            metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            logger.error(
                "checkpoint_metadata_write_failed",
                extra={"plan_id": record.plan_id, "path": str(metadata_path)},
                exc_info=True,
            )

    def _load_latest(self, plan_id: str) -> _CheckpointRecord | None:
        metadata_path = self._directory / f"{plan_id}_latest.json"
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug(
                "checkpoint_metadata_parse_failed",
                extra={"plan_id": plan_id, "path": str(metadata_path)},
                exc_info=True,
            )
            return None
        checkpoint_path = Path(str(payload.get("checkpoint_path", ""))).expanduser()
        if not checkpoint_path.exists():
            logger.warning(
                "checkpoint_metadata_missing_file",
                extra={"plan_id": plan_id, "checkpoint_path": str(checkpoint_path)},
            )
            return None
        metrics_payload = payload.get("metrics", {})
        metrics: dict[str, float] = {}
        if isinstance(metrics_payload, Mapping):
            for key, value in metrics_payload.items():
                numeric = _coerce_float(value)
                if numeric is not None:
                    metrics[str(key)] = numeric
        return _CheckpointRecord(
            plan_id=str(payload.get("plan_id", plan_id)),
            dataset_id=str(payload.get("dataset_id", "")),
            path=checkpoint_path,
            epoch=int(payload.get("epoch", 0)),
            global_step=int(payload.get("global_step", 0)),
            saved_at=str(payload.get("saved_at", "")),
            reason=str(payload.get("reason")) if payload.get("reason") is not None else None,
            trigger=str(payload.get("trigger", "manual")),
            metrics=metrics,
        )

    @staticmethod
    def _extract_metrics(trainer: Any) -> dict[str, float]:
        metrics: dict[str, float] = {}
        raw_metrics = getattr(trainer, "callback_metrics", None)
        if not isinstance(raw_metrics, Mapping):
            return metrics
        for name, value in raw_metrics.items():
            numeric = _coerce_float(value)
            if numeric is not None:
                metrics[str(name)] = numeric
        return metrics


class StreamingCheckpointCallback(_LightningCallbackBase):
    """Lightning callback that proxies lifecycle events to the checkpoint manager."""

    def __init__(self, manager: StreamingCheckpointManager) -> None:
        try:
            super().__init__()
        except Exception:  # pragma: no cover - fallback when super().__init__ is unavailable
            pass
        self._manager = manager

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:  # pragma: no cover - Lightning runtime
        self._manager.attach_trainer(trainer, pl_module)

    def on_train_batch_end(  # pragma: no cover - Lightning runtime
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._manager.maybe_save_interval()

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:  # pragma: no cover - Lightning runtime
        self._manager.detach_trainer()


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class LightningStreamingWorker(TrainingWorker):
    """
    Execute bounded TFT training jobs using streaming dataloaders.

    Args:
        config: Worker runtime configuration (caps, accelerator, telemetry).
        output_dir: Directory where worker artifacts (logits, telemetry) are written.
        teacher_factory: Optional callable returning a ready-to-fit ``TFTTeacher``.
            When omitted, the worker instantiates a default ``TFTTeacher`` derived
            from the dataset plan's streaming configuration.

    Example:
        >>> from pathlib import Path
        >>> worker = LightningStreamingWorker(StreamingWorkerConfig(), output_dir=Path("./artifacts"))
        >>> result = worker.run(plan_event)
        >>> assert "logits" in result.artifact_paths
    """

    def __init__(
        self,
        config: StreamingWorkerConfig,
        *,
        output_dir: Path,
        teacher_factory: TeacherFactory | None = None,
    ) -> None:
        super().__init__(config)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._teacher_factory = teacher_factory
        self._validation_returns_telemetry: ValidationReturnsTelemetry | None = None
        checkpoint_dir_raw = config.checkpoint_dir.strip() if config.checkpoint_dir else ""
        if checkpoint_dir_raw:
            checkpoint_dir = Path(checkpoint_dir_raw).expanduser()
            self._checkpoint_manager: StreamingCheckpointManager | None = StreamingCheckpointManager(
                checkpoint_dir,
                retention=int(config.checkpoint_retention),
                interval_seconds=(
                    float(config.checkpoint_interval_seconds)
                    if config.checkpoint_interval_seconds is not None
                    else None
                ),
                interval_steps=(
                    int(config.checkpoint_interval_steps)
                    if config.checkpoint_interval_steps is not None
                    else None
                ),
            )
        else:
            self._checkpoint_manager = None
        self._resume_record: _CheckpointRecord | None = None

    def run(self, plan: DatasetPlanEvent) -> TrainingResultEvent:
        """Run a bounded streaming training job for the provided plan."""
        if not HAS_TORCH:
            check_ml_dependencies(["torch"])
        start_ts = time.monotonic()
        self._validation_returns_telemetry = None
        context = self._prepare_context(plan)
        dataset_id = plan.dataset_id
        checkpoint_manager = self._checkpoint_manager
        resume_record: _CheckpointRecord | None = None
        if checkpoint_manager is not None:
            resume_record = checkpoint_manager.prepare_plan(plan.plan_id, dataset_id)
            self._resume_record = resume_record
            if resume_record is not None:
                _CHECKPOINT_RESUMES_TOTAL.labels(outcome="detected").inc()
                logger.info(
                    "checkpoint_resume_detected",
                    extra={
                        "plan_id": resume_record.plan_id,
                        "dataset_id": resume_record.dataset_id,
                        "checkpoint_path": str(resume_record.path),
                        "epoch": resume_record.epoch,
                        "global_step": resume_record.global_step,
                    },
                )
        else:
            self._resume_record = None
        monitor: GPUMemoryMonitor | None = None
        peak_gpu_mb: float | None = None
        interval = self.config.gpu_memory_monitor_interval_seconds
        if interval is not None and float(interval) > 0.0:
            try:
                monitor = GPUMemoryMonitor(float(interval))
                monitor.start()
            except Exception:  # pragma: no cover - defensive guard
                monitor = None

        if not context.train_metadata.shard_indices or not context.val_metadata.shard_indices:
            if monitor is not None:
                monitor.stop()
                peak_gpu_mb = monitor.max_memory_mb()
                monitor = None
            logger.warning(
                "streaming worker skipped plan due to insufficient shards",
                extra={
                    "plan_id": plan.plan_id,
                    "train_shards": len(context.train_metadata.shard_indices),
                    "val_shards": len(context.val_metadata.shard_indices),
                },
            )
            artifact_path = self._persist_logits(
                plan.plan_id,
                z_train=np.array([], dtype=np.float32),
                z_val=np.array([], dtype=np.float32),
                y_val=np.array([], dtype=np.float32),
                train_rows=StreamingRowMetadata.empty(),
                val_rows=StreamingRowMetadata.empty(),
                val_returns=np.array([], dtype=np.float32),
            )
            status = EventStatus.DEFERRED
            duration = time.monotonic() - start_ts
            _TRAINING_RUNS_TOTAL.labels(status=status.value).inc()
            _TRAINING_DURATION_SECONDS.labels(status=status.value).observe(duration)
            telemetry = self._augment_telemetry(
                context.telemetry,
                peak_gpu_mb,
                ensemble=None,
                economic=StreamingEconomicTelemetry(),
                stability=StreamingStabilityTelemetry(),
            )
            self._record_gpu_metric(dataset_id, peak_gpu_mb)
            return TrainingResultEvent(
                plan_id=plan.plan_id,
                dataset_id=dataset_id,
                model_id=self.config.model_id,
                telemetry=telemetry,
                artifact_paths={self.config.logits_artifact_key: str(artifact_path)},
                metrics={},
                status=status,
            )

        attempts = max(1, int(self.config.max_retry_attempts))
        backoff_seconds = float(self.config.retry_backoff_seconds)
        resume_applied = False

        for attempt_index in range(1, attempts + 1):
            callbacks: Sequence[Any] | None = None
            checkpoint_path: Path | None = None
            active_record: _CheckpointRecord | None = None
            if checkpoint_manager is not None:
                active_record = resume_record if attempt_index == 1 else checkpoint_manager.refresh_latest()
                if active_record is not None:
                    candidate_path = active_record.path
                    if candidate_path.exists():
                        checkpoint_path = candidate_path
                        resume_applied = True
                        _CHECKPOINT_RESUMES_TOTAL.labels(outcome="success").inc()
                        logger.info(
                            "checkpoint_resume_applied",
                            extra={
                                "plan_id": active_record.plan_id,
                                "dataset_id": active_record.dataset_id,
                                "checkpoint_path": str(candidate_path),
                                "attempt_index": attempt_index,
                                "global_step": active_record.global_step,
                                "epoch": active_record.epoch,
                            },
                        )
                    else:
                        _CHECKPOINT_RESUMES_TOTAL.labels(outcome="missing").inc()
                        logger.warning(
                            "checkpoint_resume_missing_file",
                            extra={
                                "plan_id": active_record.plan_id,
                                "dataset_id": active_record.dataset_id,
                                "checkpoint_path": str(candidate_path),
                                "attempt_index": attempt_index,
                            },
                        )
                        checkpoint_path = None
                callbacks = checkpoint_manager.create_callbacks()
            try:
                fit_result = self._execute_training_attempt(
                    plan,
                    context,
                    callbacks=callbacks,
                    checkpoint_path=checkpoint_path,
                )
                if checkpoint_manager is not None:
                    resume_record = checkpoint_manager.latest_record
                fit_result = self._maybe_attach_validation_returns(plan, fit_result)
                fit_result, ensemble_metrics, ensemble_telemetry = self._apply_ensemble(plan, fit_result)
                artifact_path = self._persist_logits(
                    plan.plan_id,
                    z_train=fit_result.z_train,
                    z_val=fit_result.z_val,
                    y_val=fit_result.y_val,
                    train_rows=fit_result.train_rows,
                    val_rows=fit_result.val_rows,
                    val_returns=fit_result.val_returns,
                )
                metrics: dict[str, float] = dict(ensemble_metrics)
                try:
                    computed_metrics = self._compute_metrics(fit_result, plan_caps=plan.caps)
                except ValidationDatasetEmptyError as exc:
                    if monitor is not None:
                        monitor.stop()
                        peak_gpu_mb = monitor.max_memory_mb()
                        monitor = None
                    else:
                        peak_gpu_mb = None
                    duration = time.monotonic() - start_ts
                    _TRAINING_RUNS_TOTAL.labels(status=EventStatus.DEFERRED.value).inc()
                    _TRAINING_DURATION_SECONDS.labels(status=EventStatus.DEFERRED.value).observe(duration)
                    failure_caps = dict(context.telemetry.caps)
                    failure_caps["validation_failure_reason"] = exc.reason
                    for key, value in exc.details.items():
                        failure_caps[f"validation_failure_{key}"] = value
                    telemetry = replace(context.telemetry, caps=failure_caps)
                    telemetry = self._augment_telemetry(
                        telemetry,
                        peak_gpu_mb,
                        ensemble=ensemble_telemetry,
                        economic=StreamingEconomicTelemetry(),
                        stability=StreamingStabilityTelemetry(),
                    )
                    telemetry = self._attach_checkpoint_telemetry(
                        telemetry,
                        checkpoint_manager=checkpoint_manager,
                        resume_record=resume_record,
                        resume_applied=resume_applied,
                    )
                    logger.warning(
                        "streaming worker deferred plan due to empty validation data",
                        extra={
                            "plan_id": plan.plan_id,
                            "dataset_id": dataset_id,
                            "validation_failure_reason": exc.reason,
                            **{f"validation_{key}": value for key, value in exc.details.items()},
                        },
                        exc_info=True,
                    )
                    self._record_gpu_metric(dataset_id, peak_gpu_mb)
                    return TrainingResultEvent(
                        plan_id=plan.plan_id,
                        dataset_id=dataset_id,
                        model_id=self.config.model_id,
                        telemetry=telemetry,
                        artifact_paths={self.config.logits_artifact_key: str(artifact_path)},
                        metrics=metrics,
                        status=EventStatus.DEFERRED,
                    )
                metrics.update(computed_metrics)
                economic_bundle, stability_bundle, economic_metrics = self._economic_and_stability_metrics(
                    fit_result,
                    metrics,
                    plan_caps=plan.caps,
                )
                if economic_metrics:
                    metrics.update(economic_metrics)
                duration = time.monotonic() - start_ts
                status = EventStatus.SUCCESS
                if duration > float(self.config.max_runtime_seconds):
                    logger.warning(
                        "streaming worker exceeded runtime budget",
                        extra={
                            "plan_id": plan.plan_id,
                            "duration_seconds": duration,
                            "budget_seconds": self.config.max_runtime_seconds,
                        },
                    )
                    status = EventStatus.PARTIAL

                _TRAINING_RUNS_TOTAL.labels(status=status.value).inc()
                _TRAINING_DURATION_SECONDS.labels(status=status.value).observe(duration)
                if attempt_index > 1:
                    _TRAINING_RETRY_ATTEMPTS.labels(outcome="recovered").inc()
                if monitor is not None:
                    monitor.stop()
                    peak_gpu_mb = monitor.max_memory_mb()
                    monitor = None
                telemetry = self._augment_telemetry(
                    context.telemetry,
                    peak_gpu_mb,
                    ensemble=ensemble_telemetry,
                    economic=economic_bundle,
                    stability=stability_bundle,
                )
                telemetry = self._attach_checkpoint_telemetry(
                    telemetry,
                    checkpoint_manager=checkpoint_manager,
                    resume_record=resume_record,
                    resume_applied=resume_applied,
                )
                self._record_gpu_metric(dataset_id, peak_gpu_mb)
                return TrainingResultEvent(
                    plan_id=plan.plan_id,
                    dataset_id=dataset_id,
                    model_id=self.config.model_id,
                    telemetry=telemetry,
                    artifact_paths={self.config.logits_artifact_key: str(artifact_path)},
                    metrics=metrics,
                    status=status,
                )
            except Exception:  # pragma: no cover - surfaced via tests
                if attempt_index >= attempts:
                    duration = time.monotonic() - start_ts
                    _TRAINING_RETRY_ATTEMPTS.labels(outcome="failed").inc()
                    _TRAINING_RUNS_TOTAL.labels(status=EventStatus.FAILED.value).inc()
                    _TRAINING_DURATION_SECONDS.labels(status=EventStatus.FAILED.value).observe(duration)
                    logger.error(
                        "streaming worker exhausted retry attempts",
                        extra={
                            "plan_id": plan.plan_id,
                            "dataset_id": dataset_id,
                            "attempt": attempt_index,
                            "max_attempts": attempts,
                        },
                        exc_info=True,
                    )
                    if monitor is not None:
                        monitor.stop()
                        peak_gpu_mb = monitor.max_memory_mb()
                        monitor = None
                        self._record_gpu_metric(dataset_id, peak_gpu_mb)
                    raise
                logger.warning(
                    "streaming worker retrying after failure",
                    extra={
                        "plan_id": plan.plan_id,
                        "dataset_id": dataset_id,
                        "attempt": attempt_index,
                        "max_attempts": attempts,
                    },
                    exc_info=True,
                )
                _TRAINING_RETRY_ATTEMPTS.labels(outcome="scheduled").inc()
                if backoff_seconds > 0.0:
                    time.sleep(backoff_seconds * attempt_index)

        if monitor is not None:
            monitor.stop()
            peak_gpu_mb = monitor.max_memory_mb()
            self._record_gpu_metric(dataset_id, peak_gpu_mb)

        raise RuntimeError("unreachable")  # pragma: no cover - defensive guard

    def _prepare_context(self, plan: DatasetPlanEvent) -> _TrainingContext:
        worker_streaming_cfg = self._apply_worker_caps(plan.streaming_config)
        amp_enabled, amp_guard_reason = self._resolve_amp_mode(plan.caps)
        limited_metadata, overall_limits = stream.apply_streaming_limits(
            plan.metadata,
            worker_streaming_cfg,
        )
        curriculum_resolution = self._resolve_train_fraction(plan.metadata_summary, plan.caps)
        train_fraction = curriculum_resolution.train_fraction
        train_metadata, val_metadata = stream.split_metadata_by_row_fraction(
            limited_metadata,
            train_fraction,
        )
        train_metadata, train_limits = stream.apply_streaming_limits(
            train_metadata,
            worker_streaming_cfg,
        )
        val_metadata, val_limits = stream.apply_streaming_limits(
            val_metadata,
            worker_streaming_cfg,
        )
        telemetry = self._build_telemetry(
            plan_caps=plan.caps,
            full_metadata=limited_metadata,
            overall_limits=overall_limits,
            train_metadata=train_metadata,
            train_limits=train_limits,
            val_metadata=val_metadata,
            val_limits=val_limits,
            worker_config=worker_streaming_cfg,
            train_fraction=train_fraction,
            curriculum_stage=curriculum_resolution.stage_label,
            curriculum_guard_reason=curriculum_resolution.guard_reason,
            amp_enabled=amp_enabled,
            amp_guard_reason=amp_guard_reason,
        )
        return _TrainingContext(
            worker_streaming_cfg=worker_streaming_cfg,
            limited_metadata=limited_metadata,
            overall_limits=overall_limits,
            train_metadata=train_metadata,
            train_limits=train_limits,
            val_metadata=val_metadata,
            val_limits=val_limits,
            telemetry=telemetry,
            train_fraction=train_fraction,
            curriculum_stage_label=curriculum_resolution.stage_label,
            curriculum_guard_reason=curriculum_resolution.guard_reason,
            amp_enabled=amp_enabled,
            amp_guard_reason=amp_guard_reason,
        )

    def _apply_worker_seed(self) -> None:
        seed = self.config.worker_seed
        if seed is None:
            return
        seed_value = int(seed)
        logger.info(
            "streaming worker applying random seed",
            extra={"worker_seed": seed_value},
        )
        random.seed(seed_value)
        np.random.seed(seed_value)
        if HAS_TORCH and torch is not None:
            try:
                torch.manual_seed(seed_value)
                if hasattr(torch, "cuda"):
                    torch.cuda.manual_seed_all(seed_value)
            except Exception:  # pragma: no cover - optional torch context
                logger.debug(
                    "torch seeding failed",
                    extra={"worker_seed": seed_value},
                    exc_info=True,
                )

    def _execute_training_attempt(
        self,
        plan: DatasetPlanEvent,
        context: _TrainingContext,
        *,
        callbacks: Sequence[Any] | None = None,
        checkpoint_path: Path | None = None,
    ) -> StreamingFitResult:
        self._apply_worker_seed()
        train_loader = stream.build_streaming_dataloader(
            plan.parquet_path,
            context.train_metadata,
            context.worker_streaming_cfg,
            metadata_is_limited=True,
            limit_summary=context.train_limits,
        )
        val_loader = stream.build_streaming_dataloader(
            plan.parquet_path,
            context.val_metadata,
            context.worker_streaming_cfg,
            metadata_is_limited=True,
            limit_summary=context.val_limits,
        )
        teacher = self._build_teacher(
            plan,
            context.worker_streaming_cfg,
            amp_enabled=context.amp_enabled,
        )
        return teacher.fit_streaming(
            parquet_path=plan.parquet_path,
            train_loader=train_loader,
            val_loader=val_loader,
            train_metadata=context.train_metadata,
            val_metadata=context.val_metadata,
            full_metadata=context.limited_metadata,
            streaming_config=context.worker_streaming_cfg,
            callbacks=callbacks,
            checkpoint_path=checkpoint_path,
        )

    @property
    def resume_record(self) -> _CheckpointRecord | None:
        """Return checkpoint metadata used to resume the most recent plan, if any."""
        return self._resume_record

    def save_checkpoint_now(self, *, reason: str, triggered_by_signal: bool = False) -> Path | None:
        """Request an immediate checkpoint flush; queued when trainer context is absent."""
        manager = self._checkpoint_manager
        if manager is None:
            logger.debug(
                "checkpoint_request_ignored",
                extra={"reason": reason},
            )
            return None
        path = manager.request_manual_save(reason, triggered_by_signal=triggered_by_signal)
        if path is None:
            logger.info(
                "checkpoint_request_deferred",
                extra={
                    "reason": reason,
                    "triggered_by_signal": triggered_by_signal,
                },
            )
        else:
            logger.info(
                "checkpoint_request_completed",
                extra={
                    "reason": reason,
                    "triggered_by_signal": triggered_by_signal,
                    "checkpoint_path": str(path),
                },
            )
        return path

    def _apply_worker_caps(self, config: TFTStreamingConfig) -> TFTStreamingConfig:
        return replace(
            config,
            max_total_rows=self.config.max_total_rows
            if self.config.max_total_rows is not None
            else config.max_total_rows,
            max_total_sequences=self.config.max_total_sequences
            if self.config.max_total_sequences is not None
            else config.max_total_sequences,
            max_shards=self.config.max_shards
            if self.config.max_shards is not None
            else config.max_shards,
            num_workers=config.num_workers,
            seed=self.config.dataset_seed if self.config.dataset_seed is not None else config.seed,
        )

    def _maybe_attach_validation_returns(
        self,
        plan: DatasetPlanEvent,
        fit_result: StreamingFitResult,
    ) -> StreamingFitResult:
        """Populate validation returns from parquet when the teacher omits them."""
        self._validation_returns_telemetry = None
        val_returns = getattr(fit_result, "val_returns", None)
        if val_returns is not None:
            try:
                if np.asarray(val_returns).size > 0:
                    return fit_result
            except Exception:
                logger.debug(
                    "streaming worker encountered invalid validation returns; attempting reload",
                    extra={"plan_id": plan.plan_id},
                    exc_info=True,
                )
        column = self.config.validation_return_column
        if column is None:
            return fit_result
        column_normalized = column.strip()
        if not column_normalized:
            return fit_result
        val_rows = fit_result.val_rows
        if val_rows is None or val_rows.size == 0:
            return fit_result
        realignment = self._realign_validation_metadata(plan, val_rows)
        mismatch_count = 0
        if realignment is not None:
            mismatch_count = realignment.missing_count
            if realignment.corrected_count > 0:
                sample_before = [str(value) for value in val_rows.instrument_ids[:5].tolist()]
                sample_after = [str(value) for value in realignment.metadata.instrument_ids[:5].tolist()]
                sample_times = [int(value) for value in val_rows.time_indices[:5].tolist()]
                logger.info(
                    "validation metadata realignment correcting instruments (before=%s, after=%s, times=%s)",
                    sample_before,
                    sample_after,
                    sample_times,
                    extra={
                        "plan_id": plan.plan_id,
                        "corrected_rows": realignment.corrected_count,
                        "missing_rows": mismatch_count,
                        "row_count": int(val_rows.size),
                        "sample_before": sample_before,
                        "sample_after": sample_after,
                        "sample_times": sample_times,
                    },
                )
                fit_result = replace(fit_result, val_rows=realignment.metadata)
                val_rows = realignment.metadata
            if mismatch_count > 0 and realignment.corrected_count == 0:
                sample_instruments = [str(value) for value in val_rows.instrument_ids[:5].tolist()]
                sample_times = [int(value) for value in val_rows.time_indices[:5].tolist()]
                logger.warning(
                    "validation metadata precise join missing rows; retaining original instruments",
                    extra={
                        "plan_id": plan.plan_id,
                        "missing_count": mismatch_count,
                        "row_count": int(val_rows.size),
                        "sample_instruments": sample_instruments,
                        "sample_times": sample_times,
                    },
                )
        outcome = self._extract_validation_returns(
            plan,
            column_normalized,
            val_rows,
            mismatch_count=mismatch_count,
        )
        if outcome is None:
            return fit_result
        updated = replace(fit_result, val_returns=outcome.returns)
        self._validation_returns_telemetry = ValidationReturnsTelemetry(
            fallback_join=outcome.diagnostics.fallback_join,
            mismatch_count=outcome.diagnostics.mismatch_count,
            missing_count=outcome.diagnostics.missing_count,
        )
        return updated

    def _realign_validation_metadata(
        self,
        plan: DatasetPlanEvent,
        val_rows: StreamingRowMetadata,
    ) -> _ValidationMetadataRealignment | None:
        """Return corrected row metadata when instruments drift from the parquet dataset."""
        if not HAS_POLARS or pl is None:
            return None
        parquet_path = plan.parquet_path
        if not parquet_path.exists():
            return None
        try:
            instruments = np.asarray(val_rows.instrument_ids, dtype=np.str_).astype(str)
            time_indices = np.asarray(val_rows.time_indices, dtype=np.int64)
        except Exception:
            logger.debug(
                "validation_metadata_realign_conversion_failed",
                extra={"plan_id": plan.plan_id},
                exc_info=True,
            )
            return None
        if time_indices.size == 0:
            return None
        identifiers = pl.DataFrame(
            {
                "instrument_id": pl.Series(instruments, dtype=pl.String),
                "__order": pl.Series(np.arange(time_indices.size, dtype=np.int64), dtype=pl.Int64),
                "time_index": pl.Series(time_indices, dtype=pl.Int64),
            },
        )
        dataset_scan = pl.scan_parquet(str(parquet_path)).select(
            [
                pl.col("time_index"),
                pl.col("instrument_id"),
            ],
        )
        dataset_with_duplicate = dataset_scan.with_columns(
            pl.col("instrument_id").alias("__dataset_instrument"),
        )
        precise_join: PolarsDataFrame
        try:
            precise_join = cast(
                PolarsDataFrame,
                (
                    identifiers.lazy()
                    .join(dataset_with_duplicate, on=["instrument_id", "time_index"], how="left")
                    .select(["__order", "__dataset_instrument"])
                    .collect()
                    .sort("__order")
                ),
            )
        except Exception:
            logger.debug(
                "validation_metadata_realign_join_failed",
                extra={"plan_id": plan.plan_id},
                exc_info=True,
            )
            return None
        precise_series = precise_join.get_column("__dataset_instrument")
        if "__dataset_instrument" not in precise_join.columns:
            logger.debug(
                "validation_metadata_realign_missing_column",
                extra={"plan_id": plan.plan_id},
            )
            return None
        if precise_series.len() != time_indices.size:
            logger.debug(
                "validation_metadata_realign_precise_size_mismatch",
                extra={
                    "plan_id": plan.plan_id,
                    "expected": int(time_indices.size),
                    "observed": int(precise_series.len()),
                },
            )
            return None
        precise_values = precise_series.to_list()
        if precise_values is None:
            return None
        dataset_instruments = instruments.copy()
        mismatch_mask = np.array([value is None for value in precise_values], dtype=bool)
        mismatch_count = int(np.count_nonzero(mismatch_mask))
        corrections = 0
        for index, precise_value in enumerate(precise_values):
            if precise_value is None:
                continue
            precise_str = str(precise_value)
            if precise_str != dataset_instruments[index]:
                corrections += 1
            dataset_instruments[index] = precise_str
        if mismatch_count == 0 and corrections == 0:
            return None
        dataset_instruments_array = np.asarray(dataset_instruments, dtype=np.str_)
        corrected_row_ids = np.asarray(
            [
                f"{instrument}::{time_value}"
                for instrument, time_value in zip(dataset_instruments.tolist(), time_indices.tolist(), strict=False)
            ],
            dtype=np.str_,
        )
        metadata = StreamingRowMetadata(
            row_ids=corrected_row_ids,
            instrument_ids=dataset_instruments_array,
            time_indices=time_indices.astype(np.int64, copy=False),
        )
        return _ValidationMetadataRealignment(
            metadata=metadata,
            corrected_count=corrections,
            missing_count=mismatch_count,
        )

    def _extract_validation_returns(
        self,
        plan: DatasetPlanEvent,
        column: str,
        val_rows: StreamingRowMetadata,
        *,
        mismatch_count: int = 0,
    ) -> _ValidationReturnsOutcome | None:
        """Return forward returns aligned with validation logits from the parquet dataset."""
        if not HAS_POLARS or pl is None:
            logger.debug(
                "validation_returns_unavailable_polars_missing",
                extra={"plan_id": plan.plan_id},
            )
            return None
        parquet_path = plan.parquet_path
        if not parquet_path.exists():
            logger.debug(
                "validation_returns_parquet_missing",
                extra={"plan_id": plan.plan_id, "path": str(parquet_path)},
            )
            return None
        try:
            instruments = np.asarray(val_rows.instrument_ids, dtype=np.str_).astype(str)
            time_indices = np.asarray(val_rows.time_indices, dtype=np.int64)
            order = np.arange(val_rows.size, dtype=np.int64)
        except Exception:
            logger.debug(
                "validation_returns_metadata_conversion_failed",
                extra={"plan_id": plan.plan_id},
                exc_info=True,
            )
            return None
        identifiers = pl.DataFrame(
            {
                "instrument_id": pl.Series(instruments, dtype=pl.String),
                "time_index": pl.Series(time_indices, dtype=pl.Int64),
                "__order": pl.Series(order, dtype=pl.Int64),
            },
        )
        dataset_scan = pl.scan_parquet(str(parquet_path)).select(
            [
                pl.col("instrument_id"),
                pl.col("time_index"),
                pl.col(column).alias("__value"),
            ],
        )
        try:
            collected = cast(
                PolarsDataFrame,
                (
                    identifiers.lazy()
                    .join(dataset_scan, on=["instrument_id", "time_index"], how="left")
                    .select(["__order", "__value"])
                    .collect()
                    .sort("__order")
                ),
            )
        except Exception:
            logger.debug(
                "validation_returns_join_failed",
                extra={"plan_id": plan.plan_id, "column": column},
                exc_info=True,
            )
            return None
        if collected is None:
            return None
        if "__value" not in collected.columns:
            logger.debug(
                "validation_returns_column_missing_post_join",
                extra={"plan_id": plan.plan_id, "column": column},
            )
            return None
        values = collected.get_column("__value")
        missing_count = int(values.null_count())
        if missing_count:
            logger.warning(
                "validation_returns_missing_rows",
                extra={
                    "plan_id": plan.plan_id,
                    "missing_count": int(missing_count),
                    "expected": int(val_rows.size),
                },
            )
            values = values.fill_null(0.0)
        returns_array = np.asarray(values.to_numpy(), dtype=np.float64).reshape(-1)
        returns = cast(npt.NDArray[np.float64], returns_array)
        if returns.size != val_rows.size:
            logger.warning(
                "validation_returns_size_mismatch",
                extra={
                    "plan_id": plan.plan_id,
                    "expected": int(val_rows.size),
                    "observed": int(returns.size),
                },
            )
            return None
        diagnostics = _ValidationReturnsDiagnostics(
            fallback_join=False,
            mismatch_count=mismatch_count,
            missing_count=missing_count,
        )
        return _ValidationReturnsOutcome(
            returns=returns,
            diagnostics=diagnostics,
        )

    def _resolve_train_fraction(
        self,
        summary: TFTStreamingSummary,
        caps: Mapping[str, float | int | None] | Mapping[str, Any],
    ) -> CurriculumResolution:
        base_fraction = float(self.config.train_fraction)
        curriculum = self.config.curriculum
        if not curriculum.enabled:
            return CurriculumResolution(
                train_fraction=base_fraction,
                stage_label=None,
                guard_reason=None,
            )
        guard_context = CurriculumGuardContext(
            total_rows=int(summary.total_rows),
            recent_roc_auc=_coerce_float(caps.get("recent_roc_auc")),
            current_backlog=_coerce_int(caps.get("streaming_backlog")),
            recent_gpu_mb=_coerce_float(caps.get("recent_peak_gpu_mb")),
        )
        resolved = curriculum.resolve_with_context(
            total_rows=int(summary.total_rows),
            fallback=curriculum.default_train_fraction,
            context=guard_context,
        )
        fraction = resolved.train_fraction
        if not (0.0 < float(fraction) < 1.0):
            return CurriculumResolution(
                train_fraction=base_fraction,
                stage_label=resolved.stage_label,
                guard_reason=resolved.guard_reason,
            )
        if resolved.guard_reason:
            logger.info(
                "curriculum guard adjusted train fraction",
                extra={
                    "plan_total_rows": summary.total_rows,
                    "baseline_fraction": base_fraction,
                    "resolved_fraction": fraction,
                    "curriculum_guard_reason": resolved.guard_reason,
                    "curriculum_stage": resolved.stage_label,
                },
            )
        elif not math.isclose(fraction, base_fraction, rel_tol=1e-6):
            logger.debug(
                "curriculum adjusted train fraction",
                extra={
                    "plan_total_rows": summary.total_rows,
                    "baseline_fraction": base_fraction,
                    "resolved_fraction": fraction,
                    "curriculum_stage": resolved.stage_label,
                },
            )
        return resolved

    def _resolve_amp_mode(
        self,
        caps: Mapping[str, float | int | None] | Mapping[str, Any],
    ) -> tuple[bool, str | None]:
        if not self.config.enable_amp:
            return False, None
        threshold = self.config.amp_guard_threshold_mb
        if threshold is None:
            return True, None
        recent_gpu = _coerce_float(caps.get("recent_peak_gpu_mb"))
        if recent_gpu is None:
            return True, None
        if recent_gpu >= float(threshold):
            reason = (
                f"recent_peak_gpu_mb={recent_gpu:.1f} exceeds guard {float(threshold):.1f} MiB"
            )
            logger.info(
                "amp guard disabled mixed precision",
                extra={
                    "recent_peak_gpu_mb": recent_gpu,
                    "amp_guard_threshold_mb": float(threshold),
                },
            )
            return False, reason
        return True, None

    def _build_teacher(
        self,
        plan: DatasetPlanEvent,
        worker_streaming_cfg: TFTStreamingConfig,
        *,
        amp_enabled: bool,
    ) -> TFTTeacher:
        if self._teacher_factory is not None:
            return self._teacher_factory(plan, worker_streaming_cfg)
        cfg = worker_streaming_cfg
        optimizer_name = self.config.optimizer.strip().lower()
        scheduler_name = self.config.lr_scheduler.strip().lower()
        scheduler_arg = None if scheduler_name == "none" else scheduler_name
        precision_arg = self.config.amp_precision if amp_enabled else self.config.precision
        loss_name = self.config.loss_name.strip().lower()
        pos_weight = (
            float(self.config.loss_pos_weight)
            if self.config.loss_pos_weight is not None
            else None
        )
        teacher_config = TFTTeacherConfig(
            loss_name=loss_name,
            pos_weight=pos_weight,
        )
        return TFTTeacher(
            teacher_config,
            max_encoder_length=cfg.max_encoder_length,
            max_prediction_length=cfg.max_prediction_length,
            max_epochs=self.config.max_epochs,
            static_categoricals=cfg.static_categoricals,
            static_reals=cfg.static_reals,
            time_varying_known_reals=cfg.time_varying_known_reals,
            time_varying_unknown_reals=cfg.time_varying_unknown_reals,
            time_idx_col=cfg.time_idx_col,
            group_id_col=cfg.group_id_col,
            target_col=cfg.target_col,
            dataloader_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            hidden_size=self.config.hidden_size,
            lstm_layers=self.config.lstm_layers,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout,
            learning_rate=float(self.config.learning_rate),
            optimizer=optimizer_name,
            lr_scheduler=scheduler_arg,
            precision=precision_arg,
        )

    def _apply_ensemble(
        self,
        plan: DatasetPlanEvent,
        fit_result: StreamingFitResult,
    ) -> tuple[StreamingFitResult, dict[str, float], StreamingEnsembleTelemetry | None]:
        ensemble_cfg = self.config.ensemble
        if not (ensemble_cfg.enabled and ensemble_cfg.members):
            primary_inventory = StreamingEnsembleMemberTelemetry(
                artifact_path="__primary__",
                weight=1.0,
                required=True,
                used=True,
                skipped_reason=None,
                train_row_count=fit_result.train_rows.size if fit_result.train_rows else None,
                validation_row_count=fit_result.val_rows.size if fit_result.val_rows else None,
            )
            telemetry = StreamingEnsembleTelemetry(
                blend_mode=ensemble_cfg.normalized_blend_mode,
                normalize_weights=ensemble_cfg.normalize_weights,
                members=(primary_inventory,),
                members_used=0,
                optional_members_skipped=0,
                misaligned_members=0,
            )
            return fit_result, {}, telemetry

        base_train = np.asarray(fit_result.z_train, dtype=np.float64).reshape(-1)
        base_val = np.asarray(fit_result.z_val, dtype=np.float64).reshape(-1)
        train_arrays: list[npt.NDArray[np.float64]] = [base_train]
        val_arrays: list[npt.NDArray[np.float64]] = [base_val]
        weights: list[float] = [1.0]
        skipped_optional = 0
        misaligned_members = 0
        inventory: list[StreamingEnsembleMemberTelemetry] = [
            StreamingEnsembleMemberTelemetry(
                artifact_path="__primary__",
                weight=1.0,
                required=True,
                used=True,
                skipped_reason=None,
                train_row_count=fit_result.train_rows.size if fit_result.train_rows else None,
                validation_row_count=fit_result.val_rows.size if fit_result.val_rows else None,
            ),
        ]
        base_train_rows = fit_result.train_rows
        base_val_rows = fit_result.val_rows
        used_members = 0

        for member in ensemble_cfg.members:
            def _record_inventory(
                *,
                used: bool,
                reason: str | None,
                payload: _LoadedMemberLogits | None,
            ) -> None:
                inventory.append(
                    StreamingEnsembleMemberTelemetry(
                        artifact_path=member.artifact_path,
                        weight=float(member.weight),
                        required=bool(member.required),
                        used=used,
                        skipped_reason=reason,
                        train_row_count=payload.train_rows.size if payload and payload.train_rows else None,
                        validation_row_count=payload.val_rows.size if payload and payload.val_rows else None,
                    ),
                )

            try:
                member_payload = self._load_member_logits(member, plan)
            except FileNotFoundError:
                _record_inventory(used=False, reason="missing_artifact", payload=None)
                if member.required:
                    raise
                skipped_optional += 1
                continue
            except ValueError:
                _record_inventory(used=False, reason="invalid_payload", payload=None)
                if member.required:
                    raise
                skipped_optional += 1
                continue
            if member_payload.z_train.shape != base_train.shape or member_payload.z_val.shape != base_val.shape:
                logger.warning(
                    "skipping ensemble member with shape mismatch",
                    extra={
                        "plan_id": plan.plan_id,
                        "dataset_id": plan.dataset_id,
                        "member_path": member.artifact_path,
                    },
                )
                if not member.required:
                    skipped_optional += 1
                _record_inventory(used=False, reason="shape_mismatch", payload=member_payload)
                continue
            alignment_issue = self._detect_alignment_issue(base_train_rows, member_payload.train_rows, split="train")
            if alignment_issue is None:
                alignment_issue = self._detect_alignment_issue(
                    base_val_rows,
                    member_payload.val_rows,
                    split="validation",
                )
            if alignment_issue is not None:
                misaligned_members += 1
                _record_inventory(used=False, reason=f"alignment:{alignment_issue}", payload=member_payload)
                logger.warning(
                    "skipping ensemble member due to row metadata misalignment",
                    extra={
                        "plan_id": plan.plan_id,
                        "dataset_id": plan.dataset_id,
                        "member_path": member.artifact_path,
                        "alignment_issue": alignment_issue,
                    },
                )
                if member.required:
                    raise ValueError(f"ensemble member {member.artifact_path} misaligned: {alignment_issue}")
                skipped_optional += 1
                continue
            train_arrays.append(member_payload.z_train)
            val_arrays.append(member_payload.z_val)
            weights.append(float(member.weight))
            used_members += 1
            _record_inventory(used=True, reason=None, payload=member_payload)

        if len(train_arrays) == 1:
            metrics_info: dict[str, float] = {"ensemble_members_used": 0.0}
            if skipped_optional:
                metrics_info["ensemble_optional_members_skipped"] = float(skipped_optional)
            if misaligned_members:
                metrics_info["ensemble_members_misaligned"] = float(misaligned_members)
            telemetry = StreamingEnsembleTelemetry(
                blend_mode=ensemble_cfg.normalized_blend_mode,
                normalize_weights=ensemble_cfg.normalize_weights,
                members=tuple(inventory),
                members_used=used_members,
                optional_members_skipped=skipped_optional,
                misaligned_members=misaligned_members,
            )
            return fit_result, metrics_info, telemetry

        blend_mode = ensemble_cfg.normalized_blend_mode
        normalized_weights: list[float]
        if blend_mode == "mean":
            normalized_value = 1.0 / float(len(train_arrays))
            normalized_weights = [normalized_value] * len(train_arrays)
        else:
            total_weight = sum(weights)
            if ensemble_cfg.normalize_weights and total_weight > 0.0:
                normalized_weights = [weight / total_weight for weight in weights]
            else:
                normalized_weights = weights

        blended_train = self._blend_logits(train_arrays, normalized_weights)
        blended_val = self._blend_logits(val_arrays, normalized_weights)
        metrics_extra: dict[str, float] = {
            "ensemble_members_used": float(used_members),
        }
        if skipped_optional:
            metrics_extra["ensemble_optional_members_skipped"] = float(skipped_optional)
        if misaligned_members:
            metrics_extra["ensemble_members_misaligned"] = float(misaligned_members)
        logger.debug(
            "applied ensemble blending",
            extra={
                "plan_id": plan.plan_id,
                "dataset_id": plan.dataset_id,
                "ensemble_members": used_members,
                "blend_mode": blend_mode,
            },
        )
        blended_result = StreamingFitResult(
            z_train=blended_train,
            z_val=blended_val,
            y_val=np.asarray(fit_result.y_val, dtype=np.float64).reshape(-1),
            train_rows=fit_result.train_rows,
            val_rows=fit_result.val_rows,
            val_returns=fit_result.val_returns,
        )
        telemetry = StreamingEnsembleTelemetry(
            blend_mode=ensemble_cfg.normalized_blend_mode,
            normalize_weights=ensemble_cfg.normalize_weights,
            members=tuple(inventory),
            members_used=used_members,
            optional_members_skipped=skipped_optional,
            misaligned_members=misaligned_members,
        )
        return blended_result, metrics_extra, telemetry

    def _load_member_logits(
        self,
        member: EnsembleMemberConfig,
        plan: DatasetPlanEvent,
    ) -> _LoadedMemberLogits:
        path = self._resolve_member_path(member.artifact_path, plan)
        if not path.exists():
            logger.warning(
                "ensemble member artifact missing",
                extra={
                    "plan_id": plan.plan_id,
                    "dataset_id": plan.dataset_id,
                    "path": str(path),
                },
            )
            raise FileNotFoundError(path)
        try:
            with np.load(path, allow_pickle=False) as payload:
                z_train = np.asarray(payload["z_train"], dtype=np.float64).reshape(-1)
                z_val = np.asarray(payload["z_val"], dtype=np.float64).reshape(-1)
                train_rows = self._extract_saved_row_metadata(payload, prefix="train")
                val_rows = self._extract_saved_row_metadata(payload, prefix="val")
        except KeyError as exc:
            logger.warning(
                "ensemble member missing required keys",
                extra={
                    "plan_id": plan.plan_id,
                    "dataset_id": plan.dataset_id,
                    "path": str(path),
                },
                exc_info=True,
            )
            raise ValueError(str(exc)) from exc
        return _LoadedMemberLogits(
            z_train=z_train,
            z_val=z_val,
            train_rows=train_rows,
            val_rows=val_rows,
        )

    def _extract_saved_row_metadata(
        self,
        payload: Mapping[str, Any],
        *,
        prefix: str,
    ) -> StreamingRowMetadata | None:
        row_ids_key = f"{prefix}_row_ids"
        instrument_key = f"{prefix}_instrument_ids"
        time_key = f"{prefix}_time_indices"
        if row_ids_key not in payload or instrument_key not in payload or time_key not in payload:
            return None
        row_ids = np.asarray(payload[row_ids_key])
        instrument_ids = np.asarray(payload[instrument_key])
        time_indices = np.asarray(payload[time_key])
        if row_ids.size != instrument_ids.size or row_ids.size != time_indices.size:
            return None
        return StreamingRowMetadata(
            row_ids=row_ids.astype(np.str_, copy=False),
            instrument_ids=instrument_ids.astype(np.str_, copy=False),
            time_indices=time_indices.astype(np.int64, copy=False),
        )

    def _detect_alignment_issue(
        self,
        base_rows: StreamingRowMetadata | None,
        member_rows: StreamingRowMetadata | None,
        *,
        split: str,
    ) -> str | None:
        if base_rows is None:
            return None
        if member_rows is None:
            return f"{split}_metadata_missing"
        if base_rows.size != member_rows.size:
            return f"{split}_size_mismatch"
        if not np.array_equal(base_rows.row_ids, member_rows.row_ids):
            return f"{split}_row_id_mismatch"
        return None

    def _resolve_member_path(self, template: str, plan: DatasetPlanEvent) -> Path:
        mapping = {
            "plan_id": plan.plan_id,
            "dataset_id": plan.dataset_id,
            "output_dir": str(self._output_dir),
        }
        try:
            resolved = template.format_map(mapping)
        except KeyError as exc:  # pragma: no cover - configuration guard
            raise ValueError(f"unknown placeholder {exc.args[0]!r} in ensemble path") from exc
        candidate = Path(resolved)
        if not candidate.is_absolute():
            candidate = (self._output_dir / candidate).resolve()
        return candidate

    @staticmethod
    def _blend_logits(
        arrays: Sequence[npt.NDArray[np.float64]],
        weights: Sequence[float],
    ) -> npt.NDArray[np.float64]:
        if len(arrays) != len(weights):  # pragma: no cover - defensive guard
            raise ValueError("arrays and weights must have equal length")
        blended = np.zeros_like(arrays[0], dtype=np.float64)
        for array, weight in zip(arrays, weights):
            blended += array * float(weight)
        return blended

    def _build_telemetry(
        self,
        *,
        plan_caps: Mapping[str, float | int | bool | None] | Mapping[str, Any],
        full_metadata: TFTStreamingMetadata,
        overall_limits: StreamingLimitSummary,
        train_metadata: TFTStreamingMetadata,
        train_limits: StreamingLimitSummary,
        val_metadata: TFTStreamingMetadata,
        val_limits: StreamingLimitSummary,
        worker_config: TFTStreamingConfig,
        train_fraction: float,
        curriculum_stage: str | None,
        curriculum_guard_reason: str | None,
        amp_enabled: bool,
        amp_guard_reason: str | None,
    ) -> StreamingRunTelemetry:
        metadata_summary = summarize_metadata(full_metadata)
        merged_caps: dict[str, object] = {
            **dict(plan_caps),
            "worker_max_total_rows": worker_config.max_total_rows,
            "worker_max_total_sequences": worker_config.max_total_sequences,
            "worker_max_shards": worker_config.max_shards,
            "worker_train_fraction": float(train_fraction),
            "worker_skipped_shards": overall_limits.skipped_shards,
            "worker_skipped_rows": overall_limits.skipped_rows,
            "worker_skipped_sequences": overall_limits.skipped_sequences,
            "worker_curriculum_enabled": self.config.curriculum.enabled,
            "worker_ensemble_enabled": self.config.ensemble.enabled,
            "worker_ensemble_members_configured": len(self.config.ensemble.members),
            "worker_amp_enabled": bool(amp_enabled),
            "worker_loss_name": self.config.loss_name.strip().lower(),
        }
        if worker_config.seed is not None and "dataset_seed" not in merged_caps:
            merged_caps["dataset_seed"] = int(worker_config.seed)
        if self.config.worker_seed is not None:
            merged_caps["worker_seed"] = int(self.config.worker_seed)
        if self.config.loss_pos_weight is not None:
            merged_caps["worker_loss_pos_weight"] = float(self.config.loss_pos_weight)
        if curriculum_stage:
            merged_caps["worker_curriculum_stage"] = curriculum_stage
        if curriculum_guard_reason:
            merged_caps["worker_curriculum_reason"] = curriculum_guard_reason
        if amp_guard_reason:
            merged_caps["worker_amp_guard_reason"] = amp_guard_reason
        train_stats = StreamingLoaderTelemetry.from_metadata(
            "train",
            train_metadata,
            train_limits,
            worker_config,
        )
        val_stats = StreamingLoaderTelemetry.from_metadata(
            "validation",
            val_metadata,
            val_limits,
            worker_config,
        )
        if train_stats.instrument_rows_selected:
            merged_caps["worker_train_instrument_rows_selected"] = dict(train_stats.instrument_rows_selected)
        if train_stats.instrument_rows_total:
            merged_caps["worker_train_instrument_rows_total"] = dict(train_stats.instrument_rows_total)
        if train_stats.instrument_rows_skipped:
            merged_caps["worker_train_instrument_rows_skipped"] = dict(train_stats.instrument_rows_skipped)
        if train_stats.instrument_sequences_selected:
            merged_caps["worker_train_instrument_sequences_selected"] = dict(
                train_stats.instrument_sequences_selected,
            )
        if train_stats.instrument_sequences_total:
            merged_caps["worker_train_instrument_sequences_total"] = dict(
                train_stats.instrument_sequences_total,
            )
        if train_stats.instrument_sequences_skipped:
            merged_caps["worker_train_instrument_sequences_skipped"] = dict(
                train_stats.instrument_sequences_skipped,
            )
        if val_stats.instrument_rows_selected:
            merged_caps["worker_validation_instrument_rows_selected"] = dict(
                val_stats.instrument_rows_selected,
            )
        if val_stats.instrument_rows_total:
            merged_caps["worker_validation_instrument_rows_total"] = dict(
                val_stats.instrument_rows_total,
            )
        if val_stats.instrument_rows_skipped:
            merged_caps["worker_validation_instrument_rows_skipped"] = dict(
                val_stats.instrument_rows_skipped,
            )
        if val_stats.instrument_sequences_selected:
            merged_caps["worker_validation_instrument_sequences_selected"] = dict(
                val_stats.instrument_sequences_selected,
            )
        if val_stats.instrument_sequences_total:
            merged_caps["worker_validation_instrument_sequences_total"] = dict(
                val_stats.instrument_sequences_total,
            )
        if val_stats.instrument_sequences_skipped:
            merged_caps["worker_validation_instrument_sequences_skipped"] = dict(
                val_stats.instrument_sequences_skipped,
            )
        return StreamingRunTelemetry(
            metadata_summary=metadata_summary,
            caps=merged_caps,
            train=train_stats,
            validation=val_stats,
        )

    def _augment_telemetry(
        self,
        telemetry: StreamingRunTelemetry,
        peak_gpu_mb: float | None,
        *,
        ensemble: StreamingEnsembleTelemetry | None,
        economic: StreamingEconomicTelemetry,
        stability: StreamingStabilityTelemetry,
    ) -> StreamingRunTelemetry:
        result = telemetry
        if peak_gpu_mb is not None:
            result = replace(result, max_gpu_memory_mb=peak_gpu_mb)
        if ensemble is not None:
            result = replace(result, ensemble=ensemble)
        if any(
            value is not None
            for value in (
                economic.slippage_adjusted_sharpe,
                economic.hit_rate,
                economic.turnover,
                economic.max_drawdown,
            )
        ):
            result = replace(result, economic=economic)
        if any(
            value is not None
            for value in (
                stability.ks_statistic,
                stability.calibration_drift,
            )
        ):
            result = replace(result, stability=stability)
        if self._validation_returns_telemetry is not None:
            result = replace(result, validation_returns=self._validation_returns_telemetry)
        return result

    def _attach_checkpoint_telemetry(
        self,
        telemetry: StreamingRunTelemetry,
        *,
        checkpoint_manager: StreamingCheckpointManager | None,
        resume_record: _CheckpointRecord | None,
        resume_applied: bool,
    ) -> StreamingRunTelemetry:
        if checkpoint_manager is None:
            return telemetry
        latest_record = checkpoint_manager.latest_record
        checkpoint_payload = StreamingCheckpointTelemetry(
            resumed=resume_applied,
            resume_global_step=resume_record.global_step if resume_record is not None else None,
            resume_epoch=resume_record.epoch if resume_record is not None else None,
            resume_checkpoint_path=str(resume_record.path) if resume_record is not None else None,
            latest_checkpoint_path=str(latest_record.path) if latest_record is not None else None,
        )
        return replace(telemetry, checkpoint=checkpoint_payload)

    def _record_gpu_metric(self, dataset_id: str, peak_gpu_mb: float | None) -> None:
        if peak_gpu_mb is None:
            return
        logger.info(
            "streaming worker GPU peak memory recorded",
            extra={
                "dataset_id": dataset_id,
                "gpu_peak_mb": peak_gpu_mb,
            },
        )
        try:
            _GPU_PEAK_MEMORY_MB.labels(dataset_id=dataset_id).set(float(peak_gpu_mb))
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "failed to publish GPU peak memory metric",
                extra={"dataset_id": dataset_id, "gpu_peak_mb": peak_gpu_mb},
                exc_info=True,
            )

    def _persist_logits(
        self,
        plan_id: str,
        *,
        z_train: npt.NDArray[np.float64] | npt.NDArray[np.float32],
        z_val: npt.NDArray[np.float64] | npt.NDArray[np.float32],
        y_val: npt.NDArray[np.float64] | npt.NDArray[np.float32],
        train_rows: StreamingRowMetadata | None = None,
        val_rows: StreamingRowMetadata | None = None,
        val_returns: npt.NDArray[np.float64] | npt.NDArray[np.float32] | None = None,
    ) -> Path:
        artifact_path = self._output_dir / f"{plan_id}_logits.npz"
        payload: dict[str, npt.NDArray[Any]] = {
            "z_train": np.asarray(z_train, dtype=np.float32).reshape(-1),
            "z_val": np.asarray(z_val, dtype=np.float32).reshape(-1),
            "y_val": np.asarray(y_val, dtype=np.float32).reshape(-1),
        }
        if val_returns is not None:
            payload["val_returns"] = np.asarray(val_returns, dtype=np.float32).reshape(-1)
        if train_rows is not None:
            payload["train_row_ids"] = np.asarray(train_rows.row_ids, dtype=np.str_)
            payload["train_instrument_ids"] = np.asarray(train_rows.instrument_ids, dtype=np.str_)
            payload["train_time_indices"] = np.asarray(train_rows.time_indices, dtype=np.int64)
        if val_rows is not None:
            payload["val_row_ids"] = np.asarray(val_rows.row_ids, dtype=np.str_)
            payload["val_instrument_ids"] = np.asarray(val_rows.instrument_ids, dtype=np.str_)
            payload["val_time_indices"] = np.asarray(val_rows.time_indices, dtype=np.int64)
        np.savez_compressed(file=artifact_path, **cast(dict[str, Any], payload))
        return artifact_path

    def _compute_metrics(
        self,
        fit_result: StreamingFitResult,
        *,
        plan_caps: Mapping[str, float | int | None],
    ) -> dict[str, float]:
        metric_name = self.config.validation_metric.strip().lower()
        y_val = np.asarray(fit_result.y_val, dtype=np.float64).reshape(-1)
        logits = cast(
            npt.NDArray[np.float64],
            np.asarray(fit_result.z_val, dtype=np.float64).reshape(-1),
        )
        if y_val.size == 0 or logits.size == 0:
            raise ValidationDatasetEmptyError(
                "validation_data_empty",
                details={
                    "y_val_size": int(y_val.size),
                    "logit_size": int(logits.size),
                },
            )

        logits = np.clip(logits, -60.0, 60.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        prevalence = float(np.mean(y_val)) if y_val.size else 0.0

        metrics: dict[str, float] = {}
        auc = roc_auc(y_val, probabilities)
        metrics["roc_auc"] = auc
        pr_value = pr_auc(y_val, probabilities)
        metrics["pr_auc"] = pr_value
        metrics["positive_rate"] = prevalence
        metrics["pr_auc_multiple"] = pr_value / prevalence if prevalence > 0.0 else 0.0
        base_log_loss = binary_logloss(y_val, probabilities)
        metrics["log_loss"] = base_log_loss
        metrics["brier_score"] = float(np.mean((probabilities - y_val) ** 2))
        metrics["calibration_ece_20"] = expected_calibration_error(probabilities, y_val, bins=20)
        metrics["calibration_ece_50"] = expected_calibration_error(probabilities, y_val, bins=50)

        if metric_name not in metrics:
            metrics[metric_name] = metrics.get("roc_auc", 0.0)
        base_calibration_metrics = {
            "log_loss": float(base_log_loss),
            "calibration_ece_20": float(metrics["calibration_ece_20"]),
            "calibration_ece_50": float(metrics["calibration_ece_50"]),
            "brier_score": float(metrics["brier_score"]),
        }
        base_probabilities = probabilities
        if self.config.enable_temperature_calibration:
            calibration_metrics = self._temperature_calibration_metrics(
                logits,
                y_val,
                base_log_loss=base_log_loss,
                base_probabilities=base_probabilities,
            )
            metrics.update(calibration_metrics)
            metrics.update(
                self._calibration_delta_metrics(
                    prefix="temperature_calibration",
                    calibration_metrics=calibration_metrics,
                    base_metrics=base_calibration_metrics,
                ),
            )
        if self.config.enable_platt_calibration:
            calibration_metrics = self._platt_calibration_metrics(
                logits,
                y_val,
                base_probabilities=base_probabilities,
            )
            metrics.update(calibration_metrics)
            metrics.update(
                self._calibration_delta_metrics(
                    prefix="platt_calibration",
                    calibration_metrics=calibration_metrics,
                    base_metrics=base_calibration_metrics,
                ),
            )
        if self.config.enable_isotonic_calibration:
            calibration_metrics = self._isotonic_calibration_metrics(
                base_probabilities,
                y_val,
            )
            metrics.update(calibration_metrics)
            metrics.update(
                self._calibration_delta_metrics(
                    prefix="isotonic_calibration",
                    calibration_metrics=calibration_metrics,
                    base_metrics=base_calibration_metrics,
                ),
            )
        return metrics

    def _economic_and_stability_metrics(
        self,
        fit_result: StreamingFitResult,
        metrics: Mapping[str, float],
        *,
        plan_caps: Mapping[str, float | int | None],
    ) -> tuple[StreamingEconomicTelemetry, StreamingStabilityTelemetry, dict[str, float]]:
        y_val = np.asarray(fit_result.y_val, dtype=np.float64).reshape(-1)
        logits_val = cast(
            npt.NDArray[np.float64],
            np.asarray(fit_result.z_val, dtype=np.float64).reshape(-1),
        )
        if y_val.size == 0 or logits_val.size == 0:
            return (
                StreamingEconomicTelemetry(),
                StreamingStabilityTelemetry(),
                {},
            )
        logits_val = cast(npt.NDArray[np.float64], np.clip(logits_val, -60.0, 60.0))
        probabilities = 1.0 / (1.0 + np.exp(-logits_val))
        train_logits = cast(
            npt.NDArray[np.float64],
            np.asarray(fit_result.z_train, dtype=np.float64).reshape(-1),
        )
        train_probabilities = None
        if train_logits.size > 0:
            train_logits = np.clip(train_logits, -60.0, 60.0)
            train_probabilities = 1.0 / (1.0 + np.exp(-train_logits))
        slippage_bps = _coerce_float(plan_caps.get("slippage_bps"))
        if slippage_bps is None:
            slippage_bps = _coerce_float(plan_caps.get("slippage_basis_points"))
        baseline_ece = _coerce_float(plan_caps.get("baseline_calibration_ece_20"))
        economic_bundle, stability_bundle, metric_mapping = compute_economic_and_stability_metrics(
            validation_probabilities=probabilities,
            validation_labels=y_val,
            training_probabilities=train_probabilities,
            validation_returns=fit_result.val_returns,
            slippage_bps=slippage_bps,
            calibration_metrics=metrics,
            baseline_calibration_ece=baseline_ece,
        )
        economic_telemetry = StreamingEconomicTelemetry(
            slippage_adjusted_sharpe=economic_bundle.slippage_adjusted_sharpe,
            hit_rate=economic_bundle.hit_rate,
            turnover=economic_bundle.turnover,
            max_drawdown=economic_bundle.max_drawdown,
        )
        stability_telemetry = StreamingStabilityTelemetry(
            ks_statistic=stability_bundle.ks_statistic,
            calibration_drift=stability_bundle.calibration_drift,
        )
        return economic_telemetry, stability_telemetry, metric_mapping

    def _temperature_calibration_metrics(
        self,
        logits: npt.NDArray[np.float64],
        labels: npt.NDArray[np.float64],
        *,
        base_log_loss: float,
        base_probabilities: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        if logits.size == 0:
            return self._calibration_summary(
                prefix="temperature_calibration",
                probabilities=base_probabilities,
                labels=labels,
                log_loss_override=base_log_loss,
                extras={"temperature": 1.0},
            )
        min_temp = float(self.config.temperature_calibration_min)
        max_temp = float(self.config.temperature_calibration_max)
        steps = max(1, int(self.config.temperature_calibration_steps))
        temperature_grid = np.linspace(min_temp, max_temp, steps)
        best_temperature = 1.0
        best_log_loss = base_log_loss
        best_probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))

        for temperature in temperature_grid:
            if temperature <= 0.0:
                continue
            scaled_logits = np.clip(logits / temperature, -60.0, 60.0)
            probabilities = 1.0 / (1.0 + np.exp(-scaled_logits))
            log_loss = binary_logloss(labels, probabilities)
            if np.isnan(log_loss) or np.isinf(log_loss):
                continue
            if log_loss < best_log_loss:
                best_log_loss = log_loss
                best_temperature = float(temperature)
                best_probabilities = probabilities

        return self._calibration_summary(
            prefix="temperature_calibration",
            probabilities=best_probabilities,
            labels=labels,
            log_loss_override=best_log_loss,
            extras={"temperature": best_temperature},
        )

    def _platt_calibration_metrics(
        self,
        logits: npt.NDArray[np.float64],
        labels: npt.NDArray[np.float64],
        *,
        base_probabilities: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        if logits.size == 0:
            return self._calibration_summary(
                prefix="platt_calibration",
                probabilities=base_probabilities,
                labels=labels,
            )
        if not HAS_SKLEARN or sklearn is None:
            check_ml_dependencies(["scikit-learn"])
        from sklearn.linear_model import LogisticRegression

        try:
            model = LogisticRegression(
                solver="lbfgs",
                max_iter=500,
            )
            features = logits.reshape(-1, 1)
            model.fit(features, labels.astype(int))
            calibrated_probabilities = model.predict_proba(features)[:, 1]
        except Exception:
            logger.debug(
                "platt calibration failed; falling back to base probabilities",
                extra={"plan_size": logits.size},
                exc_info=True,
            )
            return self._calibration_summary(
                prefix="platt_calibration",
                probabilities=base_probabilities,
                labels=labels,
            )
        return self._calibration_summary(
            prefix="platt_calibration",
            probabilities=calibrated_probabilities,
            labels=labels,
        )

    def _isotonic_calibration_metrics(
        self,
        probabilities: npt.NDArray[np.float64],
        labels: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        if probabilities.size == 0:
            return self._calibration_summary(
                prefix="isotonic_calibration",
                probabilities=probabilities,
                labels=labels,
            )
        if not HAS_SKLEARN or sklearn is None:
            check_ml_dependencies(["scikit-learn"])
        from sklearn.isotonic import IsotonicRegression

        try:
            calibrator = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
            ordered_indices = np.argsort(probabilities)
            ordered_probs = probabilities[ordered_indices]
            ordered_labels = labels[ordered_indices]
            calibrator.fit(ordered_probs, ordered_labels)
            calibrated_probabilities = calibrator.transform(probabilities)
        except Exception:
            logger.debug(
                "isotonic calibration failed; falling back to base probabilities",
                extra={"plan_size": probabilities.size},
                exc_info=True,
            )
            return self._calibration_summary(
                prefix="isotonic_calibration",
                probabilities=probabilities,
                labels=labels,
            )
        return self._calibration_summary(
            prefix="isotonic_calibration",
            probabilities=np.asarray(calibrated_probabilities, dtype=np.float64),
            labels=labels,
        )

    def _calibration_summary(
        self,
        *,
        prefix: str,
        probabilities: npt.NDArray[np.float64],
        labels: npt.NDArray[np.float64],
        log_loss_override: float | None = None,
        extras: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        clipped = np.asarray(np.clip(probabilities, 1e-12, 1.0 - 1e-12), dtype=np.float64)
        summary: dict[str, float] = {
            f"{prefix}_log_loss": float(
                log_loss_override if log_loss_override is not None else binary_logloss(labels, clipped),
            ),
            f"{prefix}_ece_20": expected_calibration_error(clipped, labels, bins=20),
            f"{prefix}_ece_50": expected_calibration_error(clipped, labels, bins=50),
            f"{prefix}_brier_score": float(np.mean((clipped - labels) ** 2)),
        }
        if extras:
            for key, value in extras.items():
                summary[f"{prefix}_{key}"] = float(value)
        return summary

    @staticmethod
    def _calibration_delta_metrics(
        *,
        prefix: str,
        calibration_metrics: Mapping[str, float],
        base_metrics: Mapping[str, float],
    ) -> dict[str, float]:
        suffix_map = {
            "log_loss": "log_loss",
            "ece_20": "calibration_ece_20",
            "ece_50": "calibration_ece_50",
            "brier_score": "brier_score",
        }
        deltas: dict[str, float] = {}
        for suffix, base_key in suffix_map.items():
            calibrated_key = f"{prefix}_{suffix}"
            calibrated_value = calibration_metrics.get(calibrated_key)
            base_value = base_metrics.get(base_key)
            if calibrated_value is None or base_value is None:
                continue
            deltas[f"{calibrated_key}_delta"] = float(calibrated_value) - float(base_value)
        return deltas


__all__ = [
    "LightningStreamingWorker",
]
