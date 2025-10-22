"""Lightning-backed streaming training worker for the event-driven pipeline."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_TORCH
from ml._imports import check_ml_dependencies
from ml.common.gpu_monitor import GPUMemoryMonitor
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram
from ml.config.events import EventStatus
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.evaluation.metrics import binary_logloss
from ml.evaluation.metrics import expected_calibration_error
from ml.evaluation.metrics import pr_auc
from ml.evaluation.metrics import roc_auc
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.event_driven.services import TrainingWorker
from ml.training.teacher import streaming_loader as stream
from ml.training.teacher.streaming_loader import StreamingLimitSummary
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingMetadata
from ml.training.teacher.streaming_loader import summarize_metadata
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry
from ml.training.teacher.tft_teacher import StreamingFitResult
from ml.training.teacher.tft_teacher import TFTTeacher
from ml.training.teacher.tft_teacher import TFTTeacherConfig


logger = logging.getLogger(__name__)


TeacherFactory = Callable[[DatasetPlanEvent, TFTStreamingConfig], TFTTeacher]

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
    def run(self, plan: DatasetPlanEvent) -> TrainingResultEvent:
        """Run a bounded streaming training job for the provided plan."""
        if not HAS_TORCH:
            check_ml_dependencies(["torch"])
        start_ts = time.monotonic()
        context = self._prepare_context(plan)
        dataset_id = plan.dataset_id
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
            )
            status = EventStatus.DEFERRED
            duration = time.monotonic() - start_ts
            _TRAINING_RUNS_TOTAL.labels(status=status.value).inc()
            _TRAINING_DURATION_SECONDS.labels(status=status.value).observe(duration)
            telemetry = self._augment_telemetry(context.telemetry, peak_gpu_mb)
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

        for attempt_index in range(1, attempts + 1):
            try:
                fit_result = self._execute_training_attempt(plan, context)
                artifact_path = self._persist_logits(
                    plan.plan_id,
                    z_train=fit_result.z_train,
                    z_val=fit_result.z_val,
                    y_val=fit_result.y_val,
                )
                metrics = self._compute_metrics(fit_result)
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
                telemetry = self._augment_telemetry(context.telemetry, peak_gpu_mb)
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
        limited_metadata, overall_limits = stream.apply_streaming_limits(
            plan.metadata,
            worker_streaming_cfg,
        )
        train_metadata, val_metadata = stream.split_metadata_by_row_fraction(
            limited_metadata,
            self.config.train_fraction,
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
        )

    def _execute_training_attempt(
        self,
        plan: DatasetPlanEvent,
        context: _TrainingContext,
    ) -> StreamingFitResult:
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
        teacher = self._build_teacher(plan, context.worker_streaming_cfg)
        return teacher.fit_streaming(
            parquet_path=plan.parquet_path,
            train_loader=train_loader,
            val_loader=val_loader,
            train_metadata=context.train_metadata,
            val_metadata=context.val_metadata,
            full_metadata=context.limited_metadata,
            streaming_config=context.worker_streaming_cfg,
        )

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
        )

    def _build_teacher(
        self,
        plan: DatasetPlanEvent,
        worker_streaming_cfg: TFTStreamingConfig,
    ) -> TFTTeacher:
        if self._teacher_factory is not None:
            return self._teacher_factory(plan, worker_streaming_cfg)
        cfg = worker_streaming_cfg
        return TFTTeacher(
            TFTTeacherConfig(),
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
        )

    def _build_telemetry(
        self,
        *,
        plan_caps: dict[str, float | int | None],
        full_metadata: TFTStreamingMetadata,
        overall_limits: StreamingLimitSummary,
        train_metadata: TFTStreamingMetadata,
        train_limits: StreamingLimitSummary,
        val_metadata: TFTStreamingMetadata,
        val_limits: StreamingLimitSummary,
        worker_config: TFTStreamingConfig,
    ) -> StreamingRunTelemetry:
        metadata_summary = summarize_metadata(full_metadata)
        merged_caps: dict[str, float | int | None] = {
            **plan_caps,
            "worker_max_total_rows": worker_config.max_total_rows,
            "worker_max_total_sequences": worker_config.max_total_sequences,
            "worker_max_shards": worker_config.max_shards,
            "worker_train_fraction": self.config.train_fraction,
            "worker_skipped_shards": overall_limits.skipped_shards,
            "worker_skipped_rows": overall_limits.skipped_rows,
            "worker_skipped_sequences": overall_limits.skipped_sequences,
        }
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
    ) -> StreamingRunTelemetry:
        if peak_gpu_mb is None:
            return telemetry
        return replace(telemetry, max_gpu_memory_mb=peak_gpu_mb)

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
    ) -> Path:
        artifact_path = self._output_dir / f"{plan_id}_logits.npz"
        np.savez_compressed(
            artifact_path,
            z_train=np.asarray(z_train, dtype=np.float32).reshape(-1),
            z_val=np.asarray(z_val, dtype=np.float32).reshape(-1),
            y_val=np.asarray(y_val, dtype=np.float32).reshape(-1),
        )
        return artifact_path

    def _compute_metrics(self, fit_result: StreamingFitResult) -> dict[str, float]:
        metric_name = self.config.validation_metric.strip().lower()
        y_val = np.asarray(fit_result.y_val, dtype=np.float64).reshape(-1)
        logits = np.asarray(fit_result.z_val, dtype=np.float64).reshape(-1)
        if y_val.size == 0 or logits.size == 0:
            return {}

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
        metrics["log_loss"] = binary_logloss(y_val, probabilities)
        metrics["brier_score"] = float(np.mean((probabilities - y_val) ** 2))
        metrics["calibration_ece_20"] = expected_calibration_error(probabilities, y_val, bins=20)
        metrics["calibration_ece_50"] = expected_calibration_error(probabilities, y_val, bins=50)

        if metric_name not in metrics:
            metrics[metric_name] = metrics.get("roc_auc", 0.0)
        return metrics


__all__ = [
    "LightningStreamingWorker",
]
