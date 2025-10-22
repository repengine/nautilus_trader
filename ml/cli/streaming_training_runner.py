#!/usr/bin/env python3
"""Run the event-driven streaming training topology end-to-end."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import time
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from ml.common.metrics_bootstrap import get_counter
from ml.common.subprocess_utils import SubprocessExecutionError
from ml.common.subprocess_utils import run_command
from ml.config.bus import MessageBusConfig
from ml.config.events import EventStatus
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.config.streaming_pipeline import TrainingOrchestratorConfig
from ml.consumers.streaming_training_service import StreamingTrainingPersistenceService
from ml.evaluation.metrics import binary_logloss
from ml.evaluation.metrics import expected_calibration_error
from ml.evaluation.metrics import pr_auc
from ml.evaluation.metrics import roc_auc
from ml.registry import DataRequirements
from ml.registry import ModelManifest
from ml.registry import ModelRegistry
from ml.registry import ModelRole
from ml.registry.feature_registry import compute_schema_hash
from ml.registry.model_registry import USE_LEGACY as REGISTRY_USE_LEGACY
from ml.training.event_driven.dataset_service import StreamingDatasetPlanner
from ml.training.event_driven.orchestrator import InMemoryStreamingOrchestrator
from ml.training.event_driven.payloads import build_plan_message
from ml.training.event_driven.payloads import build_result_message
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.event_driven.worker import LightningStreamingWorker
from ml.training.teacher.streaming_loader import TFTStreamingConfig


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus counters (idempotent thanks to bootstrap)
# ---------------------------------------------------------------------------

RUN_COUNTER = get_counter(
    "ml_streaming_training_runner_runs_total",
    "Number of streaming plans launched by the runner grouped by outcome.",
    labelnames=("status",),
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class FeatureLayout:
    """Feature layout derived from dataset metadata."""

    feature_names: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    feature_schema: Mapping[str, str]


@dataclass(slots=True, frozen=True)
class DatasetSpecification:
    """Dataset configuration required to build streaming requests."""

    dataset_id: str
    dataset_dir: Path
    metadata: Mapping[str, Any]
    report: Mapping[str, Any]
    streaming_config: TFTStreamingConfig
    feature_layout: FeatureLayout


@dataclass(slots=True)
class RunnerConfig:
    """Configuration parsed from CLI/environment for the streaming runner."""

    dataset: DatasetSpecification
    planner: DatasetServiceConfig
    worker: StreamingWorkerConfig
    orchestrator: TrainingOrchestratorConfig
    state_path: Path
    output_dir: Path
    registry_root: Path
    logits_key: str
    plan_interval_seconds: float
    max_plans: int | None
    promotion_threshold: float | None
    promotion_command: tuple[str, ...] | None
    promotion_checks: tuple[PromotionMetricCheck, ...]
    pipeline_signature: str
    pipeline_version: str
    persist_snapshot: bool


@dataclass(slots=True, frozen=True)
class PromotionMetricCheck:
    """Comparison rule applied to cohort metrics before promotion."""

    metric: str
    comparator: str
    threshold: float

    def evaluate(self, metrics: Mapping[str, float]) -> bool:
        """Return True when the metric satisfies the comparator."""
        value = metrics.get(self.metric)
        if value is None:
            return False
        if self.comparator == "ge":
            return value >= self.threshold
        if self.comparator == "le":
            return value <= self.threshold
        raise ValueError(f"Unsupported comparator {self.comparator!r}")


REQUIRED_MANIFEST_METRICS: tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "pr_auc_multiple",
    "log_loss",
    "brier_score",
    "positive_rate",
    "calibration_ece_20",
    "calibration_ece_50",
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _load_json(path: Path, *, description: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must be a JSON object at {path}")
    return payload


def _coerce_limit(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value) if int(value) > 0 else None


def _parse_metric_check(raw: str) -> PromotionMetricCheck:
    """Parse promotion metric checks of the form ``metric>=value`` or ``metric<=value``."""
    text = raw.strip()
    comparator: str
    if ">=" in text:
        metric, threshold = text.split(">=", 1)
        comparator = "ge"
    elif "<=" in text:
        metric, threshold = text.split("<=", 1)
        comparator = "le"
    else:
        raise argparse.ArgumentTypeError("Expected comparator '>=' or '<=' in promotion metric check.")
    metric_name = metric.strip().lower()
    if not metric_name:
        raise argparse.ArgumentTypeError("Metric name cannot be empty in promotion metric check.")
    try:
        threshold_value = float(threshold.strip())
    except ValueError as exc:  # pragma: no cover - argparse surfaces error
        raise argparse.ArgumentTypeError(f"Invalid threshold '{threshold}' for promotion metric check.") from exc
    return PromotionMetricCheck(metric=metric_name, comparator=comparator, threshold=threshold_value)


def _normalize_metric_value(value: Any) -> float | None:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_metrics(result_metrics: Mapping[str, Any], artifact_path: Path) -> dict[str, float]:
    """Ensure required metrics are present and JSON-serialisable."""
    metrics = {
        str(key).lower(): numeric
        for key, val in result_metrics.items()
        if (numeric := _normalize_metric_value(val)) is not None
    }

    missing = [metric for metric in REQUIRED_MANIFEST_METRICS if metric not in metrics]
    if missing:
        try:
            with np.load(artifact_path) as payload:
                logits = np.asarray(payload["z_val"], dtype=np.float64).reshape(-1)
                targets = np.asarray(payload["y_val"], dtype=np.float64).reshape(-1)
        except (FileNotFoundError, KeyError, OSError, ValueError):
            logger.error(
                "metrics_backfill_failed",
                extra={
                    "artifact_path": str(artifact_path),
                    "missing_metrics": missing,
                },
                exc_info=True,
            )
            for name in missing:
                metrics.setdefault(name, 0.0)
        else:
            if logits.size == 0 or targets.size == 0:
                for name in missing:
                    metrics.setdefault(name, 0.0)
            else:
                logits = np.clip(logits, -60.0, 60.0)
                probabilities = 1.0 / (1.0 + np.exp(-logits))
                prevalence = float(np.mean(targets)) if targets.size else 0.0
                metrics.setdefault("roc_auc", roc_auc(targets, probabilities))
                pr_value = metrics.get("pr_auc")
                if pr_value is None:
                    pr_value = pr_auc(targets, probabilities)
                    metrics["pr_auc"] = pr_value
                metrics.setdefault("positive_rate", prevalence)
                metrics.setdefault(
                    "pr_auc_multiple",
                    pr_value / prevalence if prevalence > 0.0 else 0.0,
                )
                metrics.setdefault("log_loss", binary_logloss(targets, probabilities))
                metrics.setdefault(
                    "brier_score",
                    float(np.mean((probabilities - targets) ** 2)),
                )
                metrics.setdefault(
                    "calibration_ece_20",
                    expected_calibration_error(probabilities, targets, bins=20),
                )
                metrics.setdefault(
                    "calibration_ece_50",
                    expected_calibration_error(probabilities, targets, bins=50),
                )
                newly_computed = sorted(
                    {name for name in REQUIRED_MANIFEST_METRICS if name in metrics and name not in result_metrics},
                )
                if newly_computed:
                    logger.info(
                        "metrics_backfilled",
                        extra={
                            "artifact_path": str(artifact_path),
                            "computed_metrics": newly_computed,
                        },
                    )

    return {name: float(value) for name, value in metrics.items()}


def _env_flag(name: str, default: bool) -> bool:
    """Return truthy value derived from environment variables."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _build_feature_layout(metadata: Mapping[str, Any]) -> FeatureLayout:
    columns = metadata.get("column_info", {})
    if not isinstance(columns, Mapping):
        raise ValueError("metadata column_info must be an object")

    categorical = tuple(str(value) for value in columns.get("categorical_columns", ()))
    static_reals = tuple(str(value) for value in columns.get("static_reals", ()))
    known_reals_raw = tuple(str(value) for value in columns.get("time_varying_known_reals", ()))
    known_reals = tuple(value for value in known_reals_raw if value != "time_index")
    unknown_reals = tuple(str(value) for value in columns.get("time_varying_unknown_reals", ()))
    vintage_age = tuple(str(value) for value in columns.get("vintage_age_columns", ()))

    numeric = tuple(
        dict.fromkeys(static_reals + known_reals + unknown_reals + vintage_age + ("y",)),
    )
    feature_names = tuple(
        dict.fromkeys(
            static_reals + known_reals + unknown_reals + vintage_age + categorical,
        ),
    )

    schema: dict[str, str] = {}
    for name in feature_names:
        schema[name] = "categorical" if name in categorical else "float32"
    return FeatureLayout(
        feature_names=feature_names,
        numeric_columns=numeric,
        categorical_columns=categorical,
        feature_schema=schema,
    )


def _build_streaming_config(
    metadata: Mapping[str, Any],
    *,
    batch_size: int,
    dataloader_workers: int,
    max_total_rows: int | None,
    max_total_sequences: int | None,
    max_shards: int | None,
    max_encoder_length: int,
    max_prediction_length: int,
    include_macro: bool,
    include_calendar: bool,
    include_events: bool,
    include_earnings: bool,
    include_micro: bool,
    include_l2: bool,
    include_macro_revisions: bool,
) -> TFTStreamingConfig:
    columns = metadata.get("column_info", {})
    if not isinstance(columns, Mapping):
        raise ValueError("metadata column_info must be an object")

    lags_map = metadata.get("publication_lags", {})

    def _lag_value(key: str, default: int) -> int:
        raw = lags_map.get(key, default)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return default

    macro_lag_days = _lag_value("macro_lag_days", 1 if include_macro else 0)
    earnings_lag_days = _lag_value("earnings_lag_days", 1 if include_earnings else 0)
    events_notice_minutes = _lag_value("events_notice_minutes", 0)

    layout = _build_feature_layout(metadata)
    return TFTStreamingConfig(
        time_idx_col=str(columns.get("time_idx_col", "time_index")),
        group_id_col=str(columns.get("group_id_col", "instrument_id")),
        target_col=str(columns.get("target_col", "y")),
        static_categoricals=layout.categorical_columns,
        static_reals=tuple(str(value) for value in columns.get("static_reals", ())),
        time_varying_known_reals=tuple(
            value for value in columns.get("time_varying_known_reals", ()) if value != "time_index"
        ),
        time_varying_unknown_reals=tuple(
            str(value) for value in columns.get("time_varying_unknown_reals", ())
        ),
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=batch_size,
        drop_last=False,
        shuffle_shards=False,
        seed=7,
        num_workers=dataloader_workers,
        max_total_rows=_coerce_limit(max_total_rows),
        max_total_sequences=_coerce_limit(max_total_sequences),
        max_shards=_coerce_limit(max_shards),
        include_macro=include_macro,
        include_calendar=include_calendar,
        include_events=include_events,
        include_earnings=include_earnings,
        include_micro=include_micro,
        include_l2=include_l2,
        include_macro_revisions=include_macro_revisions,
        macro_lag_days=macro_lag_days,
        earnings_lag_days=earnings_lag_days,
        events_notice_minutes=events_notice_minutes,
    )


def _resolve_dataset_spec(
    dataset_dir: Path,
    *,
    batch_size: int,
    dataloader_workers: int,
    max_total_rows: int | None,
    max_total_sequences: int | None,
    max_shards: int | None,
    max_encoder_length: int,
    max_prediction_length: int,
    include_macro: bool,
    include_calendar: bool,
    include_events: bool,
    include_earnings: bool,
    include_micro: bool,
    include_l2: bool,
    include_macro_revisions: bool,
) -> DatasetSpecification:
    metadata_path = dataset_dir / "dataset_metadata.json"
    report_path = dataset_dir / "report.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"dataset metadata missing at {metadata_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"dataset report missing at {report_path}")

    metadata = _load_json(metadata_path, description="dataset metadata")
    report = _load_json(report_path, description="dataset report")
    streaming_config = _build_streaming_config(
        metadata,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        max_total_rows=max_total_rows,
        max_total_sequences=max_total_sequences,
        max_shards=max_shards,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        include_macro=include_macro,
        include_calendar=include_calendar,
        include_events=include_events,
        include_earnings=include_earnings,
        include_micro=include_micro,
        include_l2=include_l2,
        include_macro_revisions=include_macro_revisions,
    )
    layout = _build_feature_layout(metadata)
    dataset_id = str(metadata.get("dataset_id", dataset_dir.name))
    return DatasetSpecification(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        metadata=metadata,
        report=report,
        streaming_config=streaming_config,
        feature_layout=layout,
    )


def _parquet_shape(parquet_path: Path) -> tuple[int, int]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    num_rows = parquet_file.metadata.num_rows
    num_columns = parquet_file.metadata.num_columns
    return num_rows, num_columns


def _extract_target_stats(report: Mapping[str, Any]) -> Mapping[str, float]:
    coverage = report.get("feature_coverage", {})
    if not isinstance(coverage, Mapping):
        return {}
    overall = coverage.get("overall", {})
    if not isinstance(overall, Mapping):
        return {}
    try:
        positives = float(overall.get("positives", 0.0))
        total = float(overall.get("total", 0.0))
        rate = float(overall.get("positive_rate", 0.0))
    except (TypeError, ValueError):
        positives = total = rate = 0.0
    return {"positives": positives, "total": total, "positive_rate": rate}


def _make_dataset_plan_request(spec: DatasetSpecification) -> DatasetPlanRequest:
    parquet_path = spec.dataset_dir / "dataset_with_vintage_age.parquet"
    layout = spec.feature_layout
    return DatasetPlanRequest(
        dataset_id=spec.dataset_id,
        streaming_config=spec.streaming_config,
        feature_names=layout.feature_names,
        categorical_columns=layout.categorical_columns,
        numeric_columns=layout.numeric_columns,
        parquet_path=parquet_path,
    )


class RecordingStreamingWorker(LightningStreamingWorker):
    """Lightning worker that retains the last TrainingResultEvent."""

    def __init__(self, config: StreamingWorkerConfig, *, output_dir: Path) -> None:
        super().__init__(config, output_dir=output_dir)
        self._last_result: TrainingResultEvent | None = None

    def run(self, plan: DatasetPlanEvent) -> TrainingResultEvent:
        result = super().run(plan)
        self._last_result = result
        return result

    @property
    def last_result(self) -> TrainingResultEvent | None:
        return self._last_result


def _compute_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_artifact(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(source, destination)
    return destination


def _build_manifest_payload(
    *,
    spec: DatasetSpecification,
    plan: DatasetPlanEvent,
    result: TrainingResultEvent,
    worker_config: StreamingWorkerConfig,
    registry_path: Path,
    artifact_path: Path,
    artifact_digest: str,
    state_path: Path,
    dataset_shape: tuple[int, int],
    target_stats: Mapping[str, float],
    model_id: str,
) -> dict[str, Any]:
    telemetry = result.telemetry.as_dict()
    train_stats = cast(Mapping[str, Any], telemetry.get("train", {}))
    validation_stats = cast(Mapping[str, Any], telemetry.get("validation", {}))
    resources = cast(Mapping[str, Any], telemetry.get("resources", {}))
    dataset_paths = {
        "dataset_parquet": str(spec.dataset_dir / "dataset.parquet"),
        "dataset_with_vintage_age": str(spec.dataset_dir / "dataset_with_vintage_age.parquet"),
        "metadata": str(spec.dataset_dir / "dataset_metadata.json"),
        "report_md": str(spec.dataset_dir / "report.md"),
        "report_json": str(spec.dataset_dir / "report.json"),
    }
    dataset_block = {
        "dataset_id": spec.dataset_id,
        "build_timestamp": spec.metadata.get("build_ts"),
        "shape": list(dataset_shape),
        "target": target_stats,
        "paths": dataset_paths,
        "vintage_age_columns": list(spec.metadata.get("column_info", {}).get("vintage_age_columns", ())),
    }

    training_config = {
        "max_total_rows": worker_config.max_total_rows,
        "max_total_sequences": worker_config.max_total_sequences,
        "max_shards": worker_config.max_shards,
        "train_fraction": worker_config.train_fraction,
        "max_epochs": worker_config.max_epochs,
    }

    cohort_block = {
        "plan_id": plan.plan_id,
        "dataset_id": plan.dataset_id,
        "state_snapshot": str(state_path),
        "completed_at": result.completed_at.isoformat(),
        "metrics": result.metrics,
        "artifact_paths": {
            worker_config.logits_artifact_key: result.artifact_paths[worker_config.logits_artifact_key],
        },
        "model_registry": {
            "model_id": model_id,
            "registry_root": str(registry_path),
            "artifact_rel_path": str(artifact_path.relative_to(registry_path)),
            "artifact_sha256": artifact_digest,
        },
        "training_config": training_config,
        "telemetry": {
            "caps": plan.caps,
            "selected_rows": {
                "total": plan.metadata_summary.total_rows,
                "train": train_stats.get("selected_rows"),
                "validation": validation_stats.get("selected_rows"),
            },
            "selected_sequences": {
                "train": train_stats.get("selected_sequences"),
                "validation": validation_stats.get("selected_sequences"),
            },
            "shards": {
                "total": plan.metadata_summary.total_shards,
                "max_rows": plan.metadata_summary.max_shard_rows,
            },
            "resources": dict(resources),
        },
    }

    return {
        "dataset": dataset_block,
        "cohort_run": cohort_block,
    }


def _register_model(
    *,
    registry_root: Path,
    artifact_path: Path,
    artifact_digest: str,
    spec: DatasetSpecification,
    plan: DatasetPlanEvent,
    result: TrainingResultEvent,
    worker_config: StreamingWorkerConfig,
    pipeline_signature: str,
    pipeline_version: str,
) -> str:
    if REGISTRY_USE_LEGACY:
        logger.warning("Legacy model registry is active; runner uses facade APIs.")
    registry = ModelRegistry(registry_root)
    feature_names = list(spec.feature_layout.feature_schema.keys())
    feature_dtypes = [spec.feature_layout.feature_schema[name] for name in feature_names]
    feature_schema_hash = compute_schema_hash(feature_names, feature_dtypes, pipeline_signature)
    manifest = ModelManifest(
        model_id="",
        role=ModelRole.TEACHER,
        data_requirements=DataRequirements.STREAMING,
        architecture="tft_streaming_teacher",
        feature_schema=dict(spec.feature_layout.feature_schema),
        feature_schema_hash=feature_schema_hash,
        parent_id=None,
        training_config={
            "plan_id": plan.plan_id,
            "max_total_rows": worker_config.max_total_rows,
            "max_total_sequences": worker_config.max_total_sequences,
            "max_shards": worker_config.max_shards,
            "train_fraction": worker_config.train_fraction,
            "selected_rows_total": plan.metadata_summary.total_rows,
            "selected_rows_train": result.telemetry.train.selected_rows,
            "selected_rows_validation": result.telemetry.validation.selected_rows,
            "max_gpu_memory_mb": result.telemetry.max_gpu_memory_mb,
            "max_epochs": worker_config.max_epochs,
        },
        performance_metrics=dict(result.metrics),
        deployment_constraints={
            "max_inference_latency_ms": 50.0,
            "memory_limit_mb": 1024.0,
        },
        version="1.0.0",
        serveable=False,
        artifact_format="npz",
        feature_set_id=None,
        pipeline_signature=pipeline_signature,
        pipeline_version=pipeline_version,
        decision_policy=None,
        decision_config={},
        artifact_sha256_digest=artifact_digest,
    )
    model_id = registry.register_model(
        model_path=artifact_path,
        manifest=manifest,
        auto_deploy=False,
    )
    logger.info(
        "registered_teacher",
        extra={
            "model_id": model_id,
            "artifact": str(artifact_path),
            "metrics": result.metrics,
        },
    )
    return model_id


# ---------------------------------------------------------------------------
# Runner class
# ---------------------------------------------------------------------------


class StreamingTrainingRunner:
    """Coordinate planner → worker → registry flow for streaming training."""

    def __init__(self, config: RunnerConfig) -> None:
        self._config = config
        self._stop_requested = False
        self._message_bus_cfg = MessageBusConfig.from_env()

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers to stop the runner gracefully."""

        def _handler(signum: int, _frame: Any) -> None:
            logger.info("received signal %s, stopping runner", signum)
            self._stop_requested = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def run(self) -> None:
        """Execute streaming cohorts according to configuration."""
        config = self._config
        planner = StreamingDatasetPlanner(config.planner)
        worker = RecordingStreamingWorker(config.worker, output_dir=config.output_dir)
        orchestrator = InMemoryStreamingOrchestrator(
            config=config.orchestrator,
            planner=planner,
            worker=worker,
            bus_config=self._message_bus_cfg,
            state_path=config.state_path.parent / "streaming_orchestrator_state.json",
        )
        cleared = orchestrator.clear_backlog(include_active=True)
        if cleared:
            logger.info("cleared_stale_orchestrator_state", extra={"plan_ids": list(cleared)})

        plans_run = 0
        while not self._stop_requested:
            if config.max_plans is not None and plans_run >= config.max_plans:
                break
            plan_request = _make_dataset_plan_request(config.dataset)
            plan_event = orchestrator.enqueue_training(plan_request)
            result_event = worker.last_result
            if result_event is None:
                RUN_COUNTER.labels(status=EventStatus.FAILED.value).inc()
                raise RuntimeError("worker did not produce a training result event")

            self._handle_result(plan_event, result_event)
            plans_run += 1

            interval = config.plan_interval_seconds
            if interval <= 0.0:
                break
            logger.info("sleeping before next plan", extra={"seconds": interval})
            time.sleep(interval)

    def _handle_result(self, plan: DatasetPlanEvent, result: TrainingResultEvent) -> None:
        config = self._config
        artifact_local_path = Path(result.artifact_paths[config.logits_key]).resolve()
        if not artifact_local_path.exists():
            raise FileNotFoundError(f"worker artifact not found at {artifact_local_path}")

        metrics = _normalize_metrics(result.metrics, artifact_local_path)
        result = replace(result, metrics=metrics)

        registry_artifact = config.registry_root / "staging" / artifact_local_path.name
        _copy_artifact(artifact_local_path, registry_artifact)
        artifact_digest = _compute_sha256(registry_artifact)

        model_id = _register_model(
            registry_root=config.registry_root,
            artifact_path=registry_artifact,
            artifact_digest=artifact_digest,
            spec=config.dataset,
            plan=plan,
            result=result,
            worker_config=config.worker,
            pipeline_signature=config.pipeline_signature,
            pipeline_version=config.pipeline_version,
        )

        dataset_shape = _parquet_shape(config.dataset.dataset_dir / "dataset_with_vintage_age.parquet")
        target_stats = _extract_target_stats(config.dataset.report)
        manifest_payload = _build_manifest_payload(
            spec=config.dataset,
            plan=plan,
            result=result,
            worker_config=config.worker,
            registry_path=config.registry_root,
            artifact_path=registry_artifact,
            artifact_digest=artifact_digest,
            state_path=config.state_path,
            dataset_shape=dataset_shape,
            target_stats=target_stats,
            model_id=model_id,
        )

        manifest_path = config.output_dir / f"{plan.plan_id}_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info(
            "manifest_written",
            extra={
                "manifest_path": str(manifest_path),
                "model_id": model_id,
                "metrics": metrics,
            },
        )

        promotion_gate = config.promotion_threshold
        status_label = result.status.value
        threshold_met = True
        primary_metric_name = config.worker.validation_metric
        primary_value = metrics.get(primary_metric_name)

        if promotion_gate is not None:
            if primary_value is None:
                logger.warning(
                    "promotion_metric_missing",
                    extra={
                        "metric": primary_metric_name,
                        "threshold": promotion_gate,
                        "plan_id": plan.plan_id,
                    },
                )
                threshold_met = False
            elif primary_value >= promotion_gate:
                logger.info(
                    "promotion_threshold_met",
                    extra={
                        "metric": primary_metric_name,
                        "metric_value": primary_value,
                        "threshold": promotion_gate,
                        "plan_id": plan.plan_id,
                        "model_id": model_id,
                    },
                )
            else:
                logger.info(
                    "promotion_threshold_not_met",
                    extra={
                        "metric": primary_metric_name,
                        "metric_value": primary_value,
                        "threshold": promotion_gate,
                        "plan_id": plan.plan_id,
                    },
                )
                threshold_met = False

        if threshold_met and config.promotion_checks:
            failed_checks: list[PromotionMetricCheck] = []
            for check in config.promotion_checks:
                if not check.evaluate(metrics):
                    failed_checks.append(check)
            if failed_checks:
                threshold_met = False
                for check in failed_checks:
                    logger.info(
                        "promotion_secondary_metric_not_met",
                        extra={
                            "metric": check.metric,
                            "comparator": check.comparator,
                            "threshold": check.threshold,
                            "observed": metrics.get(check.metric),
                            "plan_id": plan.plan_id,
                        },
                    )
            else:
                logger.info(
                    "promotion_secondary_metrics_met",
                    extra={
                        "plan_id": plan.plan_id,
                        "checks": [
                            {
                                "metric": check.metric,
                                "comparator": check.comparator,
                                "threshold": check.threshold,
                                "observed": metrics.get(check.metric),
                            }
                            for check in config.promotion_checks
                        ],
                    },
                )

        if threshold_met:
            status_label = "promotable"

        RUN_COUNTER.labels(status=status_label).inc()

        if threshold_met and config.promotion_command is not None:
            self._run_promotion_command(
                config.promotion_command,
                manifest_path=manifest_path,
                worker_artifact=artifact_local_path,
                registry_artifact=registry_artifact,
                model_id=model_id,
                plan=plan,
            )

        if config.persist_snapshot:
            snapshot_service = StreamingTrainingPersistenceService.create(
                state_path=config.state_path,
            )
            snapshot_service.handle(
                f"events.ml.DATASET_PLANNED.{plan.dataset_id}",
                build_plan_message(plan).as_dict(),
            )
            snapshot_service.handle(
                f"events.ml.MODEL_TRAINING_COMPLETED.{plan.dataset_id}",
                build_result_message(result).as_dict(),
            )

    def _run_promotion_command(
        self,
        command: tuple[str, ...],
        *,
        manifest_path: Path,
        worker_artifact: Path,
        registry_artifact: Path,
        model_id: str,
        plan: DatasetPlanEvent,
    ) -> None:
        """Execute the configured promotion command with cohort context placeholders."""
        context = {
            "manifest": str(manifest_path),
            "logits": str(worker_artifact),
            "worker_artifact": str(worker_artifact),
            "registry_artifact": str(registry_artifact),
            "model_id": model_id,
            "plan_id": plan.plan_id,
            "dataset_id": plan.dataset_id,
        }
        try:
            resolved = tuple(part.format(**context) for part in command)
        except KeyError as exc:  # pragma: no cover - defensive guard against misconfiguration
            logger.error(
                "promotion_command_placeholder_missing",
                extra={
                    "missing_placeholder": str(exc),
                    "plan_id": plan.plan_id,
                },
                exc_info=True,
            )
            return

        try:
            completed = run_command(resolved, check=True, capture_output=False, text=True)
            logger.info(
                "promotion_command_succeeded",
                extra={
                    "command": shlex.join(resolved),
                    "plan_id": plan.plan_id,
                    "model_id": model_id,
                    "returncode": completed.returncode,
                },
            )
        except SubprocessExecutionError as exc:
            logger.error(
                "promotion_command_failed",
                extra={
                    "command": shlex.join(resolved),
                    "plan_id": plan.plan_id,
                    "model_id": model_id,
                    "returncode": exc.returncode,
                },
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing dataset_with_vintage_age.parquet and dataset_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where streaming artifacts/logits are written.",
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path(os.environ.get("ML_MODEL_REGISTRY_PATH", "~/.nautilus/ml/models")).expanduser(),
        help="Registry root directory used for model artifacts (default: ~/.nautilus/ml/models).",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("ml_out/streaming_training_state.json"),
        help="Path for streaming state snapshot (used by dashboard control plane).",
    )
    parser.add_argument(
        "--logits-key",
        type=str,
        default="logits",
        help="Artifact key expected from the worker result payload.",
    )
    parser.add_argument(
        "--plan-interval-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait between plans (0 runs once).",
    )
    parser.add_argument(
        "--max-plans",
        type=int,
        default=1,
        help="Maximum number of cohorts to run (<=0 for unlimited).",
    )
    parser.add_argument(
        "--promotion-threshold",
        type=float,
        default=None,
        help="Validation metric threshold required for promotion (optional).",
    )
    parser.add_argument(
        "--promotion-command",
        type=str,
        default=None,
        help=(
            "Optional command invoked when promotion threshold is met. "
            "Placeholders {manifest}, {logits}, {model_id}, {plan_id}, and {dataset_id} are expanded per cohort."
        ),
    )
    parser.add_argument(
        "--promotion-metric-check",
        action="append",
        type=_parse_metric_check,
        default=None,
        help=(
            "Additional promotion constraint (repeatable). "
            "Specify as metric>=value for minima or metric<=value for maxima."
        ),
    )
    parser.add_argument(
        "--pipeline-signature",
        type=str,
        default="tft_streaming_cohort_v1",
        help="Pipeline signature recorded in the registry manifest.",
    )
    parser.add_argument(
        "--pipeline-version",
        type=str,
        default=datetime.utcnow().date().isoformat(),
        help="Pipeline version string stored in registry manifests.",
    )
    parser.add_argument(
        "--persist-snapshot",
        action="store_true",
        help="Write streaming state snapshot locally in addition to redis persistence.",
    )


def _add_planner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shard-row-budget",
        type=int,
        default=200_000,
        help="Shard row budget enforced during planning.",
    )


def _add_streaming_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-size",
        type=int,
        default=48,
        help="Streaming dataloader batch size.",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=0,
        help="Number of workers for streaming dataloaders.",
    )
    parser.add_argument(
        "--max-total-rows",
        type=int,
        default=120_000,
        help="Maximum rows per plan (<=0 disables limit).",
    )
    parser.add_argument(
        "--max-total-sequences",
        type=int,
        default=90_000,
        help="Maximum sequences per plan (<=0 disables limit).",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=32,
        help="Maximum shards per plan (<=0 disables limit).",
    )
    parser.add_argument(
        "--max-encoder-length",
        type=int,
        default=192,
        help="Encoder length for streaming dataset.",
    )
    parser.add_argument(
        "--max-prediction-length",
        type=int,
        default=24,
        help="Prediction horizon for streaming dataset.",
    )

    include_macro_default = _env_flag("ML_STREAMING_INCLUDE_MACRO", False)
    parser.add_argument(
        "--include-macro",
        dest="include_macro",
        action="store_true",
        default=include_macro_default,
        help="Enable macro feature augmentation (default derives from ML_STREAMING_INCLUDE_MACRO).",
    )
    parser.add_argument(
        "--no-include-macro",
        dest="include_macro",
        action="store_false",
        help="Disable macro feature augmentation.",
    )

    include_calendar_default = _env_flag("ML_STREAMING_INCLUDE_CALENDAR", False)
    parser.add_argument(
        "--include-calendar",
        dest="include_calendar",
        action="store_true",
        default=include_calendar_default,
        help="Enable calendar feature augmentation (default derives from ML_STREAMING_INCLUDE_CALENDAR).",
    )
    parser.add_argument(
        "--no-include-calendar",
        dest="include_calendar",
        action="store_false",
        help="Disable calendar feature augmentation.",
    )

    include_events_default = _env_flag("ML_STREAMING_INCLUDE_EVENTS", False)
    parser.add_argument(
        "--include-events",
        dest="include_events",
        action="store_true",
        default=include_events_default,
        help="Enable known-future event features (default derives from ML_STREAMING_INCLUDE_EVENTS).",
    )
    parser.add_argument(
        "--no-include-events",
        dest="include_events",
        action="store_false",
        help="Disable known-future event features.",
    )

    include_earnings_default = _env_flag("ML_STREAMING_INCLUDE_EARNINGS", False)
    parser.add_argument(
        "--include-earnings",
        dest="include_earnings",
        action="store_true",
        default=include_earnings_default,
        help="Enable earnings augmentation (default derives from ML_STREAMING_INCLUDE_EARNINGS).",
    )
    parser.add_argument(
        "--no-include-earnings",
        dest="include_earnings",
        action="store_false",
        help="Disable earnings augmentation.",
    )

    include_micro_default = _env_flag("ML_STREAMING_INCLUDE_MICRO", False)
    parser.add_argument(
        "--include-micro",
        dest="include_micro",
        action="store_true",
        default=include_micro_default,
        help="Enable microstructure feature augmentation (default derives from ML_STREAMING_INCLUDE_MICRO).",
    )
    parser.add_argument(
        "--no-include-micro",
        dest="include_micro",
        action="store_false",
        help="Disable microstructure feature augmentation.",
    )

    include_l2_default = _env_flag("ML_STREAMING_INCLUDE_L2", False)
    parser.add_argument(
        "--include-l2",
        dest="include_l2",
        action="store_true",
        default=include_l2_default,
        help="Enable L2 order book feature augmentation (default derives from ML_STREAMING_INCLUDE_L2).",
    )
    parser.add_argument(
        "--no-include-l2",
        dest="include_l2",
        action="store_false",
        help="Disable L2 order book feature augmentation.",
    )

    include_macro_revisions_default = _env_flag("ML_STREAMING_INCLUDE_MACRO_REVISIONS", False)
    parser.add_argument(
        "--include-macro-revisions",
        dest="include_macro_revisions",
        action="store_true",
        default=include_macro_revisions_default,
        help=(
            "Enable macro revision feature augmentation (default derives from ML_STREAMING_INCLUDE_MACRO_REVISIONS)."
        ),
    )
    parser.add_argument(
        "--no-include-macro-revisions",
        dest="include_macro_revisions",
        action="store_false",
        help="Disable macro revision augmentation.",
    )


def _add_worker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Maximum epochs per cohort (>=1).",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=7_200,
        help="Runtime budget per cohort before partial status is recorded.",
    )
    parser.add_argument(
        "--max-retry-attempts",
        type=int,
        default=1,
        help="Number of retry attempts for the worker.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=0.0,
        help="Backoff between worker retries.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Training fraction when splitting metadata into train/validation shards.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="PyTorch Lightning accelerator argument (cpu|gpu|auto).",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices passed to the Lightning trainer.",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=30.0,
        help="Interval (seconds) for GPU memory sampling (<=0 disables).",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=int,
        default=120,
        help="Worker heartbeat interval for orchestrator state tracking.",
    )
    parser.add_argument(
        "--validation-metric",
        type=str,
        default="roc_auc",
        help="Validation metric recorded in result events.",
    )


def _add_orchestrator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--command-topic",
        type=str,
        default="",
        help="Optional override for orchestrator command topic (format string allowed).",
    )
    parser.add_argument(
        "--result-topic",
        type=str,
        default="",
        help="Optional override for orchestrator result topic (format string allowed).",
    )
    parser.add_argument(
        "--heartbeat-topic",
        type=str,
        default="",
        help="Optional override for orchestrator heartbeat topic (format string allowed).",
    )
    parser.add_argument(
        "--max-in-flight-plans",
        type=int,
        default=8,
        help="Maximum concurrent plans tracked by orchestrator.",
    )
    parser.add_argument(
        "--dataset-retry-limit",
        type=int,
        default=2,
        help="Retry budget for dataset plans when worker heartbeats stall.",
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=int,
        default=600,
        help="Timeout for worker heartbeats before orchestrator schedules retry.",
    )
    parser.add_argument(
        "--retry-window-seconds",
        type=int,
        default=300,
        help="Cooldown window before orchestrator retries a stalled plan.",
    )
    parser.add_argument(
        "--max-plan-age-seconds",
        type=int,
        default=7_200,
        help="Maximum lifetime of a plan before it is dropped from orchestrator backlog.",
    )
    parser.add_argument(
        "--saturation-heartbeat-limit",
        type=int,
        default=5,
        help="Heartbeats without progress before marking plan as saturated.",
    )
    parser.add_argument(
        "--backlog-warning-threshold",
        type=int,
        default=10,
        help="Backlog threshold used to derive heartbeat hint and warnings.",
    )
    parser.add_argument(
        "--publish-retry-attempts",
        type=int,
        default=3,
        help="Number of attempts made when publishing events to the message bus.",
    )
    parser.add_argument(
        "--publish-retry-delay-seconds",
        type=float,
        default=0.5,
        help="Delay (seconds) between message-bus publish retries.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run continuous event-driven streaming training cohorts.",
    )
    _add_common_args(parser)
    _add_planner_args(parser)
    _add_streaming_config_args(parser)
    _add_worker_args(parser)
    _add_orchestrator_args(parser)
    return parser.parse_args(argv)


def _build_runner_config(args: argparse.Namespace) -> RunnerConfig:
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_root = args.registry_root.expanduser().resolve()
    registry_root.mkdir(parents=True, exist_ok=True)
    state_path = args.state_path.expanduser().resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_spec = _resolve_dataset_spec(
        dataset_dir,
        batch_size=args.batch_size,
        dataloader_workers=args.dataloader_workers,
        max_total_rows=args.max_total_rows,
        max_total_sequences=args.max_total_sequences,
        max_shards=args.max_shards,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        include_macro=args.include_macro,
        include_calendar=args.include_calendar,
        include_events=args.include_events,
        include_earnings=args.include_earnings,
        include_micro=args.include_micro,
        include_l2=args.include_l2,
        include_macro_revisions=args.include_macro_revisions,
    )

    planner_config = DatasetServiceConfig(
        parquet_root=str(dataset_dir),
        shard_row_budget=int(args.shard_row_budget),
        max_total_rows=_coerce_limit(args.max_total_rows),
        max_total_sequences=_coerce_limit(args.max_total_sequences),
        max_shards=_coerce_limit(args.max_shards),
        include_macro=bool(args.include_macro),
        include_calendar=bool(args.include_calendar),
        include_events=bool(args.include_events),
        include_earnings=bool(args.include_earnings),
        include_micro=bool(args.include_micro),
        include_l2=bool(args.include_l2),
        include_macro_revisions=bool(args.include_macro_revisions),
    )

    worker_config = StreamingWorkerConfig(
        max_total_rows=_coerce_limit(args.max_total_rows),
        max_total_sequences=_coerce_limit(args.max_total_sequences),
        max_shards=_coerce_limit(args.max_shards),
        max_epochs=max(1, int(args.max_epochs)),
        max_runtime_seconds=int(args.max_runtime_seconds),
        heartbeat_interval_seconds=int(args.heartbeat_interval_seconds),
        max_retry_attempts=max(1, int(args.max_retry_attempts)),
        retry_backoff_seconds=float(args.retry_backoff_seconds),
        accelerator=str(args.accelerator),
        devices=max(1, int(args.devices)),
        train_fraction=float(args.train_fraction),
        logits_artifact_key=str(args.logits_key),
        validation_metric=str(args.validation_metric).lower(),
        gpu_memory_monitor_interval_seconds=(
            None if float(args.gpu_monitor_interval) <= 0.0 else float(args.gpu_monitor_interval)
        ),
    )

    orchestrator_config = TrainingOrchestratorConfig(
        command_topic=str(args.command_topic),
        result_topic=str(args.result_topic),
        heartbeat_topic=str(args.heartbeat_topic),
        max_in_flight_plans=max(1, int(args.max_in_flight_plans)),
        dataset_retry_limit=max(0, int(args.dataset_retry_limit)),
        worker_timeout_seconds=max(30, int(args.worker_timeout_seconds)),
        retry_window_seconds=max(1, int(args.retry_window_seconds)),
        max_plan_age_seconds=max(300, int(args.max_plan_age_seconds)),
        saturation_heartbeat_limit=max(1, int(args.saturation_heartbeat_limit)),
        backlog_warning_threshold=max(0, int(args.backlog_warning_threshold)),
        publish_retry_attempts=max(1, int(args.publish_retry_attempts)),
        publish_retry_delay_seconds=max(0.0, float(args.publish_retry_delay_seconds)),
    )

    max_plans_value: int | None
    max_plans_raw = int(args.max_plans)
    if max_plans_raw <= 0:
        max_plans_value = None
    else:
        max_plans_value = max_plans_raw

    promotion_command: tuple[str, ...] | None = None
    if args.promotion_command:
        tokens = tuple(shlex.split(str(args.promotion_command)))
        if tokens:
            promotion_command = tokens

    promotion_checks: tuple[PromotionMetricCheck, ...] = ()
    if args.promotion_metric_check:
        promotion_checks = tuple(args.promotion_metric_check)

    return RunnerConfig(
        dataset=dataset_spec,
        planner=planner_config,
        worker=worker_config,
        orchestrator=orchestrator_config,
        state_path=state_path,
        output_dir=output_dir,
        registry_root=registry_root,
        logits_key=str(args.logits_key),
        plan_interval_seconds=float(args.plan_interval_seconds),
        max_plans=max_plans_value,
        promotion_threshold=float(args.promotion_threshold) if args.promotion_threshold is not None else None,
        promotion_command=promotion_command,
        promotion_checks=promotion_checks,
        pipeline_signature=str(args.pipeline_signature),
        pipeline_version=str(args.pipeline_version),
        persist_snapshot=bool(args.persist_snapshot),
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    runner_config = _build_runner_config(args)
    runner = StreamingTrainingRunner(runner_config)
    runner.install_signal_handlers()
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
