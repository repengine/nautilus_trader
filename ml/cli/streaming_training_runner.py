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
from dataclasses import field
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
from ml.config.streaming_pipeline import AzureScheduledEventsConfig
from ml.config.streaming_pipeline import CurriculumGuardRule
from ml.config.streaming_pipeline import CurriculumScheduleConfig
from ml.config.streaming_pipeline import CurriculumStageConfig
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import EnsembleMemberConfig
from ml.config.streaming_pipeline import StreamingEnsembleConfig
from ml.config.streaming_pipeline import StreamingPromotionConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.config.streaming_pipeline import TrainingOrchestratorConfig
from ml.config.streaming_pipeline import parse_curriculum_guard_spec
from ml.config.streaming_pipeline import parse_curriculum_stage_spec
from ml.config.streaming_pipeline import parse_ensemble_member_spec
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
from ml.training.event_driven.azure_events import AzureScheduledEventsWatcher
from ml.training.event_driven.azure_events import ScheduledEventNotice
from ml.training.event_driven.dataset_service import StreamingDatasetPlanner
from ml.training.event_driven.orchestrator import AdaptiveSchedulingDeferred
from ml.training.event_driven.orchestrator import InMemoryStreamingOrchestrator
from ml.training.event_driven.payloads import build_plan_message
from ml.training.event_driven.payloads import build_result_message
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.event_driven.worker import LightningStreamingWorker
from ml.training.teacher.streaming_loader import PhaseOneFeatureSignals
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
    phase_one_signals: PhaseOneFeatureSignals = field(default_factory=PhaseOneFeatureSignals)


@dataclass(slots=True, frozen=True)
class DatasetSpecification:
    """Dataset configuration required to build streaming requests."""

    dataset_id: str
    dataset_dir: Path
    metadata: Mapping[str, Any]
    report: Mapping[str, Any]
    streaming_config: TFTStreamingConfig
    feature_layout: FeatureLayout
    phase_one_signals: PhaseOneFeatureSignals = field(default_factory=PhaseOneFeatureSignals)


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
    scheduled_events: AzureScheduledEventsConfig = field(default_factory=AzureScheduledEventsConfig)


@dataclass(slots=True, frozen=True)
class PromotionMetricCheck:
    """Comparison rule applied to cohort metrics before promotion."""

    metric: str
    comparator: str
    threshold: float
    absolute: bool = False

    def evaluate(self, metrics: Mapping[str, float]) -> bool:
        """Return True when the metric satisfies the comparator."""
        value = metrics.get(self.metric)
        if value is None:
            return False
        try:
            observed = float(abs(value) if self.absolute else value)
        except (TypeError, ValueError):
            return False
        if self.comparator == "ge":
            return observed >= self.threshold
        if self.comparator == "le":
            return observed <= self.threshold
        raise ValueError(f"Unsupported comparator {self.comparator!r}")


def _observed_metric_value(
    check: PromotionMetricCheck,
    metrics: Mapping[str, float],
) -> float | None:
    """Return the metric value used for evaluation, applying absolute when requested."""
    value = metrics.get(check.metric)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return abs(numeric) if check.absolute else numeric


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


def _coerce_positive_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0.0 else None


def _promotion_checks_from_config(config: StreamingPromotionConfig) -> tuple[PromotionMetricCheck, ...]:
    """Translate promotion configuration into metric checks."""
    return tuple(
        PromotionMetricCheck(
            metric=metric,
            comparator=comparator,
            threshold=threshold,
            absolute=absolute,
        )
        for metric, comparator, threshold, absolute in config.metric_rules()
    )


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
    metric_token = metric.strip()
    absolute = False
    if "|" in metric_token:
        name_part, modifier = metric_token.split("|", 1)
        if modifier.strip().lower() != "abs":
            raise argparse.ArgumentTypeError("Unsupported promotion metric modifier; only '|abs' is allowed.")
        absolute = True
        metric_token = name_part
    metric_name = metric_token.strip().lower()
    if not metric_name:
        raise argparse.ArgumentTypeError("Metric name cannot be empty in promotion metric check.")
    try:
        threshold_value = float(threshold.strip())
    except ValueError as exc:  # pragma: no cover - argparse surfaces error
        raise argparse.ArgumentTypeError(f"Invalid threshold '{threshold}' for promotion metric check.") from exc
    return PromotionMetricCheck(
        metric=metric_name,
        comparator=comparator,
        threshold=threshold_value,
        absolute=absolute,
    )


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

    drift = metrics.get("stability_calibration_drift")
    if drift is not None and "stability_calibration_drift_abs" not in metrics:
        try:
            metrics["stability_calibration_drift_abs"] = abs(float(drift))
        except (TypeError, ValueError):
            pass

    return {name: float(value) for name, value in metrics.items()}


def _env_flag(name: str, default: bool) -> bool:
    """Return truthy value derived from environment variables."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_optional_seed(name: str) -> int | None:
    """Return non-negative seed parsed from environment or ``None`` when unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _env_optional_path(name: str) -> Path | None:
    """Return path parsed from environment or ``None`` when unset or blank."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return Path(value).expanduser()


def _env_optional_positive_float(name: str) -> float | None:
    """Return positive float parsed from environment or ``None`` when unset or invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed > 0.0 else None


def _env_optional_positive_int(name: str) -> int | None:
    """Return positive integer parsed from environment or ``None`` when unset or invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _env_truthy(name: str, default: bool = False) -> bool:
    """Return boolean flag parsed from environment."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_str_tuple(name: str) -> tuple[str, ...]:
    """Return tuple of strings derived from comma-separated environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return ()
    parts = [segment.strip() for segment in raw.split(",")]
    return tuple(part for part in parts if part)


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """Return a tuple of strings derived from ``value`` when sequence-like."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            result.append(str(item))
        return tuple(result)
    return ()


def _extract_phase_one_signals(metadata: Mapping[str, Any]) -> PhaseOneFeatureSignals:
    """Extract Phase 1 feature families from dataset metadata annotations."""
    candidate_mappings: list[Mapping[str, Any]] = []
    for key in ("phase_one_signals", "phase_one_features"):
        raw = metadata.get(key)
        if isinstance(raw, Mapping):
            candidate_mappings.append(raw)

    column_info = metadata.get("column_info")
    if isinstance(column_info, Mapping):
        for key in ("phase_one_signals", "phase_one_features"):
            raw = column_info.get(key)
            if isinstance(raw, Mapping):
                candidate_mappings.append(raw)

    def _resolve(key: str, *aliases: str) -> tuple[str, ...]:
        keys = (key,) + aliases
        for mapping in candidate_mappings:
            for lookup in keys:
                if lookup in mapping:
                    return _as_str_tuple(mapping.get(lookup))
        for lookup in keys:
            if lookup in metadata:
                return _as_str_tuple(metadata.get(lookup))
        return ()

    return PhaseOneFeatureSignals(
        macro_delta_columns=_resolve("macro_delta_columns", "macro_deltas"),
        calendar_lag_columns=_resolve("calendar_lag_columns", "calendar_lag_windows"),
        clustering_tag_columns=_resolve("clustering_tag_columns", "clustering_tags"),
        context_feature_columns=_resolve("context_feature_columns", "context_signals", "context_features"),
    )


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
    phase_one_signals = _extract_phase_one_signals(metadata)
    return FeatureLayout(
        feature_names=feature_names,
        numeric_columns=numeric,
        categorical_columns=categorical,
        feature_schema=schema,
        phase_one_signals=phase_one_signals,
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
    include_macro_deltas: bool,
    include_calendar_lags: bool,
    include_clustering_tags: bool,
    include_context_features: bool,
    dataset_seed: int | None,
    layout: FeatureLayout | None = None,
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

    if layout is None:
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
        seed=dataset_seed if dataset_seed is not None else 7,
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
        include_macro_deltas=include_macro_deltas,
        include_calendar_lags=include_calendar_lags,
        include_clustering_tags=include_clustering_tags,
        include_context_features=include_context_features,
        macro_lag_days=macro_lag_days,
        earnings_lag_days=earnings_lag_days,
        events_notice_minutes=events_notice_minutes,
        phase_one_signals=layout.phase_one_signals,
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
    include_macro_deltas: bool,
    include_calendar_lags: bool,
    include_clustering_tags: bool,
    include_context_features: bool,
    dataset_seed: int | None,
) -> DatasetSpecification:
    metadata_path = dataset_dir / "dataset_metadata.json"
    report_path = dataset_dir / "report.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"dataset metadata missing at {metadata_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"dataset report missing at {report_path}")

    metadata = _load_json(metadata_path, description="dataset metadata")
    report = _load_json(report_path, description="dataset report")
    layout = _build_feature_layout(metadata)
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
        include_macro_deltas=include_macro_deltas,
        include_calendar_lags=include_calendar_lags,
        include_clustering_tags=include_clustering_tags,
        include_context_features=include_context_features,
        dataset_seed=dataset_seed,
        layout=layout,
    )
    dataset_id = str(metadata.get("dataset_id", dataset_dir.name))
    return DatasetSpecification(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        metadata=metadata,
        report=report,
        streaming_config=streaming_config,
        feature_layout=layout,
        phase_one_signals=layout.phase_one_signals,
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
        phase_one_signals=spec.phase_one_signals,
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
        "loss_name": worker_config.loss_name,
    }
    if spec.streaming_config.seed is not None:
        training_config["dataset_seed"] = int(spec.streaming_config.seed)
    if worker_config.worker_seed is not None:
        training_config["worker_seed"] = int(worker_config.worker_seed)
    if worker_config.loss_pos_weight is not None:
        training_config["loss_pos_weight"] = float(worker_config.loss_pos_weight)

    telemetry_caps = cast(Mapping[str, Any], telemetry.get("caps", {}))
    caps_payload: dict[str, Any] = dict(plan.caps)
    caps_payload.update(telemetry_caps)
    telemetry_block: dict[str, Any] = {
        "caps": caps_payload,
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
    }
    validation_returns_payload = telemetry.get("validation_returns")
    if isinstance(validation_returns_payload, Mapping):
        telemetry_block["validation_returns"] = {
            "fallback_join": bool(validation_returns_payload.get("fallback_join")),
            "mismatch_count": int(validation_returns_payload.get("mismatch_count", 0)),
            "missing_count": int(validation_returns_payload.get("missing_count", 0)),
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
        "telemetry": telemetry_block,
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
            "loss_name": worker_config.loss_name,
            **(
                {"loss_pos_weight": float(worker_config.loss_pos_weight)}
                if worker_config.loss_pos_weight is not None
                else {}
            ),
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
        self._active_worker: LightningStreamingWorker | None = None
        self._scheduled_event_watcher: AzureScheduledEventsWatcher | None = None
        if config.scheduled_events.enabled:
            self._scheduled_event_watcher = AzureScheduledEventsWatcher(
                config.scheduled_events,
                callback=self._handle_scheduled_event_notice,
            )

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers to stop the runner gracefully."""

        def _handler(signum: int, _frame: Any) -> None:
            self._handle_signal(signum)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _handle_signal(self, signum: int) -> None:
        """Handle termination signals by requesting a graceful shutdown and checkpoint."""
        signal_enum = getattr(signal, "Signals", None)
        if signal_enum is not None:
            try:  # pragma: no cover - platform dependent path
                signal_name = signal_enum(signum).name
            except Exception:  # pragma: no cover - fallback for unexpected signals
                signal_name = str(signum)
        else:  # pragma: no cover - Signals enum unavailable on some platforms
            signal_name = str(signum)
        logger.info("received signal %s, stopping runner", signal_name)
        self._stop_requested = True
        self._request_checkpoint(reason=f"signal:{signal_name}")

    def _request_checkpoint(self, *, reason: str) -> None:
        """Flush a checkpoint on the active worker when available."""
        worker = self._active_worker
        if worker is None:
            return
        try:
            worker.save_checkpoint_now(reason=reason, triggered_by_signal=True)
        except Exception:
            logger.error(
                "checkpoint_signal_save_failed",
                extra={"reason": reason},
                exc_info=True,
            )

    def _handle_scheduled_event_notice(self, notice: ScheduledEventNotice) -> None:
        """Respond to Azure scheduled-event eviction notices by checkpointing."""
        reason_parts = [f"azure:{notice.event_type.strip().lower()}"]
        if notice.event_id:
            reason_parts.append(str(notice.event_id))
        reason = ":".join(reason_parts)
        logger.info(
            "azure_eviction_notice_received",
            extra={
                "event_id": notice.event_id,
                "event_type": notice.event_type,
                "event_status": notice.event_status,
                "resources": list(notice.resources),
                "not_before": notice.not_before,
            },
        )
        self._stop_requested = True
        self._request_checkpoint(reason=reason)

    def _start_scheduled_event_monitoring(self) -> None:
        """Start the Azure scheduled-event watcher when configured."""
        watcher = self._scheduled_event_watcher
        if watcher is not None:
            watcher.start()

    def _stop_scheduled_event_monitoring(self) -> None:
        """Stop the Azure scheduled-event watcher."""
        watcher = self._scheduled_event_watcher
        if watcher is not None:
            watcher.stop()

    def _should_defer_due_to_backlog(
        self,
        orchestrator: InMemoryStreamingOrchestrator,
    ) -> bool:
        """Return True when adaptive backlog guard should delay the next plan."""
        threshold = self._config.orchestrator.adaptive_backlog_threshold
        if threshold is None:
            return False
        active_plans = len(orchestrator.inflight_plan_ids())
        if active_plans < threshold:
            return False
        logger.info(
            "adaptive_backlog_guard_engaged",
            extra={
                "active_plans": active_plans,
                "threshold": threshold,
                "cooldown_seconds": float(self._config.orchestrator.adaptive_cooldown_seconds),
            },
        )
        return True

    def _next_plan_interval(self, result: TrainingResultEvent) -> float:
        """Return the interval before scheduling the next plan."""
        base_interval = float(self._config.plan_interval_seconds)
        if base_interval <= 0.0:
            return 0.0
        threshold = self._config.orchestrator.adaptive_gpu_threshold_mb
        peak_gpu = result.telemetry.max_gpu_memory_mb
        if threshold is None or peak_gpu is None:
            return base_interval
        if peak_gpu < threshold:
            return base_interval
        multiplier = float(self._config.orchestrator.adaptive_interval_multiplier)
        interval = base_interval * multiplier
        cooldown = float(self._config.orchestrator.adaptive_cooldown_seconds)
        adjusted = max(interval, cooldown)
        logger.info(
            "adaptive_gpu_guard_engaged",
            extra={
                "peak_gpu_mb": peak_gpu,
                "threshold": threshold,
                "base_interval": base_interval,
                "adjusted_interval": adjusted,
                "multiplier": multiplier,
            },
        )
        return adjusted

    def _sleep_with_stop_check(self, seconds: float) -> None:
        """Sleep while honouring stop requests."""
        if seconds <= 0.0:
            return
        deadline = time.monotonic() + seconds
        while not self._stop_requested:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(remaining, 1.0))

    def run(self) -> None:
        """Execute streaming cohorts according to configuration."""
        config = self._config
        planner = StreamingDatasetPlanner(config.planner)
        worker = RecordingStreamingWorker(config.worker, output_dir=config.output_dir)
        self._active_worker = worker
        self._start_scheduled_event_monitoring()
        try:
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
                if self._should_defer_due_to_backlog(orchestrator):
                    self._sleep_with_stop_check(float(config.orchestrator.adaptive_cooldown_seconds))
                    continue
                plan_request = _make_dataset_plan_request(config.dataset)
                try:
                    plan_event = orchestrator.enqueue_training(plan_request)
                except AdaptiveSchedulingDeferred as exc:
                    logger.info(
                        "adaptive_deferral_skipped_plan",
                        extra={
                            "dataset_id": config.dataset.dataset_id,
                            "reason": str(exc),
                        },
                    )
                    self._sleep_with_stop_check(float(config.orchestrator.adaptive_cooldown_seconds))
                    continue
                result_event = worker.last_result
                if result_event is None:
                    RUN_COUNTER.labels(status=EventStatus.FAILED.value).inc()
                    raise RuntimeError("worker did not produce a training result event")

                self._handle_result(plan_event, result_event)
                plans_run += 1

                interval = self._next_plan_interval(result_event)
                if interval <= 0.0:
                    break
                logger.info("sleeping before next plan", extra={"seconds": interval})
                self._sleep_with_stop_check(interval)
        finally:
            self._stop_scheduled_event_monitoring()
            self._active_worker = None

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
                    observed_value = _observed_metric_value(check, metrics)
                    logger.info(
                        "promotion_secondary_metric_not_met",
                        extra={
                            "metric": check.metric,
                            "comparator": check.comparator,
                            "threshold": check.threshold,
                            "observed": observed_value,
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
                                "observed": _observed_metric_value(check, metrics),
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
            "Specify as metric>=value for minima or metric<=value for maxima; append '|abs' to apply absolute value."
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
    dataset_seed_default = _env_optional_seed("ML_STREAMING_DATASET_SEED")
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=dataset_seed_default,
        help="Seed applied to streaming dataset iteration (default derives from ML_STREAMING_DATASET_SEED).",
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

    include_macro_deltas_default = _env_flag("ML_STREAMING_INCLUDE_MACRO_DELTAS", False)
    parser.add_argument(
        "--include-macro-deltas",
        dest="include_macro_deltas",
        action="store_true",
        default=include_macro_deltas_default,
        help="Enable macro delta feature families (default derives from ML_STREAMING_INCLUDE_MACRO_DELTAS).",
    )
    parser.add_argument(
        "--no-include-macro-deltas",
        dest="include_macro_deltas",
        action="store_false",
        help="Disable macro delta feature families.",
    )

    include_calendar_lags_default = _env_flag("ML_STREAMING_INCLUDE_CALENDAR_LAGS", False)
    parser.add_argument(
        "--include-calendar-lags",
        dest="include_calendar_lags",
        action="store_true",
        default=include_calendar_lags_default,
        help="Enable calendar lag window features (default derives from ML_STREAMING_INCLUDE_CALENDAR_LAGS).",
    )
    parser.add_argument(
        "--no-include-calendar-lags",
        dest="include_calendar_lags",
        action="store_false",
        help="Disable calendar lag window features.",
    )

    include_clustering_tags_default = _env_flag("ML_STREAMING_INCLUDE_CLUSTERING_TAGS", False)
    parser.add_argument(
        "--include-clustering-tags",
        dest="include_clustering_tags",
        action="store_true",
        default=include_clustering_tags_default,
        help="Enable clustering tag enrichment (default derives from ML_STREAMING_INCLUDE_CLUSTERING_TAGS).",
    )
    parser.add_argument(
        "--no-include-clustering-tags",
        dest="include_clustering_tags",
        action="store_false",
        help="Disable clustering tag enrichment.",
    )

    include_context_features_default = _env_flag("ML_STREAMING_INCLUDE_CONTEXT_FEATURES", False)
    parser.add_argument(
        "--include-context-features",
        dest="include_context_features",
        action="store_true",
        default=include_context_features_default,
        help="Enable additional context feature families (default derives from ML_STREAMING_INCLUDE_CONTEXT_FEATURES).",
    )
    parser.add_argument(
        "--no-include-context-features",
        dest="include_context_features",
        action="store_false",
        help="Disable additional context feature families.",
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
    worker_seed_default = _env_optional_seed("ML_STREAMING_WORKER_SEED")
    parser.add_argument(
        "--worker-seed",
        type=int,
        default=worker_seed_default,
        help="Seed applied to trainer/dataloader randomness (default derives from ML_STREAMING_WORKER_SEED).",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=30.0,
        help="Interval (seconds) for GPU memory sampling (<=0 disables).",
    )
    checkpoint_dir_default = _env_optional_path("ML_STREAMING_CHECKPOINT_DIR")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=checkpoint_dir_default,
        help="Directory where streaming checkpoints are written (default: ML_STREAMING_CHECKPOINT_DIR).",
    )
    checkpoint_interval_seconds_default = _env_optional_positive_float(
        "ML_STREAMING_CHECKPOINT_INTERVAL_SECONDS",
    )
    parser.add_argument(
        "--checkpoint-interval-seconds",
        type=float,
        default=checkpoint_interval_seconds_default,
        help="Seconds between checkpoint saves (>0 enables time-based cadence).",
    )
    checkpoint_interval_steps_default = _env_optional_positive_int(
        "ML_STREAMING_CHECKPOINT_INTERVAL_STEPS",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=checkpoint_interval_steps_default,
        help="Training steps between checkpoint saves (>0 enables step-based cadence).",
    )
    checkpoint_retention_default = _env_int("ML_STREAMING_CHECKPOINT_RETENTION", 2)
    if checkpoint_retention_default <= 0:
        checkpoint_retention_default = 2
    parser.add_argument(
        "--checkpoint-retention",
        type=int,
        default=checkpoint_retention_default,
        help="Number of checkpoint files to retain (>=1).",
    )
    azure_events_enabled_default = _env_truthy("ML_STREAMING_AZURE_EVENTS_ENABLED", False)
    parser.add_argument(
        "--azure-events-enabled",
        action=argparse.BooleanOptionalAction,
        default=azure_events_enabled_default,
        help="Enable Azure scheduled-event polling for eviction notices "
        "(default sourced from ML_STREAMING_AZURE_EVENTS_ENABLED).",
    )
    azure_poll_default = _env_optional_positive_float("ML_STREAMING_AZURE_EVENTS_POLL_SECONDS")
    parser.add_argument(
        "--azure-events-poll-interval",
        type=float,
        default=azure_poll_default if azure_poll_default is not None else 5.0,
        help="Seconds between Azure scheduled-event polls (>0).",
    )
    azure_timeout_default = _env_optional_positive_float("ML_STREAMING_AZURE_EVENTS_TIMEOUT_SECONDS")
    parser.add_argument(
        "--azure-events-timeout-seconds",
        type=float,
        default=azure_timeout_default if azure_timeout_default is not None else 2.0,
        help="Request timeout (seconds) when querying Azure scheduled events (>0).",
    )
    azure_endpoint_default = os.environ.get(
        "ML_STREAMING_AZURE_EVENTS_ENDPOINT",
        "http://169.254.169.254/metadata/scheduledevents",
    )
    parser.add_argument(
        "--azure-events-endpoint",
        type=str,
        default=azure_endpoint_default,
        help="Azure metadata endpoint for scheduled events.",
    )
    azure_api_version_default = os.environ.get("ML_STREAMING_AZURE_EVENTS_API_VERSION", "2020-07-01")
    parser.add_argument(
        "--azure-events-api-version",
        type=str,
        default=azure_api_version_default,
        help="Azure scheduled-events API version (default 2020-07-01).",
    )
    azure_resource_default = list(_env_str_tuple("ML_STREAMING_AZURE_EVENTS_RESOURCE"))
    parser.add_argument(
        "--azure-events-resource",
        action="append",
        default=azure_resource_default,
        help="Resource filter applied to scheduled events (repeatable).",
    )
    azure_event_types_default = list(_env_str_tuple("ML_STREAMING_AZURE_EVENTS_TYPES"))
    if not azure_event_types_default:
        azure_event_types_default = ["Preempt"]
    parser.add_argument(
        "--azure-events-event-type",
        action="append",
        default=azure_event_types_default,
        help="Event types triggering checkpoint saves (repeatable).",
    )
    azure_event_status_default = list(_env_str_tuple("ML_STREAMING_AZURE_EVENTS_STATUS"))
    if not azure_event_status_default:
        azure_event_status_default = ["Scheduled", "InProgress"]
    parser.add_argument(
        "--azure-events-status",
        action="append",
        default=azure_event_status_default,
        help="Event statuses triggering checkpoint saves (repeatable).",
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
    parser.add_argument(
        "--loss",
        type=str,
        choices=("bce", "poisson"),
        default="bce",
        help="Loss function for the TFT teacher (default: bce).",
    )
    parser.add_argument(
        "--loss-pos-weight",
        type=float,
        default=None,
        help="Positive class weight applied when --loss=bce (>0 disables when <=0).",
    )
    parser.add_argument(
        "--validation-return-column",
        type=str,
        default="forward_return",
        help="Column containing forward returns used for economic metrics (empty string disables).",
    )
    parser.add_argument(
        "--enable-temperature-calibration",
        action="store_true",
        help="Enable temperature scaling calibration during evaluation.",
    )
    parser.add_argument(
        "--disable-temperature-calibration",
        action="store_false",
        dest="enable_temperature_calibration",
        help="Disable temperature scaling calibration during evaluation.",
    )
    parser.set_defaults(enable_temperature_calibration=True)
    parser.add_argument(
        "--temperature-calibration-min",
        type=float,
        default=0.25,
        help="Minimum temperature considered when calibrating (must be >0).",
    )
    parser.add_argument(
        "--temperature-calibration-max",
        type=float,
        default=5.0,
        help="Maximum temperature considered when calibrating (must exceed min).",
    )
    parser.add_argument(
        "--temperature-calibration-steps",
        type=int,
        default=25,
        help="Number of evaluation steps across the temperature range (>=1).",
    )
    parser.add_argument(
        "--enable-platt-calibration",
        action="store_true",
        help="Enable Platt scaling using logistic regression for calibrated probabilities.",
    )
    parser.add_argument(
        "--enable-isotonic-calibration",
        action="store_true",
        help="Enable isotonic regression calibration for probabilities.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Lightning precision argument (e.g., 32, 16, 16-mixed, bf16).",
    )
    parser.add_argument(
        "--enable-amp",
        action="store_true",
        help="Enable automatic mixed precision using --amp-precision.",
    )
    parser.add_argument(
        "--amp-precision",
        type=str,
        default="16-mixed",
        help="Precision string used when --enable-amp is supplied (default: 16-mixed).",
    )
    parser.add_argument(
        "--amp-guard-threshold-mb",
        type=float,
        default=None,
        help="Disable AMP when recent peak GPU usage exceeds this threshold (MiB).",
    )
    parser.add_argument(
        "--enable-curriculum",
        action="store_true",
        help="Enable curriculum-aware train fraction scheduling.",
    )
    parser.add_argument(
        "--curriculum-default-train-fraction",
        type=float,
        default=None,
        help="Fallback train fraction when curriculum stages do not match (defaults to --train-fraction).",
    )
    parser.add_argument(
        "--curriculum-stage",
        action="append",
        metavar="MAX_ROWS:TRAIN_FRACTION",
        help="Curriculum stage definition (repeatable). Use '*' for unlimited rows.",
    )
    parser.add_argument(
        "--curriculum-guard",
        action="append",
        metavar="LABEL:key=value,...",
        help="Guard definition tied to a curriculum stage label (repeatable).",
    )
    parser.add_argument(
        "--enable-ensemble",
        action="store_true",
        help="Blend freshly trained logits with external artefacts before computing metrics.",
    )
    parser.add_argument(
        "--ensemble-member",
        action="append",
        metavar="PATH[:WEIGHT[:required|optional]]",
        help="External logits artefact to blend (repeatable). Paths may reference {plan_id}/{dataset_id}.",
    )
    parser.add_argument(
        "--ensemble-blend-mode",
        choices=("weighted", "mean"),
        default="weighted",
        help="Blending strategy for ensemble logits (weighted or mean).",
    )
    parser.add_argument(
        "--no-ensemble-normalize-weights",
        dest="ensemble_normalize_weights",
        action="store_false",
        help="Disable weight normalisation when blending ensemble members.",
    )
    parser.set_defaults(ensemble_normalize_weights=True)


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
    adaptive_backlog_default = _env_int("ML_STREAMING_ADAPTIVE_BACKLOG_THRESHOLD", 0)
    parser.add_argument(
        "--adaptive-backlog-threshold",
        type=int,
        default=adaptive_backlog_default,
        help="Outstanding plan count that triggers adaptive cooldown (<=0 disables).",
    )
    adaptive_gpu_default = _env_float("ML_STREAMING_ADAPTIVE_GPU_THRESHOLD_MB", 0.0)
    parser.add_argument(
        "--adaptive-gpu-threshold-mb",
        type=float,
        default=adaptive_gpu_default,
        help="Peak GPU memory (MB) that triggers adaptive interval scaling (<=0 disables).",
    )
    adaptive_cooldown_default = _env_float("ML_STREAMING_ADAPTIVE_COOLDOWN_SECONDS", 120.0)
    parser.add_argument(
        "--adaptive-cooldown-seconds",
        type=float,
        default=adaptive_cooldown_default,
        help="Cooldown applied when adaptive backlog guard engages.",
    )
    adaptive_multiplier_default = _env_float("ML_STREAMING_ADAPTIVE_INTERVAL_MULTIPLIER", 2.0)
    parser.add_argument(
        "--adaptive-interval-multiplier",
        type=float,
        default=adaptive_multiplier_default,
        help="Multiplier applied to plan interval when GPU threshold is exceeded (>=1.0).",
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

    dataset_seed = args.dataset_seed if args.dataset_seed is None or args.dataset_seed >= 0 else None
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
        include_macro_deltas=args.include_macro_deltas,
        include_calendar_lags=args.include_calendar_lags,
        include_clustering_tags=args.include_clustering_tags,
        include_context_features=args.include_context_features,
        dataset_seed=dataset_seed,
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
        include_macro_deltas=bool(args.include_macro_deltas),
        include_calendar_lags=bool(args.include_calendar_lags),
        include_clustering_tags=bool(args.include_clustering_tags),
        include_context_features=bool(args.include_context_features),
    )

    curriculum_stages: tuple[CurriculumStageConfig, ...] = ()
    if args.curriculum_stage:
        curriculum_stages = tuple(
            parse_curriculum_stage_spec(spec)
            for spec in args.curriculum_stage
        )
    curriculum_guards: tuple[CurriculumGuardRule, ...] = ()
    if args.curriculum_guard:
        curriculum_guards = tuple(
            parse_curriculum_guard_spec(spec)
            for spec in args.curriculum_guard
        )
    curriculum_default_fraction = (
        float(args.curriculum_default_train_fraction)
        if args.curriculum_default_train_fraction is not None
        else float(args.train_fraction)
    )
    curriculum_config = CurriculumScheduleConfig(
        enabled=bool(args.enable_curriculum),
        stages=curriculum_stages,
        default_train_fraction=curriculum_default_fraction,
        guards=curriculum_guards,
    )

    ensemble_members: tuple[EnsembleMemberConfig, ...] = ()
    if args.ensemble_member:
        ensemble_members = tuple(parse_ensemble_member_spec(spec) for spec in args.ensemble_member)
    ensemble_config = StreamingEnsembleConfig(
        enabled=bool(args.enable_ensemble),
        blend_mode=str(args.ensemble_blend_mode),
        normalize_weights=bool(args.ensemble_normalize_weights),
        members=ensemble_members,
    )

    validation_return_column = str(args.validation_return_column or "").strip()
    if not validation_return_column:
        validation_return_column_param: str | None = None
    else:
        validation_return_column_param = validation_return_column

    checkpoint_dir_arg = args.checkpoint_dir
    checkpoint_dir_value: str | None
    if checkpoint_dir_arg is None:
        checkpoint_dir_value = None
    else:
        checkpoint_dir_value = str(Path(checkpoint_dir_arg).expanduser())
    checkpoint_seconds_value: float | None
    if args.checkpoint_interval_seconds is None or float(args.checkpoint_interval_seconds) <= 0.0:
        checkpoint_seconds_value = None
    else:
        checkpoint_seconds_value = float(args.checkpoint_interval_seconds)
    checkpoint_steps_value: int | None
    if args.checkpoint_interval_steps is None or int(args.checkpoint_interval_steps) <= 0:
        checkpoint_steps_value = None
    else:
        checkpoint_steps_value = int(args.checkpoint_interval_steps)
    checkpoint_retention_value = max(1, int(args.checkpoint_retention))

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
        loss_name=str(args.loss).strip().lower(),
        loss_pos_weight=(
            None
            if args.loss_pos_weight is None or float(args.loss_pos_weight) <= 0.0
            else float(args.loss_pos_weight)
        ),
        gpu_memory_monitor_interval_seconds=(
            None if float(args.gpu_monitor_interval) <= 0.0 else float(args.gpu_monitor_interval)
        ),
        enable_temperature_calibration=bool(args.enable_temperature_calibration),
        temperature_calibration_min=float(args.temperature_calibration_min),
        temperature_calibration_max=float(args.temperature_calibration_max),
        temperature_calibration_steps=max(1, int(args.temperature_calibration_steps)),
        enable_platt_calibration=bool(args.enable_platt_calibration),
        enable_isotonic_calibration=bool(args.enable_isotonic_calibration),
        precision=str(args.precision),
        enable_amp=bool(args.enable_amp),
        amp_precision=str(args.amp_precision),
        amp_guard_threshold_mb=(
            None
            if args.amp_guard_threshold_mb is None
            else max(float(args.amp_guard_threshold_mb), 0.0) or None
        ),
        checkpoint_dir=checkpoint_dir_value,
        checkpoint_interval_seconds=checkpoint_seconds_value,
        checkpoint_interval_steps=checkpoint_steps_value,
        checkpoint_retention=checkpoint_retention_value,
        curriculum=curriculum_config,
        ensemble=ensemble_config,
        validation_return_column=validation_return_column_param,
        dataset_seed=dataset_spec.streaming_config.seed,
        worker_seed=args.worker_seed if args.worker_seed is None or args.worker_seed >= 0 else None,
    )

    adaptive_backlog_threshold = _coerce_limit(getattr(args, "adaptive_backlog_threshold", None))
    adaptive_gpu_threshold = _coerce_positive_float(getattr(args, "adaptive_gpu_threshold_mb", None))
    adaptive_cooldown = max(0.0, float(getattr(args, "adaptive_cooldown_seconds", 120.0)))
    adaptive_interval_multiplier = max(1.0, float(getattr(args, "adaptive_interval_multiplier", 2.0)))
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
        adaptive_backlog_threshold=adaptive_backlog_threshold,
        adaptive_gpu_threshold_mb=adaptive_gpu_threshold,
        adaptive_cooldown_seconds=max(1.0, adaptive_cooldown),
        adaptive_interval_multiplier=adaptive_interval_multiplier,
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

    promotion_config = StreamingPromotionConfig.from_env()
    if args.promotion_threshold is not None:
        promotion_threshold_value: float | None = float(args.promotion_threshold)
    else:
        promotion_threshold_value = (
            float(promotion_config.min_roc_auc)
            if promotion_config.min_roc_auc is not None
            else None
        )

    promotion_checks_from_args: tuple[PromotionMetricCheck, ...] = ()
    if args.promotion_metric_check:
        promotion_checks_from_args = tuple(args.promotion_metric_check)
    combined_checks: list[PromotionMetricCheck] = list(promotion_checks_from_args)
    seen_signatures: set[tuple[str, str, float, bool]] = {
        (check.metric, check.comparator, check.threshold, check.absolute)
        for check in combined_checks
    }
    for check in _promotion_checks_from_config(promotion_config):
        signature = (check.metric, check.comparator, check.threshold, check.absolute)
        if signature not in seen_signatures:
            combined_checks.append(check)
            seen_signatures.add(signature)
    promotion_checks = tuple(combined_checks)
    azure_endpoint_value = str(args.azure_events_endpoint or "").strip()
    if not azure_endpoint_value:
        azure_endpoint_value = "http://169.254.169.254/metadata/scheduledevents"
    azure_api_version_value = str(args.azure_events_api_version or "").strip() or "2020-07-01"
    azure_poll_interval = float(getattr(args, "azure_events_poll_interval", 5.0))
    if azure_poll_interval <= 0.0:
        azure_poll_interval = 5.0
    azure_timeout_seconds = float(getattr(args, "azure_events_timeout_seconds", 2.0))
    if azure_timeout_seconds <= 0.0:
        azure_timeout_seconds = 2.0
    azure_resource_values = tuple(
        str(item).strip()
        for item in (getattr(args, "azure_events_resource", None) or [])
        if str(item).strip()
    )
    azure_event_types_values = tuple(
        str(item).strip()
        for item in (getattr(args, "azure_events_event_type", None) or [])
        if str(item).strip()
    )
    if not azure_event_types_values:
        azure_event_types_values = ("Preempt",)
    azure_status_values = tuple(
        str(item).strip()
        for item in (getattr(args, "azure_events_status", None) or [])
        if str(item).strip()
    )
    if not azure_status_values:
        azure_status_values = ("Scheduled", "InProgress")
    azure_events_config = AzureScheduledEventsConfig(
        enabled=bool(getattr(args, "azure_events_enabled", False)),
        poll_interval_seconds=azure_poll_interval,
        request_timeout_seconds=azure_timeout_seconds,
        metadata_endpoint=azure_endpoint_value,
        api_version=azure_api_version_value,
        resource_filter=azure_resource_values,
        event_types=azure_event_types_values,
        status_filter=azure_status_values,
    )

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
        promotion_threshold=promotion_threshold_value,
        promotion_command=promotion_command,
        promotion_checks=promotion_checks,
        pipeline_signature=str(args.pipeline_signature),
        pipeline_version=str(args.pipeline_version),
        persist_snapshot=bool(args.persist_snapshot),
        scheduled_events=azure_events_config,
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
