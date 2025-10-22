#!/usr/bin/env python3
r"""
Run a single streaming TFT cohort end-to-end using the event-driven helpers.

The CLI:
* loads dataset metadata (including vintage-age features),
* plans a constrained streaming cohort via ``StreamingDatasetPlanner``,
* executes ``LightningStreamingWorker`` to produce logits/metrics, and
* mirrors the emitted events into a persistence snapshot for dashboard use.

Example:
    poetry run python -m ml.scripts.run_streaming_cohort \\
        --dataset-dir ml_out/full_tft_95 \\
        --state-path ml_out/streaming_training_state_snapshot.json \\
        --max-total-rows 120000 \\
        --max-total-sequences 90000 \\
        --max-shards 32 \\
        --output-dir ml_out/tft_streaming_artifacts/full_tft_95 \\
        --accelerator cpu
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.consumers.streaming_training_service import StreamingTrainingPersistenceService
from ml.training.event_driven.dataset_service import StreamingDatasetPlanner
from ml.training.event_driven.payloads import build_plan_message
from ml.training.event_driven.payloads import build_result_message
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.event_driven.worker import LightningStreamingWorker
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry


@dataclass(slots=True, frozen=True)
class CohortInputs:
    """Resolved input parameters for a streaming cohort run."""

    dataset_dir: Path
    metadata: dict[str, Any]
    streaming_config: TFTStreamingConfig
    feature_names: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]


def _load_metadata(dataset_dir: Path) -> dict[str, Any]:
    metadata_path = dataset_dir / "dataset_metadata.json"
    if not metadata_path.exists():
        msg = f"metadata file missing at {metadata_path}"
        raise FileNotFoundError(msg)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "metadata JSON must be an object"
        raise ValueError(msg)
    return payload


def _build_cohort_inputs(
    dataset_dir: Path,
    *,
    max_total_rows: int,
    max_total_sequences: int,
    max_shards: int,
    batch_size: int,
    dataloader_workers: int,
) -> CohortInputs:
    metadata = _load_metadata(dataset_dir)
    columns = metadata.get("column_info", {})
    if not isinstance(columns, dict):
        msg = "metadata column_info must be an object"
        raise ValueError(msg)

    categorical_columns = tuple(str(value) for value in columns.get("categorical_columns", ()))
    static_reals = tuple(str(value) for value in columns.get("static_reals", ()))
    known_reals_raw = tuple(str(value) for value in columns.get("time_varying_known_reals", ()))
    known_reals = tuple(value for value in known_reals_raw if value != "time_index")
    unknown_reals = tuple(str(value) for value in columns.get("time_varying_unknown_reals", ()))
    vintage_age = tuple(str(value) for value in columns.get("vintage_age_columns", ()))

    feature_names = tuple(
        dict.fromkeys(static_reals + known_reals + unknown_reals + vintage_age + categorical_columns),
    )
    numeric_columns = tuple(
        dict.fromkeys(static_reals + known_reals + unknown_reals + vintage_age + ("y",)),
    )

    def _coerce_limit(value: int) -> int | None:
        if int(value) <= 0:
            return None
        return int(value)

    streaming_config = TFTStreamingConfig(
        time_idx_col=str(columns.get("time_idx_col", "time_index")),
        group_id_col=str(columns.get("group_id_col", "instrument_id")),
        target_col=str(columns.get("target_col", "y")),
        static_categoricals=categorical_columns,
        static_reals=static_reals,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        max_encoder_length=192,
        max_prediction_length=24,
        batch_size=batch_size,
        drop_last=False,
        shuffle_shards=False,
        seed=7,
        num_workers=dataloader_workers,
        max_total_rows=_coerce_limit(max_total_rows),
        max_total_sequences=_coerce_limit(max_total_sequences),
        max_shards=_coerce_limit(max_shards),
    )

    return CohortInputs(
        dataset_dir=dataset_dir,
        metadata=metadata,
        streaming_config=streaming_config,
        feature_names=feature_names,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )


def _plan_dataset(
    inputs: CohortInputs,
    *,
    shard_row_budget: int,
) -> DatasetPlanEvent:
    streaming_cfg = inputs.streaming_config
    planner_max_rows = streaming_cfg.max_total_rows
    if planner_max_rows is not None and planner_max_rows < shard_row_budget:
        planner_max_rows = shard_row_budget
    planner = StreamingDatasetPlanner(
        DatasetServiceConfig(
            parquet_root=str(inputs.dataset_dir),
            shard_row_budget=shard_row_budget,
            max_total_rows=planner_max_rows,
            max_total_sequences=streaming_cfg.max_total_sequences,
            max_shards=streaming_cfg.max_shards,
        ),
    )
    request = DatasetPlanRequest(
        dataset_id=str(inputs.metadata.get("dataset_id", inputs.dataset_dir.name)),
        streaming_config=streaming_cfg,
        feature_names=inputs.feature_names,
        categorical_columns=inputs.categorical_columns,
        numeric_columns=inputs.numeric_columns,
        parquet_path=inputs.dataset_dir / "dataset_with_vintage_age.parquet",
    )
    return planner.plan(request)


def _build_worker_config(
    args: argparse.Namespace,
    *,
    model_id: str,
) -> StreamingWorkerConfig:
    def _sanitize_limit(value: int | None) -> int | None:
        if value is None:
            return None
        if int(value) <= 0:
            return None
        return int(value)

    return StreamingWorkerConfig(
        max_total_rows=_sanitize_limit(getattr(args, "max_total_rows", None)),
        max_total_sequences=_sanitize_limit(getattr(args, "max_total_sequences", None)),
        max_shards=_sanitize_limit(getattr(args, "max_shards", None)),
        max_epochs=max(1, int(getattr(args, "max_epochs", 1))),
        max_runtime_seconds=args.max_runtime_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        max_retry_attempts=args.max_retry_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        accelerator=args.accelerator,
        devices=args.devices,
        train_fraction=args.train_fraction,
        logits_artifact_key=args.logits_key,
        validation_metric=args.validation_metric.lower(),
        gpu_memory_monitor_interval_seconds=args.gpu_monitor_interval,
        model_id=model_id,
    )


def _run_worker(
    plan: DatasetPlanEvent,
    *,
    worker_config: StreamingWorkerConfig,
    output_dir: Path,
) -> tuple[StreamingRunTelemetry, dict[str, float], dict[str, str]]:
    worker = LightningStreamingWorker(worker_config, output_dir=output_dir)
    result = worker.run(plan)
    metrics_pretty = json.dumps(result.metrics, sort_keys=True)
    print(f"training status: {result.status.value}")
    print(f"metrics: {metrics_pretty}")
    print(f"gpu_peak_mb: {result.telemetry.max_gpu_memory_mb}")
    print(f"logits_path: {result.artifact_paths.get(worker_config.logits_artifact_key, 'unknown')}")
    return result.telemetry, dict(result.metrics), dict(result.artifact_paths)


def _persist_events(
    plan: DatasetPlanEvent,
    *,
    worker_config: StreamingWorkerConfig,
    telemetry: StreamingRunTelemetry,
    metrics: dict[str, float],
    artifact_paths: dict[str, str],
    state_path: Path,
) -> None:
    service = StreamingTrainingPersistenceService.create(state_path=state_path)
    plan_message = build_plan_message(plan).as_dict()
    service.handle(f"events.ml.DATASET_PLANNED.{plan.dataset_id}", plan_message)
    result_payload = TrainingResultEvent(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        model_id=worker_config.model_id,
        telemetry=telemetry,
        artifact_paths=artifact_paths,
        metrics=metrics,
        status=plan.status,
    )
    result_message = build_result_message(result_payload).as_dict()
    service.handle(f"events.ml.MODEL_TRAINING_COMPLETED.{plan.dataset_id}", result_message)
    snapshot = service.snapshot()
    print(f"state snapshot written to {state_path}")
    print(f"results stored: {tuple(snapshot.get('results', {}).keys())}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a streaming TFT cohort using the event-driven pipeline helpers.",
    )
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
        help="Directory where worker artifacts (logits, telemetry) will be stored.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("ml_out/streaming_training_state_snapshot.json"),
        help="Path to persist streaming training state snapshot JSON.",
    )
    parser.add_argument(
        "--model-id",
        default="tft_streaming_cohort",
        help="Model identifier recorded in the result event.",
    )
    parser.add_argument(
        "--max-total-rows",
        type=int,
        default=120_000,
        help="Maximum rows per cohort plan.",
    )
    parser.add_argument(
        "--max-total-sequences",
        type=int,
        default=90_000,
        help="Maximum sequences per cohort plan.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=32,
        help="Maximum shards per cohort plan.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Maximum epochs for the TFT teacher fit (>=1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=48,
        help="Batch size used by the streaming dataloaders.",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=0,
        help="Number of dataloader workers for streaming loaders.",
    )
    parser.add_argument(
        "--shard-row-budget",
        type=int,
        default=200_000,
        help="Shard row budget enforced during planning.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Train fraction applied when splitting metadata.",
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        help="Lightning accelerator argument (e.g., 'cpu', 'gpu').",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices passed to Lightning.",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=30.0,
        help="Interval in seconds for GPU memory sampling (set <=0 to disable).",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=7_200,
        help="Maximum runtime per cohort before recording partial status.",
    )
    parser.add_argument(
        "--max-retry-attempts",
        type=int,
        default=1,
        help="Streaming worker retry attempts.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=0.0,
        help="Backoff seconds between retry attempts.",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=int,
        default=120,
        help="Heartbeat interval emitted by the worker.",
    )
    parser.add_argument(
        "--logits-key",
        default="logits",
        help="Artifact key used to store logits in the result event.",
    )
    parser.add_argument(
        "--validation-metric",
        default="roc_auc",
        help="Validation metric name recorded in the result event.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    state_path = args.state_path.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = _build_cohort_inputs(
        dataset_dir=dataset_dir,
        max_total_rows=args.max_total_rows,
        max_total_sequences=args.max_total_sequences,
        max_shards=args.max_shards,
        batch_size=args.batch_size,
        dataloader_workers=args.dataloader_workers,
    )
    plan = _plan_dataset(inputs, shard_row_budget=args.shard_row_budget)
    print(f"plan_id: {plan.plan_id}")
    print(f"selected_rows: {sum(shard.row_count for shard in plan.metadata.shard_indices)}")
    print(f"skipped_rows: {plan.limits.skipped_rows}")

    worker_cfg = _build_worker_config(args, model_id=args.model_id)
    telemetry, metrics, artifact_paths = _run_worker(
        plan,
        worker_config=worker_cfg,
        output_dir=output_dir,
    )

    _persist_events(
        plan,
        worker_config=worker_cfg,
        telemetry=telemetry,
        metrics=metrics,
        artifact_paths=artifact_paths,
        state_path=state_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
