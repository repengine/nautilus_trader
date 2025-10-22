"""Helpers for constructing streaming training test payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.config.events import EventStatus
from ml.training.event_driven.payloads import build_heartbeat_message
from ml.training.event_driven.payloads import build_plan_message
from ml.training.event_driven.payloads import build_result_message
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import TrainingHeartbeatEvent
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.teacher.streaming_loader import StreamingLimitSummary
from ml.training.teacher.streaming_loader import TFTShardIndex
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingMetadata
from ml.training.teacher.streaming_loader import TFTStreamingSummary
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry


@dataclass(slots=True, frozen=True)
class StreamingTestPayloads:
    """Reusable bundle of dataset plan, result, and heartbeat events for tests.

    Args:
        plan_event: Dataset plan emitted by the planner.
        result_event: Completed training result emitted by the worker.
        heartbeat_event: Worker heartbeat emitted during training.
    """

    plan_event: DatasetPlanEvent
    result_event: TrainingResultEvent
    heartbeat_event: TrainingHeartbeatEvent

    def plan_message(self) -> dict[str, Any]:
        """Return the serialized streaming plan message."""
        return build_plan_message(self.plan_event).as_dict()

    def result_message(self) -> dict[str, Any]:
        """Return the serialized streaming result message."""
        return build_result_message(self.result_event).as_dict()

    def heartbeat_message(self) -> dict[str, Any]:
        """Return the serialized streaming heartbeat message."""
        dataset_id = self.heartbeat_event.dataset_id or self.plan_event.dataset_id
        return build_heartbeat_message(self.heartbeat_event, dataset_id=dataset_id).as_dict()


def build_streaming_test_payloads(
    *,
    dataset_id: str = "dataset",
    plan_id: str = "plan-stream",
    parquet_path: Path | None = None,
) -> StreamingTestPayloads:
    """Construct deterministic streaming plan/result/heartbeat payloads."""
    resolved_path = parquet_path or Path(f"/tmp/{dataset_id}.parquet")
    metadata = TFTStreamingMetadata(
        shard_indices=(
            TFTShardIndex(
                shard_id="shard-0",
                instrument_id="AAPL",
                row_start=0,
                row_end=4,
                row_count=5,
                time_start=1,
                time_end=5,
            ),
        ),
        numeric_stats={},
        categorical_vocab={},
        instrument_row_counts={"AAPL": 5},
    )
    summary = TFTStreamingSummary(total_shards=1, total_rows=5, max_shard_rows=5)
    limits = StreamingLimitSummary(skipped_shards=0, skipped_rows=0, skipped_sequences=0)
    streaming_cfg = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=("feature",),
        time_varying_known_reals=("bias",),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=16,
        drop_last=False,
        shuffle_shards=False,
        seed=7,
        num_workers=0,
        max_total_rows=100,
        max_total_sequences=200,
        max_shards=3,
    )
    plan_event = DatasetPlanEvent(
        plan_id=plan_id,
        dataset_id=dataset_id,
        parquet_path=resolved_path,
        metadata=metadata,
        metadata_summary=summary,
        limits=limits,
        streaming_config=streaming_cfg,
        caps={"max_total_rows": 100},
        status=EventStatus.SUCCESS,
    )
    telemetry = StreamingRunTelemetry(
        metadata_summary=summary,
        caps=plan_event.caps,
        train=StreamingLoaderTelemetry.from_metadata(
            "train",
            metadata,
            limits,
            streaming_cfg,
        ),
        validation=StreamingLoaderTelemetry.from_metadata(
            "validation",
            metadata,
            limits,
            streaming_cfg,
        ),
    )
    result_event = TrainingResultEvent(
        plan_id=plan_event.plan_id,
        dataset_id=plan_event.dataset_id,
        model_id="model",
        telemetry=telemetry,
        artifact_paths={"logits": f"/artifacts/{dataset_id}/tft_cli_logits.npz"},
        metrics={"roc_auc": 0.5},
        status=plan_event.status,
    )
    heartbeat_event = TrainingHeartbeatEvent(
        worker_id="worker",
        plan_id=plan_event.plan_id,
        dataset_id=plan_event.dataset_id,
        progress_pct=75.0,
        rss_mb=128.0,
        shards_processed=2,
    )
    return StreamingTestPayloads(
        plan_event=plan_event,
        result_event=result_event,
        heartbeat_event=heartbeat_event,
    )


__all__ = ["StreamingTestPayloads", "build_streaming_test_payloads"]
