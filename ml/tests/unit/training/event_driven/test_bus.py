from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ml.common.message_bus import MessagePublisherProtocol
from ml.config.events import EventStatus
import ml.training.event_driven.payloads as payloads_module
from ml.training.event_driven.orchestrator import PublisherOrchestratorBus
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


class _RecorderPublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


class _FailingPublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.attempts: int = 0

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:  # noqa: ARG002 - interface compatibility
        self.attempts += 1
        return False


def _plan_event() -> DatasetPlanEvent:
    metadata = TFTStreamingMetadata(
        shard_indices=(
            TFTShardIndex(
                shard_id="s0",
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
    limits = StreamingLimitSummary(skipped_shards=1, skipped_rows=3, skipped_sequences=2)
    streaming_cfg = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=("feature",),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=16,
        drop_last=False,
        shuffle_shards=True,
        seed=7,
        num_workers=0,
        max_total_rows=100,
        max_total_sequences=200,
        max_shards=3,
    )
    return DatasetPlanEvent(
        plan_id="plan-xyz",
        dataset_id="dataset",
        parquet_path=Path("/data/dataset"),
        metadata=metadata,
        metadata_summary=summary,
        limits=limits,
        streaming_config=streaming_cfg,
        caps={"max_total_rows": 100},
        status=EventStatus.SUCCESS,
    )


def _result_event() -> TrainingResultEvent:
    summary = TFTStreamingSummary(total_shards=1, total_rows=5, max_shard_rows=5)
    limits = StreamingLimitSummary()
    streaming_cfg = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=("feature",),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=16,
        drop_last=False,
        shuffle_shards=False,
        seed=7,
        num_workers=0,
    )
    metadata = TFTStreamingMetadata(
        shard_indices=(),
        numeric_stats={},
        categorical_vocab={},
        instrument_row_counts={},
    )
    telemetry = StreamingRunTelemetry(
        metadata_summary=summary,
        caps={"max_total_rows": 100},
        train=StreamingLoaderTelemetry.from_metadata("train", metadata, limits, streaming_cfg),
        validation=StreamingLoaderTelemetry.from_metadata(
            "validation",
            metadata,
            limits,
            streaming_cfg,
        ),
    )
    return TrainingResultEvent(
        plan_id="plan-xyz",
        dataset_id="dataset",
        model_id="teacher",
        telemetry=telemetry,
        artifact_paths={"logits": "/tmp/logits.npz"},
        metrics={"roc_auc": 0.55},
        status=EventStatus.SUCCESS,
    )


def test_publisher_bus_serializes_plan_event() -> None:
    recorder = _RecorderPublisher()
    bus = PublisherOrchestratorBus(recorder)
    event = _plan_event()

    bus.publish_plan("events.ml.DATASET_PLANNED.dataset", event)

    assert recorder.calls
    topic, payload = recorder.calls[0]
    assert topic == "events.ml.DATASET_PLANNED.dataset"
    assert payload["plan_id"] == "plan-xyz"
    assert payload["status"] == EventStatus.SUCCESS.value
    assert payload["payload_type"] == "streaming_plan"
    plan_body = payload["payload"]
    assert plan_body["parquet_path"] == "/data/dataset"
    assert plan_body["metadata_summary"]["total_shards"] == 1
    assert plan_body["streaming_config"]["batch_size"] == 16
    assert plan_body["created_at"].endswith("Z")


def test_publisher_bus_serializes_result_event() -> None:
    recorder = _RecorderPublisher()
    bus = PublisherOrchestratorBus(recorder)
    event = _result_event()

    bus.publish_result("events.ml.MODEL_TRAINING_COMPLETED.dataset", event)

    assert recorder.calls
    _, payload = recorder.calls[0]
    assert payload["plan_id"] == "plan-xyz"
    assert payload["payload_type"] == "streaming_result"
    result_body = payload["payload"]
    assert result_body["metrics"]["roc_auc"] == 0.55
    assert "telemetry" in result_body
    assert result_body["telemetry"]["caps"]["max_total_rows"] == 100
    assert result_body["completed_at"].endswith("Z")


def test_publisher_bus_serializes_heartbeat_event() -> None:
    recorder = _RecorderPublisher()
    bus = PublisherOrchestratorBus(recorder)
    heartbeat = TrainingHeartbeatEvent(
        worker_id="worker-1",
        plan_id="plan-xyz",
        dataset_id="dataset",
        progress_pct=12.5,
        rss_mb=256.0,
        shards_processed=3,
    )

    bus.publish_heartbeat("events.ml.WORKER_HEARTBEAT.dataset", heartbeat)

    assert recorder.calls
    _, payload = recorder.calls[0]
    assert payload["payload"]["worker_id"] == "worker-1"
    assert payload["plan_id"] == "plan-xyz"
    assert payload["dataset_id"] == "dataset"
    assert payload["payload_type"] == "streaming_heartbeat"
    heartbeat_body = payload["payload"]
    assert heartbeat_body["progress_pct"] == 12.5
    assert heartbeat_body["timestamp"].endswith("Z")


def test_publisher_bus_records_failed_payloads() -> None:
    failing = _FailingPublisher()
    bus = PublisherOrchestratorBus(failing, max_attempts=2)
    event = _plan_event()

    bus.publish_plan("events.ml.DATASET_PLANNED.dataset", event)

    assert failing.attempts == 2
    failed = bus.failed_events
    assert len(failed) == 1
    topic, payload = failed[0]
    assert topic == "events.ml.DATASET_PLANNED.dataset"
    assert payload["plan_id"] == "plan-xyz"


def test_payload_helpers_coerce_sequence_mappings() -> None:
    class _TelemetryStub:
        def as_dict(self) -> dict[str, Any]:
            return {
                "caps": [("max_total_rows", 100), ("bad-entry",)],
                "metadata": [("dataset", "dataset")],
                "train": [("selected_rows", 5)],
                "validation": [("selected_rows", 2)],
                "resources": [("gpu_mb", 256.0)],
                "ensemble": {"members": ("m1", "m2"), "weight": 0.7},
                "economic": {"sharpe": 1.2},
                "stability": {"ks_statistic": 0.1},
                "validation_returns": {"mean": 0.01},
            }

    payload = payloads_module._telemetry_dict(_TelemetryStub())
    assert payload["caps"] == {"max_total_rows": 100}
    assert payload["metadata"] == {"dataset": "dataset"}
    assert payload["train"] == {"selected_rows": 5}
    assert payload["validation"] == {"selected_rows": 2}
    assert payload["resources"] == {"gpu_mb": 256.0}
    assert payload["ensemble"]["members"] == ["m1", "m2"]
    assert payload["economic"] == {"sharpe": 1.2}
    assert payload["stability"] == {"ks_statistic": 0.1}
    assert payload["validation_returns"] == {"mean": 0.01}


def test_build_plan_message_includes_checkpoint_and_coerces_lag_values() -> None:
    event = replace(_plan_event(), checkpoint_key="checkpoint-1")
    object.__setattr__(event.streaming_config, "macro_lag_days", "bad")
    object.__setattr__(event.streaming_config, "earnings_lag_days", None)
    object.__setattr__(event.streaming_config, "events_notice_minutes", "3")

    payload = payloads_module.build_plan_message(event).as_dict()

    assert payload["payload"]["checkpoint_key"] == "checkpoint-1"
    assert payload["payload"]["publication_lags"]["macro_lag_days"] == 0
    assert payload["payload"]["publication_lags"]["earnings_lag_days"] == 0
    assert payload["payload"]["publication_lags"]["events_notice_minutes"] == 3


def test_build_heartbeat_message_marks_completion_success() -> None:
    heartbeat = TrainingHeartbeatEvent(
        worker_id="worker-2",
        plan_id=None,
        dataset_id=None,
        progress_pct=100.0,
        rss_mb=64.0,
        shards_processed=8,
    )

    payload = payloads_module.build_heartbeat_message(heartbeat, dataset_id="dataset").as_dict()

    assert payload["status"] == EventStatus.SUCCESS.value
    assert payload["plan_id"] == "dataset"
