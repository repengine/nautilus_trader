from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ml.tests.fixtures.pandera import DataFrame, Series, ensure_pandera_available

pa = ensure_pandera_available()

from ml.config.events import EventStatus
from ml.config.events import Stage
from ml.training.event_driven.payloads import build_heartbeat_message
from ml.training.event_driven.payloads import build_plan_message
from ml.training.event_driven.payloads import build_result_message
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import TrainingHeartbeatEvent
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.teacher.streaming_loader import PhaseOneFeatureSignals
from ml.training.teacher.streaming_loader import StreamingLimitSummary
from ml.training.teacher.streaming_loader import TFTShardIndex
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingMetadata
from ml.training.teacher.streaming_loader import TFTStreamingSummary
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry
from ml.training.teacher.streaming_telemetry import ValidationReturnsTelemetry


def _plan_event(tmp_path: Path) -> DatasetPlanEvent:
    phase_signals = PhaseOneFeatureSignals(
        macro_delta_columns=("PAYEMS_delta_1d",),
        calendar_lag_columns=("hours_to_fed_meeting", "hours_to_economic_release"),
        clustering_tag_columns=("event_clustering_score",),
        context_feature_columns=("is_fomc_week", "is_holiday_week"),
    )
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
        phase_one_signals=phase_signals,
    )
    summary = TFTStreamingSummary(total_shards=1, total_rows=5, max_shard_rows=5)
    limits = StreamingLimitSummary(
        skipped_shards=0,
        skipped_rows=0,
        skipped_sequences=0,
        total_instrument_rows={"AAPL": 5},
        selected_instrument_rows={"AAPL": 5},
        skipped_instrument_rows={},
        total_instrument_sequences={"AAPL": 3},
        selected_instrument_sequences={"AAPL": 3},
        skipped_instrument_sequences={},
    )
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
        max_total_rows=100,
        max_total_sequences=200,
        max_shards=3,
        include_macro=True,
        include_calendar=True,
        include_events=True,
        include_earnings=True,
        include_micro=True,
        include_l2=True,
        include_macro_revisions=True,
        include_macro_deltas=True,
        include_calendar_lags=True,
        include_clustering_tags=True,
        include_context_features=True,
        macro_lag_days=2,
        earnings_lag_days=1,
        events_notice_minutes=90,
        phase_one_signals=phase_signals,
    )
    return DatasetPlanEvent(
        plan_id="plan-contract",
        dataset_id="dataset",
        parquet_path=tmp_path / "dataset.parquet",
        metadata=metadata,
        metadata_summary=summary,
        limits=limits,
        streaming_config=streaming_cfg,
        caps={"max_total_rows": 100},
        phase_one_signals=phase_signals,
        status=EventStatus.SUCCESS,
    )


def _result_event(plan: DatasetPlanEvent) -> TrainingResultEvent:
    limits = StreamingLimitSummary()
    telemetry = StreamingRunTelemetry(
        metadata_summary=plan.metadata_summary,
        caps=plan.caps,
        train=StreamingLoaderTelemetry.from_metadata(
            "train",
            plan.metadata,
            limits,
            plan.streaming_config,
        ),
        validation=StreamingLoaderTelemetry.from_metadata(
            "validation",
            plan.metadata,
            limits,
            plan.streaming_config,
        ),
        validation_returns=ValidationReturnsTelemetry(
            fallback_join=False,
            mismatch_count=0,
            missing_count=0,
        ),
    )
    return TrainingResultEvent(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        model_id="tft-model",
        telemetry=telemetry,
        artifact_paths={"logits": "/tmp/logits.npz"},
        metrics={"roc_auc": 0.5},
        status=EventStatus.SUCCESS,
    )


def _heartbeat_event(plan: DatasetPlanEvent) -> TrainingHeartbeatEvent:
    return TrainingHeartbeatEvent(
        worker_id="worker-1",
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        progress_pct=42.0,
        rss_mb=256.0,
        shards_processed=3,
    )


def test_streaming_plan_message_contract(tmp_path: Path) -> None:
    plan = _plan_event(tmp_path)
    plan = replace(plan, checkpoint_key="checkpoint-1")
    message = build_plan_message(plan).as_dict()

    assert message["schema_version"] == "1.0.0"
    assert message["stage"] == Stage.DATASET_PLANNED.value
    assert message["source"]
    assert message["status"] == plan.status.value
    assert message["plan_id"] == plan.plan_id
    assert message["dataset_id"] == plan.dataset_id
    assert message["payload_type"] == "streaming_plan"
    payload = message["payload"]
    assert payload["created_at"].endswith("Z")
    assert payload["parquet_path"].endswith("dataset.parquet")
    assert payload["caps"]["max_total_rows"] == 100
    assert payload["checkpoint_key"] == "checkpoint-1"
    limits_payload = payload["limits"]
    assert limits_payload["skipped_shards"] == 0
    assert limits_payload["instrument_rows_total"] == {"AAPL": 5}
    assert limits_payload["instrument_rows_selected"] == {"AAPL": 5}
    assert limits_payload["instrument_sequences_total"] == {"AAPL": 3}
    assert payload["metadata_summary"]["total_rows"] == 5
    config = payload["streaming_config"]
    capability_flags = payload["capability_flags"]
    assert capability_flags == {
        "include_macro": config["include_macro"],
        "include_calendar": config["include_calendar"],
        "include_events": config["include_events"],
        "include_earnings": config["include_earnings"],
        "include_micro": config["include_micro"],
        "include_l2": config["include_l2"],
        "include_macro_revisions": config["include_macro_revisions"],
        "include_macro_deltas": config["include_macro_deltas"],
        "include_calendar_lags": config["include_calendar_lags"],
        "include_clustering_tags": config["include_clustering_tags"],
        "include_context_features": config["include_context_features"],
    }
    assert capability_flags["include_macro"] is True
    assert capability_flags["include_l2"] is True
    assert capability_flags["include_micro"] is True
    assert capability_flags["include_macro_deltas"] is True
    assert capability_flags["include_calendar_lags"] is True
    assert capability_flags["include_clustering_tags"] is True
    assert capability_flags["include_context_features"] is True
    assert config["macro_lag_days"] == 2
    assert config["earnings_lag_days"] == 1
    assert config["events_notice_minutes"] == 90
    publication_lags = payload["publication_lags"]
    assert publication_lags == {
        "macro_lag_days": 2,
        "earnings_lag_days": 1,
        "events_notice_minutes": 90,
    }
    phase_one_signals = payload["phase_one_signals"]
    assert phase_one_signals == {
        "macro_delta_columns": ["PAYEMS_delta_1d"],
        "calendar_lag_columns": ["hours_to_fed_meeting", "hours_to_economic_release"],
        "clustering_tag_columns": ["event_clustering_score"],
        "context_feature_columns": ["is_fomc_week", "is_holiday_week"],
    }
    assert config["phase_one_signals"] == phase_one_signals
    serialized = json.dumps(message)
    assert "streaming_plan" in serialized


def test_streaming_plan_message_omits_checkpoint_key_when_absent(tmp_path: Path) -> None:
    plan = _plan_event(tmp_path)
    message = build_plan_message(plan).as_dict()

    payload = message["payload"]
    assert "checkpoint_key" not in payload


def test_streaming_config_negative_lag_rejected() -> None:
    with pytest.raises(ValueError, match="macro_lag_days"):
        TFTStreamingConfig(
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
            macro_lag_days=-1,
        )


def test_streaming_result_message_contract(tmp_path: Path) -> None:
    plan = _plan_event(tmp_path)
    result = _result_event(plan)
    message = build_result_message(result).as_dict()

    assert message["schema_version"] == "1.0.0"
    assert message["stage"] == Stage.MODEL_TRAINING_COMPLETED.value
    assert message["status"] == result.status.value
    assert message["plan_id"] == plan.plan_id
    assert message["payload_type"] == "streaming_result"
    payload = message["payload"]
    assert payload["completed_at"].endswith("Z")
    assert payload["model_id"] == "tft-model"
    assert payload["metrics"]["roc_auc"] == pytest.approx(0.5)
    assert payload["artifact_paths"]["logits"].endswith(".npz")
    telemetry = payload["telemetry"]
    assert telemetry["caps"]["max_total_rows"] == 100
    assert telemetry["train"]["selected_shards"] == 1
    assert telemetry["validation_returns"]["fallback_join"] is False
    serialized = json.dumps(message)
    assert "streaming_result" in serialized


def test_streaming_heartbeat_message_contract(tmp_path: Path) -> None:
    plan = _plan_event(tmp_path)
    heartbeat = _heartbeat_event(plan)
    message = build_heartbeat_message(heartbeat, dataset_id=plan.dataset_id).as_dict()

    assert message["schema_version"] == "1.0.0"
    assert message["stage"] == Stage.WORKER_HEARTBEAT.value
    expected_status = (
        EventStatus.SUCCESS.value if heartbeat.progress_pct >= 100.0 else EventStatus.PARTIAL.value
    )
    assert message["status"] == expected_status
    assert message["plan_id"] == plan.plan_id
    assert message["dataset_id"] == plan.dataset_id
    assert message["payload_type"] == "streaming_heartbeat"
    payload = message["payload"]
    assert payload["worker_id"] == heartbeat.worker_id
    assert payload["progress_pct"] == pytest.approx(42.0)
    assert payload["rss_mb"] == pytest.approx(256.0)
    assert payload["timestamp"].endswith("Z")
    serialized = json.dumps(message)
    assert "streaming_heartbeat" in serialized


class StreamingConfigCapabilitySchema(pa.DataFrameModel):
    include_macro: Series[bool] = pa.Field(nullable=False)
    include_calendar: Series[bool] = pa.Field(nullable=False)
    include_events: Series[bool] = pa.Field(nullable=False)
    include_earnings: Series[bool] = pa.Field(nullable=False)
    include_micro: Series[bool] = pa.Field(nullable=False)
    include_l2: Series[bool] = pa.Field(nullable=False)
    include_macro_revisions: Series[bool] = pa.Field(nullable=False)


def test_calendar_event_payload_schema(tmp_path: Path) -> None:
    """
    Ensure streaming capability flags are present for calendar/event pipelines.
    """
    plan = _plan_event(tmp_path)
    message = build_plan_message(plan).as_dict()
    caps = message["payload"]["capability_flags"]

    df: DataFrame = pd.DataFrame(
        [
            {
                "include_macro": caps["include_macro"],
                "include_calendar": caps["include_calendar"],
                "include_events": caps["include_events"],
                "include_earnings": caps["include_earnings"],
                "include_micro": caps["include_micro"],
                "include_l2": caps["include_l2"],
                "include_macro_revisions": caps["include_macro_revisions"],
            },
        ],
    )

    StreamingConfigCapabilitySchema.validate(df)
