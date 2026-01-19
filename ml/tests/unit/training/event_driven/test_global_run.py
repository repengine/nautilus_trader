from __future__ import annotations

from pathlib import Path

import pytest

from ml.config.events import EventStatus
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import StreamingGlobalRunConfig
from ml.training.event_driven.global_run import StreamingGlobalRunPlanner
from ml.training.event_driven.global_run import StreamingGlobalRunStateStore
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.teacher import streaming_loader as stream
from ml.training.teacher.streaming_loader import PhaseOneFeatureSignals
from ml.training.teacher.streaming_loader import TFTShardIndex
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingMetadata
from ml.training.teacher.streaming_loader import TFTStreamingSummary
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


def _build_metadata() -> TFTStreamingMetadata:
    shards = (
        TFTShardIndex(
            shard_id="A-1",
            instrument_id="AAPL",
            row_start=0,
            row_end=9,
            row_count=10,
            time_start=0,
            time_end=9,
        ),
        TFTShardIndex(
            shard_id="A-2",
            instrument_id="AAPL",
            row_start=10,
            row_end=19,
            row_count=10,
            time_start=10,
            time_end=19,
        ),
        TFTShardIndex(
            shard_id="B-1",
            instrument_id="MSFT",
            row_start=0,
            row_end=9,
            row_count=10,
            time_start=0,
            time_end=9,
        ),
        TFTShardIndex(
            shard_id="B-2",
            instrument_id="MSFT",
            row_start=10,
            row_end=19,
            row_count=10,
            time_start=10,
            time_end=19,
        ),
    )
    instrument_counts = {"AAPL": 20, "MSFT": 20}
    return TFTStreamingMetadata(
        shard_indices=shards,
        numeric_stats={},
        categorical_vocab={},
        instrument_row_counts=instrument_counts,
        instrument_target_stats={},
        phase_one_signals=PhaseOneFeatureSignals(),
    )


def _build_request(parquet_path: Path) -> DatasetPlanRequest:
    streaming_config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=(),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        shuffle_shards=False,
        seed=7,
    )
    return DatasetPlanRequest(
        dataset_id="dataset",
        streaming_config=streaming_config,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        parquet_path=parquet_path,
    )


def _build_result_event(plan_id: str, dataset_id: str) -> TrainingResultEvent:
    summary = TFTStreamingSummary(total_shards=0, total_rows=0, max_shard_rows=0)
    train_loader = StreamingLoaderTelemetry(
        loader="train",
        total_shards=0,
        selected_shards=0,
        skipped_shards=0,
        total_rows=0,
        selected_rows=0,
        skipped_rows=0,
        total_sequences=0,
        selected_sequences=0,
        skipped_sequences=0,
    )
    val_loader = StreamingLoaderTelemetry(
        loader="validation",
        total_shards=0,
        selected_shards=0,
        skipped_shards=0,
        total_rows=0,
        selected_rows=0,
        skipped_rows=0,
        total_sequences=0,
        selected_sequences=0,
        skipped_sequences=0,
    )
    telemetry = StreamingRunTelemetry(
        metadata_summary=summary,
        caps={},
        train=train_loader,
        validation=val_loader,
    )
    return TrainingResultEvent(
        plan_id=plan_id,
        dataset_id=dataset_id,
        model_id="model-id",
        telemetry=telemetry,
        artifact_paths={"logits": "logits.npz"},
        metrics={"roc_auc": 0.5},
        status=EventStatus.SUCCESS,
    )


def test_global_run_planner_slices_training_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _build_metadata()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        stream,
        "collect_streaming_metadata",
        lambda *args, **kwargs: metadata,
    )

    planner_config = DatasetServiceConfig(parquet_root=str(tmp_path), shard_row_budget=25)
    global_config = StreamingGlobalRunConfig(
        enabled=True,
        run_id="global-run",
        state_path=str(tmp_path / "global_state.json"),
        shards_per_plan=1,
        shuffle_train_shards=False,
    )
    state_path = (
        Path(global_config.state_path)
        if global_config.state_path is not None
        else tmp_path / "global_state.json"
    )
    planner = StreamingGlobalRunPlanner(
        planner_config,
        global_config=global_config,
        state_store=StreamingGlobalRunStateStore(state_path),
        train_fraction=0.5,
        worker_max_shards=4,
    )
    request = _build_request(dataset_path)

    plan_one = planner.plan(request)
    assert plan_one.checkpoint_key == "global-run"
    assert tuple(shard.shard_id for shard in plan_one.train_metadata.shard_indices) == ("A-1",)
    assert {shard.shard_id for shard in plan_one.val_metadata.shard_indices} == {"A-2", "B-2"}

    result_one = _build_result_event(plan_one.plan_id, plan_one.dataset_id)
    assert planner.mark_plan_completed(plan_one, result_one) is False

    plan_two = planner.plan(request)
    assert tuple(shard.shard_id for shard in plan_two.train_metadata.shard_indices) == ("B-1",)
    result_two = _build_result_event(plan_two.plan_id, plan_two.dataset_id)
    assert planner.mark_plan_completed(plan_two, result_two) is True
