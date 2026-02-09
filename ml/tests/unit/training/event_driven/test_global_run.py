from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from ml.config.events import EventStatus
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import StreamingGlobalRunConfig
import ml.training.event_driven.global_run as global_run_module
from ml.training.event_driven.global_run import StreamingGlobalRunPlanner
from ml.training.event_driven.global_run import StreamingGlobalRunState
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


def _build_state(
    *,
    run_id: str = "global-run",
    dataset_id: str = "dataset",
    train_shard_ids: tuple[str, ...] = ("A-1", "B-1"),
    val_shard_ids: tuple[str, ...] = ("A-2", "B-2"),
    next_train_index: int = 0,
    pending_plan_id: str | None = None,
    pending_start: int | None = None,
    pending_end: int | None = None,
    completed_plans: int = 0,
    shards_per_plan: int = 1,
    train_fraction: float = 0.5,
    shuffle_train_shards: bool = False,
    seed: int | None = 7,
) -> StreamingGlobalRunState:
    return StreamingGlobalRunState(
        run_id=run_id,
        dataset_id=dataset_id,
        created_at="2026-02-08T00:00:00Z",
        updated_at="2026-02-08T00:00:00Z",
        train_shard_ids=train_shard_ids,
        val_shard_ids=val_shard_ids,
        next_train_index=next_train_index,
        pending_plan_id=pending_plan_id,
        pending_start=pending_start,
        pending_end=pending_end,
        completed_plans=completed_plans,
        shards_per_plan=shards_per_plan,
        train_fraction=train_fraction,
        shuffle_train_shards=shuffle_train_shards,
        seed=seed,
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


def test_global_run_state_from_dict_handles_invalid_values() -> None:
    state = StreamingGlobalRunState.from_dict(
        {
            "run_id": "run",
            "dataset_id": "dataset",
            "created_at": "created",
            "updated_at": "updated",
            "train_shard_ids": "not-a-list",
            "val_shard_ids": ["V-1"],
            "next_train_index": "bad",
            "pending_plan_id": "",
            "pending_start": "bad",
            "pending_end": "bad",
            "completed_plans": object(),
            "shards_per_plan": 0,
            "train_fraction": "bad",
            "shuffle_train_shards": True,
            "seed": "bad",
        },
    )

    assert state.train_shard_ids == ()
    assert state.val_shard_ids == ("V-1",)
    assert state.next_train_index == 0
    assert state.pending_plan_id is None
    assert state.pending_start == 0
    assert state.pending_end == 0
    assert state.shards_per_plan == 1
    assert state.train_fraction == pytest.approx(0.0)
    assert state.seed == 0
    assert state.total_plans() == 0


def test_global_run_state_store_handles_invalid_payload_and_pending_plan(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    store = StreamingGlobalRunStateStore(state_path)

    assert store.path == state_path
    assert store.load() is None

    state_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid global run state payload"):
        store.load()

    pending_state = _build_state(
        pending_plan_id="global-run-1",
        pending_start=0,
        pending_end=1,
    )
    store.save(pending_state)
    loaded_pending = store.load()
    assert loaded_pending is not None
    pending_plan = store.reserve_next_plan(loaded_pending)
    assert pending_plan.plan_id == "global-run-1"
    assert pending_plan.shard_ids == ("A-1",)

    with pytest.raises(RuntimeError, match="no remaining shards"):
        store.reserve_next_plan(_build_state(next_train_index=2))

    state_no_pending = _build_state(pending_plan_id=None, next_train_index=1)
    assert store.mark_plan_completed(state_no_pending, plan_id="unused") is state_no_pending

    with pytest.raises(ValueError, match="Pending plan mismatch"):
        store.mark_plan_completed(
            _build_state(
                pending_plan_id="global-run-2",
                pending_start=1,
                pending_end=2,
            ),
            plan_id="other-plan",
        )


def test_resolve_shards_per_plan_handles_fallbacks() -> None:
    assert (
        global_run_module._resolve_shards_per_plan(
            requested=None,
            worker_cap=2,
            config_cap=3,
            fallback=4,
        )
        == 2
    )
    assert (
        global_run_module._resolve_shards_per_plan(
            requested=0,
            worker_cap=0,
            config_cap=-1,
            fallback=0,
        )
        == 1
    )


def test_global_run_planner_has_remaining_plans_when_state_missing_or_complete(tmp_path: Path) -> None:
    planner = StreamingGlobalRunPlanner(
        DatasetServiceConfig(parquet_root=str(tmp_path)),
        global_config=StreamingGlobalRunConfig(enabled=True, state_path=str(tmp_path / "state.json")),
        state_store=StreamingGlobalRunStateStore(tmp_path / "state.json"),
        train_fraction=0.5,
        worker_max_shards=None,
    )

    assert planner.has_remaining_plans() is True
    planner._state_store.save(_build_state(next_train_index=2))
    assert planner.has_remaining_plans() is False


def test_global_run_mark_plan_completed_returns_false_for_failure_or_missing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _build_metadata()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(stream, "collect_streaming_metadata", lambda *args, **kwargs: metadata)

    global_config = StreamingGlobalRunConfig(
        enabled=True,
        run_id="global-run",
        state_path=str(tmp_path / "state.json"),
        shards_per_plan=1,
        shuffle_train_shards=False,
    )
    planner = StreamingGlobalRunPlanner(
        DatasetServiceConfig(parquet_root=str(tmp_path), shard_row_budget=25),
        global_config=global_config,
        state_store=StreamingGlobalRunStateStore(Path(global_config.state_path or "")),
        train_fraction=0.5,
        worker_max_shards=None,
    )
    request = _build_request(dataset_path)
    plan = planner.plan(request)

    failed_result = replace(_build_result_event(plan.plan_id, plan.dataset_id), status=EventStatus.FAILED)
    assert planner.mark_plan_completed(plan, failed_result) is False

    state_path = Path(global_config.state_path or "")
    state_path.unlink(missing_ok=True)
    assert planner.mark_plan_completed(plan, _build_result_event(plan.plan_id, plan.dataset_id)) is False


def test_global_run_planner_respects_plan_shard_cap_and_optional_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _build_metadata()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(stream, "collect_streaming_metadata", lambda *args, **kwargs: metadata)

    global_config = StreamingGlobalRunConfig(
        enabled=True,
        run_id="global-run",
        state_path=str(tmp_path / "state.json"),
        shards_per_plan=1,
        shuffle_train_shards=None,
    )
    planner = StreamingGlobalRunPlanner(
        DatasetServiceConfig(parquet_root=str(tmp_path), shard_row_budget=25),
        global_config=global_config,
        state_store=StreamingGlobalRunStateStore(Path(global_config.state_path or "")),
        train_fraction=0.5,
        worker_max_shards=None,
    )
    request = _build_request(dataset_path)
    request_with_cap = replace(
        request,
        streaming_config=replace(request.streaming_config, max_shards=5, seed=None, shuffle_shards=True),
    )

    plan = planner.plan(request_with_cap)

    assert plan.streaming_config.max_shards == 1
    assert "dataset_seed" not in plan.caps
    assert plan.caps["max_shards"] == 1


def test_global_run_planner_raises_when_state_run_or_dataset_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _build_metadata()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(stream, "collect_streaming_metadata", lambda *args, **kwargs: metadata)

    state_store = StreamingGlobalRunStateStore(tmp_path / "state.json")
    state_store.save(_build_state(run_id="other-run", dataset_id="dataset"))
    planner_run_mismatch = StreamingGlobalRunPlanner(
        DatasetServiceConfig(parquet_root=str(tmp_path), shard_row_budget=25),
        global_config=StreamingGlobalRunConfig(
            enabled=True,
            run_id="global-run",
            state_path=str(tmp_path / "state.json"),
        ),
        state_store=state_store,
        train_fraction=0.5,
        worker_max_shards=None,
    )
    with pytest.raises(ValueError, match="Run id mismatch"):
        planner_run_mismatch.plan(_build_request(dataset_path))

    state_store.save(_build_state(run_id="global-run", dataset_id="other-dataset"))
    planner_dataset_mismatch = StreamingGlobalRunPlanner(
        DatasetServiceConfig(parquet_root=str(tmp_path), shard_row_budget=25),
        global_config=StreamingGlobalRunConfig(
            enabled=True,
            run_id="global-run",
            state_path=str(tmp_path / "state.json"),
        ),
        state_store=state_store,
        train_fraction=0.5,
        worker_max_shards=None,
    )
    with pytest.raises(ValueError, match="Dataset mismatch"):
        planner_dataset_mismatch.plan(_build_request(dataset_path))


def test_global_run_planner_rehydrates_metadata_from_existing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _build_metadata()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(stream, "collect_streaming_metadata", lambda *args, **kwargs: metadata)

    state_store = StreamingGlobalRunStateStore(tmp_path / "state.json")
    state_store.save(
        _build_state(
            train_shard_ids=("A-1",),
            val_shard_ids=("A-2", "B-2"),
            shards_per_plan=1,
            next_train_index=0,
            pending_plan_id=None,
            pending_start=None,
            pending_end=None,
            completed_plans=0,
        ),
    )
    planner = StreamingGlobalRunPlanner(
        DatasetServiceConfig(parquet_root=str(tmp_path), shard_row_budget=25),
        global_config=StreamingGlobalRunConfig(
            enabled=True,
            run_id="global-run",
            state_path=str(tmp_path / "state.json"),
        ),
        state_store=state_store,
        train_fraction=0.5,
        worker_max_shards=None,
    )

    plan = planner.plan(_build_request(dataset_path))
    assert tuple(shard.shard_id for shard in plan.train_metadata.shard_indices) == ("A-1",)
    assert {shard.shard_id for shard in plan.val_metadata.shard_indices} == {"A-2", "B-2"}


def test_global_run_planner_raises_for_missing_default_parquet_path(tmp_path: Path) -> None:
    state_store = StreamingGlobalRunStateStore(tmp_path / "state.json")
    planner = StreamingGlobalRunPlanner(
        DatasetServiceConfig(parquet_root=str(tmp_path), shard_row_budget=25),
        global_config=StreamingGlobalRunConfig(
            enabled=True,
            run_id="global-run",
            state_path=str(tmp_path / "state.json"),
        ),
        state_store=state_store,
        train_fraction=0.5,
        worker_max_shards=None,
    )

    request = replace(_build_request(tmp_path), parquet_path=None, dataset_id="missing-dataset")
    with pytest.raises(FileNotFoundError, match="Parquet dataset not found"):
        planner.plan(request)


def test_global_run_planner_raises_when_metadata_not_initialized(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    planner = StreamingGlobalRunPlanner(
        DatasetServiceConfig(parquet_root=str(tmp_path), shard_row_budget=25),
        global_config=StreamingGlobalRunConfig(
            enabled=True,
            run_id="global-run",
            state_path=str(tmp_path / "state.json"),
        ),
        state_store=StreamingGlobalRunStateStore(tmp_path / "state.json"),
        train_fraction=0.5,
        worker_max_shards=None,
    )

    def _fake_ensure_state(
        _request: DatasetPlanRequest,
        *,
        parquet_path: Path,
        numeric_columns: tuple[str, ...],
        planner_config: TFTStreamingConfig,
    ) -> StreamingGlobalRunState:
        del parquet_path, numeric_columns, planner_config
        return _build_state()

    planner._ensure_state = _fake_ensure_state  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="Global run metadata not initialized"):
        planner.plan(_build_request(dataset_path))


def test_global_run_planner_uses_preseeded_train_metadata_when_available(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    state_store = StreamingGlobalRunStateStore(tmp_path / "state.json")
    planner = StreamingGlobalRunPlanner(
        DatasetServiceConfig(parquet_root=str(tmp_path), shard_row_budget=25),
        global_config=StreamingGlobalRunConfig(
            enabled=True,
            run_id="global-run",
            state_path=str(tmp_path / "state.json"),
            shards_per_plan=1,
        ),
        state_store=state_store,
        train_fraction=0.5,
        worker_max_shards=None,
    )

    planner._full_metadata = metadata
    planner._train_metadata = metadata
    planner._val_metadata = metadata
    planner._ordered_train_ids = tuple(shard.shard_id for shard in metadata.shard_indices)

    plan = planner.plan(_build_request(dataset_path))
    snapshot = json.loads(state_store.path.read_text(encoding="utf-8"))
    assert snapshot["pending_plan_id"] == plan.plan_id
