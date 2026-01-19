"""Utilities for multi-plan streaming runs that converge on a single model."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from pathlib import Path

from ml.config.events import EventStatus
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import StreamingGlobalRunConfig
from ml.training.event_driven.guardrails import enforce_dataset_guardrails
from ml.training.event_driven.plan_helpers import apply_service_caps
from ml.training.event_driven.plan_helpers import ensure_target_in_numeric
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import DatasetPlanner
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.teacher import streaming_loader as stream
from ml.training.teacher.streaming_loader import StreamingLimitSummary
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingMetadata
from ml.training.teacher.streaming_loader import filter_metadata_by_shard_ids
from ml.training.teacher.streaming_loader import resolve_shard_order
from ml.training.teacher.streaming_loader import split_metadata_by_row_fraction
from ml.training.teacher.streaming_loader import summarize_metadata


logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat() + "Z"


@dataclass(frozen=True, slots=True)
class GlobalRunPlan:
    """
    Plan reservation describing the next shard slice to train.

    Attributes:
        plan_id: Identifier for the plan.
        plan_index: Zero-based index within the global run.
        shard_start: Start offset into the ordered training shard list.
        shard_end: End offset (exclusive) into the ordered training shard list.
        shard_ids: Shard identifiers assigned to the plan.
        total_plans: Total number of plans in the global run.
        is_final: True when this plan exhausts training shards.
    """

    plan_id: str
    plan_index: int
    shard_start: int
    shard_end: int
    shard_ids: tuple[str, ...]
    total_plans: int
    is_final: bool


@dataclass(slots=True)
class StreamingGlobalRunState:
    """Persisted state for a multi-plan streaming run."""

    run_id: str
    dataset_id: str
    created_at: str
    updated_at: str
    train_shard_ids: tuple[str, ...]
    val_shard_ids: tuple[str, ...]
    next_train_index: int
    pending_plan_id: str | None
    pending_start: int | None
    pending_end: int | None
    completed_plans: int
    shards_per_plan: int
    train_fraction: float
    shuffle_train_shards: bool
    seed: int | None

    def is_complete(self) -> bool:
        """Return True when all training shards have been scheduled."""
        return self.pending_plan_id is None and self.next_train_index >= len(self.train_shard_ids)

    def total_plans(self) -> int:
        """Return the total number of plans required for the run."""
        if not self.train_shard_ids:
            return 0
        return max(
            1,
            math.ceil(len(self.train_shard_ids) / float(self.shards_per_plan)),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize state for JSON persistence."""
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "train_shard_ids": list(self.train_shard_ids),
            "val_shard_ids": list(self.val_shard_ids),
            "next_train_index": int(self.next_train_index),
            "pending_plan_id": self.pending_plan_id,
            "pending_start": self.pending_start,
            "pending_end": self.pending_end,
            "completed_plans": int(self.completed_plans),
            "shards_per_plan": int(self.shards_per_plan),
            "train_fraction": float(self.train_fraction),
            "shuffle_train_shards": bool(self.shuffle_train_shards),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> StreamingGlobalRunState:
        """Rehydrate state from a JSON payload."""
        def _as_int(value: object, *, default: int) -> int:
            if value is None:
                return default
            if isinstance(value, (int, float, str)):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default
            return default

        def _as_float(value: object, *, default: float) -> float:
            if value is None:
                return default
            if isinstance(value, (int, float, str)):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default
            return default

        raw_train_ids = payload.get("train_shard_ids")
        train_shard_ids = (
            tuple(str(item) for item in raw_train_ids)
            if isinstance(raw_train_ids, list)
            else ()
        )
        raw_val_ids = payload.get("val_shard_ids")
        val_shard_ids = (
            tuple(str(item) for item in raw_val_ids)
            if isinstance(raw_val_ids, list)
            else ()
        )
        pending_plan_raw = payload.get("pending_plan_id")
        pending_plan_id = (
            str(pending_plan_raw)
            if pending_plan_raw not in (None, "")
            else None
        )
        pending_start_raw = payload.get("pending_start")
        pending_end_raw = payload.get("pending_end")
        seed_raw = payload.get("seed")
        return cls(
            run_id=str(payload.get("run_id", "")),
            dataset_id=str(payload.get("dataset_id", "")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            train_shard_ids=train_shard_ids,
            val_shard_ids=val_shard_ids,
            next_train_index=_as_int(payload.get("next_train_index"), default=0),
            pending_plan_id=pending_plan_id,
            pending_start=(
                _as_int(pending_start_raw, default=0)
                if pending_start_raw is not None
                else None
            ),
            pending_end=(
                _as_int(pending_end_raw, default=0)
                if pending_end_raw is not None
                else None
            ),
            completed_plans=_as_int(payload.get("completed_plans"), default=0),
            shards_per_plan=max(1, _as_int(payload.get("shards_per_plan"), default=1)),
            train_fraction=_as_float(payload.get("train_fraction"), default=0.0),
            shuffle_train_shards=bool(payload.get("shuffle_train_shards", False)),
            seed=_as_int(seed_raw, default=0) if seed_raw is not None else None,
        )


class StreamingGlobalRunStateStore:
    """Persist and update global run state on disk."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Return the state file path."""
        return self._path

    def load(self) -> StreamingGlobalRunState | None:
        """Return the current state when present."""
        if not self._path.exists():
            return None
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid global run state payload at {self._path}")
        return StreamingGlobalRunState.from_dict(payload)

    def save(self, state: StreamingGlobalRunState) -> None:
        """Persist the provided state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(state.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def reserve_next_plan(self, state: StreamingGlobalRunState) -> GlobalRunPlan:
        """Reserve the next shard slice and persist pending plan state."""
        if state.pending_plan_id is not None:
            plan = _build_plan_from_pending(state)
            return plan
        if state.next_train_index >= len(state.train_shard_ids):
            raise RuntimeError("global run has no remaining shards to schedule")
        plan_index = state.completed_plans
        shard_start = state.next_train_index
        shard_end = min(
            shard_start + int(state.shards_per_plan),
            len(state.train_shard_ids),
        )
        plan_id = f"{state.run_id}-{plan_index + 1}"
        shard_ids = state.train_shard_ids[shard_start:shard_end]
        updated = replace(
            state,
            pending_plan_id=plan_id,
            pending_start=shard_start,
            pending_end=shard_end,
            updated_at=_utcnow_iso(),
        )
        self.save(updated)
        return GlobalRunPlan(
            plan_id=plan_id,
            plan_index=plan_index,
            shard_start=shard_start,
            shard_end=shard_end,
            shard_ids=tuple(shard_ids),
            total_plans=updated.total_plans(),
            is_final=shard_end >= len(state.train_shard_ids),
        )

    def mark_plan_completed(self, state: StreamingGlobalRunState, *, plan_id: str) -> StreamingGlobalRunState:
        """Advance the state once the pending plan succeeds."""
        if state.pending_plan_id is None:
            return state
        if state.pending_plan_id != plan_id:
            raise ValueError(
                f"Pending plan mismatch: expected {state.pending_plan_id}, received {plan_id}",
            )
        pending_end = state.pending_end or state.next_train_index
        updated = replace(
            state,
            next_train_index=int(pending_end),
            pending_plan_id=None,
            pending_start=None,
            pending_end=None,
            completed_plans=state.completed_plans + 1,
            updated_at=_utcnow_iso(),
        )
        self.save(updated)
        return updated


def _build_plan_from_pending(state: StreamingGlobalRunState) -> GlobalRunPlan:
    shard_start = int(state.pending_start or 0)
    shard_end = int(state.pending_end or shard_start)
    shard_ids = state.train_shard_ids[shard_start:shard_end]
    return GlobalRunPlan(
        plan_id=str(state.pending_plan_id),
        plan_index=state.completed_plans,
        shard_start=shard_start,
        shard_end=shard_end,
        shard_ids=tuple(shard_ids),
        total_plans=state.total_plans(),
        is_final=shard_end >= len(state.train_shard_ids),
    )


def _resolve_shards_per_plan(
    *,
    requested: int | None,
    worker_cap: int | None,
    config_cap: int | None,
    fallback: int,
) -> int:
    candidates = [requested, worker_cap, config_cap]
    for value in candidates:
        if value is not None and int(value) > 0:
            return int(value)
    return max(1, int(fallback))


def _summarize_limits(
    metadata: TFTStreamingMetadata,
    config: TFTStreamingConfig,
) -> StreamingLimitSummary:
    unlimited = replace(
        config,
        max_total_rows=None,
        max_total_sequences=None,
        max_shards=None,
    )
    _, limits = stream.apply_streaming_limits(metadata, unlimited)
    return limits


def _prepare_train_and_val_metadata(
    metadata: TFTStreamingMetadata,
    train_fraction: float,
    *,
    shuffle_train_shards: bool,
    seed: int | None,
) -> tuple[TFTStreamingMetadata, TFTStreamingMetadata, tuple[str, ...]]:
    clamped_fraction = max(0.0, min(1.0, float(train_fraction)))
    train_metadata, val_metadata = split_metadata_by_row_fraction(metadata, clamped_fraction)
    order_indices = resolve_shard_order(
        train_metadata,
        shuffle=shuffle_train_shards,
        seed=seed,
    )
    ordered_train_ids = tuple(
        train_metadata.shard_indices[int(idx)].shard_id
        for idx in order_indices
    )
    return train_metadata, val_metadata, ordered_train_ids


class StreamingGlobalRunPlanner(DatasetPlanner):
    """Dataset planner that slices training shards across multiple plans."""

    def __init__(
        self,
        config: DatasetServiceConfig,
        *,
        global_config: StreamingGlobalRunConfig,
        state_store: StreamingGlobalRunStateStore,
        train_fraction: float,
        worker_max_shards: int | None,
    ) -> None:
        super().__init__(config)
        self._global_config = global_config
        self._state_store = state_store
        self._train_fraction = float(train_fraction)
        self._worker_max_shards = worker_max_shards
        self._full_metadata: TFTStreamingMetadata | None = None
        self._train_metadata: TFTStreamingMetadata | None = None
        self._val_metadata: TFTStreamingMetadata | None = None
        self._ordered_train_ids: tuple[str, ...] | None = None
        self._resolved_config: TFTStreamingConfig | None = None

    def has_remaining_plans(self) -> bool:
        """Return True when additional training slices remain."""
        state = self._state_store.load()
        if state is None:
            return True
        return not state.is_complete()

    def mark_plan_completed(
        self,
        plan: DatasetPlanEvent,
        result: TrainingResultEvent,
    ) -> bool:
        """Advance the global run state after a successful plan."""
        if result.status != EventStatus.SUCCESS:
            logger.warning(
                "global run plan did not complete successfully; state not advanced",
                extra={
                    "plan_id": plan.plan_id,
                    "dataset_id": plan.dataset_id,
                    "status": result.status.value,
                },
            )
            return False
        state = self._state_store.load()
        if state is None:
            logger.warning(
                "global run state missing when completing plan",
                extra={"plan_id": plan.plan_id, "dataset_id": plan.dataset_id},
            )
            return False
        updated = self._state_store.mark_plan_completed(state, plan_id=plan.plan_id)
        return updated.is_complete()

    def plan(self, request: DatasetPlanRequest) -> DatasetPlanEvent:
        parquet_path = self._resolve_parquet_path(request)
        numeric_columns = ensure_target_in_numeric(
            request.numeric_columns,
            request.streaming_config.target_col,
        )
        planner_config = apply_service_caps(self.config, request.streaming_config)
        self._resolved_config = planner_config
        state = self._ensure_state(
            request,
            parquet_path=parquet_path,
            numeric_columns=numeric_columns,
            planner_config=planner_config,
        )
        plan = self._state_store.reserve_next_plan(state)
        if self._train_metadata is None or self._val_metadata is None or self._full_metadata is None:
            raise RuntimeError("Global run metadata not initialized")
        train_metadata = filter_metadata_by_shard_ids(self._train_metadata, plan.shard_ids)
        val_metadata = self._val_metadata
        union_ids = set(plan.shard_ids) | set(state.val_shard_ids)
        combined_metadata = filter_metadata_by_shard_ids(
            self._full_metadata,
            union_ids,
        )
        summary = summarize_metadata(combined_metadata)
        limits = _summarize_limits(combined_metadata, planner_config)
        status = EventStatus.SUCCESS if train_metadata.shard_indices else EventStatus.DEFERRED
        plan_streaming_config = planner_config
        if planner_config.max_shards is not None:
            plan_streaming_config = replace(
                planner_config,
                max_shards=min(int(planner_config.max_shards), int(state.shards_per_plan)),
            )
        caps: dict[str, float | int | None] = {
            "shard_row_budget": int(self.config.shard_row_budget),
            "max_shards": plan_streaming_config.max_shards,
            "max_total_rows": plan_streaming_config.max_total_rows,
            "max_total_sequences": plan_streaming_config.max_total_sequences,
            "global_plan_index": plan.plan_index,
            "global_plan_total": plan.total_plans,
            "global_shard_start": plan.shard_start,
            "global_shard_end": plan.shard_end,
            "global_shards_per_plan": state.shards_per_plan,
            "global_train_fraction": state.train_fraction,
            "global_train_shards_total": len(state.train_shard_ids),
            "global_val_shards_total": len(state.val_shard_ids),
            "global_val_overlap_shards": len(set(state.train_shard_ids) & set(state.val_shard_ids)),
        }
        if plan_streaming_config.seed is not None:
            caps["dataset_seed"] = int(plan_streaming_config.seed)

        logger.info(
            "global run plan ready",
            extra={
                "plan_id": plan.plan_id,
                "dataset_id": request.dataset_id,
                "status": status.value,
                "train_shards": len(train_metadata.shard_indices),
                "val_shards": len(val_metadata.shard_indices),
                "plan_index": plan.plan_index,
                "total_plans": plan.total_plans,
            },
        )
        plan_event = DatasetPlanEvent(
            plan_id=plan.plan_id,
            dataset_id=request.dataset_id,
            parquet_path=parquet_path,
            metadata=combined_metadata,
            metadata_summary=summary,
            limits=limits,
            streaming_config=plan_streaming_config,
            caps=caps,
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            checkpoint_key=state.run_id,
            phase_one_signals=combined_metadata.phase_one_signals,
            status=status,
        )
        enforce_dataset_guardrails(
            plan_event,
            request=request,
            service_config=self.config,
        )
        return plan_event

    def _ensure_state(
        self,
        request: DatasetPlanRequest,
        *,
        parquet_path: Path,
        numeric_columns: tuple[str, ...],
        planner_config: TFTStreamingConfig,
    ) -> StreamingGlobalRunState:
        state = self._state_store.load()
        if self._full_metadata is None:
            metadata = stream.collect_streaming_metadata(
                parquet_path,
                feature_names=request.feature_names,
                categorical_columns=request.categorical_columns,
                numeric_columns=numeric_columns,
                target_col=planner_config.target_col,
                group_id_col=planner_config.group_id_col,
                time_index_col=planner_config.time_idx_col,
                shard_row_budget=int(self.config.shard_row_budget),
                phase_one_signals=request.phase_one_signals,
            )
            self._full_metadata = metadata
        else:
            metadata = self._full_metadata

        shuffle_train = self._global_config.shuffle_train_shards
        if shuffle_train is None:
            shuffle_train = bool(planner_config.shuffle_shards)
        seed = planner_config.seed

        if state is None:
            if self._train_metadata is None or self._val_metadata is None or self._ordered_train_ids is None:
                train_metadata, val_metadata, ordered_ids = _prepare_train_and_val_metadata(
                    metadata,
                    self._train_fraction,
                    shuffle_train_shards=bool(shuffle_train),
                    seed=seed,
                )
                self._train_metadata = train_metadata
                self._val_metadata = val_metadata
                self._ordered_train_ids = ordered_ids
            shards_per_plan = _resolve_shards_per_plan(
                requested=self._global_config.shards_per_plan,
                worker_cap=self._worker_max_shards,
                config_cap=planner_config.max_shards,
                fallback=len(self._ordered_train_ids or ()),
            )
            run_id = (
                self._global_config.run_id.strip()
                if self._global_config.run_id is not None
                else f"{request.dataset_id}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
            )
            state = StreamingGlobalRunState(
                run_id=run_id,
                dataset_id=request.dataset_id,
                created_at=_utcnow_iso(),
                updated_at=_utcnow_iso(),
                train_shard_ids=self._ordered_train_ids or (),
                val_shard_ids=tuple(
                    shard.shard_id for shard in self._val_metadata.shard_indices
                ),
                next_train_index=0,
                pending_plan_id=None,
                pending_start=None,
                pending_end=None,
                completed_plans=0,
                shards_per_plan=shards_per_plan,
                train_fraction=float(self._train_fraction),
                shuffle_train_shards=bool(shuffle_train),
                seed=int(seed) if seed is not None else None,
            )
            self._state_store.save(state)
        else:
            if self._global_config.run_id and state.run_id != self._global_config.run_id:
                raise ValueError(
                    f"Run id mismatch: state has {state.run_id}, requested {self._global_config.run_id}",
                )
            if state.dataset_id != request.dataset_id:
                raise ValueError(
                    f"Dataset mismatch: state has {state.dataset_id}, requested {request.dataset_id}",
                )
            shuffle_train = state.shuffle_train_shards
            seed = state.seed

        if self._train_metadata is None or self._val_metadata is None or self._ordered_train_ids is None:
            if state is not None:
                self._train_metadata = filter_metadata_by_shard_ids(
                    metadata,
                    state.train_shard_ids,
                )
                self._val_metadata = filter_metadata_by_shard_ids(
                    metadata,
                    state.val_shard_ids,
                )
                self._ordered_train_ids = state.train_shard_ids
        return state

    def _resolve_parquet_path(self, request: DatasetPlanRequest) -> Path:
        if request.parquet_path is not None:
            parquet_path = Path(request.parquet_path)
        else:
            parquet_path = Path(self.config.parquet_root) / request.dataset_id
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet dataset not found at {parquet_path}")
        return parquet_path


__all__ = [
    "GlobalRunPlan",
    "StreamingGlobalRunPlanner",
    "StreamingGlobalRunState",
    "StreamingGlobalRunStateStore",
]
