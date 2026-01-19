"""Orchestrator, buses, and helpers for the streaming training pipeline."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

from ml.common.message_bus import MessagePublisherProtocol
from ml.common.message_bus import publisher_from_config
from ml.common.message_topics import build_topic_for_stage
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.config.bus import MessageBusConfig
from ml.config.events import Source
from ml.config.events import Stage
from ml.config.streaming_pipeline import TrainingOrchestratorConfig
from ml.training.event_driven.payloads import build_heartbeat_message
from ml.training.event_driven.payloads import build_plan_message
from ml.training.event_driven.payloads import build_result_message
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import DatasetPlanner
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.event_driven.services import OrchestratorBus
from ml.training.event_driven.services import StreamingTrainingOrchestrator
from ml.training.event_driven.services import TrainingHeartbeatEvent
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.event_driven.services import TrainingWorker


logger = logging.getLogger(__name__)

_BUS_PUBLISH_ATTEMPTS = get_counter(
    "ml_tft_streaming_bus_publish_attempts_total",
    "Total streaming bus publish attempts grouped by outcome.",
    labelnames=("outcome",),
)
_BACKLOG_GAUGE = get_gauge(
    "ml_tft_streaming_training_backlog",
    "Outstanding streaming training plans per dataset.",
    labelnames=("dataset_id",),
)
_ADAPTIVE_DEFERRALS = get_counter(
    "ml_tft_streaming_orchestrator_adaptive_deferrals_total",
    "Adaptive scheduling deferrals grouped by reason.",
    labelnames=("reason",),
)
_ADAPTIVE_COOLDOWN_SECONDS = get_gauge(
    "ml_tft_streaming_orchestrator_adaptive_cooldown_seconds",
    "Current adaptive cooldown window per dataset (seconds).",
    labelnames=("dataset_id",),
)

def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_iso_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return _utcnow()


@dataclass(slots=True)
class PublishedPlanEvent:
    """Capture a dataset plan event along with the topic it was published to."""

    topic: str
    event: DatasetPlanEvent


@dataclass(slots=True)
class PublishedResultEvent:
    """Capture a training result event along with the topic it was published to."""

    topic: str
    event: TrainingResultEvent


@dataclass(slots=True)
class PublishedHeartbeatEvent:
    """Capture a worker heartbeat along with the topic it was published to."""

    topic: str
    event: TrainingHeartbeatEvent


class AdaptiveSchedulingDeferred(RuntimeError):
    """Raised when adaptive scheduling defers plan publication."""


class InMemoryOrchestratorBus(OrchestratorBus):
    """In-memory bus collecting plan, result, and heartbeat events for testing."""

    def __init__(self) -> None:
        self.plan_events: list[PublishedPlanEvent] = []
        self.result_events: list[PublishedResultEvent] = []
        self.heartbeats: list[PublishedHeartbeatEvent] = []

    def publish_plan(self, topic: str, event: DatasetPlanEvent) -> None:
        self.plan_events.append(PublishedPlanEvent(topic=topic, event=event))

    def publish_result(self, topic: str, event: TrainingResultEvent) -> None:
        self.result_events.append(PublishedResultEvent(topic=topic, event=event))

    def publish_heartbeat(self, topic: str, event: TrainingHeartbeatEvent) -> None:
        self.heartbeats.append(PublishedHeartbeatEvent(topic=topic, event=event))


class PublisherOrchestratorBus(OrchestratorBus):
    """Orchestrator bus that forwards events to a message bus publisher."""

    def __init__(
        self,
        publisher: MessagePublisherProtocol,
        *,
        default_source: Source = Source.HISTORICAL,
        max_attempts: int = 1,
        retry_delay_seconds: float = 0.0,
    ) -> None:
        self._publisher = publisher
        self._default_source = default_source
        self._max_attempts = max(1, int(max_attempts))
        self._retry_delay = max(0.0, float(retry_delay_seconds))
        self._failed_events: list[tuple[str, dict[str, Any]]] = []

    @property
    def failed_events(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Return payloads that failed to publish after retries."""
        return tuple(self._failed_events)

    def publish_plan(self, topic: str, event: DatasetPlanEvent) -> None:
        message = build_plan_message(event, source=self._default_source)
        self._publish_with_retry(topic, message.as_dict(), context={"plan_id": event.plan_id})

    def publish_result(self, topic: str, event: TrainingResultEvent) -> None:
        message = build_result_message(event, source=self._default_source)
        self._publish_with_retry(topic, message.as_dict(), context={"plan_id": event.plan_id})

    def publish_heartbeat(self, topic: str, event: TrainingHeartbeatEvent) -> None:
        dataset_id = event.dataset_id or (event.plan_id or "UNKNOWN")
        message = build_heartbeat_message(
            event,
            dataset_id=dataset_id,
            source=self._default_source,
        )
        self._publish_with_retry(
            topic,
            message.as_dict(),
            context={"plan_id": event.plan_id or dataset_id, "worker_id": event.worker_id},
        )

    def _publish_with_retry(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        attempts = self._max_attempts
        for attempt in range(attempts):
            if self._publisher.publish(topic, payload):
                outcome = "success_after_retry" if attempt > 0 else "success"
                _BUS_PUBLISH_ATTEMPTS.labels(outcome=outcome).inc()
                return
            if attempt < attempts - 1 and self._retry_delay > 0.0:
                time.sleep(self._retry_delay)
        extra = {"topic": topic, **(context or {})}
        logger.warning(
            "streaming orchestrator failed to publish event after %s attempts",
            attempts,
            extra=extra,
        )
        self._failed_events.append((topic, payload))
        _BUS_PUBLISH_ATTEMPTS.labels(outcome="failure").inc()


@dataclass(slots=True)
class _PlanState:
    event: DatasetPlanEvent | None
    dataset_id: str
    enqueued_at: datetime = field(default_factory=_utcnow)
    last_heartbeat: datetime | None = None
    completed: bool = False
    retry_attempts: int = 0
    next_retry_at: datetime | None = None
    last_progress_pct: float = 0.0
    stalled_heartbeats: int = 0
    saturated: bool = False


@dataclass(slots=True, frozen=True)
class _PlanStateRecord:
    plan_id: str
    dataset_id: str
    enqueued_at: str
    last_heartbeat: str | None
    completed: bool
    retry_attempts: int
    next_retry_at: str | None
    last_progress_pct: float
    stalled_heartbeats: int
    saturated: bool


class _PlanStateStore(Protocol):
    def load(self) -> tuple[_PlanStateRecord, ...]:
        ...

    def save(self, states: Mapping[str, _PlanState]) -> None:
        ...


class _FileBackedPlanStateStore:
    """Persist orchestrator plan state to a JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> tuple[_PlanStateRecord, ...]:
        if not self._path.exists():
            return ()
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("failed to load orchestrator state", exc_info=True)
            return ()
        records: list[_PlanStateRecord] = []
        if isinstance(payload, dict):
            for plan_id, entry in payload.items():
                if not isinstance(entry, Mapping):
                    continue
                try:
                    records.append(
                        _PlanStateRecord(
                            plan_id=str(plan_id),
                            dataset_id=str(entry.get("dataset_id", "")),
                            enqueued_at=str(entry.get("enqueued_at", "")),
                            last_heartbeat=(
                                str(entry["last_heartbeat"])
                                if entry.get("last_heartbeat") is not None
                                else None
                            ),
                            completed=bool(entry.get("completed", False)),
                            retry_attempts=int(entry.get("retry_attempts", 0) or 0),
                            next_retry_at=(
                                str(entry["next_retry_at"])
                                if entry.get("next_retry_at") is not None
                                else None
                            ),
                            last_progress_pct=float(entry.get("last_progress_pct", 0.0) or 0.0),
                            stalled_heartbeats=int(entry.get("stalled_heartbeats", 0) or 0),
                            saturated=bool(entry.get("saturated", False)),
                        ),
                    )
                except Exception:
                    logger.debug(
                        "invalid plan state entry encountered",
                        extra={"plan_id": plan_id},
                        exc_info=True,
                    )
        return tuple(records)

    def save(self, states: Mapping[str, _PlanState]) -> None:
        try:
            serializable = {
                plan_id: {
                    "dataset_id": state.dataset_id,
                    "enqueued_at": state.enqueued_at.isoformat(),
                    "last_heartbeat": (
                        state.last_heartbeat.isoformat() if state.last_heartbeat else None
                    ),
                    "completed": state.completed,
                    "retry_attempts": state.retry_attempts,
                    "next_retry_at": (
                        state.next_retry_at.isoformat() if state.next_retry_at else None
                    ),
                    "last_progress_pct": state.last_progress_pct,
                    "stalled_heartbeats": state.stalled_heartbeats,
                    "saturated": state.saturated,
                }
                for plan_id, state in states.items()
            }
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(serializable, separators=(",", ":")), encoding="utf-8")
            tmp_path.replace(self._path)
        except Exception:
            logger.debug("failed to persist orchestrator state", exc_info=True)


class InMemoryStreamingOrchestrator(StreamingTrainingOrchestrator):
    """Simple orchestrator that uses an in-memory bus and planner."""

    def __init__(
        self,
        config: TrainingOrchestratorConfig,
        planner: DatasetPlanner,
        bus: OrchestratorBus | None = None,
        worker: TrainingWorker | None = None,
        bus_config: MessageBusConfig | None = None,
        state_path: Path | None = None,
    ) -> None:
        cfg = bus_config or MessageBusConfig.from_env()
        resolved_bus = bus
        if resolved_bus is None:
            if cfg.enabled:
                publish_attempts = int(config.publish_retry_attempts)
                resolved_bus = PublisherOrchestratorBus(
                    publisher_from_config(cfg),
                    default_source=Source.HISTORICAL,
                    max_attempts=publish_attempts,
                    retry_delay_seconds=float(config.publish_retry_delay_seconds),
                )
            else:
                resolved_bus = InMemoryOrchestratorBus()
        super().__init__(config, planner, resolved_bus)
        self._bus = resolved_bus
        self._plans: dict[str, _PlanState] = {}
        self._worker = worker
        self._bus_config = cfg
        self._known_datasets: set[str] = set()
        self._adaptive_next_allowed: dict[str, datetime] = {}
        self._adaptive_deferral_counts: dict[str, int] = {}
        self._dataset_gpu_mb: dict[str, float] = {}
        self._plan_state_store: _PlanStateStore | None = None
        if self.config.enable_state_persistence:
            path = state_path
            if path is None:
                if os.getenv("PYTEST_CURRENT_TEST"):
                    tmp_dir = Path(tempfile.mkdtemp(prefix="ml_streaming_orchestrator_"))
                    path = tmp_dir / "streaming_orchestrator_state.json"
                else:
                    path = Path("ml_out/streaming_orchestrator_state.json")
            self._plan_state_store = _FileBackedPlanStateStore(path)
            self._load_plan_states()
        self._refresh_backlog_gauge()

    def _load_plan_states(self) -> None:
        if self._plan_state_store is None:
            return
        records = self._plan_state_store.load()
        for record in records:
            if record.completed:
                continue
            enqueued_at = _parse_iso_datetime(record.enqueued_at)
            last_hb = _parse_iso_datetime(record.last_heartbeat) if record.last_heartbeat else None
            self._plans.setdefault(
                record.plan_id,
                _PlanState(
                    event=None,
                    dataset_id=record.dataset_id or "UNKNOWN",
                    enqueued_at=enqueued_at,
                    last_heartbeat=last_hb,
                    completed=False,
                    retry_attempts=record.retry_attempts,
                    next_retry_at=(
                        _parse_iso_datetime(record.next_retry_at)
                        if record.next_retry_at
                        else None
                    ),
                    last_progress_pct=record.last_progress_pct,
                    stalled_heartbeats=record.stalled_heartbeats,
                    saturated=record.saturated,
                ),
            )
            self._known_datasets.add(self._normalize_dataset_id(record.dataset_id))

    def _persist_plan_states(self) -> None:
        if self._plan_state_store is None:
            return
        self._plan_state_store.save(self._plans)

    @property
    def bus(self) -> OrchestratorBus:
        """Expose orchestrator bus for inspection."""
        return self._bus

    def attach_worker(self, worker: TrainingWorker) -> None:
        """Register a worker instance for automatic execution."""
        self._worker = worker

    def _normalize_dataset_id(self, dataset_id: str | None) -> str:
        value = (dataset_id or "UNKNOWN").strip()
        return value or "UNKNOWN"

    def _schedule_adaptive_deferral(self, dataset_id: str, reason: str, *, now: datetime) -> tuple[bool, str]:
        base_cooldown = float(self.config.adaptive_cooldown_seconds)
        multiplier = float(self.config.adaptive_interval_multiplier)
        count = self._adaptive_deferral_counts.get(dataset_id, 0)
        cooldown_seconds = base_cooldown * (multiplier**count)
        self._adaptive_deferral_counts[dataset_id] = count + 1
        self._adaptive_next_allowed[dataset_id] = now + timedelta(seconds=cooldown_seconds)
        _ADAPTIVE_DEFERRALS.labels(reason=reason).inc()
        _ADAPTIVE_COOLDOWN_SECONDS.labels(dataset_id=dataset_id).set(cooldown_seconds)
        logger.info(
            "adaptive scheduling deferred dataset",
            extra={
                "dataset_id": dataset_id,
                "reason": reason,
                "cooldown_seconds": cooldown_seconds,
                "deferral_count": count + 1,
                "next_allowed_at": self._adaptive_next_allowed[dataset_id].isoformat(),
            },
        )
        return True, reason

    def _should_defer_dataset(self, dataset_id: str) -> tuple[bool, str | None]:
        now = _utcnow()
        next_allowed = self._adaptive_next_allowed.get(dataset_id)
        if next_allowed is not None and now < next_allowed:
            _ADAPTIVE_DEFERRALS.labels(reason="cooldown").inc()
            return True, "cooldown"
        backlog_threshold = self.config.adaptive_backlog_threshold
        if backlog_threshold is not None:
            outstanding = sum(
                1
                for state in self._plans.values()
                if not state.completed and self._normalize_dataset_id(state.dataset_id) == dataset_id
            )
            if outstanding >= int(backlog_threshold):
                return self._schedule_adaptive_deferral(dataset_id, "backlog", now=now)
        gpu_threshold = self.config.adaptive_gpu_threshold_mb
        if gpu_threshold is not None:
            observed_gpu = self._dataset_gpu_mb.get(dataset_id)
            if observed_gpu is not None and observed_gpu >= float(gpu_threshold):
                return self._schedule_adaptive_deferral(dataset_id, "gpu", now=now)
        return False, None

    def _refresh_backlog_gauge(self) -> None:
        counts: dict[str, int] = {}
        for state in self._plans.values():
            if state.completed:
                continue
            key = self._normalize_dataset_id(state.dataset_id)
            counts[key] = counts.get(key, 0) + 1
        for dataset_id, count in counts.items():
            _BACKLOG_GAUGE.labels(dataset_id=dataset_id).set(float(count))
        stale_datasets = self._known_datasets - counts.keys()
        for dataset_id in stale_datasets:
            _BACKLOG_GAUGE.labels(dataset_id=dataset_id).set(0.0)
        self._known_datasets.update(counts.keys())

    def inflight_plan_ids(self) -> tuple[str, ...]:
        """Return identifiers of currently active plans."""
        return tuple(self._plans.keys())

    def saturated_plan_ids(self) -> tuple[str, ...]:
        """Return plan identifiers whose workers appear saturated."""
        return tuple(
            plan_id
            for plan_id, state in self._plans.items()
            if state.saturated and not state.completed
        )

    def clear_backlog(
        self,
        dataset_id: str | None = None,
        *,
        include_active: bool = False,
    ) -> tuple[str, ...]:
        """Remove persisted plan state optionally scoped by dataset."""
        removed: list[str] = []
        for plan_id, state in list(self._plans.items()):
            if dataset_id is not None and state.dataset_id != dataset_id:
                continue
            if not include_active and not state.completed:
                continue
            removed.append(plan_id)
            self._plans.pop(plan_id, None)
        if removed:
            self._persist_plan_states()
            self._refresh_backlog_gauge()
        return tuple(removed)

    def resume_plan(self, plan_id: str) -> bool:
        """Reset orchestration state for a plan and allow retries."""
        state = self._plans.get(plan_id)
        if state is None:
            return False
        state.completed = False
        state.retry_attempts = 0
        state.next_retry_at = None
        state.saturated = False
        state.stalled_heartbeats = 0
        state.last_progress_pct = 0.0
        state.last_heartbeat = _utcnow()
        self._persist_plan_states()
        return True

    def enqueue_training(self, request: DatasetPlanRequest) -> DatasetPlanEvent:
        """Generate a plan and publish it when within concurrency limits."""
        if len(self._plans) >= self.config.max_in_flight_plans:
            raise RuntimeError("Max in-flight plans reached")
        dataset_key = self._normalize_dataset_id(request.dataset_id)
        should_defer, reason = self._should_defer_dataset(dataset_key)
        if should_defer:
            if reason == "cooldown":
                remaining = self._adaptive_next_allowed.get(dataset_key)
                seconds_remaining = 0.0
                if remaining is not None:
                    delta = (remaining - _utcnow()).total_seconds()
                    seconds_remaining = max(0.0, float(delta))
                logger.info(
                    "adaptive scheduling cooldown active; skipping enqueue",
                    extra={
                        "dataset_id": dataset_key,
                        "cooldown_remaining_seconds": seconds_remaining,
                    },
                )
            raise AdaptiveSchedulingDeferred(
                f"adaptive scheduling deferred dataset {dataset_key} due to {reason}",
            )
        max_pending = max(1, int(self.config.dataset_retry_limit))
        outstanding = sum(
            1
            for state in self._plans.values()
            if not state.completed and self._normalize_dataset_id(state.dataset_id) == dataset_key
        )
        if outstanding >= max_pending:
            raise RuntimeError("Dataset backlog exceeded")
        plan_event = self._planner.plan(request)
        self._adaptive_deferral_counts.pop(dataset_key, None)
        if dataset_key in self._adaptive_next_allowed:
            self._adaptive_next_allowed.pop(dataset_key, None)
            _ADAPTIVE_COOLDOWN_SECONDS.labels(dataset_id=dataset_key).set(0.0)
        self._plans[plan_event.plan_id] = _PlanState(
            event=plan_event,
            dataset_id=plan_event.dataset_id,
        )
        self._known_datasets.add(self._normalize_dataset_id(plan_event.dataset_id))
        self._persist_plan_states()
        self._refresh_backlog_gauge()
        plan_topic = self._resolve_topic(
            Stage.DATASET_PLANNED,
            fallback=self.config.command_topic,
            entity_id=plan_event.dataset_id,
        )
        self._bus.publish_plan(plan_topic, plan_event)
        if self._worker is not None:
            result = self._worker.run(plan_event)
            self.handle_result(result)
        return plan_event

    def handle_heartbeat(self, heartbeat: TrainingHeartbeatEvent) -> None:
        """Record heartbeat and publish it out on the bus."""
        dataset_id: str | None = heartbeat.dataset_id
        if heartbeat.plan_id and heartbeat.plan_id in self._plans:
            state = self._plans[heartbeat.plan_id]
            state.last_heartbeat = heartbeat.timestamp
            progress = max(0.0, float(heartbeat.progress_pct))
            if progress >= state.last_progress_pct:
                if progress > state.last_progress_pct:
                    state.stalled_heartbeats = 0
                    state.last_progress_pct = progress
                    state.saturated = False
                    state.retry_attempts = 0
                    state.next_retry_at = None
                else:
                    state.stalled_heartbeats += 1
            else:
                state.last_progress_pct = progress
                state.stalled_heartbeats = 0
            if state.stalled_heartbeats >= int(self.config.saturation_heartbeat_limit):
                state.saturated = True
            if progress >= 100.0:
                state.completed = True
                state.saturated = False
                state.next_retry_at = None
            if heartbeat.dataset_id:
                state.dataset_id = heartbeat.dataset_id
            self._known_datasets.add(self._normalize_dataset_id(state.dataset_id))
            dataset_id = state.dataset_id or dataset_id
            self._persist_plan_states()
        entity_id = (dataset_id or heartbeat.plan_id or "UNKNOWN").strip() or "UNKNOWN"
        if heartbeat.dataset_id != entity_id:
            heartbeat = replace(heartbeat, dataset_id=entity_id)
        heartbeat_topic = self._resolve_topic(
            Stage.WORKER_HEARTBEAT,
            fallback=self.config.heartbeat_topic,
            entity_id=entity_id,
        )
        self._bus.publish_heartbeat(heartbeat_topic, heartbeat)

    def mark_result(self, result: TrainingResultEvent) -> None:
        """Mark plan as completed and publish training result."""
        self.handle_result(result)

    def handle_result(self, result: TrainingResultEvent) -> None:
        """Handle worker results and update orchestrator lifecycle."""
        dataset_id = result.dataset_id
        telemetry = result.telemetry
        dataset_key = self._normalize_dataset_id(dataset_id)
        if telemetry.max_gpu_memory_mb is not None:
            self._dataset_gpu_mb[dataset_key] = float(telemetry.max_gpu_memory_mb)
        if result.plan_id in self._plans:
            state = self._plans.pop(result.plan_id)
            state.completed = True
            dataset_id = state.dataset_id or dataset_id
            self._persist_plan_states()
            self._refresh_backlog_gauge()
        result_topic = self._resolve_topic(
            Stage.MODEL_TRAINING_COMPLETED,
            fallback=self.config.result_topic,
            entity_id=dataset_id or "UNKNOWN",
        )
        self._bus.publish_result(result_topic, result)

    def expired_plans(self) -> list[DatasetPlanEvent]:
        """Return plans whose heartbeats have timed out and are ready for retry."""
        now = _utcnow()
        timeout = timedelta(seconds=self.config.worker_timeout_seconds)
        retry_window = timedelta(seconds=self.config.retry_window_seconds)
        max_age = timedelta(seconds=self.config.max_plan_age_seconds)
        expired: list[DatasetPlanEvent] = []
        changed = False
        for plan_id, state in list(self._plans.items()):
            if state.completed or state.event is None:
                continue
            last_hb = state.last_heartbeat or state.enqueued_at
            timed_out = now - last_hb > timeout
            aged_out = now - state.enqueued_at > max_age
            if not (timed_out or aged_out):
                continue
            if state.retry_attempts >= int(self.config.dataset_retry_limit):
                self._plans.pop(plan_id, None)
                changed = True
                continue
            if state.next_retry_at is None:
                state.next_retry_at = now + retry_window
                changed = True
                continue
            if now < state.next_retry_at:
                continue
            state.retry_attempts += 1
            state.next_retry_at = now + retry_window
            state.stalled_heartbeats = 0
            state.saturated = False
            state.last_progress_pct = 0.0
            state.enqueued_at = now
            state.last_heartbeat = now
            expired.append(state.event)
            changed = True
        if changed:
            self._persist_plan_states()
            self._refresh_backlog_gauge()
        return expired

    def _resolve_topic(self, stage: Stage, *, fallback: str, entity_id: str) -> str:
        fallback_clean = fallback.strip()
        if fallback_clean:
            try:
                return fallback_clean.format(
                    dataset_id=entity_id,
                    plan_id=entity_id,
                )
            except Exception:
                return fallback_clean
        return build_topic_for_stage(
            stage,
            entity_id or "UNKNOWN",
            scheme=self._bus_config.scheme,
            prefix=self._bus_config.topic_prefix,
        )


__all__ = [
    "AdaptiveSchedulingDeferred",
    "InMemoryOrchestratorBus",
    "InMemoryStreamingOrchestrator",
    "PublishedHeartbeatEvent",
    "PublishedPlanEvent",
    "PublishedResultEvent",
    "PublisherOrchestratorBus",
]
