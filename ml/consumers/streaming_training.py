"""Consumers and state tracking for streaming training payloads."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ml.common.metrics_bootstrap import get_gauge
from ml.config.events import EventStatus


logger = logging.getLogger(__name__)

_BACKLOG_GAUGE = get_gauge(
    "ml_tft_streaming_training_backlog",
    "Number of outstanding streaming training plans awaiting completion.",
    labelnames=("dataset_id",),
)
_WORKER_PROGRESS_GAUGE = get_gauge(
    "ml_tft_streaming_worker_progress_pct",
    "Latest progress percentage reported by streaming workers.",
    labelnames=("worker_id",),
)
_WORKER_RSS_GAUGE = get_gauge(
    "ml_tft_streaming_worker_rss_mb",
    "Latest resident set size (MB) reported by streaming workers.",
    labelnames=("worker_id",),
)
_WORKER_COUNT_GAUGE = get_gauge(
    "ml_tft_streaming_workers_active",
    "Current number of active streaming workers per dataset.",
    labelnames=("dataset_id",),
)
_RESULT_METRIC_GAUGE = get_gauge(
    "ml_tft_streaming_validation_metric",
    "Validation metrics emitted by streaming training results.",
    labelnames=("dataset_id", "plan_id", "metric"),
)


class ObservabilitySink(Protocol):
    """Minimal protocol for observability collectors."""

    def add_metric(
        self,
        *,
        metric_name: str,
        metric_type: str,
        value: float,
        timestamp: int,
        labels: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> None:
        """Record a metric sample."""


def _parse_timestamp(value: str) -> datetime:
    """Return a timezone-aware ``datetime`` parsed from the provided ISO-8601 string."""
    if value.endswith("Z"):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.fromisoformat(value)


@dataclass(slots=True)
class StreamingPlanRecord:
    """Snapshot of a dataset plan emitted on the streaming pipeline bus."""

    plan_id: str
    dataset_id: str
    status: EventStatus
    created_at: datetime
    caps: Mapping[str, float | int | None]
    limits: Mapping[str, Any]
    metadata_summary: Mapping[str, int]
    streaming_config: Mapping[str, Any]
    correlation_id: str
    topic: str

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the plan record."""
        return {
            "plan_id": self.plan_id,
            "dataset_id": self.dataset_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "caps": dict(self.caps),
            "limits": dict(self.limits),
            "metadata_summary": dict(self.metadata_summary),
            "streaming_config": dict(self.streaming_config),
            "correlation_id": self.correlation_id,
            "topic": self.topic,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StreamingPlanRecord:
        """Instantiate a record from a serialized snapshot."""
        return cls(
            plan_id=str(data.get("plan_id", "")),
            dataset_id=str(data.get("dataset_id", "")),
            status=EventStatus(str(data.get("status", EventStatus.SUCCESS.value))),
            created_at=_parse_timestamp(str(data.get("created_at", ""))),
            caps=dict(data.get("caps", {})),
            limits=dict(data.get("limits", {})),
            metadata_summary=dict(data.get("metadata_summary", {})),
            streaming_config=dict(data.get("streaming_config", {})),
            correlation_id=str(data.get("correlation_id", "")),
            topic=str(data.get("topic", "")),
        )


@dataclass(slots=True)
class StreamingResultRecord:
    """Snapshot of a completed streaming training job."""

    plan_id: str
    dataset_id: str
    status: EventStatus
    completed_at: datetime
    model_id: str
    metrics: Mapping[str, float]
    artifact_paths: Mapping[str, str]
    telemetry: Mapping[str, Any]
    correlation_id: str
    topic: str

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the result record."""
        return {
            "plan_id": self.plan_id,
            "dataset_id": self.dataset_id,
            "status": self.status.value,
            "completed_at": self.completed_at.isoformat(),
            "model_id": self.model_id,
            "metrics": dict(self.metrics),
            "artifact_paths": dict(self.artifact_paths),
            "telemetry": dict(self.telemetry),
            "correlation_id": self.correlation_id,
            "topic": self.topic,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StreamingResultRecord:
        """Instantiate a record from a serialized snapshot."""
        return cls(
            plan_id=str(data.get("plan_id", "")),
            dataset_id=str(data.get("dataset_id", "")),
            status=EventStatus(str(data.get("status", EventStatus.SUCCESS.value))),
            completed_at=_parse_timestamp(str(data.get("completed_at", ""))),
            model_id=str(data.get("model_id", "")),
            metrics=dict(data.get("metrics", {})),
            artifact_paths=dict(data.get("artifact_paths", {})),
            telemetry=dict(data.get("telemetry", {})),
            correlation_id=str(data.get("correlation_id", "")),
            topic=str(data.get("topic", "")),
        )


@dataclass(slots=True)
class StreamingHeartbeatRecord:
    """Snapshot of a worker heartbeat update."""

    plan_id: str
    dataset_id: str
    status: EventStatus
    worker_id: str
    progress_pct: float
    rss_mb: float
    shards_processed: int
    timestamp: datetime
    correlation_id: str
    topic: str

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the heartbeat record."""
        return {
            "plan_id": self.plan_id,
            "dataset_id": self.dataset_id,
            "status": self.status.value,
            "worker_id": self.worker_id,
            "progress_pct": self.progress_pct,
            "rss_mb": self.rss_mb,
            "shards_processed": self.shards_processed,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "topic": self.topic,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StreamingHeartbeatRecord:
        """Instantiate a record from a serialized snapshot."""
        return cls(
            plan_id=str(data.get("plan_id", "")),
            dataset_id=str(data.get("dataset_id", "")),
            status=EventStatus(str(data.get("status", EventStatus.PARTIAL.value))),
            worker_id=str(data.get("worker_id", "")),
            progress_pct=float(data.get("progress_pct", 0.0)),
            rss_mb=float(data.get("rss_mb", 0.0)),
            shards_processed=int(data.get("shards_processed", 0)),
            timestamp=_parse_timestamp(str(data.get("timestamp", ""))),
            correlation_id=str(data.get("correlation_id", "")),
            topic=str(data.get("topic", "")),
        )


class StreamingTrainingStateStore(Protocol):
    """Protocol describing state persistence for streaming training consumers."""

    def record_plan(self, record: StreamingPlanRecord) -> None:
        """Persist metadata for a dataset plan message."""

    def record_result(self, record: StreamingResultRecord) -> None:
        """Persist metadata for a training result message."""

    def record_heartbeat(self, record: StreamingHeartbeatRecord) -> None:
        """Persist metadata for a worker heartbeat message."""

    def get_plan(self, plan_id: str) -> StreamingPlanRecord | None:
        """Return the latest plan record for ``plan_id`` if available."""

    def get_result(self, plan_id: str) -> StreamingResultRecord | None:
        """Return the latest training result for ``plan_id`` if available."""

    def latest_heartbeat(self, worker_id: str) -> StreamingHeartbeatRecord | None:
        """Return the latest heartbeat recorded for ``worker_id``."""

    def outstanding_plan_ids(self) -> tuple[str, ...]:
        """Return plan identifiers that have not yet produced a result."""

    def outstanding_plan_ids_for_dataset(self, dataset_id: str) -> tuple[str, ...]:
        """Return outstanding plan identifiers scoped to ``dataset_id``."""

    def snapshot(self) -> Mapping[str, Any]:
        """Return a serializable snapshot of the store."""

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore state from a serialized snapshot."""

    def get_stream_cursor(self) -> str | None:
        """Return the last processed Redis stream ID, if any."""

    def update_stream_cursor(self, cursor: str) -> None:
        """Persist the latest processed Redis stream ID."""


class InMemoryStreamingTrainingStateStore(StreamingTrainingStateStore):
    """In-memory implementation of :class:`StreamingTrainingStateStore`."""

    def __init__(self) -> None:
        self._plans: dict[str, StreamingPlanRecord] = {}
        self._results: dict[str, StreamingResultRecord] = {}
        self._heartbeats: dict[str, StreamingHeartbeatRecord] = {}
        self._stream_cursor: str | None = None

    def record_plan(self, record: StreamingPlanRecord) -> None:
        self._plans[record.plan_id] = record

    def record_result(self, record: StreamingResultRecord) -> None:
        self._results[record.plan_id] = record
        if record.plan_id in self._plans:
            plan = self._plans[record.plan_id]
            self._plans[record.plan_id] = replace(plan, status=record.status)

    def record_heartbeat(self, record: StreamingHeartbeatRecord) -> None:
        key = f"{record.worker_id}::{record.plan_id}"
        self._heartbeats[key] = record

    def get_plan(self, plan_id: str) -> StreamingPlanRecord | None:
        return self._plans.get(plan_id)

    def get_result(self, plan_id: str) -> StreamingResultRecord | None:
        return self._results.get(plan_id)

    def latest_heartbeat(self, worker_id: str) -> StreamingHeartbeatRecord | None:
        matches = [hb for key, hb in self._heartbeats.items() if key.startswith(f"{worker_id}::")]
        if not matches:
            return None
        matches.sort(key=lambda hb: hb.timestamp, reverse=True)
        return matches[0]

    def outstanding_plan_ids(self) -> tuple[str, ...]:
        return tuple(pid for pid in self._plans if pid not in self._results)

    def outstanding_plan_ids_for_dataset(self, dataset_id: str) -> tuple[str, ...]:
        return tuple(
            pid
            for pid, plan in self._plans.items()
            if plan.dataset_id == dataset_id and pid not in self._results
        )

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "plans": {pid: record.as_dict() for pid, record in self._plans.items()},
            "results": {pid: record.as_dict() for pid, record in self._results.items()},
            "heartbeats": {key: record.as_dict() for key, record in self._heartbeats.items()},
            "stream_cursor": self._stream_cursor,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        plans_snapshot = snapshot.get("plans", {})
        results_snapshot = snapshot.get("results", {})
        heartbeats_snapshot = snapshot.get("heartbeats", {})
        cursor_snapshot = snapshot.get("stream_cursor")

        self._plans = {
            plan_id: StreamingPlanRecord.from_dict(data)
            for plan_id, data in plans_snapshot.items()
        }
        self._results = {
            plan_id: StreamingResultRecord.from_dict(data)
            for plan_id, data in results_snapshot.items()
        }
        self._heartbeats = {
            key: StreamingHeartbeatRecord.from_dict(data)
            for key, data in heartbeats_snapshot.items()
        }
        if isinstance(cursor_snapshot, str):
            self._stream_cursor = cursor_snapshot
        else:
            self._stream_cursor = None

    def get_stream_cursor(self) -> str | None:
        return self._stream_cursor

    def update_stream_cursor(self, cursor: str) -> None:
        self._stream_cursor = cursor.strip() or None


class FileBackedStreamingTrainingStateStore(StreamingTrainingStateStore):
    """Persist streaming training state snapshots to a JSON file."""

    def __init__(
        self,
        path: Path,
        *,
        delegate: StreamingTrainingStateStore | None = None,
    ) -> None:
        self._path = path
        self._delegate = delegate or InMemoryStreamingTrainingStateStore()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                snapshot = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(snapshot, Mapping):
                    self._delegate.restore(snapshot)
            except Exception:
                logger.warning(
                    "streaming training store failed to restore snapshot",
                    extra={"path": str(self._path)},
                    exc_info=True,
                )
        self._persist()

    def record_plan(self, record: StreamingPlanRecord) -> None:
        self._delegate.record_plan(record)
        self._persist()

    def record_result(self, record: StreamingResultRecord) -> None:
        self._delegate.record_result(record)
        self._persist()

    def record_heartbeat(self, record: StreamingHeartbeatRecord) -> None:
        self._delegate.record_heartbeat(record)
        self._persist()

    def get_plan(self, plan_id: str) -> StreamingPlanRecord | None:
        return self._delegate.get_plan(plan_id)

    def get_result(self, plan_id: str) -> StreamingResultRecord | None:
        return self._delegate.get_result(plan_id)

    def latest_heartbeat(self, worker_id: str) -> StreamingHeartbeatRecord | None:
        return self._delegate.latest_heartbeat(worker_id)

    def outstanding_plan_ids(self) -> tuple[str, ...]:
        return self._delegate.outstanding_plan_ids()

    def outstanding_plan_ids_for_dataset(self, dataset_id: str) -> tuple[str, ...]:
        return self._delegate.outstanding_plan_ids_for_dataset(dataset_id)

    def snapshot(self) -> Mapping[str, Any]:
        return self._delegate.snapshot()

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self._delegate.restore(snapshot)
        self._persist()

    def get_stream_cursor(self) -> str | None:
        return self._delegate.get_stream_cursor()

    def update_stream_cursor(self, cursor: str) -> None:
        self._delegate.update_stream_cursor(cursor)
        self._persist()

    def _persist(self) -> None:
        try:
            snapshot = self._delegate.snapshot()
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as temp_file:
                json.dump(snapshot, temp_file, separators=(",", ":"), ensure_ascii=False)
            tmp_path.replace(self._path)
        except Exception:
            logger.warning(
                "streaming training store failed to persist snapshot",
                extra={"path": str(self._path)},
                exc_info=True,
            )


class StreamingTrainingConsumer:
    """
    Consume streaming training payloads and persist state.

    Args:
        state_store: Optional backing store for streaming plan/result/heartbeat state. When
            omitted, an in-memory store is instantiated.
        observability: Optional observability sink used to mirror backlog and heartbeat metrics.

    Example:
        >>> consumer = StreamingTrainingConsumer()
        >>> consumer.handle(topic, payload)
        >>> consumer.state_store.outstanding_plan_ids()
        ('plan-123',)
    """

    def __init__(
        self,
        state_store: StreamingTrainingStateStore | None = None,
        *,
        observability: ObservabilitySink | None = None,
    ) -> None:
        self._state_store = state_store or InMemoryStreamingTrainingStateStore()
        self._seen_correlations: set[str] = set()
        self._observability = observability

    @property
    def state_store(self) -> StreamingTrainingStateStore:
        """Return the backing state store."""
        return self._state_store

    def handle(self, topic: str, payload: dict[str, Any]) -> None:
        payload_type = str(payload.get("payload_type", "")).strip()
        correlation_id = str(payload.get("correlation_id", "")).strip()
        if not payload_type or not correlation_id:
            return
        if correlation_id in self._seen_correlations:
            return

        try:
            if payload_type == "streaming_plan":
                plan_record = self._parse_plan(topic, payload)
                self._state_store.record_plan(plan_record)
                self._update_backlog_metric(plan_record.dataset_id)
            elif payload_type == "streaming_result":
                result_record = self._parse_result(topic, payload)
                self._state_store.record_result(result_record)
                self._record_result_metrics(result_record)
                self._update_backlog_metric(result_record.dataset_id)
            elif payload_type == "streaming_heartbeat":
                heartbeat_record = self._parse_heartbeat(topic, payload)
                self._state_store.record_heartbeat(heartbeat_record)
                self._record_heartbeat_metrics(heartbeat_record)
            else:
                logger.debug("ignoring unsupported payload_type %s", payload_type)
                return
            self._seen_correlations.add(correlation_id)
        except Exception:  # pragma: no cover - defensive guard
            logger.warning(
                "failed to process streaming payload",
                extra={"topic": topic, "payload_type": payload_type},
                exc_info=True,
            )

    def _parse_plan(self, topic: str, payload: Mapping[str, Any]) -> StreamingPlanRecord:
        body = payload.get("payload", {})
        return StreamingPlanRecord(
            plan_id=str(payload.get("plan_id", "")),
            dataset_id=str(payload.get("dataset_id", "")),
            status=EventStatus(str(payload.get("status"))),
            created_at=_parse_timestamp(str(body.get("created_at", ""))),
            caps=dict(body.get("caps", {})),
            limits=dict(body.get("limits", {})),
            metadata_summary=dict(body.get("metadata_summary", {})),
            streaming_config=dict(body.get("streaming_config", {})),
            correlation_id=str(payload.get("correlation_id", "")),
            topic=topic,
        )

    def _parse_result(self, topic: str, payload: Mapping[str, Any]) -> StreamingResultRecord:
        body = payload.get("payload", {})
        return StreamingResultRecord(
            plan_id=str(payload.get("plan_id", "")),
            dataset_id=str(payload.get("dataset_id", "")),
            status=EventStatus(str(payload.get("status"))),
            completed_at=_parse_timestamp(str(body.get("completed_at", ""))),
            model_id=str(body.get("model_id", "")),
            metrics=dict(body.get("metrics", {})),
            artifact_paths=dict(body.get("artifact_paths", {})),
            telemetry=dict(body.get("telemetry", {})),
            correlation_id=str(payload.get("correlation_id", "")),
            topic=topic,
        )

    def _parse_heartbeat(self, topic: str, payload: Mapping[str, Any]) -> StreamingHeartbeatRecord:
        body = payload.get("payload", {})
        return StreamingHeartbeatRecord(
            plan_id=str(payload.get("plan_id", "")),
            dataset_id=str(payload.get("dataset_id", "")),
            status=EventStatus(str(payload.get("status"))),
            worker_id=str(body.get("worker_id", "")),
            progress_pct=float(body.get("progress_pct", 0.0)),
            rss_mb=float(body.get("rss_mb", 0.0)),
            shards_processed=int(body.get("shards_processed", 0)),
            timestamp=_parse_timestamp(str(body.get("timestamp", ""))),
            correlation_id=str(payload.get("correlation_id", "")),
            topic=topic,
        )

    def _update_backlog_metric(self, dataset_id: str) -> None:
        outstanding = len(self._state_store.outstanding_plan_ids_for_dataset(dataset_id))
        dataset_key = dataset_id or "UNKNOWN"
        _BACKLOG_GAUGE.labels(dataset_id=dataset_key).set(float(outstanding))
        if self._observability is not None:
            timestamp_ns = int(datetime.utcnow().timestamp() * 1_000_000_000)
            self._observability.add_metric(
                metric_name="ml_tft_streaming_training_backlog",
                metric_type="gauge",
                value=float(outstanding),
                timestamp=timestamp_ns,
                labels={"dataset_id": dataset_key},
            )

    def _record_heartbeat_metrics(self, record: StreamingHeartbeatRecord) -> None:
        worker_key = record.worker_id or "UNKNOWN"
        _WORKER_PROGRESS_GAUGE.labels(worker_id=worker_key).set(record.progress_pct)
        _WORKER_RSS_GAUGE.labels(worker_id=worker_key).set(record.rss_mb)
        dataset_key = record.dataset_id or "UNKNOWN"
        snapshot = self._state_store.snapshot()
        heartbeats_raw = snapshot.get("heartbeats", {})
        active_workers: set[str] = set()
        if isinstance(heartbeats_raw, Mapping):
            for heartbeat_payload in heartbeats_raw.values():
                if not isinstance(heartbeat_payload, Mapping):
                    continue
                worker_id = str(heartbeat_payload.get("worker_id", "")).strip()
                dataset_match = str(heartbeat_payload.get("dataset_id", dataset_key)).strip()
                if worker_id and dataset_match == dataset_key:
                    active_workers.add(worker_id)
        active_workers.discard("")
        _WORKER_COUNT_GAUGE.labels(dataset_id=dataset_key).set(float(len(active_workers)))
        if self._observability is not None:
            timestamp_ns = int(record.timestamp.timestamp() * 1_000_000_000)
            self._observability.add_metric(
                metric_name="ml_tft_streaming_worker_progress_pct",
                metric_type="gauge",
                value=float(record.progress_pct),
                timestamp=timestamp_ns,
                labels={
                    "worker_id": worker_key,
                    "plan_id": record.plan_id,
                    "dataset_id": dataset_key,
                },
            )
            self._observability.add_metric(
                metric_name="ml_tft_streaming_worker_rss_mb",
                metric_type="gauge",
                value=float(record.rss_mb),
                timestamp=timestamp_ns,
                labels={
                    "worker_id": worker_key,
                    "plan_id": record.plan_id,
                    "dataset_id": dataset_key,
                },
            )
            self._observability.add_metric(
                metric_name="ml_tft_streaming_workers_active",
                metric_type="gauge",
                value=float(len(active_workers)),
                timestamp=timestamp_ns,
                labels={"dataset_id": dataset_key},
            )

    def _record_result_metrics(self, record: StreamingResultRecord) -> None:
        if not record.metrics:
            return
        dataset_key = record.dataset_id or "UNKNOWN"
        plan_key = record.plan_id or "UNKNOWN"
        completed_at = record.completed_at if isinstance(record.completed_at, datetime) else datetime.utcnow()
        timestamp_ns = int(completed_at.timestamp() * 1_000_000_000)
        for metric_name, value in record.metrics.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            _RESULT_METRIC_GAUGE.labels(
                dataset_id=dataset_key,
                plan_id=plan_key,
                metric=metric_name,
            ).set(numeric_value)
            if self._observability is not None:
                self._observability.add_metric(
                    metric_name="ml_tft_streaming_validation_metric",
                    metric_type="gauge",
                    value=numeric_value,
                    timestamp=timestamp_ns,
                    labels={
                        "dataset_id": dataset_key,
                        "plan_id": plan_key,
                        "metric": metric_name,
                    },
                )


class SubscriptionBus(Protocol):
    """Protocol for buses supporting pattern subscriptions."""

    def subscribe(self, pattern: str, handler: Callable[[str, dict[str, Any]], None]) -> None:
        """Register a topic handler."""


def attach_streaming_training_monitor(
    bus: SubscriptionBus,
    *,
    state_path: Path,
    observability: ObservabilitySink | None = None,
    topic_pattern: str = "events.ml.#",
) -> StreamingTrainingConsumer:
    """Attach a streaming training consumer to the provided bus."""
    store = FileBackedStreamingTrainingStateStore(state_path)
    consumer = StreamingTrainingConsumer(state_store=store, observability=observability)

    def _handler(topic: str, payload: dict[str, Any]) -> None:
        consumer.handle(topic, dict(payload))

    bus.subscribe(topic_pattern, _handler)
    return consumer


__all__ = [
    "FileBackedStreamingTrainingStateStore",
    "InMemoryStreamingTrainingStateStore",
    "StreamingHeartbeatRecord",
    "StreamingPlanRecord",
    "StreamingResultRecord",
    "StreamingTrainingConsumer",
    "StreamingTrainingStateStore",
    "attach_streaming_training_monitor",
]
