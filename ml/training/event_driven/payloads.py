"""Schema helpers for event-driven streaming training payloads."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import TrainingHeartbeatEvent
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry


SCHEMA_VERSION: Final[str] = "1.0.0"
_NAMESPACE: Final[uuid.UUID] = uuid.UUID("3c3f1b7b-6866-4e2c-9d77-fad9b21681ef")
_DEFAULT_SOURCE: Final[Source] = Source.HISTORICAL
_ISO_TIMESPEC: Final[str] = "microseconds"


def _isoformat(timestamp: datetime) -> str:
    """Return an ISO-8601 formatted timestamp with microsecond precision (UTC assumed)."""
    return timestamp.replace(tzinfo=None).isoformat(timespec=_ISO_TIMESPEC) + "Z"


def _make_correlation_id(*parts: str) -> str:
    """Generate a deterministic correlation identifier using UUIDv5."""
    joined = "|".join(parts)
    return str(uuid.uuid5(_NAMESPACE, joined))


def _to_serializable_sequence(values: Sequence[str] | tuple[str, ...]) -> list[str]:
    return [str(item) for item in values]


def _telemetry_dict(telemetry: StreamingRunTelemetry) -> dict[str, Any]:
    payload = telemetry.as_dict()

    def _coerce_mapping(value: object) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return {str(k): v for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items: list[tuple[str, Any]] = []
            for element in value:
                if (
                    isinstance(element, Sequence)
                    and not isinstance(element, (str, bytes))
                    and len(element) == 2
                ):
                    key_obj, value_obj = element[0], element[1]
                    items.append((str(key_obj), value_obj))
            if items:
                return dict(items)
        return {}

    result: dict[str, Any] = {
        "caps": _coerce_mapping(payload.get("caps")),
        "metadata": _coerce_mapping(payload.get("metadata")),
        "train": _coerce_mapping(payload.get("train")),
        "validation": _coerce_mapping(payload.get("validation")),
        "resources": _coerce_mapping(payload.get("resources")),
    }
    ensemble_payload = payload.get("ensemble")
    if isinstance(ensemble_payload, Mapping):
        members = ensemble_payload.get("members")
        result["ensemble"] = {
            **{
                key: value
                for key, value in ensemble_payload.items()
                if key != "members"
            },
            "members": list(members) if isinstance(members, Sequence) else [],
        }
    economic_payload = payload.get("economic")
    if isinstance(economic_payload, Mapping):
        result["economic"] = {str(key): economic_payload[key] for key in economic_payload}
    stability_payload = payload.get("stability")
    if isinstance(stability_payload, Mapping):
        result["stability"] = {str(key): stability_payload[key] for key in stability_payload}
    validation_returns_payload = payload.get("validation_returns")
    if isinstance(validation_returns_payload, Mapping):
        result["validation_returns"] = {
            str(key): validation_returns_payload[key]
            for key in validation_returns_payload
        }
    return result


def _capability_flags_from_config(config: TFTStreamingConfig) -> dict[str, bool]:
    return {
        "include_macro": bool(getattr(config, "include_macro", False)),
        "include_calendar": bool(getattr(config, "include_calendar", False)),
        "include_events": bool(getattr(config, "include_events", False)),
        "include_earnings": bool(getattr(config, "include_earnings", False)),
        "include_micro": bool(getattr(config, "include_micro", False)),
        "include_l2": bool(getattr(config, "include_l2", False)),
        "include_macro_revisions": bool(getattr(config, "include_macro_revisions", False)),
        "include_macro_deltas": bool(getattr(config, "include_macro_deltas", False)),
        "include_calendar_lags": bool(getattr(config, "include_calendar_lags", False)),
        "include_clustering_tags": bool(getattr(config, "include_clustering_tags", False)),
        "include_context_features": bool(getattr(config, "include_context_features", False)),
    }


def _publication_lags_from_config(config: TFTStreamingConfig) -> dict[str, int]:
    def _coerce(name: str) -> int:
        raw = getattr(config, name, 0)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    return {
        "macro_lag_days": _coerce("macro_lag_days"),
        "earnings_lag_days": _coerce("earnings_lag_days"),
        "events_notice_minutes": _coerce("events_notice_minutes"),
    }


@dataclass(slots=True, frozen=True)
class StreamingPlanMessage:
    """Schema for dataset plan events emitted to the message bus."""

    schema_version: str
    stage: Stage
    source: Source
    status: EventStatus
    correlation_id: str
    dataset_id: str
    plan_id: str
    checkpoint_key: str | None
    created_at: str
    parquet_path: str
    caps: Mapping[str, float | int | None]
    limits: Mapping[str, Any]
    metadata_summary: Mapping[str, int]
    streaming_config: Mapping[str, Any]
    capability_flags: Mapping[str, bool]
    publication_lags: Mapping[str, int]
    phase_one_signals: Mapping[str, Sequence[str]]

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "created_at": self.created_at,
            "parquet_path": self.parquet_path,
            "caps": dict(self.caps),
            "limits": dict(self.limits),
            "metadata_summary": dict(self.metadata_summary),
            "streaming_config": dict(self.streaming_config),
            "capability_flags": dict(self.capability_flags),
            "publication_lags": dict(self.publication_lags),
            "phase_one_signals": {
                key: list(values) for key, values in self.phase_one_signals.items()
            },
        }
        if self.checkpoint_key is not None:
            payload["checkpoint_key"] = self.checkpoint_key
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "source": self.source.value,
            "status": self.status.value,
            "correlation_id": self.correlation_id,
            "dataset_id": self.dataset_id,
            "plan_id": self.plan_id,
            "payload_type": "streaming_plan",
            "payload": payload,
        }


@dataclass(slots=True, frozen=True)
class StreamingResultMessage:
    """Schema for bounded streaming training results."""

    schema_version: str
    stage: Stage
    source: Source
    status: EventStatus
    correlation_id: str
    dataset_id: str
    plan_id: str
    completed_at: str
    model_id: str
    metrics: Mapping[str, float]
    artifact_paths: Mapping[str, str]
    telemetry: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "source": self.source.value,
            "status": self.status.value,
            "correlation_id": self.correlation_id,
            "dataset_id": self.dataset_id,
            "plan_id": self.plan_id,
            "payload_type": "streaming_result",
            "payload": {
                "completed_at": self.completed_at,
                "model_id": self.model_id,
                "metrics": dict(self.metrics),
                "artifact_paths": dict(self.artifact_paths),
                "telemetry": dict(self.telemetry),
            },
        }


@dataclass(slots=True, frozen=True)
class StreamingHeartbeatMessage:
    """Schema for worker heartbeat events published to the bus."""

    schema_version: str
    stage: Stage
    source: Source
    status: EventStatus
    correlation_id: str
    dataset_id: str
    plan_id: str
    worker_id: str
    progress_pct: float
    rss_mb: float
    shards_processed: int
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "source": self.source.value,
            "status": self.status.value,
            "correlation_id": self.correlation_id,
            "dataset_id": self.dataset_id,
            "plan_id": self.plan_id,
            "payload_type": "streaming_heartbeat",
            "payload": {
                "worker_id": self.worker_id,
                "progress_pct": self.progress_pct,
                "rss_mb": self.rss_mb,
                "shards_processed": self.shards_processed,
                "timestamp": self.timestamp,
            },
        }


def build_plan_message(plan: DatasetPlanEvent, *, source: Source = _DEFAULT_SOURCE) -> StreamingPlanMessage:
    """Create a message payload for the provided dataset plan."""
    summary = plan.metadata_summary
    limits = plan.limits
    config = plan.streaming_config
    capability_flags = _capability_flags_from_config(config)
    phase_one_payload = plan.phase_one_signals.as_payload()
    streaming_config = {
        "time_idx_col": config.time_idx_col,
        "group_id_col": config.group_id_col,
        "target_col": config.target_col,
        "static_categoricals": _to_serializable_sequence(config.static_categoricals),
        "static_reals": _to_serializable_sequence(config.static_reals),
        "time_varying_known_reals": _to_serializable_sequence(config.time_varying_known_reals),
        "time_varying_unknown_reals": _to_serializable_sequence(config.time_varying_unknown_reals),
        "batch_size": config.batch_size,
        "max_encoder_length": config.max_encoder_length,
        "max_prediction_length": config.max_prediction_length,
        "max_total_rows": config.max_total_rows,
        "max_total_sequences": config.max_total_sequences,
        "max_shards": config.max_shards,
        "drop_last": config.drop_last,
        "shuffle_shards": config.shuffle_shards,
        "seed": config.seed,
        "num_workers": config.num_workers,
        "include_macro": config.include_macro,
        "include_calendar": config.include_calendar,
        "include_events": config.include_events,
        "include_earnings": config.include_earnings,
        "include_micro": config.include_micro,
        "include_l2": config.include_l2,
        "include_macro_revisions": config.include_macro_revisions,
        "include_macro_deltas": config.include_macro_deltas,
        "include_calendar_lags": config.include_calendar_lags,
        "include_clustering_tags": config.include_clustering_tags,
        "include_context_features": config.include_context_features,
        "macro_lag_days": config.macro_lag_days,
        "earnings_lag_days": config.earnings_lag_days,
        "events_notice_minutes": config.events_notice_minutes,
        "phase_one_signals": phase_one_payload,
    }
    return StreamingPlanMessage(
        schema_version=SCHEMA_VERSION,
        stage=Stage.DATASET_PLANNED,
        source=source,
        status=plan.status,
        correlation_id=_make_correlation_id(plan.dataset_id, plan.plan_id, "plan"),
        dataset_id=plan.dataset_id,
        plan_id=plan.plan_id,
        checkpoint_key=plan.checkpoint_key,
        created_at=_isoformat(plan.created_at),
        parquet_path=str(plan.parquet_path),
        caps=dict(plan.caps),
        limits={
            "skipped_shards": limits.skipped_shards,
            "skipped_rows": limits.skipped_rows,
            "skipped_sequences": limits.skipped_sequences,
            "instrument_rows_total": dict(limits.total_instrument_rows),
            "instrument_rows_selected": dict(limits.selected_instrument_rows),
            "instrument_rows_skipped": dict(limits.skipped_instrument_rows),
            "instrument_sequences_total": dict(limits.total_instrument_sequences),
            "instrument_sequences_selected": dict(limits.selected_instrument_sequences),
            "instrument_sequences_skipped": dict(limits.skipped_instrument_sequences),
        },
        metadata_summary={
            "total_shards": summary.total_shards,
            "total_rows": summary.total_rows,
            "max_shard_rows": summary.max_shard_rows,
        },
        streaming_config=streaming_config,
        capability_flags=capability_flags,
        publication_lags=_publication_lags_from_config(config),
        phase_one_signals=phase_one_payload,
    )


def build_result_message(
    result: TrainingResultEvent,
    *,
    source: Source = _DEFAULT_SOURCE,
    telemetry: StreamingRunTelemetry | None = None,
) -> StreamingResultMessage:
    """Create a message payload for a completed streaming training job."""
    telemetry_obj = telemetry or result.telemetry
    return StreamingResultMessage(
        schema_version=SCHEMA_VERSION,
        stage=Stage.MODEL_TRAINING_COMPLETED,
        source=source,
        status=result.status,
        correlation_id=_make_correlation_id(result.dataset_id, result.plan_id, result.model_id, "result"),
        dataset_id=result.dataset_id,
        plan_id=result.plan_id,
        completed_at=_isoformat(result.completed_at),
        model_id=result.model_id,
        metrics=dict(result.metrics),
        artifact_paths=dict(result.artifact_paths),
        telemetry=_telemetry_dict(telemetry_obj),
    )


def build_heartbeat_message(
    heartbeat: TrainingHeartbeatEvent,
    *,
    dataset_id: str,
    source: Source = _DEFAULT_SOURCE,
) -> StreamingHeartbeatMessage:
    """Create a message payload for worker heartbeat updates."""
    status = EventStatus.SUCCESS if heartbeat.progress_pct >= 100.0 else EventStatus.PARTIAL
    plan_id = heartbeat.plan_id or dataset_id
    return StreamingHeartbeatMessage(
        schema_version=SCHEMA_VERSION,
        stage=Stage.WORKER_HEARTBEAT,
        source=source,
        status=status,
        correlation_id=_make_correlation_id(dataset_id, plan_id, heartbeat.worker_id, "heartbeat"),
        dataset_id=dataset_id,
        plan_id=plan_id,
        worker_id=heartbeat.worker_id,
        progress_pct=float(heartbeat.progress_pct),
        rss_mb=float(heartbeat.rss_mb),
        shards_processed=int(heartbeat.shards_processed),
        timestamp=_isoformat(heartbeat.timestamp),
    )


__all__ = [
    "SCHEMA_VERSION",
    "StreamingHeartbeatMessage",
    "StreamingPlanMessage",
    "StreamingResultMessage",
    "build_heartbeat_message",
    "build_plan_message",
    "build_result_message",
]
