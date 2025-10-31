"""Streaming training persistence service for bus-driven payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.config.bus import MessageBusConfig
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer
from ml.consumers.streaming_training import FileBackedStreamingTrainingStateStore
from ml.consumers.streaming_training import ObservabilitySink
from ml.consumers.streaming_training import StreamingTrainingConsumer
from ml.consumers.streaming_training import StreamingTrainingStateStore


class StreamingEventGate:
    """Idempotent gate tailored to streaming plan/result payloads."""

    __slots__ = ("_seen_ids",)

    def __init__(self) -> None:
        self._seen_ids: set[str] = set()

    def process(self, payload: Mapping[str, Any]) -> bool:
        """Return True for unseen correlation or plan identifiers."""
        correlation_id = str(payload.get("correlation_id", "")).strip()
        dataset_id = str(payload.get("dataset_id", "")).strip()
        plan_id = str(payload.get("plan_id", "")).strip()
        stage = str(payload.get("stage", "")).strip()

        dedupe_key = correlation_id or ":".join(
            value for value in (dataset_id, plan_id, stage) if value
        )
        if not dedupe_key:
            return True
        if dedupe_key in self._seen_ids:
            return False
        self._seen_ids.add(dedupe_key)
        return True


@dataclass(slots=True)
class StreamingTrainingPersistenceService:
    """Persist streaming training events to a durable state store."""

    state_path: Path
    state_store: StreamingTrainingStateStore
    consumer: StreamingTrainingConsumer

    @classmethod
    def create(
        cls,
        *,
        state_path: Path,
        observability: ObservabilitySink | None = None,
        state_store: StreamingTrainingStateStore | None = None,
    ) -> StreamingTrainingPersistenceService:
        """Instantiate the service with a file-backed state store."""
        resolved_path = state_path.expanduser()
        store = state_store or FileBackedStreamingTrainingStateStore(resolved_path)
        consumer = StreamingTrainingConsumer(state_store=store, observability=observability)
        return cls(
            state_path=resolved_path,
            state_store=store,
            consumer=consumer,
        )

    def handle(self, topic: str, payload: Mapping[str, Any]) -> None:
        """Persist a single streaming training payload."""
        self.consumer.handle(topic, dict(payload))

    def snapshot(self) -> Mapping[str, Any]:
        """Return the current state snapshot."""
        return self.state_store.snapshot()

    def create_stream_consumer(
        self,
        config: MessageBusConfig | None = None,
    ) -> RedisStreamsConsumer:
        """Build a Redis streams consumer wired to this persistence service."""
        cfg = config or MessageBusConfig.from_env()
        if not cfg.enabled or cfg.backend != "redis":
            raise RuntimeError("streaming persistence requires redis backend")

        def _handler(topic: str, event: dict[str, Any]) -> None:
            self.handle(topic, event)

        return RedisStreamsConsumer(
            url=cfg.redis_url,
            stream=cfg.redis_stream,
            handler=_handler,
            gate=StreamingEventGate(),
        )


__all__ = ["StreamingEventGate", "StreamingTrainingPersistenceService"]
