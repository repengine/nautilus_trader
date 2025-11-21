"""Long-running worker that persists streaming training events from Redis Streams."""

from __future__ import annotations

import logging
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any, Protocol

from ml.config.bus import MessageBusConfig
from ml.config.streaming_pipeline import StreamingPersistenceConfig
from ml.consumers.streaming_training import ObservabilitySink
from ml.consumers.streaming_training import StreamingTrainingStateStore
from ml.consumers.streaming_training_service import StreamingTrainingPersistenceService


if TYPE_CHECKING:
    from ml.observability.service import ObservabilityService


@dataclass(slots=True)
class _ObservabilityServiceAdapter:
    service: ObservabilityService

    def add_metric(
        self,
        *,
        metric_name: str,
        metric_type: str,
        value: float,
        timestamp: int,
        labels: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> None:
        normalized: dict[str, Any]
        if labels is None:
            normalized = {}
        elif isinstance(labels, Mapping):
            normalized = {str(key): val for key, val in labels.items()}
        else:
            normalized = {str(key): val for key, val in labels}
        self.service.add_metric(
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            labels=normalized,
        )


logger = logging.getLogger(__name__)


class _PollableConsumer(Protocol):
    """Protocol describing the poll interface used by the persistence worker."""

    def poll_once(self, *, count: int, block_ms: int, last_id: str = "$") -> int:
        """Poll a backend once and return the number of processed entries."""

    @property
    def last_entry_id(self) -> str | None:
        """Return the last processed backend cursor, if any."""


ConsumerFactory = Callable[
    [StreamingTrainingPersistenceService, MessageBusConfig],
    _PollableConsumer,
]


@dataclass(slots=True)
class StreamingTrainingPersistenceWorker:
    """
    Run the streaming training persistence loop against Redis Streams.

    Args:
        config: Configuration controlling polling cadence and persistence paths.
        message_bus_config: Message bus configuration used to discover Redis endpoints.
        observability: Optional observability sink for mirroring backlog metrics.
        state_store: Optional state store implementation used instead of the default file store.
        consumer_factory: Optional factory that builds a custom consumer, primarily for tests.

    Example:
        >>> worker = StreamingTrainingPersistenceWorker(
        ...     StreamingPersistenceConfig(),
        ...     message_bus_config=MessageBusConfig.from_env(),
        ... )
        >>> worker.run_forever()  # doctest: +SKIP
    """

    config: StreamingPersistenceConfig
    message_bus_config: MessageBusConfig = field(default_factory=MessageBusConfig.from_env)
    observability: ObservabilitySink | None = None
    state_store: StreamingTrainingStateStore | None = None
    consumer_factory: ConsumerFactory | None = None

    _service: StreamingTrainingPersistenceService | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _consumer: _PollableConsumer | None = field(default=None, init=False, repr=False)
    _stop_event: Event = field(default_factory=Event, init=False, repr=False)
    _observability_initialized: bool = field(default=False, init=False, repr=False)
    _resolved_observability: ObservabilitySink | None = field(default=None, init=False, repr=False)
    _last_stream_id: str | None = field(default=None, init=False, repr=False)

    def poll_once(self) -> int:
        """Poll Redis Streams a single time using configured limits."""
        if not self.config.enabled:
            return 0
        consumer = self._ensure_consumer()
        if consumer is None:
            return 0
        last_id = self._last_stream_id or "0-0"
        try:
            processed = consumer.poll_once(
                count=int(self.config.batch_size),
                block_ms=int(self.config.block_ms),
                last_id=last_id,
            )
        except Exception:
            logger.warning(
                "streaming persistence poll failed",
                extra={"state_path": self.config.state_path},
                exc_info=True,
            )
            return 0
        self._update_cursor_from_consumer(consumer)
        return processed

    def run_forever(self) -> None:
        """Start the persistence loop until :meth:`stop` is invoked."""
        if not self.config.enabled:
            logger.info(
                "streaming persistence worker disabled",
                extra={"state_path": self.config.state_path},
            )
            return
        self._stop_event.clear()
        idle_interval = float(self.config.poll_interval_seconds)
        while not self._stop_event.is_set():
            processed = self.poll_once()
            if processed == 0 and idle_interval > 0.0:
                self._stop_event.wait(timeout=idle_interval)

    def stop(self) -> None:
        """Signal the worker loop to exit."""
        self._stop_event.set()

    @property
    def service(self) -> StreamingTrainingPersistenceService:
        """Return the persistence service scoped to this worker."""
        return self._ensure_service()

    def _ensure_service(self) -> StreamingTrainingPersistenceService:
        if self._service is not None:
            return self._service
        state_path = Path(self.config.state_path).expanduser()
        observability = self.observability or self._get_observability_sink()
        self._service = StreamingTrainingPersistenceService.create(
            state_path=state_path,
            observability=observability,
            state_store=self.state_store,
        )
        if self.observability is None:
            self.observability = observability
        cursor = self._service.state_store.get_stream_cursor()
        if isinstance(cursor, str) and cursor.strip():
            self._last_stream_id = cursor.strip()
        return self._service

    def _ensure_consumer(self) -> _PollableConsumer | None:
        if self._consumer is not None:
            return self._consumer
        service = self._ensure_service()
        try:
            if self.consumer_factory is not None:
                consumer = self.consumer_factory(service, self.message_bus_config)
            else:
                consumer = service.create_stream_consumer(self.message_bus_config)
        except Exception:
            logger.warning(
                "streaming persistence consumer initialization failed",
                extra={
                    "backend": self.message_bus_config.backend,
                    "enabled": self.message_bus_config.enabled,
                },
                exc_info=True,
            )
            return None
        self._consumer = consumer
        return consumer

    def _update_cursor_from_consumer(self, consumer: _PollableConsumer) -> None:
        new_cursor = consumer.last_entry_id
        if not new_cursor:
            return
        normalized = new_cursor.strip()
        if not normalized or normalized == self._last_stream_id:
            return
        try:
            self._ensure_service().state_store.update_stream_cursor(normalized)
        except Exception:
            logger.warning(
                "streaming persistence worker failed to persist redis cursor",
                extra={"state_path": self.config.state_path},
                exc_info=True,
            )
            return
        self._last_stream_id = normalized

    def _get_observability_sink(self) -> ObservabilitySink | None:
        if self.observability is not None:
            return self.observability
        if self._observability_initialized:
            return self._resolved_observability
        self._observability_initialized = True
        try:
            from ml.observability.service import ObservabilityService

            adapter = _ObservabilityServiceAdapter(ObservabilityService())
            self._resolved_observability = adapter
        except Exception:
            logger.debug(
                "streaming persistence worker failed to initialize observability service",
                extra={"state_path": str(self.config.state_path)},
                exc_info=True,
            )
            self._resolved_observability = None
        return self._resolved_observability


__all__ = ["ConsumerFactory", "StreamingTrainingPersistenceWorker"]
