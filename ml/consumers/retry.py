"""
Retry/DLQ consumer wrapper with synchronous bounded retries for tests/examples.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field

from ml.common.message_bus import MessagePublisherProtocol
from ml.consumers.protocols import Envelope


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 3


HandlerFunc = Callable[[str, Envelope], None]


@dataclass(slots=True)
class RetriableConsumer:
    """
    Wrap a handler with bounded synchronous retries and DLQ publishing.

    For simplicity and deterministic testing, failures are retried immediately up
    to ``policy.max_attempts``. On final failure, the envelope is published to the
    DLQ with topic ``dlq.{stage}``.

    """

    handler: HandlerFunc
    dlq: MessagePublisherProtocol
    policy: RetryPolicy = field(default_factory=RetryPolicy)

    _attempts: dict[str, int] = field(default_factory=dict)

    def handle(self, topic: str, envelope: Envelope) -> None:
        eid = envelope["id"]
        attempts = self._attempts.get(eid, 0)
        while attempts < self.policy.max_attempts:
            try:
                self.handler(topic, envelope)
                # Success; reset
                self._attempts.pop(eid, None)
                return
            except Exception:
                attempts += 1
                self._attempts[eid] = attempts
        # Give up to DLQ
        dlq_topic = f"dlq.{envelope['stage']}"
        self.dlq.publish(
            dlq_topic,
            {
                "id": envelope["id"],
                "parent_id": envelope["parent_id"],
                "instrument_id": envelope["instrument_id"],
                "ts_event": envelope["ts_event"],
                "stage": envelope["stage"],
                "correlation_id": envelope["correlation_id"],
                "payload": envelope["payload"],
                "attempts": attempts,
            },
        )


__all__ = ["RetriableConsumer", "RetryPolicy"]
