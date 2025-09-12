"""
Consumer protocols and event envelope type for cross-domain processing.

This module defines a minimal, strictly-typed envelope carried on the message bus
between pipeline stages and consumer protocols used to implement aggregators, lineage
writers, and retry/DLQ handlers.

"""

from __future__ import annotations

from typing import Protocol, TypeAlias, TypedDict

from ml.config.events import Stage


StageLike: TypeAlias = Stage | str


class Envelope(TypedDict):
    """
    Canonical event envelope passed between stages.

    Fields
    ------
    id:
        Unique event identifier (UUID string recommended).
    parent_id:
        Optional parent event identifier for lineage (None for roots).
    instrument_id:
        Normalized instrument identifier for routing and lineage.
    ts_event:
        Event timestamp in nanoseconds since epoch.
    stage:
        Pipeline stage name (e.g., "FEATURE_COMPUTED").
    correlation_id:
        Correlation identifier tying a chain of events together.
    payload:
        Opaque payload content for downstream processing.

    """

    id: str
    parent_id: str | None
    instrument_id: str
    ts_event: int
    stage: StageLike
    correlation_id: str
    payload: dict[str, object]


class ConsumerProtocol(Protocol):
    """
    Protocol for message consumers.
    """

    def handle(self, topic: str, envelope: Envelope) -> None:
        """
        Process a message delivered on a topic.
        """
        ...


__all__ = ["ConsumerProtocol", "Envelope", "StageLike"]
