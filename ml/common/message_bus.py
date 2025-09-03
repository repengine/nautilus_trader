"""
Lightweight message bus publisher protocol and no-op implementation.

This module defines a minimal, typed interface for publishing ML events to an
external message bus. Implementations should be provided by deployment layers;
tests may inject fakes/mocks. By default, a `NoopPublisher` is safe to use and
performs no publishing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MessagePublisherProtocol(Protocol):
    """Protocol for message bus publishers."""

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        """Publish a payload to a topic; returns True on success."""
        ...


class NoopPublisher:
    """No-op publisher implementation (safe default)."""

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        return False


__all__ = ["MessagePublisherProtocol", "NoopPublisher"]

