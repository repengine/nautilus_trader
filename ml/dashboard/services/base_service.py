"""Shared types and base classes for dashboard integration services."""

from __future__ import annotations

import asyncio
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:
    from ml.core.integration import MLIntegrationManager

_T = TypeVar("_T")

integration_calls = get_counter(
    "ml_dashboard_integration_calls_total",
    "Total integration layer calls",
    labelnames=["component", "operation", "status"],
)

integration_latency = get_histogram(
    "ml_dashboard_integration_latency_seconds",
    "Integration layer operation latency",
)


@dataclass(slots=True)
class IntegrationContext:
    """Structured metadata attached to integration requests."""

    user_id: str | None = None
    session_id: str | None = None
    correlation_id: str | None = None
    source: str = "dashboard"
    metadata: dict[str, Any] | None = None


class BaseIntegrationService(ABC):
    """Base class that tracks context and metrics for integration services."""

    def __init__(self, integration_manager: MLIntegrationManager | None) -> None:
        self._integration = integration_manager
        self._context: IntegrationContext | None = None

    @abstractmethod
    def get_service_name(self) -> str:
        """Return a stable service name used for metrics labels."""

    def set_context(self, context: IntegrationContext) -> None:
        """Attach request-scoped context metadata."""
        self._context = context

    def set_integration_manager(self, integration_manager: MLIntegrationManager | None) -> None:
        """Update the underlying integration manager reference."""
        self._integration = integration_manager

    def _track_operation(self, *, operation: str, status: str) -> None:
        """Increment Prometheus counters for the operation lifecycle."""
        integration_calls.labels(
            component=self.get_service_name(),
            operation=operation,
            status=status,
        ).inc()

    async def _run_async(self, func: Callable[[], _T]) -> _T:
        """Execute a blocking function via the default executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Return health information for the underlying integration."""


__all__ = [
    "BaseIntegrationService",
    "IntegrationContext",
    "integration_calls",
    "integration_latency",
]
