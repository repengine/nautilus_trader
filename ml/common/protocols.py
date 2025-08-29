"""
Universal ML component protocol and mixin.

This module defines a runtime-checkable protocol which standardizes health reporting,
performance metrics, and configuration validation across ML components (actors, stores,
registries).

Components can adopt the protocol by inheriting from MLComponentMixin or by implementing
the required methods directly.

"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MLComponentProtocol(Protocol):
    """
    Standard interface for ML components.

    Implementations should keep these methods out of the hot path.

    """

    def get_health_status(self) -> dict[str, Any]:
        """
        Return current health information for the component.

        Should be safe to call from monitoring endpoints and during startup checks.
        Avoid expensive work.

        """

    def get_performance_metrics(self) -> dict[str, float]:
        """
        Return lightweight performance metrics for the component.

        This is not a replacement for Prometheus. Intended for quick diagnostics (e.g.,
        averages or counters).

        """

    def validate_configuration(self) -> list[str]:
        """
        Validate component configuration, returning a list of issues.

        Return an empty list when configuration is valid. Avoid raising in normal
        operation; integration layers may escalate based on strictness settings.

        """


class MLComponentMixin:
    """
    Safe default implementations for the universal component protocol.

    - get_health_status: basic ok status with timestamp
    - get_performance_metrics: empty mapping
    - validate_configuration: no issues

    """

    _component_name: str | None = None  # Optional override in subclasses

    def get_health_status(self) -> dict[str, Any]:  # pragma: no cover - trivial
        name = getattr(self, "_component_name", None) or self.__class__.__name__
        return {
            "component": name,
            "status": "ok",
            "timestamp": time.time(),
        }

    def get_performance_metrics(self) -> dict[str, float]:  # pragma: no cover - trivial
        return {}

    def validate_configuration(self) -> list[str]:  # pragma: no cover - trivial
        return []


__all__ = ["MLComponentMixin", "MLComponentProtocol"]
