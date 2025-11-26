"""
Service controller component for Dashboard service.

Extracted from DashboardService to follow single-responsibility principle.
Delegates service control actions (start/stop/restart) to ServiceControllerProtocol.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ml.common.logging_config import bind_log_context
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.dashboard.exceptions import ServiceControlUnsupportedError


if TYPE_CHECKING:
    from ml.dashboard.controllers import ServiceControllerProtocol


logger = logging.getLogger(__name__)


_REQS_TOTAL = get_counter(
    "ml_dashboard_requests_total",
    "Total dashboard API requests",
    labels=["route", "method", "status"],
)
_LATENCY_SECONDS = get_histogram(
    "ml_dashboard_latency_seconds",
    "Dashboard API latency (seconds)",
    labels=["route"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


@runtime_checkable
class ServiceControllerComponentProtocol(Protocol):
    """Protocol for service control operations."""

    def control_service(self, name: str, action: str) -> dict[str, Any]:
        """Control a service with the specified action."""
        ...


@dataclass
class ServiceControllerComponent:
    """
    Component for controlling services via start/stop/restart actions.

    Extracted from DashboardService to follow single-responsibility principle.
    Delegates to an injected ServiceControllerProtocol (e.g., ComposeServiceController
    or NoopServiceController).

    Attributes:
        controller: The service controller to delegate actions to.
    """

    controller: ServiceControllerProtocol

    def control_service(self, name: str, action: str) -> dict[str, Any]:
        """
        Control a service with the specified action.

        Supported actions:
        - start: Start the service
        - stop: Stop the service
        - restart: Restart the service

        Args:
            name: Service name (e.g., "ml_signal_actor", "ml_strategy")
            action: Action to perform ("start", "stop", "restart")

        Returns:
            Dictionary with control result:
            - ok: Whether the action succeeded
            - action: The action performed
            - service: The service name
            - error: Error message (if applicable)

        Raises:
            ValueError: If action is not recognized.
            ServiceControlUnsupportedError: If service control is unavailable.

        Example:
            >>> controller = ServiceControllerComponent(NoopServiceController())
            >>> result = controller.control_service("ml_pipeline", "restart")
            >>> assert result["action"] == "restart"
            >>> assert result["service"] == "ml_pipeline"
        """
        route = "/api/services/{name}:action"
        start_time = time.perf_counter()
        try:
            # Bind logging context for structured logs
            bind_log_context(component="ml.dashboard", action=action, service=name)

            # Validate action first (before protocol check)
            if action not in ("start", "stop", "restart"):
                raise ValueError(f"unknown action: {action}")

            # Runtime protocol check (defensive)
            if not hasattr(self.controller, action):
                raise ServiceControlUnsupportedError("service control unavailable")

            # Dispatch action
            result = False
            if action == "start":
                result = self.controller.start(name)
            elif action == "stop":
                result = self.controller.stop(name)
            elif action == "restart":
                result = self.controller.restart(name)

            # Record success metrics
            status = "success" if result else "noop"
            _REQS_TOTAL.labels(route=route, method="POST", status=status).inc()

            return {"ok": result, "action": action, "service": name}

        except ServiceControlUnsupportedError:
            _REQS_TOTAL.labels(route=route, method="POST", status="unsupported").inc()
            logger.debug(
                "service control unsupported",
                extra={"service": name, "action": action},
                exc_info=True,
            )
            return {"ok": False, "action": action, "service": name, "error": "unsupported"}

        except ValueError as exc:
            _REQS_TOTAL.labels(route=route, method="POST", status="invalid_action").inc()
            logger.warning(
                "invalid service control action",
                extra={"service": name, "action": action, "error": str(exc)},
                exc_info=True,
            )
            return {"ok": False, "action": action, "service": name, "error": str(exc)}

        except Exception as exc:
            _REQS_TOTAL.labels(route=route, method="POST", status="error").inc()
            logger.error(
                "service control failed",
                extra={"service": name, "action": action},
                exc_info=True,
            )
            return {"ok": False, "action": action, "service": name, "error": str(exc)}

        finally:
            _LATENCY_SECONDS.labels(route=route).observe(time.perf_counter() - start_time)


__all__ = [
    "ServiceControllerComponent",
    "ServiceControllerComponentProtocol",
]
