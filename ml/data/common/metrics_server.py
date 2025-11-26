"""
Metrics server component extracted from DataScheduler.

This component handles Prometheus metrics server management:
- Starting the HTTP server for metrics exposure
- Graceful error handling for import/startup failures

Extracted from legacy DataScheduler (lines 472-500):
- _start_metrics_server() method

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from ml.monitoring.server import MetricsServer


logger = logging.getLogger(__name__)


class MetricsServerProtocol(Protocol):
    """
    Protocol for metrics server operations.

    This protocol defines the contract for metrics server components,
    enabling duck typing for testing and alternative implementations.

    Methods
    -------
    start_metrics_server
        Start the Prometheus HTTP server on the specified port.

    """

    def start_metrics_server(self, port: int) -> Any | None:
        """
        Start the HTTP server for Prometheus metrics.

        Args:
            port: Port number for the metrics server.

        Returns:
            MetricsServer instance if successful, None on failure.

        """
        ...


class MetricsServerComponent:
    """
    Component for Prometheus metrics server management.

    This component extracts metrics server responsibilities from DataScheduler,
    providing focused methods for:
    - Starting the Prometheus HTTP server
    - Graceful error handling for import failures
    - Graceful error handling for startup failures

    All methods are designed to handle errors gracefully and log appropriate
    warnings without raising exceptions that would prevent scheduler operation.

    Example:
        >>> component = MetricsServerComponent()
        >>> server = component.start_metrics_server(port=8080)
        >>> if server is not None:
        ...     print(f"Metrics available at http://localhost:8080/metrics")

    """

    def start_metrics_server(self, port: int) -> MetricsServer | None:
        """
        Start the HTTP server for Prometheus metrics.

        Creates a MetricsServer instance with the specified port and starts
        the HTTP server for exposing Prometheus metrics. The server provides:
        - /metrics endpoint for Prometheus scraping
        - /health endpoint for health checks

        This method handles all exceptions gracefully:
        - Import errors (monitoring module unavailable)
        - Startup errors (port already in use, network issues)

        Args:
            port: Port number for the metrics server. Must be a valid port
                number (typically 1024-65535 for non-privileged ports).

        Returns:
            MetricsServer instance if startup is successful, None on any
            failure. The server is already started when returned.

        Example:
            >>> component = MetricsServerComponent()
            >>> server = component.start_metrics_server(port=9090)
            >>> if server is not None:
            ...     print(f"Server running on port {server.get_port()}")
            ...     # Later: server.stop()

        """
        try:
            from ml.monitoring._config import MonitoringConfig
            from ml.monitoring.server import MetricsServer

            # Create monitoring config with specified port
            monitoring_config = MonitoringConfig(
                enabled=True,
                metrics_port=port,
            )

            metrics_server = MetricsServer(config=monitoring_config)
            metrics_server.start()
            logger.info(f"Started metrics server on port {port}")
            return metrics_server

        except Exception:
            logger.warning(
                "Failed to start metrics server",
                exc_info=True,
            )
            return None

    def stop_metrics_server(self, server: MetricsServer | None) -> None:
        """
        Stop a running metrics server gracefully.

        This method safely stops a MetricsServer instance, handling
        None values and exceptions gracefully.

        Args:
            server: MetricsServer instance to stop, or None.

        Example:
            >>> component = MetricsServerComponent()
            >>> server = component.start_metrics_server(port=8080)
            >>> # ... use server ...
            >>> component.stop_metrics_server(server)

        """
        if server is None:
            return

        try:
            server.stop()
            logger.info("Stopped metrics server")
        except Exception:
            logger.warning(
                "Error stopping metrics server",
                exc_info=True,
            )


__all__ = [
    "MetricsServerComponent",
    "MetricsServerProtocol",
]
