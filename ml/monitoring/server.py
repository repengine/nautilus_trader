
"""
Prometheus metrics server for ML monitoring.

This module provides a lightweight HTTP server for exposing Prometheus metrics with
graceful shutdown handling and error recovery.

"""

from __future__ import annotations

import logging
import socket
import threading
import time
import types
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from socketserver import ThreadingMixIn
from typing import TYPE_CHECKING, Any, Self

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig


if TYPE_CHECKING:
    pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """
    Threaded HTTP server for concurrent request handling.

    Allows the server to handle multiple requests concurrently while maintaining thread
    safety.

    """

    daemon_threads = True
    allow_reuse_address = True


class MetricsHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for Prometheus metrics endpoint.

    Handles GET requests to the /metrics endpoint and returns Prometheus-formatted
    metrics.

    """

    def do_GET(self) -> None:
        """
        Handle GET requests.
        """
        if self.path == "/metrics":
            self._handle_metrics()
        elif self.path == "/health":
            self._handle_health()
        else:
            self.send_error(404, "Not Found")

    def _handle_metrics(self) -> None:
        """
        Handle metrics endpoint.
        """
        if not HAS_PROMETHEUS:
            self.send_error(503, "Prometheus client not available")
            return

        try:
            from ml._imports import generate_latest

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(generate_latest())
        except Exception as e:
            self.send_error(500, f"Error generating metrics: {e}")

    def _handle_health(self) -> None:
        """
        Handle health check endpoint.
        """
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "healthy", "service": "ml-metrics"}')

    def log_message(self, format: str, *args: Any) -> None:
        """
        Override to use Python logging instead of stderr.
        """
        logging.getLogger("MetricsServer").info(format % args)


class MetricsServer:
    """
    Lightweight HTTP server for exposing Prometheus metrics.

    This server provides a /metrics endpoint for Prometheus to scrape
    and a /health endpoint for health checks.

    Parameters
    ----------
    config : MonitoringConfig
        Configuration for the metrics server.

    """

    def __init__(self, config: MonitoringConfig) -> None:
        """
        Initialize the metrics server.

        Parameters
        ----------
        config : MonitoringConfig
            Configuration for the metrics server.

        """
        self._config = config
        self._server: ThreadedHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._running = False
        self._logger = logging.getLogger(self.__class__.__name__)

        # Check if Prometheus is available when server is needed
        self._prometheus_available = HAS_PROMETHEUS

    def start(self) -> None:
        """
        Start the metrics server.

        Raises
        ------
        RuntimeError
            If the server is already running or fails to start.

        """
        if self._running:
            raise RuntimeError("Metrics server is already running")

        if not self._config.enabled:
            self._logger.info("Metrics server disabled in configuration")
            return

        if not self._prometheus_available:
            self._logger.warning(
                "Prometheus client not available. Install with: pip install 'nautilus-trader[ml]'",
            )
            return

        try:
            # Create server
            self._server = ThreadedHTTPServer(
                ("127.0.0.1", self._config.metrics_port),
                MetricsHandler,
            )
            self._server.timeout = self._config.server_timeout

            # Start server in background thread
            self._server_thread = threading.Thread(
                target=self._run_server,
                name="MetricsServer",
                daemon=True,
            )
            self._running = True
            self._server_thread.start()

            self._logger.info(f"Metrics server started on port {self._config.metrics_port}")

        except OSError as e:
            self._running = False
            if e.errno == 98:  # Address already in use
                raise RuntimeError(
                    f"Port {self._config.metrics_port} is already in use. "
                    f"Choose a different port or stop the conflicting service.",
                ) from e
            else:
                raise RuntimeError(f"Failed to start metrics server: {e}") from e
        except Exception as e:
            self._running = False
            raise RuntimeError(f"Failed to start metrics server: {e}") from e

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the metrics server gracefully.

        Parameters
        ----------
        timeout : float, default 10.0
            Maximum time to wait for graceful shutdown.

        """
        if not self._running:
            return

        self._running = False

        if self._server is not None:
            # Shutdown server
            self._server.shutdown()
            self._server.server_close()

            # Wait for server thread to finish
            if self._server_thread is not None and self._server_thread.is_alive():
                self._server_thread.join(timeout=timeout)
                if self._server_thread.is_alive():
                    self._logger.warning("Server thread did not terminate gracefully")

        self._server = None
        self._server_thread = None
        self._logger.info("Metrics server stopped")

    def _run_server(self) -> None:
        """
        Run the HTTP server (internal method).
        """
        if self._server is None:
            return

        try:
            self._server.serve_forever()
        except Exception as e:
            if self._running:  # Only log if not shutting down
                self._logger.error(f"Metrics server error: {e}")
        finally:
            self._running = False

    def is_running(self) -> bool:
        """
        Check if the server is running.

        Returns
        -------
        bool
            True if the server is running.

        """
        return self._running

    def get_port(self) -> int:
        """
        Get the configured server port.

        Returns
        -------
        int
            The server port number.

        """
        return self._config.metrics_port

    def get_metrics_url(self) -> str:
        """
        Get the full URL for the metrics endpoint.

        Returns
        -------
        str
            The metrics endpoint URL.

        """
        return f"http://localhost:{self._config.metrics_port}/metrics"

    def get_health_url(self) -> str:
        """
        Get the full URL for the health endpoint.

        Returns
        -------
        str
            The health endpoint URL.

        """
        return f"http://localhost:{self._config.metrics_port}/health"

    def wait_for_ready(self, timeout: float = 30.0) -> bool:
        """
        Wait for the server to be ready to accept connections.

        Parameters
        ----------
        timeout : float, default 30.0
            Maximum time to wait for the server to be ready.

        Returns
        -------
        bool
            True if the server is ready, False if timeout occurred.

        """
        if not self._running:
            return False

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_port_open():
                return True
            time.sleep(0.1)

        return False

    def _is_port_open(self) -> bool:
        """
        Check if the server port is accepting connections.
        """
        try:
            with socket.create_connection(("localhost", self._config.metrics_port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            return False

    def __enter__(self) -> Self:
        """
        Context manager entry.
        """
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """
        Context manager exit.
        """
        self.stop()
