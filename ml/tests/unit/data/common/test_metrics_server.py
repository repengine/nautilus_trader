"""
Unit tests for MetricsServerComponent.

This module contains 6 tests for the metrics server component:
- test_start_metrics_server_success
- test_start_metrics_server_custom_port
- test_start_metrics_server_failure_graceful
- test_start_metrics_server_prometheus_unavailable
- test_stop_metrics_server
- test_metrics_server_returns_none_on_import_error

"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ml.data.common.metrics_server import (
    MetricsServerComponent,
    MetricsServerProtocol,
)


if TYPE_CHECKING:
    pass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def component() -> MetricsServerComponent:
    """
    Fixture providing a fresh MetricsServerComponent instance.
    """
    return MetricsServerComponent()


@pytest.fixture
def mock_metrics_server() -> MagicMock:
    """
    Fixture providing a mock MetricsServer instance.
    """
    mock_server = MagicMock()
    mock_server.get_port.return_value = 8080
    mock_server.is_running.return_value = True
    return mock_server


# ============================================================================
# Protocol Conformance Test
# ============================================================================


class TestProtocolConformance:
    """
    Tests to verify MetricsServerComponent conforms to MetricsServerProtocol.
    """

    def test_component_conforms_to_protocol(
        self,
        component: MetricsServerComponent,
    ) -> None:
        """
        Verify MetricsServerComponent implements MetricsServerProtocol.
        """
        # Protocol conformance is checked by structural typing
        # The component should have the required method signature
        assert hasattr(component, "start_metrics_server")
        assert callable(component.start_metrics_server)

        # Verify the method accepts port parameter
        import inspect
        sig = inspect.signature(component.start_metrics_server)
        params = list(sig.parameters.keys())
        assert "port" in params


# ============================================================================
# Happy Path Tests
# ============================================================================


class TestHappyPath:
    """
    Happy path tests for MetricsServerComponent.
    """

    def test_start_metrics_server_success(
        self,
        component: MetricsServerComponent,
        mock_metrics_server: MagicMock,
    ) -> None:
        """
        Test successful metrics server startup.

        Verifies that:
        - MonitoringConfig is created with correct parameters
        - MetricsServer is instantiated and started
        - The server instance is returned
        """
        with patch(
            "ml.monitoring.server.MetricsServer"
        ) as mock_server_class, patch(
            "ml.monitoring._config.MonitoringConfig"
        ) as mock_config_class:
            # Setup mocks
            mock_server_class.return_value = mock_metrics_server
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            # Start server
            result = component.start_metrics_server(port=8080)

            # Verify MonitoringConfig was created correctly
            mock_config_class.assert_called_once_with(
                enabled=True,
                metrics_port=8080,
            )

            # Verify MetricsServer was instantiated and started
            mock_server_class.assert_called_once_with(config=mock_config)
            mock_metrics_server.start.assert_called_once()

            # Verify return value
            assert result is mock_metrics_server

    def test_start_metrics_server_custom_port(
        self,
        component: MetricsServerComponent,
        mock_metrics_server: MagicMock,
    ) -> None:
        """
        Test metrics server startup with custom port.

        Verifies that custom port values are passed through correctly.
        """
        custom_port = 9090

        with patch(
            "ml.monitoring.server.MetricsServer"
        ) as mock_server_class, patch(
            "ml.monitoring._config.MonitoringConfig"
        ) as mock_config_class:
            # Setup mocks
            mock_server_class.return_value = mock_metrics_server
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            # Start server with custom port
            result = component.start_metrics_server(port=custom_port)

            # Verify custom port was used
            mock_config_class.assert_called_once_with(
                enabled=True,
                metrics_port=custom_port,
            )

            assert result is mock_metrics_server


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """
    Error handling tests for MetricsServerComponent.
    """

    def test_start_metrics_server_failure_graceful(
        self,
        component: MetricsServerComponent,
    ) -> None:
        """
        Test graceful handling of server startup failure.

        Verifies that:
        - Server startup exceptions are caught
        - None is returned instead of raising
        - Error is logged with exc_info=True
        """
        with patch(
            "ml.monitoring.server.MetricsServer"
        ) as mock_server_class, patch(
            "ml.monitoring._config.MonitoringConfig"
        ), patch(
            "ml.data.common.metrics_server.logger"
        ) as mock_logger:
            # Make server.start() raise an exception
            mock_server = MagicMock()
            mock_server.start.side_effect = RuntimeError("Port already in use")
            mock_server_class.return_value = mock_server

            # Start server - should not raise
            result = component.start_metrics_server(port=8080)

            # Verify graceful failure
            assert result is None

            # Verify warning was logged with exc_info
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "Failed to start metrics server" in call_args[0][0]
            assert call_args[1].get("exc_info") is True

    def test_start_metrics_server_prometheus_unavailable(
        self,
        component: MetricsServerComponent,
    ) -> None:
        """
        Test graceful handling when Prometheus client is unavailable.

        Verifies that the component handles the case where the monitoring
        module cannot start due to missing prometheus_client.
        """
        with patch(
            "ml.monitoring.server.MetricsServer"
        ) as mock_server_class, patch(
            "ml.monitoring._config.MonitoringConfig"
        ), patch(
            "ml.data.common.metrics_server.logger"
        ) as mock_logger:
            # Simulate Prometheus unavailable scenario
            mock_server = MagicMock()
            mock_server.start.side_effect = ImportError(
                "prometheus_client not installed"
            )
            mock_server_class.return_value = mock_server

            # Start server - should not raise
            result = component.start_metrics_server(port=8080)

            # Verify graceful failure
            assert result is None

            # Verify warning was logged
            mock_logger.warning.assert_called_once()

    def test_metrics_server_returns_none_on_import_error(
        self,
        component: MetricsServerComponent,
    ) -> None:
        """
        Test that import errors during startup return None gracefully.

        Verifies that if the monitoring module cannot be imported,
        the component returns None without raising.
        """
        import builtins

        original_import = builtins.__import__

        def mock_import(
            name: str,
            globals: dict | None = None,
            locals: dict | None = None,
            fromlist: tuple = (),
            level: int = 0,
        ):  # type: ignore
            if name.startswith("ml.monitoring"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, globals, locals, fromlist, level)

        with patch.object(
            builtins, "__import__", side_effect=mock_import
        ), patch(
            "ml.data.common.metrics_server.logger"
        ) as mock_logger:
            result = component.start_metrics_server(port=8080)

            # Verify graceful failure
            assert result is None

            # Verify warning was logged
            mock_logger.warning.assert_called_once()


# ============================================================================
# Stop Server Tests
# ============================================================================


class TestStopServer:
    """
    Tests for stopping the metrics server.
    """

    def test_stop_metrics_server(
        self,
        component: MetricsServerComponent,
        mock_metrics_server: MagicMock,
    ) -> None:
        """
        Test that server.stop() is callable via component.

        Verifies that:
        - stop_metrics_server accepts a server instance
        - Calls server.stop() method
        - Handles None gracefully
        """
        # Test stopping a running server
        component.stop_metrics_server(mock_metrics_server)

        # Verify stop was called
        mock_metrics_server.stop.assert_called_once()

    def test_stop_metrics_server_handles_none(
        self,
        component: MetricsServerComponent,
    ) -> None:
        """
        Test that stop_metrics_server handles None gracefully.
        """
        # Should not raise
        component.stop_metrics_server(None)

    def test_stop_metrics_server_handles_exception(
        self,
        component: MetricsServerComponent,
        mock_metrics_server: MagicMock,
    ) -> None:
        """
        Test that stop_metrics_server handles exceptions gracefully.
        """
        # Make stop() raise an exception
        mock_metrics_server.stop.side_effect = RuntimeError("Stop failed")

        with patch(
            "ml.data.common.metrics_server.logger"
        ) as mock_logger:
            # Should not raise
            component.stop_metrics_server(mock_metrics_server)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
