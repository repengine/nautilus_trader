"""
Unit tests for ServiceControllerComponent.

Tests the service control component that delegates to ServiceControllerProtocol
for start/stop/restart actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ml.dashboard.common.service_controller import ServiceControllerComponent
from ml.dashboard.controllers import NoopServiceController, ServiceControllerProtocol
from ml.dashboard.exceptions import ServiceControlUnsupportedError


# -------------------------------------------------------------------------
# Mock Controllers
# -------------------------------------------------------------------------


@dataclass
class MockSuccessController:
    """Mock controller that always succeeds."""

    start_called: bool = False
    stop_called: bool = False
    restart_called: bool = False
    last_service: str = ""

    def start(self, name: str) -> bool:
        self.start_called = True
        self.last_service = name
        return True

    def stop(self, name: str) -> bool:
        self.stop_called = True
        self.last_service = name
        return True

    def restart(self, name: str) -> bool:
        self.restart_called = True
        self.last_service = name
        return True


@dataclass
class MockFailureController:
    """Mock controller that always fails."""

    def start(self, name: str) -> bool:
        return False

    def stop(self, name: str) -> bool:
        return False

    def restart(self, name: str) -> bool:
        return False


@dataclass
class MockExceptionController:
    """Mock controller that raises exceptions."""

    def start(self, name: str) -> bool:
        raise RuntimeError("start failed")

    def stop(self, name: str) -> bool:
        raise RuntimeError("stop failed")

    def restart(self, name: str) -> bool:
        raise RuntimeError("restart failed")


@dataclass
class MockUnsupportedController:
    """Mock controller that lacks protocol methods."""

    def unsupported_method(self) -> None:
        pass


# -------------------------------------------------------------------------
# Tests: control_service - start action
# -------------------------------------------------------------------------


def test_control_service_start_success() -> None:
    """Test successful start action."""
    mock_controller = MockSuccessController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_pipeline", action="start")

    assert result["ok"] is True
    assert result["action"] == "start"
    assert result["service"] == "ml_pipeline"
    assert "error" not in result
    assert mock_controller.start_called is True
    assert mock_controller.last_service == "ml_pipeline"


def test_control_service_start_failure() -> None:
    """Test start action that returns False."""
    mock_controller = MockFailureController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_signal_actor", action="start")

    assert result["ok"] is False
    assert result["action"] == "start"
    assert result["service"] == "ml_signal_actor"


# -------------------------------------------------------------------------
# Tests: control_service - stop action
# -------------------------------------------------------------------------


def test_control_service_stop_success() -> None:
    """Test successful stop action."""
    mock_controller = MockSuccessController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_strategy", action="stop")

    assert result["ok"] is True
    assert result["action"] == "stop"
    assert result["service"] == "ml_strategy"
    assert "error" not in result
    assert mock_controller.stop_called is True
    assert mock_controller.last_service == "ml_strategy"


def test_control_service_stop_failure() -> None:
    """Test stop action that returns False."""
    mock_controller = MockFailureController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_pipeline", action="stop")

    assert result["ok"] is False
    assert result["action"] == "stop"
    assert result["service"] == "ml_pipeline"


# -------------------------------------------------------------------------
# Tests: control_service - restart action
# -------------------------------------------------------------------------


def test_control_service_restart_success() -> None:
    """Test successful restart action."""
    mock_controller = MockSuccessController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_signal_actor", action="restart")

    assert result["ok"] is True
    assert result["action"] == "restart"
    assert result["service"] == "ml_signal_actor"
    assert "error" not in result
    assert mock_controller.restart_called is True
    assert mock_controller.last_service == "ml_signal_actor"


def test_control_service_restart_failure() -> None:
    """Test restart action that returns False."""
    mock_controller = MockFailureController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_strategy", action="restart")

    assert result["ok"] is False
    assert result["action"] == "restart"
    assert result["service"] == "ml_strategy"


# -------------------------------------------------------------------------
# Tests: control_service - invalid action
# -------------------------------------------------------------------------


def test_control_service_invalid_action() -> None:
    """Test that invalid action raises ValueError and returns error."""
    mock_controller = MockSuccessController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_pipeline", action="invalid_action")

    assert result["ok"] is False
    assert result["action"] == "invalid_action"
    assert result["service"] == "ml_pipeline"
    assert "error" in result
    assert "unknown action" in result["error"]


def test_control_service_empty_action() -> None:
    """Test that empty action is treated as invalid."""
    mock_controller = MockSuccessController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_pipeline", action="")

    assert result["ok"] is False
    assert result["action"] == ""
    assert "error" in result


# -------------------------------------------------------------------------
# Tests: control_service - exception handling
# -------------------------------------------------------------------------


def test_control_service_exception_during_start() -> None:
    """Test that exceptions during start are caught and returned as error."""
    mock_controller = MockExceptionController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_pipeline", action="start")

    assert result["ok"] is False
    assert result["action"] == "start"
    assert result["service"] == "ml_pipeline"
    assert "error" in result
    assert "start failed" in result["error"]


def test_control_service_exception_during_stop() -> None:
    """Test that exceptions during stop are caught and returned as error."""
    mock_controller = MockExceptionController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_strategy", action="stop")

    assert result["ok"] is False
    assert result["action"] == "stop"
    assert result["service"] == "ml_strategy"
    assert "error" in result
    assert "stop failed" in result["error"]


def test_control_service_exception_during_restart() -> None:
    """Test that exceptions during restart are caught and returned as error."""
    mock_controller = MockExceptionController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_signal_actor", action="restart")

    assert result["ok"] is False
    assert result["action"] == "restart"
    assert result["service"] == "ml_signal_actor"
    assert "error" in result
    assert "restart failed" in result["error"]


# -------------------------------------------------------------------------
# Tests: control_service - unsupported controller
# -------------------------------------------------------------------------


def test_control_service_unsupported_controller() -> None:
    """Test behavior when controller doesn't support the action."""
    mock_controller = MockUnsupportedController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    result = component.control_service(name="ml_pipeline", action="start")

    assert result["ok"] is False
    assert result["action"] == "start"
    assert result["service"] == "ml_pipeline"
    assert result["error"] == "unsupported"


# -------------------------------------------------------------------------
# Tests: control_service - NoopServiceController integration
# -------------------------------------------------------------------------


def test_control_service_with_noop_controller_start() -> None:
    """Test that NoopServiceController returns False for start."""
    controller = NoopServiceController()
    component = ServiceControllerComponent(controller=controller)

    result = component.control_service(name="ml_pipeline", action="start")

    assert result["ok"] is False
    assert result["action"] == "start"
    assert result["service"] == "ml_pipeline"


def test_control_service_with_noop_controller_stop() -> None:
    """Test that NoopServiceController returns False for stop."""
    controller = NoopServiceController()
    component = ServiceControllerComponent(controller=controller)

    result = component.control_service(name="ml_strategy", action="stop")

    assert result["ok"] is False
    assert result["action"] == "stop"
    assert result["service"] == "ml_strategy"


def test_control_service_with_noop_controller_restart() -> None:
    """Test that NoopServiceController returns False for restart."""
    controller = NoopServiceController()
    component = ServiceControllerComponent(controller=controller)

    result = component.control_service(name="ml_signal_actor", action="restart")

    assert result["ok"] is False
    assert result["action"] == "restart"
    assert result["service"] == "ml_signal_actor"


# -------------------------------------------------------------------------
# Tests: control_service - service name variations
# -------------------------------------------------------------------------


def test_control_service_with_different_service_names() -> None:
    """Test control_service with various valid service names."""
    mock_controller = MockSuccessController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    services = ["ml_signal_actor", "ml_strategy", "ml_pipeline", "prometheus", "grafana"]
    for service in services:
        result = component.control_service(name=service, action="start")
        assert result["ok"] is True
        assert result["service"] == service
        assert mock_controller.last_service == service


# -------------------------------------------------------------------------
# Tests: Protocol conformance
# -------------------------------------------------------------------------


def test_service_controller_component_conforms_to_protocol() -> None:
    """Test that ServiceControllerComponent conforms to its protocol."""
    from ml.dashboard.common.service_controller import (
        ServiceControllerComponentProtocol,
    )

    mock_controller = MockSuccessController()
    component = ServiceControllerComponent(controller=mock_controller)  # type: ignore

    # Runtime protocol check
    assert isinstance(component, ServiceControllerComponentProtocol)
    assert hasattr(component, "control_service")
    assert callable(component.control_service)
