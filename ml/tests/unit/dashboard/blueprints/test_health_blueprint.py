"""Tests for Health and Services Blueprint.

This module tests the health blueprint routes:
- GET /api/health/system
- GET /api/services
- POST /api/services/<name>:action

Tests:
1. test_health_system_returns_200
2. test_services_list_returns_200
3. test_services_action_start_success
4. test_services_action_stop_success
5. test_services_action_restart_success
6. test_services_action_invalid_action_400
7. test_services_action_unauthorized_401
8. test_services_action_delegates_to_service
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Blueprint, Flask
from flask.testing import FlaskClient

from ml.dashboard.blueprints.health import register_health_routes


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_service() -> MagicMock:
    """Provide a mock DashboardService."""
    svc = MagicMock()
    svc.get_system_health.return_value = {
        "healthy": True,
        "components": {
            "database": {"status": "ok", "latency_ms": 5},
            "redis": {"status": "ok", "latency_ms": 2},
            "model_store": {"status": "ok"},
        },
        "timestamp": "2025-01-15T12:00:00Z",
    }
    svc.list_services.return_value = [
        {"name": "ml_signal_actor", "ports": {"http": 8080}},
        {"name": "ml_pipeline", "ports": {"http": 8081}},
    ]
    svc.control_service.return_value = {"ok": True, "message": "Service controlled"}
    return svc


@pytest.fixture
def app(mock_service: MagicMock) -> Flask:
    """Provide Flask test application with health blueprint."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    # Track auth state for require_token mock
    app.config["AUTH_ENABLED"] = True

    def require_token() -> bool:
        return app.config.get("AUTH_ENABLED", True)

    # Create a FRESH blueprint for each test to avoid Flask's
    # "already registered" error
    health_bp = Blueprint("health", __name__, url_prefix="/api")

    # Register routes with mock service
    register_health_routes(health_bp, mock_service, require_token)
    app.register_blueprint(health_bp)

    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Provide Flask test client."""
    return app.test_client()


# ============================================================================
# TEST: health_system_returns_200
# ============================================================================


class TestHealthSystemReturns200:
    """Test GET /api/health/system returns 200."""

    def test_health_system_returns_200(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that GET /api/health/system returns HTTP 200 with health data."""
        response = client.get("/api/health/system")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["healthy"] is True
        assert "components" in data
        assert "timestamp" in data
        mock_service.get_system_health.assert_called_once()


# ============================================================================
# TEST: services_list_returns_200
# ============================================================================


class TestServicesListReturns200:
    """Test GET /api/services returns 200."""

    def test_services_list_returns_200(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that GET /api/services returns HTTP 200 with services list."""
        response = client.get("/api/services")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "ml_signal_actor"
        mock_service.list_services.assert_called_once()


# ============================================================================
# TEST: services_action_start_success
# ============================================================================


class TestServicesActionStartSuccess:
    """Test POST /api/services/<name>:action with start action."""

    def test_services_action_start_success(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that POST /api/services/<name>:action with start returns 202."""
        mock_service.control_service.return_value = {
            "ok": True,
            "message": "Service started",
        }

        response = client.post(
            "/api/services/ml_signal_actor:action",
            json={"action": "start"},
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["ok"] is True
        assert data["message"] == "Service started"
        mock_service.control_service.assert_called_once_with("ml_signal_actor", "start")


# ============================================================================
# TEST: services_action_stop_success
# ============================================================================


class TestServicesActionStopSuccess:
    """Test POST /api/services/<name>:action with stop action."""

    def test_services_action_stop_success(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that POST /api/services/<name>:action with stop returns 202."""
        mock_service.control_service.return_value = {
            "ok": True,
            "message": "Service stopped",
        }

        response = client.post(
            "/api/services/ml_pipeline:action",
            json={"action": "stop"},
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["ok"] is True
        assert data["message"] == "Service stopped"
        mock_service.control_service.assert_called_once_with("ml_pipeline", "stop")


# ============================================================================
# TEST: services_action_restart_success
# ============================================================================


class TestServicesActionRestartSuccess:
    """Test POST /api/services/<name>:action with restart action."""

    def test_services_action_restart_success(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that POST /api/services/<name>:action with restart returns 202."""
        mock_service.control_service.return_value = {
            "ok": True,
            "message": "Service restarted",
        }

        response = client.post(
            "/api/services/ml_signal_actor:action",
            json={"action": "restart"},
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["ok"] is True
        assert data["message"] == "Service restarted"
        mock_service.control_service.assert_called_once_with("ml_signal_actor", "restart")


# ============================================================================
# TEST: services_action_invalid_action_400
# ============================================================================


class TestServicesActionInvalidAction400:
    """Test POST /api/services/<name>:action with invalid action."""

    def test_services_action_invalid_action_400(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that POST /api/services/<name>:action with invalid action returns 400."""
        response = client.post(
            "/api/services/ml_signal_actor:action",
            json={"action": "invalid_action"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "invalid_action"
        mock_service.control_service.assert_not_called()

    def test_services_action_empty_action_400(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that POST /api/services/<name>:action with empty action returns 400."""
        response = client.post(
            "/api/services/ml_signal_actor:action",
            json={"action": ""},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "invalid_action"
        mock_service.control_service.assert_not_called()

    def test_services_action_missing_action_400(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that POST /api/services/<name>:action with missing action returns 400."""
        response = client.post(
            "/api/services/ml_signal_actor:action",
            json={},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "invalid_action"
        mock_service.control_service.assert_not_called()


# ============================================================================
# TEST: services_action_unauthorized_401
# ============================================================================


class TestServicesActionUnauthorized401:
    """Test POST /api/services/<name>:action without authentication."""

    def test_services_action_unauthorized_401(
        self,
        app: Flask,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that POST /api/services/<name>:action without auth returns 401."""
        # Disable auth in the app config
        app.config["AUTH_ENABLED"] = False

        response = client.post(
            "/api/services/ml_signal_actor:action",
            json={"action": "start"},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data is not None
        assert data["error"] == "unauthorized"
        mock_service.control_service.assert_not_called()


# ============================================================================
# TEST: services_action_delegates_to_service
# ============================================================================


class TestServicesActionDelegatesToService:
    """Test POST /api/services/<name>:action delegates to DashboardService."""

    def test_services_action_delegates_to_service(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that POST /api/services/<name>:action correctly delegates to service."""
        mock_service.control_service.return_value = {
            "ok": True,
            "service_name": "test_service",
            "action": "start",
            "result": "success",
            "timestamp": "2025-01-15T12:00:00Z",
        }

        response = client.post(
            "/api/services/test_service:action",
            json={"action": "start"},
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["service_name"] == "test_service"
        assert data["action"] == "start"
        assert data["result"] == "success"

        # Verify the service was called with correct arguments
        mock_service.control_service.assert_called_once_with("test_service", "start")

    def test_services_action_returns_200_on_failure(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
    ) -> None:
        """Test that POST /api/services/<name>:action returns 200 when ok=False."""
        mock_service.control_service.return_value = {
            "ok": False,
            "error": "Service not found",
        }

        response = client.post(
            "/api/services/nonexistent_service:action",
            json={"action": "start"},
        )

        # When ok=False, should return 200 (not 202)
        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["ok"] is False
        assert data["error"] == "Service not found"


__all__ = [
    "TestHealthSystemReturns200",
    "TestServicesActionDelegatesToService",
    "TestServicesActionInvalidAction400",
    "TestServicesActionRestartSuccess",
    "TestServicesActionStartSuccess",
    "TestServicesActionStopSuccess",
    "TestServicesActionUnauthorized401",
    "TestServicesListReturns200",
]
