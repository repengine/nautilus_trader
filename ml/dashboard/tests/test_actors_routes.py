"""
Tests for Actor Management API Routes.

Comprehensive test coverage for actor deployment, lifecycle management,
hot reloading, and health monitoring.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from ml.dashboard.app import create_app
from ml.dashboard.config import DashboardConfig
from ml.dashboard.services.actors_service import ActorDeploymentResult
from ml.dashboard.services.actors_service import ActorHealthSnapshot
from ml.dashboard.services.actors_service import ActorHotReloadResult
from ml.dashboard.services.actors_service import ActorLifecycleResult


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def app() -> Flask:
    """Provide Flask test application."""
    from ml.dashboard.config import DashboardToken

    config = DashboardConfig(
        auth_tokens=(DashboardToken(value="test-token-123"),),
        db_connection="postgresql://test:test@localhost:5434/test",
    )
    return create_app(config)


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Provide Flask test client."""
    return app.test_client()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Provide authentication headers."""
    return {"X-ML-DASHBOARD-TOKEN": "test-token-123"}


# ============================================================================
# DEPLOY ENDPOINT TESTS
# ============================================================================


class TestActorsDeployEndpoint:
    """Test /api/actors/deploy endpoint."""

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_deploy_actor_success(
        self,
        mock_service_class: MagicMock,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test successful actor deployment."""
        # Setup mock
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        mock_result = ActorDeploymentResult(
            success=True,
            actor_id="test_actor_1",
            status="DEPLOYED",
            message="Actor deployed successfully",
            error=None,
        )

        # Create async mock for deploy_actor
        async_mock = AsyncMock(return_value=mock_result)
        mock_service.deploy_actor = async_mock

        # Make request
        response = client.post(
            "/api/actors/deploy",
            json={
                "actor_type": "MLSignalActor",
                "config": {"model_id": "test_model"},
                "run_id": "run_123",
            },
            headers=auth_headers,
        )

        # Verify response
        assert response.status_code == 202
        data = response.get_json()
        assert data["success"] is True
        assert data["actor_id"] == "test_actor_1"
        assert data["status"] == "DEPLOYED"
        assert data["message"] == "Actor deployed successfully"
        assert data["error"] is None

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_deploy_actor_failure(
        self,
        mock_service_class: MagicMock,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test failed actor deployment."""
        # Setup mock
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        mock_result = ActorDeploymentResult(
            success=False,
            actor_id="",
            status="FAILED",
            message=None,
            error="Model test_model not found",
        )

        async_mock = AsyncMock(return_value=mock_result)
        mock_service.deploy_actor = async_mock

        # Make request
        response = client.post(
            "/api/actors/deploy",
            json={
                "actor_type": "MLSignalActor",
                "config": {"model_id": "test_model"},
            },
            headers=auth_headers,
        )

        # Verify response
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert data["status"] == "FAILED"
        assert "Model test_model not found" in data["error"]

    def test_deploy_actor_unauthorized(self, client: FlaskClient) -> None:
        """Test deployment without authentication."""
        response = client.post(
            "/api/actors/deploy",
            json={"actor_type": "MLSignalActor", "config": {"model_id": "test"}},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data["error"] == "unauthorized"

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_deploy_actor_with_defaults(
        self,
        mock_service_class: MagicMock,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test deployment with minimal payload uses defaults."""
        # Setup mock
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        mock_result = ActorDeploymentResult(
            success=True,
            actor_id="default_actor",
            status="DEPLOYED",
            message="Actor deployed",
            error=None,
        )

        async_mock = AsyncMock(return_value=mock_result)
        mock_service.deploy_actor = async_mock

        # Make request with minimal payload
        response = client.post(
            "/api/actors/deploy",
            json={"config": {"model_id": "model_123"}},
            headers=auth_headers,
        )

        # Verify response
        assert response.status_code == 202
        data = response.get_json()
        assert data["success"] is True


# ============================================================================
# HOT RELOAD ENDPOINT TESTS
# ============================================================================


class TestActorsHotReloadEndpoint:
    """Test /api/actors/hot-reload endpoint."""

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_hot_reload_success(
        self,
        mock_service_class: MagicMock,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test successful hot reload."""
        # Setup mock
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        mock_result = ActorHotReloadResult(
            success=True,
            actor_id="test_actor_1",
            new_model_id="model_v2",
            status="RELOADED",
            error=None,
        )

        async_mock = AsyncMock(return_value=mock_result)
        mock_service.hot_reload_model = async_mock

        # Make request
        response = client.post(
            "/api/actors/hot-reload",
            json={"actor_id": "test_actor_1", "model_id": "model_v2"},
            headers=auth_headers,
        )

        # Verify response
        assert response.status_code == 202
        data = response.get_json()
        assert data["success"] is True
        assert data["actor_id"] == "test_actor_1"
        assert data["new_model_id"] == "model_v2"
        assert data["status"] == "RELOADED"

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_hot_reload_actor_not_found(
        self,
        mock_service_class: MagicMock,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test hot reload with non-existent actor."""
        # Setup mock
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        mock_result = ActorHotReloadResult(
            success=False,
            actor_id="missing_actor",
            new_model_id="model_v2",
            status="NOT_FOUND",
            error="Actor missing_actor not found",
        )

        async_mock = AsyncMock(return_value=mock_result)
        mock_service.hot_reload_model = async_mock

        # Make request
        response = client.post(
            "/api/actors/hot-reload",
            json={"actor_id": "missing_actor", "model_id": "model_v2"},
            headers=auth_headers,
        )

        # Verify response
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert data["status"] == "NOT_FOUND"

    def test_hot_reload_missing_parameters(
        self, client: FlaskClient, auth_headers: dict[str, str]
    ) -> None:
        """Test hot reload with missing required parameters."""
        # Missing model_id
        response = client.post(
            "/api/actors/hot-reload",
            json={"actor_id": "test_actor"},
            headers=auth_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "required" in data["error"]

        # Missing actor_id
        response = client.post(
            "/api/actors/hot-reload",
            json={"model_id": "model_123"},
            headers=auth_headers,
        )
        assert response.status_code == 400

        # Missing both
        response = client.post(
            "/api/actors/hot-reload", json={}, headers=auth_headers
        )
        assert response.status_code == 400

    def test_hot_reload_unauthorized(self, client: FlaskClient) -> None:
        """Test hot reload without authentication."""
        response = client.post(
            "/api/actors/hot-reload",
            json={"actor_id": "test", "model_id": "model"},
        )
        assert response.status_code == 401


# ============================================================================
# PAUSE ENDPOINT TESTS
# ============================================================================


class TestActorsPauseEndpoint:
    """Test /api/actors/pause endpoint."""

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_pause_actor_success(
        self,
        mock_service_class: MagicMock,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test successful actor pause."""
        # Setup mock
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        mock_result = ActorLifecycleResult(
            success=True,
            actor_id="test_actor_1",
            status="PAUSED",
            message="Actor paused",
            error=None,
        )

        async_mock = AsyncMock(return_value=mock_result)
        mock_service.pause_actor = async_mock

        # Make request
        response = client.post(
            "/api/actors/pause",
            json={"actor_id": "test_actor_1", "reason": "Maintenance"},
            headers=auth_headers,
        )

        # Verify response
        assert response.status_code == 202
        data = response.get_json()
        assert data["success"] is True
        assert data["actor_id"] == "test_actor_1"
        assert data["status"] == "PAUSED"

    def test_pause_actor_missing_id(
        self, client: FlaskClient, auth_headers: dict[str, str]
    ) -> None:
        """Test pause without actor_id."""
        response = client.post(
            "/api/actors/pause", json={}, headers=auth_headers
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "required" in data["error"]


# ============================================================================
# RESUME ENDPOINT TESTS
# ============================================================================


class TestActorsResumeEndpoint:
    """Test /api/actors/resume endpoint."""

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_resume_actor_success(
        self,
        mock_service_class: MagicMock,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test successful actor resume."""
        # Setup mock
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        mock_result = ActorLifecycleResult(
            success=True,
            actor_id="test_actor_1",
            status="RUNNING",
            message="Actor resumed",
            error=None,
        )

        async_mock = AsyncMock(return_value=mock_result)
        mock_service.resume_actor = async_mock

        # Make request
        response = client.post(
            "/api/actors/resume",
            json={"actor_id": "test_actor_1"},
            headers=auth_headers,
        )

        # Verify response
        assert response.status_code == 202
        data = response.get_json()
        assert data["success"] is True
        assert data["status"] == "RUNNING"

    def test_resume_actor_missing_id(
        self, client: FlaskClient, auth_headers: dict[str, str]
    ) -> None:
        """Test resume without actor_id."""
        response = client.post(
            "/api/actors/resume", json={}, headers=auth_headers
        )
        assert response.status_code == 400


# ============================================================================
# STOP ENDPOINT TESTS
# ============================================================================


class TestActorsStopEndpoint:
    """Test /api/actors/stop endpoint."""

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_stop_actor_success(
        self,
        mock_service_class: MagicMock,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test successful actor stop."""
        # Setup mock
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        mock_result = ActorLifecycleResult(
            success=True,
            actor_id="test_actor_1",
            status="STOPPED",
            message="Actor stopped",
            error=None,
        )

        async_mock = AsyncMock(return_value=mock_result)
        mock_service.stop_actor = async_mock

        # Make request
        response = client.post(
            "/api/actors/stop",
            json={"actor_id": "test_actor_1", "force": False},
            headers=auth_headers,
        )

        # Verify response
        assert response.status_code == 202
        data = response.get_json()
        assert data["success"] is True
        assert data["status"] == "STOPPED"

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_stop_actor_force(
        self,
        mock_service_class: MagicMock,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test force stop of actor."""
        # Setup mock
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        mock_result = ActorLifecycleResult(
            success=True,
            actor_id="test_actor_1",
            status="STOPPED",
            message="Actor force stopped",
            error=None,
        )

        async_mock = AsyncMock(return_value=mock_result)
        mock_service.stop_actor = async_mock

        # Make request with force=True
        response = client.post(
            "/api/actors/stop",
            json={"actor_id": "test_actor_1", "force": True, "reason": "Emergency"},
            headers=auth_headers,
        )

        # Verify response
        assert response.status_code == 202
        data = response.get_json()
        assert data["success"] is True

    def test_stop_actor_missing_id(
        self, client: FlaskClient, auth_headers: dict[str, str]
    ) -> None:
        """Test stop without actor_id."""
        response = client.post("/api/actors/stop", json={}, headers=auth_headers)
        assert response.status_code == 400


# ============================================================================
# HEALTH ENDPOINT TESTS
# ============================================================================


class TestActorsHealthEndpoint:
    """Test /api/actors/health endpoint."""

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_get_actors_health_success(
        self, mock_service_class: MagicMock, client: FlaskClient
    ) -> None:
        """Test successful health check."""
        # Setup mock
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        mock_snapshot = ActorHealthSnapshot(
            total_actors=3,
            healthy_actors=2,
            unhealthy_actors=1,
            paused_actors=1,
            actors={
                "actor_1": {"healthy": True, "status": "RUNNING"},
                "actor_2": {"healthy": True, "status": "RUNNING"},
                "actor_3": {"healthy": False, "status": "PAUSED"},
            },
        )

        async_mock = AsyncMock(return_value=mock_snapshot)
        mock_service.get_actor_health = async_mock

        # Make request (no auth required for GET)
        response = client.get("/api/actors/health")

        # Verify response
        assert response.status_code == 200
        data = response.get_json()
        assert data["total_actors"] == 3
        assert data["healthy_actors"] == 2
        assert data["unhealthy_actors"] == 1
        assert data["paused_actors"] == 1
        assert len(data["actors"]) == 3
        assert "actor_1" in data["actors"]

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_get_actors_health_empty(
        self, mock_service_class: MagicMock, client: FlaskClient
    ) -> None:
        """Test health check with no actors."""
        # Setup mock
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        mock_snapshot = ActorHealthSnapshot(
            total_actors=0,
            healthy_actors=0,
            unhealthy_actors=0,
            paused_actors=0,
            actors={},
        )

        async_mock = AsyncMock(return_value=mock_snapshot)
        mock_service.get_actor_health = async_mock

        # Make request
        response = client.get("/api/actors/health")

        # Verify response
        assert response.status_code == 200
        data = response.get_json()
        assert data["total_actors"] == 0
        assert data["actors"] == {}


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestActorsIntegration:
    """Integration tests for actor lifecycle."""

    @patch("ml.dashboard.services.actors_service.ActorIntegrationService")
    def test_full_actor_lifecycle(
        self,
        mock_service_class: MagicMock,
        client: FlaskClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test complete actor lifecycle: deploy -> pause -> resume -> stop."""
        # Setup mock service
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # 1. Deploy
        deploy_result = ActorDeploymentResult(
            success=True,
            actor_id="lifecycle_actor",
            status="DEPLOYED",
            message="Deployed",
            error=None,
        )
        mock_service.deploy_actor = AsyncMock(return_value=deploy_result)

        response = client.post(
            "/api/actors/deploy",
            json={"config": {"model_id": "test"}},
            headers=auth_headers,
        )
        assert response.status_code == 202

        # 2. Pause
        pause_result = ActorLifecycleResult(
            success=True,
            actor_id="lifecycle_actor",
            status="PAUSED",
            message="Paused",
            error=None,
        )
        mock_service.pause_actor = AsyncMock(return_value=pause_result)

        response = client.post(
            "/api/actors/pause",
            json={"actor_id": "lifecycle_actor"},
            headers=auth_headers,
        )
        assert response.status_code == 202

        # 3. Resume
        resume_result = ActorLifecycleResult(
            success=True,
            actor_id="lifecycle_actor",
            status="RUNNING",
            message="Resumed",
            error=None,
        )
        mock_service.resume_actor = AsyncMock(return_value=resume_result)

        response = client.post(
            "/api/actors/resume",
            json={"actor_id": "lifecycle_actor"},
            headers=auth_headers,
        )
        assert response.status_code == 202

        # 4. Stop
        stop_result = ActorLifecycleResult(
            success=True,
            actor_id="lifecycle_actor",
            status="STOPPED",
            message="Stopped",
            error=None,
        )
        mock_service.stop_actor = AsyncMock(return_value=stop_result)

        response = client.post(
            "/api/actors/stop",
            json={"actor_id": "lifecycle_actor"},
            headers=auth_headers,
        )
        assert response.status_code == 202


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
