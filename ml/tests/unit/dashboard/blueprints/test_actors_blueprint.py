"""Tests for Actors Blueprint.

This module tests the actors blueprint routes:
- POST /api/actors/deploy
- POST /api/actors/hot-reload
- POST /api/actors/pause
- POST /api/actors/resume
- POST /api/actors/stop
- GET /api/actors/health

Tests:
1. test_actors_deploy_success
2. test_actors_deploy_unauthorized_401
3. test_actors_deploy_invalid_type_400
4. test_actors_hot_reload_success
5. test_actors_hot_reload_missing_params_400
6. test_actors_hot_reload_unauthorized_401
7. test_actors_pause_success
8. test_actors_pause_missing_actor_id_400
9. test_actors_pause_unauthorized_401
10. test_actors_resume_success
11. test_actors_resume_missing_actor_id_400
12. test_actors_resume_unauthorized_401
13. test_actors_stop_success
14. test_actors_stop_force_success
15. test_actors_health_returns_200
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Blueprint, Flask
from flask.testing import FlaskClient

from ml.dashboard.blueprints.actors import register_actors_routes

# Module path for patching services (imported inside route functions)
ACTORS_SERVICE_PATH = "ml.dashboard.services.actors_service"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_service() -> MagicMock:
    """Provide a mock DashboardService."""
    svc = MagicMock()
    svc._pipeline_integration_manager = None
    return svc


@pytest.fixture
def app(mock_service: MagicMock) -> Flask:
    """Provide Flask test application with actors blueprint."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    # Track auth state for require_token mock
    app.config["AUTH_ENABLED"] = True

    def require_token() -> bool:
        return app.config.get("AUTH_ENABLED", True)

    # Create a FRESH blueprint for each test to avoid Flask's
    # "already registered" error
    actors_bp = Blueprint("actors", __name__, url_prefix="/api/actors")

    # Register routes with mock service
    register_actors_routes(actors_bp, mock_service, require_token)
    app.register_blueprint(actors_bp)

    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Provide Flask test client."""
    return app.test_client()


# ============================================================================
# TEST: actors_deploy_success
# ============================================================================


class TestActorsDeploySuccess:
    """Test POST /api/actors/deploy success."""

    def test_actors_deploy_success(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/deploy returns 202 on success."""
        with patch(
            f"{ACTORS_SERVICE_PATH}.ActorIntegrationService"
        ) as mock_svc_class:
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.actor_id = "MLSignalActor_model_123"
            mock_result.status = "DEPLOYED"
            mock_result.message = "Actor deployed successfully"
            mock_result.error = None

            async def mock_deploy(*args: object, **kwargs: object) -> MagicMock:
                return mock_result

            mock_instance.deploy_actor = mock_deploy

            response = client.post(
                "/api/actors/deploy",
                json={
                    "actor_type": "MLSignalActor",
                    "config": {"model_id": "model_123"},
                    "run_id": "run_001",
                },
            )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["actor_id"] == "MLSignalActor_model_123"
        assert data["status"] == "DEPLOYED"


# ============================================================================
# TEST: actors_deploy_unauthorized_401
# ============================================================================


class TestActorsDeployUnauthorized401:
    """Test POST /api/actors/deploy unauthorized."""

    def test_actors_deploy_unauthorized_401(
        self,
        app: Flask,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/deploy without auth returns 401."""
        app.config["AUTH_ENABLED"] = False

        response = client.post(
            "/api/actors/deploy",
            json={"actor_type": "MLSignalActor", "config": {}},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data is not None
        assert data["error"] == "unauthorized"


# ============================================================================
# TEST: actors_deploy_invalid_type_400
# ============================================================================


class TestActorsDeployInvalidType400:
    """Test POST /api/actors/deploy with invalid actor type."""

    def test_actors_deploy_invalid_type_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/deploy returns 400 for invalid type."""
        with patch(
            f"{ACTORS_SERVICE_PATH}.ActorIntegrationService"
        ) as mock_svc_class:
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = False
            mock_result.actor_id = ""
            mock_result.status = "INVALID"
            mock_result.message = None
            mock_result.error = "Invalid actor type: InvalidActor"

            async def mock_deploy(*args: object, **kwargs: object) -> MagicMock:
                return mock_result

            mock_instance.deploy_actor = mock_deploy

            response = client.post(
                "/api/actors/deploy",
                json={"actor_type": "InvalidActor", "config": {}},
            )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["success"] is False
        assert data["status"] == "INVALID"


# ============================================================================
# TEST: actors_hot_reload_success
# ============================================================================


class TestActorsHotReloadSuccess:
    """Test POST /api/actors/hot-reload success."""

    def test_actors_hot_reload_success(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/hot-reload returns 202 on success."""
        with patch(
            f"{ACTORS_SERVICE_PATH}.ActorIntegrationService"
        ) as mock_svc_class:
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.actor_id = "actor_001"
            mock_result.new_model_id = "model_v2"
            mock_result.status = "RELOADED"
            mock_result.error = None

            async def mock_hot_reload(*args: object, **kwargs: object) -> MagicMock:
                return mock_result

            mock_instance.hot_reload_model = mock_hot_reload

            response = client.post(
                "/api/actors/hot-reload",
                json={"actor_id": "actor_001", "model_id": "model_v2"},
            )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["new_model_id"] == "model_v2"
        assert data["status"] == "RELOADED"


# ============================================================================
# TEST: actors_hot_reload_missing_params_400
# ============================================================================


class TestActorsHotReloadMissingParams400:
    """Test POST /api/actors/hot-reload with missing params."""

    def test_actors_hot_reload_missing_actor_id_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/hot-reload without actor_id returns 400."""
        response = client.post(
            "/api/actors/hot-reload",
            json={"model_id": "model_v2"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "actor_id and model_id are required"

    def test_actors_hot_reload_missing_model_id_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/hot-reload without model_id returns 400."""
        response = client.post(
            "/api/actors/hot-reload",
            json={"actor_id": "actor_001"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "actor_id and model_id are required"


# ============================================================================
# TEST: actors_hot_reload_unauthorized_401
# ============================================================================


class TestActorsHotReloadUnauthorized401:
    """Test POST /api/actors/hot-reload unauthorized."""

    def test_actors_hot_reload_unauthorized_401(
        self,
        app: Flask,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/hot-reload without auth returns 401."""
        app.config["AUTH_ENABLED"] = False

        response = client.post(
            "/api/actors/hot-reload",
            json={"actor_id": "actor_001", "model_id": "model_v2"},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data is not None
        assert data["error"] == "unauthorized"


# ============================================================================
# TEST: actors_pause_success
# ============================================================================


class TestActorsPauseSuccess:
    """Test POST /api/actors/pause success."""

    def test_actors_pause_success(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/pause returns 202 on success."""
        with patch(
            f"{ACTORS_SERVICE_PATH}.ActorIntegrationService"
        ) as mock_svc_class:
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.actor_id = "actor_001"
            mock_result.status = "PAUSED"
            mock_result.message = "Actor paused"
            mock_result.error = None

            async def mock_pause(*args: object, **kwargs: object) -> MagicMock:
                return mock_result

            mock_instance.pause_actor = mock_pause

            response = client.post(
                "/api/actors/pause",
                json={"actor_id": "actor_001", "reason": "Maintenance"},
            )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["status"] == "PAUSED"


# ============================================================================
# TEST: actors_pause_missing_actor_id_400
# ============================================================================


class TestActorsPauseMissingActorId400:
    """Test POST /api/actors/pause with missing actor_id."""

    def test_actors_pause_missing_actor_id_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/pause without actor_id returns 400."""
        response = client.post(
            "/api/actors/pause",
            json={"reason": "Maintenance"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "actor_id is required"


# ============================================================================
# TEST: actors_pause_unauthorized_401
# ============================================================================


class TestActorsPauseUnauthorized401:
    """Test POST /api/actors/pause unauthorized."""

    def test_actors_pause_unauthorized_401(
        self,
        app: Flask,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/pause without auth returns 401."""
        app.config["AUTH_ENABLED"] = False

        response = client.post(
            "/api/actors/pause",
            json={"actor_id": "actor_001"},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data is not None
        assert data["error"] == "unauthorized"


# ============================================================================
# TEST: actors_resume_success
# ============================================================================


class TestActorsResumeSuccess:
    """Test POST /api/actors/resume success."""

    def test_actors_resume_success(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/resume returns 202 on success."""
        with patch(
            f"{ACTORS_SERVICE_PATH}.ActorIntegrationService"
        ) as mock_svc_class:
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.actor_id = "actor_001"
            mock_result.status = "RUNNING"
            mock_result.message = "Actor resumed"
            mock_result.error = None

            async def mock_resume(*args: object, **kwargs: object) -> MagicMock:
                return mock_result

            mock_instance.resume_actor = mock_resume

            response = client.post(
                "/api/actors/resume",
                json={"actor_id": "actor_001"},
            )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["status"] == "RUNNING"


# ============================================================================
# TEST: actors_resume_missing_actor_id_400
# ============================================================================


class TestActorsResumeMissingActorId400:
    """Test POST /api/actors/resume with missing actor_id."""

    def test_actors_resume_missing_actor_id_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/resume without actor_id returns 400."""
        response = client.post(
            "/api/actors/resume",
            json={},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "actor_id is required"


# ============================================================================
# TEST: actors_resume_unauthorized_401
# ============================================================================


class TestActorsResumeUnauthorized401:
    """Test POST /api/actors/resume unauthorized."""

    def test_actors_resume_unauthorized_401(
        self,
        app: Flask,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/resume without auth returns 401."""
        app.config["AUTH_ENABLED"] = False

        response = client.post(
            "/api/actors/resume",
            json={"actor_id": "actor_001"},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data is not None
        assert data["error"] == "unauthorized"


# ============================================================================
# TEST: actors_stop_success
# ============================================================================


class TestActorsStopSuccess:
    """Test POST /api/actors/stop success."""

    def test_actors_stop_success(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/stop returns 202 on success."""
        with patch(
            f"{ACTORS_SERVICE_PATH}.ActorIntegrationService"
        ) as mock_svc_class:
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.actor_id = "actor_001"
            mock_result.status = "STOPPED"
            mock_result.message = "Actor stopped"
            mock_result.error = None

            async def mock_stop(*args: object, **kwargs: object) -> MagicMock:
                return mock_result

            mock_instance.stop_actor = mock_stop

            response = client.post(
                "/api/actors/stop",
                json={"actor_id": "actor_001", "reason": "Shutdown"},
            )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["status"] == "STOPPED"

    def test_actors_stop_missing_actor_id_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/stop without actor_id returns 400."""
        response = client.post(
            "/api/actors/stop",
            json={"force": True},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "actor_id is required"

    def test_actors_stop_unauthorized_401(
        self,
        app: Flask,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/stop without auth returns 401."""
        app.config["AUTH_ENABLED"] = False

        response = client.post(
            "/api/actors/stop",
            json={"actor_id": "actor_001"},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data is not None
        assert data["error"] == "unauthorized"


# ============================================================================
# TEST: actors_stop_force_success
# ============================================================================


class TestActorsStopForceSuccess:
    """Test POST /api/actors/stop with force flag."""

    def test_actors_stop_force_success(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that POST /api/actors/stop with force=True returns 202."""
        with patch(
            f"{ACTORS_SERVICE_PATH}.ActorIntegrationService"
        ) as mock_svc_class:
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.actor_id = "actor_001"
            mock_result.status = "STOPPED"
            mock_result.message = "Actor force stopped"
            mock_result.error = None

            async def mock_stop(*args: object, **kwargs: object) -> MagicMock:
                return mock_result

            mock_instance.stop_actor = mock_stop

            response = client.post(
                "/api/actors/stop",
                json={
                    "actor_id": "actor_001",
                    "force": True,
                    "reason": "Emergency shutdown",
                },
            )

        assert response.status_code == 202
        data = response.get_json()
        assert data is not None
        assert data["success"] is True
        assert data["status"] == "STOPPED"


# ============================================================================
# TEST: actors_health_returns_200
# ============================================================================


class TestActorsHealthReturns200:
    """Test GET /api/actors/health returns 200."""

    def test_actors_health_returns_200(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that GET /api/actors/health returns 200 with health data."""
        with patch(
            f"{ACTORS_SERVICE_PATH}.ActorIntegrationService"
        ) as mock_svc_class:
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_snapshot = MagicMock()
            mock_snapshot.total_actors = 3
            mock_snapshot.healthy_actors = 2
            mock_snapshot.unhealthy_actors = 1
            mock_snapshot.paused_actors = 0
            mock_snapshot.actors = {
                "actor_001": {"healthy": True, "status": "RUNNING"},
                "actor_002": {"healthy": True, "status": "RUNNING"},
                "actor_003": {"healthy": False, "status": "ERROR"},
            }

            async def mock_get_health(*args: object, **kwargs: object) -> MagicMock:
                return mock_snapshot

            mock_instance.get_actor_health = mock_get_health

            response = client.get("/api/actors/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["total_actors"] == 3
        assert data["healthy_actors"] == 2
        assert data["unhealthy_actors"] == 1
        assert data["paused_actors"] == 0
        assert len(data["actors"]) == 3

    def test_actors_health_empty_actors(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that GET /api/actors/health returns 200 with no actors."""
        with patch(
            f"{ACTORS_SERVICE_PATH}.ActorIntegrationService"
        ) as mock_svc_class:
            mock_instance = MagicMock()
            mock_svc_class.return_value = mock_instance

            mock_snapshot = MagicMock()
            mock_snapshot.total_actors = 0
            mock_snapshot.healthy_actors = 0
            mock_snapshot.unhealthy_actors = 0
            mock_snapshot.paused_actors = 0
            mock_snapshot.actors = {}

            async def mock_get_health(*args: object, **kwargs: object) -> MagicMock:
                return mock_snapshot

            mock_instance.get_actor_health = mock_get_health

            response = client.get("/api/actors/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None
        assert data["total_actors"] == 0
        assert data["actors"] == {}


__all__ = [
    "TestActorsDeployInvalidType400",
    "TestActorsDeploySuccess",
    "TestActorsDeployUnauthorized401",
    "TestActorsHealthReturns200",
    "TestActorsHotReloadMissingParams400",
    "TestActorsHotReloadSuccess",
    "TestActorsHotReloadUnauthorized401",
    "TestActorsPauseMissingActorId400",
    "TestActorsPauseSuccess",
    "TestActorsPauseUnauthorized401",
    "TestActorsResumeMissingActorId400",
    "TestActorsResumeSuccess",
    "TestActorsResumeUnauthorized401",
    "TestActorsStopForceSuccess",
    "TestActorsStopSuccess",
]
