"""Tests for Control Blueprint.

This module tests the control blueprint routes:
- POST /api/control/actors/start - Start an actor
- POST /api/control/actors/stop - Stop an actor
- POST /api/control/pipeline/trigger - Trigger pipeline execution
- POST /api/control/ingestion/start - Start data ingestion
- POST /api/control/ingestion/backfill - Trigger historical backfill
- POST /api/control/emergency/stop - Emergency stop all components
- GET /api/control/status - Get system control status

Tests (15):
1. test_control_start_actor_success
2. test_control_start_actor_invalid_id_400
3. test_control_start_actor_unauthorized_401
4. test_control_stop_actor_success
5. test_control_stop_actor_invalid_id_400
6. test_control_stop_actor_not_found
7. test_control_trigger_pipeline_queued_202
8. test_control_trigger_pipeline_unavailable_503
9. test_control_trigger_pipeline_unauthorized_401
10. test_control_start_ingestion_success
11. test_control_start_ingestion_no_symbols_400
12. test_control_backfill_success
13. test_control_backfill_missing_params_400
14. test_control_emergency_stop_success
15. test_control_system_status_success
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from flask import Blueprint, Flask
from flask.testing import FlaskClient

from ml.dashboard.blueprints.control import register_control_routes


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_service() -> MagicMock:
    """Provide a mock DashboardService with pipeline methods."""
    svc = MagicMock()

    # Default trigger_pipeline response (success/queued)
    svc.trigger_pipeline.return_value = {
        "success": True,
        "job_id": "job_123",
        "pipeline_type": "full",
        "status": "QUEUED",
        "message": "Pipeline queued successfully",
    }

    return svc


@pytest.fixture
def mock_control_panel() -> MagicMock:
    """Provide a mock SimpleControlPanel."""
    panel = MagicMock()

    # Default start_actor response
    panel.start_actor.return_value = {
        "success": True,
        "actor_id": "test_actor",
        "status": "running",
    }

    # Default stop_actor response
    panel.stop_actor.return_value = {
        "success": True,
        "actor_id": "test_actor",
        "status": "stopped",
    }

    # Default trigger_pipeline response
    panel.trigger_pipeline.return_value = {
        "success": True,
        "run_id": "run_20250115_120000",
        "job_id": "job_123",
        "mode": "full",
        "status": "queued",
        "start_time": "2025-01-15T12:00:00+00:00",
    }

    # Default start_ingestion response
    panel.start_ingestion.return_value = {
        "success": True,
        "task_id": "ingest_20250115_120000",
        "symbols": ["SPY", "QQQ"],
        "source": "databento",
    }

    # Default emergency_stop_all response
    panel.emergency_stop_all.return_value = {
        "success": True,
        "stopped_components": {
            "actors": ["actor_1", "actor_2"],
            "pipelines": ["run_1"],
            "ingestion": ["ingest_1"],
        },
        "stop_time": "2025-01-15T12:00:00+00:00",
    }

    # Default get_system_status response
    panel.get_system_status.return_value = {
        "actors": {"active": 2, "max": 10, "instances": {}},
        "pipelines": {"active": 1, "max": 5, "runs": {}},
        "ingestion": {"active_tasks": 0, "tasks": {}},
        "stores": {
            "data": {"healthy": True, "fallback": False},
            "model": {"healthy": True, "fallback": False},
            "feature": {"healthy": True, "fallback": False},
            "strategy": {"healthy": True, "fallback": False},
        },
        "timestamp": "2025-01-15T12:00:00+00:00",
    }

    return panel


@pytest.fixture
def mock_dashboard_control_panel() -> MagicMock:
    """Provide a mock DashboardControlPanel for backfill tests."""
    panel = MagicMock()

    # Async method mock
    async def mock_trigger_backfill(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "task_id": "backfill_20250115_120000",
            "symbols": ["SPY", "QQQ"],
            "date_range": "2024-01-01T00:00:00 to 2024-12-31T00:00:00",
        }

    panel.trigger_backfill = mock_trigger_backfill
    return panel


@pytest.fixture
def app(mock_service: MagicMock, mock_control_panel: MagicMock) -> Flask:
    """Provide Flask test application with control blueprint."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    # Track auth state for require_token mock
    app.config["AUTH_ENABLED"] = True

    def require_token() -> bool:
        return app.config.get("AUTH_ENABLED", True)

    # Create a FRESH blueprint for each test
    bp = Blueprint("control", __name__, url_prefix="/api/control")

    # Register routes with mock service
    register_control_routes(bp, mock_service, require_token)
    app.register_blueprint(bp)

    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Provide Flask test client."""
    return app.test_client()


# ============================================================================
# TEST 1: test_control_start_actor_success
# ============================================================================


class TestControlStartActorSuccess:
    """Test POST /api/control/actors/start returns 202 on success."""

    def test_control_start_actor_success(
        self,
        client: FlaskClient,
        mock_control_panel: MagicMock,
    ) -> None:
        """Test that successful actor start returns HTTP 202."""
        with patch(
            "ml.dashboard.control_simple.SimpleControlPanel"
        ) as mock_cls:
            mock_cls.from_env.return_value = mock_control_panel

            response = client.post(
                "/api/control/actors/start",
                json={"actor_id": "test_actor", "actor_type": "signal", "config": {}},
            )

            assert response.status_code == 202
            data = response.get_json()
            assert data is not None
            assert data["success"] is True
            assert data["actor_id"] == "test_actor"
            mock_control_panel.start_actor.assert_called_once_with(
                "test_actor", "signal", {}
            )


# ============================================================================
# TEST 2: test_control_start_actor_invalid_id_400
# ============================================================================


class TestControlStartActorInvalidId400:
    """Test POST /api/control/actors/start returns 400 for missing actor_id."""

    def test_control_start_actor_invalid_id_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that missing actor_id returns HTTP 400."""
        response = client.post(
            "/api/control/actors/start",
            json={"actor_type": "signal"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "invalid_actor_id"

    def test_control_start_actor_empty_id_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that empty actor_id returns HTTP 400."""
        response = client.post(
            "/api/control/actors/start",
            json={"actor_id": "", "actor_type": "signal"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "invalid_actor_id"


# ============================================================================
# TEST 3: test_control_start_actor_unauthorized_401
# ============================================================================


class TestControlStartActorUnauthorized401:
    """Test POST /api/control/actors/start returns 401 when unauthorized."""

    def test_control_start_actor_unauthorized_401(
        self,
        app: Flask,
        client: FlaskClient,
    ) -> None:
        """Test that unauthorized request returns HTTP 401."""
        app.config["AUTH_ENABLED"] = False

        response = client.post(
            "/api/control/actors/start",
            json={"actor_id": "test_actor"},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data is not None
        assert data["error"] == "unauthorized"


# ============================================================================
# TEST 4: test_control_stop_actor_success
# ============================================================================


class TestControlStopActorSuccess:
    """Test POST /api/control/actors/stop returns 200 on success."""

    def test_control_stop_actor_success(
        self,
        client: FlaskClient,
        mock_control_panel: MagicMock,
    ) -> None:
        """Test that successful actor stop returns HTTP 200."""
        with patch(
            "ml.dashboard.control_simple.SimpleControlPanel"
        ) as mock_cls:
            mock_cls.from_env.return_value = mock_control_panel

            response = client.post(
                "/api/control/actors/stop",
                json={"actor_id": "test_actor"},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data is not None
            assert data["success"] is True
            assert data["status"] == "stopped"
            mock_control_panel.stop_actor.assert_called_once_with("test_actor")


# ============================================================================
# TEST 5: test_control_stop_actor_invalid_id_400
# ============================================================================


class TestControlStopActorInvalidId400:
    """Test POST /api/control/actors/stop returns 400 for missing actor_id."""

    def test_control_stop_actor_invalid_id_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that missing actor_id returns HTTP 400."""
        response = client.post(
            "/api/control/actors/stop",
            json={},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "invalid_actor_id"


# ============================================================================
# TEST 6: test_control_stop_actor_not_found
# ============================================================================


class TestControlStopActorNotFound:
    """Test POST /api/control/actors/stop handles not found actors."""

    def test_control_stop_actor_not_found(
        self,
        client: FlaskClient,
        mock_control_panel: MagicMock,
    ) -> None:
        """Test that stopping non-existent actor returns appropriate response."""
        mock_control_panel.stop_actor.return_value = {
            "success": False,
            "error": "Actor not found",
        }

        with patch(
            "ml.dashboard.control_simple.SimpleControlPanel"
        ) as mock_cls:
            mock_cls.from_env.return_value = mock_control_panel

            response = client.post(
                "/api/control/actors/stop",
                json={"actor_id": "nonexistent_actor"},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data is not None
            assert data["success"] is False
            assert data["error"] == "Actor not found"


# ============================================================================
# TEST 7: test_control_trigger_pipeline_queued_202
# ============================================================================


class TestControlTriggerPipelineQueued202:
    """Test POST /api/control/pipeline/trigger returns 202 when queued."""

    def test_control_trigger_pipeline_queued_202(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
        mock_control_panel: MagicMock,
    ) -> None:
        """Test that successful pipeline trigger returns HTTP 202."""
        with patch(
            "ml.dashboard.control_simple.SimpleControlPanel"
        ) as mock_cls:
            mock_cls.from_env.return_value = mock_control_panel

            response = client.post(
                "/api/control/pipeline/trigger",
                json={"pipeline_type": "full", "config": {"param": "value"}},
            )

            assert response.status_code == 202
            data = response.get_json()
            assert data is not None
            assert data["success"] is True
            assert data["status"] == "QUEUED"
            assert data["job_id"] == "job_123"
            assert "control_run_id" in data
            assert "control_status" in data


# ============================================================================
# TEST 8: test_control_trigger_pipeline_unavailable_503
# ============================================================================


class TestControlTriggerPipelineUnavailable503:
    """Test POST /api/control/pipeline/trigger returns 503 when unavailable."""

    def test_control_trigger_pipeline_unavailable_503(
        self,
        client: FlaskClient,
        mock_service: MagicMock,
        mock_control_panel: MagicMock,
    ) -> None:
        """Test that unavailable pipeline service returns HTTP 503."""
        mock_service.trigger_pipeline.return_value = {
            "success": False,
            "status": "UNAVAILABLE",
            "message": "Pipeline service not available",
        }

        with patch(
            "ml.dashboard.control_simple.SimpleControlPanel"
        ) as mock_cls:
            mock_cls.from_env.return_value = mock_control_panel

            response = client.post(
                "/api/control/pipeline/trigger",
                json={"pipeline_type": "full"},
            )

            assert response.status_code == 503
            data = response.get_json()
            assert data["status"] == "UNAVAILABLE"


# ============================================================================
# TEST 9: test_control_trigger_pipeline_unauthorized_401
# ============================================================================


class TestControlTriggerPipelineUnauthorized401:
    """Test POST /api/control/pipeline/trigger returns 401 when unauthorized."""

    def test_control_trigger_pipeline_unauthorized_401(
        self,
        app: Flask,
        client: FlaskClient,
    ) -> None:
        """Test that unauthorized request returns HTTP 401."""
        app.config["AUTH_ENABLED"] = False

        response = client.post(
            "/api/control/pipeline/trigger",
            json={"pipeline_type": "full"},
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data["error"] == "unauthorized"


# ============================================================================
# TEST 10: test_control_start_ingestion_success
# ============================================================================


class TestControlStartIngestionSuccess:
    """Test POST /api/control/ingestion/start returns 202 on success."""

    def test_control_start_ingestion_success(
        self,
        client: FlaskClient,
        mock_control_panel: MagicMock,
    ) -> None:
        """Test that successful ingestion start returns HTTP 202."""
        with patch(
            "ml.dashboard.control_simple.SimpleControlPanel"
        ) as mock_cls:
            mock_cls.from_env.return_value = mock_control_panel

            response = client.post(
                "/api/control/ingestion/start",
                json={"symbols": ["SPY", "QQQ"], "source": "databento"},
            )

            assert response.status_code == 202
            data = response.get_json()
            assert data is not None
            assert data["success"] is True
            assert data["symbols"] == ["SPY", "QQQ"]
            mock_control_panel.start_ingestion.assert_called_once_with(
                ["SPY", "QQQ"], "databento"
            )


# ============================================================================
# TEST 11: test_control_start_ingestion_no_symbols_400
# ============================================================================


class TestControlStartIngestionNoSymbols400:
    """Test POST /api/control/ingestion/start returns 400 for empty symbols."""

    def test_control_start_ingestion_no_symbols_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that empty symbols list returns HTTP 400."""
        response = client.post(
            "/api/control/ingestion/start",
            json={"symbols": [], "source": "databento"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "no_symbols"

    def test_control_start_ingestion_missing_symbols_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that missing symbols key returns HTTP 400."""
        response = client.post(
            "/api/control/ingestion/start",
            json={"source": "databento"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "no_symbols"


# ============================================================================
# TEST 12: test_control_backfill_success
# ============================================================================


class TestControlBackfillSuccess:
    """Test POST /api/control/ingestion/backfill returns 202 on success."""

    def test_control_backfill_success(
        self,
        client: FlaskClient,
        mock_dashboard_control_panel: MagicMock,
    ) -> None:
        """Test that successful backfill returns HTTP 202."""
        with patch(
            "ml.dashboard.control_panel.DashboardControlPanel"
        ) as mock_cls:
            mock_cls.from_env.return_value = mock_dashboard_control_panel

            response = client.post(
                "/api/control/ingestion/backfill",
                json={
                    "symbols": ["SPY", "QQQ"],
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
            )

            assert response.status_code == 202
            data = response.get_json()
            assert data is not None
            assert data["success"] is True


# ============================================================================
# TEST 13: test_control_backfill_missing_params_400
# ============================================================================


class TestControlBackfillMissingParams400:
    """Test POST /api/control/ingestion/backfill returns 400 for missing params."""

    def test_control_backfill_missing_symbols_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that missing symbols returns HTTP 400."""
        response = client.post(
            "/api/control/ingestion/backfill",
            json={"start_date": "2024-01-01", "end_date": "2024-12-31"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "missing_params"

    def test_control_backfill_missing_start_date_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that missing start_date returns HTTP 400."""
        response = client.post(
            "/api/control/ingestion/backfill",
            json={"symbols": ["SPY"], "end_date": "2024-12-31"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "missing_params"

    def test_control_backfill_missing_end_date_400(
        self,
        client: FlaskClient,
    ) -> None:
        """Test that missing end_date returns HTTP 400."""
        response = client.post(
            "/api/control/ingestion/backfill",
            json={"symbols": ["SPY"], "start_date": "2024-01-01"},
        )

        assert response.status_code == 400
        data = response.get_json()
        assert data is not None
        assert data["error"] == "missing_params"


# ============================================================================
# TEST 14: test_control_emergency_stop_success
# ============================================================================


class TestControlEmergencyStopSuccess:
    """Test POST /api/control/emergency/stop returns 200 on success."""

    def test_control_emergency_stop_success(
        self,
        client: FlaskClient,
        mock_control_panel: MagicMock,
    ) -> None:
        """Test that emergency stop returns HTTP 200."""
        with patch(
            "ml.dashboard.control_simple.SimpleControlPanel"
        ) as mock_cls:
            mock_cls.from_env.return_value = mock_control_panel

            response = client.post("/api/control/emergency/stop")

            assert response.status_code == 200
            data = response.get_json()
            assert data is not None
            assert data["success"] is True
            assert "stopped_components" in data
            mock_control_panel.emergency_stop_all.assert_called_once()


# ============================================================================
# TEST 15: test_control_system_status_success
# ============================================================================


class TestControlSystemStatusSuccess:
    """Test GET /api/control/status returns 200 with system status."""

    def test_control_system_status_success(
        self,
        client: FlaskClient,
        mock_control_panel: MagicMock,
    ) -> None:
        """Test that system status returns HTTP 200 with data."""
        with patch(
            "ml.dashboard.control_simple.SimpleControlPanel"
        ) as mock_cls:
            mock_cls.from_env.return_value = mock_control_panel

            response = client.get("/api/control/status")

            assert response.status_code == 200
            data = response.get_json()
            assert data is not None
            assert "actors" in data
            assert "pipelines" in data
            assert "ingestion" in data
            assert "stores" in data
            assert "timestamp" in data
            mock_control_panel.get_system_status.assert_called_once()


__all__ = [
    "TestControlBackfillMissingParams400",
    "TestControlBackfillSuccess",
    "TestControlEmergencyStopSuccess",
    "TestControlStartActorInvalidId400",
    "TestControlStartActorSuccess",
    "TestControlStartActorUnauthorized401",
    "TestControlStartIngestionNoSymbols400",
    "TestControlStartIngestionSuccess",
    "TestControlStopActorInvalidId400",
    "TestControlStopActorNotFound",
    "TestControlStopActorSuccess",
    "TestControlSystemStatusSuccess",
    "TestControlTriggerPipelineQueued202",
    "TestControlTriggerPipelineUnauthorized401",
    "TestControlTriggerPipelineUnavailable503",
]
