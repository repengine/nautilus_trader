"""
Contract tests for Dashboard App facade routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from ml.dashboard.config import DashboardConfig


if TYPE_CHECKING:
    from flask import Flask
    from flask.testing import FlaskClient


@pytest.fixture
def dashboard_config() -> DashboardConfig:
    """Create dashboard configuration for contract testing."""
    return DashboardConfig(
        db_connection="",
        actor_port=8081,
        strategy_port=8082,
        pipeline_port=8083,
        prometheus_url="http://localhost:9090",
        grafana_url="http://localhost:3000",
        grafana_api_token="test_api_token",
        grafana_username=None,
        grafana_password=None,
        grafana_provision_on_start=False,
        compose_enabled=False,
        store_health_enabled=False,
    )


@pytest.fixture
def facade_app(dashboard_config: DashboardConfig) -> Flask:
    """Create facade app instance."""
    from ml.dashboard.app_facade import create_app_facade

    app = create_app_facade(dashboard_config)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def facade_client(facade_app: Flask) -> FlaskClient:
    """Create test client for facade app."""
    return facade_app.test_client()


class TestHealthEndpoints:
    """Contract tests for health endpoints."""

    @patch("ml.dashboard.common.health_aggregator.requests.get")
    @patch("ml.dashboard.service.requests.get")
    def test_health_system_returns_structure(
        self,
        mock_service_get: Mock,
        mock_agg_get: Mock,
        facade_client: FlaskClient,
    ) -> None:
        """GET /api/health/system returns expected structure."""
        for mock_get in (mock_service_get, mock_agg_get):
            mock_response = Mock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_get.return_value = mock_response

        response = facade_client.get("/api/health/system")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, dict)
        assert isinstance(data.get("services"), dict)
        assert isinstance(data.get("dependencies"), dict)


class TestServiceEndpoints:
    """Contract tests for service endpoints."""

    @patch("ml.dashboard.common.health_aggregator._to_url")
    @patch("ml.dashboard.service._to_url")
    def test_services_list_returns_list(
        self,
        mock_service_url: Mock,
        mock_agg_url: Mock,
        facade_client: FlaskClient,
    ) -> None:
        """GET /api/services returns a list payload."""
        for mock_to_url in (mock_service_url, mock_agg_url):
            mock_to_url.side_effect = lambda port, path, **kwargs: f"http://localhost:{port}{path}"

        response = facade_client.get("/api/services")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, dict)
        assert isinstance(data.get("services"), list)
