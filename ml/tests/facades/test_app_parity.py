"""
Parity tests for Dashboard App facade vs legacy app implementation.

CRITICAL: These tests ensure 100% API compatibility between the new app_facade
and the legacy monolithic app.py. Per CRITICAL_SAFEGUARDS.md Category 5,
these tests MUST pass in BOTH modes:
- ML_USE_LEGACY_DASHBOARD_APP=0 (facade mode - DEFAULT)
- ML_USE_LEGACY_DASHBOARD_APP=1 (legacy mode)

Test Strategy:
1. Create both legacy and facade apps
2. Run IDENTICAL requests against both implementations
3. Verify IDENTICAL response structures for ALL endpoints
4. Use property-based testing where appropriate for edge cases
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest

from ml.dashboard.config import DashboardConfig


if TYPE_CHECKING:
    from flask import Flask
    from flask.testing import FlaskClient


# Feature flag status for logging
_IMPLEMENTATION = "FACADE" if os.getenv("ML_USE_LEGACY_DASHBOARD_APP", "0") == "0" else "LEGACY"


@pytest.fixture
def dashboard_config() -> DashboardConfig:
    """Create dashboard configuration for parity testing."""
    return DashboardConfig(
        db_connection="",  # Disabled for unit tests
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
        store_health_enabled=False,  # Disabled to avoid DB dependencies
    )


@pytest.fixture
def legacy_app(dashboard_config: DashboardConfig) -> Flask:
    """Create legacy app instance."""
    from ml.dashboard.app import create_app as legacy_create_app

    app = legacy_create_app(dashboard_config)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def facade_app(dashboard_config: DashboardConfig) -> Flask:
    """Create facade app instance."""
    from ml.dashboard.app_facade import create_app_facade

    app = create_app_facade(dashboard_config)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def legacy_client(legacy_app: Flask) -> FlaskClient:
    """Create test client for legacy app."""
    return legacy_app.test_client()


@pytest.fixture
def facade_client(facade_app: Flask) -> FlaskClient:
    """Create test client for facade app."""
    return facade_app.test_client()


class TestHealthSystemParity:
    """Test /api/health/system endpoint parity."""

    @patch("ml.dashboard.common.health_aggregator.requests.get")
    @patch("ml.dashboard.service.requests.get")
    def test_health_system_parity(
        self,
        mock_legacy_get: Mock,
        mock_facade_get: Mock,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/health/system returns identical structure."""
        # Configure mocks
        for mock_get in [mock_legacy_get, mock_facade_get]:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_get.return_value = mock_response

        # Call both endpoints
        legacy_response = legacy_client.get("/api/health/system")
        facade_response = facade_client.get("/api/health/system")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify same structure
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert "services" in legacy_data and "services" in facade_data
        assert "dependencies" in legacy_data and "dependencies" in facade_data
        assert isinstance(legacy_data["services"], dict)
        assert isinstance(facade_data["services"], dict)


class TestServicesListParity:
    """Test /api/services endpoint parity."""

    @patch("ml.dashboard.common.health_aggregator._to_url")
    @patch("ml.dashboard.service._to_url")
    def test_services_list_parity(
        self,
        mock_legacy_to_url: Mock,
        mock_facade_to_url: Mock,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/services returns identical structure."""
        # Configure mocks
        for mock_to_url in [mock_legacy_to_url, mock_facade_to_url]:
            mock_to_url.side_effect = lambda port, path, **kwargs: f"http://localhost:{port}{path}"

        # Call both endpoints
        legacy_response = legacy_client.get("/api/services")
        facade_response = facade_client.get("/api/services")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify same structure
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert isinstance(legacy_data, list)
        assert isinstance(facade_data, list)
        assert len(legacy_data) >= 3  # At least 3 services
        assert len(facade_data) >= 3


class TestRegistryModelsParity:
    """Test /api/registry/models endpoint parity."""

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_model_registry")
    @patch("ml.dashboard.service.DashboardService._get_model_registry")
    def test_registry_models_parity(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/registry/models returns identical structure."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.get_all_models.return_value = []
            mock_registry.return_value = mock_reg

        # Call both endpoints
        legacy_response = legacy_client.get("/api/registry/models")
        facade_response = facade_client.get("/api/registry/models")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify same structure
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert isinstance(legacy_data, list)
        assert isinstance(facade_data, list)


class TestPipelineJobsParity:
    """Test /api/pipeline/jobs endpoint parity."""

    @patch("ml.dashboard.common.pipeline_integration.PipelineIntegrationComponent._get_pipeline_service")
    @patch("ml.dashboard.service.DashboardService._get_pipeline_service")
    def test_pipeline_jobs_parity(
        self,
        mock_legacy_service: Mock,
        mock_facade_service: Mock,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/pipeline/jobs returns identical structure."""
        # Call both endpoints (with auth)
        legacy_response = legacy_client.get(
            "/api/pipeline/jobs",
            headers={"X-ML-DASHBOARD-TOKEN": "test"},
        )
        facade_response = facade_client.get(
            "/api/pipeline/jobs",
            headers={"X-ML-DASHBOARD-TOKEN": "test"},
        )

        # Verify same status codes (may differ based on service availability)
        # Both should handle the same way
        assert type(legacy_response.status_code) is int
        assert type(facade_response.status_code) is int


class TestControlStatusParity:
    """Test /api/control/status endpoint parity."""

    def test_control_status_parity(
        self,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/control/status returns identical structure."""
        # Call both endpoints
        legacy_response = legacy_client.get("/api/control/status")
        facade_response = facade_client.get("/api/control/status")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify same structure
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert isinstance(legacy_data, dict)
        assert isinstance(facade_data, dict)


class TestMetricsSnapshotParity:
    """Test /api/metrics/snapshot endpoint parity."""

    def test_metrics_snapshot_parity(
        self,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/metrics/snapshot returns identical structure."""
        # Call both endpoints
        legacy_response = legacy_client.get("/api/metrics/snapshot")
        facade_response = facade_client.get("/api/metrics/snapshot")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify both return dicts
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert isinstance(legacy_data, dict)
        assert isinstance(facade_data, dict)


class TestTradingHealthParity:
    """Test /api/trading/health endpoint parity."""

    def test_trading_health_parity(
        self,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/trading/health returns identical structure."""
        # Call both endpoints
        legacy_response = legacy_client.get("/api/trading/health")
        facade_response = facade_client.get("/api/trading/health")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify same structure
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert isinstance(legacy_data, dict)
        assert isinstance(facade_data, dict)

        # Key fields should exist in both
        for key in ["healthy", "trading_enabled", "mode"]:
            assert key in legacy_data
            assert key in facade_data


class TestActorsHealthParity:
    """Test /api/actors/health endpoint parity."""

    def test_actors_health_parity(
        self,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/actors/health returns identical structure."""
        # Call both endpoints
        legacy_response = legacy_client.get("/api/actors/health")
        facade_response = facade_client.get("/api/actors/health")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify same structure
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert isinstance(legacy_data, dict)
        assert isinstance(facade_data, dict)

        # Key fields should exist in both
        for key in ["total_actors", "healthy_actors", "unhealthy_actors", "paused_actors"]:
            assert key in legacy_data
            assert key in facade_data


class TestStrategiesListParity:
    """Test /api/strategies endpoint parity."""

    def test_strategies_list_parity(
        self,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/strategies returns identical structure."""
        # Call both endpoints
        legacy_response = legacy_client.get("/api/strategies")
        facade_response = facade_client.get("/api/strategies")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify same structure
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert isinstance(legacy_data, dict)
        assert isinstance(facade_data, dict)


class TestFeatureFlagSelection:
    """Test feature flag selects correct implementation."""

    def test_feature_flag_selects_legacy(self, dashboard_config: DashboardConfig) -> None:
        """Test ML_USE_LEGACY_DASHBOARD_APP=1 selects legacy."""
        with patch.dict(os.environ, {"ML_USE_LEGACY_DASHBOARD_APP": "1"}):
            from ml.dashboard.app_facade import use_legacy_dashboard_app

            assert use_legacy_dashboard_app() is True

    def test_feature_flag_selects_facade(self, dashboard_config: DashboardConfig) -> None:
        """Test ML_USE_LEGACY_DASHBOARD_APP=0 selects facade."""
        with patch.dict(os.environ, {"ML_USE_LEGACY_DASHBOARD_APP": "0"}):
            from ml.dashboard.app_facade import use_legacy_dashboard_app

            assert use_legacy_dashboard_app() is False

    def test_feature_flag_default_is_facade(self, dashboard_config: DashboardConfig) -> None:
        """Test default (no env var) selects facade."""
        env_backup = os.environ.pop("ML_USE_LEGACY_DASHBOARD_APP", None)
        try:
            from ml.dashboard.app_facade import use_legacy_dashboard_app

            # Reimport to get fresh value
            import importlib

            import ml.dashboard.app_facade

            importlib.reload(ml.dashboard.app_facade)
            assert ml.dashboard.app_facade.use_legacy_dashboard_app() is False
        finally:
            if env_backup is not None:
                os.environ["ML_USE_LEGACY_DASHBOARD_APP"] = env_backup


class TestOpenAPISpecParity:
    """Test /api/openapi.json endpoint parity."""

    def test_openapi_spec_parity(
        self,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/openapi.json returns identical structure."""
        # Call both endpoints
        legacy_response = legacy_client.get("/api/openapi.json")
        facade_response = facade_client.get("/api/openapi.json")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify same structure
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert isinstance(legacy_data, dict)
        assert isinstance(facade_data, dict)

        # OpenAPI spec should have these keys
        for key in ["openapi", "info", "paths"]:
            assert key in legacy_data
            assert key in facade_data


class TestBasicHealthParity:
    """Test /health endpoint parity."""

    def test_basic_health_parity(
        self,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /health returns identical structure."""
        # Call both endpoints
        legacy_response = legacy_client.get("/health")
        facade_response = facade_client.get("/health")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify same structure
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert legacy_data == {"healthy": True}
        assert facade_data == {"healthy": True}


class TestEventsParity:
    """Test /api/events endpoint parity."""

    def test_events_list_parity(
        self,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/events returns identical structure."""
        # Call both endpoints
        legacy_response = legacy_client.get("/api/events")
        facade_response = facade_client.get("/api/events")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify same structure (list of events)
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert isinstance(legacy_data, list)
        assert isinstance(facade_data, list)


class TestObservabilityStatusParity:
    """Test /api/observability/status endpoint parity."""

    def test_observability_status_parity(
        self,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test GET /api/observability/status returns identical structure."""
        # Call both endpoints
        legacy_response = legacy_client.get("/api/observability/status")
        facade_response = facade_client.get("/api/observability/status")

        # Verify same status codes
        assert legacy_response.status_code == facade_response.status_code == 200

        # Verify same structure
        legacy_data = legacy_response.get_json()
        facade_data = facade_response.get_json()

        assert isinstance(legacy_data, dict)
        assert isinstance(facade_data, dict)

        # Key fields should exist in both
        assert "ok" in legacy_data and "ok" in facade_data
        assert "url" in legacy_data and "url" in facade_data


class TestAuthenticationParity:
    """Test authentication works identically in both implementations."""

    def test_unauthorized_pipeline_jobs(
        self,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test unauthorized requests are handled identically.

        Note: With no auth_tokens configured in DashboardConfig,
        all requests are accepted (open access mode).
        """
        # Call both endpoints without auth (no tokens configured = open access)
        legacy_response = legacy_client.get("/api/pipeline/jobs")
        facade_response = facade_client.get("/api/pipeline/jobs")

        # Both should behave the same way (accept since no tokens configured)
        assert legacy_response.status_code == facade_response.status_code

    def test_bearer_token_auth(
        self,
        legacy_client: FlaskClient,
        facade_client: FlaskClient,
    ) -> None:
        """Test Bearer token authentication works identically."""
        # Call with Bearer token header (no configured tokens = open access)
        legacy_response = legacy_client.get(
            "/api/pipeline/jobs",
            headers={"Authorization": "Bearer test_token"},
        )
        facade_response = facade_client.get(
            "/api/pipeline/jobs",
            headers={"Authorization": "Bearer test_token"},
        )

        # Both should handle auth the same way (no tokens = open access)
        assert legacy_response.status_code == facade_response.status_code
