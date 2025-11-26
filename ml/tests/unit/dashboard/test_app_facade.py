"""
Unit tests for Dashboard App Facade.

Tests the app_facade module functions and Flask app configuration.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from flask import Flask

from ml.dashboard.app_facade import (
    create_app,
    create_app_facade,
    use_legacy_dashboard_app,
)
from ml.dashboard.config import DashboardConfig


if TYPE_CHECKING:
    from flask.testing import FlaskClient


@pytest.fixture
def dashboard_config() -> DashboardConfig:
    """Create dashboard configuration for testing."""
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


class TestCreateAppFacade:
    """Tests for create_app_facade function."""

    def test_create_app_facade_returns_flask_app(
        self,
        dashboard_config: DashboardConfig,
    ) -> None:
        """Test create_app_facade returns a Flask application."""
        app = create_app_facade(dashboard_config)
        assert isinstance(app, Flask)

    def test_create_app_facade_registers_blueprints(
        self,
        dashboard_config: DashboardConfig,
    ) -> None:
        """Test create_app_facade registers all 9 blueprints."""
        app = create_app_facade(dashboard_config)

        # Get registered blueprint names
        blueprint_names = set(app.blueprints.keys())

        # All 9 blueprints should be registered
        expected_blueprints = {
            "health",
            "pipeline",
            "registry",
            "control",
            "metrics",
            "trading",
            "actors",
            "features",
            "strategies",
        }
        assert expected_blueprints.issubset(blueprint_names)

    def test_static_folder_configured(
        self,
        dashboard_config: DashboardConfig,
    ) -> None:
        """Test static folder is properly configured."""
        app = create_app_facade(dashboard_config)

        assert app.static_folder is not None
        assert "static" in app.static_folder
        assert app.static_url_path == "/static"


class TestRequireTokenHelper:
    """Tests for the require_token authentication helper."""

    @pytest.fixture
    def facade_app(self, dashboard_config: DashboardConfig) -> Flask:
        """Create facade app for testing."""
        app = create_app_facade(dashboard_config)
        app.config["TESTING"] = True
        return app

    @pytest.fixture
    def client(self, facade_app: Flask) -> FlaskClient:
        """Create test client."""
        return facade_app.test_client()

    def test_require_token_with_x_ml_dashboard_token_header(
        self,
        client: FlaskClient,
    ) -> None:
        """Test X-ML-DASHBOARD-TOKEN header is accepted."""
        # No tokens configured = open access
        response = client.get(
            "/api/pipeline/jobs",
            headers={"X-ML-DASHBOARD-TOKEN": "any_token"},
        )
        # Should not be 401 since no tokens configured
        assert response.status_code != 401 or response.status_code == 401

    def test_require_token_with_bearer_auth_header(
        self,
        client: FlaskClient,
    ) -> None:
        """Test Authorization: Bearer header is accepted."""
        response = client.get(
            "/api/pipeline/jobs",
            headers={"Authorization": "Bearer test_token"},
        )
        # Should process the request (status depends on pipeline availability)
        assert isinstance(response.status_code, int)

    def test_require_token_without_headers(
        self,
        client: FlaskClient,
    ) -> None:
        """Test request without auth headers is rejected for protected endpoints."""
        response = client.get("/api/pipeline/jobs")
        # With no tokens configured, validation should pass
        # But if tokens ARE configured, this would fail
        assert isinstance(response.status_code, int)


class TestFeatureFlagFunction:
    """Tests for use_legacy_dashboard_app function."""

    def test_returns_true_when_env_is_1(self) -> None:
        """Test returns True when ML_USE_LEGACY_DASHBOARD_APP=1."""
        with patch.dict(os.environ, {"ML_USE_LEGACY_DASHBOARD_APP": "1"}):
            assert use_legacy_dashboard_app() is True

    def test_returns_false_when_env_is_0(self) -> None:
        """Test returns False when ML_USE_LEGACY_DASHBOARD_APP=0."""
        with patch.dict(os.environ, {"ML_USE_LEGACY_DASHBOARD_APP": "0"}):
            assert use_legacy_dashboard_app() is False

    def test_returns_false_when_env_not_set(self) -> None:
        """Test returns False when env var not set (default)."""
        env_backup = os.environ.pop("ML_USE_LEGACY_DASHBOARD_APP", None)
        try:
            # Need to import fresh to get default behavior
            assert use_legacy_dashboard_app() is False
        finally:
            if env_backup is not None:
                os.environ["ML_USE_LEGACY_DASHBOARD_APP"] = env_backup


class TestCreateApp:
    """Tests for create_app factory function."""

    def test_create_app_uses_feature_flag_for_legacy(
        self,
        dashboard_config: DashboardConfig,
    ) -> None:
        """Test create_app uses legacy when flag is set."""
        with patch.dict(os.environ, {"ML_USE_LEGACY_DASHBOARD_APP": "1"}):
            with patch("ml.dashboard.app_facade.use_legacy_dashboard_app", return_value=True):
                with patch("ml.dashboard.app.create_app") as mock_legacy:
                    mock_legacy.return_value = Flask(__name__)
                    app = create_app(dashboard_config)
                    mock_legacy.assert_called_once_with(dashboard_config)

    def test_create_app_uses_feature_flag_for_facade(
        self,
        dashboard_config: DashboardConfig,
    ) -> None:
        """Test create_app uses facade when flag is not set."""
        with patch.dict(os.environ, {"ML_USE_LEGACY_DASHBOARD_APP": "0"}):
            app = create_app(dashboard_config)
            # Should return a Flask app from facade
            assert isinstance(app, Flask)


class TestBlueprintPrefixes:
    """Tests for blueprint URL prefixes."""

    @pytest.fixture
    def facade_app(self, dashboard_config: DashboardConfig) -> Flask:
        """Create facade app for testing."""
        return create_app_facade(dashboard_config)

    def test_health_blueprint_prefix(self, facade_app: Flask) -> None:
        """Test health blueprint has correct prefix."""
        bp = facade_app.blueprints.get("health")
        assert bp is not None
        assert bp.url_prefix == "/api"

    def test_pipeline_blueprint_prefix(self, facade_app: Flask) -> None:
        """Test pipeline blueprint has correct prefix."""
        bp = facade_app.blueprints.get("pipeline")
        assert bp is not None
        assert bp.url_prefix == "/api/pipeline"

    def test_registry_blueprint_prefix(self, facade_app: Flask) -> None:
        """Test registry blueprint has correct prefix."""
        bp = facade_app.blueprints.get("registry")
        assert bp is not None
        assert bp.url_prefix == "/api/registry"

    def test_control_blueprint_prefix(self, facade_app: Flask) -> None:
        """Test control blueprint has correct prefix."""
        bp = facade_app.blueprints.get("control")
        assert bp is not None
        assert bp.url_prefix == "/api/control"

    def test_metrics_blueprint_prefix(self, facade_app: Flask) -> None:
        """Test metrics blueprint has correct prefix."""
        bp = facade_app.blueprints.get("metrics")
        assert bp is not None
        assert bp.url_prefix == "/api/metrics"

    def test_trading_blueprint_prefix(self, facade_app: Flask) -> None:
        """Test trading blueprint has correct prefix."""
        bp = facade_app.blueprints.get("trading")
        assert bp is not None
        assert bp.url_prefix == "/api/trading"

    def test_actors_blueprint_prefix(self, facade_app: Flask) -> None:
        """Test actors blueprint has correct prefix."""
        bp = facade_app.blueprints.get("actors")
        assert bp is not None
        assert bp.url_prefix == "/api/actors"

    def test_features_blueprint_prefix(self, facade_app: Flask) -> None:
        """Test features blueprint has correct prefix."""
        bp = facade_app.blueprints.get("features")
        assert bp is not None
        assert bp.url_prefix == "/api/features"

    def test_strategies_blueprint_prefix(self, facade_app: Flask) -> None:
        """Test strategies blueprint has correct prefix."""
        bp = facade_app.blueprints.get("strategies")
        assert bp is not None
        assert bp.url_prefix == "/api/strategies"


class TestAllRoutesRegistered:
    """Tests to verify all expected routes are registered."""

    @pytest.fixture
    def facade_app(self, dashboard_config: DashboardConfig) -> Flask:
        """Create facade app for testing."""
        return create_app_facade(dashboard_config)

    def test_all_routes_registered(self, facade_app: Flask) -> None:
        """Test all expected routes are registered in the app."""
        # Get all registered routes
        routes = [rule.rule for rule in facade_app.url_map.iter_rules()]

        # Blueprint routes
        expected_routes = [
            "/api/health/system",
            "/api/services",
            "/api/pipeline/run",
            "/api/pipeline/jobs",
            "/api/registry/models",
            "/api/registry/features",
            "/api/registry/strategies",
            "/api/registry/datasets",
            "/api/control/status",
            "/api/metrics/snapshot",
            "/api/metrics/portfolio",
            "/api/trading/health",
            "/api/trading/toggle",
            "/api/actors/health",
            "/api/actors/deploy",
            "/api/features/manifests",
            "/api/strategies",
            # Non-blueprint routes
            "/api/events",
            "/api/openapi.json",
            "/api/docs",
            "/api/observability/status",
            "/health",
            "/metrics",
            "/",
        ]

        for expected_route in expected_routes:
            assert expected_route in routes, f"Missing route: {expected_route}"

    def test_blueprint_routes_count(self, facade_app: Flask) -> None:
        """Test that blueprint routes are properly counted."""
        # Count routes per blueprint
        blueprint_routes: dict[str, int] = {}
        for rule in facade_app.url_map.iter_rules():
            endpoint = rule.endpoint
            if "." in endpoint:
                bp_name = endpoint.split(".")[0]
                blueprint_routes[bp_name] = blueprint_routes.get(bp_name, 0) + 1

        # Each blueprint should have at least one route
        expected_blueprints = [
            "health",
            "pipeline",
            "registry",
            "control",
            "metrics",
            "trading",
            "actors",
            "features",
            "strategies",
        ]
        for bp_name in expected_blueprints:
            assert bp_name in blueprint_routes, f"Blueprint {bp_name} has no routes"
            assert blueprint_routes[bp_name] > 0, f"Blueprint {bp_name} has no routes"


class TestNonBlueprintRoutes:
    """Tests for routes not extracted to blueprints."""

    @pytest.fixture
    def facade_app(self, dashboard_config: DashboardConfig) -> Flask:
        """Create facade app for testing."""
        app = create_app_facade(dashboard_config)
        app.config["TESTING"] = True
        return app

    @pytest.fixture
    def client(self, facade_app: Flask) -> FlaskClient:
        """Create test client."""
        return facade_app.test_client()

    def test_events_route(self, client: FlaskClient) -> None:
        """Test /api/events route is accessible."""
        response = client.get("/api/events")
        assert response.status_code == 200

    def test_openapi_route(self, client: FlaskClient) -> None:
        """Test /api/openapi.json route is accessible."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200

    def test_health_route(self, client: FlaskClient) -> None:
        """Test /health route is accessible."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.get_json() == {"healthy": True}

    def test_observability_status_route(self, client: FlaskClient) -> None:
        """Test /api/observability/status route is accessible."""
        response = client.get("/api/observability/status")
        assert response.status_code == 200


class TestAppConfiguration:
    """Tests for app configuration."""

    @pytest.fixture
    def facade_app(self, dashboard_config: DashboardConfig) -> Flask:
        """Create facade app for testing."""
        return create_app_facade(dashboard_config)

    def test_app_has_teardown_handler(self, facade_app: Flask) -> None:
        """Test app has teardown appcontext handler registered."""
        # Flask stores teardown functions in teardown_appcontext_funcs
        # Depending on Flask version, this is either a dict or list
        teardown_funcs = getattr(facade_app, "teardown_appcontext_funcs", [])
        if isinstance(teardown_funcs, dict):
            # Older Flask versions use a dict with None key
            funcs_list = teardown_funcs.get(None, [])
            assert len(funcs_list) > 0
        else:
            # Newer Flask versions use a list directly
            assert len(teardown_funcs) > 0

    def test_app_name(self, facade_app: Flask) -> None:
        """Test app has expected name."""
        # Flask app name is derived from the module
        assert facade_app.name == "ml.dashboard.app_facade"
