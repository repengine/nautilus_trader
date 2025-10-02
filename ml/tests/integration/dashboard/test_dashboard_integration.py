from __future__ import annotations

import os

import pytest
from flask.testing import FlaskClient

from ml.dashboard import DashboardConfig, create_app


@pytest.mark.integration
def test_integration_dashboard_endpoints_basic() -> None:
    os.environ["ML_DASHBOARD_USE_COMPOSE"] = "0"
    app = create_app(DashboardConfig.from_env())
    client: FlaskClient = app.test_client()

    # Health/system should return dict with services/dependencies keys
    r1 = client.get("/api/health/system")
    assert r1.status_code == 200
    body = r1.get_json()
    assert isinstance(body, dict)
    assert "services" in body and "dependencies" in body

    # Metrics endpoint responds
    r2 = client.get("/metrics")
    assert r2.status_code == 200
    assert b"ml_dashboard_requests_total" in r2.data


@pytest.mark.integration
def test_integration_registry_listings() -> None:
    os.environ["ML_DASHBOARD_USE_COMPOSE"] = "0"
    app = create_app(DashboardConfig.from_env())
    client: FlaskClient = app.test_client()

    for path in ("/api/registry/features", "/api/registry/strategies", "/api/registry/datasets"):
        r = client.get(path)
        assert r.status_code == 200
        assert isinstance(r.get_json(), list)


@pytest.mark.integration
def test_integration_trading_health() -> None:
    """Test trading health endpoint."""
    os.environ["ML_DASHBOARD_USE_COMPOSE"] = "0"
    app = create_app(DashboardConfig.from_env())
    client: FlaskClient = app.test_client()

    r = client.get("/api/trading/health")
    assert r.status_code == 200
    body = r.get_json()
    assert isinstance(body, dict)
    assert "healthy" in body
    assert "trading_enabled" in body
    assert "mode" in body


@pytest.mark.integration
def test_integration_trading_market_data() -> None:
    """Test market data endpoint."""
    os.environ["ML_DASHBOARD_USE_COMPOSE"] = "0"
    app = create_app(DashboardConfig.from_env())
    client: FlaskClient = app.test_client()

    r = client.get("/api/trading/market-data")
    assert r.status_code == 200
    body = r.get_json()
    assert isinstance(body, dict)
    assert "generated_at" in body
    assert "total_positions" in body
    assert "total_exposure" in body


@pytest.mark.integration
def test_integration_trading_toggle_unauthorized() -> None:
    """Test trading toggle requires auth."""
    os.environ["ML_DASHBOARD_USE_COMPOSE"] = "0"
    os.environ["ML_DASHBOARD_TOKENS"] = "test-token-123"
    app = create_app(DashboardConfig.from_env())
    client: FlaskClient = app.test_client()

    r = client.post("/api/trading/toggle", json={"enable": True})
    assert r.status_code == 401
    body = r.get_json()
    assert body["error"] == "unauthorized"

    # Clean up
    os.environ.pop("ML_DASHBOARD_TOKENS", None)


@pytest.mark.integration
def test_integration_trading_emergency_unauthorized() -> None:
    """Test emergency stop requires auth."""
    os.environ["ML_DASHBOARD_USE_COMPOSE"] = "0"
    os.environ["ML_DASHBOARD_TOKENS"] = "test-token-456"
    app = create_app(DashboardConfig.from_env())
    client: FlaskClient = app.test_client()

    r = client.post("/api/trading/emergency")
    assert r.status_code == 401
    body = r.get_json()
    assert body["error"] == "unauthorized"

    # Clean up
    os.environ.pop("ML_DASHBOARD_TOKENS", None)

