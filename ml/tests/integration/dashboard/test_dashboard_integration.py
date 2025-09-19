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

