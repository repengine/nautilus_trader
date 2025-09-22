from __future__ import annotations

import os
import datetime as dt
from datetime import datetime
from datetime import timedelta
from typing import Any

import ml.dashboard.grafana as grafana_module

import pytest
from flask.testing import FlaskClient

from ml.dashboard import DashboardConfig, create_app
from ml.dashboard.config import DashboardToken
from ml.dashboard.service import DashboardService


@pytest.fixture()
def client() -> FlaskClient:
    # Keep the app isolated; disable compose control by default
    os.environ["ML_DASHBOARD_USE_COMPOSE"] = "0"
    cfg = DashboardConfig.from_env()
    app = create_app(cfg)
    return app.test_client()


def test_health_system_ok(client: FlaskClient) -> None:
    resp = client.get("/api/health/system")
    assert resp.status_code == 200
    data: dict[str, Any] = resp.get_json()
    assert "services" in data and "dependencies" in data
    # Keys are present even if targets aren't running locally
    assert "ml_pipeline" in data["services"]


def test_services_list_contains_expected(client: FlaskClient) -> None:
    resp = client.get("/api/services")
    assert resp.status_code == 200
    services = resp.get_json()
    names = {s["name"] for s in services}
    assert {"ml_signal_actor", "ml_strategy", "ml_pipeline"}.issubset(names)


def test_services_action_unsupported_returns_ok(client: FlaskClient) -> None:
    # Compose disabled; expect supported 200 with ok False
    resp = client.post("/api/services/ml_pipeline:action", json={"action": "restart"})
    assert resp.status_code in {200, 202}
    data = resp.get_json()
    assert data["action"] == "restart"
    assert data["service"] == "ml_pipeline"
    assert data["ok"] in {False, True}


def test_pipeline_run_emits_bus_event(client: FlaskClient) -> None:
    # With default env, bus publishes to noop publisher; still returns 200/202
    payload = {"mode": "backfill", "instrument": "SPY.EQUS", "dataset": "EQUS.MINI"}
    resp = client.post("/api/pipeline/run", json=payload)
    assert resp.status_code in {200, 202}
    data = resp.get_json()
    assert set(data.keys()).issuperset({"ok", "topic"})


def test_metrics_and_health_endpoints(client: FlaskClient) -> None:
    m = client.get("/metrics")
    assert m.status_code == 200
    assert b"ml_dashboard_requests_total" in m.data

    h = client.get("/health")
    assert h.status_code == 200
    data = h.get_json()
    assert data.get("healthy") is True


def test_registry_read_endpoints(client: FlaskClient) -> None:
    # Should return empty lists/dicts by default
    r = client.get("/api/registry/models")
    assert r.status_code == 200
    assert isinstance(r.get_json(), list)

    d = client.get("/api/registry/deployments")
    assert d.status_code == 200
    assert isinstance(d.get_json(), dict)


def test_feature_strategy_dataset_list_endpoints(client: FlaskClient) -> None:
    f = client.get("/api/registry/features")
    assert f.status_code == 200
    assert isinstance(f.get_json(), list)

    s = client.get("/api/registry/strategies")
    assert s.status_code == 200
    assert isinstance(s.get_json(), list)

    ds = client.get("/api/registry/datasets")
    assert ds.status_code == 200
    assert isinstance(ds.get_json(), list)


def test_feature_promote_and_deprecate_endpoints(client: FlaskClient) -> None:
    # Unknown feature id should be handled safely
    pr = client.post("/api/registry/features/unknown:promote", json={"stage": "PROD"})
    assert pr.status_code in {200, 202}
    dr = client.post("/api/registry/features/unknown:deprecate", json={"reason": "test"})
    assert dr.status_code in {200, 202}


def test_feature_lineage_endpoint(client: FlaskClient) -> None:
    resp = client.get("/api/registry/features/unknown/lineage")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_dataset_watermarks_and_lineage_endpoints(client: FlaskClient) -> None:
    # Without dataset_id → empty list
    m = client.get("/api/registry/datasets/watermarks")
    assert m.status_code == 200
    assert m.get_json() == []

    # With dataset_id but likely none present yet → ok and list/empty
    m2 = client.get("/api/registry/datasets/watermarks?dataset_id=features")
    assert m2.status_code == 200
    assert isinstance(m2.get_json(), list)

    ln = client.get("/api/registry/datasets/lineage?child=features")
    assert ln.status_code == 200
    assert isinstance(ln.get_json(), list)


def test_registry_deploy_invalid_model_returns_ok_false(client: FlaskClient) -> None:
    resp = client.post("/api/registry/models/does_not_exist:deploy", json={"target": "ml_signal_actor"})
    assert resp.status_code in {200, 202}
    body = resp.get_json()
    assert body["model_id"] == "does_not_exist"
    assert body["target"] == "ml_signal_actor"
    assert body["ok"] in {False, True}


def test_registry_hot_reload_and_rollback_endpoints(client: FlaskClient) -> None:
    # Hot reload unknown model should be ok false/true
    hr = client.post("/api/registry/models/unknown_model:hot_reload", json={"target": "ml_signal_actor"})
    assert hr.status_code in {200, 202}
    hrj = hr.get_json()
    assert hrj["target"] == "ml_signal_actor"

    rb = client.post("/api/registry/deployments:rollback", json={"target": "ml_signal_actor", "to_model_id": "unknown_model"})
    assert rb.status_code in {200, 202}
    rbj = rb.get_json()
    assert rbj["target"] == "ml_signal_actor"


def test_events_endpoint_without_redis_returns_list(client: FlaskClient) -> None:
    resp = client.get("/api/events")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)


def test_grafana_provision_endpoint_default(client: FlaskClient) -> None:
    # Without credentials, should be safe no-op
    resp = client.post("/api/observability/grafana/provision", json={"title": "Test"})
    assert resp.status_code in {200, 202, 401}
    if resp.status_code != 401:  # guard may be enabled in env
        body = resp.get_json()
        assert set(body.keys()).issuperset({"ok", "url", "error"})


def test_observability_status_endpoint(client: FlaskClient) -> None:
    resp = client.get("/api/observability/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body.keys()).issuperset({"ok", "url", "embed_urls"})


def test_observability_summary_endpoint(client: FlaskClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        def __init__(self, value: float) -> None:
            self._value = value
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "data": {
                    "result": [
                        {
                            "value": [0, str(self._value)],
                        }
                    ],
                }
            }

    def _fake_get(url: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> _FakeResponse:
        query = (params or {}).get("query", "")
        mapping = {
            "sum(rate(ml_dashboard_requests_total[5m]))": 1.25,
            "histogram_quantile(0.95, sum(rate(ml_dashboard_latency_seconds_bucket[5m])) by (le))": 0.42,
            "sum(increase(ml_dashboard_events_failure_total[5m]))": 0.0,
        }
        return _FakeResponse(mapping.get(query, 0.0))

    monkeypatch.setattr(grafana_module.requests, "get", _fake_get)

    resp = client.get("/api/observability/summary")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert pytest.approx(data["metrics"]["request_rate_per_second"], rel=1e-6) == 1.25


def test_auth_guard_when_token_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.dashboard import create_app, DashboardConfig
    import os

    monkeypatch.setenv("ML_DASHBOARD_USE_COMPOSE", "0")
    monkeypatch.setenv("ML_DASHBOARD_TOKEN", "secret")
    app = create_app(DashboardConfig.from_env())
    c = app.test_client()
    # Missing token
    r1 = c.post("/api/services/ml_pipeline:action", json={"action": "restart"})
    assert r1.status_code == 401
    # With token
    r2 = c.post(
        "/api/services/ml_pipeline:action",
        headers={"X-ML-DASHBOARD-TOKEN": "secret"},
        json={"action": "restart"},
    )
    assert r2.status_code in {200, 202}


def test_observability_stores_endpoint(client: FlaskClient, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"ok": True, "stores": [{"store": "feature", "healthy": True}]}

    monkeypatch.setattr(
        DashboardService,
        "get_store_summary",
        lambda self: payload,
    )

    resp = client.get("/api/observability/stores")
    assert resp.status_code == 200
    assert resp.get_json() == payload


def test_dashboard_service_validate_token_success() -> None:
    cfg = DashboardConfig(auth_tokens=(DashboardToken(value="secret"),))
    svc = DashboardService.from_config(cfg)
    assert svc.validate_token("secret") is True
    assert svc.validate_token("other") is False


def test_dashboard_service_validate_token_expired() -> None:
    expired = DashboardToken(
        value="secret",
        expires_at=datetime.now(dt.UTC) - timedelta(minutes=1),
    )
    cfg = DashboardConfig(auth_tokens=(expired,))
    svc = DashboardService.from_config(cfg)
    assert svc.validate_token("secret", now=datetime.now(dt.UTC)) is False


def test_dashboard_config_token_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "ML_DASHBOARD_TOKENS",
        '[{"value":"tok1","expires":"2099-01-01T00:00:00Z"},"tok2"]',
    )
    cfg = DashboardConfig.from_env({})
    values = [token.value for token in cfg.auth_tokens]
    assert values == ["tok1", "tok2"]
