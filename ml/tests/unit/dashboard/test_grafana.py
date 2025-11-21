from __future__ import annotations

from typing import Any

import pytest

from ml.dashboard.grafana import (
    GrafanaConfig,
    GrafanaProvisionResult,
    PrometheusQueryHelper,
    build_dashboard,
    default_panel_bundles,
    provision_dashboard,
)

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")


def test_build_dashboard_sets_datasource_variable() -> None:
    cfg = GrafanaConfig(url="http://grafana.local")
    dashboard = build_dashboard(
        title="Test",
        uid=cfg.dashboard_uid,
        tags=["nautilus-ml"],
        refresh="15s",
        time_from="now-1h",
        time_to="now",
        bundles=default_panel_bundles(),
        datasource_uid="prometheus-main",
    )
    assert dashboard["title"] == "Test"
    assert dashboard["uid"] == cfg.dashboard_uid
    assert dashboard["panels"], "expected default bundles to populate panels"
    datasource_vars = [var for var in dashboard["templating"]["list"] if var["name"] == "datasource"]
    assert datasource_vars
    current = datasource_vars[0].get("current")
    assert current and current.get("value") == "prometheus-main"


def test_provision_dashboard_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {"slug": "ml-test"}

        @staticmethod
        def raise_for_status() -> None:  # pragma: no cover - compatibility
            return None

        text = "ok"

    def _fake_post(url: str, data: str, headers: dict[str, str], auth: tuple[str, str] | None, timeout: float) -> _Response:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = data
        captured["auth"] = auth
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("ml.dashboard.grafana.requests.post", _fake_post)
    cfg = GrafanaConfig(url="http://grafana.local", api_token="token", dashboard_uid="ml-x")
    result = provision_dashboard(cfg, overwrite=True, title="Custom", bundles=default_panel_bundles())
    assert isinstance(result, GrafanaProvisionResult)
    assert result.ok is True
    assert result.url == "http://grafana.local/d/ml-test"
    assert captured["url"].endswith("/api/dashboards/db")
    assert "Authorization" in captured["headers"]


def test_prometheus_query_helper_collect_scalars(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status_code = 200

        def __init__(self, value: float) -> None:
            self._value = value

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "data": {
                    "result": [
                        {
                            "value": [0, str(self._value)],
                        }
                    ]
                }
            }

    def _fake_get(url: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> _Response:
        query = (params or {}).get("query", "")
        mapping = {
            "q1": 3.14,
            "q2": 2.71,
        }
        return _Response(mapping.get(query, 0.0))

    monkeypatch.setattr("ml.dashboard.grafana.requests.get", _fake_get)

    helper = PrometheusQueryHelper(base_url="http://prom.local", timeout_seconds=0.1, max_attempts=1)
    values = helper.collect_scalars({"first": "q1", "second": "q2"})
    assert pytest.approx(values["first"], rel=1e-6) == 3.14
    assert pytest.approx(values["second"], rel=1e-6) == 2.71

@pytest.fixture(autouse=True)
def _isolated_prom_registry(isolated_prometheus_registry: Any) -> None:
    """Ensure Prometheus state is isolated when helper instantiates metrics."""
    del isolated_prometheus_registry
