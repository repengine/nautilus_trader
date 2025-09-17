from __future__ import annotations

from typing import Callable

from ml.deployment.metrics_http import build_app


def test_health_and_metrics_endpoints() -> None:
    healthy = False

    def is_healthy() -> bool:
        return healthy

    app = build_app(is_healthy)
    client = app.test_client()

    # Initially unhealthy
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.is_json and resp.json == {"healthy": False}

    # Flip to healthy
    nonlocal_healthy: dict[str, bool] = {"v": True}

    def _is_healthy() -> bool:
        return nonlocal_healthy["v"]

    app2 = build_app(_is_healthy)
    client2 = app2.test_client()
    resp2 = client2.get("/health")
    assert resp2.status_code == 200
    assert resp2.is_json and resp2.json == {"healthy": True}

    # Metrics should return text content (may be empty if client not installed)
    m = client2.get("/metrics")
    assert m.status_code == 200
    assert isinstance(m.data, (bytes, bytearray))

