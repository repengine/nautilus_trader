"""
Grafana provisioning helpers (cold path).

Builds a simple dashboard JSON and provisions it via Grafana HTTP API when
`GRAFANA_URL` and either `GRAFANA_API_TOKEN` or basic auth are provided.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True, frozen=True)
class GrafanaConfig:
    url: str
    api_token: str | None = None
    username: str | None = None
    password: str | None = None
    folder_uid: str | None = None


def build_dashboard(title: str = "Nautilus ML Dashboard") -> dict[str, Any]:
    """
    Return a small dashboard JSON with control-plane metrics panels.
    """
    panels: list[dict[str, Any]] = []

    # Control-plane requests rate
    panels.append(
        {
            "type": "graph",
            "title": "Dashboard Requests (rate)",
            "targets": [
                {
                    "expr": "sum by (status) (rate(ml_dashboard_requests_total[5m]))",
                    "legendFormat": "{{status}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 9, "w": 12, "x": 0, "y": 0},
        }
    )

    # Control-plane latency P95
    panels.append(
        {
            "type": "stat",
            "title": "Latency P95 (s)",
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, sum(rate(ml_dashboard_latency_seconds_bucket[5m])) by (le))",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 9, "w": 12, "x": 12, "y": 0},
        }
    )

    # Service status (Prometheus up)
    panels.append(
        {
            "type": "stat",
            "title": "Services Up",
            "targets": [
                {
                    "expr": 'sum(up{job=~"ml_pipeline|ml_signal_actor"})',
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 7, "w": 6, "x": 0, "y": 9},
        }
    )

    # Actor predictions rate (if exposed by monitoring collector)
    panels.append(
        {
            "type": "graph",
            "title": "Actor Predictions Rate",
            "targets": [
                {"expr": "sum(rate(ml_predictions_total[5m]))", "refId": "A"},
            ],
            "gridPos": {"h": 7, "w": 9, "x": 6, "y": 9},
        }
    )

    # Strategy signals/trades rate
    panels.append(
        {
            "type": "graph",
            "title": "Strategy Signals / Trades Rate",
            "targets": [
                {"expr": "sum(rate(nautilus_ml_signals_received_total[5m]))", "legendFormat": "signals", "refId": "A"},
                {"expr": "sum(rate(nautilus_ml_trades_executed_total[5m]))", "legendFormat": "trades", "refId": "B"},
            ],
            "gridPos": {"h": 7, "w": 9, "x": 15, "y": 9},
        }
    )

    # Strategy signal->trade latency P95
    panels.append(
        {
            "type": "stat",
            "title": "Signal→Trade Latency P95 (s)",
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, sum(rate(nautilus_ml_signal_to_trade_latency_seconds_bucket[5m])) by (le))",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 7, "w": 6, "x": 0, "y": 16},
        }
    )

    dashboard: dict[str, Any] = {
        "title": title,
        "uid": None,  # Let Grafana assign
        "panels": panels,
        "schemaVersion": 38,
        "version": 0,
        "tags": ["nautilus-ml", "control-plane"],
        "time": {"from": "now-6h", "to": "now"},
        "templating": {"list": []},
    }
    return dashboard


def provision_dashboard(cfg: GrafanaConfig, *, overwrite: bool = True, title: str | None = None) -> tuple[bool, str | None]:
    """
    Provision a dashboard via Grafana HTTP API.

    Returns (ok, url) where url is the dashboard path if created.
    """
    dashboard = build_dashboard(title or "Nautilus ML Dashboard")
    payload = {"dashboard": dashboard, "overwrite": overwrite, "folderUid": cfg.folder_uid}

    headers = {"Content-Type": "application/json"}
    auth: tuple[str, str] | None = None
    if cfg.api_token:
        headers["Authorization"] = f"Bearer {cfg.api_token}"
    elif cfg.username and cfg.password:
        auth = (cfg.username, cfg.password)

    url = cfg.url.rstrip("/") + "/api/dashboards/db"

    try:
        resp = requests.post(url, data=json.dumps(payload), headers=headers, auth=auth, timeout=5.0)
        if resp.status_code // 100 != 2:
            return False, None
        body = resp.json()
        slug = body.get("slug")
        if slug:
            return True, cfg.url.rstrip("/") + "/d/" + slug
        return True, None
    except Exception:
        return False, None


__all__ = ["GrafanaConfig", "build_dashboard", "provision_dashboard"]
