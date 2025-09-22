"""
Grafana provisioning helpers (cold path).

This module composes reusable dashboard panel bundles using the shared
``ml.monitoring.dashboard_factory`` utilities and exposes a thin provisioning
helper. It remains cold-path only and avoids direct Prometheus or Grafana
imports beyond simple HTTP calls.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any, cast
from urllib.parse import urljoin

import requests

from ml.common.retry_utils import retry_with_backoff
from ml.monitoring.dashboard_factory import GrafanaDashboardFactory


logger = logging.getLogger(__name__)

@dataclass(slots=True, frozen=True)
class GrafanaConfig:
    url: str
    api_token: str | None = None
    username: str | None = None
    password: str | None = None
    folder_uid: str | None = None
    datasource_uid: str | None = None
    dashboard_uid: str = "ml-control-plane"
    dashboard_title: str = "Nautilus ML Control Plane"
    dashboard_tags: tuple[str, ...] = ("nautilus-ml", "control-plane")
    refresh_interval: str = "30s"
    dashboard_time_from: str = "now-6h"
    dashboard_time_to: str = "now"


@dataclass(slots=True, frozen=True)
class GrafanaProvisionResult:
    ok: bool
    url: str | None
    status_code: int | None = None
    error: str | None = None


@dataclass(slots=True)
class _PanelIdAllocator:
    """Allocate incrementing panel identifiers for deterministic JSON."""

    start: int = 1
    _next: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_next", self.start)

    def next(self) -> int:
        value = self._next
        object.__setattr__(self, "_next", value + 1)
        return value


BundleBuilder = Callable[[GrafanaDashboardFactory, _PanelIdAllocator, int], tuple[list[dict[str, Any]], int]]


@dataclass(slots=True, frozen=True)
class GrafanaPanelBundle:
    """Container for reusable Grafana panel bundles."""

    key: str
    builder: BundleBuilder

    def build(
        self,
        factory: GrafanaDashboardFactory,
        allocator: _PanelIdAllocator,
        start_y: int,
    ) -> tuple[list[dict[str, Any]], int]:
        return self.builder(factory, allocator, start_y)


def _control_plane_bundle(
    factory: GrafanaDashboardFactory,
    allocator: _PanelIdAllocator,
    start_y: int,
) -> tuple[list[dict[str, Any]], int]:
    panels: list[dict[str, Any]] = []
    panels.append(factory.panel_factory.create_row_panel("Control Plane", allocator.next(), start_y))
    current_y = start_y + 1

    panels.append(
        factory.panel_factory.create_stat_panel(
            title="Dashboard Requests /s",
            expr="sum(rate(ml_dashboard_requests_total[5m]))",
            panel_id=allocator.next(),
            grid_pos={"h": 6, "w": 8, "x": 0, "y": current_y},
            unit="req/s",
        )
    )
    panels.append(
        factory.panel_factory.create_stat_panel(
            title="Latency P95 (s)",
            expr="histogram_quantile(0.95, sum(rate(ml_dashboard_latency_seconds_bucket[5m])) by (le))",
            panel_id=allocator.next(),
            grid_pos={"h": 6, "w": 8, "x": 8, "y": current_y},
            unit="s",
        )
    )
    panels.append(
        factory.panel_factory.create_stat_panel(
            title="Event Poll Failures /5m",
            expr="sum(increase(ml_dashboard_events_failure_total[5m]))",
            panel_id=allocator.next(),
            grid_pos={"h": 6, "w": 8, "x": 16, "y": current_y},
            unit="short",
        )
    )

    current_y += 6

    panels.append(
        factory.panel_factory.create_timeseries_panel(
            title="Requests by Status",
            targets=[
                {
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                    "expr": "sum(rate(ml_dashboard_requests_total[5m])) by (status)",
                    "legendFormat": "{{status}}",
                    "refId": "A",
                }
            ],
            panel_id=allocator.next(),
            grid_pos={"h": 8, "w": 24, "x": 0, "y": current_y},
            unit="req/s",
        )
    )

    current_y += 8

    return panels, current_y


def _inference_bundle(
    factory: GrafanaDashboardFactory,
    allocator: _PanelIdAllocator,
    start_y: int,
) -> tuple[list[dict[str, Any]], int]:
    panels: list[dict[str, Any]] = []
    panels.append(factory.panel_factory.create_row_panel("Inference", allocator.next(), start_y))
    current_y = start_y + 1

    panels.append(
        factory.panel_factory.create_timeseries_panel(
            title="Predictions /s",
            targets=[
                {
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                    "expr": "sum(rate(ml_predictions_total[5m]))",
                    "legendFormat": "predictions",
                    "refId": "A",
                }
            ],
            panel_id=allocator.next(),
            grid_pos={"h": 8, "w": 12, "x": 0, "y": current_y},
            unit="req/s",
        )
    )
    panels.append(
        factory.panel_factory.create_timeseries_panel(
            title="Signal → Trade Latency P95",
            targets=[
                {
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                    "expr": "histogram_quantile(0.95, sum(rate(nautilus_ml_signal_to_trade_latency_seconds_bucket[5m])) by (le))",
                    "legendFormat": "p95",
                    "refId": "A",
                }
            ],
            panel_id=allocator.next(),
            grid_pos={"h": 8, "w": 12, "x": 12, "y": current_y},
            unit="s",
        )
    )

    current_y += 8

    return panels, current_y


def default_panel_bundles() -> tuple[GrafanaPanelBundle, ...]:
    """Return the default set of Grafana panel bundles."""
    return (
        GrafanaPanelBundle(key="control_plane", builder=_control_plane_bundle),
        GrafanaPanelBundle(key="inference", builder=_inference_bundle),
    )


def build_dashboard(
    *,
    title: str,
    uid: str,
    tags: Sequence[str],
    refresh: str,
    time_from: str,
    time_to: str,
    bundles: Sequence[GrafanaPanelBundle] | None = None,
    datasource_uid: str | None = None,
) -> dict[str, Any]:
    """Compose a Grafana dashboard using reusable bundles."""
    factory = GrafanaDashboardFactory()
    dashboard = factory.create_base_dashboard(
        title=title,
        uid=uid,
        tags=list(tags),
        refresh=refresh,
        time_from=time_from,
        time_to=time_to,
    )

    allocator = _PanelIdAllocator()
    current_y = 0
    for bundle in bundles or default_panel_bundles():
        built_panels, current_y = bundle.build(factory, allocator, current_y)
        dashboard["panels"].extend(built_panels)

    if datasource_uid:
        for variable in dashboard.get("templating", {}).get("list", []):
            if variable.get("name") == "datasource":
                variable.setdefault("current", {})
                variable["current"].update({"selected": True, "text": datasource_uid, "value": datasource_uid})
                break

    return dashboard


def provision_dashboard(
    cfg: GrafanaConfig,
    *,
    overwrite: bool = True,
    title: str | None = None,
    bundles: Sequence[GrafanaPanelBundle] | None = None,
) -> GrafanaProvisionResult:
    """
    Provision a dashboard via Grafana HTTP API.

    Returns a :class:`GrafanaProvisionResult` describing the outcome.
    """
    dashboard = build_dashboard(
        title=title or cfg.dashboard_title,
        uid=cfg.dashboard_uid,
        tags=cfg.dashboard_tags,
        refresh=cfg.refresh_interval,
        time_from=cfg.dashboard_time_from,
        time_to=cfg.dashboard_time_to,
        bundles=bundles,
        datasource_uid=cfg.datasource_uid,
    )
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
            logger.debug(
                "grafana provisioning failed",
                extra={"status": resp.status_code, "text": resp.text[:256]},
            )
            return GrafanaProvisionResult(ok=False, url=None, status_code=resp.status_code, error="http_error")
        body = resp.json()
        slug = body.get("slug")
        if slug:
            return GrafanaProvisionResult(
                ok=True,
                url=cfg.url.rstrip("/") + "/d/" + slug,
                status_code=resp.status_code,
                error=None,
            )
        return GrafanaProvisionResult(ok=True, url=None, status_code=resp.status_code, error=None)
    except Exception:  # pragma: no cover - defensive network guard
        logger.debug("grafana provisioning raised", exc_info=True)
        return GrafanaProvisionResult(ok=False, url=None, status_code=None, error="exception")


@dataclass(slots=True)
class PrometheusQueryHelper:
    """Cold-path helper for Prometheus instant queries."""

    base_url: str
    timeout_seconds: float = 2.5
    max_attempts: int = 3
    initial_delay: float = 0.2
    max_delay: float = 1.0
    jitter: float = 0.1

    def _query_endpoint(self) -> str:
        return urljoin(self.base_url.rstrip("/") + "/", "api/v1/query")

    def instant_value(self, query: str) -> float | None:
        """Execute an instant PromQL query and return the scalar value if available."""

        def _call() -> Mapping[str, Any]:
            resp = requests.get(
                self._query_endpoint(),
                params={"query": query},
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
            return cast(Mapping[str, Any], resp.json())

        try:
            payload = retry_with_backoff(
                _call,
                max_attempts=self.max_attempts,
                initial_delay=self.initial_delay,
                max_delay=self.max_delay,
                jitter=self.jitter,
            )
        except Exception:
            logger.debug("prometheus query failed", extra={"query": query}, exc_info=True)
            return None

        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, Mapping):
            return None
        result = data.get("result")
        if not isinstance(result, list) or not result:
            return None
        first = result[0]
        if not isinstance(first, Mapping):
            return None
        value = first.get("value")
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        raw_value = value[1]
        try:
            return float(raw_value)
        except Exception:
            return None

    def collect_scalars(self, queries: Mapping[str, str]) -> dict[str, float | None]:
        """Return a mapping of key → scalar result for the provided queries."""
        return {name: self.instant_value(expr) for name, expr in queries.items()}


__all__ = [
    "GrafanaConfig",
    "GrafanaPanelBundle",
    "GrafanaProvisionResult",
    "PrometheusQueryHelper",
    "build_dashboard",
    "default_panel_bundles",
    "provision_dashboard",
]
