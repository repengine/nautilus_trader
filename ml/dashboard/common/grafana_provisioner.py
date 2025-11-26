"""
Grafana provisioner component for Dashboard service.

Extracted from DashboardService to follow single-responsibility principle.
Handles Grafana dashboard provisioning, status checks, and Prometheus queries.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.dashboard.grafana import GrafanaConfig
from ml.dashboard.grafana import GrafanaProvisionResult
from ml.dashboard.grafana import PrometheusQueryHelper
from ml.dashboard.grafana import default_panel_bundles
from ml.dashboard.grafana import provision_dashboard


if TYPE_CHECKING:
    from ml.dashboard.config import DashboardConfig


logger = logging.getLogger(__name__)


_REQS_TOTAL = get_counter(
    "ml_dashboard_requests_total",
    "Total dashboard API requests",
    labels=["route", "method", "status"],
)
_LATENCY_SECONDS = get_histogram(
    "ml_dashboard_latency_seconds",
    "Dashboard API latency (seconds)",
    labels=["route"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


class GrafanaProvisionerProtocol(Protocol):
    """Protocol for Grafana provisioning and observability operations."""

    def provision_grafana_dashboard(
        self, *, title: str | None = None, force: bool = False
    ) -> dict[str, Any]:
        """Provision Grafana dashboard with optional title override and force flag."""
        ...

    def get_grafana_status(self) -> dict[str, Any]:
        """Get current Grafana dashboard status and URLs."""
        ...

    def get_prometheus_summary(self) -> dict[str, Any]:
        """Get summary of Prometheus metrics for dashboard monitoring."""
        ...


@dataclass(slots=True)
class _GrafanaStatus:
    """
    Track Grafana provisioning attempts and cache status.

    This internal state prevents redundant provisioning requests and stores
    the last known provisioning result.
    """

    ok: bool = False
    url: str | None = None
    status_code: int | None = None
    error: str | None = None
    last_attempt_epoch: float | None = None


@dataclass
class GrafanaProvisionerComponent:
    """
    Component for Grafana dashboard provisioning and observability queries.

    Extracted from DashboardService to follow single-responsibility principle.
    Responsible for:
    - Provisioning Grafana dashboards via API
    - Checking Grafana provisioning status
    - Querying Prometheus metrics for dashboard summaries

    Uses caching to avoid redundant provisioning requests and tracks the last
    known provisioning status.
    """

    config: DashboardConfig
    _grafana_status: _GrafanaStatus = field(init=False, repr=False, default_factory=_GrafanaStatus)
    _prometheus_helper: PrometheusQueryHelper | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Initialize Prometheus helper if URL is configured."""
        if self.config.prometheus_url:
            self._prometheus_helper = PrometheusQueryHelper(
                base_url=self.config.prometheus_url,
                timeout_seconds=self.config.prometheus_query_timeout_seconds,
            )

    def provision_grafana_dashboard(
        self, *, title: str | None = None, force: bool = False
    ) -> dict[str, Any]:
        """
        Provision a Grafana dashboard with optional title override.

        Uses caching to avoid redundant provisioning unless force=True.
        Records metrics for monitoring provisioning success/failure.

        Args:
            title: Optional dashboard title override (uses config default if None)
            force: Force reprovisioning even if cached status is OK

        Returns:
            Dictionary with provisioning result containing:
            - ok: Whether provisioning succeeded
            - url: Dashboard URL (if successful)
            - status_code: HTTP status code from Grafana API
            - error: Error message (if failed)
            - cached: Whether result came from cache (only present if cached)

        Example:
            >>> provisioner = GrafanaProvisionerComponent(config)
            >>> result = provisioner.provision_grafana_dashboard(force=True)
            >>> assert result["ok"] is True
            >>> assert "url" in result
        """
        route = "/api/observability/grafana/provision"
        start = time.perf_counter()
        try:
            cfg = self._build_grafana_config()
            # Check cache if not forcing and previous attempt succeeded
            if (
                not force
                and self._grafana_status.ok
                and self._grafana_status.url is not None
                and (title is None or title == cfg.dashboard_title)
            ):
                _REQS_TOTAL.labels(route=route, method="POST", status="cached").inc()
                return {
                    "ok": True,
                    "url": self._grafana_status.url,
                    "cached": True,
                    "status_code": self._grafana_status.status_code,
                    "error": self._grafana_status.error,
                }

            # Provision dashboard via Grafana API
            result: GrafanaProvisionResult = provision_dashboard(
                cfg,
                overwrite=True,
                title=title,
                bundles=default_panel_bundles(),
            )
            resolved_url = result.url or (
                self.config.grafana_dashboard_url() if result.ok else None
            )
            status_label = "success" if result.ok else "error"
            _REQS_TOTAL.labels(route=route, method="POST", status=status_label).inc()

            # Update cached status
            self._grafana_status = _GrafanaStatus(
                ok=result.ok,
                url=resolved_url,
                status_code=result.status_code,
                error=result.error,
                last_attempt_epoch=time.time(),
            )
            return {
                "ok": result.ok,
                "url": resolved_url,
                "status_code": result.status_code,
                "error": result.error,
            }
        except Exception:
            logger.debug("grafana provisioning error", exc_info=True)
            _REQS_TOTAL.labels(route=route, method="POST", status="exception").inc()

            # Preserve URL from previous successful attempt if available
            self._grafana_status = _GrafanaStatus(
                ok=False,
                url=self._grafana_status.url,
                status_code=None,
                error="exception",
                last_attempt_epoch=time.time(),
            )
            return {"ok": False, "url": self._grafana_status.url, "error": "exception"}
        finally:
            _LATENCY_SECONDS.labels(route=route).observe(time.perf_counter() - start)

    def get_grafana_status(self) -> dict[str, Any]:
        """
        Get current Grafana dashboard provisioning status.

        Returns the cached provisioning status from the last attempt, including
        dashboard URLs and embed URLs configured in the dashboard config.

        Returns:
            Dictionary with status information containing:
            - ok: Whether last provisioning succeeded
            - url: Dashboard URL (from cache or config default)
            - status_code: HTTP status code from last provisioning
            - error: Error message from last provisioning (if failed)
            - last_attempt_epoch: Unix timestamp of last provisioning attempt
            - embed_urls: List of embed panel URLs (if configured)

        Example:
            >>> provisioner = GrafanaProvisionerComponent(config)
            >>> status = provisioner.get_grafana_status()
            >>> assert "ok" in status
            >>> assert "url" in status
            >>> assert "embed_urls" in status
        """
        status = self._grafana_status
        return {
            "ok": status.ok,
            "url": status.url or self.config.grafana_dashboard_url(),
            "status_code": status.status_code,
            "error": status.error,
            "last_attempt_epoch": status.last_attempt_epoch,
            "embed_urls": self.config.grafana_embed_urls(),
        }

    def get_prometheus_summary(self) -> dict[str, Any]:
        """
        Get summary of Prometheus metrics for dashboard monitoring.

        Queries Prometheus for key dashboard metrics including request rate,
        latency percentiles, and event failures. Requires Prometheus URL to be
        configured.

        Returns:
            Dictionary with Prometheus query results containing:
            - ok: Whether queries succeeded
            - metrics: Dictionary of metric name to value (if ok=True)
            - updated_at: Unix timestamp when metrics were collected (if ok=True)
            - reason: Reason for failure (if ok=False) - "disabled" or "error"

        Example:
            >>> provisioner = GrafanaProvisionerComponent(config)
            >>> summary = provisioner.get_prometheus_summary()
            >>> assert summary["ok"] is True
            >>> assert "request_rate_per_second" in summary["metrics"]
        """
        route = "/api/observability/summary"
        start = time.perf_counter()
        try:
            helper = self._prometheus_helper
            if helper is None:
                _REQS_TOTAL.labels(route=route, method="GET", status="disabled").inc()
                return {"ok": False, "metrics": {}, "reason": "disabled"}

            # Collect key dashboard metrics from Prometheus
            metrics = helper.collect_scalars(
                {
                    "request_rate_per_second": "sum(rate(ml_dashboard_requests_total[5m]))",
                    "latency_p95_seconds": "histogram_quantile(0.95, sum(rate(ml_dashboard_latency_seconds_bucket[5m])) by (le))",
                    "event_failures_increase": "sum(increase(ml_dashboard_events_failure_total[5m]))",
                },
            )
            _REQS_TOTAL.labels(route=route, method="GET", status="success").inc()
            return {"ok": True, "metrics": metrics, "updated_at": time.time()}
        except Exception:
            logger.debug("prometheus summary failed", exc_info=True)
            _REQS_TOTAL.labels(route=route, method="GET", status="error").inc()
            return {"ok": False, "metrics": {}, "reason": "error"}
        finally:
            _LATENCY_SECONDS.labels(route=route).observe(time.perf_counter() - start)

    def _build_grafana_config(self) -> GrafanaConfig:
        """Build GrafanaConfig from DashboardConfig fields."""
        return GrafanaConfig(
            url=self.config.grafana_url,
            api_token=self.config.grafana_api_token,
            username=self.config.grafana_username,
            password=self.config.grafana_password,
            folder_uid=self.config.grafana_folder_uid,
            datasource_uid=self.config.grafana_datasource_uid,
            dashboard_uid=self.config.grafana_dashboard_uid,
            dashboard_title=self.config.grafana_dashboard_title,
            refresh_interval=self.config.grafana_refresh_interval,
        )


__all__ = [
    "GrafanaProvisionerComponent",
    "GrafanaProvisionerProtocol",
]
