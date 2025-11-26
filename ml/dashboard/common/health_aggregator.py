"""
Health aggregator component for Dashboard service.

Extracted from DashboardService to follow single-responsibility principle.
Aggregates system health from services, dependencies, and stores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urljoin

import requests

from ml.common.retry_utils import retry_with_backoff


if TYPE_CHECKING:
    from ml.dashboard.config import DashboardConfig


logger = logging.getLogger(__name__)


class HealthAggregatorProtocol(Protocol):
    """Protocol for health aggregation operations."""

    def get_system_health(self) -> dict[str, Any]:
        """Get aggregated system health status."""
        ...

    def list_services(self) -> list[dict[str, Any]]:
        """List all services with their health status."""
        ...

    def get_store_summary(self) -> dict[str, Any]:
        """Get summary of all store health metrics."""
        ...


def _safe_get(url: str, timeout: float) -> tuple[bool, int]:
    """Perform GET request with retry and return (ok, status_code)."""

    def _call() -> tuple[bool, int]:
        resp = requests.get(url, timeout=timeout)
        return resp.ok, resp.status_code

    try:
        return retry_with_backoff(
            _call,
            max_attempts=3,
            initial_delay=0.1,
            max_delay=0.5,
            jitter=0.2,
        )
    except Exception:
        logger.debug("health probe request failed", extra={"url": url}, exc_info=True)
        return False, 0


def _to_url(host_port: int, path: str, service_name: str | None = None) -> str:
    """Build URL for service endpoint with optional Docker networking support."""
    import os

    # Check for service-specific URL environment variable first (for Docker networking)
    if service_name:
        service_url_map = {
            "ml_signal_actor": os.getenv("ML_SIGNAL_ACTOR_URL"),
            "ml_strategy": os.getenv("ML_STRATEGY_URL"),
            "ml_pipeline": os.getenv("ML_PIPELINE_URL"),
        }
        service_url = service_url_map.get(service_name)
        if service_url:
            return urljoin(service_url.rstrip("/") + "/", path.lstrip("/"))
    return urljoin(f"http://localhost:{host_port}/", path.lstrip("/"))


@dataclass
class HealthAggregatorComponent:
    """
    Component for aggregating system health metrics.

    Extracted from DashboardService to follow single-responsibility principle.
    Responsible for probing service endpoints and aggregating health status.
    """

    config: DashboardConfig

    def get_system_health(self) -> dict[str, Any]:
        """
        Aggregate health across core services and dependencies.

        This is a read-only aggregation that pings known endpoints for liveness.

        Returns:
            Dictionary containing health status for services and dependencies.

        Example:
            >>> aggregator = HealthAggregatorComponent(config)
            >>> health = aggregator.get_system_health()
            >>> assert "services" in health
            >>> assert "dependencies" in health
        """
        cfg = self.config
        health: dict[str, Any] = {
            "services": {},
            "dependencies": {},
        }

        # ML services
        for name, port, hpath in (
            ("ml_signal_actor", cfg.actor_port, "/health"),
            ("ml_strategy", cfg.strategy_port, "/health"),
            ("ml_pipeline", cfg.pipeline_port, "/health"),
        ):
            ok, code = _safe_get(
                _to_url(port, hpath, service_name=name), cfg.request_timeout_seconds
            )
            health["services"][name] = {"healthy": ok, "status_code": code}

        # Observability dependencies
        prom_health_url = urljoin(self.config.prometheus_url.rstrip("/") + "/", "-/healthy")
        ok_prom, code_prom = _safe_get(prom_health_url, cfg.request_timeout_seconds)
        grafana_health_url = urljoin(self.config.grafana_url.rstrip("/") + "/", "api/health")
        ok_graf, code_graf = _safe_get(grafana_health_url, cfg.request_timeout_seconds)

        health["dependencies"]["prometheus"] = {"healthy": ok_prom, "status_code": code_prom}
        health["dependencies"]["grafana"] = {"healthy": ok_graf, "status_code": code_graf}

        return health

    def list_services(self) -> list[dict[str, Any]]:
        """
        List all services with their endpoint information.

        Returns:
            List of service dictionaries with name, ports, and endpoints.

        Example:
            >>> aggregator = HealthAggregatorComponent(config)
            >>> services = aggregator.list_services()
            >>> assert len(services) == 3
            >>> assert services[0]["name"] == "ml_signal_actor"
        """
        cfg = self.config
        return [
            {
                "name": "ml_signal_actor",
                "ports": {"http": cfg.actor_port},
                "endpoints": {
                    "health": _to_url(cfg.actor_port, "/health", service_name="ml_signal_actor"),
                    "metrics": _to_url(cfg.actor_port, "/metrics", service_name="ml_signal_actor"),
                },
            },
            {
                "name": "ml_strategy",
                "ports": {"http": cfg.strategy_port},
                "endpoints": {
                    "health": _to_url(cfg.strategy_port, "/health", service_name="ml_strategy"),
                    "metrics": _to_url(cfg.strategy_port, "/metrics", service_name="ml_strategy"),
                },
            },
            {
                "name": "ml_pipeline",
                "ports": {"http": cfg.pipeline_port},
                "endpoints": {
                    "health": _to_url(cfg.pipeline_port, "/health", service_name="ml_pipeline"),
                    "metrics": _to_url(cfg.pipeline_port, "/metrics", service_name="ml_pipeline"),
                },
            },
        ]

    def get_store_summary(self) -> dict[str, Any]:
        """
        Get summary of all store health metrics.

        This method aggregates health information from all stores (feature, model, strategy).
        Respects the store_health_enabled config flag.

        Returns:
            Dictionary with store health summaries and metadata.

        Example:
            >>> aggregator = HealthAggregatorComponent(config)
            >>> summary = aggregator.get_store_summary()
            >>> assert "ok" in summary
            >>> assert "stores" in summary
        """
        # Import store health utilities
        try:
            from ml.core.db_engine import EngineManager
            from ml.dashboard.store_health import summarize_all_stores
        except Exception:
            logger.debug("store health imports failed", exc_info=True)
            return {"ok": False, "stores": [], "reason": "import_error"}

        if not self.config.store_health_enabled:
            return {"ok": False, "stores": [], "reason": "disabled"}

        if not self.config.db_connection:
            return {"ok": False, "stores": [], "reason": "no_db_connection"}

        # Get database engine
        try:
            engine = EngineManager.get_engine(self.config.db_connection)
        except Exception:
            logger.debug("dashboard db engine unavailable", exc_info=True)
            engine = None

        # Initialize store clients lazily
        feature_store = None
        model_store = None
        strategy_store = None

        try:
            from ml.stores.feature_store import FeatureStore

            feature_store = FeatureStore(self.config.db_connection, enable_publishing=False)
        except Exception:
            logger.debug("feature store init failed", exc_info=True)

        try:
            from ml.stores.model_store import ModelStore

            model_store = ModelStore(self.config.db_connection, enable_publishing=False)
        except Exception:
            logger.debug("model store init failed", exc_info=True)

        try:
            from ml.stores.strategy_store import StrategyStore

            strategy_store = StrategyStore(self.config.db_connection, enable_publishing=False)
        except Exception:
            logger.debug("strategy store init failed", exc_info=True)

        # Aggregate store summaries
        try:
            import datetime as dt
            from datetime import datetime

            summaries = summarize_all_stores(
                feature_store=feature_store,
                model_store=model_store,
                strategy_store=strategy_store,
                engine=engine,
                top_dataset_limit=self.config.store_health_top_datasets,
            )

            payload = {
                "ok": True,
                "generated_at": datetime.now(dt.UTC).isoformat(),
                "stores": [summary.as_dict() for summary in summaries],
            }
            return payload
        except Exception:
            logger.debug("store summary collection failed", exc_info=True)
            return {"ok": False, "stores": [], "reason": "error"}


__all__ = [
    "HealthAggregatorComponent",
    "HealthAggregatorProtocol",
]
