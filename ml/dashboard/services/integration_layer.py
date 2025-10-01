"""Factory helpers for dashboard integration services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml.dashboard.services.actors_service import ActorIntegrationService
from ml.dashboard.services.base_service import BaseIntegrationService
from ml.dashboard.services.base_service import IntegrationContext
from ml.dashboard.services.metrics_service import StoreIntegrationService
from ml.dashboard.services.pipelines_service import PipelineIntegrationService
from ml.dashboard.services.system_service import SystemConnectorService
from ml.dashboard.services.trading_service import TradingIntegrationService


if TYPE_CHECKING:
    from ml.core.integration import MLIntegrationManager

_SERVICE_FACTORIES: dict[str, type[BaseIntegrationService]] = {
    "store": StoreIntegrationService,
    "actor": ActorIntegrationService,
    "pipeline": PipelineIntegrationService,
    "trading": TradingIntegrationService,
    "system": SystemConnectorService,
}

_integration_services: dict[str, BaseIntegrationService] = {}


def get_integration_service(
    service_type: str,
    integration_manager: MLIntegrationManager | None = None,
) -> BaseIntegrationService:
    """Return a singleton instance of the requested integration service."""
    try:
        service_class = _SERVICE_FACTORIES[service_type]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown service type: {service_type}") from exc

    service = _integration_services.get(service_type)
    if service is None:
        service = service_class(integration_manager)
        _integration_services[service_type] = service
    else:
        service.set_integration_manager(integration_manager)

    return service


__all__ = [
    "ActorIntegrationService",
    "BaseIntegrationService",
    "IntegrationContext",
    "PipelineIntegrationService",
    "StoreIntegrationService",
    "SystemConnectorService",
    "TradingIntegrationService",
    "get_integration_service",
]
