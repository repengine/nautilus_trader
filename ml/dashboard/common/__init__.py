"""Dashboard service components for single-responsibility decomposition."""

from __future__ import annotations

from ml.dashboard.common.authentication import AuthenticationComponent
from ml.dashboard.common.authentication import AuthenticationProtocol
from ml.dashboard.common.event_polling import EventPollingComponent
from ml.dashboard.common.event_polling import EventPollingProtocol
from ml.dashboard.common.grafana_provisioner import GrafanaProvisionerComponent
from ml.dashboard.common.grafana_provisioner import GrafanaProvisionerProtocol
from ml.dashboard.common.health_aggregator import HealthAggregatorComponent
from ml.dashboard.common.health_aggregator import HealthAggregatorProtocol
from ml.dashboard.common.metrics_collector import MetricsCollectorComponent
from ml.dashboard.common.metrics_collector import MetricsCollectorProtocol
from ml.dashboard.common.pipeline_integration import PipelineIntegrationComponent
from ml.dashboard.common.pipeline_integration import PipelineIntegrationProtocol
from ml.dashboard.common.registry_manager import RegistryManagerComponent
from ml.dashboard.common.registry_manager import RegistryManagerProtocol
from ml.dashboard.common.service_controller import ServiceControllerComponent
from ml.dashboard.common.service_controller import ServiceControllerComponentProtocol


__all__ = [
    "AuthenticationComponent",
    "AuthenticationProtocol",
    "EventPollingComponent",
    "EventPollingProtocol",
    "GrafanaProvisionerComponent",
    "GrafanaProvisionerProtocol",
    "HealthAggregatorComponent",
    "HealthAggregatorProtocol",
    "MetricsCollectorComponent",
    "MetricsCollectorProtocol",
    "PipelineIntegrationComponent",
    "PipelineIntegrationProtocol",
    "RegistryManagerComponent",
    "RegistryManagerProtocol",
    "ServiceControllerComponent",
    "ServiceControllerComponentProtocol",
]
