"""
Compatibility facade for dashboard service operations.

This facade preserves the legacy DashboardService API while delegating to the
extracted dashboard components with lazy initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ml.dashboard import service as dashboard_service
from ml.dashboard.common import AuthenticationComponent
from ml.dashboard.common import EventPollingComponent
from ml.dashboard.common import GrafanaProvisionerComponent
from ml.dashboard.common import HealthAggregatorComponent
from ml.dashboard.common import MetricsCollectorComponent
from ml.dashboard.common import PipelineIntegrationComponent
from ml.dashboard.common import RegistryManagerComponent
from ml.dashboard.common import ServiceControllerComponent
from ml.dashboard.config import DashboardConfig
from ml.dashboard.controllers import ComposeServiceController
from ml.dashboard.controllers import NoopServiceController
from ml.dashboard.controllers import ServiceControllerProtocol
from ml.dashboard.metrics_snapshot import DashboardMetricsSnapshot
from ml.dashboard.metrics_snapshot import DashboardSuccessReport
from ml.dashboard.service import DashboardService


@dataclass
class DashboardServiceFacade:
    """
    Facade delegating dashboard operations to component implementations.

    Attributes:
        config: Dashboard configuration.
        controller: Service controller used for start/stop/restart actions.
    """

    config: DashboardConfig
    controller: ServiceControllerProtocol
    _health_aggregator: HealthAggregatorComponent | None = None
    _registry_manager: RegistryManagerComponent | None = None
    _grafana_provisioner: GrafanaProvisionerComponent | None = None
    _metrics_collector: MetricsCollectorComponent | None = None
    _pipeline_integration: PipelineIntegrationComponent | None = None
    _service_controller: ServiceControllerComponent | None = None
    _event_polling: EventPollingComponent | None = None
    _authentication: AuthenticationComponent | None = None

    @classmethod
    def from_config(cls, config: DashboardConfig) -> DashboardServiceFacade:
        """
        Build a facade using the provided dashboard configuration.

        Args:
            config: Dashboard configuration.

        Returns:
            Initialized dashboard service facade.
        """
        controller: ServiceControllerProtocol
        if config.compose_enabled:
            controller = ComposeServiceController(config.compose_file)
        else:
            controller = NoopServiceController()
        return cls(config=config, controller=controller)

    def get_system_health(self) -> dict[str, Any]:
        """
        Return aggregated system health status.

        Returns:
            Health aggregation payload.
        """
        return self._get_health_aggregator().get_system_health()

    def list_services(self) -> list[dict[str, Any]]:
        """
        List registered services and endpoints.

        Returns:
            List of service metadata entries.
        """
        return self._get_health_aggregator().list_services()

    def get_store_summary(self) -> dict[str, Any]:
        """
        Return store health summary.

        Returns:
            Store health payload.
        """
        return self._get_health_aggregator().get_store_summary()

    def list_models(self) -> list[dict[str, Any]]:
        """
        List models from the registry manager.

        Returns:
            List of model metadata entries.
        """
        return self._get_registry_manager().list_models()

    def get_model_performance_history(self, model_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        """
        Fetch model performance history.

        Args:
            model_id: Model identifier.
            limit: Maximum number of history entries.

        Returns:
            List of performance history records.
        """
        return self._get_registry_manager().get_model_performance_history(model_id, limit=limit)

    def list_deployments(self) -> dict[str, list[str]]:
        """
        List model deployments by stage.

        Returns:
            Mapping of deployment stages to model IDs.
        """
        return self._get_registry_manager().list_deployments()

    def list_features(self, *, role: str | None = None, stage: str | None = None) -> list[dict[str, Any]]:
        """
        List feature sets with optional filters.

        Args:
            role: Optional role filter.
            stage: Optional stage filter.

        Returns:
            List of feature set metadata entries.
        """
        return self._get_registry_manager().list_features(role=role, stage=stage)

    def get_feature_lineage(self, feature_set_id: str) -> list[dict[str, Any]]:
        """
        Get lineage for a feature set.

        Args:
            feature_set_id: Feature set identifier.

        Returns:
            List of lineage records.
        """
        return self._get_registry_manager().get_feature_lineage(feature_set_id)

    def list_strategies(self) -> list[dict[str, Any]]:
        """
        List strategies from the registry manager.

        Returns:
            List of strategy metadata entries.
        """
        return self._get_registry_manager().list_strategies()

    def get_strategy_details(self, strategy_id: str) -> dict[str, Any] | None:
        """
        Fetch strategy details.

        Args:
            strategy_id: Strategy identifier.

        Returns:
            Strategy details or None if not found.
        """
        return self._get_registry_manager().get_strategy_details(strategy_id)

    def check_strategy_compatibility(
        self,
        strategy_id: str,
        active: list[str],
    ) -> dict[str, Any]:
        """
        Check compatibility for a strategy against active ones.

        Args:
            strategy_id: Strategy identifier.
            active: List of active strategy identifiers.

        Returns:
            Compatibility report.
        """
        return self._get_registry_manager().check_strategy_compatibility(strategy_id, active)

    def promote_feature(
        self,
        feature_set_id: str,
        *,
        stage: str | None = None,
        gates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Promote a feature set to a target stage.

        Args:
            feature_set_id: Feature set identifier.
            stage: Target stage override.
            gates: Optional gating configuration.

        Returns:
            Promotion result payload.
        """
        return self._get_registry_manager().promote_feature(feature_set_id, stage=stage, gates=gates)

    def deprecate_feature(self, feature_set_id: str, *, reason: str | None = None) -> dict[str, Any]:
        """
        Deprecate a feature set.

        Args:
            feature_set_id: Feature set identifier.
            reason: Optional deprecation reason.

        Returns:
            Deprecation result payload.
        """
        return self._get_registry_manager().deprecate_feature(feature_set_id, reason=reason)

    def list_datasets(self) -> list[dict[str, Any]]:
        """
        List datasets registered in the data registry.

        Returns:
            List of dataset metadata entries.
        """
        return self._get_registry_manager().list_datasets()

    def list_watermarks(
        self,
        *,
        dataset_id: str,
        instrument: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        List watermark records.

        Args:
            dataset_id: Dataset identifier.
            instrument: Optional instrument filter.
            source: Optional source filter.
            limit: Maximum number of entries.

        Returns:
            List of watermark metadata entries.
        """
        return self._get_registry_manager().list_watermarks(
            dataset_id=dataset_id,
            instrument=instrument,
            source=source,
            limit=limit,
        )

    def list_dataset_lineage(
        self,
        *,
        child: str | None = None,
        parent: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        List dataset lineage relationships.

        Args:
            child: Optional child dataset filter.
            parent: Optional parent dataset filter.
            limit: Maximum number of entries.

        Returns:
            List of lineage records.
        """
        return self._get_registry_manager().list_dataset_lineage(
            child=child,
            parent=parent,
            limit=limit,
        )

    def provision_grafana_dashboard(self, *, title: str | None = None, force: bool = False) -> dict[str, Any]:
        """
        Provision the Grafana dashboard.

        Args:
            title: Optional dashboard title override.
            force: Force reprovisioning.

        Returns:
            Provisioning result payload.
        """
        return self._get_grafana_provisioner().provision_grafana_dashboard(title=title, force=force)

    def get_grafana_status(self) -> dict[str, Any]:
        """
        Return Grafana provisioning status.

        Returns:
            Grafana status payload.
        """
        return self._get_grafana_provisioner().get_grafana_status()

    def get_prometheus_summary(self) -> dict[str, Any]:
        """
        Return Prometheus summary metrics.

        Returns:
            Prometheus summary payload.
        """
        return self._get_grafana_provisioner().get_prometheus_summary()

    def get_metrics_snapshot(self) -> DashboardMetricsSnapshot:
        """
        Return aggregated dashboard metrics.

        Returns:
            Dashboard metrics snapshot.
        """
        return self._get_metrics_collector().get_metrics_snapshot()

    def evaluate_success_criteria(self) -> DashboardSuccessReport:
        """
        Evaluate dashboard success criteria against metrics.

        Returns:
            Success criteria evaluation report.
        """
        return self._get_metrics_collector().evaluate_success_criteria()

    def trigger_pipeline(self, pipeline_type: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Trigger a pipeline job.

        Args:
            pipeline_type: Pipeline type identifier.
            config: Pipeline configuration payload.

        Returns:
            Job status payload.
        """
        return self._get_pipeline_integration().trigger_pipeline(pipeline_type, config)

    def trigger_orchestrator_task(self, task_name: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Trigger a specific orchestrator task.

        Args:
            task_name: Orchestrator task name.
            config: Task configuration payload.

        Returns:
            Task execution payload.
        """
        return self._get_pipeline_integration().trigger_orchestrator_task(task_name, config)

    def list_pipeline_jobs(self) -> dict[str, Any]:
        """
        List pipeline jobs.

        Returns:
            Pipeline job listing payload.
        """
        return self._get_pipeline_integration().list_pipeline_jobs()

    def get_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """
        Get pipeline job details.

        Args:
            job_id: Pipeline job identifier.

        Returns:
            Job detail payload.
        """
        return self._get_pipeline_integration().get_pipeline_job(job_id)

    def purge_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """
        Purge a pipeline job.

        Args:
            job_id: Pipeline job identifier.

        Returns:
            Purge result payload.
        """
        return self._get_pipeline_integration().purge_pipeline_job(job_id)

    def build_dataset_pipeline(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Trigger a dataset build pipeline.

        Args:
            config: Pipeline configuration payload.

        Returns:
            Job status payload.
        """
        return self._get_pipeline_integration().build_dataset_pipeline(config)

    def train_model_pipeline(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Trigger a model training pipeline.

        Args:
            config: Pipeline configuration payload.

        Returns:
            Job status payload.
        """
        return self._get_pipeline_integration().train_model_pipeline(config)

    def run_hpo_pipeline(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Trigger a hyperparameter optimization pipeline.

        Args:
            config: Pipeline configuration payload.

        Returns:
            Job status payload.
        """
        return self._get_pipeline_integration().run_hpo_pipeline(config)

    def get_pipeline_progress(self, job_id: str) -> dict[str, Any]:
        """
        Get pipeline job progress.

        Args:
            job_id: Pipeline job identifier.

        Returns:
            Job progress payload.
        """
        return self._get_pipeline_integration().get_pipeline_progress(job_id)

    def cancel_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """
        Cancel a running pipeline job.

        Args:
            job_id: Pipeline job identifier.

        Returns:
            Cancellation result payload.
        """
        return self._get_pipeline_integration().cancel_pipeline_job(job_id)

    def get_integration_manager(self) -> object | None:
        """
        Return the pipeline integration manager.

        Returns:
            Integration manager instance or None.
        """
        return self._get_pipeline_integration().get_integration_manager()

    def control_service(self, name: str, action: str) -> dict[str, Any]:
        """
        Control a dashboard-managed service.

        Args:
            name: Service name.
            action: Action to perform ("start", "stop", "restart").

        Returns:
            Control action result payload.
        """
        return self._get_service_controller().control_service(name, action)

    def list_events(
        self,
        *,
        limit: int = 100,
        stage: str | None = None,
        source: str | None = None,
        instrument_substr: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List events from the event polling component.

        Args:
            limit: Maximum number of events.
            stage: Optional stage filter.
            source: Optional source filter.
            instrument_substr: Optional instrument substring filter.

        Returns:
            List of event payloads.
        """
        return self._get_event_polling().list_events(
            limit=limit,
            stage=stage,
            source=source,
            instrument_substr=instrument_substr,
        )

    def start_event_polling(self, interval_seconds: float) -> None:
        """
        Start background event polling.

        Args:
            interval_seconds: Polling interval in seconds.
        """
        self._get_event_polling().start_event_polling(interval_seconds)

    def stop_event_polling(self) -> None:
        """
        Stop background event polling.
        """
        self._get_event_polling().stop_event_polling()

    def validate_token(self, token: str | None, *, now: datetime | None = None) -> bool:
        """
        Validate a dashboard token.

        Args:
            token: Token string.
            now: Optional timestamp override.

        Returns:
            True when the token is valid.
        """
        return self._get_authentication().validate_token(token, now=now)

    def _get_health_aggregator(self) -> HealthAggregatorComponent:
        if self._health_aggregator is None:
            self._health_aggregator = HealthAggregatorComponent(self.config)
        return self._health_aggregator

    def _get_registry_manager(self) -> RegistryManagerComponent:
        if self._registry_manager is None:
            self._registry_manager = RegistryManagerComponent(self.config)
        return self._registry_manager

    def _get_grafana_provisioner(self) -> GrafanaProvisionerComponent:
        if self._grafana_provisioner is None:
            self._grafana_provisioner = GrafanaProvisionerComponent(self.config)
        return self._grafana_provisioner

    def _get_metrics_collector(self) -> MetricsCollectorComponent:
        if self._metrics_collector is None:
            self._metrics_collector = MetricsCollectorComponent(
                config=self.config,
                registry_cache_hits=dashboard_service._REGISTRY_CACHE_HITS,
                registry_cache_misses=dashboard_service._REGISTRY_CACHE_MISSES,
                registry_histogram=dashboard_service._REGISTRY_LATENCY_SECONDS,
                event_cache_hits=dashboard_service._EVENT_CACHE_HITS,
                event_cache_misses=dashboard_service._EVENT_CACHE_MISSES,
                request_counter=dashboard_service._REQS_TOTAL,
                store_histogram=dashboard_service._STORE_SUMMARY_SECONDS,
            )
        return self._metrics_collector

    def _get_pipeline_integration(self) -> PipelineIntegrationComponent:
        if self._pipeline_integration is None:
            self._pipeline_integration = PipelineIntegrationComponent(self.config)
        return self._pipeline_integration

    def _get_service_controller(self) -> ServiceControllerComponent:
        if self._service_controller is None:
            self._service_controller = ServiceControllerComponent(self.controller)
        return self._service_controller

    def _get_event_polling(self) -> EventPollingComponent:
        if self._event_polling is None:
            self._event_polling = EventPollingComponent(
                ttl_seconds=self.config.events_cache_ttl_seconds,
                max_entries=self.config.events_cache_max_entries,
            )
        return self._event_polling

    def _get_authentication(self) -> AuthenticationComponent:
        if self._authentication is None:
            self._authentication = AuthenticationComponent(self.config.auth_tokens)
        return self._authentication


__all__ = [
    "DashboardService",
    "DashboardServiceFacade",
]
