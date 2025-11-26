"""
Dashboard service facade implementation for gradual migration.

Wires all 8 components together and provides feature flag support for progressive rollout.
Maintains exact API compatibility with legacy DashboardService.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ml.dashboard.common import AuthenticationComponent
from ml.dashboard.common import EventPollingComponent
from ml.dashboard.common import GrafanaProvisionerComponent
from ml.dashboard.common import HealthAggregatorComponent
from ml.dashboard.common import MetricsCollectorComponent
from ml.dashboard.common import PipelineIntegrationComponent
from ml.dashboard.common import RegistryManagerComponent
from ml.dashboard.common import ServiceControllerComponent
from ml.dashboard.controllers import ComposeServiceController
from ml.dashboard.controllers import NoopServiceController
from ml.dashboard.controllers import ServiceControllerProtocol
from ml.dashboard.metrics_snapshot import DashboardMetricsSnapshot
from ml.dashboard.metrics_snapshot import DashboardSuccessReport


if TYPE_CHECKING:
    from ml.core.integration import MLIntegrationManager
    from ml.dashboard.config import DashboardConfig


logger = logging.getLogger(__name__)


@dataclass(init=False)
class DashboardServiceFacade:
    """
    Dashboard service facade wiring all 8 components together.

    Provides exact API compatibility with legacy DashboardService while delegating
    to single-responsibility components. Supports gradual migration via feature flag:
    ML_USE_LEGACY_DASHBOARD_SERVICE.

    Architecture:
    - HealthAggregatorComponent: System health, service listing, store summary
    - RegistryManagerComponent: All 13 registry operations with caching
    - GrafanaProvisionerComponent: Dashboard provisioning, status, Prometheus summary
    - MetricsCollectorComponent: Metrics snapshot, success criteria evaluation
    - PipelineIntegrationComponent: All 10 pipeline operations
    - ServiceControllerComponent: Service control (start/stop/restart)
    - EventPollingComponent: Event listing and background polling
    - AuthenticationComponent: Token validation

    All methods delegate to appropriate components - zero business logic in facade.
    """

    config: DashboardConfig
    controller: ServiceControllerProtocol

    # Component instances (lazy-initialized)
    _health_aggregator: HealthAggregatorComponent | None = field(default=None, init=False, repr=False)
    _registry_manager: RegistryManagerComponent | None = field(default=None, init=False, repr=False)
    _grafana_provisioner: GrafanaProvisionerComponent | None = field(default=None, init=False, repr=False)
    _metrics_collector: MetricsCollectorComponent | None = field(default=None, init=False, repr=False)
    _pipeline_integration: PipelineIntegrationComponent | None = field(default=None, init=False, repr=False)
    _service_controller: ServiceControllerComponent | None = field(default=None, init=False, repr=False)
    _event_polling: EventPollingComponent | None = field(default=None, init=False, repr=False)
    _authentication: AuthenticationComponent | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: DashboardConfig) -> DashboardServiceFacade:
        """
        Create facade instance from configuration.

        Args:
            config: Dashboard configuration

        Returns:
            Configured DashboardServiceFacade instance
        """
        controller: ServiceControllerProtocol
        if config.compose_enabled:
            controller = ComposeServiceController(config.compose_file)
        else:
            controller = NoopServiceController()
        facade = cls(config=config, controller=controller)

        # Provision Grafana on startup if configured
        if config.grafana_provision_on_start:
            try:
                facade.provision_grafana_dashboard(title=config.grafana_dashboard_title, force=True)
            except Exception:
                logger.debug("initial grafana provisioning failed", exc_info=True)
        return facade

    def __init__(self, config: DashboardConfig, controller: ServiceControllerProtocol) -> None:
        """
        Initialize facade with configuration and controller.

        Args:
            config: Dashboard configuration
            controller: Service controller for start/stop/restart operations
        """
        self.config = config
        self.controller = controller
        self._health_aggregator = None
        self._registry_manager = None
        self._grafana_provisioner = None
        self._metrics_collector = None
        self._pipeline_integration = None
        self._service_controller = None
        self._event_polling = None
        self._authentication = None

    # -----------------
    # Component Getters (Lazy Initialization)
    # -----------------
    def _get_health_aggregator(self) -> HealthAggregatorComponent:
        """Get or create health aggregator component."""
        if self._health_aggregator is None:
            self._health_aggregator = HealthAggregatorComponent(config=self.config)
        return self._health_aggregator

    def _get_registry_manager(self) -> RegistryManagerComponent:
        """Get or create registry manager component."""
        if self._registry_manager is None:
            self._registry_manager = RegistryManagerComponent(config=self.config)
        return self._registry_manager

    def _get_grafana_provisioner(self) -> GrafanaProvisionerComponent:
        """Get or create Grafana provisioner component."""
        if self._grafana_provisioner is None:
            self._grafana_provisioner = GrafanaProvisionerComponent(config=self.config)
        return self._grafana_provisioner

    def _get_metrics_collector(self) -> MetricsCollectorComponent:
        """Get or create metrics collector component."""
        if self._metrics_collector is None:
            # Import metrics from dashboard.service module
            from ml.dashboard.service import _EVENT_CACHE_HITS
            from ml.dashboard.service import _EVENT_CACHE_MISSES
            from ml.dashboard.service import _REGISTRY_CACHE_HITS
            from ml.dashboard.service import _REGISTRY_CACHE_MISSES
            from ml.dashboard.service import _REGISTRY_LATENCY_SECONDS
            from ml.dashboard.service import _REQS_TOTAL
            from ml.dashboard.service import _STORE_SUMMARY_SECONDS

            self._metrics_collector = MetricsCollectorComponent(
                config=self.config,
                registry_cache_hits=_REGISTRY_CACHE_HITS,
                registry_cache_misses=_REGISTRY_CACHE_MISSES,
                registry_histogram=_REGISTRY_LATENCY_SECONDS,
                event_cache_hits=_EVENT_CACHE_HITS,
                event_cache_misses=_EVENT_CACHE_MISSES,
                request_counter=_REQS_TOTAL,
                store_histogram=_STORE_SUMMARY_SECONDS,
            )
        return self._metrics_collector

    def _get_pipeline_integration(self) -> PipelineIntegrationComponent:
        """Get or create pipeline integration component."""
        if self._pipeline_integration is None:
            self._pipeline_integration = PipelineIntegrationComponent(config=self.config)
        return self._pipeline_integration

    def _get_service_controller(self) -> ServiceControllerComponent:
        """Get or create service controller component."""
        if self._service_controller is None:
            self._service_controller = ServiceControllerComponent(controller=self.controller)
        return self._service_controller

    def _get_event_polling(self) -> EventPollingComponent:
        """Get or create event polling component."""
        if self._event_polling is None:
            self._event_polling = EventPollingComponent(
                ttl_seconds=self.config.events_cache_ttl_seconds,
                max_entries=self.config.events_cache_max_entries,
            )
        return self._event_polling

    def _get_authentication(self) -> AuthenticationComponent:
        """Get or create authentication component."""
        if self._authentication is None:
            self._authentication = AuthenticationComponent(tokens=self.config.auth_tokens)
        return self._authentication

    # -----------------
    # Health & Metadata (3 methods) - HealthAggregatorComponent
    # -----------------
    def get_system_health(self) -> dict[str, Any]:
        """
        Aggregate health across core services and dependencies.

        Returns:
            Dictionary containing health status for services and dependencies
        """
        return self._get_health_aggregator().get_system_health()

    def list_services(self) -> list[dict[str, Any]]:
        """
        List all services with their health status.

        Returns:
            List of service metadata with ports and endpoints
        """
        return self._get_health_aggregator().list_services()

    def get_store_summary(self) -> dict[str, Any]:
        """
        Get summary of all store health metrics.

        Returns:
            Dictionary containing store health summaries and metadata
        """
        return self._get_health_aggregator().get_store_summary()

    # -----------------
    # Registry Operations (13 methods) - RegistryManagerComponent
    # -----------------
    def list_models(self) -> list[dict[str, Any]]:
        """
        List all registered models.

        Returns:
            List of model metadata dictionaries
        """
        return self._get_registry_manager().list_models()

    def get_model_performance_history(
        self,
        model_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get performance history for a specific model.

        Args:
            model_id: Model identifier
            limit: Maximum number of history entries to return

        Returns:
            List of performance history entries
        """
        return self._get_registry_manager().get_model_performance_history(model_id, limit=limit)

    def list_deployments(self) -> dict[str, list[str]]:
        """
        List active model deployments.

        Returns:
            Dictionary mapping deployment targets to model IDs
        """
        return self._get_registry_manager().list_deployments()

    def list_features(
        self,
        role: str | None = None,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List feature sets with optional filtering.

        Args:
            role: Filter by feature role (optional)
            stage: Filter by feature stage (optional)

        Returns:
            List of feature set metadata dictionaries
        """
        return self._get_registry_manager().list_features(role=role, stage=stage)

    def get_feature_lineage(self, feature_set_id: str) -> list[dict[str, Any]]:
        """
        Get lineage information for a feature set.

        Args:
            feature_set_id: Feature set identifier

        Returns:
            List of lineage entries (parent/child relationships)
        """
        return self._get_registry_manager().get_feature_lineage(feature_set_id)

    def list_strategies(self) -> list[dict[str, Any]]:
        """
        List all registered strategies.

        Returns:
            List of strategy metadata dictionaries
        """
        return self._get_registry_manager().list_strategies()

    def get_strategy_details(self, strategy_id: str) -> dict[str, Any] | None:
        """
        Get detailed information for a specific strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Strategy details dictionary or None if not found
        """
        return self._get_registry_manager().get_strategy_details(strategy_id)

    def check_strategy_compatibility(
        self,
        strategy_id: str,
        active: list[str],
    ) -> dict[str, Any]:
        """
        Check if a strategy is compatible with active strategies.

        Args:
            strategy_id: Strategy identifier to check
            active: List of currently active strategy IDs

        Returns:
            Dictionary containing compatibility status
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
        Promote a feature set to a new stage.

        Args:
            feature_set_id: Feature set identifier
            stage: Target stage (defaults to PROD)
            gates: Quality gates to validate before promotion (optional)

        Returns:
            Dictionary containing promotion result
        """
        return self._get_registry_manager().promote_feature(
            feature_set_id,
            stage=stage,
            gates=gates,
        )

    def deprecate_feature(
        self,
        feature_set_id: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Deprecate a feature set.

        Args:
            feature_set_id: Feature set identifier
            reason: Deprecation reason (optional)

        Returns:
            Dictionary containing deprecation result
        """
        return self._get_registry_manager().deprecate_feature(feature_set_id, reason=reason)

    def list_datasets(self) -> list[dict[str, Any]]:
        """
        List all registered datasets.

        Returns:
            List of dataset metadata dictionaries
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
        List watermarks for a dataset.

        Args:
            dataset_id: Dataset identifier
            instrument: Filter by instrument ID (optional)
            source: Filter by source (optional)
            limit: Maximum number of watermarks to return

        Returns:
            List of watermark entries
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
        List dataset lineage entries.

        Args:
            child: Filter by child dataset ID (optional)
            parent: Filter by parent dataset ID (optional)
            limit: Maximum number of lineage entries to return

        Returns:
            List of dataset lineage records
        """
        return self._get_registry_manager().list_dataset_lineage(
            child=child,
            parent=parent,
            limit=limit,
        )

    # -----------------
    # Grafana & Observability (3 methods) - GrafanaProvisionerComponent
    # -----------------
    def provision_grafana_dashboard(
        self,
        *,
        title: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Provision Grafana dashboard.

        Args:
            title: Dashboard title (optional, uses config default)
            force: Force re-provisioning even if cached

        Returns:
            Dictionary containing provisioning result with URL and status
        """
        return self._get_grafana_provisioner().provision_grafana_dashboard(
            title=title,
            force=force,
        )

    def get_grafana_status(self) -> dict[str, Any]:
        """
        Get Grafana dashboard status.

        Returns:
            Dictionary containing Grafana status and URLs
        """
        return self._get_grafana_provisioner().get_grafana_status()

    def get_prometheus_summary(self) -> dict[str, Any]:
        """
        Get Prometheus metrics summary.

        Returns:
            Dictionary containing aggregated Prometheus metrics
        """
        return self._get_grafana_provisioner().get_prometheus_summary()

    # -----------------
    # Metrics & Success Criteria (2 methods) - MetricsCollectorComponent
    # -----------------
    def get_metrics_snapshot(self) -> DashboardMetricsSnapshot:
        """
        Get current metrics snapshot.

        Returns:
            DashboardMetricsSnapshot containing aggregated metrics
        """
        return self._get_metrics_collector().get_metrics_snapshot()

    def evaluate_success_criteria(self) -> DashboardSuccessReport:
        """
        Evaluate dashboard success criteria.

        Returns:
            DashboardSuccessReport with success status and violations
        """
        return self._get_metrics_collector().evaluate_success_criteria()

    # -----------------
    # Pipeline Operations (10 methods) - PipelineIntegrationComponent
    # -----------------
    def trigger_pipeline(
        self,
        pipeline_type: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Trigger a pipeline execution.

        Args:
            pipeline_type: Type of pipeline to trigger
            config: Pipeline configuration parameters

        Returns:
            Dictionary containing job ID and status
        """
        return self._get_pipeline_integration().trigger_pipeline(pipeline_type, config)

    def trigger_orchestrator_task(
        self,
        task: str,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Trigger an orchestrator task.

        Args:
            task: Task name to execute
            config: Task configuration parameters (optional)

        Returns:
            Dictionary containing task execution result
        """
        return self._get_pipeline_integration().trigger_orchestrator_task(task, config)

    def list_pipeline_jobs(self) -> dict[str, Any]:
        """
        List all pipeline jobs.

        Returns:
            Dictionary containing list of pipeline jobs
        """
        return self._get_pipeline_integration().list_pipeline_jobs()

    def get_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """
        Get details for a specific pipeline job.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary containing job details
        """
        return self._get_pipeline_integration().get_pipeline_job(job_id)

    def purge_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """
        Purge a completed pipeline job.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary containing purge result
        """
        return self._get_pipeline_integration().purge_pipeline_job(job_id)

    def build_dataset_pipeline(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """
        Build a dataset pipeline.

        Args:
            config: Dataset build configuration

        Returns:
            Dictionary containing job ID and status
        """
        return self._get_pipeline_integration().build_dataset_pipeline(config)

    def train_model_pipeline(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """
        Train a model pipeline.

        Args:
            config: Model training configuration

        Returns:
            Dictionary containing job ID and status
        """
        return self._get_pipeline_integration().train_model_pipeline(config)

    def run_hpo_pipeline(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """
        Run hyperparameter optimization pipeline.

        Args:
            config: HPO configuration

        Returns:
            Dictionary containing job ID and status
        """
        return self._get_pipeline_integration().run_hpo_pipeline(config)

    def get_pipeline_progress(self, job_id: str) -> dict[str, Any]:
        """
        Get progress for a pipeline job.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary containing progress information
        """
        return self._get_pipeline_integration().get_pipeline_progress(job_id)

    def cancel_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """
        Cancel a running pipeline job.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary containing cancellation result
        """
        return self._get_pipeline_integration().cancel_pipeline_job(job_id)

    # -----------------
    # Service Control (1 method) - ServiceControllerComponent
    # -----------------
    def control_service(self, name: str, action: str) -> dict[str, Any]:
        """
        Control a service (start/stop/restart).

        Args:
            name: Service name
            action: Action to perform (start/stop/restart)

        Returns:
            Dictionary containing control action result
        """
        return self._get_service_controller().control_service(name, action)

    # -----------------
    # Event Polling (3 methods) - EventPollingComponent
    # -----------------
    def list_events(
        self,
        *,
        limit: int = 100,
        stage: str | None = None,
        source: str | None = None,
        instrument_substr: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List recent events from message bus.

        Args:
            limit: Maximum number of events to return
            stage: Filter by stage (optional)
            source: Filter by source (optional)
            instrument_substr: Filter by instrument substring (optional)

        Returns:
            List of event dictionaries
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
            interval_seconds: Polling interval in seconds
        """
        self._get_event_polling().start_event_polling(interval_seconds)

    def stop_event_polling(self) -> None:
        """Stop background event polling."""
        self._get_event_polling().stop_event_polling()

    # -----------------
    # Authentication (1 method) - AuthenticationComponent
    # -----------------
    def validate_token(
        self,
        provided: str | None,
        *,
        now: datetime | None = None,
    ) -> bool:
        """
        Validate authentication token.

        Args:
            provided: Token to validate
            now: Current time for validation (optional, defaults to now)

        Returns:
            True if token is valid, False otherwise
        """
        return self._get_authentication().validate_token(provided, now=now)

    # -----------------
    # Integration Manager Access (1 method) - PipelineIntegrationComponent
    # -----------------
    def get_integration_manager(self) -> MLIntegrationManager | None:
        """
        Get the MLIntegrationManager instance if available.

        Returns:
            MLIntegrationManager instance or None
        """
        return self._get_pipeline_integration().get_integration_manager()


__all__ = ["DashboardServiceFacade"]
