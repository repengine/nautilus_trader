"""
Unit tests for DashboardServiceFacade.

Tests verify that the facade correctly delegates to all 8 components
and maintains exact API compatibility with legacy DashboardService.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone, UTC
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.dashboard.config import DashboardConfig
from ml.dashboard.controllers import NoopServiceController
from ml.dashboard.metrics_snapshot import DashboardMetricsSnapshot
from ml.dashboard.metrics_snapshot import DashboardSuccessReport
from ml.dashboard.service_facade import DashboardServiceFacade


@pytest.fixture
def dashboard_config() -> DashboardConfig:
    """Create minimal dashboard configuration for testing."""
    return DashboardConfig(
        db_connection="",
        actor_port=8081,
        strategy_port=8082,
        pipeline_port=8083,
        prometheus_url="http://localhost:9090",
        grafana_url="http://localhost:3000",
        grafana_api_token="test_token",
        grafana_provision_on_start=False,
        compose_enabled=False,
        store_health_enabled=False,
    )


@pytest.fixture
def facade(dashboard_config: DashboardConfig) -> DashboardServiceFacade:
    """Create facade instance for testing."""
    controller = NoopServiceController()
    return DashboardServiceFacade(config=dashboard_config, controller=controller)


class TestFacadeInitialization:
    """Test facade initialization and configuration."""

    def test_from_config_creates_instance(self, dashboard_config: DashboardConfig) -> None:
        """Test from_config class method creates proper instance."""
        facade = DashboardServiceFacade.from_config(dashboard_config)
        assert facade.config == dashboard_config
        assert isinstance(facade, DashboardServiceFacade)

    def test_init_sets_config_and_controller(self, dashboard_config: DashboardConfig) -> None:
        """Test __init__ properly sets config and controller."""
        controller = NoopServiceController()
        facade = DashboardServiceFacade(config=dashboard_config, controller=controller)
        assert facade.config == dashboard_config
        assert facade.controller == controller

    def test_components_lazily_initialized(self, facade: DashboardServiceFacade) -> None:
        """Test components are None until first access."""
        assert facade._health_aggregator is None
        assert facade._registry_manager is None
        assert facade._grafana_provisioner is None
        assert facade._metrics_collector is None
        assert facade._pipeline_integration is None
        assert facade._service_controller is None
        assert facade._event_polling is None
        assert facade._authentication is None


class TestHealthAggregatorDelegation:
    """Test delegation to HealthAggregatorComponent."""

    def test_get_system_health_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test get_system_health delegates to HealthAggregatorComponent."""
        with patch.object(facade, "_get_health_aggregator") as mock_getter:
            mock_component = Mock()
            mock_component.get_system_health.return_value = {"services": {}, "dependencies": {}}
            mock_getter.return_value = mock_component

            result = facade.get_system_health()

            mock_component.get_system_health.assert_called_once()
            assert result == {"services": {}, "dependencies": {}}

    def test_list_services_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test list_services delegates to HealthAggregatorComponent."""
        with patch.object(facade, "_get_health_aggregator") as mock_getter:
            mock_component = Mock()
            expected = [{"name": "ml_signal_actor", "ports": {"http": 8081}}]
            mock_component.list_services.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.list_services()

            mock_component.list_services.assert_called_once()
            assert result == expected

    def test_get_store_summary_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test get_store_summary delegates to HealthAggregatorComponent."""
        with patch.object(facade, "_get_health_aggregator") as mock_getter:
            mock_component = Mock()
            expected = {"ok": True, "stores": []}
            mock_component.get_store_summary.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.get_store_summary()

            mock_component.get_store_summary.assert_called_once()
            assert result == expected


class TestRegistryManagerDelegation:
    """Test delegation to RegistryManagerComponent."""

    def test_list_models_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test list_models delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = [{"model_id": "test_model", "version": "1.0.0"}]
            mock_component.list_models.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.list_models()

            mock_component.list_models.assert_called_once()
            assert result == expected

    def test_get_model_performance_history_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test get_model_performance_history delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = [{"sharpe_ratio": 1.5, "timestamp": "2024-01-01"}]
            mock_component.get_model_performance_history.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.get_model_performance_history("test_model", limit=50)

            mock_component.get_model_performance_history.assert_called_once_with("test_model", limit=50)
            assert result == expected

    def test_list_deployments_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test list_deployments delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = {"production": ["model_v1", "model_v2"]}
            mock_component.list_deployments.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.list_deployments()

            mock_component.list_deployments.assert_called_once()
            assert result == expected

    def test_list_features_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test list_features delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = [{"feature_set_id": "test_features", "stage": "PROD"}]
            mock_component.list_features.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.list_features(role="training", stage="PROD")

            mock_component.list_features.assert_called_once_with(role="training", stage="PROD")
            assert result == expected

    def test_get_feature_lineage_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test get_feature_lineage delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = [{"feature_set_id": "parent_features"}]
            mock_component.get_feature_lineage.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.get_feature_lineage("test_features")

            mock_component.get_feature_lineage.assert_called_once_with("test_features")
            assert result == expected

    def test_list_strategies_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test list_strategies delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = [{"strategy_id": "test_strategy", "type": "ML"}]
            mock_component.list_strategies.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.list_strategies()

            mock_component.list_strategies.assert_called_once()
            assert result == expected

    def test_get_strategy_details_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test get_strategy_details delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = {"strategy_id": "test_strategy", "type": "ML", "version": "1.0.0"}
            mock_component.get_strategy_details.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.get_strategy_details("test_strategy")

            mock_component.get_strategy_details.assert_called_once_with("test_strategy")
            assert result == expected

    def test_check_strategy_compatibility_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test check_strategy_compatibility delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = {"strategy_id": "new_strategy", "compatible": True}
            mock_component.check_strategy_compatibility.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.check_strategy_compatibility("new_strategy", ["existing_strategy"])

            mock_component.check_strategy_compatibility.assert_called_once_with(
                "new_strategy",
                ["existing_strategy"],
            )
            assert result == expected

    def test_promote_feature_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test promote_feature delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = {"ok": True, "feature_set_id": "test_features", "stage": "PROD"}
            mock_component.promote_feature.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.promote_feature("test_features", stage="PROD", gates=[])

            mock_component.promote_feature.assert_called_once_with(
                "test_features",
                stage="PROD",
                gates=[],
            )
            assert result == expected

    def test_deprecate_feature_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test deprecate_feature delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = {"ok": True, "feature_set_id": "old_features"}
            mock_component.deprecate_feature.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.deprecate_feature("old_features", reason="superseded")

            mock_component.deprecate_feature.assert_called_once_with("old_features", reason="superseded")
            assert result == expected

    def test_list_datasets_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test list_datasets delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = [{"dataset_id": "test_dataset", "dataset_type": "training"}]
            mock_component.list_datasets.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.list_datasets()

            mock_component.list_datasets.assert_called_once()
            assert result == expected

    def test_list_watermarks_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test list_watermarks delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = [{"dataset_id": "test_dataset", "last_success_ns": 1000}]
            mock_component.list_watermarks.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.list_watermarks(
                dataset_id="test_dataset",
                instrument="SPY",
                source="databento",
                limit=10,
            )

            mock_component.list_watermarks.assert_called_once_with(
                dataset_id="test_dataset",
                instrument="SPY",
                source="databento",
                limit=10,
            )
            assert result == expected

    def test_list_dataset_lineage_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test list_dataset_lineage delegates to RegistryManagerComponent."""
        with patch.object(facade, "_get_registry_manager") as mock_getter:
            mock_component = Mock()
            expected = [{"child_dataset_id": "child", "parent_dataset_id": "parent"}]
            mock_component.list_dataset_lineage.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.list_dataset_lineage(child="child", parent="parent", limit=20)

            mock_component.list_dataset_lineage.assert_called_once_with(
                child="child",
                parent="parent",
                limit=20,
            )
            assert result == expected


class TestGrafanaProvisionerDelegation:
    """Test delegation to GrafanaProvisionerComponent."""

    def test_provision_grafana_dashboard_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test provision_grafana_dashboard delegates to GrafanaProvisionerComponent."""
        with patch.object(facade, "_get_grafana_provisioner") as mock_getter:
            mock_component = Mock()
            expected = {"ok": True, "url": "http://localhost:3000/d/test"}
            mock_component.provision_grafana_dashboard.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.provision_grafana_dashboard(title="Test Dashboard", force=True)

            mock_component.provision_grafana_dashboard.assert_called_once_with(
                title="Test Dashboard",
                force=True,
            )
            assert result == expected

    def test_get_grafana_status_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test get_grafana_status delegates to GrafanaProvisionerComponent."""
        with patch.object(facade, "_get_grafana_provisioner") as mock_getter:
            mock_component = Mock()
            expected = {"ok": True, "url": "http://localhost:3000"}
            mock_component.get_grafana_status.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.get_grafana_status()

            mock_component.get_grafana_status.assert_called_once()
            assert result == expected

    def test_get_prometheus_summary_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test get_prometheus_summary delegates to GrafanaProvisionerComponent."""
        with patch.object(facade, "_get_grafana_provisioner") as mock_getter:
            mock_component = Mock()
            expected = {"ok": True, "metrics": {"request_rate": 10.5}}
            mock_component.get_prometheus_summary.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.get_prometheus_summary()

            mock_component.get_prometheus_summary.assert_called_once()
            assert result == expected


class TestMetricsCollectorDelegation:
    """Test delegation to MetricsCollectorComponent."""

    def test_get_metrics_snapshot_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test get_metrics_snapshot delegates to MetricsCollectorComponent."""
        with patch.object(facade, "_get_metrics_collector") as mock_getter:
            mock_component = Mock()
            # Use correct structure from ml/dashboard/metrics_snapshot.py
            from ml.dashboard.metrics_snapshot import CacheStats, RequestStats
            expected = DashboardMetricsSnapshot(
                registry_cache=CacheStats(hits=90.0, misses=10.0),
                event_cache=CacheStats(hits=85.0, misses=15.0),
                grafana_provisioning=RequestStats(successes=95.0, total=100.0),
                store_summary_p95_seconds=0.05,
                registry_latency_p95_seconds=0.01,
            )
            mock_component.get_metrics_snapshot.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.get_metrics_snapshot()

            mock_component.get_metrics_snapshot.assert_called_once()
            assert result == expected

    def test_evaluate_success_criteria_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test evaluate_success_criteria delegates to MetricsCollectorComponent."""
        with patch.object(facade, "_get_metrics_collector") as mock_getter:
            mock_component = Mock()
            # Use correct structure from ml/dashboard/metrics_snapshot.py
            expected = DashboardSuccessReport(
                registry_latency_p95_seconds=0.15,
                event_cache_hit_ratio=0.85,
                grafana_success_rate=0.98,
                store_summary_p95_seconds=0.50,
            )
            mock_component.evaluate_success_criteria.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.evaluate_success_criteria()

            mock_component.evaluate_success_criteria.assert_called_once()
            assert result == expected


class TestPipelineIntegrationDelegation:
    """Test delegation to PipelineIntegrationComponent."""

    def test_trigger_pipeline_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test trigger_pipeline delegates to PipelineIntegrationComponent."""
        with patch.object(facade, "_get_pipeline_integration") as mock_getter:
            mock_component = Mock()
            expected = {"success": True, "job_id": "job123", "status": "RUNNING"}
            mock_component.trigger_pipeline.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.trigger_pipeline("train_model", {"dataset_id": "test"})

            mock_component.trigger_pipeline.assert_called_once_with("train_model", {"dataset_id": "test"})
            assert result == expected

    def test_trigger_orchestrator_task_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test trigger_orchestrator_task delegates to PipelineIntegrationComponent."""
        with patch.object(facade, "_get_pipeline_integration") as mock_getter:
            mock_component = Mock()
            expected = {"ok": True, "result": {"status": "started"}}
            mock_component.trigger_orchestrator_task.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.trigger_orchestrator_task("backfill", {"instruments": ["SPY"]})

            mock_component.trigger_orchestrator_task.assert_called_once_with(
                "backfill",
                {"instruments": ["SPY"]},
            )
            assert result == expected

    def test_list_pipeline_jobs_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test list_pipeline_jobs delegates to PipelineIntegrationComponent."""
        with patch.object(facade, "_get_pipeline_integration") as mock_getter:
            mock_component = Mock()
            expected = {"status": "success", "jobs": []}
            mock_component.list_pipeline_jobs.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.list_pipeline_jobs()

            mock_component.list_pipeline_jobs.assert_called_once()
            assert result == expected

    def test_get_pipeline_job_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test get_pipeline_job delegates to PipelineIntegrationComponent."""
        with patch.object(facade, "_get_pipeline_integration") as mock_getter:
            mock_component = Mock()
            expected = {"status": "success", "job": {"job_id": "job123"}}
            mock_component.get_pipeline_job.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.get_pipeline_job("job123")

            mock_component.get_pipeline_job.assert_called_once_with("job123")
            assert result == expected

    def test_purge_pipeline_job_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test purge_pipeline_job delegates to PipelineIntegrationComponent."""
        with patch.object(facade, "_get_pipeline_integration") as mock_getter:
            mock_component = Mock()
            expected = {"status": "purged", "result": {"success": True}}
            mock_component.purge_pipeline_job.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.purge_pipeline_job("job123")

            mock_component.purge_pipeline_job.assert_called_once_with("job123")
            assert result == expected

    def test_build_dataset_pipeline_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test build_dataset_pipeline delegates to PipelineIntegrationComponent."""
        with patch.object(facade, "_get_pipeline_integration") as mock_getter:
            mock_component = Mock()
            expected = {"success": True, "job_id": "job456"}
            mock_component.build_dataset_pipeline.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.build_dataset_pipeline({"dataset_id": "test"})

            mock_component.build_dataset_pipeline.assert_called_once_with({"dataset_id": "test"})
            assert result == expected

    def test_train_model_pipeline_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test train_model_pipeline delegates to PipelineIntegrationComponent."""
        with patch.object(facade, "_get_pipeline_integration") as mock_getter:
            mock_component = Mock()
            expected = {"success": True, "job_id": "job789"}
            mock_component.train_model_pipeline.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.train_model_pipeline({"model_type": "xgboost"})

            mock_component.train_model_pipeline.assert_called_once_with({"model_type": "xgboost"})
            assert result == expected

    def test_run_hpo_pipeline_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test run_hpo_pipeline delegates to PipelineIntegrationComponent."""
        with patch.object(facade, "_get_pipeline_integration") as mock_getter:
            mock_component = Mock()
            expected = {"success": True, "job_id": "job999"}
            mock_component.run_hpo_pipeline.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.run_hpo_pipeline({"search_method": "bayesian"})

            mock_component.run_hpo_pipeline.assert_called_once_with({"search_method": "bayesian"})
            assert result == expected

    def test_get_pipeline_progress_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test get_pipeline_progress delegates to PipelineIntegrationComponent."""
        with patch.object(facade, "_get_pipeline_integration") as mock_getter:
            mock_component = Mock()
            expected = {"status": "success", "progress": {"progress": 0.5}}
            mock_component.get_pipeline_progress.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.get_pipeline_progress("job123")

            mock_component.get_pipeline_progress.assert_called_once_with("job123")
            assert result == expected

    def test_cancel_pipeline_job_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test cancel_pipeline_job delegates to PipelineIntegrationComponent."""
        with patch.object(facade, "_get_pipeline_integration") as mock_getter:
            mock_component = Mock()
            expected = {"success": True, "job_id": "job123", "status": "CANCELLED"}
            mock_component.cancel_pipeline_job.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.cancel_pipeline_job("job123")

            mock_component.cancel_pipeline_job.assert_called_once_with("job123")
            assert result == expected

    def test_get_integration_manager_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test get_integration_manager delegates to PipelineIntegrationComponent."""
        with patch.object(facade, "_get_pipeline_integration") as mock_getter:
            mock_component = Mock()
            mock_manager = MagicMock()
            mock_component.get_integration_manager.return_value = mock_manager
            mock_getter.return_value = mock_component

            result = facade.get_integration_manager()

            mock_component.get_integration_manager.assert_called_once()
            assert result == mock_manager


class TestServiceControllerDelegation:
    """Test delegation to ServiceControllerComponent."""

    def test_control_service_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test control_service delegates to ServiceControllerComponent."""
        with patch.object(facade, "_get_service_controller") as mock_getter:
            mock_component = Mock()
            expected = {"ok": True, "action": "start", "service": "ml_signal_actor"}
            mock_component.control_service.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.control_service("ml_signal_actor", "start")

            mock_component.control_service.assert_called_once_with("ml_signal_actor", "start")
            assert result == expected


class TestEventPollingDelegation:
    """Test delegation to EventPollingComponent."""

    def test_list_events_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test list_events delegates to EventPollingComponent."""
        with patch.object(facade, "_get_event_polling") as mock_getter:
            mock_component = Mock()
            expected = [{"id": "event1", "topic": "ml.features", "payload": {}}]
            mock_component.list_events.return_value = expected
            mock_getter.return_value = mock_component

            result = facade.list_events(
                limit=50,
                stage="FEATURE_COMPUTED",
                source="databento",
                instrument_substr="SPY",
            )

            mock_component.list_events.assert_called_once_with(
                limit=50,
                stage="FEATURE_COMPUTED",
                source="databento",
                instrument_substr="SPY",
            )
            assert result == expected

    def test_start_event_polling_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test start_event_polling delegates to EventPollingComponent."""
        with patch.object(facade, "_get_event_polling") as mock_getter:
            mock_component = Mock()
            mock_getter.return_value = mock_component

            facade.start_event_polling(5.0)

            mock_component.start_event_polling.assert_called_once_with(5.0)

    def test_stop_event_polling_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test stop_event_polling delegates to EventPollingComponent."""
        with patch.object(facade, "_get_event_polling") as mock_getter:
            mock_component = Mock()
            mock_getter.return_value = mock_component

            facade.stop_event_polling()

            mock_component.stop_event_polling.assert_called_once()


class TestAuthenticationDelegation:
    """Test delegation to AuthenticationComponent."""

    def test_validate_token_delegates(self, facade: DashboardServiceFacade) -> None:
        """Test validate_token delegates to AuthenticationComponent."""
        with patch.object(facade, "_get_authentication") as mock_getter:
            mock_component = Mock()
            mock_component.validate_token.return_value = True
            mock_getter.return_value = mock_component

            now = datetime.now(UTC)
            result = facade.validate_token("test_token", now=now)

            mock_component.validate_token.assert_called_once_with("test_token", now=now)
            assert result is True


class TestComponentLazyLoading:
    """Test that components are lazily loaded and cached."""

    def test_health_aggregator_cached(self, facade: DashboardServiceFacade) -> None:
        """Test HealthAggregatorComponent is created once and cached."""
        component1 = facade._get_health_aggregator()
        component2 = facade._get_health_aggregator()
        assert component1 is component2

    def test_registry_manager_cached(self, facade: DashboardServiceFacade) -> None:
        """Test RegistryManagerComponent is created once and cached."""
        component1 = facade._get_registry_manager()
        component2 = facade._get_registry_manager()
        assert component1 is component2

    def test_grafana_provisioner_cached(self, facade: DashboardServiceFacade) -> None:
        """Test GrafanaProvisionerComponent is created once and cached."""
        component1 = facade._get_grafana_provisioner()
        component2 = facade._get_grafana_provisioner()
        assert component1 is component2

    def test_metrics_collector_cached(self, facade: DashboardServiceFacade) -> None:
        """Test MetricsCollectorComponent is created once and cached."""
        component1 = facade._get_metrics_collector()
        component2 = facade._get_metrics_collector()
        assert component1 is component2

    def test_pipeline_integration_cached(self, facade: DashboardServiceFacade) -> None:
        """Test PipelineIntegrationComponent is created once and cached."""
        component1 = facade._get_pipeline_integration()
        component2 = facade._get_pipeline_integration()
        assert component1 is component2

    def test_service_controller_cached(self, facade: DashboardServiceFacade) -> None:
        """Test ServiceControllerComponent is created once and cached."""
        component1 = facade._get_service_controller()
        component2 = facade._get_service_controller()
        assert component1 is component2

    def test_event_polling_cached(self, facade: DashboardServiceFacade) -> None:
        """Test EventPollingComponent is created once and cached."""
        component1 = facade._get_event_polling()
        component2 = facade._get_event_polling()
        assert component1 is component2

    def test_authentication_cached(self, facade: DashboardServiceFacade) -> None:
        """Test AuthenticationComponent is created once and cached."""
        component1 = facade._get_authentication()
        component2 = facade._get_authentication()
        assert component1 is component2
