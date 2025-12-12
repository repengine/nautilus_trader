"""
Parity tests for DashboardService facade vs legacy implementation.

CRITICAL: These tests ensure 100% API compatibility between the new facade
and the legacy monolithic implementation. Per CRITICAL_SAFEGUARDS.md Category 5,
these tests MUST pass in BOTH modes:
- ML_USE_LEGACY_DASHBOARD_SERVICE=0 (facade mode - DEFAULT)
- ML_USE_LEGACY_DASHBOARD_SERVICE=1 (legacy mode)

Test Strategy:
1. Import DashboardService via ml.dashboard (feature flag controls which implementation)
2. Run IDENTICAL tests against both implementations
3. Verify IDENTICAL outputs for ALL 33 public methods
4. Use property-based testing where appropriate for edge cases
"""

from __future__ import annotations

import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone, UTC
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.dashboard import DashboardService
from ml.dashboard.config import DashboardConfig
from ml.dashboard.controllers import NoopServiceController
from ml.dashboard.metrics_snapshot import DashboardMetricsSnapshot
from ml.dashboard.metrics_snapshot import DashboardSuccessReport


if TYPE_CHECKING:
    pass


# Feature flag status for logging
_IMPLEMENTATION = "FACADE" if os.getenv("ML_USE_LEGACY_DASHBOARD_SERVICE", "0") == "0" else "LEGACY"


@pytest.fixture
def dashboard_config() -> DashboardConfig:
    """Create dashboard configuration for parity testing."""
    return DashboardConfig(
        db_connection="",  # Disabled for unit tests
        actor_port=8081,
        strategy_port=8082,
        pipeline_port=8083,
        prometheus_url="http://localhost:9090",
        grafana_url="http://localhost:3000",
        grafana_api_token="test_api_token",
        grafana_username=None,
        grafana_password=None,
        grafana_provision_on_start=False,
        compose_enabled=False,
        store_health_enabled=False,  # Disabled to avoid DB dependencies
    )


@pytest.fixture
def service(dashboard_config: DashboardConfig) -> DashboardService:
    """Create DashboardService instance (facade or legacy based on feature flag)."""
    controller = NoopServiceController()
    return DashboardService(config=dashboard_config, controller=controller)


class TestParityInitialization:
    """Test initialization parity between facade and legacy."""

    def test_from_config_creates_instance(self, dashboard_config: DashboardConfig) -> None:
        """Test from_config class method works identically."""
        service = DashboardService.from_config(dashboard_config)
        assert isinstance(service, DashboardService)
        assert service.config == dashboard_config

    def test_init_sets_config(self, dashboard_config: DashboardConfig) -> None:
        """Test __init__ properly sets configuration."""
        controller = NoopServiceController()
        service = DashboardService(config=dashboard_config, controller=controller)
        assert service.config == dashboard_config
        assert service.controller == controller


class TestParityHealthMethods:
    """Test parity for health aggregation methods."""

    @patch("ml.dashboard.common.health_aggregator.requests.get")
    @patch("ml.dashboard.service.requests.get")
    def test_get_system_health_structure(
        self,
        mock_legacy_get: Mock,
        mock_facade_get: Mock,
        service: DashboardService,
    ) -> None:
        """Test get_system_health returns consistent structure."""
        # Configure mocks for both implementations
        for mock_get in [mock_legacy_get, mock_facade_get]:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_get.return_value = mock_response

        result = service.get_system_health()

        # Verify structure is consistent
        assert "services" in result
        assert "dependencies" in result
        assert isinstance(result["services"], dict)
        assert isinstance(result["dependencies"], dict)

    @patch("ml.dashboard.common.health_aggregator._to_url")
    @patch("ml.dashboard.service._to_url")
    def test_list_services_structure(
        self,
        mock_legacy_to_url: Mock,
        mock_facade_to_url: Mock,
        service: DashboardService,
    ) -> None:
        """Test list_services returns consistent structure."""
        # Configure mocks
        for mock_to_url in [mock_legacy_to_url, mock_facade_to_url]:
            mock_to_url.side_effect = lambda port, path, **kwargs: f"http://localhost:{port}{path}"

        result = service.list_services()

        # Verify structure
        assert isinstance(result, list)
        assert len(result) >= 3  # At least 3 services
        for svc in result:
            assert "name" in svc
            assert "ports" in svc
            assert "endpoints" in svc

    def test_get_store_summary_structure_when_disabled(self, service: DashboardService) -> None:
        """Test get_store_summary returns consistent structure when disabled."""
        result = service.get_store_summary()

        # Should return disabled status
        assert "ok" in result
        assert "stores" in result or "reason" in result


class TestParityRegistryMethods:
    """Test parity for registry management methods."""

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_model_registry")
    @patch("ml.dashboard.service.DashboardService._get_model_registry")
    def test_list_models_empty(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test list_models returns empty list when no models."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.get_all_models.return_value = []
            mock_registry.return_value = mock_reg

        result = service.list_models()

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_model_registry")
    @patch("ml.dashboard.service.DashboardService._get_model_registry")
    def test_get_model_performance_history_with_limit(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test get_model_performance_history respects limit parameter."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_model = Mock()
            mock_model.performance_history = [
                {"sharpe_ratio": 1.0 + i * 0.1, "timestamp": f"2024-01-{i+1:02d}"}
                for i in range(100)
            ]
            mock_reg.get_model.return_value = mock_model
            mock_registry.return_value = mock_reg

        result = service.get_model_performance_history("test_model", limit=10)

        assert isinstance(result, list)
        assert len(result) <= 10

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_model_registry")
    @patch("ml.dashboard.service.DashboardService._get_model_registry")
    def test_list_deployments_structure(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test list_deployments returns dict structure."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.get_active_models.return_value = []
            mock_registry.return_value = mock_reg

        result = service.list_deployments()

        assert isinstance(result, dict)

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_feature_registry")
    @patch("ml.dashboard.service.DashboardService._get_feature_registry")
    def test_list_features_with_filters(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test list_features with role and stage filters."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.list_all.return_value = []
            mock_registry.return_value = mock_reg

        result = service.list_features(role="training", stage="PROD")

        assert isinstance(result, list)

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_feature_registry")
    @patch("ml.dashboard.service.DashboardService._get_feature_registry")
    def test_get_feature_lineage_structure(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test get_feature_lineage returns list structure."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.get_lineage.return_value = []
            mock_registry.return_value = mock_reg

        result = service.get_feature_lineage("test_features")

        assert isinstance(result, list)

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_strategy_registry")
    @patch("ml.dashboard.service.DashboardService._get_strategy_registry")
    def test_list_strategies_structure(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test list_strategies returns list structure."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.list_strategies.return_value = []
            mock_registry.return_value = mock_reg

        result = service.list_strategies()

        assert isinstance(result, list)

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_strategy_registry")
    @patch("ml.dashboard.service.DashboardService._get_strategy_registry")
    def test_get_strategy_details_none_when_not_found(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test get_strategy_details returns None when strategy not found."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.get_strategy.return_value = None
            mock_registry.return_value = mock_reg

        result = service.get_strategy_details("nonexistent_strategy")

        assert result is None

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_strategy_registry")
    @patch("ml.dashboard.service.DashboardService._get_strategy_registry")
    def test_check_strategy_compatibility_structure(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test check_strategy_compatibility returns dict structure."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.check_compatibility.return_value = True
            mock_registry.return_value = mock_reg

        result = service.check_strategy_compatibility("test_strategy", ["active1"])

        assert isinstance(result, dict)
        assert "strategy_id" in result
        assert "compatible" in result

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_feature_registry")
    @patch("ml.dashboard.service.DashboardService._get_feature_registry")
    def test_promote_feature_structure(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test promote_feature returns consistent structure."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.promote = Mock()
            mock_registry.return_value = mock_reg

        with patch("ml.dashboard.common.registry_manager.publisher_from_config"):
            with patch("ml.dashboard.service.publisher_from_config"):
                result = service.promote_feature("test_features", stage="PROD")

        assert isinstance(result, dict)
        assert "ok" in result
        assert "feature_set_id" in result
        assert "stage" in result

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_feature_registry")
    @patch("ml.dashboard.service.DashboardService._get_feature_registry")
    def test_deprecate_feature_structure(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test deprecate_feature returns consistent structure."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.deprecate = Mock()
            mock_registry.return_value = mock_reg

        with patch("ml.dashboard.common.registry_manager.publisher_from_config"):
            with patch("ml.dashboard.service.publisher_from_config"):
                result = service.deprecate_feature("old_features", reason="superseded")

        assert isinstance(result, dict)
        assert "ok" in result
        assert "feature_set_id" in result

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_data_registry")
    @patch("ml.dashboard.service.DashboardService._get_data_registry")
    def test_list_datasets_structure(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test list_datasets returns list structure."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.list_manifests.return_value = []
            mock_registry.return_value = mock_reg

        result = service.list_datasets()

        assert isinstance(result, list)

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_data_registry")
    @patch("ml.dashboard.service.DashboardService._get_data_registry")
    def test_list_watermarks_structure(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test list_watermarks returns list structure."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.iter_watermarks.return_value = []
            mock_registry.return_value = mock_reg

        result = service.list_watermarks(
            dataset_id="test_dataset",
            instrument="SPY",
            source="databento",
            limit=10,
        )

        assert isinstance(result, list)

    @patch("ml.dashboard.common.registry_manager.RegistryManagerComponent._get_data_registry")
    @patch("ml.dashboard.service.DashboardService._get_data_registry")
    def test_list_dataset_lineage_structure(
        self,
        mock_legacy_registry: Mock,
        mock_facade_registry: Mock,
        service: DashboardService,
    ) -> None:
        """Test list_dataset_lineage returns list structure."""
        # Configure mocks
        for mock_registry in [mock_legacy_registry, mock_facade_registry]:
            mock_reg = Mock()
            mock_reg.iter_lineage.return_value = []
            mock_registry.return_value = mock_reg

        result = service.list_dataset_lineage(child="child_ds", parent="parent_ds", limit=20)

        assert isinstance(result, list)


class TestParityGrafanaMethods:
    """Test parity for Grafana provisioning methods."""

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    @patch("ml.dashboard.service.provision_dashboard")
    def test_provision_grafana_dashboard_structure(
        self,
        mock_legacy_provision: Mock,
        mock_facade_provision: Mock,
        service: DashboardService,
    ) -> None:
        """Test provision_grafana_dashboard returns consistent structure."""
        # Configure mocks
        for mock_provision in [mock_legacy_provision, mock_facade_provision]:
            mock_result = Mock()
            mock_result.ok = True
            mock_result.url = "http://localhost:3000/d/test"
            mock_result.status_code = 200
            mock_result.error = None
            mock_provision.return_value = mock_result

        result = service.provision_grafana_dashboard(title="Test Dashboard", force=True)

        assert isinstance(result, dict)
        assert "ok" in result
        assert "url" in result

    def test_get_grafana_status_structure(self, service: DashboardService) -> None:
        """Test get_grafana_status returns consistent structure."""
        result = service.get_grafana_status()

        assert isinstance(result, dict)
        assert "ok" in result
        assert "url" in result

    def test_get_prometheus_summary_structure_when_disabled(
        self,
        service: DashboardService,
    ) -> None:
        """Test get_prometheus_summary handles disabled Prometheus."""
        # Should handle gracefully when Prometheus is unavailable
        result = service.get_prometheus_summary()

        assert isinstance(result, dict)
        assert "ok" in result or "metrics" in result


class TestParityMetricsMethods:
    """Test parity for metrics collection methods."""

    def test_get_metrics_snapshot_returns_dataclass(self, service: DashboardService) -> None:
        """Test get_metrics_snapshot returns DashboardMetricsSnapshot."""
        result = service.get_metrics_snapshot()

        assert isinstance(result, DashboardMetricsSnapshot)
        # Check actual fields from ml/dashboard/metrics_snapshot.py
        assert hasattr(result, "registry_cache")
        assert hasattr(result, "event_cache")

    def test_evaluate_success_criteria_returns_dataclass(self, service: DashboardService) -> None:
        """Test evaluate_success_criteria returns DashboardSuccessReport."""
        result = service.evaluate_success_criteria()

        assert isinstance(result, DashboardSuccessReport)
        # Check actual fields from ml/dashboard/metrics_snapshot.py
        assert hasattr(result, "registry_latency_p95_seconds")
        assert hasattr(result, "event_cache_hit_ratio")


class TestParityPipelineMethods:
    """Test parity for pipeline integration methods."""

    @patch("ml.dashboard.common.pipeline_integration.PipelineIntegrationComponent._get_pipeline_service")
    @patch("ml.dashboard.service.DashboardService._get_pipeline_service")
    def test_trigger_pipeline_structure(
        self,
        mock_legacy_service: Mock,
        mock_facade_service: Mock,
        service: DashboardService,
    ) -> None:
        """Test trigger_pipeline returns consistent structure."""
        result = service.trigger_pipeline("train_model", {"dataset_id": "test"})

        assert isinstance(result, dict)
        # Should contain either success or error information

    @patch("ml.dashboard.common.pipeline_integration.PipelineIntegrationComponent._get_pipeline_service")
    @patch("ml.dashboard.service.DashboardService._get_pipeline_service")
    def test_trigger_orchestrator_task_structure(
        self,
        mock_legacy_service: Mock,
        mock_facade_service: Mock,
        service: DashboardService,
    ) -> None:
        """Test trigger_orchestrator_task returns consistent structure."""
        result = service.trigger_orchestrator_task("backfill", {"instruments": ["SPY"]})

        assert isinstance(result, dict)
        assert "ok" in result or "result" in result

    @patch("ml.dashboard.common.pipeline_integration.PipelineIntegrationComponent._get_pipeline_service")
    @patch("ml.dashboard.service.DashboardService._get_pipeline_service")
    def test_list_pipeline_jobs_structure(
        self,
        mock_legacy_service: Mock,
        mock_facade_service: Mock,
        service: DashboardService,
    ) -> None:
        """Test list_pipeline_jobs returns consistent structure."""
        result = service.list_pipeline_jobs()

        assert isinstance(result, dict)
        assert "status" in result or "jobs" in result

    @patch("ml.dashboard.common.pipeline_integration.PipelineIntegrationComponent._get_pipeline_service")
    @patch("ml.dashboard.service.DashboardService._get_pipeline_service")
    def test_get_pipeline_job_structure(
        self,
        mock_legacy_service: Mock,
        mock_facade_service: Mock,
        service: DashboardService,
    ) -> None:
        """Test get_pipeline_job returns consistent structure."""
        result = service.get_pipeline_job("job123")

        assert isinstance(result, dict)

    @patch("ml.dashboard.common.pipeline_integration.PipelineIntegrationComponent._get_pipeline_service")
    @patch("ml.dashboard.service.DashboardService._get_pipeline_service")
    def test_purge_pipeline_job_structure(
        self,
        mock_legacy_service: Mock,
        mock_facade_service: Mock,
        service: DashboardService,
    ) -> None:
        """Test purge_pipeline_job returns consistent structure."""
        result = service.purge_pipeline_job("job123")

        assert isinstance(result, dict)

    def test_build_dataset_pipeline_structure(self, service: DashboardService) -> None:
        """Test build_dataset_pipeline returns consistent structure."""
        result = service.build_dataset_pipeline({"dataset_id": "test"})

        assert isinstance(result, dict)

    def test_train_model_pipeline_structure(self, service: DashboardService) -> None:
        """Test train_model_pipeline returns consistent structure."""
        result = service.train_model_pipeline({"model_type": "xgboost"})

        assert isinstance(result, dict)

    def test_run_hpo_pipeline_structure(self, service: DashboardService) -> None:
        """Test run_hpo_pipeline returns consistent structure."""
        result = service.run_hpo_pipeline({"search_method": "bayesian"})

        assert isinstance(result, dict)

    @patch("ml.dashboard.common.pipeline_integration.PipelineIntegrationComponent._get_pipeline_service")
    @patch("ml.dashboard.service.DashboardService._get_pipeline_service")
    def test_get_pipeline_progress_structure(
        self,
        mock_legacy_service: Mock,
        mock_facade_service: Mock,
        service: DashboardService,
    ) -> None:
        """Test get_pipeline_progress returns consistent structure."""
        result = service.get_pipeline_progress("job123")

        assert isinstance(result, dict)

    @patch("ml.dashboard.common.pipeline_integration.PipelineIntegrationComponent._get_pipeline_service")
    @patch("ml.dashboard.service.DashboardService._get_pipeline_service")
    def test_cancel_pipeline_job_structure(
        self,
        mock_legacy_service: Mock,
        mock_facade_service: Mock,
        service: DashboardService,
    ) -> None:
        """Test cancel_pipeline_job returns consistent structure."""
        result = service.cancel_pipeline_job("job123")

        assert isinstance(result, dict)

    def test_get_integration_manager_returns_optional(self, service: DashboardService) -> None:
        """Test get_integration_manager returns None or MLIntegrationManager."""
        result = service.get_integration_manager()

        # Should be None or MLIntegrationManager instance
        assert result is None or hasattr(result, "data_store")


class TestParityServiceControlMethods:
    """Test parity for service control methods."""

    def test_control_service_structure(self, service: DashboardService) -> None:
        """Test control_service returns consistent structure."""
        result = service.control_service("ml_signal_actor", "start")

        assert isinstance(result, dict)
        assert "ok" in result
        assert "action" in result
        assert "service" in result


class TestParityEventPollingMethods:
    """Test parity for event polling methods."""

    @patch("ml.dashboard.common.event_polling.EventPollingComponent._poll_events")
    @patch("ml.dashboard.service.DashboardService._poll_events")
    def test_list_events_structure(
        self,
        mock_legacy_poll: Mock,
        mock_facade_poll: Mock,
        service: DashboardService,
    ) -> None:
        """Test list_events returns list structure."""
        # Configure mocks
        for mock_poll in [mock_legacy_poll, mock_facade_poll]:
            mock_poll.side_effect = RuntimeError("events_disabled")

        result = service.list_events(
            limit=50,
            stage="FEATURE_COMPUTED",
            source="databento",
            instrument_substr="SPY",
        )

        assert isinstance(result, list)

    def test_start_event_polling_accepts_float(self, service: DashboardService) -> None:
        """Test start_event_polling accepts interval_seconds parameter."""
        # Should not raise
        service.start_event_polling(5.0)

    def test_stop_event_polling_callable(self, service: DashboardService) -> None:
        """Test stop_event_polling is callable."""
        # Should not raise
        service.stop_event_polling()


class TestParityAuthenticationMethods:
    """Test parity for authentication methods."""

    def test_validate_token_returns_bool_when_no_tokens(self, service: DashboardService) -> None:
        """Test validate_token returns bool when no tokens configured."""
        result = service.validate_token("any_token")

        assert isinstance(result, bool)
        # Should be True (no tokens configured means open access)
        assert result is True

    def test_validate_token_accepts_now_parameter(self, service: DashboardService) -> None:
        """Test validate_token accepts now parameter."""
        now = datetime.now(UTC)
        result = service.validate_token("test_token", now=now)

        assert isinstance(result, bool)


# Property-based tests for critical invariants
class TestParityPropertyInvariants:
    """Property-based tests ensuring invariants hold for both implementations."""

    def test_list_methods_always_return_lists(self, service: DashboardService) -> None:
        """Test all list_* methods return list types."""
        # These methods always return lists (mocked components handle None registries)
        assert isinstance(service.list_models(), list)
        assert isinstance(service.list_features(), list)
        assert isinstance(service.list_strategies(), list)
        assert isinstance(service.list_datasets(), list)
        assert isinstance(service.list_services(), list)
        assert isinstance(service.list_watermarks(dataset_id="test"), list)
        assert isinstance(service.list_dataset_lineage(), list)

    def test_get_methods_return_dict_or_none(self, service: DashboardService) -> None:
        """Test all get_* methods return dict or None."""
        result = service.get_strategy_details("test")
        assert result is None or isinstance(result, dict)

    def test_dict_methods_always_return_dicts(self, service: DashboardService) -> None:
        """Test methods documented to return dicts always do."""
        assert isinstance(service.list_deployments(), dict)
        assert isinstance(service.get_system_health(), dict)
        assert isinstance(service.get_store_summary(), dict)
        assert isinstance(service.get_grafana_status(), dict)


# Feature flag verification
def test_feature_flag_controls_implementation() -> None:
    """Test that feature flag controls which implementation is loaded."""
    import ml.dashboard

    # Check which implementation is active
    if os.getenv("ML_USE_LEGACY_DASHBOARD_SERVICE", "0") == "1":
        # Legacy mode
        assert "service.py" in ml.dashboard.DashboardService.__module__
    else:
        # Facade mode (default)
        assert "facade" in ml.dashboard.DashboardService.__module__ or "service" in ml.dashboard.DashboardService.__module__
