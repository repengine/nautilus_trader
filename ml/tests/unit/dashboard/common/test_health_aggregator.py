"""Unit tests for HealthAggregatorComponent.

Tests extracted health aggregation logic isolated from DashboardService.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ml.dashboard.common.health_aggregator import HealthAggregatorComponent
from ml.dashboard.config import DashboardConfig
from ml.tests.utils.db import build_postgres_url

TEST_DB_CONNECTION = build_postgres_url(user="test", password="test", database="test_db")


@pytest.fixture
def dashboard_config() -> DashboardConfig:
    """Create a minimal dashboard config for testing."""
    return DashboardConfig(
        actor_port=8000,
        strategy_port=8001,
        pipeline_port=8081,
        grafana_port=3000,
        prometheus_port=9090,
        grafana_url="http://localhost:3000",
        prometheus_url="http://localhost:9090",
        request_timeout_seconds=2.5,
        store_health_enabled=True,
        db_connection=TEST_DB_CONNECTION,
        store_health_top_datasets=5,
    )


@pytest.fixture
def health_aggregator(dashboard_config: DashboardConfig) -> HealthAggregatorComponent:
    """Create a health aggregator component instance."""
    return HealthAggregatorComponent(config=dashboard_config)


class TestHealthAggregatorComponentInstantiation:
    """Test component instantiation and initialization."""

    def test_component_instantiation_with_valid_config(
        self, dashboard_config: DashboardConfig
    ) -> None:
        """Component instantiates with valid config."""
        component = HealthAggregatorComponent(config=dashboard_config)
        assert component.config == dashboard_config

    def test_component_has_required_methods(self, health_aggregator: HealthAggregatorComponent) -> None:
        """Component implements required protocol methods."""
        assert hasattr(health_aggregator, "get_system_health")
        assert callable(health_aggregator.get_system_health)
        assert hasattr(health_aggregator, "list_services")
        assert callable(health_aggregator.list_services)
        assert hasattr(health_aggregator, "get_store_summary")
        assert callable(health_aggregator.get_store_summary)


class TestGetSystemHealth:
    """Test get_system_health method."""

    @patch("ml.dashboard.common.health_aggregator._safe_get")
    def test_get_system_health_all_services_healthy(
        self,
        mock_safe_get: MagicMock,
        health_aggregator: HealthAggregatorComponent,
    ) -> None:
        """System health shows all services healthy when probes succeed."""
        # All health checks return success
        mock_safe_get.return_value = (True, 200)

        health = health_aggregator.get_system_health()

        assert health["services"]["ml_signal_actor"]["healthy"] is True
        assert health["services"]["ml_signal_actor"]["status_code"] == 200
        assert health["services"]["ml_strategy"]["healthy"] is True
        assert health["services"]["ml_strategy"]["status_code"] == 200
        assert health["services"]["ml_pipeline"]["healthy"] is True
        assert health["services"]["ml_pipeline"]["status_code"] == 200
        assert health["dependencies"]["prometheus"]["healthy"] is True
        assert health["dependencies"]["prometheus"]["status_code"] == 200
        assert health["dependencies"]["grafana"]["healthy"] is True
        assert health["dependencies"]["grafana"]["status_code"] == 200

    @patch("ml.dashboard.common.health_aggregator._safe_get")
    def test_get_system_health_partial_failure(
        self,
        mock_safe_get: MagicMock,
        health_aggregator: HealthAggregatorComponent,
    ) -> None:
        """System health shows partial failures correctly."""
        # First service fails, others succeed
        mock_safe_get.side_effect = [
            (False, 503),  # ml_signal_actor
            (True, 200),   # ml_strategy
            (True, 200),   # ml_pipeline
            (True, 200),   # prometheus
            (True, 200),   # grafana
        ]

        health = health_aggregator.get_system_health()

        assert health["services"]["ml_signal_actor"]["healthy"] is False
        assert health["services"]["ml_signal_actor"]["status_code"] == 503
        assert health["services"]["ml_strategy"]["healthy"] is True
        assert health["services"]["ml_pipeline"]["healthy"] is True

    @patch("ml.dashboard.common.health_aggregator._safe_get")
    def test_get_system_health_all_services_down(
        self,
        mock_safe_get: MagicMock,
        health_aggregator: HealthAggregatorComponent,
    ) -> None:
        """System health shows all services down when all probes fail."""
        # All health checks fail
        mock_safe_get.return_value = (False, 0)

        health = health_aggregator.get_system_health()

        assert health["services"]["ml_signal_actor"]["healthy"] is False
        assert health["services"]["ml_strategy"]["healthy"] is False
        assert health["services"]["ml_pipeline"]["healthy"] is False
        assert health["dependencies"]["prometheus"]["healthy"] is False
        assert health["dependencies"]["grafana"]["healthy"] is False

    @patch("ml.dashboard.common.health_aggregator._safe_get")
    def test_get_system_health_returns_dict_structure(
        self,
        mock_safe_get: MagicMock,
        health_aggregator: HealthAggregatorComponent,
    ) -> None:
        """System health returns expected dictionary structure."""
        mock_safe_get.return_value = (True, 200)

        health = health_aggregator.get_system_health()

        assert isinstance(health, dict)
        assert "services" in health
        assert "dependencies" in health
        assert isinstance(health["services"], dict)
        assert isinstance(health["dependencies"], dict)


class TestListServices:
    """Test list_services method."""

    def test_list_services_returns_all_services(
        self, health_aggregator: HealthAggregatorComponent
    ) -> None:
        """List services returns all three ML services."""
        services = health_aggregator.list_services()

        assert len(services) == 3
        service_names = {s["name"] for s in services}
        assert service_names == {"ml_signal_actor", "ml_strategy", "ml_pipeline"}

    def test_list_services_includes_ports(
        self, health_aggregator: HealthAggregatorComponent
    ) -> None:
        """List services includes port information for each service."""
        services = health_aggregator.list_services()

        for service in services:
            assert "ports" in service
            assert "http" in service["ports"]
            assert isinstance(service["ports"]["http"], int)

    def test_list_services_includes_endpoints(
        self, health_aggregator: HealthAggregatorComponent
    ) -> None:
        """List services includes health and metrics endpoints."""
        services = health_aggregator.list_services()

        for service in services:
            assert "endpoints" in service
            assert "health" in service["endpoints"]
            assert "metrics" in service["endpoints"]
            assert service["endpoints"]["health"].endswith("/health")
            assert service["endpoints"]["metrics"].endswith("/metrics")

    def test_list_services_uses_correct_ports(
        self, dashboard_config: DashboardConfig
    ) -> None:
        """List services uses ports from config."""
        aggregator = HealthAggregatorComponent(config=dashboard_config)
        services = aggregator.list_services()

        service_map = {s["name"]: s for s in services}
        assert service_map["ml_signal_actor"]["ports"]["http"] == dashboard_config.actor_port
        assert service_map["ml_strategy"]["ports"]["http"] == dashboard_config.strategy_port
        assert service_map["ml_pipeline"]["ports"]["http"] == dashboard_config.pipeline_port


class TestGetStoreSummary:
    """Test get_store_summary method."""

    def test_get_store_summary_disabled_returns_disabled(
        self, dashboard_config: DashboardConfig
    ) -> None:
        """Store summary returns disabled status when store_health_enabled is False."""
        config = DashboardConfig(
            store_health_enabled=False,
            db_connection=TEST_DB_CONNECTION,
        )
        aggregator = HealthAggregatorComponent(config=config)

        summary = aggregator.get_store_summary()

        assert summary["ok"] is False
        assert summary["reason"] == "disabled"
        assert summary["stores"] == []

    def test_get_store_summary_no_db_connection(
        self, dashboard_config: DashboardConfig
    ) -> None:
        """Store summary returns error when no db_connection configured."""
        config = DashboardConfig(
            store_health_enabled=True,
            db_connection=None,
        )
        aggregator = HealthAggregatorComponent(config=config)

        summary = aggregator.get_store_summary()

        assert summary["ok"] is False
        assert summary["reason"] == "no_db_connection"
        assert summary["stores"] == []

    @patch("ml.core.db_engine.EngineManager.get_engine")
    @patch("ml.dashboard.store_health.summarize_all_stores")
    def test_get_store_summary_success(
        self,
        mock_summarize: MagicMock,
        mock_get_engine: MagicMock,
        health_aggregator: HealthAggregatorComponent,
    ) -> None:
        """Store summary aggregates stores successfully."""
        # Mock engine
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        # Mock store summary
        mock_summary = MagicMock()
        mock_summary.as_dict.return_value = {
            "store": "feature",
            "healthy": True,
            "total_records": 1000,
        }
        mock_summarize.return_value = [mock_summary]

        with patch("ml.stores.feature_store.FeatureStore"), \
             patch("ml.stores.model_store.ModelStore"), \
             patch("ml.stores.strategy_store.StrategyStore"):
            summary = health_aggregator.get_store_summary()

        assert summary["ok"] is True
        assert "generated_at" in summary
        assert len(summary["stores"]) == 1
        assert summary["stores"][0]["store"] == "feature"
        assert summary["stores"][0]["healthy"] is True

    @patch("ml.core.db_engine.EngineManager.get_engine")
    def test_get_store_summary_engine_failure_graceful(
        self,
        mock_get_engine: MagicMock,
        health_aggregator: HealthAggregatorComponent,
    ) -> None:
        """Store summary handles engine initialization failure gracefully."""
        mock_get_engine.side_effect = Exception("DB connection failed")

        # Should still attempt to summarize with engine=None
        with patch("ml.dashboard.store_health.summarize_all_stores") as mock_summarize:
            mock_summarize.return_value = []
            with patch("ml.stores.feature_store.FeatureStore"), \
                 patch("ml.stores.model_store.ModelStore"), \
                 patch("ml.stores.strategy_store.StrategyStore"):
                summary = health_aggregator.get_store_summary()

            # summarize_all_stores was called with engine=None
            call_kwargs = mock_summarize.call_args[1]
            assert call_kwargs["engine"] is None

    @patch("ml.core.db_engine.EngineManager.get_engine")
    @patch("ml.dashboard.store_health.summarize_all_stores")
    def test_get_store_summary_summarize_failure(
        self,
        mock_summarize: MagicMock,
        mock_get_engine: MagicMock,
        health_aggregator: HealthAggregatorComponent,
    ) -> None:
        """Store summary returns error when summarization fails."""
        mock_get_engine.return_value = MagicMock()
        mock_summarize.side_effect = Exception("Summarization error")

        with patch("ml.stores.feature_store.FeatureStore"), \
             patch("ml.stores.model_store.ModelStore"), \
             patch("ml.stores.strategy_store.StrategyStore"):
            summary = health_aggregator.get_store_summary()

        assert summary["ok"] is False
        assert summary["reason"] == "error"
        assert summary["stores"] == []


class TestSafeGetHelper:
    """Test _safe_get helper function."""

    @patch("ml.dashboard.common.health_aggregator.retry_with_backoff")
    def test_safe_get_success(self, mock_retry: MagicMock) -> None:
        """_safe_get returns success when request succeeds."""
        from ml.dashboard.common.health_aggregator import _safe_get

        mock_retry.return_value = (True, 200)

        ok, code = _safe_get("http://localhost:8000/health", timeout=2.5)

        assert ok is True
        assert code == 200

    @patch("ml.dashboard.common.health_aggregator.retry_with_backoff")
    def test_safe_get_failure(self, mock_retry: MagicMock) -> None:
        """_safe_get returns failure when request fails."""
        from ml.dashboard.common.health_aggregator import _safe_get

        mock_retry.side_effect = Exception("Connection timeout")

        ok, code = _safe_get("http://localhost:8000/health", timeout=2.5)

        assert ok is False
        assert code == 0


class TestToUrlHelper:
    """Test _to_url helper function."""

    def test_to_url_without_service_name(self) -> None:
        """_to_url builds localhost URL when no service name provided."""
        from ml.dashboard.common.health_aggregator import _to_url

        url = _to_url(8000, "/health", service_name=None)

        assert url == "http://localhost:8000/health"

    def test_to_url_with_service_name_no_env(self) -> None:
        """_to_url builds localhost URL when service name provided but no env var."""
        from ml.dashboard.common.health_aggregator import _to_url

        with patch.dict("os.environ", {}, clear=False):
            url = _to_url(8000, "/health", service_name="ml_signal_actor")

        assert url == "http://localhost:8000/health"

    def test_to_url_with_service_name_and_env(self) -> None:
        """_to_url uses environment variable URL when provided."""
        from ml.dashboard.common.health_aggregator import _to_url

        with patch.dict("os.environ", {"ML_SIGNAL_ACTOR_URL": "http://actor:8000"}, clear=False):
            url = _to_url(8000, "/health", service_name="ml_signal_actor")

        assert url == "http://actor:8000/health"


class TestProtocolCompliance:
    """Test protocol compliance."""

    def test_component_satisfies_protocol(
        self, health_aggregator: HealthAggregatorComponent
    ) -> None:
        """Component satisfies HealthAggregatorProtocol."""
        from ml.dashboard.common.health_aggregator import HealthAggregatorProtocol

        # Structural typing - check methods exist and callable
        assert callable(getattr(health_aggregator, "get_system_health", None))
        assert callable(getattr(health_aggregator, "list_services", None))
        assert callable(getattr(health_aggregator, "get_store_summary", None))

    def test_protocol_type_annotations(self) -> None:
        """Protocol methods have correct type annotations."""
        from ml.dashboard.common.health_aggregator import HealthAggregatorProtocol

        assert hasattr(HealthAggregatorProtocol, "get_system_health")
        assert hasattr(HealthAggregatorProtocol, "list_services")
        assert hasattr(HealthAggregatorProtocol, "get_store_summary")
