"""
Unit tests for GrafanaProvisionerComponent.

Tests cover:
- Dashboard provisioning (success, failure, force mode)
- Grafana status retrieval
- Prometheus summary queries
- Caching behavior
- Error handling and fallbacks
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.dashboard.common.grafana_provisioner import GrafanaProvisionerComponent
from ml.dashboard.config import DashboardConfig
from ml.dashboard.grafana import GrafanaProvisionResult


@pytest.fixture
def base_config() -> DashboardConfig:
    """Create base dashboard config for testing."""
    return DashboardConfig(
        grafana_url="http://localhost:3000",
        grafana_api_token="test-token",
        grafana_dashboard_uid="test-dashboard",
        grafana_dashboard_title="Test Dashboard",
        prometheus_url="http://localhost:9090",
        prometheus_query_timeout_seconds=2.0,
    )


@pytest.fixture
def config_no_prometheus() -> DashboardConfig:
    """Create config without Prometheus URL."""
    return DashboardConfig(
        grafana_url="http://localhost:3000",
        grafana_api_token="test-token",
        prometheus_url="",  # No Prometheus configured
    )


@pytest.fixture
def config_with_embed() -> DashboardConfig:
    """Create config with embed URLs enabled."""
    return DashboardConfig(
        grafana_url="http://localhost:3000",
        grafana_dashboard_uid="test-dashboard",
        grafana_dashboard_title="Test Dashboard",
        grafana_embed_enabled=True,
        grafana_embed_panels=(1, 2, 3),
        grafana_embed_theme="dark",
        grafana_embed_org_id=1,
    )


class TestGrafanaProvisionerInit:
    """Test component initialization."""

    def test_init_with_prometheus_url(self, base_config: DashboardConfig) -> None:
        """Test that Prometheus helper is initialized when URL is configured."""
        provisioner = GrafanaProvisionerComponent(base_config)

        assert provisioner.config == base_config
        assert provisioner._prometheus_helper is not None
        assert provisioner._prometheus_helper.base_url == "http://localhost:9090"
        assert provisioner._prometheus_helper.timeout_seconds == 2.0
        assert provisioner._grafana_status.ok is False
        assert provisioner._grafana_status.url is None

    def test_init_without_prometheus_url(self, config_no_prometheus: DashboardConfig) -> None:
        """Test that Prometheus helper is None when URL is not configured."""
        provisioner = GrafanaProvisionerComponent(config_no_prometheus)

        assert provisioner._prometheus_helper is None
        assert provisioner._grafana_status.ok is False


class TestProvisionGrafanaDashboard:
    """Test provision_grafana_dashboard method."""

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_provision_success(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test successful dashboard provisioning."""
        # Mock successful provisioning
        mock_provision.return_value = GrafanaProvisionResult(
            ok=True,
            url="http://localhost:3000/d/test-dashboard/test-dashboard",
            status_code=200,
            error=None,
        )

        provisioner = GrafanaProvisionerComponent(base_config)
        result = provisioner.provision_grafana_dashboard()

        assert result["ok"] is True
        assert "http://localhost:3000" in result["url"]
        assert result["status_code"] == 200
        assert result.get("error") is None
        assert "cached" not in result  # First provision, not cached

        # Verify internal state updated
        assert provisioner._grafana_status.ok is True
        assert provisioner._grafana_status.url == "http://localhost:3000/d/test-dashboard/test-dashboard"
        assert provisioner._grafana_status.status_code == 200

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_provision_cached(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test that cached status prevents redundant provisioning."""
        # First successful provisioning
        mock_provision.return_value = GrafanaProvisionResult(
            ok=True,
            url="http://localhost:3000/d/test-dashboard/test-dashboard",
            status_code=200,
            error=None,
        )

        provisioner = GrafanaProvisionerComponent(base_config)
        result1 = provisioner.provision_grafana_dashboard()
        assert result1["ok"] is True
        assert mock_provision.call_count == 1

        # Second call should use cache
        result2 = provisioner.provision_grafana_dashboard()
        assert result2["ok"] is True
        assert result2["cached"] is True
        assert result2["url"] == result1["url"]
        assert mock_provision.call_count == 1  # Not called again

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_provision_force_bypasses_cache(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test that force=True bypasses cache."""
        mock_provision.return_value = GrafanaProvisionResult(
            ok=True,
            url="http://localhost:3000/d/test-dashboard/test-dashboard",
            status_code=200,
            error=None,
        )

        provisioner = GrafanaProvisionerComponent(base_config)
        result1 = provisioner.provision_grafana_dashboard()
        assert result1["ok"] is True
        assert mock_provision.call_count == 1

        # Force re-provision
        result2 = provisioner.provision_grafana_dashboard(force=True)
        assert result2["ok"] is True
        assert "cached" not in result2
        assert mock_provision.call_count == 2  # Called again

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_provision_with_title_override(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test provisioning with custom title."""
        mock_provision.return_value = GrafanaProvisionResult(
            ok=True,
            url="http://localhost:3000/d/test-dashboard/custom-title",
            status_code=200,
            error=None,
        )

        provisioner = GrafanaProvisionerComponent(base_config)
        result = provisioner.provision_grafana_dashboard(title="Custom Title")

        assert result["ok"] is True
        # Verify provision_dashboard was called with title
        mock_provision.assert_called_once()
        call_kwargs = mock_provision.call_args.kwargs
        assert call_kwargs["title"] == "Custom Title"

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_provision_title_change_bypasses_cache(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test that title change bypasses cache."""
        mock_provision.return_value = GrafanaProvisionResult(
            ok=True,
            url="http://localhost:3000/d/test-dashboard/test-dashboard",
            status_code=200,
            error=None,
        )

        provisioner = GrafanaProvisionerComponent(base_config)
        result1 = provisioner.provision_grafana_dashboard()
        assert result1["ok"] is True
        assert mock_provision.call_count == 1

        # Different title should bypass cache
        result2 = provisioner.provision_grafana_dashboard(title="Different Title")
        assert result2["ok"] is True
        assert "cached" not in result2
        assert mock_provision.call_count == 2

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_provision_failure(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test provisioning failure handling."""
        mock_provision.return_value = GrafanaProvisionResult(
            ok=False,
            url=None,
            status_code=401,
            error="unauthorized",
        )

        provisioner = GrafanaProvisionerComponent(base_config)
        result = provisioner.provision_grafana_dashboard()

        assert result["ok"] is False
        assert result["url"] is None
        assert result["status_code"] == 401
        assert result["error"] == "unauthorized"

        # Verify internal state reflects failure
        assert provisioner._grafana_status.ok is False
        assert provisioner._grafana_status.error == "unauthorized"

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_provision_exception_handling(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test exception handling during provisioning."""
        mock_provision.side_effect = Exception("Network error")

        provisioner = GrafanaProvisionerComponent(base_config)
        result = provisioner.provision_grafana_dashboard()

        assert result["ok"] is False
        assert result["error"] == "exception"
        assert provisioner._grafana_status.error == "exception"

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_provision_exception_preserves_previous_url(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test that exception preserves URL from previous successful attempt."""
        # First successful provisioning
        mock_provision.return_value = GrafanaProvisionResult(
            ok=True,
            url="http://localhost:3000/d/test-dashboard/test-dashboard",
            status_code=200,
            error=None,
        )

        provisioner = GrafanaProvisionerComponent(base_config)
        result1 = provisioner.provision_grafana_dashboard()
        assert result1["ok"] is True
        previous_url = result1["url"]

        # Force re-provision that fails
        mock_provision.side_effect = Exception("Network error")
        result2 = provisioner.provision_grafana_dashboard(force=True)

        assert result2["ok"] is False
        assert result2["url"] == previous_url  # URL preserved


class TestGetGrafanaStatus:
    """Test get_grafana_status method."""

    def test_status_initial_state(self, base_config: DashboardConfig) -> None:
        """Test status before any provisioning attempt."""
        provisioner = GrafanaProvisionerComponent(base_config)
        status = provisioner.get_grafana_status()

        assert status["ok"] is False
        assert status["url"] == "http://localhost:3000/d/test-dashboard/test-dashboard"
        assert status["status_code"] is None
        assert status["error"] is None
        assert status["last_attempt_epoch"] is None
        assert status["embed_urls"] == ()

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_status_after_successful_provision(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test status reflects successful provisioning."""
        mock_provision.return_value = GrafanaProvisionResult(
            ok=True,
            url="http://localhost:3000/d/test-dashboard/test-dashboard",
            status_code=200,
            error=None,
        )

        provisioner = GrafanaProvisionerComponent(base_config)
        provisioner.provision_grafana_dashboard()

        status = provisioner.get_grafana_status()
        assert status["ok"] is True
        assert "http://localhost:3000" in status["url"]
        assert status["status_code"] == 200
        assert status["error"] is None
        assert isinstance(status["last_attempt_epoch"], float)

    def test_status_with_embed_urls(self, config_with_embed: DashboardConfig) -> None:
        """Test status includes embed URLs when configured."""
        provisioner = GrafanaProvisionerComponent(config_with_embed)
        status = provisioner.get_grafana_status()

        assert status["embed_urls"] != ()
        assert len(status["embed_urls"]) == 3
        for url in status["embed_urls"]:
            assert "d-solo" in url
            assert "theme=dark" in url
            assert "orgId=1" in url

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_status_after_failure(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test status reflects provisioning failure."""
        mock_provision.return_value = GrafanaProvisionResult(
            ok=False,
            url=None,
            status_code=500,
            error="internal error",
        )

        provisioner = GrafanaProvisionerComponent(base_config)
        provisioner.provision_grafana_dashboard()

        status = provisioner.get_grafana_status()
        assert status["ok"] is False
        assert status["status_code"] == 500
        assert status["error"] == "internal error"


class TestGetPrometheusSummary:
    """Test get_prometheus_summary method."""

    def test_summary_disabled_when_no_url(
        self, config_no_prometheus: DashboardConfig
    ) -> None:
        """Test summary returns disabled when Prometheus URL not configured."""
        provisioner = GrafanaProvisionerComponent(config_no_prometheus)
        summary = provisioner.get_prometheus_summary()

        assert summary["ok"] is False
        assert summary["metrics"] == {}
        assert summary["reason"] == "disabled"

    @patch("ml.dashboard.grafana.PrometheusQueryHelper.collect_scalars")
    def test_summary_success(
        self, mock_collect: Mock, base_config: DashboardConfig
    ) -> None:
        """Test successful Prometheus metrics collection."""
        mock_collect.return_value = {
            "request_rate_per_second": 10.5,
            "latency_p95_seconds": 0.025,
            "event_failures_increase": 0.0,
        }

        provisioner = GrafanaProvisionerComponent(base_config)
        summary = provisioner.get_prometheus_summary()

        assert summary["ok"] is True
        assert summary["metrics"]["request_rate_per_second"] == 10.5
        assert summary["metrics"]["latency_p95_seconds"] == 0.025
        assert summary["metrics"]["event_failures_increase"] == 0.0
        assert "updated_at" in summary
        assert isinstance(summary["updated_at"], float)

        # Verify queries were called
        mock_collect.assert_called_once()
        queries = mock_collect.call_args.args[0]
        assert "request_rate_per_second" in queries
        assert "latency_p95_seconds" in queries
        assert "event_failures_increase" in queries

    @patch("ml.dashboard.grafana.PrometheusQueryHelper.collect_scalars")
    def test_summary_with_null_metrics(
        self, mock_collect: Mock, base_config: DashboardConfig
    ) -> None:
        """Test summary when Prometheus returns None for some metrics."""
        mock_collect.return_value = {
            "request_rate_per_second": None,  # Query failed
            "latency_p95_seconds": 0.025,
            "event_failures_increase": None,
        }

        provisioner = GrafanaProvisionerComponent(base_config)
        summary = provisioner.get_prometheus_summary()

        assert summary["ok"] is True
        assert summary["metrics"]["request_rate_per_second"] is None
        assert summary["metrics"]["latency_p95_seconds"] == 0.025
        assert summary["metrics"]["event_failures_increase"] is None

    @patch("ml.dashboard.grafana.PrometheusQueryHelper.collect_scalars")
    def test_summary_exception_handling(
        self, mock_collect: Mock, base_config: DashboardConfig
    ) -> None:
        """Test exception handling during metrics collection."""
        mock_collect.side_effect = Exception("Connection timeout")

        provisioner = GrafanaProvisionerComponent(base_config)
        summary = provisioner.get_prometheus_summary()

        assert summary["ok"] is False
        assert summary["metrics"] == {}
        assert summary["reason"] == "error"


class TestBuildGrafanaConfig:
    """Test _build_grafana_config helper method."""

    def test_build_config_all_fields(self, base_config: DashboardConfig) -> None:
        """Test that all config fields are correctly mapped."""
        provisioner = GrafanaProvisionerComponent(base_config)
        grafana_config = provisioner._build_grafana_config()

        assert grafana_config.url == "http://localhost:3000"
        assert grafana_config.api_token == "test-token"
        assert grafana_config.dashboard_uid == "test-dashboard"
        assert grafana_config.dashboard_title == "Test Dashboard"

    def test_build_config_with_credentials(self) -> None:
        """Test config builder with username/password."""
        config = DashboardConfig(
            grafana_url="http://grafana:3000",
            grafana_username="admin",
            grafana_password="secret",
            grafana_folder_uid="folder-123",
            grafana_datasource_uid="prom-ds",
            grafana_refresh_interval="1m",
        )
        provisioner = GrafanaProvisionerComponent(config)
        grafana_config = provisioner._build_grafana_config()

        assert grafana_config.username == "admin"
        assert grafana_config.password == "secret"
        assert grafana_config.folder_uid == "folder-123"
        assert grafana_config.datasource_uid == "prom-ds"
        assert grafana_config.refresh_interval == "1m"


class TestMetricsRecording:
    """Test that metrics are properly recorded."""

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    @patch("ml.dashboard.common.grafana_provisioner._REQS_TOTAL")
    @patch("ml.dashboard.common.grafana_provisioner._LATENCY_SECONDS")
    def test_provision_records_metrics_on_success(
        self,
        mock_latency: Mock,
        mock_counter: Mock,
        mock_provision: Mock,
        base_config: DashboardConfig,
    ) -> None:
        """Test that successful provisioning records metrics."""
        mock_provision.return_value = GrafanaProvisionResult(
            ok=True,
            url="http://localhost:3000/d/test-dashboard/test-dashboard",
            status_code=200,
        )

        provisioner = GrafanaProvisionerComponent(base_config)
        provisioner.provision_grafana_dashboard()

        # Verify counter incremented
        mock_counter.labels.assert_called()
        # Verify latency recorded
        mock_latency.labels.assert_called()

    @patch("ml.dashboard.common.grafana_provisioner._REQS_TOTAL")
    @patch("ml.dashboard.grafana.PrometheusQueryHelper.collect_scalars")
    def test_summary_records_metrics_on_disabled(
        self,
        mock_collect: Mock,
        mock_counter: Mock,
        config_no_prometheus: DashboardConfig,
    ) -> None:
        """Test that disabled Prometheus records appropriate metric."""
        provisioner = GrafanaProvisionerComponent(config_no_prometheus)
        provisioner.get_prometheus_summary()

        # Verify disabled status recorded
        mock_counter.labels.assert_called()


class TestProtocolConformance:
    """Test that component conforms to GrafanaProvisionerProtocol."""

    def test_component_has_all_protocol_methods(
        self, base_config: DashboardConfig
    ) -> None:
        """Test that component implements all protocol methods."""
        from ml.dashboard.common.grafana_provisioner import (
            GrafanaProvisionerProtocol,
        )

        provisioner = GrafanaProvisionerComponent(base_config)

        # Verify protocol methods exist
        assert hasattr(provisioner, "provision_grafana_dashboard")
        assert callable(provisioner.provision_grafana_dashboard)
        assert hasattr(provisioner, "get_grafana_status")
        assert callable(provisioner.get_grafana_status)
        assert hasattr(provisioner, "get_prometheus_summary")
        assert callable(provisioner.get_prometheus_summary)

        # Verify structural typing works
        def accepts_protocol(p: GrafanaProvisionerProtocol) -> None:
            p.provision_grafana_dashboard()
            p.get_grafana_status()
            p.get_prometheus_summary()

        accepts_protocol(provisioner)  # Should not raise


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_provision_with_empty_title(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test provisioning with empty string title."""
        mock_provision.return_value = GrafanaProvisionResult(
            ok=True,
            url="http://localhost:3000/d/test-dashboard/test-dashboard",
            status_code=200,
        )

        provisioner = GrafanaProvisionerComponent(base_config)
        result = provisioner.provision_grafana_dashboard(title="")

        assert result["ok"] is True
        # Verify empty title was passed through
        call_kwargs = mock_provision.call_args.kwargs
        assert call_kwargs["title"] == ""

    @patch("ml.dashboard.common.grafana_provisioner.provision_dashboard")
    def test_provision_timing_consistency(
        self, mock_provision: Mock, base_config: DashboardConfig
    ) -> None:
        """Test that timing is recorded for both success and failure."""
        mock_provision.return_value = GrafanaProvisionResult(ok=True, url="test")

        provisioner = GrafanaProvisionerComponent(base_config)
        start = time.time()
        provisioner.provision_grafana_dashboard()
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should be fast when mocked
        assert provisioner._grafana_status.last_attempt_epoch is not None

    def test_config_url_trailing_slash_handling(self) -> None:
        """Test that URLs with trailing slashes are handled correctly."""
        config = DashboardConfig(
            grafana_url="http://localhost:3000/",  # Trailing slash
            prometheus_url="http://localhost:9090/",  # Trailing slash
        )
        provisioner = GrafanaProvisionerComponent(config)

        # Should handle URLs correctly regardless of trailing slash
        assert provisioner._prometheus_helper is not None
