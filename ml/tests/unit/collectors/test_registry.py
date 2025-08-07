# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Tests for the MLMetricsRegistry.
"""

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.registry import MLMetricsRegistry


class TestMLMetricsRegistry:
    """
    Test suite for MLMetricsRegistry functionality.
    """

    @pytest.fixture(autouse=True)
    def setup_test(self, prometheus_registry_cleanup, metric_name_manager):
        """
        Set up test with proper cleanup and unique names.
        """
        self.metric_name_manager = metric_name_manager

    @pytest.fixture
    def config(self):
        """
        Create test configuration.
        """
        prefix = self.metric_name_manager.get_unique_name("ml")
        return MonitoringConfig(
            enabled=True,
            metrics_port=8081,  # Use different port to avoid conflicts
            metrics_prefix=prefix,
        )

    @pytest.fixture
    def disabled_config(self):
        """
        Create disabled configuration.
        """
        return MonitoringConfig(enabled=False)

    @pytest.fixture
    def registry(self, config):
        """
        Create MLMetricsRegistry for testing.
        """
        return MLMetricsRegistry(config, enable_background_monitoring=False)

    @pytest.fixture
    def disabled_registry(self, disabled_config):
        """
        Create disabled MLMetricsRegistry for testing.
        """
        return MLMetricsRegistry(disabled_config, enable_background_monitoring=False)

    def test_initialization(self, registry, config):
        """
        Test registry initialization.
        """
        assert registry.config == config
        assert not registry.started
        assert registry.enabled == HAS_PROMETHEUS

        # Check that all collectors are created
        collectors = registry.list_collectors()
        expected_collectors = ["ml", "model", "data", "features", "performance", "resources"]

        for expected in expected_collectors:
            assert expected in collectors

    def test_disabled_initialization(self, disabled_registry):
        """
        Test disabled registry initialization.
        """
        assert not disabled_registry.enabled
        assert not disabled_registry.started

    def test_get_collector(self, registry):
        """
        Test getting specific collectors.
        """
        # Test valid collectors
        data_collector = registry.get_collector("data")
        assert data_collector is not None

        model_collector = registry.get_collector("model")
        assert model_collector is not None

        # Test invalid collector type
        with pytest.raises(ValueError, match="Unknown collector type"):
            registry.get_collector("invalid")

    def test_list_collectors(self, registry):
        """
        Test listing all collector types.
        """
        collectors = registry.list_collectors()

        assert isinstance(collectors, list)
        assert len(collectors) == 6
        assert "ml" in collectors
        assert "data" in collectors
        assert "features" in collectors

    def test_get_all_collectors(self, registry):
        """
        Test getting all collectors as dictionary.
        """
        all_collectors = registry.get_all_collectors()

        assert isinstance(all_collectors, dict)
        assert len(all_collectors) == 6

        # Verify it returns a copy (modifications don't affect original)
        all_collectors["test"] = "value"
        assert "test" not in registry.get_all_collectors()

    def test_health_check(self, registry):
        """
        Test comprehensive health check.
        """
        health = registry.health_check()

        assert "status" in health
        assert "started" in health
        assert "enabled_collectors" in health
        assert "total_collectors" in health
        assert "total_metrics" in health
        assert "server_port" in health
        assert "background_monitoring" in health
        assert "collectors" in health

        assert health["total_collectors"] == 6
        assert health["server_port"] == 8081
        assert health["started"] is False  # Not started yet

    def test_start_stop_lifecycle(self, registry):
        """
        Test start/stop lifecycle.
        """
        assert not registry.started

        # Start registry
        registry.start()
        assert registry.started

        # Starting again should be idempotent
        registry.start()
        assert registry.started

        # Stop registry
        registry.stop()
        assert not registry.started

        # Stopping again should be idempotent
        registry.stop()
        assert not registry.started

    def test_context_manager(self, registry):
        """
        Test registry as context manager.
        """
        assert not registry.started

        with registry as reg:
            assert reg.started
            assert reg is registry

        assert not registry.started

    def test_reset_all_metrics(self, registry):
        """
        Test resetting all metrics across collectors.
        """
        # This should complete without errors
        registry.reset_all_metrics()

    def test_get_metrics_summary(self, registry):
        """
        Test getting metrics summary.
        """
        summary = registry.get_metrics_summary()

        assert "registry_status" in summary
        assert "total_collectors" in summary
        assert "enabled_collectors" in summary

        assert summary["total_collectors"] == 6
        assert summary["registry_status"] == "stopped"

    def test_configure_collector_not_implemented(self, registry):
        """
        Test that collector configuration raises NotImplementedError.
        """
        with pytest.raises(NotImplementedError):
            registry.configure_collector("data", param1="value1")

    def test_prometheus_registry_access(self, registry):
        """
        Test accessing Prometheus registry.
        """
        prom_registry = registry.get_prometheus_registry()

        if HAS_PROMETHEUS:
            assert prom_registry is not None
        else:
            assert prom_registry is None

    def test_export_metrics(self, registry):
        """
        Test exporting metrics in Prometheus format.
        """
        metrics_text = registry.export_metrics()

        if HAS_PROMETHEUS and registry.enabled:
            assert isinstance(metrics_text, str)
            # Should contain some metric lines if enabled
        else:
            assert metrics_text == ""

    def test_disabled_registry_operations(self, disabled_registry):
        """
        Test that disabled registry operations work correctly.
        """
        assert not disabled_registry.enabled

        # All operations should complete without error
        disabled_registry.start()
        disabled_registry.stop()

        health = disabled_registry.health_check()
        assert health["enabled_collectors"] == 0  # All disabled

        summary = disabled_registry.get_metrics_summary()
        assert summary["enabled_collectors"] == 0

        # Context manager should work
        with disabled_registry:
            pass

    def test_string_representation(self, registry):
        """
        Test string representation.
        """
        repr_str = repr(registry)

        assert "MLMetricsRegistry" in repr_str
        assert "started=False" in repr_str
        assert "server_port=8081" in repr_str

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_enabled_registry_with_prometheus(self, registry):
        """
        Test enabled registry when Prometheus is available.
        """
        assert registry.enabled

        health = registry.health_check()
        assert health["enabled_collectors"] > 0

        summary = registry.get_metrics_summary()
        assert summary["enabled_collectors"] > 0

    def test_background_monitoring_flag(self, config):
        """
        Test background monitoring flag.
        """
        # Test with background monitoring enabled
        registry_with_bg = MLMetricsRegistry(config, enable_background_monitoring=True)
        assert registry_with_bg._enable_background_monitoring

        # Test with background monitoring disabled - use different prefix
        prefix2 = self.metric_name_manager.get_unique_name("ml2")
        config2 = MonitoringConfig(
            enabled=True,
            metrics_port=8082,  # Different port
            metrics_prefix=prefix2,
        )
        registry_no_bg = MLMetricsRegistry(config2, enable_background_monitoring=False)
        assert not registry_no_bg._enable_background_monitoring

    def test_collector_health_in_registry(self, registry):
        """
        Test that individual collector health is included in registry health.
        """
        health = registry.health_check()

        assert "collectors" in health
        collectors_health = health["collectors"]

        # Should have health info for each collector
        for collector_type in ["ml", "data", "features", "model", "performance", "resources"]:
            assert collector_type in collectors_health
            collector_health = collectors_health[collector_type]

            assert "enabled" in collector_health
            assert "metrics_count" in collector_health
            assert "collector_type" in collector_health
