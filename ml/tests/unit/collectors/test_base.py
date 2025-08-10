
"""
Tests for the BaseMetricsCollector abstract class.
"""

from typing import Any

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.base import BaseMetricsCollector


class TestBaseMetricsCollector:
    """
    Test suite for BaseMetricsCollector functionality.
    """

    @pytest.fixture(autouse=True)
    def setup_test(self, prometheus_registry_cleanup: Any, metric_name_manager: Any) -> None:
        """
        Set up test with proper cleanup and unique names.
        """
        self.metric_name_manager = metric_name_manager

    def test_initialization_with_disabled_config(self) -> None:
        """
        Test collector initialization when monitoring is disabled.
        """
        config = MonitoringConfig(enabled=False)

        # Create a concrete implementation for testing
        class ConcreteCollector(BaseMetricsCollector):
            def _initialize_metrics(self) -> None:
                pass

        collector = ConcreteCollector(config)

        assert collector.config == config
        assert not collector.enabled
        assert collector.get_metric_count() == 0
        assert collector.metrics == {}

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_initialization_with_enabled_config(self) -> None:
        """
        Test collector initialization when monitoring is enabled.
        """
        config = MonitoringConfig(enabled=True)

        class ConcreteCollector(BaseMetricsCollector):
            def __init__(self, config: MonitoringConfig, name_manager: Any) -> None:
                self.name_manager = name_manager
                super().__init__(config)

            def _initialize_metrics(self) -> None:
                from ml._imports import Counter

                counter_name = self.name_manager.get_unique_name("counter")
                counter = Counter(counter_name, "Test counter")
                self._register_metric("test_counter", counter)

        collector = ConcreteCollector(config, self.metric_name_manager)

        assert collector.config == config
        assert collector.enabled
        assert collector.get_metric_count() == 1
        assert "test_counter" in collector.metrics

    def test_health_check_basic_functionality(self) -> None:
        """
        Test health check returns expected information.
        """
        config = MonitoringConfig(enabled=True)

        class ConcreteCollector(BaseMetricsCollector):
            def _initialize_metrics(self) -> None:
                pass

        collector = ConcreteCollector(config)
        health = collector.health_check()

        assert "enabled" in health
        assert "metrics_count" in health
        assert "prometheus_available" in health
        assert "config_valid" in health
        assert "collector_type" in health

        assert health["collector_type"] == "ConcreteCollector"
        assert health["prometheus_available"] == HAS_PROMETHEUS

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_metric_registration_and_retrieval(self) -> None:
        """
        Test metric registration and value retrieval.
        """
        config = MonitoringConfig(enabled=True)

        class ConcreteCollector(BaseMetricsCollector):
            def __init__(self, config: MonitoringConfig, name_manager: Any) -> None:
                self.name_manager = name_manager
                super().__init__(config)

            def _initialize_metrics(self) -> None:
                from ml._imports import Gauge

                gauge_name = self.name_manager.get_unique_name("gauge")
                gauge = Gauge(gauge_name, "Test gauge")
                self._register_metric("test_gauge", gauge)
                gauge.set(42.0)

        collector = ConcreteCollector(config, self.metric_name_manager)

        # Test metric retrieval
        value = collector.get_metric_value("test_gauge")
        assert value == 42.0

        # Test non-existent metric
        assert collector.get_metric_value("non_existent") is None

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_labeled_metric_retrieval(self) -> None:
        """
        Test retrieval of labeled metrics.
        """
        config = MonitoringConfig(enabled=True)

        class ConcreteCollector(BaseMetricsCollector):
            def __init__(self, config: MonitoringConfig, name_manager: Any) -> None:
                self.name_manager = name_manager
                super().__init__(config)

            def _initialize_metrics(self) -> None:
                from ml._imports import Gauge

                gauge_name = self.name_manager.get_unique_name("labeled_gauge")
                gauge = Gauge(gauge_name, "Test labeled gauge", ["label1"])
                self._register_metric("test_labeled_gauge", gauge)
                gauge.labels(label1="value1").set(100.0)

        collector = ConcreteCollector(config, self.metric_name_manager)

        # Test labeled metric retrieval
        value = collector.get_metric_value("test_labeled_gauge", {"label1": "value1"})
        assert value == 100.0

        # Test non-existent label combination
        value = collector.get_metric_value("test_labeled_gauge", {"label1": "nonexistent"})
        assert value == 0.0  # Default value for unset gauge

    def test_safe_record_error_handling(self) -> None:
        """
        Test that _safe_record handles errors gracefully.
        """
        config = MonitoringConfig(enabled=True)

        class ConcreteCollector(BaseMetricsCollector):
            def _initialize_metrics(self) -> None:
                pass

        collector = ConcreteCollector(config)

        # This should not raise an exception
        def failing_operation() -> None:
            raise ValueError("Test error")

        collector._safe_record("test_operation", failing_operation)
        # Should complete without raising

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_reset_metrics_functionality(self) -> None:
        """
        Test metrics reset functionality.
        """
        config = MonitoringConfig(enabled=True)

        class ConcreteCollector(BaseMetricsCollector):
            def __init__(self, config: MonitoringConfig, name_manager: Any) -> None:
                self.name_manager = name_manager
                super().__init__(config)

            def _initialize_metrics(self) -> None:
                from ml._imports import Counter
                from ml._imports import Gauge

                counter_name = self.name_manager.get_unique_name("counter")
                gauge_name = self.name_manager.get_unique_name("gauge")
                counter = Counter(counter_name, "Test counter")
                gauge = Gauge(gauge_name, "Test gauge")
                self._register_metric("test_counter", counter)
                self._register_metric("test_gauge", gauge)

                # Set some values
                counter.inc(5)
                gauge.set(10.0)

        collector = ConcreteCollector(config, self.metric_name_manager)

        # Verify initial values
        assert collector.get_metric_value("test_gauge") == 10.0

        # Reset metrics
        collector.reset_metrics()

        # Gauge should be reset to 0
        assert collector.get_metric_value("test_gauge") == 0.0

    def test_config_validation(self) -> None:
        """
        Test configuration validation.
        """
        config = MonitoringConfig(enabled=True, metrics_prefix="test")

        class ConcreteCollector(BaseMetricsCollector):
            def _initialize_metrics(self) -> None:
                pass

        collector = ConcreteCollector(config)
        assert collector._validate_config()

        # Test with invalid config (missing attributes)
        class InvalidCollector(BaseMetricsCollector):
            def _initialize_metrics(self) -> None:
                pass

            def _validate_config(self) -> bool:
                # Simulate missing attributes
                return hasattr(self._config, "nonexistent_attr")

        invalid_collector = InvalidCollector(config)
        assert not invalid_collector._validate_config()

    def test_string_representation(self) -> None:
        """
        Test string representation of collector.
        """
        config = MonitoringConfig(enabled=True)

        class ConcreteCollector(BaseMetricsCollector):
            def _initialize_metrics(self) -> None:
                pass

        collector = ConcreteCollector(config)
        repr_str = repr(collector)

        assert "ConcreteCollector" in repr_str
        assert "enabled=True" in repr_str
        assert "metrics_count=0" in repr_str

    def test_disabled_collector_behavior(self) -> None:
        """
        Test that disabled collectors handle operations gracefully.
        """
        config = MonitoringConfig(enabled=False)

        class ConcreteCollector(BaseMetricsCollector):
            def _initialize_metrics(self) -> None:
                # Should not be called when disabled
                raise RuntimeError("Should not initialize when disabled")

        collector = ConcreteCollector(config)

        # All operations should be no-ops
        assert not collector.enabled
        assert collector.get_metric_value("any_metric") is None
        collector.reset_metrics()  # Should not raise
        collector._safe_record("test", lambda: None)  # Should not raise

        health = collector.health_check()
        assert not health["enabled"]
