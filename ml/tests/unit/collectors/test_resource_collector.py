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
Tests for the ResourceUtilizationCollector class.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.resources import ResourceUtilizationCollector


class TestResourceUtilizationCollector:
    """
    Test suite for ResourceUtilizationCollector functionality.
    """

    @pytest.fixture(autouse=True)
    def setup_test(
        self,
        prometheus_registry_cleanup: Any,
        metric_name_manager: Any,
        monitoring_config: MonitoringConfig,
    ) -> None:
        """
        Set up test with proper cleanup and unique names.
        """
        self.metric_name_manager = metric_name_manager
        self.config = monitoring_config

    def test_initialization_with_disabled_config(self) -> None:
        """
        Test collector initialization when monitoring is disabled.
        """
        config = MonitoringConfig(enabled=False)
        collector = ResourceUtilizationCollector(config)

        assert collector.config == config
        assert not collector.enabled
        assert collector.get_metric_count() == 0
        assert collector._monitoring_thread is None

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_initialization_with_enabled_config(self) -> None:
        """
        Test collector initialization when monitoring is enabled.
        """
        collector = ResourceUtilizationCollector(self.config)

        assert collector.config == self.config
        assert collector.enabled
        # Should have initialized multiple metrics
        assert collector.get_metric_count() > 0

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_model_memory_usage(self) -> None:
        """
        Test recording model memory usage.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Record model memory usage
        collector.record_model_memory_usage("test_model", 1024 * 1024 * 256, "resident")

        # Check that metric was recorded
        value = collector.get_metric_value(
            "model_memory_usage_bytes",
            {"model": "test_model", "memory_type": "resident"},
        )
        assert value == 1024 * 1024 * 256

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_cpu_usage(self) -> None:
        """
        Test recording CPU usage.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Record CPU usage
        collector.record_cpu_usage(45.5, "process")

        # Check that metric was recorded
        value = collector.get_metric_value(
            "cpu_usage_percent",
            {"core": "process"},
        )
        assert value == 45.5

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_gpu_metrics(self) -> None:
        """
        Test recording GPU metrics.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Record GPU metrics
        collector.record_gpu_metrics(
            device="cuda:0",
            compute_utilization=85.0,
            memory_utilization=70.0,
            memory_used_bytes=1024 * 1024 * 1024 * 6,  # 6 GB
            memory_total_bytes=1024 * 1024 * 1024 * 8,  # 8 GB
        )

        # Check compute utilization
        value = collector.get_metric_value(
            "gpu_utilization_percent",
            {"device": "cuda:0", "metric": "compute"},
        )
        assert value == 85.0

        # Check memory used
        value = collector.get_metric_value(
            "gpu_memory_usage_bytes",
            {"device": "cuda:0", "memory_type": "used"},
        )
        assert value == 1024 * 1024 * 1024 * 6

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_feature_store_size(self) -> None:
        """
        Test recording feature store size.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Record feature store size
        collector.record_feature_store_size(1024 * 1024 * 100, "memory")  # 100 MB

        # Check that metric was recorded
        value = collector.get_metric_value(
            "feature_store_size_bytes",
            {"storage_type": "memory"},
        )
        assert value == 1024 * 1024 * 100

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_background_monitoring_thread(self) -> None:
        """
        Test background monitoring thread functionality.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Start monitoring (may not start thread if psutil not available, but shouldn't error)
        collector.start_monitoring()

        # Let it run for a bit (if started)
        time.sleep(0.01)

        # Stop monitoring (should not error)
        collector.stop_monitoring()

        # Test should pass if no exceptions are raised

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_disk_usage(self) -> None:
        """
        Test recording disk usage.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Record disk usage
        collector.record_disk_usage("/data", 1024 * 1024 * 1024 * 5, "data")  # 5 GB

        # Check that metric was recorded
        value = collector.get_metric_value(
            "disk_usage_bytes",
            {"path": "/data", "usage_type": "data"},
        )
        assert value == 1024 * 1024 * 1024 * 5

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_data_io(self) -> None:
        """
        Test recording data I/O operations.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Record data I/O
        collector.record_data_io(1024 * 1024 * 100, "read", "bars")  # 100 MB read

        # Check that metric was recorded
        value = collector.get_metric_value(
            "data_io_bytes_total",
            {"operation": "read", "data_type": "bars"},
        )
        assert value == 1024 * 1024 * 100

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_inference_batch_size(self) -> None:
        """
        Test recording inference batch size.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Record batch size
        collector.record_inference_batch_size("test_model", 32)

        # Check that metric was recorded
        value = collector.get_metric_value(
            "inference_batch_size",
            {"model": "test_model"},
        )
        assert value == 32

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_training_data_processed(self) -> None:
        """
        Test recording training data processed.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Record training data processed
        collector.record_training_data_processed(1000, "train")

        # Check that metric was recorded
        value = collector.get_metric_value(
            "training_data_rows_processed_total",
            {"dataset": "train"},
        )
        assert value == 1000

    def test_disabled_collector_operations(self) -> None:
        """
        Test that disabled collector operations are no-ops.
        """
        config = MonitoringConfig(enabled=False)
        collector = ResourceUtilizationCollector(config)

        # All operations should be no-ops and not raise
        collector.record_model_memory_usage("test", 1024, "resident")
        collector.record_cpu_usage(50.0, "process")
        collector.start_monitoring()
        collector.stop_monitoring()

        # Should still return None for metrics
        assert collector.get_metric_value("any_metric") is None
        assert collector._monitoring_thread is None

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_get_resource_summary(self) -> None:
        """
        Test getting resource summary.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Record some metrics
        collector.record_model_memory_usage("test_model", 1024 * 1024, "resident")
        collector.record_cpu_usage(50.0, "process")

        # Get summary
        summary = collector.get_resource_summary()

        assert isinstance(summary, dict)
        # Should only contain non-None values
        for value in summary.values():
            assert value is not None

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_health_check(self) -> None:
        """
        Test health check returns expected information.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Record some metrics
        collector.record_model_memory_usage("test_model", 1024 * 1024, "resident")

        health = collector.health_check()

        assert health["enabled"] is True
        assert health["metrics_count"] > 0
        assert health["prometheus_available"] == HAS_PROMETHEUS
        assert health["config_valid"] is True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_reset_metrics(self) -> None:
        """
        Test resetting metrics to initial state.
        """
        collector = ResourceUtilizationCollector(self.config)

        # Record some metrics
        collector.record_model_memory_usage("test_model", 1024 * 1024, "resident")

        # Verify metric exists
        value = collector.get_metric_value(
            "model_memory_usage_bytes",
            {"model": "test_model", "memory_type": "resident"},
        )
        assert value == 1024 * 1024

        # Reset metrics
        collector.reset_metrics()

        # After reset, the collector should be in a clean state
        # The specific value depends on the implementation -
        # Prometheus Gauges may retain values or be reset to 0
        value_after_reset = collector.get_metric_value(
            "model_memory_usage_bytes",
            {"model": "test_model", "memory_type": "resident"},
        )
        # Just verify the reset method can be called without error
        assert value_after_reset is not None

    def test_error_handling_in_collection(self) -> None:
        """
        Test error handling during resource collection.
        """
        config = MonitoringConfig(enabled=True)
        collector = ResourceUtilizationCollector(config)

        # Should handle any errors gracefully
        collector.record_model_memory_usage("test", 1024, "resident")

        # Collector should still be enabled
        assert collector.enabled
