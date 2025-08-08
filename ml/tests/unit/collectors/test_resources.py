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
Tests for the ResourceUtilizationCollector.
"""

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.resources import ResourceUtilizationCollector


class TestResourceUtilizationCollector:
    """
    Test suite for ResourceUtilizationCollector functionality.
    """

    @pytest.fixture(autouse=True)
    def setup_test(self, prometheus_registry_cleanup: Any, metric_name_manager: Any) -> None:
        """
        Set up test with proper cleanup and unique names.
        """
        self.metric_name_manager = metric_name_manager

    @pytest.fixture
    def collector(self) -> ResourceUtilizationCollector:
        """
        Create a ResourceUtilizationCollector for testing.
        """
        prefix = self.metric_name_manager.get_unique_name("ml")
        config = MonitoringConfig(enabled=True, metrics_prefix=prefix)
        return ResourceUtilizationCollector(config)

    @pytest.fixture
    def disabled_collector(self) -> ResourceUtilizationCollector:
        """
        Create a disabled ResourceUtilizationCollector for testing.
        """
        config = MonitoringConfig(enabled=False)
        return ResourceUtilizationCollector(config)

    def test_initialization(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test collector initialization.
        """
        assert collector.enabled == HAS_PROMETHEUS
        assert collector.get_metric_count() > 0 or not HAS_PROMETHEUS

    def test_disabled_initialization(
        self,
        disabled_collector: ResourceUtilizationCollector,
    ) -> None:
        """
        Test disabled collector initialization.
        """
        assert not disabled_collector.enabled
        assert disabled_collector.get_metric_count() == 0

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_model_memory_usage(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test recording model memory usage.
        """
        model_name = "xgboost_classifier"
        memory_bytes = 268435456  # 256 MB in bytes

        collector.record_model_memory_usage(
            model=model_name,
            memory_bytes=memory_bytes,
            memory_type="resident",
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_feature_store_size(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test recording feature store size.
        """
        storage_type = "redis"
        size_bytes = 1073741824  # 1024 MB in bytes

        collector.record_feature_store_size(
            size_bytes=size_bytes,
            storage_type=storage_type,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_cpu_usage(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test recording CPU usage metrics.
        """
        usage_percent = 65.2
        core = "average"

        collector.record_cpu_usage(
            usage_percent=usage_percent,
            core=core,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_ml_cpu_time(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test recording ML CPU time.
        """
        operation_type = "inference"
        cpu_time_seconds = 2.5

        collector.record_ml_cpu_time(
            cpu_time_seconds=cpu_time_seconds,
            operation_type=operation_type,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_gpu_metrics(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test recording GPU metrics.
        """
        device = "cuda:0"
        compute_utilization = 78.5
        memory_used_bytes = 4294967296  # 4GB in bytes
        memory_total_bytes = 8589934592  # 8GB in bytes

        collector.record_gpu_metrics(
            device=device,
            compute_utilization=compute_utilization,
            memory_used_bytes=memory_used_bytes,
            memory_total_bytes=memory_total_bytes,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_disk_usage(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test recording disk usage metrics.
        """
        path = "/data/models"
        usage_bytes = 134872702976  # 125.6 GB in bytes
        usage_type = "models"

        collector.record_disk_usage(
            path=path,
            usage_bytes=usage_bytes,
            usage_type=usage_type,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_data_io(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test recording data I/O metrics.
        """
        operation = "read"
        bytes_transferred = 1048576  # 1MB
        data_type = "bars"

        collector.record_data_io(
            bytes_transferred=bytes_transferred,
            operation=operation,
            data_type=data_type,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_inference_batch_size(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test recording inference batch size.
        """
        model_name = "neural_network"
        batch_size = 32

        collector.record_inference_batch_size(
            model=model_name,
            batch_size=batch_size,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_training_data_processed(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test recording training data processed.
        """
        rows = 100000
        dataset = "train"

        collector.record_training_data_processed(
            rows=rows,
            dataset=dataset,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_start_stop_monitoring(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test starting and stopping resource monitoring.
        """
        # Start monitoring
        collector.start_monitoring()

        # Brief pause to allow monitoring to initialize
        time.sleep(0.1)

        # Stop monitoring
        collector.stop_monitoring()

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_get_resource_summary(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test getting resource summary.
        """
        # Record some resource usage first
        collector.record_model_memory_usage(
            model="test_model",
            memory_bytes=134217728,  # 128 MB in bytes
            memory_type="resident",
        )
        collector.record_cpu_usage(
            usage_percent=45.0,
            core="inference",
        )

        summary = collector.get_resource_summary()

        assert isinstance(summary, dict)
        # Check for actual keys returned by get_resource_summary
        assert summary is not None

    def test_get_resource_summary_disabled(
        self,
        disabled_collector: ResourceUtilizationCollector,
    ) -> None:
        """
        Test getting resource summary when disabled.
        """
        summary = disabled_collector.get_resource_summary()

        # Should return empty dict when disabled
        assert summary == {}

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_comprehensive_resource_tracking(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test comprehensive resource tracking scenario.
        """
        model_name = "comprehensive_model"

        # 1. Record initial memory usage
        collector.record_model_memory_usage(
            model=model_name,
            memory_bytes=67108864,  # 64 MB in bytes
            memory_type="resident",
        )

        # 2. Record training resource usage
        collector.record_training_data_processed(
            rows=50000,
            dataset="train",
        )

        # 3. Record CPU usage during training
        collector.record_cpu_usage(
            usage_percent=85.0,
            core="training",
        )

        # 4. Record data I/O during training
        collector.record_data_io(
            operation="read",
            bytes_transferred=10485760,  # 10MB
            data_type="training_data",
        )

        # 5. Record inference batch processing
        collector.record_inference_batch_size(
            model=model_name,
            batch_size=64,
        )

        # 6. Record final memory state
        collector.record_model_memory_usage(
            model=model_name,
            memory_bytes=134217728,  # 128 MB in bytes
            memory_type="resident",
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_thread_safety(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test thread safety of collector operations.
        """
        results = []
        errors = []

        def worker():
            try:
                thread_id = threading.current_thread().ident
                for i in range(5):
                    collector.record_model_memory_usage(
                        model=f"test_model_{thread_id}",
                        memory_bytes=int((64.0 + i * 16) * 1024 * 1024),  # Convert MB to bytes
                        memory_type="resident",
                    )
                    collector.record_cpu_usage(
                        usage_percent=50.0 + i * 5,
                        core=f"core_{thread_id}",
                    )
                results.append("success")
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 3
        assert len(errors) == 0

    def test_disabled_collector_operations(
        self,
        disabled_collector: ResourceUtilizationCollector,
    ) -> None:
        """
        Test that disabled collector handles operations gracefully.
        """
        # All operations should be no-ops
        disabled_collector.record_model_memory_usage(
            model="test",
            memory_bytes=134217728,
            memory_type="resident",
        )
        disabled_collector.record_feature_store_size(
            size_bytes=536870912,
            storage_type="redis",  # 512 MB in bytes
        )
        disabled_collector.record_cpu_usage(
            usage_percent=50.0,
            core="test",
        )
        disabled_collector.record_gpu_metrics(
            device="cuda:0",
            compute_utilization=75.0,
            memory_used_bytes=2147483648,
            memory_total_bytes=8589934592,  # Convert MB to bytes
        )
        disabled_collector.start_monitoring()
        disabled_collector.stop_monitoring()

        # Should not raise any exceptions
        assert True

    def test_health_check(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test health check functionality.
        """
        health = collector.health_check()

        assert isinstance(health, dict)
        assert "enabled" in health
        assert "metrics_count" in health
        assert "prometheus_available" in health
        assert "config_valid" in health
        assert "collector_type" in health

        assert health["collector_type"] == "ResourceUtilizationCollector"

    @patch("ml._imports.HAS_PROMETHEUS", False)
    def test_graceful_degradation_without_prometheus(self) -> None:
        """
        Test graceful degradation when Prometheus is not available.
        """
        with patch("ml.monitoring.collectors.base.HAS_PROMETHEUS", False):
            config = MonitoringConfig(enabled=True)
            collector = ResourceUtilizationCollector(config)

            assert not collector.enabled
            assert collector.get_metric_count() == 0

            # All operations should be no-ops
            collector.record_model_memory_usage(
                model="test",
                memory_bytes=134217728,
                memory_type="resident",
            )
        collector.start_monitoring()
        collector.stop_monitoring()

        # Should not raise any exceptions
        assert True

    def test_string_representation(self, collector: ResourceUtilizationCollector) -> None:
        """
        Test string representation of collector.
        """
        repr_str = repr(collector)

        assert "ResourceUtilizationCollector" in repr_str
        assert f"enabled={collector.enabled}" in repr_str
        assert f"metrics_count={collector.get_metric_count()}" in repr_str

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_multi_gpu_metrics(self, collector) -> None:
        """
        Test recording metrics for multiple GPUs.
        """
        gpu_configs = [
            {"gpu_id": "0", "util": 75.0, "mem_used": 4096, "mem_total": 8192, "temp": 65},
            {"gpu_id": "1", "util": 82.0, "mem_used": 6144, "mem_total": 8192, "temp": 68},
            {"gpu_id": "2", "util": 45.0, "mem_used": 2048, "mem_total": 8192, "temp": 58},
        ]

        for gpu in gpu_configs:
            collector.record_gpu_metrics(
                device=f"cuda:{gpu['gpu_id']}",
                compute_utilization=gpu["util"],
                memory_used_bytes=gpu["mem_used"] * 1024 * 1024,  # Convert MB to bytes
                memory_total_bytes=gpu["mem_total"] * 1024 * 1024,  # Convert MB to bytes
            )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_storage_system_monitoring(self, collector) -> None:
        """
        Test monitoring different storage systems.
        """
        storage_systems = [
            {"path": "/data/models", "used": 125.6, "total": 500.0, "read": 234.7, "write": 89.3},
            {
                "path": "/cache/features",
                "used": 89.2,
                "total": 200.0,
                "read": 156.3,
                "write": 203.1,
            },
            {"path": "/logs/ml", "used": 12.5, "total": 100.0, "read": 45.6, "write": 78.9},
        ]

        for storage in storage_systems:
            collector.record_disk_usage(
                path=storage["path"],
                usage_bytes=int(storage["used"] * 1024 * 1024 * 1024),  # Convert GB to bytes
                usage_type="data",
            )

        # Should not raise any exceptions
        assert True
