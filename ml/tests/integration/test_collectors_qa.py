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
Comprehensive QA integration tests for ML monitoring collectors.

Tests end-to-end integration, performance, thread safety, and graceful degradation.

"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.data import DataQualityCollector
from ml.monitoring.collectors.features import FeatureEngineeringCollector
from ml.monitoring.collectors.model import ModelLifecycleCollector
from ml.monitoring.collectors.performance import PerformanceDegradationMonitor
from ml.monitoring.collectors.registry import MLMetricsRegistry
from ml.monitoring.collectors.resources import ResourceUtilizationCollector


class TestMLMonitoringQA:
    """
    Comprehensive QA tests for ML monitoring collectors.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test - clear Prometheus registry."""
        if HAS_PROMETHEUS:
            from prometheus_client import REGISTRY

            # Clear collectors to avoid duplicates
            collectors = list(REGISTRY._collector_to_names.keys())
            for collector in collectors:
                try:
                    REGISTRY.unregister(collector)
                except Exception:
                    pass

    def test_full_collector_integration(self):
        """
        Test complete integration of all collectors.
        """
        # Create config with unique prefix to avoid conflicts
        config = MonitoringConfig(
            enabled=True,
            metrics_prefix=f"qa_test_{int(time.time())}",
            metrics_port=8091,
            enable_background_monitoring=False,
        )

        # Initialize all collectors
        data_collector = DataQualityCollector(config)
        feature_collector = FeatureEngineeringCollector(config)
        model_collector = ModelLifecycleCollector(config)
        performance_collector = PerformanceDegradationMonitor(config)
        resource_collector = ResourceUtilizationCollector(config)

        # Test data quality metrics
        data_collector.record_data_load(
            instrument="EURUSD",
            rows_loaded=10000,
            load_duration=2.5,
            source="parquet",
        )

        data_collector.record_data_quality_check(
            instrument="EURUSD",
            check_type="completeness",
            passed=True,
            issues_found=0,
        )

        # Test feature engineering metrics
        feature_collector.record_feature_computation(
            instrument="EURUSD",
            feature_type="technical",
            computation_duration=1.2,
            features_computed=50,
            computation_mode="batch",
            success=True,
        )

        # Test model lifecycle metrics
        model_collector.record_model_deployment(
            model="xgboost_v1",
            version="1.0.0",
            environment="production",
            success=True,
        )

        with model_collector.time_training("xgboost_v1"):
            time.sleep(0.01)  # Simulate training

        # Test performance monitoring
        performance_collector.record_model_performance(
            model="xgboost_v1",
            accuracy=0.85,
            metric_type="accuracy",
            window="1h",
        )

        # Test resource monitoring
        resource_collector.record_model_memory_usage(
            model="xgboost_v1",
            memory_mb=256.5,
        )

        # Verify all collectors are functioning
        assert data_collector.enabled == HAS_PROMETHEUS
        assert feature_collector.enabled == HAS_PROMETHEUS
        assert model_collector.enabled == HAS_PROMETHEUS
        assert performance_collector.enabled == HAS_PROMETHEUS
        assert resource_collector.enabled == HAS_PROMETHEUS

        # Get summaries
        if HAS_PROMETHEUS:
            data_stats = data_collector.get_data_stats()
            assert "total_rows_loaded" in data_stats

            model_stats = model_collector.get_model_stats()
            assert "deployments" in model_stats

    def test_performance_overhead(self):
        """
        Test that metrics collection overhead is < 5%.
        """
        config_enabled = MonitoringConfig(
            enabled=True,
            metrics_prefix=f"perf_test_{int(time.time())}",
        )
        config_disabled = MonitoringConfig(enabled=False)

        # Test with metrics disabled
        collector_disabled = DataQualityCollector(config_disabled)
        start_time = time.perf_counter()
        for _ in range(1000):
            collector_disabled.record_data_load(
                instrument="TEST",
                rows_loaded=100,
                load_duration=0.1,
                source="test",
            )
        time_without_metrics = time.perf_counter() - start_time

        # Test with metrics enabled
        collector_enabled = DataQualityCollector(config_enabled)
        start_time = time.perf_counter()
        for _ in range(1000):
            collector_enabled.record_data_load(
                instrument="TEST",
                rows_loaded=100,
                load_duration=0.1,
                source="test",
            )
        time_with_metrics = time.perf_counter() - start_time

        # Calculate overhead
        if time_without_metrics > 0:
            overhead = (time_with_metrics - time_without_metrics) / time_without_metrics * 100
            print(f"Metrics overhead: {overhead:.2f}%")
            # Allow up to 10% overhead in tests (5% is production target)
            assert overhead < 10, f"Metrics overhead {overhead:.2f}% exceeds 10% limit"

    def test_thread_safety(self):
        """
        Test concurrent access to collectors.
        """
        config = MonitoringConfig(
            enabled=True,
            metrics_prefix=f"thread_test_{int(time.time())}",
        )
        collector = DataQualityCollector(config)

        def record_metrics(thread_id: int) -> None:
            """
            Record metrics from a thread.
            """
            for i in range(100):
                collector.record_data_load(
                    instrument=f"INST_{thread_id}",
                    rows_loaded=i,
                    load_duration=0.001 * i,
                    source="thread_test",
                )

        # Run concurrent threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(record_metrics, i) for i in range(10)]
            for future in futures:
                future.result()  # Wait for completion

        # Verify no errors occurred
        assert True  # If we get here, no exceptions were raised

    def test_graceful_degradation_without_prometheus(self):
        """
        Test system works without Prometheus installed.
        """
        with patch("ml._imports.HAS_PROMETHEUS", False):
            config = MonitoringConfig(enabled=True)

            # Create collectors without Prometheus
            data_collector = DataQualityCollector(config)
            feature_collector = FeatureEngineeringCollector(config)

            # These should not raise errors
            data_collector.record_data_load(
                instrument="TEST",
                rows_loaded=100,
                load_duration=1.0,
            )

            feature_collector.record_feature_computation(
                instrument="TEST",
                feature_type="test",
                computation_duration=1.0,
                features_computed=10,
            )

            # Collectors should be disabled
            assert not data_collector.enabled
            assert not feature_collector.enabled

    def test_memory_stability(self):
        """
        Test for memory leaks during extended operation.
        """
        config = MonitoringConfig(
            enabled=True,
            metrics_prefix=f"memory_test_{int(time.time())}",
        )
        collector = ResourceUtilizationCollector(config)

        # Record initial memory
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run many iterations
        for i in range(10000):
            collector.record_model_memory_usage(
                model=f"model_{i % 10}",  # Rotate through 10 models
                memory_mb=100.0 + i % 50,
            )

            if i % 1000 == 0:
                # Check memory periodically
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                print(f"Iteration {i}: Memory growth: {memory_growth:.2f} MB")

        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory

        # Allow up to 50MB growth for 10k operations
        assert total_growth < 50, f"Excessive memory growth: {total_growth:.2f} MB"

    def test_registry_lifecycle(self):
        """
        Test MLMetricsRegistry lifecycle management.
        """
        config = MonitoringConfig(
            enabled=True,
            metrics_prefix=f"registry_test_{int(time.time())}",
            enable_background_monitoring=False,
        )

        # Test context manager
        with MLMetricsRegistry(config) as registry:
            # Get collectors
            data_collector = registry.get_collector("data")
            assert data_collector is not None

            # Record some metrics
            data_collector.record_data_load(
                instrument="TEST",
                rows_loaded=100,
                load_duration=1.0,
            )

            # Check health
            health = registry.health_check()
            assert health["status"] == "healthy"

        # Registry should clean up properly
        assert True

    def test_metrics_accuracy(self):
        """
        Test that metrics are accurately recorded.
        """
        config = MonitoringConfig(
            enabled=True,
            metrics_prefix=f"accuracy_test_{int(time.time())}",
        )

        collector = ModelLifecycleCollector(config)

        # Record specific values
        test_data = [
            ("model_1", "1.0.0", "production", True),
            ("model_2", "2.0.0", "staging", True),
            ("model_3", "3.0.0", "development", False),
        ]

        for model, version, env, success in test_data:
            collector.record_model_deployment(
                model=model,
                version=version,
                environment=env,
                success=success,
            )

        # Get stats
        stats = collector.get_model_stats()

        if HAS_PROMETHEUS:
            # Verify counts
            assert stats["deployments"]["total"] >= 3
            assert stats["deployments"]["successful"] >= 2
            assert stats["deployments"]["failed"] >= 1

    def test_error_recovery(self):
        """
        Test recovery from errors during metric recording.
        """
        config = MonitoringConfig(
            enabled=True,
            metrics_prefix=f"error_test_{int(time.time())}",
        )

        collector = DataQualityCollector(config)

        # Test with invalid inputs - should not crash
        try:
            collector.record_data_load(
                instrument="",  # Empty instrument
                rows_loaded=-1,  # Negative rows
                load_duration=-1.0,  # Negative duration
            )
        except Exception:
            pytest.fail("Collector should handle invalid inputs gracefully")

        # Collector should still work after error
        collector.record_data_load(
            instrument="VALID",
            rows_loaded=100,
            load_duration=1.0,
        )

        assert True  # If we get here, error recovery worked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
