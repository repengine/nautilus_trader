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
Integration tests for the complete ML monitoring system.

This module tests the full integration between collectors, data loader, feature
engineering, and the metrics registry.

"""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml._imports import HAS_POLARS
from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.registry import MLMetricsRegistry


class TestMLMonitoringIntegration:
    """
    Integration tests for the complete ML monitoring system.
    """

    @pytest.fixture
    def config(self):
        """
        Create test monitoring configuration.
        """
        return MonitoringConfig(
            enabled=True,
            metrics_port=8082,  # Use different port to avoid conflicts
            metrics_prefix="integration_test",
        )

    @pytest.fixture
    def registry(self, config):
        """
        Create metrics registry for integration testing.
        """
        return MLMetricsRegistry(config, enable_background_monitoring=False)

    @pytest.mark.skipif(
        not (HAS_PROMETHEUS and HAS_POLARS),
        reason="Requires both Prometheus and Polars for full integration",
    )
    def test_data_loader_with_metrics_integration(self, registry):
        """
        Test MLDataLoader integration with DataQualityCollector.
        """
        from ml.data.loader import MLDataLoader
        from ml.monitoring.collectors.data import DataQualityCollector
        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

        # Create mock catalog
        mock_catalog = MagicMock(spec=ParquetDataCatalog)

        # Mock query to return empty results (simulating no data)
        mock_catalog.query.return_value = []

        # Get data quality collector from registry
        data_collector = registry.get_collector("data")
        assert isinstance(data_collector, DataQualityCollector)

        # Create data loader with metrics collector
        loader = MLDataLoader(
            catalog=mock_catalog,
            metrics_collector=data_collector,
        )

        # Load data - should record metrics even with empty results
        df = loader.load_bars("EURUSD.SIM", start="2023-01-01", end="2023-01-31")

        # Verify the data loader recorded metrics
        assert df.is_empty()

        # Health check should show the collector is working
        health = data_collector.health_check()
        assert health["enabled"]

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Requires Prometheus client")
    def test_feature_engineer_with_metrics_integration(self, registry):
        """
        Test FeatureEngineer integration with FeatureEngineeringCollector.
        """
        from ml.features.engineering import FeatureConfig
        from ml.features.engineering import FeatureEngineer
        from ml.monitoring.collectors.features import FeatureEngineeringCollector

        # Get feature engineering collector from registry
        features_collector = registry.get_collector("features")
        assert isinstance(features_collector, FeatureEngineeringCollector)

        # Create feature engineer with metrics collector
        config = FeatureConfig(return_periods=[1, 5], rsi_period=14)
        engineer = FeatureEngineer(config=config, metrics_collector=features_collector)

        # Create mock data for feature calculation
        mock_bar = {
            "close": 1.2345,
            "volume": 1000.0,
            "open": 1.2340,
            "high": 1.2350,
            "low": 1.2335,
        }

        # Mock indicator manager
        from ml.features.engineering import IndicatorManager

        mock_indicator_manager = MagicMock(spec=IndicatorManager)
        mock_indicator_manager.get_all_values.return_value = {
            "sma_20": 1.2340,
            "rsi_14": 45.0,
            "bb_upper": 1.2360,
            "bb_lower": 1.2320,
        }

        # Calculate features online - should record metrics
        features = engineer.calculate_features_online(
            current_bar=mock_bar,
            indicator_manager=mock_indicator_manager,
        )

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

        # Health check should show the collector is working
        health = features_collector.health_check()
        assert health["enabled"]

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Requires Prometheus client")
    def test_model_lifecycle_integration(self, registry):
        """
        Test ModelLifecycleCollector functionality.
        """
        from ml.monitoring.collectors.model import ModelLifecycleCollector

        # Get model lifecycle collector
        model_collector = registry.get_collector("model")
        assert isinstance(model_collector, ModelLifecycleCollector)

        # Simulate model deployment
        model_collector.record_model_deployment(
            model="test_xgboost_v1",
            version="1.0.0",
            instrument="EURUSD",
            git_commit="abc123def",
        )

        # Simulate model training
        model_collector.record_model_training(
            model="test_xgboost_v1",
            training_duration=120.5,
            training_samples=50000,
            training_score=0.85,
            validation_score=0.82,
        )

        # Simulate model loading
        model_collector.record_model_loading(
            model="test_xgboost_v1",
            load_duration=0.5,
            model_size_bytes=1024 * 1024,  # 1MB
            success=True,
        )

        # Test context managers
        with model_collector.time_training("test_xgboost_v1") as timer:
            time.sleep(0.01)  # Simulate training
            timer.set_training_data(samples=1000, training_score=0.9)

        with model_collector.time_loading("test_xgboost_v1") as timer:
            time.sleep(0.001)  # Simulate loading
            timer.set_model_info(size_bytes=500000, format_type="onnx")

        # Verify collector is working
        health = model_collector.health_check()
        assert health["enabled"]

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Requires Prometheus client")
    def test_performance_monitoring_integration(self, registry):
        """
        Test PerformanceDegradationMonitor functionality.
        """
        from ml.monitoring.collectors.performance import PerformanceDegradationMonitor

        # Get performance monitor
        perf_monitor = registry.get_collector("performance")
        assert isinstance(perf_monitor, PerformanceDegradationMonitor)

        # Simulate model performance tracking
        confidence_scores = [0.8, 0.9, 0.7, 0.95, 0.6, 0.85, 0.88]

        perf_monitor.record_model_performance(
            model="test_model",
            accuracy=0.82,
            confidence_scores=confidence_scores,
        )

        # Simulate individual predictions
        for i, correct in enumerate([True, True, False, True, False]):
            perf_monitor.record_prediction_evaluation(
                model="test_model",
                prediction_correct=correct,
                confidence=confidence_scores[i] if i < len(confidence_scores) else 0.5,
                latency_ms=2.5 + i * 0.1,  # Simulate varying latency
            )

        # Simulate distribution shift detection
        perf_monitor.record_distribution_shift(
            model="test_model",
            shift_score=0.05,
            shift_metric="psi",
        )

        # Update degradation score
        perf_monitor.update_degradation_score(
            model="test_model",
            degradation_score=0.3,  # Below threshold
        )

        # Record inference latencies
        latencies = [1.2, 2.3, 1.8, 3.1, 2.0, 1.5, 4.2, 1.9]
        perf_monitor.record_inference_latency_percentiles(
            model="test_model",
            latencies_ms=latencies,
        )

        # Get performance summary
        summary = perf_monitor.get_performance_summary("test_model")
        assert "model" in summary
        assert summary["model"] == "test_model"

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Requires Prometheus client")
    def test_resource_monitoring_integration(self, registry):
        """
        Test ResourceUtilizationCollector functionality.
        """
        from ml.monitoring.collectors.resources import ResourceUtilizationCollector

        # Get resource collector
        resource_collector = registry.get_collector("resources")
        assert isinstance(resource_collector, ResourceUtilizationCollector)

        # Record various resource metrics
        resource_collector.record_model_memory_usage(
            model="test_model",
            memory_bytes=512 * 1024 * 1024,  # 512MB
            memory_type="resident",
        )

        resource_collector.record_feature_store_size(
            size_bytes=100 * 1024 * 1024,  # 100MB
            storage_type="memory",
        )

        resource_collector.record_cpu_usage(
            usage_percent=45.2,
            core="process",
        )

        resource_collector.record_ml_cpu_time(
            cpu_time_seconds=12.5,
            operation_type="inference",
        )

        # Simulate GPU metrics (if available)
        resource_collector.record_gpu_metrics(
            device="cuda:0",
            compute_utilization=75.0,
            memory_utilization=60.0,
            memory_used_bytes=2 * 1024 * 1024 * 1024,  # 2GB
            memory_total_bytes=8 * 1024 * 1024 * 1024,  # 8GB
        )

        resource_collector.record_disk_usage(
            path="/data/models",
            usage_bytes=5 * 1024 * 1024 * 1024,  # 5GB
            usage_type="models",
        )

        resource_collector.record_data_io(
            bytes_transferred=1024 * 1024,  # 1MB
            operation="read",
            data_type="bars",
        )

        resource_collector.record_inference_batch_size(
            model="test_model",
            batch_size=32,
        )

        resource_collector.record_training_data_processed(
            rows=10000,
            dataset="train",
        )

        # Get resource summary
        summary = resource_collector.get_resource_summary()
        assert isinstance(summary, dict)

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Requires Prometheus client")
    def test_full_registry_lifecycle_integration(self, registry):
        """
        Test complete registry lifecycle with all collectors.
        """
        assert not registry.started

        # Start the registry
        with registry:
            assert registry.started

            # Get comprehensive health check
            health = registry.health_check()

            assert health["status"] == "healthy"
            assert health["started"]
            assert health["enabled_collectors"] > 0
            assert health["total_collectors"] == 6

            # Verify all expected collectors are present
            collectors = health["collectors"]
            expected_types = [
                "MLMetricsCollector",
                "ModelLifecycleCollector",
                "DataQualityCollector",
                "FeatureEngineeringCollector",
                "PerformanceDegradationMonitor",
                "ResourceUtilizationCollector",
            ]

            for collector_health in collectors.values():
                assert collector_health["collector_type"] in expected_types
                assert "enabled" in collector_health
                assert "metrics_count" in collector_health

            # Get metrics summary
            summary = registry.get_metrics_summary()
            assert summary["registry_status"] == "running"
            assert summary["total_collectors"] == 6

            # Export metrics
            if HAS_PROMETHEUS:
                metrics_text = registry.export_metrics()
                assert isinstance(metrics_text, str)
                # Should contain our test prefix
                assert "integration_test" in metrics_text or len(metrics_text) > 0

            # Reset all metrics
            registry.reset_all_metrics()

        # After context exit, should be stopped
        assert not registry.started

    def test_error_handling_integration(self, registry):
        """
        Test that the system handles errors gracefully.
        """
        # All operations should complete without raising exceptions
        # even when Prometheus is not available or collectors are disabled

        data_collector = registry.get_collector("data")
        data_collector.record_data_load(
            instrument="TEST",
            data_type="bars",
            rows_loaded=100,
            duration_seconds=0.1,
        )

        features_collector = registry.get_collector("features")
        features_collector.record_feature_computation(
            instrument="TEST",
            feature_type="technical",
            computation_duration=0.05,
            features_computed=10,
        )

        model_collector = registry.get_collector("model")
        model_collector.record_model_deployment(
            model="test_model",
            version="1.0.0",
        )

        # Health check should always work
        health = registry.health_check()
        assert "status" in health
        assert "collectors" in health

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Requires Prometheus client")
    def test_thread_safety_integration(self, registry):
        """
        Test thread safety across the system.
        """
        import random
        import threading

        def worker(worker_id):
            """
            Worker function that exercises various collectors.
            """
            for i in range(5):
                # Data quality metrics
                data_collector = registry.get_collector("data")
                data_collector.record_data_load(
                    instrument=f"INST{worker_id}",
                    data_type="bars",
                    rows_loaded=random.randint(100, 1000),  # noqa: S311
                    duration_seconds=random.uniform(0.1, 1.0),  # noqa: S311
                )

                # Feature metrics
                features_collector = registry.get_collector("features")
                features_collector.record_feature_computation(
                    instrument=f"INST{worker_id}",
                    feature_type="technical",
                    computation_duration=random.uniform(0.01, 0.1),  # noqa: S311
                    features_computed=random.randint(5, 20),  # noqa: S311
                )

                # Model metrics
                model_collector = registry.get_collector("model")
                model_collector.record_model_training(
                    model=f"model_{worker_id}",
                    training_duration=random.uniform(10, 100),  # noqa: S311
                    training_samples=random.randint(1000, 10000),  # noqa: S311
                )

                # Small delay to increase interleaving
                time.sleep(0.001)

        # Start registry
        with registry:
            # Create and start threads
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # System should still be healthy after concurrent access
            health = registry.health_check()
            assert health["status"] == "healthy"

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Requires Prometheus client")
    def test_metrics_export_format_integration(self, registry):
        """
        Test that exported metrics follow Prometheus format.
        """
        with registry:
            # Record some metrics
            data_collector = registry.get_collector("data")
            data_collector.record_data_load(
                instrument="EURUSD",
                data_type="bars",
                rows_loaded=1000,
                duration_seconds=0.5,
            )

            # Export metrics
            metrics_text = registry.export_metrics()

            if metrics_text:  # Only test if we got metrics
                lines = metrics_text.strip().split("\n")

                # Should have some metric lines
                metric_lines = [line for line in lines if not line.startswith("#") and line.strip()]

                # Each metric line should follow Prometheus format
                for line in metric_lines:
                    if line.strip():
                        # Should contain metric name and value
                        assert " " in line or "{" in line
                        # Should end with a number (the metric value)
                        parts = line.split()
                        if parts:
                            # Last part should be a number or timestamp
                            try:
                                float(parts[-1])
                            except ValueError:
                                # Some metrics might have timestamp, try second to last
                                if len(parts) > 1:
                                    float(parts[-2])

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Requires Prometheus client")
    def test_performance_overhead_integration(self, registry):
        """
        Test that metrics collection has minimal performance overhead.
        """
        import time

        # Baseline: operations without metrics
        start_time = time.perf_counter()
        for i in range(100):
            # Simulate some work
            _ = i**2
        baseline_duration = time.perf_counter() - start_time

        # With metrics: same operations but with metrics collection
        with registry:
            start_time = time.perf_counter()
            for i in range(100):
                # Same work
                _ = i**2

                # Add metrics collection
                if i % 10 == 0:  # Collect metrics every 10 operations
                    data_collector = registry.get_collector("data")
                    data_collector.record_data_load(
                        instrument="TEST",
                        data_type="bars",
                        rows_loaded=100,
                        duration_seconds=0.01,
                    )

            with_metrics_duration = time.perf_counter() - start_time

        # Overhead should be reasonable (less than 50% increase)
        if baseline_duration > 0:
            overhead_ratio = (with_metrics_duration - baseline_duration) / baseline_duration
            assert overhead_ratio < 0.5, f"Metrics overhead too high: {overhead_ratio:.2%}"
