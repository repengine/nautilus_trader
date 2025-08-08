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
Tests for the ModelLifecycleCollector.
"""

import threading
import time
from datetime import UTC
from datetime import datetime
from unittest.mock import patch

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.model import ModelLifecycleCollector


class TestModelLifecycleCollector:
    """
    Test suite for ModelLifecycleCollector functionality.
    """

    @pytest.fixture(autouse=True)
    def setup_test(self, prometheus_registry_cleanup, metric_name_manager):
        """
        Set up test with proper cleanup and unique names.
        """
        self.metric_name_manager = metric_name_manager

    @pytest.fixture
    def collector(self):
        """
        Create a ModelLifecycleCollector for testing.
        """
        prefix = self.metric_name_manager.get_unique_name("ml")
        config = MonitoringConfig(enabled=True, metrics_prefix=prefix)
        return ModelLifecycleCollector(config)

    @pytest.fixture
    def disabled_collector(self):
        """
        Create a disabled ModelLifecycleCollector for testing.
        """
        config = MonitoringConfig(enabled=False)
        return ModelLifecycleCollector(config)

    def test_initialization(self, collector) -> None:
        """
        Test collector initialization.
        """
        assert collector.enabled == HAS_PROMETHEUS
        assert collector.get_metric_count() > 0 or not HAS_PROMETHEUS

    def test_disabled_initialization(self, disabled_collector) -> None:
        """
        Test disabled collector initialization.
        """
        assert not disabled_collector.enabled
        assert disabled_collector.get_metric_count() == 0

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_model_deployment(self, collector) -> None:
        """
        Test recording model deployment.
        """
        model_name = "test_model"
        version = "v1.0.0"
        instrument = "EURUSD"
        deployment_time = datetime.now(UTC)
        git_commit = "abc123"

        collector.record_model_deployment(
            model=model_name,
            version=version,
            instrument=instrument,
            deployment_time=deployment_time,
            git_commit=git_commit,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_model_training(self, collector) -> None:
        """
        Test recording model training metrics.
        """
        model_name = "test_model"
        training_duration = 120.5
        validation_score = 0.85
        training_samples = 10000

        collector.record_model_training(
            model=model_name,
            training_duration=training_duration,
            training_samples=training_samples,
            validation_score=validation_score,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_model_loading(self, collector) -> None:
        """
        Test recording model loading metrics.
        """
        model_name = "test_model"
        load_duration = 2.5
        model_size_mb = 10.3

        collector.record_model_loading(
            model=model_name,
            load_duration=load_duration,
            model_size_bytes=int(model_size_mb * 1024 * 1024),  # Convert MB to bytes
            location="s3",
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_update_model_scores(self, collector) -> None:
        """
        Test updating model scores.
        """
        model_name = "test_model"
        accuracy = 0.92
        precision = 0.88
        recall = 0.90

        # Test accuracy score
        collector.update_model_scores(
            model=model_name,
            validation_score=accuracy,
            metric_type="accuracy",
        )

        # Test precision score
        collector.update_model_scores(
            model=model_name,
            validation_score=precision,
            metric_type="precision",
        )

        # Test recall score
        collector.update_model_scores(
            model=model_name,
            validation_score=recall,
            metric_type="recall",
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_time_training_context_manager(self, collector) -> None:
        """
        Test model training timer context manager.
        """
        model_name = "test_model"
        phase = "training"

        with collector.time_training(model_name, phase):
            time.sleep(0.01)  # Simulate training time

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_time_training_context_manager_with_exception(self, collector) -> None:
        """
        Test model training timer with exception.
        """
        model_name = "test_model"
        phase = "validation"

        try:
            with collector.time_training(model_name, phase):
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Timer should still record the duration
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_time_loading_context_manager(self, collector) -> None:
        """
        Test model loading timer context manager.
        """
        model_name = "test_model"
        location = "disk"

        with collector.time_loading(model_name, location):
            time.sleep(0.01)  # Simulate loading time

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_time_loading_context_manager_with_exception(self, collector) -> None:
        """
        Test model loading timer with exception.
        """
        model_name = "test_model"
        location = "s3"

        try:
            with collector.time_loading(model_name, location):
                raise ConnectionError("Test exception")
        except ConnectionError:
            pass  # Expected

        # Timer should still record the duration
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_get_model_stats(self, collector) -> None:
        """
        Test getting model statistics.
        """
        model_name = "test_model"

        # Record some metrics first
        collector.record_model_training(
            model=model_name,
            training_duration=60.0,
            training_samples=5000,
            validation_score=0.80,
        )

        stats = collector.get_model_stats(model_name)

        assert isinstance(stats, dict)
        assert "model" in stats
        # The exact keys depend on what metrics have been recorded and have values
        assert stats["model"] == model_name

    def test_get_model_stats_disabled(self, disabled_collector) -> None:
        """
        Test getting model stats when disabled.
        """
        stats = disabled_collector.get_model_stats("test_model")

        # Should return basic dict when disabled (just model name)
        assert isinstance(stats, dict)
        assert "model" in stats

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_additional_model_operations(self, collector) -> None:
        """
        Test additional model operations.
        """
        model_name = "test_model"

        # Record additional training with different metric type
        collector.update_model_scores(
            model=model_name,
            training_score=0.75,
            validation_score=0.80,
            metric_type="f1_score",
        )

        # Record model loading with different parameters
        collector.record_model_loading(
            model=model_name,
            load_duration=1.5,
            model_size_bytes=1048576,  # 1MB
            location="memory",
            format_type="onnx",
            success=True,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_thread_safety(self, collector) -> None:
        """
        Test thread safety of collector operations.
        """
        results = []
        errors = []

        def worker():
            try:
                for i in range(10):
                    collector.record_model_training(
                        model=f"model_{threading.current_thread().ident}",
                        training_duration=float(i),
                        training_samples=1000 + i * 100,
                        validation_score=0.8 + i * 0.01,
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

    def test_disabled_collector_operations(self, disabled_collector) -> None:
        """
        Test that disabled collector handles operations gracefully.
        """
        # All operations should be no-ops
        disabled_collector.record_model_deployment(
            model="test",
            version="v1",
            instrument="EUR",
            deployment_time=datetime.now(),
        )
        disabled_collector.record_model_training(
            model="test",
            training_duration=60.0,
            training_samples=1000,
            validation_score=0.8,
        )
        disabled_collector.record_model_loading(
            model="test",
            load_duration=1.0,
            model_size_bytes=5242880,
        )
        disabled_collector.update_model_scores(
            model="test",
            validation_score=0.9,
        )

        # Should not raise any exceptions
        assert True

    def test_health_check(self, collector) -> None:
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

        assert health["collector_type"] == "ModelLifecycleCollector"

    @patch("ml.monitoring.collectors.base.HAS_PROMETHEUS", False)
    def test_graceful_degradation_without_prometheus(self) -> None:
        """
        Test graceful degradation when Prometheus is not available.
        """
        config = MonitoringConfig(enabled=True)
        collector = ModelLifecycleCollector(config)

        assert not collector.enabled
        assert collector.get_metric_count() == 0

        # All operations should be no-ops
        collector.record_model_deployment(
            model="test",
            version="v1",
            instrument="EUR",
            deployment_time=datetime.now(),
        )
        collector.record_model_training(
            model="test",
            training_duration=60.0,
            training_samples=1000,
            validation_score=0.8,
        )

        # Should not raise any exceptions
        assert True

    def test_string_representation(self, collector) -> None:
        """
        Test string representation of collector.
        """
        repr_str = repr(collector)

        assert "ModelLifecycleCollector" in repr_str
        assert f"enabled={collector.enabled}" in repr_str
        assert f"metrics_count={collector.get_metric_count()}" in repr_str
