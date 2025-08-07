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
Tests for the PerformanceDegradationMonitor.
"""

import threading
from unittest.mock import patch

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.performance import PerformanceDegradationMonitor


class TestPerformanceDegradationMonitor:
    """
    Test suite for PerformanceDegradationMonitor functionality.
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
        Create a PerformanceDegradationMonitor for testing.
        """
        prefix = self.metric_name_manager.get_unique_name("ml")
        config = MonitoringConfig(enabled=True, metrics_prefix=prefix)
        return PerformanceDegradationMonitor(config)

    @pytest.fixture
    def disabled_collector(self):
        """
        Create a disabled PerformanceDegradationMonitor for testing.
        """
        config = MonitoringConfig(enabled=False)
        return PerformanceDegradationMonitor(config)

    def test_initialization(self, collector):
        """
        Test collector initialization.
        """
        assert collector.enabled == HAS_PROMETHEUS
        assert collector.get_metric_count() > 0 or not HAS_PROMETHEUS

    def test_disabled_initialization(self, disabled_collector):
        """
        Test disabled collector initialization.
        """
        assert not disabled_collector.enabled
        assert disabled_collector.get_metric_count() == 0

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_model_performance(self, collector):
        """
        Test recording model performance metrics.
        """
        model_name = "trend_predictor"
        instrument = "EURUSD"

        collector.record_model_performance(
            model=model_name,
            instrument=instrument,
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_prediction_evaluation(self, collector):
        """
        Test recording prediction evaluation metrics.
        """
        model_name = "volatility_predictor"
        instrument = "GBPUSD"

        collector.record_prediction_evaluation(
            model=model_name,
            instrument=instrument,
            mae=0.05,
            mse=0.0025,
            mape=3.2,
            directional_accuracy=0.78,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_distribution_shift(self, collector):
        """
        Test recording distribution shift detection.
        """
        model_name = "price_classifier"
        instrument = "USDJPY"

        collector.record_distribution_shift(
            model=model_name,
            instrument=instrument,
            shift_detected=True,
            kl_divergence=0.45,
            psi_score=0.32,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_update_degradation_score(self, collector):
        """
        Test updating degradation score.
        """
        model_name = "momentum_predictor"
        instrument = "AUDUSD"
        degradation_score = 0.25

        collector.update_degradation_score(
            model=model_name,
            instrument=instrument,
            degradation_score=degradation_score,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_inference_latency_percentiles(self, collector):
        """
        Test recording inference latency percentiles.
        """
        model_name = "risk_classifier"
        instrument = "NZDUSD"

        collector.record_inference_latency_percentiles(
            model=model_name,
            instrument=instrument,
            p50=2.5,
            p95=8.2,
            p99=15.7,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_trigger_retraining_alert(self, collector):
        """
        Test triggering retraining alert.
        """
        model_name = "sentiment_analyzer"
        instrument = "CADCHF"
        reason = "performance_degradation"
        threshold_value = 0.4

        collector.trigger_retraining_alert(
            model=model_name,
            instrument=instrument,
            reason=reason,
            threshold_value=threshold_value,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_retraining_completion(self, collector):
        """
        Test recording retraining completion.
        """
        model_name = "pattern_detector"
        instrument = "CHFJPY"
        old_performance = 0.75
        new_performance = 0.88

        collector.record_retraining_completion(
            model=model_name,
            instrument=instrument,
            old_performance=old_performance,
            new_performance=new_performance,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_get_performance_summary(self, collector):
        """
        Test getting performance summary.
        """
        model_name = "price_predictor"
        instrument = "EURGBP"

        # Record some performance metrics first
        collector.record_model_performance(
            model=model_name,
            instrument=instrument,
            accuracy=0.82,
            precision=0.79,
            recall=0.85,
            f1_score=0.82,
        )
        collector.update_degradation_score(
            model=model_name,
            instrument=instrument,
            degradation_score=0.15,
        )

        summary = collector.get_performance_summary(model_name, instrument)

        assert isinstance(summary, dict)
        assert "model" in summary
        assert "instrument" in summary
        assert "current_accuracy" in summary
        assert "degradation_score" in summary
        assert "retraining_alerts" in summary

    def test_get_performance_summary_disabled(self, disabled_collector):
        """
        Test getting performance summary when disabled.
        """
        summary = disabled_collector.get_performance_summary("test_model", "EURUSD")

        # Should return empty dict when disabled
        assert summary == {}

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_performance_degradation_workflow(self, collector):
        """
        Test complete performance degradation workflow.
        """
        model_name = "comprehensive_predictor"
        instrument = "BTCUSD"

        # 1. Record initial good performance
        collector.record_model_performance(
            model=model_name,
            instrument=instrument,
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
        )

        # 2. Detect distribution shift
        collector.record_distribution_shift(
            model=model_name,
            instrument=instrument,
            shift_detected=True,
            kl_divergence=0.35,
            psi_score=0.28,
        )

        # 3. Record degraded performance
        collector.record_model_performance(
            model=model_name,
            instrument=instrument,
            accuracy=0.72,
            precision=0.68,
            recall=0.75,
            f1_score=0.71,
        )

        # 4. Update degradation score
        collector.update_degradation_score(
            model=model_name,
            instrument=instrument,
            degradation_score=0.45,
        )

        # 5. Trigger retraining alert
        collector.trigger_retraining_alert(
            model=model_name,
            instrument=instrument,
            reason="accuracy_drop",
            threshold_value=0.75,
        )

        # 6. Complete retraining
        collector.record_retraining_completion(
            model=model_name,
            instrument=instrument,
            old_performance=0.72,
            new_performance=0.89,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_thread_safety(self, collector):
        """
        Test thread safety of collector operations.
        """
        results = []
        errors = []

        def worker():
            try:
                thread_id = threading.current_thread().ident
                for i in range(5):
                    collector.record_model_performance(
                        model=f"test_model_{thread_id}",
                        instrument="EURUSD",
                        accuracy=0.80 + i * 0.01,
                        precision=0.78 + i * 0.01,
                        recall=0.82 + i * 0.01,
                        f1_score=0.80 + i * 0.01,
                    )
                    collector.update_degradation_score(
                        model=f"test_model_{thread_id}",
                        instrument="EURUSD",
                        degradation_score=0.1 + i * 0.05,
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

    def test_disabled_collector_operations(self, disabled_collector):
        """
        Test that disabled collector handles operations gracefully.
        """
        # All operations should be no-ops
        disabled_collector.record_model_performance(
            model="test",
            instrument="EUR",
            accuracy=0.8,
            precision=0.8,
            recall=0.8,
            f1_score=0.8,
        )
        disabled_collector.record_prediction_evaluation(
            model="test",
            instrument="EUR",
            mae=0.1,
            mse=0.01,
            mape=5.0,
            directional_accuracy=0.7,
        )
        disabled_collector.record_distribution_shift(
            model="test",
            instrument="EUR",
            shift_detected=True,
            kl_divergence=0.3,
            psi_score=0.2,
        )
        disabled_collector.trigger_retraining_alert(
            model="test",
            instrument="EUR",
            reason="test",
            threshold_value=0.5,
        )

        # Should not raise any exceptions
        assert True

    def test_health_check(self, collector):
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

        assert health["collector_type"] == "PerformanceDegradationMonitor"

    @patch("ml._imports.HAS_PROMETHEUS", False)
    def test_graceful_degradation_without_prometheus(self):
        """
        Test graceful degradation when Prometheus is not available.
        """
        config = MonitoringConfig(enabled=True)
        collector = PerformanceDegradationMonitor(config)

        assert not collector.enabled
        assert collector.get_metric_count() == 0

        # All operations should be no-ops
        collector.record_model_performance(
            model="test",
            instrument="EUR",
            accuracy=0.8,
            precision=0.8,
            recall=0.8,
            f1_score=0.8,
        )
        collector.trigger_retraining_alert(
            model="test",
            instrument="EUR",
            reason="test",
            threshold_value=0.5,
        )

        # Should not raise any exceptions
        assert True

    def test_string_representation(self, collector):
        """
        Test string representation of collector.
        """
        repr_str = repr(collector)

        assert "PerformanceDegradationMonitor" in repr_str
        assert f"enabled={collector.enabled}" in repr_str
        assert f"metrics_count={collector.get_metric_count()}" in repr_str

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_latency_percentiles_comprehensive(self, collector):
        """
        Test comprehensive latency percentile recording.
        """
        models = ["fast_model", "balanced_model", "accurate_model"]
        instrument = "GOLD"

        # Simulate different performance characteristics
        latency_profiles = [
            {"p50": 1.2, "p95": 3.5, "p99": 7.8},  # Fast model
            {"p50": 2.5, "p95": 6.2, "p99": 12.5},  # Balanced model
            {"p50": 4.1, "p95": 9.8, "p99": 18.2},  # Accurate model
        ]

        for model, profile in zip(models, latency_profiles):
            collector.record_inference_latency_percentiles(
                model=model,
                instrument=instrument,
                **profile,
            )

        # Should not raise any exceptions
        assert True
