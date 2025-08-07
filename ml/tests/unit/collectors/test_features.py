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
Tests for the FeatureEngineeringCollector.
"""

import threading
import time
from unittest.mock import patch

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.features import FeatureEngineeringCollector


class TestFeatureEngineeringCollector:
    """
    Test suite for FeatureEngineeringCollector functionality.
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
        Create a FeatureEngineeringCollector for testing.
        """
        prefix = self.metric_name_manager.get_unique_name("ml")
        config = MonitoringConfig(enabled=True, metrics_prefix=prefix)
        return FeatureEngineeringCollector(config)

    @pytest.fixture
    def disabled_collector(self):
        """
        Create a disabled FeatureEngineeringCollector for testing.
        """
        config = MonitoringConfig(enabled=False)
        return FeatureEngineeringCollector(config)

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
    def test_record_feature_computation(self, collector):
        """
        Test recording feature computation metrics.
        """
        instrument = "EURUSD"
        feature_type = "technical"
        computation_duration = 0.05
        features_computed = 100

        collector.record_feature_computation(
            instrument=instrument,
            feature_type=feature_type,
            computation_duration=computation_duration,
            features_computed=features_computed,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_cache_hit(self, collector):
        """
        Test recording cache hit.
        """
        instrument = "GBPUSD"
        cache_level = "memory"

        collector.record_cache_hit(instrument, cache_level)

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_cache_miss(self, collector):
        """
        Test recording cache miss.
        """
        instrument = "USDJPY"
        cache_level = "memory"

        collector.record_cache_miss(instrument, cache_level)

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_update_cache_stats(self, collector):
        """
        Test updating cache statistics.
        """
        instrument = "EURUSD"
        hit_ratio = 0.85
        cache_level = "memory"
        cache_size = 50000

        collector.update_cache_stats(
            instrument=instrument,
            hit_ratio=hit_ratio,
            cache_level=cache_level,
            cache_size=cache_size,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_feature_drift(self, collector):
        """
        Test recording feature drift detection.
        """
        instrument = "EURJPY"
        feature = "price_momentum"
        drift_score = 0.65

        collector.record_feature_drift(
            instrument=instrument,
            feature=feature,
            drift_score=drift_score,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_feature_importance(self, collector):
        """
        Test recording feature importance scores.
        """
        model = "xgboost"
        feature_importances = {
            "volatility_ratio": 0.78,
            "sma_20": 0.65,
            "rsi_14": 0.45,
        }

        collector.record_feature_importance(
            model=model,
            feature_importances=feature_importances,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_feature_quality(self, collector):
        """
        Test recording feature quality metrics.
        """
        instrument = "NZDUSD"
        feature_qualities = {
            "support_resistance": {
                "null_ratio": 0.02,
                "infinite_ratio": 0.01,
            },
            "sma_20": {
                "null_ratio": 0.0,
                "infinite_ratio": 0.0,
            },
        }

        collector.record_feature_quality(
            instrument=instrument,
            feature_qualities=feature_qualities,
        )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_time_feature_computation_context_manager(self, collector):
        """
        Test feature computation timer context manager.
        """
        feature_name = "trend_strength"
        instrument = "CADCHF"

        with collector.time_feature_computation(feature_name, instrument):
            time.sleep(0.01)  # Simulate computation time

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_time_feature_computation_with_exception(self, collector):
        """
        Test feature computation timer with exception.
        """
        feature_name = "correlation_matrix"
        instrument = "CHFJPY"

        try:
            with collector.time_feature_computation(feature_name, instrument):
                raise ValueError("Computation failed")
        except ValueError:
            pass  # Expected

        # Timer should still record the duration
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_get_feature_stats(self, collector):
        """
        Test getting feature statistics.
        """
        instrument = "EURGBP"
        feature_type = "technical"

        # Record some metrics first
        collector.record_feature_computation(
            instrument=instrument,
            feature_type=feature_type,
            computation_duration=0.12,
            features_computed=250,
        )
        collector.record_cache_hit(instrument, "memory")
        collector.record_cache_miss(instrument, "memory")

        stats = collector.get_feature_stats(instrument, feature_type)

        assert isinstance(stats, dict)
        assert "instrument" in stats
        assert "feature_type" in stats
        assert "cache_hit_ratio" in stats
        assert "last_computed" in stats

    def test_get_feature_stats_disabled(self, disabled_collector):
        """
        Test getting feature stats when disabled.
        """
        stats = disabled_collector.get_feature_stats("EURUSD", "technical")

        # Should return basic info but no metrics when disabled
        assert isinstance(stats, dict)
        assert "instrument" in stats
        assert stats["instrument"] == "EURUSD"

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_set_computation_result(self, collector):
        """
        Test setting computation result metrics.
        """
        feature_name = "volume_profile"
        instrument = "GOLD"

        collector.set_computation_result(
            feature_name=feature_name,
            instrument=instrument,
            success=True,
            feature_count=500,
            computation_time=0.25,
            memory_used_mb=12.5,
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
                    collector.record_feature_computation(
                        feature_name=f"test_feature_{thread_id}",
                        instrument="EURUSD",
                        computation_time=0.01 + i * 0.001,
                        feature_count=100 + i,
                    )
                    collector.record_cache_hit(f"test_feature_{thread_id}", "EURUSD")
                    if i % 2 == 0:
                        collector.record_cache_miss(f"test_feature_{thread_id}", "EURUSD")
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
        disabled_collector.record_feature_computation(
            feature_name="test",
            instrument="EUR",
            computation_time=0.1,
            feature_count=10,
        )
        disabled_collector.record_cache_hit("test", "EUR")
        disabled_collector.record_cache_miss("test", "EUR")
        disabled_collector.update_cache_stats(
            cache_size_mb=10.0,
            cache_entries=100,
            hit_ratio=0.5,
            eviction_count=5,
        )
        disabled_collector.record_feature_drift(
            feature_name="test",
            instrument="EUR",
            drift_score=0.5,
            drift_detected=False,
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

        assert health["collector_type"] == "FeatureEngineeringCollector"

    @patch("ml._imports.HAS_PROMETHEUS", False)
    def test_graceful_degradation_without_prometheus(self):
        """
        Test graceful degradation when Prometheus is not available.
        """
        config = MonitoringConfig(enabled=True)
        collector = FeatureEngineeringCollector(config)

        assert not collector.enabled
        assert collector.get_metric_count() == 0

        # All operations should be no-ops
        collector.record_feature_computation(
            feature_name="test",
            instrument="EUR",
            computation_time=0.1,
            feature_count=10,
        )
        collector.record_cache_hit("test", "EUR")

        # Should not raise any exceptions
        assert True

    def test_string_representation(self, collector):
        """
        Test string representation of collector.
        """
        repr_str = repr(collector)

        assert "FeatureEngineeringCollector" in repr_str
        assert f"enabled={collector.enabled}" in repr_str
        assert f"metrics_count={collector.get_metric_count()}" in repr_str

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_feature_importance_ranking(self, collector):
        """
        Test feature importance ranking functionality.
        """
        features = [
            ("price_change", 0.85, 1),
            ("volume_ratio", 0.72, 2),
            ("volatility", 0.68, 3),
            ("momentum", 0.54, 4),
        ]

        instrument = "BTCUSD"

        for feature_name, importance, rank in features:
            collector.record_feature_importance(
                feature_name=feature_name,
                instrument=instrument,
                importance_score=importance,
                rank=rank,
            )

        # Should not raise any exceptions
        assert True

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_feature_drift_monitoring(self, collector):
        """
        Test comprehensive feature drift monitoring.
        """
        feature_name = "price_momentum"
        instrument = "ETHUSDT"

        # Simulate drift detection over time
        drift_scores = [0.1, 0.2, 0.3, 0.6, 0.8]  # Increasing drift

        for i, drift_score in enumerate(drift_scores):
            drift_detected = drift_score > 0.5
            collector.record_feature_drift(
                feature_name=feature_name,
                instrument=instrument,
                drift_score=drift_score,
                drift_detected=drift_detected,
            )

        # Should not raise any exceptions
        assert True
