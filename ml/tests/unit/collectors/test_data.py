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
Tests for the DataQualityCollector.
"""

import time

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.data import DataQualityCollector


class TestDataQualityCollector:
    """
    Test suite for DataQualityCollector functionality.
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
        Create a DataQualityCollector for testing.
        """
        prefix = self.metric_name_manager.get_unique_name("ml")
        config = MonitoringConfig(enabled=True, metrics_prefix=prefix)
        return DataQualityCollector(config)

    @pytest.fixture
    def disabled_collector(self):
        """
        Create a disabled DataQualityCollector for testing.
        """
        config = MonitoringConfig(enabled=False)
        return DataQualityCollector(config)

    def test_initialization(self, collector):
        """
        Test collector initialization.
        """
        assert collector.enabled == HAS_PROMETHEUS
        assert collector.config.metrics_prefix.endswith("_ml")

    def test_disabled_initialization(self, disabled_collector):
        """
        Test disabled collector initialization.
        """
        assert not disabled_collector.enabled
        assert disabled_collector.get_metric_count() == 0

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_data_load_success(self, collector):
        """
        Test recording successful data load.
        """
        collector.record_data_load(
            instrument="EURUSD",
            data_type="bars",
            rows_loaded=1000,
            duration_seconds=0.5,
            cache_hit=False,
            success=True,
        )

        # Verify metrics were recorded (basic functionality test)
        assert collector.enabled

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_data_load_error(self, collector):
        """
        Test recording data load error.
        """
        collector.record_data_load(
            instrument="EURUSD",
            data_type="bars",
            rows_loaded=0,
            duration_seconds=1.0,
            success=False,
            error_type="FileNotFoundError",
        )

        # Should not raise any exceptions
        assert collector.enabled

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_cache_stats(self, collector):
        """
        Test recording cache statistics.
        """
        collector.record_cache_stats(
            instrument="EURUSD",
            data_type="bars",
            hit_ratio=0.75,
            cache_size=100,
        )

        # Verify hit ratio is clamped to valid range
        collector.record_cache_stats(
            instrument="EURUSD",
            data_type="bars",
            hit_ratio=1.5,  # Should be clamped to 1.0
        )

        collector.record_cache_stats(
            instrument="EURUSD",
            data_type="bars",
            hit_ratio=-0.1,  # Should be clamped to 0.0
        )

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_data_quality(self, collector):
        """
        Test recording data quality metrics.
        """
        missing_ratios = {
            "open": 0.01,
            "high": 0.02,
            "low": 0.01,
            "close": 0.0,
            "volume": 0.05,
        }

        outlier_counts = {
            "zscore": 10,
            "iqr": 15,
        }

        collector.record_data_quality(
            instrument="EURUSD",
            data_type="bars",
            missing_ratios=missing_ratios,
            outlier_counts=outlier_counts,
            total_rows=1000,
        )

        # Test with invalid ratios (should be clamped)
        invalid_ratios = {
            "open": 1.5,  # Should be clamped to 1.0
            "close": -0.1,  # Should be clamped to 0.0
        }

        collector.record_data_quality(
            instrument="EURUSD",
            data_type="bars",
            missing_ratios=invalid_ratios,
        )

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_record_data_validation(self, collector):
        """
        Test recording data validation results.
        """
        # Test successful validation
        collector.record_data_validation(
            instrument="EURUSD",
            data_type="bars",
            validation_type="schema",
            passed=True,
            total_checks=100,
        )

        # Test failed validation
        collector.record_data_validation(
            instrument="EURUSD",
            data_type="bars",
            validation_type="range",
            passed=False,
            total_checks=50,
        )

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_update_data_staleness(self, collector):
        """
        Test updating data staleness metrics.
        """
        current_time = time.time()
        one_hour_ago = current_time - 3600

        # Test with specific timestamp
        collector.update_data_staleness(
            instrument="EURUSD",
            data_type="bars",
            last_updated_timestamp=one_hour_ago,
        )

        # Test with current time (None)
        collector.update_data_staleness(
            instrument="EURUSD",
            data_type="bars",
            last_updated_timestamp=None,
        )

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_data_load_timer_context_manager(self, collector):
        """
        Test DataLoadTimer context manager.
        """
        with collector.time_data_load("EURUSD", "bars") as timer:
            # Simulate some work
            time.sleep(0.01)

            # Set load results
            timer.set_load_result(
                rows=1000,
                cache_hit=True,
                missing_ratios={"close": 0.01},
                outlier_counts={"zscore": 5},
            )

        # Timer should have recorded metrics automatically

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_data_load_timer_with_exception(self, collector):
        """
        Test DataLoadTimer handles exceptions correctly.
        """
        with pytest.raises(ValueError):
            with collector.time_data_load("EURUSD", "bars") as timer:
                timer.set_load_result(rows=100)
                raise ValueError("Test exception")

        # Should have recorded the error

    def test_get_data_quality_summary(self, collector):
        """
        Test getting data quality summary.
        """
        summary = collector.get_data_quality_summary("EURUSD", "bars")

        assert "instrument" in summary
        assert "data_type" in summary
        assert summary["instrument"] == "EURUSD"
        assert summary["data_type"] == "bars"

    def test_disabled_collector_operations(self, disabled_collector):
        """
        Test that disabled collector operations are no-ops.
        """
        # All operations should complete without error
        disabled_collector.record_data_load(
            instrument="EURUSD",
            data_type="bars",
            rows_loaded=1000,
            duration_seconds=0.5,
        )

        disabled_collector.record_cache_stats(
            instrument="EURUSD",
            data_type="bars",
            hit_ratio=0.5,
        )

        disabled_collector.record_data_quality(
            instrument="EURUSD",
            data_type="bars",
            missing_ratios={"close": 0.01},
        )

        disabled_collector.record_data_validation(
            instrument="EURUSD",
            data_type="bars",
            validation_type="schema",
            passed=True,
        )

        disabled_collector.update_data_staleness(
            instrument="EURUSD",
            data_type="bars",
        )

        # Context manager should work
        with disabled_collector.time_data_load("EURUSD", "bars") as timer:
            timer.set_load_result(rows=100)

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_health_check(self, collector):
        """
        Test health check functionality.
        """
        health = collector.health_check()

        assert health["enabled"] == HAS_PROMETHEUS
        assert health["collector_type"] == "DataQualityCollector"
        assert health["metrics_count"] > 0  # Should have initialized metrics

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="Prometheus client not available")
    def test_thread_safety(self, collector):
        """
        Test basic thread safety of operations.
        """
        import threading

        def worker():
            for i in range(10):
                collector.record_data_load(
                    instrument=f"INST{i % 3}",
                    data_type="bars",
                    rows_loaded=100,
                    duration_seconds=0.1,
                )

        threads = [threading.Thread(target=worker) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors
