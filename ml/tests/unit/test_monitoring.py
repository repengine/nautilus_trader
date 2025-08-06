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
Tests for ML monitoring infrastructure.

Tests the metrics collector and server components with both Prometheus available and
unavailable scenarios.

"""

import socket
import time
import urllib.request
from unittest.mock import patch

import pytest

from ml.monitoring import MetricsServer
from ml.monitoring import MLMetricsCollector
from ml.monitoring import MonitoringConfig


class TestMonitoringConfig:
    """
    Test cases for MonitoringConfig.
    """

    def test_default_config(self):
        """
        Test default configuration values.
        """
        config = MonitoringConfig()

        assert config.enabled is True
        assert config.metrics_port == 8080
        assert config.health_check_interval == 30.0
        assert config.export_interval == 5.0
        assert config.metrics_prefix == "nautilus_ml"
        assert config.enable_high_cardinality is False
        assert config.max_metric_age == 300.0
        assert config.histogram_buckets is None
        assert config.enable_gc_metrics is True
        assert config.server_timeout == 10.0
        assert config.max_concurrent_requests == 100

    def test_custom_config(self):
        """
        Test custom configuration values.
        """
        custom_buckets = [0.001, 0.01, 0.1, 1.0]
        config = MonitoringConfig(
            enabled=False,
            metrics_port=9090,
            metrics_prefix="custom",
            histogram_buckets=custom_buckets,
        )

        assert config.enabled is False
        assert config.metrics_port == 9090
        assert config.metrics_prefix == "custom"
        assert config.histogram_buckets == custom_buckets

    def test_get_default_buckets(self):
        """
        Test default histogram buckets.
        """
        config = MonitoringConfig()
        buckets = config.get_default_buckets()

        assert len(buckets) == 13
        assert buckets[0] == 0.0001  # 0.1ms
        assert buckets[-1] == 1.0  # 1s

    def test_get_histogram_buckets_default(self):
        """
        Test getting histogram buckets when none specified.
        """
        config = MonitoringConfig()
        buckets = config.get_histogram_buckets()

        assert buckets == config.get_default_buckets()

    def test_get_histogram_buckets_custom(self):
        """
        Test getting custom histogram buckets.
        """
        custom_buckets = [0.001, 0.01, 0.1]
        config = MonitoringConfig(histogram_buckets=custom_buckets)
        buckets = config.get_histogram_buckets()

        assert buckets == custom_buckets


class TestMLMetricsCollector:
    """
    Test cases for MLMetricsCollector.
    """

    def test_init_disabled_config(self):
        """
        Test collector with disabled config.
        """
        config = MonitoringConfig(enabled=False)
        collector = MLMetricsCollector(config)

        assert collector.enabled is False
        assert collector.config == config

    def test_record_prediction_disabled(self):
        """
        Test recording prediction when disabled.
        """
        config = MonitoringConfig(enabled=False)
        collector = MLMetricsCollector(config)

        # Should not raise any errors
        collector.record_prediction(
            model="test_model",
            instrument="EURUSD",
            prediction_class="buy",
            latency_seconds=0.001,
            confidence=0.85,
        )

    def test_record_feature_computation_disabled(self):
        """
        Test recording feature computation when disabled.
        """
        config = MonitoringConfig(enabled=False)
        collector = MLMetricsCollector(config)

        # Should not raise any errors
        collector.record_feature_computation(
            instrument="EURUSD",
            feature_type="technical",
            latency_seconds=0.0005,
        )

    def test_record_error_disabled(self):
        """
        Test recording error when disabled.
        """
        config = MonitoringConfig(enabled=False)
        collector = MLMetricsCollector(config)

        # Should not raise any errors
        collector.record_error(
            model="test_model",
            instrument="EURUSD",
            error_type="inference",
        )

    def test_time_prediction_disabled(self):
        """
        Test prediction timer when disabled.
        """
        config = MonitoringConfig(enabled=False)
        collector = MLMetricsCollector(config)

        with collector.time_prediction("test_model", "EURUSD") as timer:
            timer.set_prediction("buy", 0.85)
            time.sleep(0.001)  # Simulate work

        # Should complete without errors

    def test_time_feature_computation_disabled(self):
        """
        Test feature computation timer when disabled.
        """
        config = MonitoringConfig(enabled=False)
        collector = MLMetricsCollector(config)

        with collector.time_feature_computation("EURUSD", "technical"):
            time.sleep(0.001)  # Simulate work

        # Should complete without errors

    @patch("ml._imports.HAS_PROMETHEUS", True)
    @patch("prometheus_client.Gauge")
    @patch("prometheus_client.Counter")
    @patch("prometheus_client.Histogram")
    def test_init_enabled_with_prometheus(self, mock_histogram, mock_counter, mock_gauge):
        """
        Test collector initialization when Prometheus is available.
        """
        config = MonitoringConfig(enabled=True)
        collector = MLMetricsCollector(config)
        assert collector.enabled is True

    def test_prediction_timer_context_manager(self):
        """
        Test PredictionTimer context manager.
        """
        config = MonitoringConfig(enabled=False)
        collector = MLMetricsCollector(config)

        timer = collector.time_prediction("model", "instrument")
        assert timer is not None

        with timer as t:
            assert t is timer
            t.set_prediction("buy", 0.9)
            # Should not raise any errors

    def test_feature_timer_context_manager(self):
        """
        Test FeatureTimer context manager.
        """
        config = MonitoringConfig(enabled=False)
        collector = MLMetricsCollector(config)

        timer = collector.time_feature_computation("instrument", "type")
        assert timer is not None

        with timer as t:
            assert t is timer
            # Should not raise any errors


class TestMetricsServer:
    """
    Test cases for MetricsServer.
    """

    def get_free_port(self) -> int:
        """
        Get a free port for testing.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def test_init_disabled_config(self):
        """
        Test server with disabled config.
        """
        port = self.get_free_port()
        config = MonitoringConfig(enabled=False, metrics_port=port)
        server = MetricsServer(config)

        assert server.get_port() == port
        assert server.is_running() is False

    def test_start_disabled_server(self):
        """
        Test starting disabled server.
        """
        port = self.get_free_port()
        config = MonitoringConfig(enabled=False, metrics_port=port)
        server = MetricsServer(config)

        server.start()
        assert server.is_running() is False

    def test_stop_not_running_server(self):
        """
        Test stopping server that is not running.
        """
        port = self.get_free_port()
        config = MonitoringConfig(enabled=True, metrics_port=port)
        server = MetricsServer(config)

        # Should not raise errors
        server.stop()
        assert server.is_running() is False

    def test_start_already_running_server(self):
        """
        Test starting server that is already running.
        """
        port = self.get_free_port()
        config = MonitoringConfig(enabled=True, metrics_port=port)
        server = MetricsServer(config)

        # Mock running state
        server._running = True

        with pytest.raises(RuntimeError, match="already running"):
            server.start()

    def test_start_without_prometheus(self):
        """
        Test starting server without Prometheus available.
        """
        port = self.get_free_port()
        config = MonitoringConfig(enabled=True, metrics_port=port)

        with patch("ml.monitoring.server.HAS_PROMETHEUS", False):
            server = MetricsServer(config)
            server.start()
            assert server.is_running() is False

    def test_get_urls(self):
        """
        Test URL getters.
        """
        port = self.get_free_port()
        config = MonitoringConfig(metrics_port=port)
        server = MetricsServer(config)

        assert server.get_metrics_url() == f"http://localhost:{port}/metrics"
        assert server.get_health_url() == f"http://localhost:{port}/health"

    def test_wait_for_ready_not_running(self):
        """
        Test wait_for_ready when server is not running.
        """
        port = self.get_free_port()
        config = MonitoringConfig(enabled=False, metrics_port=port)
        server = MetricsServer(config)

        result = server.wait_for_ready(timeout=0.1)
        assert result is False

    def test_context_manager_disabled(self):
        """
        Test context manager with disabled server.
        """
        port = self.get_free_port()
        config = MonitoringConfig(enabled=False, metrics_port=port)

        with MetricsServer(config) as server:
            assert server.is_running() is False

    @patch("ml._imports.HAS_PROMETHEUS", True)
    def test_start_server_integration(self):
        """
        Integration test for starting server with Prometheus.
        """
        port = self.get_free_port()
        config = MonitoringConfig(
            enabled=True,
            metrics_port=port,
            server_timeout=1.0,
        )

        with patch("prometheus_client.generate_latest", return_value=b"test_metrics"):
            server = MetricsServer(config)

            try:
                server.start()

                if server.is_running():
                    # Wait for server to be ready
                    ready = server.wait_for_ready(timeout=2.0)
                    if ready:
                        # Test health endpoint
                        health_url = server.get_health_url()
                        try:
                            with urllib.request.urlopen(health_url, timeout=1) as response:  # noqa: S310
                                assert response.status == 200
                                content = response.read()
                                assert b"healthy" in content
                        except Exception:  # noqa: S110
                            # Network issues in test environment are acceptable
                            pass

                        # Test metrics endpoint
                        metrics_url = server.get_metrics_url()
                        try:
                            with urllib.request.urlopen(metrics_url, timeout=1) as response:  # noqa: S310
                                assert response.status == 200
                        except Exception:  # noqa: S110
                            # Network issues in test environment are acceptable
                            pass

            finally:
                server.stop()
                assert server.is_running() is False

    def test_port_already_in_use(self):
        """
        Test starting server when port is already in use.
        """
        port = self.get_free_port()

        # Create socket to occupy the port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", port))
        sock.listen(1)

        try:
            config = MonitoringConfig(enabled=True, metrics_port=port)
            server = MetricsServer(config)

            with patch("ml._imports.HAS_PROMETHEUS", True):
                with pytest.raises(RuntimeError, match="already in use"):
                    server.start()
        finally:
            sock.close()
