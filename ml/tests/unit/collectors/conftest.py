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
Test configuration and fixtures for metrics collectors tests.

This module provides common fixtures and utilities for testing metrics collectors,
including Prometheus registry management and test isolation.

"""

import uuid
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig


class MetricNameManager:
    """
    Manages unique metric names for test isolation.
    """

    def __init__(self):
        """
        Initialize the metric name manager.
        """
        self._prefix = f"test_{uuid.uuid4().hex[:8]}_"

    def get_unique_name(self, base_name: str) -> str:
        """
        Get a unique metric name for testing.
        """
        return f"{self._prefix}{base_name}"


@pytest.fixture
def metric_name_manager():
    """
    Provide a metric name manager for unique metric names.
    """
    return MetricNameManager()


@pytest.fixture
def monitoring_config():
    """
    Provide a basic monitoring configuration.
    """
    return MonitoringConfig(enabled=True, metrics_port=8081)


@pytest.fixture
def disabled_monitoring_config():
    """
    Provide a disabled monitoring configuration.
    """
    return MonitoringConfig(enabled=False)


@pytest.fixture(autouse=True)
def mock_prometheus_when_unavailable():
    """
    Mock Prometheus imports when not available to prevent import errors.
    """
    if not HAS_PROMETHEUS:
        with patch("ml._imports.HAS_PROMETHEUS", True):
            with (
                patch("ml._imports.Counter") as mock_counter,
                patch("ml._imports.Gauge") as mock_gauge,
                patch("ml._imports.Histogram") as mock_histogram,
            ):

                # Create mock metric classes that behave like Prometheus metrics
                mock_counter.return_value = MagicMock()
                mock_gauge.return_value = MagicMock()
                mock_histogram.return_value = MagicMock()

                yield {
                    "Counter": mock_counter,
                    "Gauge": mock_gauge,
                    "Histogram": mock_histogram,
                }
    else:
        yield None


@pytest.fixture
def prometheus_registry_cleanup():
    """
    Clean up Prometheus registry after each test to prevent conflicts.
    """
    collectors_before = set()

    if HAS_PROMETHEUS:
        try:
            from prometheus_client import REGISTRY

            collectors_before = set(REGISTRY._collector_to_names.keys())
        except ImportError:
            pass

    yield

    if HAS_PROMETHEUS:
        try:
            from prometheus_client import REGISTRY

            collectors_after = set(REGISTRY._collector_to_names.keys())
            new_collectors = collectors_after - collectors_before

            for collector in new_collectors:
                try:
                    REGISTRY.unregister(collector)
                except (KeyError, ValueError):
                    # Collector may have already been unregistered
                    pass
        except ImportError:
            pass


@pytest.fixture
def mock_data_catalog():
    """
    Provide a mock data catalog for testing.
    """
    catalog = MagicMock()
    catalog.instruments.return_value = ["EURUSD.SIM", "GBPUSD.SIM"]
    return catalog
