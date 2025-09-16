"""
Monitoring collectors test fixtures consolidated for reuse.

Extracted from ml/tests/unit/monitoring/collectors/conftest.py to reduce fixture
scattering and make discovery easier for contributors and tools.

"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml._imports import HAS_PROMETHEUS
from ml.monitoring._config import MonitoringConfig


class MetricNameManager:
    """
    Manages unique metric names for test isolation.
    """

    def __init__(self) -> None:
        self._prefix = f"test_{uuid.uuid4().hex[:8]}_"

    def get_unique_name(self, base_name: str) -> str:
        """
        Get a unique metric name for testing.
        """

        return f"{self._prefix}{base_name}"


@pytest.fixture
def metric_name_manager() -> MetricNameManager:
    """
    Provide a metric name manager for unique metric names.
    """

    return MetricNameManager()


@pytest.fixture
def monitoring_config(metric_name_manager: MetricNameManager) -> MonitoringConfig:
    """
    Provide a basic monitoring configuration with unique metrics prefix.
    """

    return MonitoringConfig(
        enabled=True,
        metrics_port=8081,
        metrics_prefix=metric_name_manager._prefix.rstrip("_"),
    )


@pytest.fixture
def disabled_monitoring_config() -> MonitoringConfig:
    """
    Provide a disabled monitoring configuration.
    """

    return MonitoringConfig(enabled=False)


@pytest.fixture(autouse=True)
def mock_prometheus_when_unavailable() -> Any:
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


@pytest.fixture(autouse=True)
def prometheus_registry_cleanup() -> Any:
    """
    Clean up Prometheus registry after each test to prevent conflicts.
    """

    names_before: set[str] = set()

    if HAS_PROMETHEUS:
        try:
            from prometheus_client import REGISTRY

            names_before = set(REGISTRY._names_to_collectors.keys())
        except (ImportError, AttributeError):
            pass

    yield

    if HAS_PROMETHEUS:
        try:
            from prometheus_client import REGISTRY

            names_after = set(REGISTRY._names_to_collectors.keys())
            new_names = names_after - names_before
            for name in new_names:
                try:
                    collector = REGISTRY._names_to_collectors.get(name)
                    if collector:
                        REGISTRY.unregister(collector)
                except (KeyError, ValueError, AttributeError):
                    pass
        except (ImportError, AttributeError):
            pass


@pytest.fixture
def mock_data_catalog() -> MagicMock:
    """
    Provide a mock data catalog for testing.
    """

    catalog = MagicMock()
    catalog.instruments.return_value = ["EURUSD.SIM", "GBPUSD.SIM"]
    return catalog


__all__ = [
    "MetricNameManager",
    "disabled_monitoring_config",
    "metric_name_manager",
    "mock_data_catalog",
    "mock_prometheus_when_unavailable",
    "monitoring_config",
    "prometheus_registry_cleanup",
]
