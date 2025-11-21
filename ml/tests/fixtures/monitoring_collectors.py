"""
Monitoring collectors test fixtures consolidated for reuse.

Extracted from ml/tests/unit/monitoring/collectors/conftest.py to reduce fixture
scattering and make discovery easier for contributors and tools.

"""

from __future__ import annotations

import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
import sys
from typing import Any, Callable, Generator, Iterable
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


@pytest.fixture(autouse=False)
def mock_prometheus_when_unavailable() -> Any:
    """
    Mock Prometheus imports when not available to prevent import errors.

    This fixture is OPT-IN - tests must explicitly request it in their signature.
    Use this fixture only in tests that need Prometheus mocking when the package
    is not installed.

    Usage:
        def test_something(mock_prometheus_when_unavailable):
            # Your test code here
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


@pytest.fixture()
def prometheus_registry_cleanup() -> Any:
    """Clean up Prometheus registry after each test (opt-in only).

    This fixture unregisters metrics created during a test to prevent
    registry conflicts. Only use this fixture in tests that explicitly
    need registry isolation.

    WARNING: This fixture was changed from autouse=True to opt-in
    to prevent destructive cleanup of metrics needed by subsequent tests.
    Most tests should use unique metric names instead of relying on cleanup.

    Usage:
        def test_something(prometheus_registry_cleanup):
            # Your test code here
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


def _snapshot_registry_state(registry: Any) -> set[str]:
    names: set[str] = set()
    names_to_collectors = getattr(registry, "_names_to_collectors", None)
    if isinstance(names_to_collectors, dict):
        names = set(names_to_collectors.keys())
    return names


@dataclass(frozen=True, slots=True)
class PrometheusRegistryHarness:
    """Container exposing the active Prometheus registry and helpers."""

    registry: Any
    reload_modules: Callable[[Iterable[str] | str], None]


@contextmanager
def patch_prometheus_registry(
    modules: Iterable[str] | str | None = None,
) -> Generator[PrometheusRegistryHarness, None, None]:
    """
    Provide deterministic cleanup around the global Prometheus registry.

    Args:
        modules: Optional module (or iterable of modules) to evict from
            ``sys.modules`` so they can be re-imported with fresh state.

    Yields:
        PrometheusRegistryHarness exposing the active registry and a helper
        to drop modules from the import cache.

    Raises:
        pytest.SkipTest: When Prometheus is unavailable in the runtime.
    """

    if not HAS_PROMETHEUS:
        pytest.skip("prometheus_client not available; patch_prometheus_registry requires it")

    from prometheus_client import REGISTRY  # Local import to avoid hard dependency
    from ml.common import metrics_bootstrap

    metrics_bootstrap.reset_metrics_cache()
    baseline_names = _snapshot_registry_state(REGISTRY)

    def _reload_modules(targets: Iterable[str] | str) -> None:
        resolved = (targets,) if isinstance(targets, str) else tuple(targets)
        for module_path in resolved:
            sys.modules.pop(module_path, None)

    harness = PrometheusRegistryHarness(
        registry=REGISTRY,
        reload_modules=_reload_modules,
    )

    if modules:
        harness.reload_modules(modules)

    try:
        yield harness
    finally:
        metrics_bootstrap.reset_metrics_cache()
        try:
            current_names = _snapshot_registry_state(REGISTRY)
            extra_names = current_names - baseline_names
            names_to_collectors = getattr(REGISTRY, "_names_to_collectors", {})
            for name in extra_names:
                collector = names_to_collectors.get(name)
                if collector is not None:
                    with suppress(KeyError, ValueError, AttributeError):
                        REGISTRY.unregister(collector)
        except AttributeError:
            pass


@pytest.fixture
def isolated_prometheus_registry() -> Generator[PrometheusRegistryHarness, None, None]:
    """
    Provide deterministic registry cleanup for tests.

    Returns:
        PrometheusRegistryHarness exposing the shared registry and helpers.
    """

    with patch_prometheus_registry() as harness:
        yield harness


__all__ = [
    "MetricNameManager",
    "PrometheusRegistryHarness",
    "disabled_monitoring_config",
    "isolated_prometheus_registry",
    "metric_name_manager",
    "mock_data_catalog",
    "mock_prometheus_when_unavailable",
    "monitoring_config",
    "patch_prometheus_registry",
    "prometheus_registry_cleanup",
]
