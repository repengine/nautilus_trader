"""
Integration tests for ML module with Nautilus Trader infrastructure.

This package contains integration tests that validate ML components work correctly
with Nautilus Trader's core infrastructure including:
- ParquetDataCatalog for data loading
- BacktestEngine for strategy testing
- Nautilus indicators for feature computation
- Message bus for Actor communication

"""

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

import pytest

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)
