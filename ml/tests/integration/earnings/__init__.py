"""
Integration tests for earnings data pipeline.

This module contains end-to-end integration tests for the earnings data pipeline,
including EDGAR fetching, PostgreSQL storage, feature computation, and data quality validation.
"""

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

import pytest

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)
