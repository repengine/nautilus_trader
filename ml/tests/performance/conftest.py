"""
Performance test package configuration.

This conftest ensures that all tests under ml/tests/performance are:
- Marked with the `performance` marker for easy selection/exclusion.
- Skipped when running under coverage or xdist, where microbenchmarks and
  zero-allocation assertions are invalid/unstable due to tracing overhead
  and parallel native extensions.

Rationale
- Coverage tracing perturbs latency and allocations, invalidating guardrails.
- xdist parallelism can oversubscribe BLAS/onnxruntime/XGBoost and crash
  workers. We run guardrails separately (see Makefile targets).
"""

from __future__ import annotations

import os
import sys
import pytest


def _under_xdist() -> bool:
    return bool(os.getenv("PYTEST_XDIST_WORKER"))


def _under_coverage() -> bool:
    # Heuristics to detect pytest-cov/coverage active in the current process
    if os.getenv("COV_CORE_SOURCE") or os.getenv("COVERAGE_PROCESS_START"):
        return True
    addopts = os.getenv("PYTEST_ADDOPTS", "")
    if "--cov" in addopts:
        return True
    gettrace = getattr(sys, "gettrace", None)
    if callable(gettrace) and gettrace():  # Coverage sets a C tracer
        return True
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Mark all tests in this package as performance for consistent selection
    for item in items:
        if "performance" not in item.keywords:
            item.add_marker(pytest.mark.performance)

    # Skip performance tests under coverage or xdist (see rationale above)
    if _under_xdist() or _under_coverage():
        skip_marker = pytest.mark.skip(
            reason="Skip performance tests under coverage or xdist (invalid/unstable)"
        )
        for item in items:
            item.add_marker(skip_marker)

