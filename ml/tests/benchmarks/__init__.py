"""
Benchmark tests for ML components.

This package contains performance benchmarks that validate optimization targets and
ensure ML components meet latency requirements for production trading.
"""

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

__all__ = ("pytest_plugins",)
