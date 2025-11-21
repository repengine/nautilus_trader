"""
Performance and benchmark tests for ML components.

This package contains performance-critical tests that verify ML components
meet latency and memory requirements. These tests focus on:

- Hot path performance (< 5ms end-to-end signal generation)
- Zero allocation guarantees (no GC pressure during trading)
- Memory leak detection (stable over 24h runs)
- Regression testing (ensures performance doesn't degrade)
- Benchmark comparisons (baseline vs current performance)

Performance tests are automatically run when ML inference or feature
computation files change, with 20% regression tolerance.
"""

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

__all__ = ("pytest_plugins",)
