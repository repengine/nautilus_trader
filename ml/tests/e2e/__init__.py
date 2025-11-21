"""
End-to-end tests for ML refactoring phases.

These tests verify that refactored components actually work in real-world
scenarios by performing full operations with real data structures while relying
on the canonical fixture plug-in.
"""

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)
