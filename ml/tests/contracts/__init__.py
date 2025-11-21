"""
Contract tests for ML components.

This package contains behavioral contract tests that verify ML components
conform to expected interfaces and behaviors. These tests focus on:

- Actor contract compliance (lifecycle, messaging, error handling)
- Model interface contracts (predict, fit, serialize behaviors)
- Strategy contract compliance (trading strategy behaviors)
- Training pipeline contracts (data flow, model persistence)
- Registry contracts (model versioning, metadata management)

Contract tests ensure components can be safely composed and that breaking
changes to interfaces are caught early.
"""

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

__all__ = ("pytest_plugins",)
