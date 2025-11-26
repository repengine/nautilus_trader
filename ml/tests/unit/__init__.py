"""
Unit tests for ML components.

All modules under this package rely on the shared pytest plug-in so fixtures stay
centralized and automatically available.
"""

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

__all__ = ("pytest_plugins",)
