"""
Unit actor test package bootstrap.

Ensures the canonical pytest plug-in registers fixtures when importing tests
under ``ml.tests.unit.actors``.
"""

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

__all__ = ("pytest_plugins",)
