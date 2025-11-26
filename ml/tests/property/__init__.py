"""
Property test package bootstrap.

The property suite relies on the shared pytest plug-in under
``ml.tests.fixtures.pytest_plugins`` so every module automatically registers
canonical fixtures without importing ``ml.tests.conftest``.
"""

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)
