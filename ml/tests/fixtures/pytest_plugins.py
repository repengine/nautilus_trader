#!/usr/bin/env python3
"""
pytest plug-in registrations for ML fixtures.

This module discovers every fixture module under ``ml.tests.fixtures`` so
repository-level ``conftest.py`` files can register them without diverging
from the canonical fixture list.  Adding a new fixture module now requires
no manual updates—dropping a ``*.py`` file beside the others is enough.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import iter_modules
from typing import Final

_FIXTURE_PACKAGE: Final = "ml.tests.fixtures"
_FIXTURE_DIR: Final = Path(__file__).resolve().parent
_EXCLUDED_NAMES: Final = {"__init__", "pytest_plugins"}
_EXCLUDED_PREFIXES: Final = ("__", "test_")
# Modules that do not register pytest fixtures (helper factories, docs, etc.)
PLUGIN_DISCOVERY_EXCLUDES: Final = frozenset({"model_factory"})


def _should_register(module_name: str) -> bool:
    """Return whether a fixture submodule should be auto-registered."""

    if module_name in _EXCLUDED_NAMES:
        return False
    if module_name.startswith(_EXCLUDED_PREFIXES):
        return False
    return True


def discover_fixture_plugins() -> tuple[str, ...]:
    """
    Discover all fixture modules that should be exposed via pytest plug-ins.
    """

    plugin_modules: list[str] = []
    for module in iter_modules([str(_FIXTURE_DIR)]):
        if _should_register(module.name) and module.name not in PLUGIN_DISCOVERY_EXCLUDES:
            plugin_modules.append(f"{_FIXTURE_PACKAGE}.{module.name}")
    priority = {"database_fixtures": 0, "dummy_model": 1}
    plugin_modules.sort(
        key=lambda name: (priority.get(name.rsplit(".", 1)[-1], 99), name),
    )
    return tuple(plugin_modules)


pytest_plugins: tuple[str, ...] = discover_fixture_plugins()

__all__ = ("PLUGIN_DISCOVERY_EXCLUDES", "discover_fixture_plugins", "pytest_plugins")
