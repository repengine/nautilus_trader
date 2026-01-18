#!/usr/bin/env python3
"""
Regression tests ensuring the canonical fixture index stays authoritative.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from pkgutil import iter_modules
from types import ModuleType

import ml.tests.fixtures as fixtures

_FIXTURE_MODULE_NAMES = (
    "common",
    "database_fixtures",
    "datasets",
    "dummy_model",
    "integration",
    "mock_services",
    "mock_stores",
    "model_factory",
    "monitoring_collectors",
    "observability",
    "pandera",
    "runtime",
    "security",
    "stores",
    "streaming_events",
    "universes",
)


def _load_fixture_modules() -> dict[str, ModuleType]:
    """Load fixture modules on demand to avoid early imports."""
    return {
        name: importlib.import_module(f"{fixtures.__name__}.{name}")
        for name in _FIXTURE_MODULE_NAMES
    }


def test_fixtures_all_is_alphabetical() -> None:
    """The __all__ index should remain alphabetically sorted for quick scanning."""

    assert fixtures.__all__ == sorted(fixtures.__all__)


def test_fixtures_all_covers_submodules() -> None:
    """Ensure __all__ remains the single source of truth for fixture exports."""

    modules = _load_fixture_modules()
    expected = (
        set(modules["common"].__all__)
        | set(modules["database_fixtures"].__all__)
        | set(modules["datasets"].__all__)
        | set(modules["dummy_model"].__all__)
        | set(modules["integration"].__all__)
        | set(modules["mock_services"].__all__)
        | set(modules["mock_stores"].__all__)
        | set(modules["model_factory"].__all__)
        | set(modules["monitoring_collectors"].__all__)
        | set(modules["observability"].__all__)
        | set(modules["pandera"].__all__)
        | set(modules["runtime"].__all__)
        | set(modules["security"].__all__)
        | set(modules["stores"].__all__)
        | set(modules["streaming_events"].__all__)
        | set(modules["universes"].__all__)
    )
    actual = set(fixtures.__all__)

    missing = sorted(expected - actual)
    assert not missing, f"ml.tests.fixtures is missing exports: {missing}"

    for name in fixtures.__all__:
        # getattr should succeed for every exported name to keep the index authoritative.
        getattr(fixtures, name)


def test_fixtures_all_has_no_duplicates() -> None:
    """Guard against accidental duplicate exports that break alphabetical order."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in fixtures.__all__:
        if name in seen:
            duplicates.append(name)
        else:
            seen.add(name)

    assert not duplicates, f"Duplicate fixture exports detected: {sorted(duplicates)}"


def test_fixtures_module_is_importable() -> None:
    """The lint check doubles as a lightweight import smoke test for CI."""

    assert importlib.import_module("ml.tests.fixtures") is fixtures


def test_pytest_plugin_registry_matches_fixture_modules() -> None:
    """Ensure the pytest plug-in auto-loader stays aligned with fixture modules."""

    plugin_module = importlib.import_module("ml.tests.fixtures.pytest_plugins")
    discovery_excludes = getattr(plugin_module, "PLUGIN_DISCOVERY_EXCLUDES", frozenset())

    fixtures_dir = Path(fixtures.__file__).resolve().parent
    expected = {
        f"{fixtures.__name__}.{module.name}"
        for module in iter_modules([str(fixtures_dir)])
        if module.name not in {"__init__", "pytest_plugins"}
        and not module.name.startswith(("__", "test_"))
        and module.name not in discovery_excludes
    }

    assert set(plugin_module.pytest_plugins) == expected
