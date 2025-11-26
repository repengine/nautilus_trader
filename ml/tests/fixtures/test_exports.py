#!/usr/bin/env python3
"""
Regression tests ensuring the canonical fixture index stays authoritative.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from pkgutil import iter_modules

import ml.tests.fixtures as fixtures
from ml.tests.fixtures import common
from ml.tests.fixtures import database_fixtures
from ml.tests.fixtures import datasets
from ml.tests.fixtures import dummy_model
from ml.tests.fixtures import integration
from ml.tests.fixtures import mock_services
from ml.tests.fixtures import mock_stores
from ml.tests.fixtures import model_factory
from ml.tests.fixtures import monitoring_collectors
from ml.tests.fixtures import observability
from ml.tests.fixtures import security
from ml.tests.fixtures import streaming_events
from ml.tests.fixtures import pandera as pandera_fixtures
from ml.tests.fixtures import runtime
from ml.tests.fixtures import stores
from ml.tests.fixtures import universes


def test_fixtures_all_is_alphabetical() -> None:
    """The __all__ index should remain alphabetically sorted for quick scanning."""

    assert fixtures.__all__ == sorted(fixtures.__all__)


def test_fixtures_all_covers_submodules() -> None:
    """Ensure __all__ remains the single source of truth for fixture exports."""

    expected = (
        set(common.__all__)
        | set(database_fixtures.__all__)
        | set(datasets.__all__)
        | set(dummy_model.__all__)
        | set(integration.__all__)
        | set(mock_services.__all__)
        | set(mock_stores.__all__)
        | set(model_factory.__all__)
        | set(monitoring_collectors.__all__)
        | set(observability.__all__)
        | set(pandera_fixtures.__all__)
        | set(runtime.__all__)
        | set(security.__all__)
        | set(stores.__all__)
        | set(streaming_events.__all__)
        | set(universes.__all__)
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
