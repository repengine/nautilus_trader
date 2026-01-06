#!/usr/bin/env python3
# ruff: noqa: RUF022
"""
Test fixtures for ML testing.

This package lazily re-exports every helper under ``ml.tests.fixtures`` so
pytest's assertion rewriting can instrument each fixture module before it is
imported.  Attribute lookups only import the submodule that defines the requested
symbol, keeping startup lean for both pytest and ad-hoc scripts.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from pkgutil import iter_modules
from types import ModuleType
from typing import Any, Final, Iterable

_FIXTURE_DIR: Final[Path] = Path(__file__).resolve().parent
_EXCLUDED_MODULES: Final = frozenset({"__init__", "pytest_plugins"})
_EXCLUDED_PREFIXES: Final = ("__", "test_")
_BUILDER_EXPORTS: Final = frozenset({"DataBuilder", "MLConfigBuilder", "MockBuilder", "RegistryBuilder"})

_MODULE_CACHE: dict[str, ModuleType] = {}
_ALL_EXPORTS: list[str] | None = None


def _discover_fixture_modules() -> tuple[str, ...]:
    module_names: list[str] = []
    for module in iter_modules([str(_FIXTURE_DIR)]):
        if module.name in _EXCLUDED_MODULES or module.name.startswith(_EXCLUDED_PREFIXES):
            continue
        module_names.append(module.name)
    return tuple(sorted(module_names))


_FIXTURE_MODULES: Final = _discover_fixture_modules()


def _load_module(module_name: str) -> ModuleType:
    qualified_name = f"{__name__}.{module_name}"
    module = _MODULE_CACHE.get(qualified_name)
    if module is None:
        module = importlib.import_module(qualified_name)
        _MODULE_CACHE[qualified_name] = module
    return module


def _iter_module_exports(module: ModuleType) -> Iterable[str]:
    exports = getattr(module, "__all__", None)
    if exports is not None:
        return tuple(exports)
    return tuple(name for name in vars(module) if not name.startswith("_"))


def _build_all_exports() -> list[str]:
    global _ALL_EXPORTS
    if _ALL_EXPORTS is not None:
        return _ALL_EXPORTS

    names: set[str] = set(_BUILDER_EXPORTS)
    for module_name in _FIXTURE_MODULES:
        module = _load_module(module_name)
        names.update(_iter_module_exports(module))

    _ALL_EXPORTS = sorted(names)
    return _ALL_EXPORTS


def _resolve_from_fixture_modules(name: str) -> Any:
    for module_name in _FIXTURE_MODULES:
        module = _load_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


def _load_builder_attribute(name: str) -> Any:
    from ml.tests import builders as _builders

    return getattr(_builders, name)


def __getattr__(name: str) -> Any:  # pragma: no cover - import-time helper
    if name == "__all__":
        exports = _build_all_exports()
        globals()["__all__"] = exports
        return exports

    if name in _FIXTURE_MODULES:
        module = _load_module(name)
        globals()[name] = module
        return module

    if name in _BUILDER_EXPORTS:
        value = _load_builder_attribute(name)
        globals()[name] = value
        return value

    try:
        value = _resolve_from_fixture_modules(name)
    except AttributeError as exc:  # pragma: no cover - mirrors default behavior
        raise AttributeError(name) from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - developer convenience
    dynamic_names = set(globals())
    dynamic_names.update(_build_all_exports())
    return sorted(dynamic_names)
