"""
Shared task-level entry points for ML CLI commands.

Each submodule exposes typed, reusable functions orchestrating cold-path
operations. The ``ml.cli`` package imports from here to keep CLI wrappers thin
and testable.

"""

from __future__ import annotations

import importlib
from types import ModuleType


__all__ = [
    "datasets",
    "db",
    "dev",
    "ingest",
    "monitoring",
    "observability",
    "registry",
    "training",
]

_SUBMODULES = {name: f"{__name__}.{name}" for name in __all__}


def __getattr__(name: str) -> ModuleType:
    if name in _SUBMODULES:
        module = importlib.import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(__all__)))
