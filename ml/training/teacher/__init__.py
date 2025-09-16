"""
Teacher models and CLIs for training pipelines.

Public API exposes the CLI module for tests and orchestrators via a lazy attribute to
avoid import-time overhead.

"""

from __future__ import annotations

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType


__all__ = [
    "tft_cli",
]


def __getattr__(name: str) -> _ModuleType:
    if name == "tft_cli":
        return _import_module("ml.training.teacher.tft_cli")
    raise AttributeError(name)
