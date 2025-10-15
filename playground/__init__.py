"""Playground research utilities for experimental ML workflows."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


__all__ = ["risk_model"]


def __getattr__(name: str) -> ModuleType:
    if name == "risk_model":
        return import_module("playground.risk_model")
    msg = f"module 'playground' has no attribute '{name}'"
    raise AttributeError(msg)
