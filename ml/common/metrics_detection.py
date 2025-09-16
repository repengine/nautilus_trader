"""
Isolated metrics backend detection to prevent circular imports.

This module performs a minimal, side-effect-free check for the availability of the
Prometheus metrics backend, without importing any internal ML modules.

"""

from __future__ import annotations


try:
    import importlib as _importlib

    _importlib.import_module("prometheus_client")
    HAS_METRICS_BACKEND: bool = True
except Exception:  # pragma: no cover - optional dependency
    HAS_METRICS_BACKEND = False

__all__ = ["HAS_METRICS_BACKEND"]
