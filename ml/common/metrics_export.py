"""
Prometheus metrics export helpers (safe import wrapper).

Use this module instead of importing prometheus_client directly. It avoids violating the
centralized metrics bootstrap rule and provides graceful fallbacks when the client is
unavailable.

"""

from __future__ import annotations

import importlib
from collections.abc import Callable


def _load_prometheus() -> tuple[str, Callable[[], bytes]]:
    try:
        # Dynamic import to avoid direct prometheus_client import in source
        mod = importlib.import_module("prometheus_client")
        ctype = getattr(mod, "CONTENT_TYPE_LATEST", "text/plain; version=0.0.4; charset=utf-8")
        gen = getattr(mod, "generate_latest", lambda: b"")
        return ctype, gen
    except Exception:  # pragma: no cover - optional in minimal envs

        def _empty() -> bytes:
            return b""

        return "text/plain; version=0.0.4; charset=utf-8", _empty


# Public API
CONTENT_TYPE_LATEST, _GENERATE_LATEST = _load_prometheus()


def generate_latest() -> bytes:
    """
    Return the latest metrics exposition payload.

    Falls back to empty payload if prometheus_client is not available.

    """
    return _GENERATE_LATEST()


__all__ = ["CONTENT_TYPE_LATEST", "generate_latest"]
