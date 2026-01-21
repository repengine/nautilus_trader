"""Resource usage helpers for memory monitoring."""

from __future__ import annotations

from typing import Any, cast


try:  # pragma: no cover - platform dependent
    import resource as _resource
except ImportError:  # pragma: no cover - windows fallback
    _resource = cast(Any, None)


def current_rss_mb() -> float | None:
    """Return current process RSS in MiB when available."""
    if _resource is None:
        return None
    try:
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
    except Exception:
        return None
    rss_value = float(getattr(usage, "ru_maxrss", 0.0))
    if rss_value <= 0.0:
        return None
    if rss_value > 1e8:
        return rss_value / (1024.0 * 1024.0)
    return rss_value / 1024.0


__all__ = ["current_rss_mb"]
