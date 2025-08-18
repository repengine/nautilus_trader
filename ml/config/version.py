"""
Package version helper.
"""

from __future__ import annotations

from collections.abc import Callable


_pkg_version_fn: Callable[[str], str] | None
try:
    from importlib.metadata import version as _pkg_version_fn
except Exception:  # pragma: no cover
    _pkg_version_fn = None

# Make the type explicit for mypy
_pkg_version: Callable[[str], str] | None = _pkg_version_fn


def get_package_version(dist_name: str = "nautilus-trader-ml") -> str | None:
    try:
        if _pkg_version is None:
            return None
        return _pkg_version(dist_name)
    except Exception:
        return None


__all__ = ["get_package_version"]
