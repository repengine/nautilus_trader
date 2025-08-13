"""
Package version helper.
"""

from __future__ import annotations


try:
    from importlib.metadata import version as _pkg_version
except Exception:  # pragma: no cover
    _pkg_version = None


def get_package_version(dist_name: str = "nautilus-trader-ml") -> str | None:
    try:
        if _pkg_version is None:
            return None
        return _pkg_version(dist_name)
    except Exception:
        return None


__all__ = ["get_package_version"]
