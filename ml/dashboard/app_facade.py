"""
Compatibility shim for dashboard app facade imports.
"""

from __future__ import annotations

from ml.dashboard.app import create_app as create_app_facade


__all__ = [
    "create_app_facade",
]
