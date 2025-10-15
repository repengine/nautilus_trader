"""
Domain exceptions for the Dashboard module.
"""

from __future__ import annotations


class DashboardError(Exception):
    """Base error for dashboard operations."""


class ServiceControlUnsupportedError(DashboardError):
    """Raised when service control is not enabled or unavailable."""


class ServiceActionFailedError(DashboardError):
    """Raised when a control action fails."""


__all__ = [
    "DashboardError",
    "ServiceActionFailedError",
    "ServiceControlUnsupportedError",
]
