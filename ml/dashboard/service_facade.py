"""
Compatibility shim for dashboard service facade imports.
"""

from __future__ import annotations

from ml.dashboard.service import DashboardService


DashboardServiceFacade = DashboardService

__all__ = [
    "DashboardService",
    "DashboardServiceFacade",
]
