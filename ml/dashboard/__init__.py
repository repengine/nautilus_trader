"""
Dashboard control and visualization API (cold path).

This package provides a small, typed control-plane service exposing health and
orchestration endpoints for the Nautilus Trader ML system. It adheres to the
project's Universal ML Architecture Patterns:

- Cold-path only (no hot-path modifications).
- Centralized metrics bootstrap (`ml.common.metrics_bootstrap`).
- Structured logging via `ml.common.logging_config` (structlog + stdlib interop).
- Events and topics: use `ml.config.events` enums and
  `ml.common.message_topics.build_topic_for_stage` when publishing.

Public API is intentionally minimal; consumers should import from this package
only (not internal modules).

Feature Flag:
    ML_USE_LEGACY_DASHBOARD_SERVICE=1 → Use legacy monolithic DashboardService
    ML_USE_LEGACY_DASHBOARD_SERVICE=0 → Use new DashboardServiceFacade (default)
"""

import os

from ml.dashboard.app import create_app
from ml.dashboard.config import DashboardConfig


# Feature flag for gradual migration
_USE_LEGACY = os.getenv("ML_USE_LEGACY_DASHBOARD_SERVICE", "0") == "1"

if _USE_LEGACY:
    from ml.dashboard.service import DashboardService
else:
    from ml.dashboard.service_facade import DashboardServiceFacade as DashboardService


__all__ = [
    "DashboardConfig",
    "DashboardService",
    "create_app",
]
