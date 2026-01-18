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

"""

from ml.dashboard.app import create_app
from ml.dashboard.config import DashboardConfig
from ml.dashboard.service import DashboardService


__all__ = [
    "DashboardConfig",
    "DashboardService",
    "create_app",
]
