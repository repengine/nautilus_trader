"""Dashboard bootstrap utilities (kept separate to avoid heavy imports)."""

from __future__ import annotations


__all__ = [
    "DEFAULT_COMPOSE_FILE",
    "DEFAULT_HEALTH_CHECKS",
    "DEFAULT_SERVICES",
    "DashboardBootstrapError",
    "HealthCheck",
    "HealthStatus",
    "build_welcome_summary",
]

from .welcome import DEFAULT_COMPOSE_FILE
from .welcome import DEFAULT_HEALTH_CHECKS
from .welcome import DEFAULT_SERVICES
from .welcome import DashboardBootstrapError
from .welcome import HealthCheck
from .welcome import HealthStatus
from .welcome import build_welcome_summary

