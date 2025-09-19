"""
Typed configuration for the Dashboard control-plane service (cold path).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


_DEFAULT_TIMEOUT_SECONDS: Final[float] = 2.5


@dataclass(slots=True, frozen=True)
class DashboardConfig:
    """
    Configuration for the Dashboard service.

    Attributes
    ----------
    compose_enabled : bool
        Whether to enable Docker Compose service control (cold path only).
    compose_file : Path | None
        Optional override for compose file. When None, detection is attempted
        by callers; control actions should be disabled if not found.
    request_timeout_seconds : float
        Default HTTP timeout for health probes.
    actor_port : int
        Host port for the ML signal actor health/metrics endpoints.
    strategy_port : int
        Host port for the ML strategy node health/metrics endpoints.
    pipeline_port : int
        Host port mapped to the pipeline service health/metrics endpoints.
    grafana_port : int
        Host port for Grafana (`/api/health`).
    prometheus_port : int
        Host port for Prometheus (`/-/healthy`).
    redis_port : int
        Host port for Redis (used for liveness-only pings in optional checks).
    """

    compose_enabled: bool = False
    compose_file: Path | None = None
    request_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    actor_port: int = 8000
    strategy_port: int = 8001
    pipeline_port: int = 8081
    grafana_port: int = 3000
    prometheus_port: int = 9090
    redis_port: int = 6380

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> DashboardConfig:
        """
        Build a config from environment variables.
        """
        e = env or {}

        def _truthy(name: str, default: bool) -> bool:
            val = e.get(name)
            if val is None:
                import os

                val = os.getenv(name)
            if val is None:
                return default
            return val.strip().lower() in {"1", "true", "yes", "y", "on"}

        def _int(name: str, default: int) -> int:
            raw = e.get(name)
            if raw is None:
                import os

                raw = os.getenv(name)
            try:
                return int(raw) if raw is not None and raw != "" else default
            except Exception:
                return default

        def _float(name: str, default: float) -> float:
            raw = e.get(name)
            if raw is None:
                import os

                raw = os.getenv(name)
            try:
                return float(raw) if raw is not None and raw != "" else default
            except Exception:
                return default

        compose_path = e.get("ML_DASHBOARD_COMPOSE_FILE")
        if compose_path is None:
            import os

            compose_path = os.getenv("ML_DASHBOARD_COMPOSE_FILE")

        return DashboardConfig(
            compose_enabled=_truthy("ML_DASHBOARD_USE_COMPOSE", False),
            compose_file=Path(compose_path) if compose_path else None,
            request_timeout_seconds=_float("ML_DASHBOARD_TIMEOUT", _DEFAULT_TIMEOUT_SECONDS),
            actor_port=_int("ML_ACTOR_HOST_PORT", 8000),
            strategy_port=_int("ML_STRATEGY_HOST_PORT", 8001),
            pipeline_port=_int("ML_PIPELINE_HOST_PORT", 8081),
            grafana_port=_int("GRAFANA_HOST_PORT", 3000),
            prometheus_port=_int("PROMETHEUS_HOST_PORT", 9090),
            redis_port=_int("REDIS_HOST_PORT", 6380),
        )


__all__ = ["DashboardConfig"]

