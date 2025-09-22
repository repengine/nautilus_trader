"""Utilities for orchestrating the dashboard stack from a single CLI entrypoint."""

from __future__ import annotations

import subprocess
import textwrap
import time
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import requests
from requests import Response


ServiceKind = Literal["service", "dependency", "dashboard"]


DEFAULT_COMPOSE_FILE = "ml/deployment/docker-compose.yml"
DEFAULT_SERVICES: tuple[str, ...] = (
    "postgres",
    "redis",
    "ml_signal_actor",
    "ml_strategy",
    "ml_pipeline",
    "prometheus",
    "grafana",
    "ml_dashboard",
)


@dataclass(slots=True, frozen=True)
class HealthCheck:
    """HTTP health probe definition."""

    name: str
    kind: ServiceKind
    url: str
    expected_status: tuple[int, ...] = (200,)


@dataclass(slots=True, frozen=True)
class HealthStatus:
    """Result of a health probe."""

    check: HealthCheck
    healthy: bool
    status_code: int | None
    error: str | None


DEFAULT_HEALTH_CHECKS: tuple[HealthCheck, ...] = (
    HealthCheck(name="ML Signal Actor", kind="service", url="http://localhost:8000/health"),
    HealthCheck(name="ML Strategy", kind="service", url="http://localhost:8001/health"),
    HealthCheck(name="ML Pipeline", kind="service", url="http://localhost:8081/health"),
    HealthCheck(name="Dashboard API", kind="dashboard", url="http://localhost:8010/health"),
    HealthCheck(name="Prometheus", kind="dependency", url="http://localhost:9090/-/healthy"),
    HealthCheck(name="Grafana", kind="dependency", url="http://localhost:3000/api/health"),
)


class DashboardBootstrapError(RuntimeError):
    """Raised when the dashboard bootstrap CLI cannot continue."""


def start_services(
    *,
    compose_file: str = DEFAULT_COMPOSE_FILE,
    services: Sequence[str] = DEFAULT_SERVICES,
    detach: bool = True,
) -> None:
    """Invoke docker compose to start the dashboard stack."""
    command = ["docker", "compose", "-f", compose_file, "up"]
    if detach:
        command.append("-d")
    command.extend(services)

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - environment dependent
        raise DashboardBootstrapError(
            "docker compose not found; ensure Docker is installed and available",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise DashboardBootstrapError(
            f"docker compose failed with exit code {exc.returncode}: {exc.stderr.decode().strip()}",
        ) from exc


def probe_health(check: HealthCheck, *, timeout: float) -> HealthStatus:
    """Perform a single HTTP health probe."""
    try:
        response: Response = requests.get(check.url, timeout=timeout)
        code = response.status_code
        healthy = code in check.expected_status
        return HealthStatus(check=check, healthy=healthy, status_code=code, error=None)
    except requests.RequestException as exc:
        return HealthStatus(
            check=check,
            healthy=False,
            status_code=None,
            error=str(exc),
        )


def gather_health(
    checks: Iterable[HealthCheck],
    *,
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> list[HealthStatus]:
    """Probe health checks with basic retry support."""
    remaining_retries = max(0, retries)
    pending = list(checks)
    statuses: list[HealthStatus] = []

    while pending:
        statuses = [probe_health(check, timeout=timeout) for check in pending]
        unhealthy = [status for status in statuses if not status.healthy]
        if not unhealthy or remaining_retries <= 0:
            break
        time.sleep(sleep_seconds)
        remaining_retries -= 1
        pending = [status.check for status in unhealthy]

    return statuses


def format_summary(statuses: Sequence[HealthStatus]) -> str:
    """Render a human-friendly summary for the dashboard welcome screen."""
    lines: list[str] = []
    lines.append("Nautilus Trader ML Dashboard")
    lines.append("=" * len(lines[0]))
    lines.append("")

    icon = {True: "✅", False: "❌"}
    for group in ("service", "dependency", "dashboard"):
        group_statuses = [status for status in statuses if status.check.kind == group]
        if not group_statuses:
            continue
        lines.append(group.title())
        lines.append("-" * len(group))
        for status in group_statuses:
            note: str
            if status.healthy:
                note = f"HTTP {status.status_code}" if status.status_code is not None else "healthy"
            else:
                if status.status_code is not None:
                    note = f"HTTP {status.status_code}"
                elif status.error:
                    note = status.error.split("\n", maxsplit=1)[0]
                else:
                    note = "unreachable"
            lines.append(f"  {icon[status.healthy]} {status.check.name}: {note}")
        lines.append("")

    lines.append("Quick Links")
    lines.append("-----------")
    lines.append("  • Dashboard UI  : http://localhost:8010/")
    lines.append("  • Grafana       : http://localhost:3000/")
    lines.append("  • Prometheus    : http://localhost:9090/")
    lines.append("")
    lines.append(textwrap.fill(
        "Run `docker compose -f ml/deployment/docker-compose.yml logs -f ml_dashboard` "
        "for live logs, or use `CTRL+C` to stop the welcome screen once you've "
        "verified all components.",
        width=90,
    ))
    return "\n".join(lines).strip()


def build_welcome_summary(
    *,
    compose_file: str = DEFAULT_COMPOSE_FILE,
    services: Sequence[str] = DEFAULT_SERVICES,
    checks: Sequence[HealthCheck] = DEFAULT_HEALTH_CHECKS,
    timeout_seconds: float = 5.0,
    retries: int = 5,
    retry_interval_seconds: float = 2.0,
    start: bool = True,
) -> str:
    """Start services (optional) and return a formatted welcome summary."""
    if start:
        start_services(compose_file=compose_file, services=services)

    statuses = gather_health(
        checks,
        timeout=timeout_seconds,
        retries=retries,
        sleep_seconds=retry_interval_seconds,
    )
    return format_summary(statuses)
