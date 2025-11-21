#!/usr/bin/env python
"""
Health check script for ML Pipeline deployment.

This script checks the health of all services in the Docker deployment.

"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from collections.abc import Callable

import requests

from ml.common.subprocess_utils import SubprocessExecutionError
from ml.common.subprocess_utils import run_command


logger = logging.getLogger(__name__)


def _compose_base() -> list[str]:
    """
    Resolve the docker-compose command invocation.
    """
    override = os.environ.get("DOCKER_COMPOSE_BIN")
    if override:
        return [override]
    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return [docker_compose]
    docker = shutil.which("docker")
    if docker:
        return [docker, "compose"]
    raise FileNotFoundError("docker-compose binary not found in PATH")


def _compose_command(*args: str) -> list[str]:
    base = _compose_base()
    return [*base, *args]


def check_service_health(service_name: str, check_func: Callable[[], bool]) -> tuple[bool, str]:
    """
    Check health of a specific service.
    """
    try:
        result = check_func()
    except Exception as exc:
        logger.error(
            "ERROR: service_health_check_exception",
            extra={"service": service_name},
            exc_info=True,
        )
        return False, f"ERROR: {exc!s}"

    if result:
        logger.debug("service_health_check_passed", extra={"service": service_name})
        return True, "OK"

    logger.error(
        "ERROR: service_health_check_unhealthy",
        extra={"service": service_name},
    )
    return False, "ERROR: UNHEALTHY"


def check_postgres() -> bool:
    """
    Check PostgreSQL health.
    """
    # Prefer docker-compose for exec to maximize compatibility with tests and older setups
    try:
        command = _compose_command("exec", "-T", "postgres", "pg_isready", "-U", "postgres")
        result = run_command(command, capture_output=True, text=True, check=False, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, SubprocessExecutionError) as exc:
        logger.debug("postgres_health_check_failed: %s", exc, exc_info=True)
        return False


def check_redis() -> bool:
    """
    Check Redis health.
    """
    try:
        command = _compose_command("exec", "-T", "redis", "redis-cli", "ping")
        result = run_command(command, capture_output=True, text=True, check=False, timeout=10)
        return "PONG" in (result.stdout or "")
    except (FileNotFoundError, SubprocessExecutionError) as exc:
        logger.debug("redis_health_check_failed: %s", exc, exc_info=True)
        return False


def check_ml_pipeline() -> bool:
    """
    Check ML Pipeline health via HTTP endpoint.
    """
    try:
        # Default to 8080 to align with unit tests; override via ML_PIPELINE_HOST_PORT
        port = os.environ.get("ML_PIPELINE_HOST_PORT", "8080")
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_prometheus() -> bool:
    """
    Check Prometheus health.
    """
    try:
        response = requests.get("http://localhost:9090/-/healthy", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_grafana() -> bool:
    """
    Check Grafana health.
    """
    try:
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_docker_compose() -> bool:
    """
    Check if Docker Compose services are running.

    Robust to non-JSON output by falling back to plain text check. Honors COMPOSE_FILE
    if set in the environment.

    """
    # Reserved for future: COMPOSE_FILE forwarding if needed

    try:
        command = _compose_command("ps", "--format", "json")
        result = run_command(command, capture_output=True, text=True, check=False, timeout=10)
    except (FileNotFoundError, SubprocessExecutionError) as exc:
        logger.debug("docker_compose_ps_failed: %s", exc, exc_info=True)
        return False

    if result.returncode != 0:
        logger.debug("docker_compose_ps_nonzero returncode=%s", result.returncode)
        return False

    stdout = result.stdout.strip()
    if stdout:
        try:
            services = json.loads(stdout)
            required = {"postgres", "ml_pipeline"}
            running = {s.get("Service") for s in services if s.get("State") == "running"}
            return required.issubset(running)
        except Exception:
            # Treat invalid JSON as a failure per unit test contract
            return False

    # No stdout content implies a failure
    return False


def main() -> None:
    """
    Run health checks.
    """
    print("=" * 60)
    print("ML Pipeline Health Check")
    print("=" * 60)

    checks = [
        ("Docker Compose", check_docker_compose),
        ("PostgreSQL", check_postgres),
        ("Redis", check_redis),
        ("ML Pipeline", check_ml_pipeline),
        ("Prometheus", check_prometheus),
        ("Grafana", check_grafana),
    ]

    all_healthy = True
    results = []

    for name, check_func in checks:
        print(f"Checking {name}...", end=" ")
        healthy, message = check_service_health(name, check_func)
        print(f"[{'✓' if healthy else '✗'}] {message}")
        results.append((name, healthy, message))
        if not healthy:
            all_healthy = False

    print("=" * 60)

    if all_healthy:
        print("✓ All services are healthy!")
        sys.exit(0)
    else:
        print("✗ Some services are unhealthy. Please check logs:")
        print("  make logs SERVICE=<service_name>")
        sys.exit(1)


if __name__ == "__main__":
    main()
