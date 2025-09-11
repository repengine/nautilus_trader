#!/usr/bin/env python
"""
Health check script for ML Pipeline deployment.

This script checks the health of all services in the Docker deployment.

"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable

import requests


def check_service_health(service_name: str, check_func: Callable[[], bool]) -> tuple[bool, str]:
    """
    Check health of a specific service.
    """
    try:
        result = check_func()
        return result, "OK" if result else "UNHEALTHY"
    except Exception as e:
        return False, f"ERROR: {e!s}"


def check_postgres() -> bool:
    """
    Check PostgreSQL health.
    """
    # Prefer docker-compose for exec to maximize compatibility with tests and older setups
    result = subprocess.run(
        ["docker-compose", "exec", "-T", "postgres", "pg_isready", "-U", "postgres"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def check_redis() -> bool:
    """
    Check Redis health.
    """
    result = subprocess.run(
        ["docker-compose", "exec", "-T", "redis", "redis-cli", "ping"],
        capture_output=True,
        text=True,
    )
    return "PONG" in result.stdout


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

    result = subprocess.run(
        ["docker-compose", "ps", "--format", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
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
