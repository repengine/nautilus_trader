#!/usr/bin/env python3
"""
Database test helpers.

These utilities centralize how tests derive the Postgres connection port so
test suites consistently honor the `TEST_DB_PORT` environment variable.
"""

from __future__ import annotations

import os

DEFAULT_TEST_DB_PORT = os.getenv("TEST_DB_PORT", "5434")


def get_test_db_port() -> str:
    """Return the Postgres port to use for tests.

    Returns:
        The port defined by `TEST_DB_PORT`, falling back to the default test
        port when the environment variable is not set.
    """
    return os.getenv("TEST_DB_PORT", DEFAULT_TEST_DB_PORT)


def build_postgres_url(
    *,
    user: str = "postgres",
    password: str | None = None,
    host: str = "localhost",
    database: str = "nautilus",
    port: str | None = None,
) -> str:
    """Compose a Postgres connection string using the test port by default.

    Args:
        user: Database user.
        password: Database password.
        host: Database host.
        database: Database name.
        port: Optional port override; defaults to the test port when omitted.

    Returns:
        A Postgres connection string targeting the configured test port.
    """
    resolved_port = port or get_test_db_port()
    resolved_password = password or os.getenv("TEST_DB_PASSWORD", "postgres")
    return f"postgresql://{user}:{resolved_password}@{host}:{resolved_port}/{database}"


__all__ = ["DEFAULT_TEST_DB_PORT", "build_postgres_url", "get_test_db_port"]
