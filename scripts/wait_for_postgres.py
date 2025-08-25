#!/usr/bin/env python3
"""
Wait for PostgreSQL to become available.

Reads DATABASE_URL from environment or uses a sensible default for local tests.
Exits with code 0 when a connection succeeds, non-zero on timeout.
"""

from __future__ import annotations

import os
import sys
import time
from typing import NoReturn

import psycopg2  # type: ignore[import-not-found]


def main() -> NoReturn:
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/nautilus",
    )
    timeout = int(os.environ.get("DB_WAIT_TIMEOUT", "60"))
    interval = float(os.environ.get("DB_WAIT_INTERVAL", "1.0"))

    start = time.time()
    last_err: Exception | None = None
    while (time.time() - start) < timeout:
        try:
            conn = psycopg2.connect(url)
            conn.close()
            print("PostgreSQL is ready.")
            sys.exit(0)
        except Exception as e:  # pragma: no cover - best effort wait loop
            last_err = e
            time.sleep(interval)

    print(f"Timed out waiting for PostgreSQL after {timeout}s: {last_err}")
    sys.exit(1)


if __name__ == "__main__":
    main()

