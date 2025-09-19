"""
Run the Dashboard API server (cold path).

This module provides a simple entrypoint to run the dashboard API using Flask's
built-in server (sufficient for local/dev; use a WSGI server for prod as needed).
"""

from __future__ import annotations

import os
from typing import NoReturn

from ml.dashboard import DashboardConfig
from ml.dashboard import create_app


def main() -> NoReturn:  # pragma: no cover - process runner
    cfg = DashboardConfig.from_env()
    app = create_app(cfg)
    # Default to loopback; container compose sets 0.0.0.0 explicitly
    host = os.getenv("ML_DASHBOARD_HOST", "127.0.0.1")
    port_str = os.getenv("ML_DASHBOARD_PORT", "8010")
    try:
        port = int(port_str)
    except Exception:
        port = 8010
    app.run(host=host, port=port)
    raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover - process runner
    main()
