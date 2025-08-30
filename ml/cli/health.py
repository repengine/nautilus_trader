#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

from ml.core.integration import MLIntegrationManager


def main() -> None:
    parser = argparse.ArgumentParser(description="ML system health summary")
    parser.add_argument(
        "--db-connection",
        dest="db_connection",
        type=str,
        default=None,
        help="PostgreSQL connection string (optional)",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Raise on protocol validation failures",
    )
    args = parser.parse_args()

    mgr = MLIntegrationManager(
        db_connection=args.db_connection,
        auto_start_postgres=False,
        auto_migrate=False,
        ensure_healthy=False,
        strict_protocol_validation=args.strict,
    )

    summary: dict[str, Any] = mgr.aggregate_health()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

