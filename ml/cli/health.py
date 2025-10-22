#!/usr/bin/env python3
"""
Thin wrapper delegating ML integration health aggregation to tasks.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ml.core.integration import HealthSummary
from ml.tasks.monitoring import aggregate_integration_health


__all__ = ["main"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary: HealthSummary = aggregate_integration_health(
        args.db_connection,
        strict_protocol_validation=args.strict,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
