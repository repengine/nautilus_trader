#!/usr/bin/env python3
"""
CLI entrypoint for pipeline health checks.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid as _uuid
from collections.abc import Sequence

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.monitoring.health import HAS_PSYCOPG2
from ml.monitoring.health import HealthStatus
from ml.monitoring.health import format_human_output
from ml.monitoring.health import format_json_output
from ml.monitoring.health import run_pipeline_health_checks


__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check ML pipeline health")
    parser.add_argument(
        "--connection-string",
        default=os.environ.get(
            "ML_DB_CONNECTION",
            "postgresql://postgres:postgres@localhost:5432/nautilus",
        ),
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--critical-only",
        action="store_true",
        help="Show only critical issues",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run pipeline health checks and render report output.

    Args:
        argv: Optional CLI argument vector (excluding executable).

    Returns:
        Exit code (0=healthy, 1=warning, 2=critical/failure).
    """
    configure_logging()
    run_id = f"cli_check_pipeline_health_{_uuid.uuid4().hex[:8]}"
    bind_log_context(run_id=run_id, component="ml.cli.check_pipeline_health")

    if not HAS_PSYCOPG2:
        print(
            "Error: psycopg2 is required. Install with: pip install psycopg2-binary",
            file=sys.stderr,
        )
        return 2

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        component_health, overall_status, exit_code = run_pipeline_health_checks(
            str(args.connection_string),
        )

        if bool(args.critical_only):
            component_health = {
                key: value
                for key, value in component_health.items()
                if value.status == HealthStatus.CRITICAL
            }
            if not component_health:
                print("No critical issues found")
                return 0

        if bool(args.json):
            output = format_json_output(component_health, overall_status, exit_code)
        else:
            output = format_human_output(component_health, overall_status)

        export_path = args.export
        if isinstance(export_path, str) and export_path:
            with open(export_path, "w", encoding="utf-8") as handle:
                handle.write(output)
            print(f"Results exported to {export_path}")
        else:
            print(output)

        return int(exit_code)
    except Exception as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
