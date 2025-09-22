#!/usr/bin/env python3
"""Bootstrap the Nautilus Trader ML dashboard stack and display a welcome screen."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ml.dashboard_bootstrap import DEFAULT_COMPOSE_FILE
from ml.dashboard_bootstrap import DEFAULT_HEALTH_CHECKS
from ml.dashboard_bootstrap import DEFAULT_SERVICES
from ml.dashboard_bootstrap import DashboardBootstrapError
from ml.dashboard_bootstrap import HealthCheck
from ml.dashboard_bootstrap import build_welcome_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the ML dashboard stack")
    parser.add_argument(
        "--compose-file",
        default=DEFAULT_COMPOSE_FILE,
        help="docker compose file to use",
    )
    parser.add_argument(
        "--service",
        action="append",
        dest="services",
        help="explicit service list (repeatable)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="health probe timeout (seconds)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="health probe retry attempts",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=2.0,
        help="seconds between health retries",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="skip docker compose up and only display status",
    )
    parser.add_argument(
        "--checks",
        dest="checks",
        action="append",
        help="custom health check in the form name=url (repeatable)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def parse_checks(overrides: list[str] | None) -> tuple[HealthCheck, ...]:
    if not overrides:
        return DEFAULT_HEALTH_CHECKS
    custom: list[HealthCheck] = []
    for raw in overrides:
        if "=" not in raw:
            raise DashboardBootstrapError(
                f"Invalid check format '{raw}'. Expected name=url",
            )
        name, url = raw.split("=", maxsplit=1)
        custom.append(HealthCheck(name=name.strip(), kind="service", url=url.strip()))
    return tuple(custom)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    services = tuple(args.services) if args.services else DEFAULT_SERVICES
    checks = parse_checks(args.checks)

    try:
        summary = build_welcome_summary(
            compose_file=args.compose_file,
            services=services,
            checks=checks,
            timeout_seconds=args.timeout,
            retries=args.retries,
            retry_interval_seconds=args.retry_interval,
            start=not args.status_only,
        )
    except DashboardBootstrapError as exc:
        print(f"Error: {exc}")
        return 1

    print(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
