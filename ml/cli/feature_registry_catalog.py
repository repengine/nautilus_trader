#!/usr/bin/env python3
"""
CLI for summarising FeatureRegistry manifests.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from ml.registry.feature_registry import FeatureRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.tools.feature_catalog import build_feature_catalog


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a summary report for registered feature manifests.",
    )
    parser.add_argument(
        "registry_dir",
        help="Directory for registry artifacts (JSON backend) and manifest attachments.",
    )
    parser.add_argument(
        "--postgres-dsn",
        default=None,
        help="Optional PostgreSQL DSN. When provided, the report is generated from the database backend.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Defaults to human-readable text.",
    )
    parser.add_argument(
        "--fail-on-inconsistency",
        action="store_true",
        help="Return a non-zero exit code if any capability mismatches are detected.",
    )
    return parser


def _load_registry(registry_dir: Path, postgres_dsn: str | None) -> FeatureRegistry:
    if postgres_dsn:
        config = PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string=postgres_dsn,
        )
        return FeatureRegistry(registry_dir, persistence_config=config)
    return FeatureRegistry(registry_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    registry_dir = Path(args.registry_dir).expanduser().resolve()
    try:
        registry = _load_registry(registry_dir, args.postgres_dsn)
        report = build_feature_catalog(registry)
    except Exception as exc:  # pragma: no cover - CLI failure path
        logger.error("feature_registry_catalog_failed", exc_info=True, extra={"registry_dir": str(registry_dir)})
        print(f"Failed to build catalog: {exc}")
        return 1

    if args.format == "json":
        print(report.to_json())
    else:
        print(report.render_text())

    if args.fail_on_inconsistency and report.has_inconsistencies:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
