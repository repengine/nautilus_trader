#!/usr/bin/env python3
"""
CLI wrapper for ML database migrations.

Delegates planning/execution to canonical store migration modules so this module remains a
lightweight entry point with predictable typing.

"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import structlog

from ml.common.db_connections import ConnectionRole
from ml.common.db_connections import collect_postgres_candidates
from ml.common.db_connections import select_first_working_connection
from ml.stores.common.sql_splitter import split_sql_statements as _split_statements
from ml.stores.migrations_runner import MigrationResult
from ml.stores.migrations_runner import MigrationSchema
from ml.stores.migrations_runner import apply_database_migrations
from ml.stores.migrations_runner import apply_migration_files as apply_files
from ml.stores.migrations_runner import build_migration_plan


# Backwards compatibility for tests importing from the CLI module
build_plan = build_migration_plan
split_statements = _split_statements


__all__ = ["apply_files", "build_plan", "main", "split_statements"]


LOGGER = structlog.get_logger(__name__)


def _parse_schema(value: str) -> MigrationSchema:
    try:
        return MigrationSchema(value)
    except ValueError as exc:  # pragma: no cover - argparse guards
        msg = f"Unsupported schema selection: {value}"
        raise argparse.ArgumentTypeError(msg) from exc


def _print_plan(plan_files: Sequence[Path]) -> None:
    print("Migration plan:")
    for file in plan_files:
        print(f" - {file}")


def _print_summary(result: MigrationResult) -> None:
    print("\nMigration Summary")
    print("=================")
    print(f"Applied: {result.applied}")
    print(f"Skipped: {result.skipped}")
    print(f"Warnings: {result.warnings}")
    print(f"Errors: {result.errors}")
    if result.files_applied:
        print("\nFiles Applied:")
        for file in result.files_applied:
            print(f" - {file}")
    if result.files_skipped:
        print("\nFiles Skipped (missing):")
        for file in result.files_skipped:
            print(f" - {file}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply ML database migrations")
    parser.add_argument("--db-url", dest="db_url", default=None, help="PostgreSQL connection URL")
    parser.add_argument("--schema", type=_parse_schema, default=MigrationSchema.BOTH)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include optional migrations (hardening, views, fixes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be applied without executing",
    )
    parser.add_argument("--print-only", action="store_true", help="Print migration plan and exit")
    args = parser.parse_args(argv)

    candidates = collect_postgres_candidates(
        ConnectionRole.MIGRATION,
        explicit=args.db_url,
    )
    if not candidates.urls:
        raise SystemExit("No PostgreSQL connection candidates found. Set --db-url or DATABASE_URL.")

    try:
        db_url = select_first_working_connection(candidates.urls)
    except RuntimeError as exc:
        db_url = candidates.urls[0]
        LOGGER.warning(
            "PostgreSQL connectivity probe failed; using first candidate",
            connection=db_url,
            error=str(exc),
        )

    plan = build_migration_plan(include_optional=args.full, schema=args.schema)
    if args.print_only:
        _print_plan(plan.files)
        return 0

    result = apply_database_migrations(
        db_url,
        include_optional=args.full,
        schema=args.schema,
        dry_run=args.dry_run,
    )
    _print_summary(result)
    return 0 if result.succeeded else 2


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
