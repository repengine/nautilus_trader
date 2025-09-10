#!/usr/bin/env python3
"""
Apply ML database migrations to a PostgreSQL instance.

This lightweight runner executes the canonical SQL migrations for the ML system
in a safe, idempotent manner with a concise summary.

Usage examples:
  uv run --active --no-sync python -m ml.scripts.apply_migrations --db-url postgresql://... --full
  uv run --active --no-sync python -m ml.scripts.apply_migrations --print-only

Environment:
  DATABASE_URL may be used instead of --db-url

"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager


# Canonical baseline list mirrors MLIntegrationManager._run_migrations
BASE_MIGRATIONS: list[str] = [
    "ml/registry/migrations/001_initial_schema.sql",
    "ml/stores/migrations/001_stores_schema.sql",
    "ml/stores/migrations/002_auto_partitioning.sql",
    "ml/stores/migrations/003_market_data.sql",
    "ml/stores/migrations/004_data_registry.sql",
    "ml/stores/migrations/007_add_event_metadata.sql",
]

# Optional extras (applied when --full or --include-optional is set)
OPTIONAL_MIGRATIONS: list[str] = [
    # Hardening & views
    "ml/stores/migrations/005_schema_hardening.sql",
    "ml/stores/migrations/005_views.sql",
    # Test-time optimizations/indices
    "ml/stores/migrations/006_disable_partition_triggers.sql",
    "ml/stores/migrations/007_brin_indexes.sql",
    # Registry extension
    "ml/registry/migrations/002_add_cold_path_fields.sql",
    # Emergency fixes
    "ml/migrations/999_fix_partitions_immediate.sql",
]


@dataclass
class Result:
    applied: int = 0
    skipped: int = 0
    warnings: int = 0
    errors: int = 0
    files_applied: list[str] | None = None
    files_skipped: list[str] | None = None


def _read_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _split_statements(sql: str) -> Iterable[str]:
    """
    Split SQL into executable statements, respecting dollar-quoted and string blocks.

    Handles $$...$$ and $tag$...$tag$ function bodies and single-quoted strings.

    """
    stmts: list[str] = []
    buf: list[str] = []
    in_single = False
    in_dollar = False
    dollar_tag = ""  # e.g., '', or 'tag'
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # Detect start/end of dollar-quoted blocks: $tag$
        if not in_single:
            if ch == "$":
                # Read tag
                j = i + 1
                tag = []
                while (j < n and sql[j].isalnum()) or (j < n and sql[j] == "_"):
                    tag.append(sql[j])
                    j += 1
                if j < n and sql[j] == "$":
                    token = "".join(tag)
                    if not in_dollar:
                        in_dollar = True
                        dollar_tag = token
                    else:
                        # Closing tag only if matches
                        if token == dollar_tag:
                            in_dollar = False
                            dollar_tag = ""
                    # Append the full delimiter
                    buf.append(sql[i : j + 1])
                    i = j + 1
                    continue

        # Toggle single-quoted string state (handle escaping by doubling '')
        if not in_dollar and ch == "'":
            if in_single and nxt == "'":
                # Escaped quote inside string
                buf.append("''")
                i += 2
                continue
            in_single = not in_single

        # Statement delimiter when not inside quotes/blocks
        if ch == ";" and not in_single and not in_dollar:
            stmt = "".join(buf).strip()
            if stmt:
                stmts.append(stmt)
            buf.clear()
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts


def apply_files(engine: Engine, files: list[Path], *, dry_run: bool = False) -> Result:
    res = Result(files_applied=[], files_skipped=[])
    for file in files:
        if not file.exists():
            res.skipped += 1
            res.files_skipped.append(str(file))
            continue
        if dry_run:
            res.applied += 1
            res.files_applied.append(str(file))
            continue
        try:
            sql = _read_sql(file)
            with engine.begin() as conn:
                for stmt in _split_statements(sql):
                    try:
                        conn.execute(text(stmt))
                    except Exception as e:  # tolerate idempotent conflicts
                        msg = str(e).lower()
                        if (
                            "already exists" in msg
                            or "does not exist" in msg
                            or "duplicate key" in msg
                        ):
                            res.warnings += 1
                        else:
                            # Record error but continue to next file
                            res.errors += 1
            res.applied += 1
            res.files_applied.append(str(file))
        except Exception:
            res.errors += 1
    return res


def build_plan(full: bool, schema: str) -> list[Path]:
    # Filter by schema selection
    selected: list[str]
    if schema == "stores":
        selected = [p for p in BASE_MIGRATIONS if "/stores/" in p]
    elif schema == "registry":
        selected = [p for p in BASE_MIGRATIONS if "/registry/" in p]
    else:
        selected = list(BASE_MIGRATIONS)

    optionals: list[str] = []
    if full:
        if schema == "stores":
            optionals = [p for p in OPTIONAL_MIGRATIONS if "/stores/" in p]
        elif schema == "registry":
            optionals = [p for p in OPTIONAL_MIGRATIONS if "/registry/" in p]
        else:
            optionals = list(OPTIONAL_MIGRATIONS)

    return [Path(p) for p in [*selected, *optionals]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply ML database migrations")
    parser.add_argument("--db-url", dest="db_url", default=None, help="PostgreSQL connection URL")
    parser.add_argument("--schema", choices=["stores", "registry", "both"], default="both")
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

    # Resolve DB URL
    import os

    db_url = (
        args.db_url
        or os.getenv("DATABASE_URL")
        or "postgresql://postgres:postgres@localhost:5432/nautilus"
    )

    plan = build_plan(args.full, args.schema)
    if args.print_only:
        print("Migration plan:")
        for p in plan:
            print(" -", p)
        return 0

    engine = EngineManager.get_engine(db_url)
    result = apply_files(engine, plan, dry_run=args.dry_run)

    # Summary
    print("\nMigration Summary")
    print("=================")
    print(f"Applied: {result.applied}")
    print(f"Skipped: {result.skipped}")
    print(f"Warnings: {result.warnings}")
    print(f"Errors: {result.errors}")
    if result.files_applied:
        print("\nFiles Applied:")
        for f in result.files_applied:
            print(" -", f)
    if result.files_skipped:
        print("\nFiles Skipped (missing):")
        for f in result.files_skipped:
            print(" -", f)

    return 0 if result.errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
