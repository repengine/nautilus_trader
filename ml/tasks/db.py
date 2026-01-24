"""
Database-oriented cold-path tasks reused by ML CLI entry points.

The helpers centralize migration planning and execution while keeping the command-line
wrappers completely declarative. All functions are explicitly typed and safe to use from
tests or other orchestration layers.

"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Final

import structlog
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.engine import RootTransaction
from sqlalchemy.exc import SQLAlchemyError

from ml.core.db_engine import EngineManager


LOGGER = structlog.get_logger(__name__)

IDEMPOTENT_ERROR_PHRASES: Final[tuple[str, ...]] = (
    "already exists",
    "does not exist",
    "duplicate key",
    "is not partitioned",
)

# Canonical baseline list (consolidated 2025-10-01 from 18 fragmented migrations)
_BASE_MIGRATIONS: Final[tuple[str, ...]] = (
    "ml/registry/migrations/001_initial_schema.sql",
    "ml/registry/migrations/002_add_cold_path_fields.sql",
    "ml/registry/migrations/003_add_artifact_digest.sql",
    "ml/stores/migrations_bootstrap/001_bootstrap.sql",
)

# Optional extras (no longer needed - consolidated into bootstrap)
_OPTIONAL_MIGRATIONS: Final[tuple[str, ...]] = ()

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_migration_path(path: str) -> Path:
    """
    Resolve migration paths relative to the repository root when needed.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    resolved = _REPO_ROOT / path
    if resolved.exists():
        return resolved
    return candidate


class MigrationSchema(str, Enum):
    """
    Schema selection for migration plans.
    """

    STORES = "stores"
    REGISTRY = "registry"
    BOTH = "both"

    def allows(self, migration_path: str) -> bool:
        """
        Return ``True`` when ``migration_path`` is included for this schema.
        """
        if self is MigrationSchema.BOTH:
            return True
        if self is MigrationSchema.STORES:
            return "/stores/" in migration_path
        if self is MigrationSchema.REGISTRY:
            return "/registry/" in migration_path
        return False


@dataclass(slots=True, frozen=True)
class MigrationPlan:
    """
    Concrete plan describing which SQL files will be executed.
    """

    files: tuple[Path, ...]


@dataclass(slots=True)
class MigrationResult:
    """
    Outcome details returned after executing a migration plan.
    """

    applied: int = 0
    skipped: int = 0
    warnings: int = 0
    errors: int = 0
    files_applied: list[Path] = field(default_factory=list)
    files_skipped: list[Path] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """
        Return ``True`` when no errors were recorded.
        """
        return self.errors == 0


# ---------------------------------------------------------------------------
# Planning helpers
# ---------------------------------------------------------------------------


def build_migration_plan(
    *,
    include_optional: bool,
    schema: MigrationSchema,
    base: Sequence[str] | None = None,
    optional: Sequence[str] | None = None,
) -> MigrationPlan:
    """
    Construct a migration plan filtered by ``schema``.

    Parameters
    ----------
    include_optional:
        When ``True`` the optional migration list is appended to the plan.
    schema:
        Which schema subset should be included.
    base:
        Override for the canonical baseline migration list (used in tests).
    optional:
        Override for the optional migration list (used in tests).

    """
    base_paths = base or _BASE_MIGRATIONS
    optional_paths = optional or _OPTIONAL_MIGRATIONS

    ordered: list[Path] = []
    for candidate in base_paths:
        if not schema.allows(candidate):
            continue
        ordered.append(_resolve_migration_path(candidate))

    if include_optional:
        for candidate in optional_paths:
            if not schema.allows(candidate):
                continue
            ordered.append(_resolve_migration_path(candidate))

    plan = MigrationPlan(files=tuple(ordered))
    LOGGER.debug("Built migration plan", schema=schema.value, count=len(plan.files))
    return plan


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def split_sql_statements(sql: str) -> Iterable[str]:
    """
    Yield SQL statements while respecting dollar-quoted/string literals.
    """
    statements: list[str] = []
    buffer: list[str] = []
    in_single = False
    in_dollar = False
    in_line_comment = False
    dollar_tag = ""
    i = 0
    length = len(sql)

    while i < length:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < length else ""

        if not in_line_comment and not in_single and ch == "$":
            j = i + 1
            tag: list[str] = []
            while (j < length and sql[j].isalnum()) or (j < length and sql[j] == "_"):
                tag.append(sql[j])
                j += 1
            if j < length and sql[j] == "$":
                token = "".join(tag)
                if not in_dollar:
                    in_dollar = True
                    dollar_tag = token
                elif token == dollar_tag:
                    in_dollar = False
                    dollar_tag = ""
                buffer.append(sql[i : j + 1])
                i = j + 1
                continue

        if not in_line_comment and not in_dollar and ch == "'":
            if in_single and nxt == "'":
                buffer.append("''")
                i += 2
                continue
            in_single = not in_single

        if not in_single and not in_dollar:
            if not in_line_comment and ch == "-" and nxt == "-":
                in_line_comment = True
                i += 2
                continue
            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
                    buffer.append(ch)
                i += 1
                continue

        if ch == ";" and not in_single and not in_dollar and not in_line_comment:
            stmt = "".join(buffer).strip()
            if stmt and _has_meaningful_sql(stmt):
                statements.append(stmt)
            buffer.clear()
            i += 1
            continue

        buffer.append(ch)
        i += 1

    tail = "".join(buffer).strip()
    if tail and _has_meaningful_sql(tail):
        statements.append(tail)

    return statements


def _has_meaningful_sql(statement: str) -> bool:
    """Return True if ``statement`` contains executable SQL once comments are ignored."""
    # Drop full-line comments (``--``) and whitespace before checking emptiness.
    non_comment_lines = []
    for line in statement.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("--") or not stripped:
            continue
        non_comment_lines.append(line)

    if not "".join(non_comment_lines).strip():
        return False

    # Remove simple block comments in a lightweight scan so comment-only
    # buffers such as ``/* doc */;`` are ignored. This deliberately avoids
    # string parsing complexity because we only care about all-comment chunks.
    cleaned: list[str] = []
    in_block = False
    i = 0
    length = len(statement)
    while i < length:
        ch = statement[i]
        nxt = statement[i + 1] if i + 1 < length else ""

        if not in_block and ch == "/" and nxt == "*":
            in_block = True
            i += 2
            continue

        if in_block and ch == "*" and nxt == "/":
            in_block = False
            i += 2
            continue

        if not in_block:
            cleaned.append(ch)
        i += 1

    return "".join(cleaned).strip() != ""


def apply_migration_files(
    engine: Engine,
    plan: MigrationPlan,
    *,
    dry_run: bool = False,
) -> MigrationResult:
    """
    Execute a migration plan using the provided SQLAlchemy ``engine``.
    """
    result = MigrationResult()
    for path in plan.files:
        if not path.exists():
            LOGGER.warning("Migration file missing", file=str(path))
            result.skipped += 1
            result.files_skipped.append(path)
            continue

        if dry_run:
            result.applied += 1
            result.files_applied.append(path)
            LOGGER.info("Dry-run migration", file=str(path))
            continue

        try:
            sql_text = path.read_text(encoding="utf-8")
        except Exception as exc:
            LOGGER.error("Unable to read migration file", file=str(path), exc_info=exc)
            result.errors += 1
            continue

        statements = tuple(split_sql_statements(sql_text))
        if not statements:
            LOGGER.debug("Skipping empty migration file %s", path)
            continue

        try:
            with engine.connect() as connection:
                transaction: RootTransaction | None = connection.begin()
                try:
                    for statement in statements:
                        try:
                            if _requires_dedicated_transaction(statement):
                                if transaction is None:
                                    transaction = connection.begin()
                                transaction.commit()
                                transaction = None
                                _execute_autocommit_statement(engine, statement)
                                transaction = connection.begin()
                            else:
                                connection.execute(text(statement))
                        except SQLAlchemyError as exc:
                            message = str(exc).lower()
                            if any(phrase in message for phrase in IDEMPOTENT_ERROR_PHRASES):
                                LOGGER.warning(
                                    "Idempotent migration warning",
                                    file=str(path),
                                    statement_preview=statement[:80],
                                    exc_info=exc,
                                )
                                result.warnings += 1
                            else:
                                raise
                except SQLAlchemyError as exc:
                    if transaction is not None and transaction.is_active:
                        transaction.rollback()
                    LOGGER.error(
                        "Migration statement failed",
                        file=str(path),
                        exc_info=exc,
                    )
                    result.errors += 1
                    continue
                else:
                    if transaction is not None and transaction.is_active:
                        transaction.commit()
        except SQLAlchemyError as exc:
            LOGGER.error("Migration file execution failed", file=str(path), exc_info=exc)
            result.errors += 1
            continue

        result.applied += 1
        result.files_applied.append(path)
        LOGGER.info("Applied migration file", file=str(path))

    return result


def _requires_dedicated_transaction(statement: str) -> bool:
    stripped = statement.strip().lower()
    return (
        stripped.startswith("select create_monthly_partitions(")
        or stripped.startswith("select create_event_partitions(")
    )


def _execute_autocommit_statement(engine: Engine, statement: str) -> None:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(text(statement))


# ---------------------------------------------------------------------------
# High-level orchestration helpers
# ---------------------------------------------------------------------------


def apply_database_migrations(
    db_url: str,
    *,
    include_optional: bool,
    schema: MigrationSchema,
    dry_run: bool = False,
) -> MigrationResult:
    """
    Plan and apply database migrations using ``EngineManager``.
    """
    plan = build_migration_plan(include_optional=include_optional, schema=schema)
    engine = EngineManager.get_engine(db_url)
    LOGGER.info(
        "Applying migrations",
        db_url=db_url,
        dry_run=dry_run,
        schema=schema.value,
        optional=include_optional,
        files=len(plan.files),
    )
    return apply_migration_files(engine, plan, dry_run=dry_run)


__all__ = [
    "MigrationPlan",
    "MigrationResult",
    "MigrationSchema",
    "apply_database_migrations",
    "apply_migration_files",
    "build_migration_plan",
    "split_sql_statements",
]
