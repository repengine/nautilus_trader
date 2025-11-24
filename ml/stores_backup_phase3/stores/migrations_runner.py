"""
Database migration runner for the Nautilus ML stores schema.

This module discovers SQL files under :mod:`ml.stores.migrations`, executes any
pending migrations against a PostgreSQL database, and records progress in the
``ml_schema_migrations`` table.  The runner is safe to invoke on every service
startup—already applied files are skipped and checksum mismatches raise a
descriptive error so schema drift is caught early.

The module also exposes a thin CLI wrapper::

    poetry run python -m ml.stores.migrations_runner apply --db-url postgresql://...

and a schema health check helper that ensures the critical ``market_data`` table
matches the ingestion requirements (bid/ask columns, generated spread/mid_price,
and the ``(instrument_id, ts_event)`` primary key).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from sqlalchemy import inspect
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.exc import SQLAlchemyError

from ml.core.db_engine import EngineManager
from ml.tasks.db import split_sql_statements


LOGGER = logging.getLogger(__name__)

EngineFactory = Callable[[str], Engine]

TRACKING_TABLE_NAME = "ml_schema_migrations"
REQUIRED_MEASURE_COLUMNS: tuple[str, ...] = ("bid", "ask", "bid_size", "ask_size")
REQUIRED_GENERATED_COLUMNS: tuple[str, ...] = ("spread", "mid_price")
PRIMARY_KEY_COLUMNS: tuple[str, ...] = ("instrument_id", "ts_event")
DEFAULT_SCHEMA_CANDIDATES: tuple[str | None, ...] = (None, "public")
REQUIRED_INSTRUMENTATION_TABLES: tuple[str, ...] = ("ml_data_events", "ml_data_watermarks")
_TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOW_MIGRATION_DRIFT = frozenset({"1", "true", "yes", "on"})


def _allow_migration_drift() -> bool:
    """
    Return ``True`` when operators permit checksum drift for already applied migrations.
    """
    token = os.getenv("ML_ALLOW_MIGRATION_DRIFT")
    if token is None:
        return False
    return token.strip().lower() in _ALLOW_MIGRATION_DRIFT


class MigrationRunnerError(RuntimeError):
    """Raised when migrations cannot be discovered or applied."""


class SchemaHealthCheckError(RuntimeError):
    """Raised when the schema health check detects a blocking issue."""


@dataclass(slots=True, frozen=True)
class MigrationSummary:
    """
    Summary returned after applying (or planning) migrations.
    """

    planned: tuple[Path, ...]
    applied: tuple[Path, ...]
    already_applied: tuple[Path, ...]
    pending: tuple[Path, ...]
    dry_run: bool

    @property
    def applied_count(self) -> int:
        """Number of migrations executed in this run."""
        return len(self.applied)

    @property
    def already_applied_count(self) -> int:
        """Number of migrations skipped because they were previously recorded."""
        return len(self.already_applied)

    @property
    def pending_count(self) -> int:
        """Number of migrations that still need to run (dry-run preview)."""
        return len(self.pending)


@dataclass(slots=True, frozen=True)
class SchemaHealthReport:
    """
    Result of the market_data schema verification.
    """

    table_name: str
    schema: str | None
    missing_columns: tuple[str, ...]
    missing_generated_columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    dialect: str

    @property
    def healthy(self) -> bool:
        """
        Return ``True`` when the schema satisfies the ingestion contract.
        """
        return (
            not self.missing_columns
            and not self.missing_generated_columns
            and tuple(self.primary_key) == PRIMARY_KEY_COLUMNS
        )

    def describe(self) -> str:
        """
        Human-readable summary suitable for logging/errors.
        """
        parts: list[str] = [
            f"table={self.table_name}",
            f"schema={self.schema or '<default>'}",
            f"primary_key={self.primary_key or '<missing>'}",
            f"dialect={self.dialect}",
        ]
        if self.missing_columns:
            parts.append(f"missing_columns={','.join(self.missing_columns)}")
        if self.missing_generated_columns:
            parts.append(f"missing_generated={','.join(self.missing_generated_columns)}")
        return " ".join(parts)


class MigrationRunner:
    """
    Apply SQL migrations from ``ml/stores/migrations`` with checksum tracking.
    """

    def __init__(
        self,
        *,
        db_url: str,
        migrations_path: str | Path | None = None,
        tracking_table: str = TRACKING_TABLE_NAME,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        """
        Parameters
        ----------
        db_url:
            SQLAlchemy-style database URL.
        migrations_path:
            Directory containing ``*.sql`` files (defaults to ``ml/stores/migrations``).
        tracking_table:
            Table used to persist applied migration checksums.
        engine_factory:
            Optional override used by tests to control engine instantiation.
        """
        self._db_url = db_url
        self._engine_factory = engine_factory or EngineManager.get_engine
        self._engine: Engine | None = None
        self._tracking_table = self._validate_tracking_table(tracking_table)
        base_dir = Path(__file__).resolve().parent
        default_path = base_dir / "migrations"
        self._migrations_path = Path(migrations_path) if migrations_path else default_path
        if not self._migrations_path.exists():
            msg = f"Migrations directory not found: {self._migrations_path}"
            raise MigrationRunnerError(msg)
        if not self._migrations_path.is_dir():
            msg = f"Migrations path must be a directory: {self._migrations_path}"
            raise MigrationRunnerError(msg)

    @property
    def engine(self) -> Engine:
        """
        Return the SQLAlchemy engine used by the runner.
        """
        if self._engine is None:
            self._engine = self._engine_factory(self._db_url)
        return self._engine

    @property
    def tracking_table(self) -> str:
        """
        Name of the table used for migration bookkeeping.
        """
        return self._tracking_table

    @staticmethod
    def _validate_tracking_table(table_name: str) -> str:
        if not _TABLE_NAME_PATTERN.fullmatch(table_name):
            msg = f"Invalid tracking table identifier: {table_name}"
            raise MigrationRunnerError(msg)
        return table_name

    def discover_migration_files(self) -> tuple[Path, ...]:
        """
        Return the ordered list of migration files.
        """
        files = [
            path
            for path in sorted(self._migrations_path.glob("*.sql"), key=lambda p: p.name)
            if path.is_file()
        ]
        if not files:
            msg = f"No SQL migrations found under {self._migrations_path}"
            raise MigrationRunnerError(msg)
        return tuple(files)

    def apply_pending_migrations(self, *, dry_run: bool = False) -> MigrationSummary:
        """
        Apply migrations that have not been recorded yet.
        """
        files = self.discover_migration_files()
        engine = self.engine

        with engine.begin() as connection:
            self._ensure_tracking_table(connection)
            applied_map = self._load_applied_migrations(connection)

        pending_files = tuple(path for path in files if path.name not in applied_map)
        already_applied = tuple(path for path in files if path.name in applied_map)

        for path in already_applied:
            recorded_checksum = applied_map.get(path.name)
            current_checksum = self._compute_checksum(path)
            if recorded_checksum is None:
                continue
            if recorded_checksum != current_checksum:
                if _allow_migration_drift():
                    LOGGER.warning(
                        "migrations_runner.checksum_drift file=%s recorded=%s current=%s",
                        path,
                        recorded_checksum,
                        current_checksum,
                    )
                    with engine.begin() as connection:
                        self._record_applied_migration(connection, path.name, current_checksum)
                    continue
                msg = (
                    f"Recorded checksum for {path.name} ({recorded_checksum}) "
                    f"does not match current file ({current_checksum})"
                )
                raise MigrationRunnerError(msg)

        applied_this_run: list[Path] = []

        if dry_run:
            LOGGER.info(
                "migrations_runner.dry_run pending=%d already_applied=%d",
                len(pending_files),
                len(already_applied),
            )
            return MigrationSummary(
                planned=files,
                applied=tuple(),
                already_applied=already_applied,
                pending=pending_files,
                dry_run=True,
            )

        for path in pending_files:
            checksum = self._compute_checksum(path)
            recorded_checksum = applied_map.get(path.name)
            if recorded_checksum is not None and recorded_checksum != checksum:
                if _allow_migration_drift():
                    LOGGER.warning(
                        "migrations_runner.checksum_drift_pending file=%s recorded=%s current=%s",
                        path,
                        recorded_checksum,
                        checksum,
                    )
                else:
                    msg = (
                        f"Checksum mismatch for {path.name}: "
                        f"expected {recorded_checksum}, observed {checksum}"
                    )
                    raise MigrationRunnerError(msg)

            self._apply_file(engine, path)
            applied_this_run.append(path)
            with engine.begin() as connection:
                self._record_applied_migration(connection, path.name, checksum)
            LOGGER.info("migrations_runner.applied file=%s", path)

        summary = MigrationSummary(
            planned=files,
            applied=tuple(applied_this_run),
            already_applied=already_applied,
            pending=tuple(
                path
                for path in files
                if path not in already_applied and path not in applied_this_run
            ),
            dry_run=False,
        )
        LOGGER.info(
            "migrations_runner.summary applied=%d already_applied=%d total=%d",
            summary.applied_count,
            summary.already_applied_count,
            len(files),
        )
        return summary

    def _compute_checksum(self, path: Path) -> str:
        try:
            payload = path.read_bytes()
        except Exception as exc:  # pragma: no cover - filesystem error
            msg = f"Unable to read migration file {path}"
            raise MigrationRunnerError(msg) from exc
        return hashlib.sha256(payload).hexdigest()

    def _ensure_tracking_table(self, connection: Any) -> None:
        ddl = text(
            f"""
            CREATE TABLE IF NOT EXISTS {self._tracking_table} (
                filename TEXT PRIMARY KEY,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        connection.execute(ddl)

    def _load_applied_migrations(self, connection: Any) -> dict[str, str]:
        query = text(f"SELECT filename, checksum FROM {self._tracking_table}")
        result = connection.execute(query)
        return {row[0]: row[1] for row in result}

    def _record_applied_migration(self, connection: Any, filename: str, checksum: str) -> None:
        statement = text(
            f"""
            INSERT INTO {self._tracking_table} (filename, checksum)
            VALUES (:filename, :checksum)
            ON CONFLICT (filename) DO UPDATE SET checksum = EXCLUDED.checksum
            """,
        )
        connection.execute(statement, {"filename": filename, "checksum": checksum})

    def _apply_file(self, engine: Engine, path: Path) -> None:
        try:
            sql_text = path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - filesystem error
            msg = f"Unable to read migration file {path}"
            raise MigrationRunnerError(msg) from exc

        statements = tuple(split_sql_statements(sql_text))
        if not statements:
            LOGGER.debug("Skipping empty migration file %s", path)
            return

        try:
            with engine.begin() as connection:
                for statement in statements:
                    connection.execute(text(statement))
        except SQLAlchemyError as exc:
            msg = f"Failed to apply migration {path.name}"
            raise MigrationRunnerError(msg) from exc


def is_postgres_url(url: str) -> bool:
    """
    Return ``True`` when ``url`` points to a PostgreSQL backend.
    """
    try:
        parsed = make_url(url)
    except Exception:
        return False
    backend = parsed.get_backend_name()
    if backend is None:
        return False
    return backend.startswith("postgres")


def verify_market_data_schema(
    engine: Engine,
    *,
    table_name: str = "market_data",
) -> SchemaHealthReport:
    """
    Ensure the ``market_data`` table contains required columns and primary key.
    """
    inspector = inspect(engine)
    schema, columns = _resolve_table_columns(inspector, table_name)
    column_map = {column["name"]: column for column in columns}

    missing_columns = tuple(col for col in REQUIRED_MEASURE_COLUMNS if col not in column_map)

    generated_from_db: set[str] | None = None
    missing_generated: list[str] = []
    for column in REQUIRED_GENERATED_COLUMNS:
        metadata = column_map.get(column)
        if metadata is None:
            missing_generated.append(column)
            continue
        if _column_is_generated(metadata):
            continue
        if engine.dialect.name == "postgresql":
            if generated_from_db is None:
                generated_from_db = _query_generated_columns(
                    engine,
                    schema or inspector.default_schema_name or "public",
                    table_name,
                )
            if column in generated_from_db:
                continue
        missing_generated.append(column)

    pk_info = inspector.get_pk_constraint(table_name, schema=schema)
    primary_key = tuple(pk_info.get("constrained_columns") or ())

    report = SchemaHealthReport(
        table_name=table_name,
        schema=schema,
        missing_columns=missing_columns,
        missing_generated_columns=tuple(missing_generated),
        primary_key=primary_key,
        dialect=engine.dialect.name,
    )
    if not report.healthy:
        raise SchemaHealthCheckError(f"market_data schema invalid: {report.describe()}")
    return report


def verify_instrumentation_tables(
    engine: Engine,
    *,
    required_tables: Sequence[str] | None = None,
    schema: str | None = None,
) -> None:
    """
    Ensure instrumentation tables exist before the scheduler starts emitting events.
    """
    inspector = inspect(engine)
    tables = tuple(required_tables or REQUIRED_INSTRUMENTATION_TABLES)

    schema_candidates: list[str | None] = []
    if schema is not None:
        schema_candidates.append(schema)
    env_schema = os.getenv("ML_DB_SCHEMA")
    if env_schema:
        schema_candidates.append(env_schema)
    schema_candidates.extend(DEFAULT_SCHEMA_CANDIDATES)
    default_schema = cast(str | None, getattr(inspector, "default_schema_name", None))
    if default_schema:
        schema_candidates.append(default_schema)
    schema_candidates.append(None)

    ordered_schemas: list[str | None] = []
    seen: set[str | None] = set()
    for candidate in schema_candidates:
        if candidate in seen:
            continue
        ordered_schemas.append(candidate)
        seen.add(candidate)

    missing: list[str] = []
    for table in tables:
        if any(inspector.has_table(table, schema=candidate) for candidate in ordered_schemas):
            continue
        missing.append(table)

    if missing:
        formatted_schemas = ", ".join(schema or "<default>" for schema in ordered_schemas)
        msg = (
            "Instrumentation tables missing. Ensure ml/stores/migrations_runner "
            "has been executed before starting ingestion. Missing tables: "
            f"{', '.join(missing)}. Checked schemas: {formatted_schemas}"
        )
        raise SchemaHealthCheckError(msg)


def _resolve_table_columns(
    inspector: Any,
    table_name: str,
) -> tuple[str | None, list[Mapping[str, Any]]]:
    candidates: list[str | None] = []
    env_schema = os.getenv("ML_DB_SCHEMA")
    if env_schema:
        candidates.append(env_schema)
    candidates.extend(candidate for candidate in DEFAULT_SCHEMA_CANDIDATES if candidate is not None)
    candidates.append(cast(str | None, getattr(inspector, "default_schema_name", None)))
    # Deduplicate while preserving order.
    ordered_candidates: list[str | None] = []
    seen: set[str | None] = set()
    for candidate in (None, *candidates):
        if candidate in seen:
            continue
        ordered_candidates.append(candidate)
        seen.add(candidate)

    last_error: Exception | None = None
    for schema in ordered_candidates:
        try:
            columns = inspector.get_columns(table_name, schema=schema)
        except NoSuchTableError as exc:
            last_error = exc
            continue
        if columns:
            return schema, columns
        # Table exists but no columns (unlikely)—treat as found.
        return schema, columns

    msg = f"Table '{table_name}' not found in schemas {ordered_candidates}"
    raise SchemaHealthCheckError(msg) from last_error


def _column_is_generated(column: Mapping[str, Any]) -> bool:
    computed = column.get("computed")
    if isinstance(computed, Mapping):
        sqltext = computed.get("sqltext")
        persisted = computed.get("persisted")
        if sqltext:
            if persisted is None:
                return True
            return bool(persisted)
    dialect_options = column.get("dialect_options")
    if isinstance(dialect_options, Mapping):
        if dialect_options.get("postgresql_generated"):
            return True
    return False


def _query_generated_columns(engine: Engine, schema: str, table_name: str) -> set[str]:
    statement = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
          AND is_generated = 'ALWAYS'
        """,
    )
    with engine.connect() as connection:
        rows = connection.execute(statement, {"schema": schema, "table": table_name})
        return {row[0] for row in rows}


def _default_db_url() -> str | None:
    for key in ("DB_CONNECTION", "DATABASE_URL", "NAUTILUS_DB"):
        value = os.getenv(key)
        if value:
            return value
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply Nautilus ML store migrations")
    parser.add_argument(
        "command",
        choices=("apply", "plan"),
        nargs="?",
        default="apply",
        help="Action to perform (default: apply)",
    )
    parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        help="PostgreSQL connection string (falls back to DB_CONNECTION/DATABASE_URL)",
    )
    parser.add_argument(
        "--path",
        dest="path",
        default=None,
        help="Custom migrations directory (defaults to ml/stores/migrations)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pending migrations without executing them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entry point for manual migration management.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    db_url = args.db_url or _default_db_url()
    if not db_url:
        parser.error("Set --db-url or export DB_CONNECTION/DATABASE_URL")

    runner = MigrationRunner(db_url=db_url, migrations_path=args.path)

    if args.command == "plan" or args.dry_run:
        summary = runner.apply_pending_migrations(dry_run=True)
        print("Pending migrations:")
        for path in summary.pending:
            print(f" - {path}")
        print(
            f"{summary.pending_count} pending, "
            f"{summary.already_applied_count} already applied",
        )
        return 0

    summary = runner.apply_pending_migrations(dry_run=False)
    print(
        f"Applied {summary.applied_count} migration(s); "
        f"{summary.already_applied_count} already recorded.",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())


__all__ = [
    "MigrationRunner",
    "MigrationRunnerError",
    "MigrationSummary",
    "SchemaHealthCheckError",
    "SchemaHealthReport",
    "is_postgres_url",
    "main",
    "verify_instrumentation_tables",
    "verify_market_data_schema",
]
