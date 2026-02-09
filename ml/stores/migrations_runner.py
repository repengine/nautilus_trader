"""
Database migration runner for the Nautilus ML stores schema.

This module discovers SQL files under the store migrations directories
(``migrations_bootstrap``, ``migrations``, and ``migrations_legacy``), executes
pending migrations against a PostgreSQL database, and records progress in the
``ml_schema_migrations`` table. The runner is safe to invoke on every service
startup—already applied files are skipped and checksum mismatches raise a
descriptive error so schema drift is caught early.

The module also exposes a thin CLI wrapper::

    poetry run python -m ml.stores.migrations_runner apply --db-url postgresql://...

and a schema health check helper that ensures the critical market data tables
match the ingestion requirements (legacy ``market_data`` or per-class tables with
``(instrument_id, ts_event)`` primary keys).
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
from dataclasses import field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Final, cast

from sqlalchemy import inspect
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.engine import RootTransaction
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.exc import SQLAlchemyError

from ml.config.market_data import MarketDataTableConfig
from ml.config.market_data import MarketDataTableProfile
from ml.core.db_engine import EngineManager
from ml.stores.common.sql_splitter import split_sql_statements


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
IDEMPOTENT_ERROR_PHRASES: Final[tuple[str, ...]] = (
    "already exists",
    "does not exist",
    "duplicate key",
    "is not partitioned",
)
_BASE_MIGRATIONS: Final[tuple[str, ...]] = (
    "ml/registry/migrations/001_initial_schema.sql",
    "ml/registry/migrations/002_add_cold_path_fields.sql",
    "ml/registry/migrations/003_add_artifact_digest.sql",
    "ml/stores/migrations_bootstrap/001_bootstrap.sql",
)
_OPTIONAL_MIGRATIONS: Final[tuple[str, ...]] = ()
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _allow_migration_drift() -> bool:
    """
    Return ``True`` when operators permit checksum drift for already applied migrations.
    """
    token = os.getenv("ML_ALLOW_MIGRATION_DRIFT")
    if token is None:
        return False
    return token.strip().lower() in _ALLOW_MIGRATION_DRIFT


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


class MigrationProfile(str, Enum):
    """
    Supported migration profiles for selecting bootstrap or legacy histories.
    """

    AUTO = "auto"
    BOOTSTRAP = "bootstrap"
    LEGACY = "legacy"
    INCREMENTAL = "incremental"

    @classmethod
    def from_env(cls, token: str | None) -> MigrationProfile:
        if token is None:
            return cls.AUTO
        value = token.strip().lower()
        if not value:
            return cls.AUTO
        for candidate in cls:
            if candidate.value == value:
                return candidate
        msg = f"Unsupported migration profile: {token}"
        raise MigrationRunnerError(msg)


@dataclass(slots=True, frozen=True)
class ProfiledMigrationSummary:
    """
    Aggregated migration summary across bootstrap/incremental/legacy runs.
    """

    profile: MigrationProfile
    bootstrap: MigrationSummary | None
    incremental: MigrationSummary | None
    legacy: MigrationSummary | None

    def _summaries(self) -> tuple[MigrationSummary, ...]:
        summaries = [summary for summary in (self.bootstrap, self.incremental, self.legacy) if summary]
        return tuple(summaries)

    @property
    def applied_count(self) -> int:
        return sum(summary.applied_count for summary in self._summaries())

    @property
    def already_applied_count(self) -> int:
        return sum(summary.already_applied_count for summary in self._summaries())

    @property
    def pending_count(self) -> int:
        return sum(summary.pending_count for summary in self._summaries())

    @property
    def planned(self) -> tuple[Path, ...]:
        return tuple(chain.from_iterable(summary.planned for summary in self._summaries()))

    @property
    def pending(self) -> tuple[Path, ...]:
        return tuple(chain.from_iterable(summary.pending for summary in self._summaries()))

    @property
    def dry_run(self) -> bool:
        return any(summary.dry_run for summary in self._summaries())


@dataclass(slots=True, frozen=True)
class SchemaHealthReport:
    """
    Result of a table schema verification.
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


@dataclass(slots=True, frozen=True)
class MarketDataSchemaReport:
    """
    Aggregate health report for market data tables.
    """

    profile: MarketDataTableProfile
    tables: tuple[SchemaHealthReport, ...]

    @property
    def healthy(self) -> bool:
        return all(table.healthy for table in self.tables)

    def describe(self) -> str:
        parts = [f"profile={self.profile.value}"]
        parts.extend(table.describe() for table in self.tables)
        return " | ".join(parts)


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
    LOGGER.debug("Built migration plan", extra={"schema": schema.value, "count": len(plan.files)})
    return plan


def apply_migration_files(
    engine: Engine,
    plan: MigrationPlan,
    *,
    dry_run: bool = False,
) -> MigrationResult:
    """
    Execute a migration plan using the provided SQLAlchemy ``engine``.

    Parameters
    ----------
    engine:
        SQLAlchemy engine used for execution.
    plan:
        Ordered migration plan generated by :func:`build_migration_plan`.
    dry_run:
        When ``True``, report planned file application without executing SQL.

    Returns
    -------
    MigrationResult
        Aggregated migration execution outcome.
    """
    result = MigrationResult()
    for path in plan.files:
        if not path.exists():
            LOGGER.warning("Migration file missing", extra={"file": str(path)})
            result.skipped += 1
            result.files_skipped.append(path)
            continue

        if dry_run:
            result.applied += 1
            result.files_applied.append(path)
            LOGGER.info("Dry-run migration", extra={"file": str(path)})
            continue

        try:
            sql_text = path.read_text(encoding="utf-8")
        except Exception as exc:
            LOGGER.error("Unable to read migration file", extra={"file": str(path)}, exc_info=exc)
            result.errors += 1
            continue

        statements = tuple(split_sql_statements(sql_text))
        if not statements:
            LOGGER.debug("Skipping empty migration file %s", path)
            continue

        try:
            warning_count = _execute_migration_statements(
                engine=engine,
                statements=statements,
                path=path,
                tolerate_idempotent_errors=True,
            )
        except SQLAlchemyError as exc:
            LOGGER.error("Migration file execution failed", extra={"file": str(path)}, exc_info=exc)
            result.errors += 1
            continue

        result.warnings += warning_count
        result.applied += 1
        result.files_applied.append(path)
        LOGGER.info("Applied migration file", extra={"file": str(path)})

    return result


def apply_database_migrations(
    db_url: str,
    *,
    include_optional: bool,
    schema: MigrationSchema,
    dry_run: bool = False,
) -> MigrationResult:
    """
    Plan and apply database migrations using ``EngineManager``.

    Parameters
    ----------
    db_url:
        SQLAlchemy-compatible database URL.
    include_optional:
        Include optional migration files when ``True``.
    schema:
        Schema filter for baseline migration files.
    dry_run:
        When ``True``, return a preview without executing SQL.

    Returns
    -------
    MigrationResult
        Aggregated migration application result.
    """
    plan = build_migration_plan(include_optional=include_optional, schema=schema)
    engine = EngineManager.get_engine(db_url)
    LOGGER.info(
        "Applying migrations",
        extra={
            "db_url": db_url,
            "dry_run": dry_run,
            "schema": schema.value,
            "optional": include_optional,
            "files": len(plan.files),
        },
    )
    return apply_migration_files(engine, plan, dry_run=dry_run)


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
        allow_empty: bool = False,
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
        allow_empty:
            When True, allow missing SQL files and return an empty plan.
        """
        self._db_url = db_url
        self._engine_factory = engine_factory or EngineManager.get_engine
        self._engine: Engine | None = None
        self._tracking_table = self._validate_tracking_table(tracking_table)
        self._allow_empty = allow_empty
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
            if self._allow_empty:
                return tuple()
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

    def load_applied_migrations(self) -> dict[str, str]:
        """
        Return the recorded migration checksums after ensuring the tracking table.
        """
        with self.engine.begin() as connection:
            self._ensure_tracking_table(connection)
            return self._load_applied_migrations(connection)

    def _ensure_tracking_table(self, connection: Any) -> None:
        ddl = text(
            f"CREATE TABLE IF NOT EXISTS {self._tracking_table} (\n"  # nosec B608: table name validated
            "    filename TEXT PRIMARY KEY,\n"
            "    checksum TEXT NOT NULL,\n"
            "    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP\n"
            ")",
        )
        connection.execute(ddl)

    def _load_applied_migrations(self, connection: Any) -> dict[str, str]:
        query = text(
            f"SELECT filename, checksum FROM {self._tracking_table}"  # nosec B608: table name validated
        )
        result = connection.execute(query)
        return {row[0]: row[1] for row in result}

    def _record_applied_migration(self, connection: Any, filename: str, checksum: str) -> None:
        statement = text(
            f"INSERT INTO {self._tracking_table} (filename, checksum)\n"  # nosec B608: table name validated
            "VALUES (:filename, :checksum)\n"
            "ON CONFLICT (filename) DO UPDATE SET checksum = EXCLUDED.checksum",
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
            _execute_migration_statements(
                engine=engine,
                statements=statements,
                path=path,
                tolerate_idempotent_errors=False,
            )
        except SQLAlchemyError as exc:
            msg = f"Failed to apply migration {path.name}"
            raise MigrationRunnerError(msg) from exc


def _execute_migration_statements(
    *,
    engine: Engine,
    statements: Sequence[str],
    path: Path,
    tolerate_idempotent_errors: bool,
) -> int:
    """
    Execute migration statements with transaction handling.

    Parameters
    ----------
    engine:
        SQLAlchemy engine used to execute statements.
    statements:
        Statements to execute in order.
    path:
        Migration file path used for logging context.
    tolerate_idempotent_errors:
        When ``True``, known idempotent SQL errors are converted to warnings.

    Returns
    -------
    int
        Number of idempotent warnings emitted while executing ``statements``.
    """
    warning_count = 0
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
                    if tolerate_idempotent_errors and any(
                        phrase in message for phrase in IDEMPOTENT_ERROR_PHRASES
                    ):
                        LOGGER.warning(
                            "Idempotent migration warning",
                            extra={
                                "file": str(path),
                                "statement_preview": statement[:80],
                            },
                            exc_info=exc,
                        )
                        warning_count += 1
                    else:
                        raise
        except SQLAlchemyError:
            if transaction is not None and transaction.is_active:
                transaction.rollback()
            raise
        else:
            if transaction is not None and transaction.is_active:
                transaction.commit()

    return warning_count


def _requires_dedicated_transaction(statement: str) -> bool:
    stripped = statement.strip().lower()
    return stripped.startswith(
        ("select create_monthly_partitions(", "select create_event_partitions("),
    )


def _execute_autocommit_statement(engine: Engine, statement: str) -> None:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.execute(text(statement))


def _default_migration_paths() -> dict[str, Path]:
    base_dir = Path(__file__).resolve().parent
    return {
        "bootstrap": base_dir / "migrations_bootstrap",
        "legacy": base_dir / "migrations_legacy",
        "incremental": base_dir / "migrations",
    }


def _resolve_profile(profile: MigrationProfile | str | None) -> MigrationProfile:
    if isinstance(profile, MigrationProfile):
        return profile
    token = profile if profile is not None else os.getenv("ML_MIGRATIONS_PROFILE")
    return MigrationProfile.from_env(token)


def apply_profiled_migrations(
    *,
    db_url: str,
    profile: MigrationProfile | str | None = None,
    bootstrap_path: str | Path | None = None,
    legacy_path: str | Path | None = None,
    incremental_path: str | Path | None = None,
    tracking_table: str = TRACKING_TABLE_NAME,
    engine_factory: EngineFactory | None = None,
    dry_run: bool = False,
) -> ProfiledMigrationSummary:
    """
    Apply migrations using bootstrap or legacy profiles based on tracking state.

    The default ``auto`` profile applies the bootstrap migrations only when the
    tracking table is empty, then applies incremental migrations. The ``legacy``
    profile applies only the legacy migration history. The ``incremental``
    profile skips bootstrap and runs incremental migrations only.
    """
    resolved_profile = _resolve_profile(profile)
    paths = _default_migration_paths()
    bootstrap = Path(bootstrap_path) if bootstrap_path else paths["bootstrap"]
    legacy = Path(legacy_path) if legacy_path else paths["legacy"]
    incremental = Path(incremental_path) if incremental_path else paths["incremental"]

    incremental_runner = MigrationRunner(
        db_url=db_url,
        migrations_path=incremental,
        tracking_table=tracking_table,
        engine_factory=engine_factory,
        allow_empty=True,
    )

    if resolved_profile is MigrationProfile.LEGACY:
        legacy_runner = MigrationRunner(
            db_url=db_url,
            migrations_path=legacy,
            tracking_table=tracking_table,
            engine_factory=engine_factory,
            allow_empty=False,
        )
        legacy_summary = legacy_runner.apply_pending_migrations(dry_run=dry_run)
        return ProfiledMigrationSummary(
            profile=resolved_profile,
            bootstrap=None,
            incremental=None,
            legacy=legacy_summary,
        )

    applied_map = incremental_runner.load_applied_migrations()
    bootstrap_summary: MigrationSummary | None = None
    incremental_summary: MigrationSummary | None = None

    if resolved_profile in {MigrationProfile.AUTO, MigrationProfile.BOOTSTRAP}:
        if applied_map:
            if resolved_profile is MigrationProfile.BOOTSTRAP:
                msg = "Bootstrap profile requires an empty migration tracking table"
                raise MigrationRunnerError(msg)
        else:
            bootstrap_runner = MigrationRunner(
                db_url=db_url,
                migrations_path=bootstrap,
                tracking_table=tracking_table,
                engine_factory=engine_factory,
                allow_empty=False,
            )
            bootstrap_summary = bootstrap_runner.apply_pending_migrations(dry_run=dry_run)

    if resolved_profile in {MigrationProfile.AUTO, MigrationProfile.INCREMENTAL, MigrationProfile.BOOTSTRAP}:
        incremental_summary = incremental_runner.apply_pending_migrations(dry_run=dry_run)

    return ProfiledMigrationSummary(
        profile=resolved_profile,
        bootstrap=bootstrap_summary,
        incremental=incremental_summary,
        legacy=None,
    )


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


def _relation_kind(inspector: Any, table_name: str) -> str | None:
    get_views = getattr(inspector, "get_view_names", None)
    schema_candidates: tuple[str | None, ...] = DEFAULT_SCHEMA_CANDIDATES
    default_schema = getattr(inspector, "default_schema_name", None)
    if default_schema and default_schema not in schema_candidates:
        schema_candidates = (*schema_candidates, default_schema)
    for schema in schema_candidates:
        try:
            if table_name in inspector.get_table_names(schema=schema):
                return "table"
        except Exception:
            continue
        if callable(get_views):
            try:
                if table_name in get_views(schema=schema):
                    return "view"
            except Exception:
                continue
    return None


def _resolve_market_data_profile(
    engine: Engine,
    *,
    config: MarketDataTableConfig,
) -> MarketDataTableProfile:
    if config.profile is not MarketDataTableProfile.AUTO:
        return config.profile
    if engine.dialect.name != "postgresql":
        return MarketDataTableProfile.LEGACY
    inspector = inspect(engine)
    relation = _relation_kind(inspector, config.legacy_table)
    if relation == "table":
        return MarketDataTableProfile.LEGACY
    if relation == "view":
        return MarketDataTableProfile.CLASS_TABLES
    return MarketDataTableProfile.CLASS_TABLES


def _verify_table_schema(
    engine: Engine,
    *,
    table_name: str,
    required_columns: Sequence[str],
    required_generated: Sequence[str] = (),
) -> SchemaHealthReport:
    inspector = inspect(engine)
    schema, columns = _resolve_table_columns(inspector, table_name)
    column_map = {column["name"]: column for column in columns}

    missing_columns = tuple(col for col in required_columns if col not in column_map)

    generated_from_db: set[str] | None = None
    missing_generated: list[str] = []
    for column in required_generated:
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

    return SchemaHealthReport(
        table_name=table_name,
        schema=schema,
        missing_columns=missing_columns,
        missing_generated_columns=tuple(missing_generated),
        primary_key=primary_key,
        dialect=engine.dialect.name,
    )


def verify_market_data_schema(
    engine: Engine,
    *,
    table_config: MarketDataTableConfig | None = None,
) -> MarketDataSchemaReport:
    """
    Ensure market data tables satisfy the ingestion contract.
    """
    config = table_config or MarketDataTableConfig.from_env()
    profile = _resolve_market_data_profile(engine, config=config)

    reports: list[SchemaHealthReport] = []
    if profile is MarketDataTableProfile.LEGACY:
        report = _verify_table_schema(
            engine,
            table_name=config.legacy_table,
            required_columns=REQUIRED_MEASURE_COLUMNS,
            required_generated=REQUIRED_GENERATED_COLUMNS,
        )
        reports.append(report)
    else:
        expectations = (
            (config.bar_table, ("instrument_id", "ts_event", "ts_init", "open", "high", "low", "close")),
            (
                config.quote_tick_table,
                ("instrument_id", "ts_event", "ts_init", "bid", "ask", "bid_size", "ask_size"),
            ),
            (
                config.tbbo_table,
                ("instrument_id", "ts_event", "ts_init", "bid", "ask", "bid_size", "ask_size"),
            ),
            (
                config.mbp1_table,
                ("instrument_id", "ts_event", "ts_init", "bid", "ask", "bid_size", "ask_size"),
            ),
            (
                config.mbp10_table,
                ("instrument_id", "ts_event", "ts_init", "bids", "asks", "bid_counts", "ask_counts"),
            ),
            (
                config.mbo_table,
                ("instrument_id", "ts_event", "ts_init", "action", "order_payload"),
            ),
            (config.trade_tick_table, ("instrument_id", "ts_event", "ts_init", "last")),
        )
        for table_name, required_columns in expectations:
            required_generated = (
                REQUIRED_GENERATED_COLUMNS
                if table_name
                in {
                    config.quote_tick_table,
                    config.tbbo_table,
                    config.mbp1_table,
                }
                else ()
            )
            report = _verify_table_schema(
                engine,
                table_name=table_name,
                required_columns=required_columns,
                required_generated=required_generated,
            )
            reports.append(report)

    market_report = MarketDataSchemaReport(profile=profile, tables=tuple(reports))
    if not market_report.healthy:
        raise SchemaHealthCheckError(f"market_data schema invalid: {market_report.describe()}")
    return market_report


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
        help="Custom incremental migrations directory (defaults to ml/stores/migrations)",
    )
    parser.add_argument(
        "--bootstrap-path",
        dest="bootstrap_path",
        default=None,
        help="Custom bootstrap migrations directory (defaults to ml/stores/migrations_bootstrap)",
    )
    parser.add_argument(
        "--legacy-path",
        dest="legacy_path",
        default=None,
        help="Custom legacy migrations directory (defaults to ml/stores/migrations_legacy)",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        choices=tuple(profile.value for profile in MigrationProfile),
        default=None,
        help="Migration profile selector (auto, bootstrap, legacy, incremental)",
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

    use_profile = args.profile or args.bootstrap_path or args.legacy_path
    dry_run = bool(args.command == "plan" or args.dry_run)

    if args.path and not use_profile:
        runner = MigrationRunner(db_url=db_url, migrations_path=args.path)
        summary = runner.apply_pending_migrations(dry_run=dry_run)
        profiled = ProfiledMigrationSummary(
            profile=MigrationProfile.INCREMENTAL,
            bootstrap=None,
            incremental=summary,
            legacy=None,
        )
    else:
        profiled = apply_profiled_migrations(
            db_url=db_url,
            profile=args.profile,
            bootstrap_path=args.bootstrap_path,
            legacy_path=args.legacy_path,
            incremental_path=args.path,
            dry_run=dry_run,
        )

    if profiled.dry_run:
        print("Pending migrations:")
        for path in profiled.pending:
            print(f" - {path}")
        print(
            f"{profiled.pending_count} pending, "
            f"{profiled.already_applied_count} already applied",
        )
        return 0

    print(
        f"Applied {profiled.applied_count} migration(s); "
        f"{profiled.already_applied_count} already recorded.",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())


__all__ = [
    "MarketDataSchemaReport",
    "MigrationPlan",
    "MigrationProfile",
    "MigrationResult",
    "MigrationRunner",
    "MigrationRunnerError",
    "MigrationSchema",
    "MigrationSummary",
    "ProfiledMigrationSummary",
    "SchemaHealthCheckError",
    "SchemaHealthReport",
    "apply_database_migrations",
    "apply_migration_files",
    "apply_profiled_migrations",
    "build_migration_plan",
    "is_postgres_url",
    "main",
    "split_sql_statements",
    "verify_instrumentation_tables",
    "verify_market_data_schema",
]
