from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import NoSuchTableError

from ml.stores.migrations_runner import MigrationRunner
from ml.stores.migrations_runner import MigrationRunnerError
from ml.stores.migrations_runner import SchemaHealthCheckError
from ml.stores.migrations_runner import apply_profiled_migrations
from ml.config.market_data import MarketDataTableProfile
from ml.stores.migrations_runner import verify_market_data_schema
from ml.stores.migrations_runner import verify_instrumentation_tables


def _sqlite_engine(db_path: Path):
    return create_engine(f"sqlite:///{db_path}", future=True)


def test_runner_applies_and_tracks_migrations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "runner.sqlite"
    engine = _sqlite_engine(db_path)

    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    migration_file = migrations_dir / "001_create_table.sql"
    migration_file.write_text(
        """
        CREATE TABLE IF NOT EXISTS market_data (
            instrument_id TEXT PRIMARY KEY,
            bid REAL
        );
        """,
        encoding="utf-8",
    )

    runner = MigrationRunner(
        db_url=f"sqlite:///{db_path}",
        migrations_path=migrations_dir,
        engine_factory=lambda _: engine,
    )

    summary = runner.apply_pending_migrations()
    assert summary.applied_count == 1
    assert summary.already_applied_count == 0
    assert summary.pending_count == 0

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT COUNT(*) FROM ml_schema_migrations"))
        assert rows.scalar_one() == 1

    repeat_summary = runner.apply_pending_migrations()
    assert repeat_summary.applied_count == 0
    assert repeat_summary.already_applied_count == 1
    assert repeat_summary.pending_count == 0


def test_runner_detects_checksum_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "runner.sqlite"
    engine = _sqlite_engine(db_path)
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    migration_file = migrations_dir / "001_mutable.sql"
    migration_file.write_text("CREATE TABLE foo (id INTEGER PRIMARY KEY);", encoding="utf-8")

    runner = MigrationRunner(
        db_url=f"sqlite:///{db_path}",
        migrations_path=migrations_dir,
        engine_factory=lambda _: engine,
    )
    runner.apply_pending_migrations()

    migration_file.write_text(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY, note TEXT);",
        encoding="utf-8",
    )

    with pytest.raises(MigrationRunnerError):
        runner.apply_pending_migrations()


def test_runner_allows_checksum_mismatch_when_env_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "runner.sqlite"
    engine = _sqlite_engine(db_path)
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    migration_file = migrations_dir / "001_mutable.sql"
    migration_file.write_text("CREATE TABLE foo (id INTEGER PRIMARY KEY);", encoding="utf-8")

    runner = MigrationRunner(
        db_url=f"sqlite:///{db_path}",
        migrations_path=migrations_dir,
        engine_factory=lambda _: engine,
    )
    runner.apply_pending_migrations()

    migration_file.write_text(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY, note TEXT);",
        encoding="utf-8",
    )
    monkeypatch.setenv("ML_ALLOW_MIGRATION_DRIFT", "1")

    runner.apply_pending_migrations()


def _write_migration(directory: Path, name: str, sql: str) -> Path:
    path = directory / name
    path.write_text(sql, encoding="utf-8")
    return path


def test_apply_profiled_migrations_auto_applies_bootstrap_and_incremental(tmp_path: Path) -> None:
    db_path = tmp_path / "profiled.sqlite"
    db_url = f"sqlite:///{db_path}"
    bootstrap_dir = tmp_path / "bootstrap"
    incremental_dir = tmp_path / "incremental"
    bootstrap_dir.mkdir()
    incremental_dir.mkdir()

    _write_migration(
        bootstrap_dir,
        "001_bootstrap.sql",
        "CREATE TABLE bootstrap_table (id INTEGER PRIMARY KEY);",
    )
    _write_migration(
        incremental_dir,
        "002_incremental.sql",
        "CREATE TABLE incremental_table (id INTEGER PRIMARY KEY);",
    )

    summary = apply_profiled_migrations(
        db_url=db_url,
        bootstrap_path=bootstrap_dir,
        incremental_path=incremental_dir,
    )

    assert summary.applied_count == 2

    engine = _sqlite_engine(db_path)
    with engine.connect() as conn:
        assert (
            conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='bootstrap_table'",
                ),
            ).scalar_one()
            == "bootstrap_table"
        )
        assert (
            conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='incremental_table'",
                ),
            ).scalar_one()
            == "incremental_table"
        )


def test_apply_profiled_migrations_bootstrap_rejects_non_empty_tracking(tmp_path: Path) -> None:
    db_path = tmp_path / "profiled.sqlite"
    db_url = f"sqlite:///{db_path}"
    bootstrap_dir = tmp_path / "bootstrap"
    incremental_dir = tmp_path / "incremental"
    bootstrap_dir.mkdir()
    incremental_dir.mkdir()

    _write_migration(
        bootstrap_dir,
        "001_bootstrap.sql",
        "CREATE TABLE bootstrap_table (id INTEGER PRIMARY KEY);",
    )
    _write_migration(
        incremental_dir,
        "002_incremental.sql",
        "CREATE TABLE incremental_table (id INTEGER PRIMARY KEY);",
    )

    apply_profiled_migrations(
        db_url=db_url,
        bootstrap_path=bootstrap_dir,
        incremental_path=incremental_dir,
    )

    with pytest.raises(MigrationRunnerError):
        apply_profiled_migrations(
            db_url=db_url,
            profile="bootstrap",
            bootstrap_path=bootstrap_dir,
            incremental_path=incremental_dir,
        )


class _InspectorStub:
    default_schema_name = "public"

    def __init__(self, columns: list[dict[str, Any]]) -> None:
        self._columns = columns

    def get_columns(self, table_name: str, schema: str | None = None) -> list[dict[str, Any]]:
        if table_name != "market_data":
            raise NoSuchTableError(table_name)
        return self._columns

    def get_pk_constraint(self, table_name: str, schema: str | None = None) -> dict[str, Any]:
        if table_name != "market_data":
            raise NoSuchTableError(table_name)
        return {"constrained_columns": ["instrument_id", "ts_event"]}


class _DummyEngine:
    def __init__(self) -> None:
        self.dialect = SimpleNamespace(name="sqlite")

    def connect(self):
        raise AssertionError("connect should not be called for sqlite schema check")


class _InstrumentationInspectorStub:
    default_schema_name = "public"

    def __init__(self, tables: set[str]) -> None:
        self._tables = tables

    def has_table(self, table_name: str, schema: str | None = None) -> bool:
        schema_token = schema or "public"
        return f"{schema_token}.{table_name}" in self._tables


def test_verify_market_data_schema_success(monkeypatch: pytest.MonkeyPatch) -> None:
    columns = [
        {"name": "instrument_id"},
        {"name": "ts_event"},
        {"name": "ts_init"},
        {"name": "bid"},
        {"name": "ask"},
        {"name": "bid_size"},
        {"name": "ask_size"},
        {"name": "spread", "computed": {"sqltext": "ask - bid", "persisted": True}},
        {"name": "mid_price", "computed": {"sqltext": "(bid + ask) / 2", "persisted": True}},
    ]
    inspector = _InspectorStub(columns)
    monkeypatch.setattr("ml.stores.migrations_runner.inspect", lambda _: inspector)
    report = verify_market_data_schema(_DummyEngine())
    assert report.healthy
    assert report.profile is MarketDataTableProfile.LEGACY
    assert report.tables[0].table_name == "market_data"


def test_verify_market_data_schema_missing_column(monkeypatch: pytest.MonkeyPatch) -> None:
    columns = [
        {"name": "instrument_id"},
        {"name": "ts_event"},
        {"name": "bid"},
        {"name": "ask"},
    ]
    inspector = _InspectorStub(columns)
    monkeypatch.setattr("ml.stores.migrations_runner.inspect", lambda _: inspector)
    with pytest.raises(SchemaHealthCheckError):
        verify_market_data_schema(_DummyEngine())


def test_verify_instrumentation_tables_success(monkeypatch: pytest.MonkeyPatch) -> None:
    inspector = _InstrumentationInspectorStub(
        {"public.ml_data_events", "public.ml_data_watermarks"},
    )
    monkeypatch.setattr("ml.stores.migrations_runner.inspect", lambda _: inspector)
    verify_instrumentation_tables(_DummyEngine())


def test_verify_instrumentation_tables_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    inspector = _InstrumentationInspectorStub({"public.ml_data_events"})
    monkeypatch.setattr("ml.stores.migrations_runner.inspect", lambda _: inspector)
    with pytest.raises(SchemaHealthCheckError):
        verify_instrumentation_tables(_DummyEngine())
