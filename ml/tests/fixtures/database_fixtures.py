#!/usr/bin/env python3

"""
Database fixtures for ML module testing.

This module provides:
- Test database initialization
- Schema creation utilities
- Test data seeding
- Database cleanup utilities
- Transaction management for test isolation

"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


# Track schema initialization per connection URL to avoid re-running migrations
_SCHEMA_INITIALIZED: dict[str, bool] = {}

import pandas as pd
from sqlalchemy import MetaData
from sqlalchemy import event
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from ml.core.db_engine import EngineManager


class TestDatabase:
    """
    Test database manager for ML tests.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        connection_string: str | None = None,
        use_in_memory: bool = False,  # Default to False, prefer PostgreSQL
        auto_rollback: bool = True,
        echo: bool = False,
    ):
        """
        Initialize test database.

        Parameters
        ----------
        engine : Engine, optional
            Pre-configured SQLAlchemy engine (takes precedence)
        connection_string : str, optional
            Database connection string (auto-generated if not provided)
        use_in_memory : bool, default False
            Use in-memory database for unit tests (deprecated, use PostgreSQL)
        auto_rollback : bool, default True
            Automatically rollback transactions after each test
        echo : bool, default False
            Echo SQL statements

        """
        self.use_in_memory = use_in_memory
        self.auto_rollback = auto_rollback
        self.echo = echo

        # Use provided engine if available
        if engine:
            self.engine = engine
            # Use provided connection_string if available, otherwise extract from engine
            # Note: str(engine.url) masks the password with *** which breaks subsequent connections
            if connection_string:
                self.connection_string = connection_string
            else:
                # Fallback - this will have masked password
                self.connection_string = str(engine.url)
        else:
            # Set up connection string
            if connection_string:
                self.connection_string = connection_string
            else:
                # Default to PostgreSQL from environment
                import os

                self.connection_string = os.getenv(
                    "DATABASE_URL",
                    "postgresql://postgres:postgres@localhost:5432/nautilus",
                )

            # Use EngineManager to get or create engine
            # This ensures proper connection pooling and prevents "too many clients" errors
            if "sqlite" in self.connection_string:
                # SQLite-specific settings
                connect_args = (
                    {"check_same_thread": False} if "memory" in self.connection_string else {}
                )
                self.engine = EngineManager.get_engine(
                    self.connection_string,
                    echo=echo,
                    connect_args=connect_args,
                )
                # Enable foreign keys for SQLite
                event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
            else:
                # PostgreSQL or other databases
                # Use conservative pool settings for tests
                self.engine = EngineManager.get_engine(
                    self.connection_string,
                    echo=echo,
                    pool_size=2,  # Conservative for tests
                    max_overflow=3,  # Conservative for tests
                )

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

        # Track if schema is initialized
        self._schema_initialized = False

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_conn: Any, connection_record: Any) -> None:
        """
        Enable foreign key constraints for SQLite.
        """
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    def init_schema(self, schema_files: list[Path] | None = None) -> None:
        """
        Initialize database schema.

        Parameters
        ----------
        schema_files : list[Path], optional
            SQL schema files to execute (uses default ML schema if not provided)

        """
        # Short-circuit when schema already initialized for this engine URL
        try:
            engine_key = str(self.engine.url)
        except Exception:
            engine_key = "default"

        # Detect stale/partial initialization for PostgreSQL and force re-run if needed
        needs_init = True
        try:
            with self.engine.connect() as _conn:
                probe = _conn.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.routines WHERE routine_name = 'create_monthly_partitions')",
                    ),
                ).scalar()
                # If helper function exists, assume base schema applied
                needs_init = not bool(probe)
        except Exception:
            # On any error, fall back to running initialization
            needs_init = True

        if (
            self._schema_initialized or _SCHEMA_INITIALIZED.get(engine_key, False)
        ) and not needs_init:
            self._schema_initialized = True
            return

        # Get default schema files if not provided
        if schema_files is None:
            schema_files = self._get_default_schema_files()

        # Execute schema files
        with self.engine.connect() as conn:
            for schema_file in schema_files:
                if schema_file.exists():
                    sql = schema_file.read_text()

                    # Handle PostgreSQL-specific syntax for SQLite
                    if "sqlite" in self.connection_string:
                        sql = self._adapt_sql_for_sqlite(sql)

                    # Execute SQL statements, respecting dollar-quoted function bodies
                    from typing import Callable, Iterable, cast as _cast

                    try:
                        from ml.cli.apply_migrations import (
                            _split_statements as _split,
                        )  # noqa: WPS433

                        splitter: Callable[[str], Iterable[str]] = _split
                    except Exception:
                        # Fallback simple splitter if import fails
                        def _fallback_splitter(x: str) -> list[str]:
                            return [s for s in x.split(";") if s.strip()]

                        splitter = _fallback_splitter

                    for statement in splitter(sql):
                        statement = statement.strip()
                        if statement and not statement.startswith("--"):
                            try:
                                conn.execute(text(statement))
                            except Exception as e:
                                # Skip errors for unsupported features in SQLite
                                if "sqlite" in self.connection_string and any(
                                    keyword in str(e).lower()
                                    for keyword in ["partition", "inherit", "tablespace"]
                                ):
                                    continue
                                raise

            conn.commit()

            # Ensure helper function exists for PostgreSQL test expectations
            try:
                exists = conn.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM pg_proc WHERE proname='create_monthly_partitions')",
                    ),
                ).scalar()
                if not bool(exists):
                    conn.execute(
                        text(
                            """
CREATE OR REPLACE FUNCTION create_monthly_partitions(
    table_name TEXT,
    start_date DATE,
    num_months INTEGER
)
RETURNS VOID AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_ns BIGINT;
    end_ns BIGINT;
BEGIN
    FOR i IN 0..num_months-1 LOOP
        partition_date := start_date + (i || ' months')::INTERVAL;
        partition_name := table_name || '_' || TO_CHAR(partition_date, 'YYYY_MM');
        start_ns := EXTRACT(EPOCH FROM partition_date) * 1000000000;
        end_ns := EXTRACT(EPOCH FROM partition_date + '1 month'::INTERVAL) * 1000000000;

        EXECUTE format('
            CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
            FOR VALUES FROM (%L) TO (%L)',
            partition_name, table_name, start_ns, end_ns
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;
                            """,
                        ),
                    )
                    conn.commit()
            except Exception:
                # Allow tests to surface specific DB issues
                pass

        # Apply canonical baseline via migration runner (idempotent, best-effort)
        try:
            from ml.cli.apply_migrations import apply_files as _apply_files
            from ml.cli.apply_migrations import build_plan as _build_plan

            plan = _build_plan(full=True, schema="both")
            _apply_files(self.engine, plan, dry_run=False)
        except Exception:
            # Ignore if migration helpers are unavailable in the environment
            pass

        self._schema_initialized = True
        _SCHEMA_INITIALIZED[engine_key] = True

    def _get_default_schema_files(self) -> list[Path]:
        """
        Get default ML schema files.
        """
        migrations_dir = Path(__file__).parent.parent.parent.parent / "stores" / "migrations"

        # Core schema files for testing
        schema_files = [
            migrations_dir / "001_stores_schema.sql",
            migrations_dir / "003_market_data.sql",
            migrations_dir / "004_data_registry.sql",
        ]

        # Filter to existing files
        return [f for f in schema_files if f.exists()]

    def _adapt_sql_for_sqlite(self, sql: str) -> str:
        """
        Adapt PostgreSQL SQL for SQLite compatibility.
        """
        # Remove PostgreSQL-specific features
        replacements = [
            ("SERIAL", "INTEGER"),
            ("BIGSERIAL", "INTEGER"),
            ("JSONB", "TEXT"),
            ("JSON", "TEXT"),
            ("UUID", "TEXT"),
            ("BYTEA", "BLOB"),
            ("TIMESTAMP WITH TIME ZONE", "TIMESTAMP"),
            ("TIMESTAMPTZ", "TIMESTAMP"),
            ("::BIGINT", ""),
            ("::INTEGER", ""),
            ("::TEXT", ""),
            ("ON CONFLICT DO NOTHING", "OR IGNORE"),
            ("PARTITION BY", "-- PARTITION BY"),  # Comment out partitioning
            ("INHERITS", "-- INHERITS"),  # Comment out inheritance
            ("IF NOT EXISTS", ""),  # SQLite doesn't support for all statements
        ]

        for old, new in replacements:
            sql = sql.replace(old, new)

        # Remove CREATE EXTENSION statements
        lines = sql.split("\n")
        filtered_lines = []
        for line in lines:
            if not any(
                keyword in line.upper()
                for keyword in ["CREATE EXTENSION", "CREATE INDEX CONCURRENTLY"]
            ):
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def seed_test_data(self, data_type: str = "basic") -> None:
        """
        Seed database with test data.

        Parameters
        ----------
        data_type : str, default "basic"
            Type of test data to seed ("basic", "full", "minimal")

        """
        with self.get_session() as session:
            if data_type == "basic":
                self._seed_basic_data(session)
            elif data_type == "full":
                self._seed_full_data(session)
            elif data_type == "minimal":
                self._seed_minimal_data(session)

            session.commit()

    def _seed_basic_data(self, session: Session) -> None:
        """
        Seed basic test data.
        """
        # Add instruments
        session.execute(
            text(
                """
                INSERT INTO ml_instruments (instrument_id, symbol, asset_type, tick_size, lot_size)
                VALUES
                    ('EURUSD.SIM', 'EURUSD', 'FX', 0.00001, 1000),
                    ('SPY.XNAS', 'SPY', 'EQUITY', 0.01, 1),
                    ('BTCUSD.COINBASE', 'BTCUSD', 'CRYPTO', 0.01, 0.001)
                ON CONFLICT DO NOTHING
            """,
            ),
        )

        # Add sample feature values
        import time

        current_ns = int(time.time() * 1e9)

        session.execute(
            text(
                """
                INSERT INTO ml_feature_values (
                    feature_set_id, instrument_id, ts_event, ts_init, values, is_live
                )
                VALUES (
                    'test_features_v1',
                    'EURUSD.SIM',
                    :ts_event,
                    :ts_init,
                    '{"sma_20": 1.0900, "rsi": 55.5, "volume": 12345}'::text,
                    false
                )
                ON CONFLICT DO NOTHING
            """,
            ),
            {"ts_event": current_ns, "ts_init": current_ns + 1000},
        )

    def _seed_full_data(self, session: Session) -> None:
        """
        Seed comprehensive test data.
        """
        # Start with basic data
        self._seed_basic_data(session)

        # Add more instruments
        instruments = [
            ("GBPUSD.SIM", "GBPUSD", "FX", 0.00001, 1000),
            ("QQQ.XNAS", "QQQ", "EQUITY", 0.01, 1),
            ("AAPL.XNAS", "AAPL", "EQUITY", 0.01, 1),
        ]

        for inst_id, symbol, asset_type, tick_size, lot_size in instruments:
            session.execute(
                text(
                    """
                    INSERT INTO ml_instruments (instrument_id, symbol, asset_type, tick_size, lot_size)
                    VALUES (:inst_id, :symbol, :asset_type, :tick_size, :lot_size)
                    ON CONFLICT DO NOTHING
                """,
                ),
                {
                    "inst_id": inst_id,
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "tick_size": tick_size,
                    "lot_size": lot_size,
                },
            )

        # Add historical feature values
        import time

        base_ts = int(time.time() * 1e9)

        for i in range(100):
            ts = base_ts - i * 60 * 1e9  # 1-minute intervals

            for inst_id in ["EURUSD.SIM", "GBPUSD.SIM", "SPY.XNAS"]:
                session.execute(
                    text(
                        """
                        INSERT INTO ml_feature_values (
                            feature_set_id, instrument_id, ts_event, ts_init, values, is_live
                        )
                        VALUES (
                            'test_features_v1',
                            :inst_id,
                            :ts_event,
                            :ts_init,
                            :values,
                            false
                        )
                        ON CONFLICT DO NOTHING
                    """,
                    ),
                    {
                        "inst_id": inst_id,
                        "ts_event": ts,
                        "ts_init": ts + 1000,
                        "values": f'{{"sma_20": {1.09 + i * 0.0001}, "rsi": {50 + i % 30}, "volume": {10000 + i * 100}}}',
                    },
                )

    def _seed_minimal_data(self, session: Session) -> None:
        """
        Seed minimal test data.
        """
        # Just one instrument
        session.execute(
            text(
                """
                INSERT INTO ml_instruments (instrument_id, symbol, asset_type, tick_size, lot_size)
                VALUES ('TEST.SIM', 'TEST', 'EQUITY', 0.01, 1)
                ON CONFLICT DO NOTHING
            """,
            ),
        )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session with automatic cleanup.

        Yields
        ------
        Session
            Database session

        """
        session = self.SessionLocal()
        try:
            yield session
            if not self.auto_rollback:
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            if self.auto_rollback:
                session.rollback()
            session.close()

    def clear_all_data(self) -> None:
        """
        Clear all data from database (preserves schema).
        """
        with self.engine.connect() as conn:
            # Get all tables
            metadata = MetaData()
            metadata.reflect(bind=self.engine)

            # Delete data from all tables (in reverse dependency order)
            for table in reversed(metadata.sorted_tables):
                conn.execute(
                    text(f"DELETE FROM {table.name}"),
                )

            conn.commit()

    def drop_all(self) -> None:
        """
        Drop all tables and schema.
        """
        with self.engine.connect() as conn:
            # Get all tables
            metadata = MetaData()
            metadata.reflect(bind=self.engine)

            # Drop all tables
            metadata.drop_all(bind=self.engine)

            conn.commit()

        self._schema_initialized = False

    def cleanup(self) -> None:
        """
        Clean up test database resources.
        """
        # Note: We don't dispose the engine here anymore as it's managed by EngineManager
        # The engine will be disposed by the cleanup_engines fixture after each test

        # Remove temporary file if using file-based SQLite
        if hasattr(self, "db_path") and self.db_path.exists():
            try:
                self.db_path.unlink()
            except Exception:
                import logging as _logging

                _logging.getLogger(__name__).debug(
                    "Failed to unlink temp DB path; ignoring",
                    exc_info=True,
                )

    def execute_sql(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        """
        Execute arbitrary SQL statement.

        Parameters
        ----------
        sql : str
            SQL statement to execute
        params : dict, optional
            Parameters for SQL statement

        Returns
        -------
        Any
            Query result

        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            conn.commit()
            return result

    def get_table_data(self, table_name: str) -> pd.DataFrame:
        """
        Get all data from a table as DataFrame.

        Parameters
        ----------
        table_name : str
            Name of table to query

        Returns
        -------
        pd.DataFrame
            Table data

        """
        return pd.read_sql(
            f"SELECT * FROM {table_name}",
            self.engine,
        )


def create_test_database(**kwargs: Any) -> TestDatabase:
    """
    Factory function to create test database.

    Parameters
    ----------
    **kwargs
        Arguments passed to TestDatabase constructor

    Returns
    -------
    TestDatabase
        Configured test database instance

    """
    return TestDatabase(**kwargs)


@contextmanager
def temp_database(
    use_in_memory: bool = True,
    init_schema: bool = True,
    seed_data: str | None = None,
) -> Generator[TestDatabase, None, None]:
    """
    Context manager for temporary test database.

    Parameters
    ----------
    use_in_memory : bool, default True
        Use in-memory database
    init_schema : bool, default True
        Initialize database schema
    seed_data : str, optional
        Type of test data to seed

    Yields
    ------
    TestDatabase
        Temporary test database

    """
    db = TestDatabase(use_in_memory=use_in_memory)

    try:
        if init_schema:
            db.init_schema()

        if seed_data:
            db.seed_test_data(seed_data)

        yield db
    finally:
        db.cleanup()


class DatabaseSnapshot:
    """
    Utility for database state snapshots and restoration.
    """

    def __init__(self, database: TestDatabase):
        """
        Initialize snapshot utility.
        """
        self.database = database
        self.snapshots: dict[str, dict[str, pd.DataFrame]] = {}

    def take_snapshot(self, name: str = "default") -> None:
        """
        Take snapshot of current database state.
        """
        snapshot = {}

        # Get all tables
        metadata = MetaData()
        metadata.reflect(bind=self.database.engine)

        # Save data from each table
        for table in metadata.tables:
            df = self.database.get_table_data(table)
            snapshot[table] = df

        self.snapshots[name] = snapshot

    def restore_snapshot(self, name: str = "default") -> None:
        """
        Restore database to snapshot state.
        """
        if name not in self.snapshots:
            raise ValueError(f"Snapshot '{name}' not found")

        snapshot = self.snapshots[name]

        # Clear all data
        self.database.clear_all_data()

        # Restore data for each table
        with self.database.get_session() as session:
            for table_name, df in snapshot.items():
                if not df.empty:
                    df.to_sql(table_name, self.database.engine, if_exists="append", index=False)
            session.commit()

    def has_snapshot(self, name: str = "default") -> bool:
        """
        Check if snapshot exists.
        """
        return name in self.snapshots
