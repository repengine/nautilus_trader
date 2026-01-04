#!/usr/bin/env python3

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

"""
Test PostgreSQL integration is working properly.
"""

import pytest
from sqlalchemy import text

from ml.core.db_engine import EngineManager


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
def test_postgres_connection(cloned_test_database: str) -> None:
    """Test that we can connect to PostgreSQL."""
    assert cloned_test_database.startswith("postgresql")
    engine = EngineManager.get_engine(cloned_test_database)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


def test_database_fixture_works(cloned_test_database: str) -> None:
    """Test that the cloned database provides a working connection."""
    engine = EngineManager.get_engine(cloned_test_database)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


@pytest.mark.database
@pytest.mark.serial
def test_database_cleanup(cloned_test_database: str) -> None:
    """Cloned database starts clean and allows writes."""
    engine = EngineManager.get_engine(cloned_test_database)
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'ml_feature_values'
                )
                """,
            ),
        )
        table_exists = result.scalar()

        if table_exists:
            result = conn.execute(text("SELECT COUNT(*) FROM ml_feature_values"))
            assert result.scalar() == 0

            conn.execute(
                text(
                    """
                    INSERT INTO ml_feature_values
                    (feature_set_id, instrument_id, ts_event, ts_init, values)
                    VALUES ('test', 'EUR/USD', 1000000000, 1000000001, '{}')
                    """,
                ),
            )

            result = conn.execute(text("SELECT COUNT(*) FROM ml_feature_values"))
            assert result.scalar() == 1


@pytest.mark.database
@pytest.mark.serial
def test_clean_postgres_db_fixture(cloned_test_database: str) -> None:
    """Cloned DB provides isolated state."""
    assert cloned_test_database


@pytest.mark.database
@pytest.mark.serial
def test_migrations_applied(cloned_test_database: str) -> None:
    """Cloned template has expected migrated tables."""
    engine = EngineManager.get_engine(cloned_test_database)
    with engine.connect() as conn:
        tables_to_check = [
            "ml_feature_values",
            "ml_model_predictions",
            "ml_strategy_signals",
            "ml_data_events",
            "ml_data_watermarks",
        ]

        for table in tables_to_check:
            result = conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = :table
                    )
                    """,
                ),
                {"table": table},
            )
            assert result.scalar(), f"Table {table} should exist after migrations"

        result = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relkind = 'p'
                AND n.nspname = 'public'
                """,
            ),
        )
        assert result.scalar() > 0, "Should have partitioned tables"


@pytest.mark.database
@pytest.mark.serial
def test_postgres_specific_features(cloned_test_database: str) -> None:
    """Ensure PostgreSQL-specific features are available."""
    engine = EngineManager.get_engine(cloned_test_database)
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_proc
                    WHERE proname = 'create_monthly_partitions'
                )
                """,
            ),
        )
        assert result.scalar(), "PL/pgSQL function should exist"

        result = conn.execute(
            text(
                """
                SELECT NOW()::DATE AS today,
                       EXTRACT(EPOCH FROM NOW()) AS epoch,
                       ARRAY[1,2,3] AS test_array
                """,
            ),
        )
        row = result.first()
        assert row is not None
        assert row.today is not None
        assert row.epoch > 0
        assert row.test_array == [1, 2, 3]




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
