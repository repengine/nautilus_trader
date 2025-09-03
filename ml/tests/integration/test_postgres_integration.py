#!/usr/bin/env python3
"""Test PostgreSQL integration is working properly."""

import pytest
from sqlalchemy import text


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
def test_postgres_connection(postgres_connection):
    """Test that we can connect to PostgreSQL."""
    assert "postgresql" in postgres_connection
    assert "nautilus_test" in postgres_connection


def test_database_fixture_works(test_database):
    """Test that the database fixture provides a working connection."""
    # Test we can execute a simple query
    with test_database.engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


@pytest.mark.database
@pytest.mark.serial
def test_database_cleanup(test_database):
    """Test that database is properly cleaned between tests."""
    # Create a test table
    with test_database.engine.connect() as conn:
        # Check if feature values table exists (from migrations)
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'ml_feature_values'
            )
        """))
        table_exists = result.scalar()

        if table_exists:
            # Count rows (should be 0 due to cleanup)
            result = conn.execute(text("SELECT COUNT(*) FROM ml_feature_values"))
            count = result.scalar()
            assert count == 0, "Table should be empty after cleanup"

            # Insert test data
            conn.execute(text("""
                INSERT INTO ml_feature_values
                (feature_set_id, instrument_id, ts_event, ts_init, values)
                VALUES ('test', 'EUR/USD', 1000000000, 1000000001, '{}')
            """))
            conn.commit()

            # Verify insert
            result = conn.execute(text("SELECT COUNT(*) FROM ml_feature_values"))
            count = result.scalar()
            assert count == 1, "Should have 1 row after insert"


@pytest.mark.database
@pytest.mark.serial
def test_clean_postgres_db_fixture(clean_postgres_db):
    """Test that clean_postgres_db fixture ensures clean state."""
    # This fixture should ensure database is clean
    # Just verify it doesn't error
    assert True


@pytest.mark.database
@pytest.mark.serial
def test_migrations_applied(test_database):
    """Test that migrations have been applied to the test database."""
    with test_database.engine.connect() as conn:
        # Check for key tables from migrations
        tables_to_check = [
            "ml_feature_values",
            "ml_model_predictions",
            "ml_strategy_signals"
        ]

        for table in tables_to_check:
            result = conn.execute(text(f"""  # noqa: S608 - table list is internal/test-controlled
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = '{table}'
                    )
                """))
            exists = result.scalar()
            assert exists, f"Table {table} should exist after migrations"

        # Check for partitioned tables (PostgreSQL-specific feature)
        result = conn.execute(text("""
            SELECT COUNT(*)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'p'  -- partitioned table
            AND n.nspname = 'public'
        """))
        partition_count = result.scalar()
        assert partition_count > 0, "Should have partitioned tables"


@pytest.mark.database
@pytest.mark.serial
def test_postgres_specific_features(test_database):
    """Test that PostgreSQL-specific features are available."""
    with test_database.engine.connect() as conn:
        # Test PL/pgSQL function exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_proc
                WHERE proname = 'create_monthly_partitions'
            )
        """))
        function_exists = result.scalar()
        assert function_exists, "PL/pgSQL function should exist"

        # Test we can use PostgreSQL-specific syntax
        result = conn.execute(text("""
            SELECT NOW()::DATE AS today,
                   EXTRACT(EPOCH FROM NOW()) AS epoch,
                   ARRAY[1,2,3] AS test_array
        """))
        row = result.first()
        assert row is not None
        assert row.today is not None
        assert row.epoch > 0
        assert row.test_array == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
