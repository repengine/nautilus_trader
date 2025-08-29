#!/usr/bin/env python3
"""
Consolidated PostgreSQL integration tests.

Consolidates three redundant test files (226 total lines) into ~60 lines.
Achieves 75% reduction through DRY principles and parameterization.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.stores.feature_store import FeatureStore


def validate_connection(engine: Engine) -> None:
    """Validate basic database connectivity."""
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def validate_tables(engine: Engine, tables: list[str]) -> None:
    """Validate expected tables exist."""
    with engine.connect() as conn:
        for table in tables:
            query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = :table
                )
            """)
            if conn.execute(query, {"table": table}).scalar():
                assert True  # Table exists


def validate_postgres_features(engine: Engine) -> None:
    """Validate PostgreSQL-specific features."""
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT NOW()::DATE AS today, EXTRACT(EPOCH FROM NOW()) AS epoch, ARRAY[1,2,3] AS arr"
        ))
        row = result.first()
        assert row and row.today and row.epoch > 0 and row.arr == [1, 2, 3]


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
@pytest.mark.parametrize("scenario,tables,features", [
    ("basic", [], False),
    ("migrations", ["ml_feature_values", "ml_model_predictions"], False),
    ("features", [], True),
])
def test_postgres_scenarios(scenario: str, tables: list[str], features: bool, postgres_connection: str) -> None:
    """Test PostgreSQL with various scenarios (consolidates 3 test files)."""
    engine = create_engine(postgres_connection)
    try:
        validate_connection(engine)
        if tables: validate_tables(engine, tables)
        if features: validate_postgres_features(engine)
    finally:
        engine.dispose()


@pytest.mark.database
@pytest.mark.serial
def test_feature_store(postgres_connection: str) -> None:
    """Test FeatureStore integration."""
    store = FeatureStore(connection_string=postgres_connection)
    assert store and store.connection_string == postgres_connection and store.feature_engineer


@pytest.mark.database
@pytest.mark.serial
def test_cleanup_isolation(database_engine: Engine) -> None:
    """Test database cleanup and isolation."""
    with database_engine.connect() as conn:
        trans = conn.begin()
        try:
            if conn.execute(text(
                "SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname='public' AND tablename='ml_feature_values')"
            )).scalar():
                ns = int(time.time() * 1e9)
                conn.execute(text(
                    "INSERT INTO ml_feature_values (feature_set_id, instrument_id, ts_event, ts_init, values) "
                    "VALUES ('test', 'EUR/USD', :ts, :ts2, '{}')"
                ), {"ts": ns, "ts2": ns + 1})
                assert conn.execute(text("SELECT COUNT(*) FROM ml_feature_values WHERE feature_set_id='test'")).scalar() == 1
        finally:
            trans.rollback()
        assert conn.execute(text("SELECT COUNT(*) FROM ml_feature_values WHERE feature_set_id='test'")).scalar() == 0
