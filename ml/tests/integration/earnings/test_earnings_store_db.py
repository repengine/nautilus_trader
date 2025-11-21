"""Integration tests covering the PostgreSQL-backed earnings store."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

from ml.stores.earnings_store import EarningsStore

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures(
        "isolated_prometheus_registry",
        "mock_tracing_backend",
        "isolated_orchestrator_env",
    ),
]


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/nautilus_trader",
)


@pytest.fixture
def temporary_earnings_store() -> EarningsStore:
    """Provision an isolated schema for exercising PostgreSQL DDL."""

    schema = f"ml_test_earnings_{uuid4().hex[:8]}"
    store = EarningsStore(connection_string=DATABASE_URL, schema=schema)
    try:
        yield store
    finally:
        with store._engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))


@pytest.mark.database
@pytest.mark.serial
def test_earnings_store_bootstraps_tables(temporary_earnings_store: EarningsStore) -> None:
    """First write succeeds because the store provisions its own tables."""

    store = temporary_earnings_store
    ticker = "EARNINGS_BOOT"

    store.write_actuals(
        ticker=ticker,
        period_end="2024-03-31",
        filing_date="2024-04-25",
        eps_diluted=1.23,
        revenue=94_000_000_000.0,
        ts_event=1_712_009_600_000_000_000,
        ts_init=1_712_009_601_000_000_000,
    )
    store.write_estimates(
        ticker=ticker,
        estimate_date="2024-03-01",
        period_end="2024-03-31",
        eps_consensus=1.10,
        ts_event=1_711_728_000_000_000_000,
        ts_init=1_711_728_001_000_000_000,
    )

    with store._engine.connect() as conn:
        count_actuals = conn.execute(
            text(
                "SELECT COUNT(*) FROM "
                f"{store._schema}.earnings_actuals WHERE ticker = :ticker",
            ),
            {"ticker": ticker},
        ).scalar_one()
        count_estimates = conn.execute(
            text(
                "SELECT COUNT(*) FROM "
                f"{store._schema}.earnings_estimates WHERE ticker = :ticker",
            ),
            {"ticker": ticker},
        ).scalar_one()

    assert count_actuals == 1
    assert count_estimates == 1


@pytest.mark.database
@pytest.mark.serial
def test_earnings_store_queries_order_by_ts_event(temporary_earnings_store: EarningsStore) -> None:
    """Query builders order by period/estimate date and ts_event for restatements."""

    store = temporary_earnings_store

    actuals_query = store._build_actuals_query(
        ticker="EARNINGS_ORDER",
        start_date=None,
        end_date=None,
        as_of_ts=None,
    )
    estimates_query = store._build_estimates_query(
        ticker="EARNINGS_ORDER",
        period_end="2024-03-31",
        as_of_ts=None,
    )

    compiled_actuals = str(
        actuals_query.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"render_postcompile": True},
        ),
    )
    compiled_estimates = str(
        estimates_query.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"render_postcompile": True},
        ),
    )

    actuals_sql = compiled_actuals.lower()
    estimates_sql = compiled_estimates.lower()

    assert "order by" in actuals_sql
    assert "period_end desc" in actuals_sql
    assert "ts_event desc" in actuals_sql

    assert "order by" in estimates_sql
    assert "estimate_date desc" in estimates_sql
    assert "ts_event desc" in estimates_sql
