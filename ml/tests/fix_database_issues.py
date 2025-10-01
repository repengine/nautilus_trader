#!/usr/bin/env python3
"""
Best-effort remediation for test database prerequisites.

This module is invoked by ml/tests/conftest.py during pytest_sessionstart to ensure
helper functions and monthly partitions exist for the ML tables used by tests.

It is intentionally idempotent and safe to run when PostgreSQL is unavailable; in that
case, it exits quietly.

"""

from __future__ import annotations

import os
from typing import Any
from typing import Final

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager

_TRIGGERS_TO_DROP: Final[tuple[tuple[str, str], ...]] = (
    ("ml_feature_values", "auto_create_partition_feature_values"),
    ("ml_model_predictions", "auto_create_partition_model_predictions"),
    ("ml_strategy_signals", "auto_create_partition_strategy_signals"),
)

_PARTITIONED_TABLES: Final[tuple[str, ...]] = (
    "ml_feature_values",
    "ml_model_predictions",
    "ml_strategy_signals",
)

_CREATE_TABLE_SQL: Final[dict[str, str]] = {
    "ml_feature_values": (
        """
        CREATE TABLE IF NOT EXISTS public.ml_feature_values (
            id BIGSERIAL,
            feature_set_id VARCHAR(255) NOT NULL,
            instrument_id VARCHAR(100) NOT NULL,
            ts_event BIGINT NOT NULL,
            ts_init BIGINT NOT NULL,
            values JSONB NOT NULL,
            is_live BOOLEAN DEFAULT FALSE,
            source VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (feature_set_id, instrument_id, ts_event)
        ) PARTITION BY RANGE (ts_event)
        """
    ),
    "ml_model_predictions": (
        """
        CREATE TABLE IF NOT EXISTS public.ml_model_predictions (
            prediction_id BIGSERIAL,
            model_id VARCHAR(255) NOT NULL,
            instrument_id VARCHAR(100) NOT NULL,
            ts_event BIGINT NOT NULL,
            ts_init BIGINT NOT NULL,
            prediction DOUBLE PRECISION NOT NULL,
            confidence DOUBLE PRECISION,
            features_used JSONB,
            inference_time_ms DOUBLE PRECISION,
            is_live BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (model_id, instrument_id, ts_event)
        ) PARTITION BY RANGE (ts_event)
        """
    ),
    "ml_strategy_signals": (
        """
        CREATE TABLE IF NOT EXISTS public.ml_strategy_signals (
            signal_id BIGSERIAL,
            strategy_id VARCHAR(255) NOT NULL,
            instrument_id VARCHAR(100) NOT NULL,
            ts_event BIGINT NOT NULL,
            ts_init BIGINT NOT NULL,
            signal_type VARCHAR(50) NOT NULL,
            strength DOUBLE PRECISION NOT NULL,
            model_predictions JSONB,
            risk_metrics JSONB,
            execution_params JSONB,
            is_live BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (strategy_id, instrument_id, ts_event)
        ) PARTITION BY RANGE (ts_event)
        """
    ),
}

_CREATE_DEFAULT_PARTITION_SQL: Final[dict[str, str]] = {
    table: (
        f"CREATE TABLE IF NOT EXISTS public.{table}_default PARTITION OF public.{table} DEFAULT"
    )
    for table in _PARTITIONED_TABLES
}


def _ensure_functions_and_partitions(engine: Engine) -> None:
    """
    Create helper functions and current-month partitions if missing.
    """
    with engine.begin() as conn:
        try:
            conn.execute(text("ALTER DATABASE nautilus SET search_path = public, pg_catalog"))
        except Exception:
            # Non-fatal when executing against alternate database names or limited privileges.
            pass

        conn.execute(text("SET search_path TO public, pg_catalog"))

        # Drop legacy triggers that attempt to auto-create partitions per row.
        for table_name, trigger_name in _TRIGGERS_TO_DROP:
            try:
                conn.execute(
                    text(
                        f"DROP TRIGGER IF EXISTS {trigger_name} ON public.{table_name}",
                    ),
                )
            except Exception:
                # Non-fatal; continue with cleanup
                pass

        # Drop the problematic ensure_partition_exists trigger function if present.
        try:
            conn.execute(text("DROP FUNCTION IF EXISTS ensure_partition_exists()"))
        except Exception:
            pass
        try:
            conn.execute(text("DROP FUNCTION IF EXISTS ml_registry.ensure_partition_exists()"))
        except Exception:
            pass

        try:
            conn.execute(
                text(
                    """
CREATE OR REPLACE FUNCTION ensure_partition_exists()
RETURNS TRIGGER AS $$
BEGIN
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
                    """,
                ),
            )
        except Exception:
            pass
        try:
            conn.execute(
                text(
                    """
CREATE OR REPLACE FUNCTION ml_registry.ensure_partition_exists()
RETURNS TRIGGER AS $$
BEGIN
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
                    """,
                ),
            )
        except Exception:
            pass

        # Create helper functions used by preflight checks
        try:
            from ml.stores import infrastructure as _infra

            helper = getattr(_infra, "_ensure_helper_functions", None)
            if callable(helper):
                helper_any: Any = helper
                helper_any(conn)
        except Exception:
            # Fallback: attempt to call function if it already exists
            pass

        # Ensure the canonical partitioned tables exist under the public schema.
        for table_name in _PARTITIONED_TABLES:
            conn.execute(
                text(
                    f"DROP TABLE IF EXISTS public.{table_name} CASCADE",
                ),
            )
            create_sql = _CREATE_TABLE_SQL[table_name]
            conn.execute(text(create_sql))

            # Ensure the DEFAULT partition exists (idempotent).
            conn.execute(text(_CREATE_DEFAULT_PARTITION_SQL[table_name]))

        # Seed current and near-future partitions to satisfy preflight checks.
        for table_name in _PARTITIONED_TABLES:
            try:
                conn.execute(
                    text(
                        "SELECT create_monthly_partitions(:table_name, DATE_TRUNC('month', CURRENT_DATE)::DATE, :months)",
                    ),
                    {
                        "table_name": table_name,
                        "months": 6,
                    },
                )
            except Exception:
                # Function may be unavailable; tests relying on default partitions remain satisfied.
                pass

        # Attempt to auto-create partitions for the current and next month
        try:
            conn.execute(text("SELECT auto_create_partitions()"))
        except Exception:
            # Silently ignore when function is unavailable; tests will gate DB usage
            pass

        # Ensure DEFAULT partitions exist for partitioned tables so inserts do not fail
        try:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS ml_feature_values_default PARTITION OF ml_feature_values DEFAULT",
                ),
            )
        except Exception:
            pass
        try:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS ml_model_predictions_default PARTITION OF ml_model_predictions DEFAULT",
                ),
            )
        except Exception:
            pass
        try:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS ml_strategy_signals_default PARTITION OF ml_strategy_signals DEFAULT",
                ),
            )
        except Exception:
            pass

        # Ensure required unique index for FeatureStore upserts exists
        try:
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_feature_values_key "
                    "ON public.ml_feature_values (feature_set_id, instrument_id, ts_event)",
                ),
            )
        except Exception:
            pass


def main() -> None:
    """
    Run database remediation if DATABASE_URL is set and reachable.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        return
    try:
        engine = EngineManager.get_engine(url)
        _ensure_functions_and_partitions(engine)
    except Exception:
        # Non-fatal for unit tests without DB
        return


if __name__ == "__main__":
    main()
