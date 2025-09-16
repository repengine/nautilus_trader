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

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager


def _ensure_functions_and_partitions(engine: Engine) -> None:
    """Create helper functions and current-month partitions if missing."""
    with engine.begin() as conn:
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

        # Attempt to auto-create partitions for the current and next month
        try:
            conn.execute(text("SELECT auto_create_partitions()"))
        except Exception:
            # Silently ignore when function is unavailable; tests will gate DB usage
            pass

        # Ensure DEFAULT partitions exist for partitioned tables so inserts do not fail
        try:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS ml_feature_values_default PARTITION OF ml_feature_values DEFAULT",
            ))
        except Exception:
            pass
        try:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS ml_model_predictions_default PARTITION OF ml_model_predictions DEFAULT",
            ))
        except Exception:
            pass
        try:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS ml_strategy_signals_default PARTITION OF ml_strategy_signals DEFAULT",
            ))
        except Exception:
            pass

        # Ensure required unique index for FeatureStore upserts exists
        try:
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_feature_values_key ON public.ml_feature_values (feature_set_id, instrument_id, ts_event)",
            ))
        except Exception:
            pass


def main() -> None:
    """Run database remediation if DATABASE_URL is set and reachable."""
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
