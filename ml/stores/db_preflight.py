"""
Lightweight database preflight checks for ML stores.

Verifies presence of required SQL functions and current-month partitions for time-
partitioned tables. Intended for optional use at service startup or in operational
scripts.

"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sqlalchemy import text

from ml.core.db_engine import EngineManager


logger = logging.getLogger(__name__)


REQUIRED_FUNCTIONS = [
    "emit_data_event",
    "update_watermark",
]

PARTITIONED_TABLES = [
    "ml_feature_values",
    "ml_model_predictions",
    "ml_strategy_signals",
]


def _ensure_helper_functions(conn: Any) -> None:
    """
    Ensure helper functions exist (idempotent).
    """
    # create_monthly_partitions is used by migrations and tests
    try:
        exists = conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_proc WHERE proname='create_monthly_partitions')"),
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
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
            partition_name, table_name, start_ns, end_ns
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;
                    """,
                ),
            )
            # Transaction is handled by caller when using engine.begin()
    except Exception:
        # Proceed; partition checks will surface issues
        pass


def check_db_prereqs(connection_string: str) -> dict[str, bool | str]:
    """
    Run preflight checks with best-effort remediation and return a status summary.

    Checks:
    - Required SQL functions exist (update_watermark, emit_data_event)
    - Current-month partition exists for partitioned tables
    - If partitions are missing, attempt to create via auto_create_partitions()

    """
    engine = EngineManager.get_engine(connection_string)
    summary: dict[str, bool | str] = {"ok": True}

    try:
        # Use explicit transactional contexts for any DDL/remediation
        with engine.begin() as conn:
            # Ensure helper exists for partition creation
            _ensure_helper_functions(conn)

            # Check functions
            for fn in REQUIRED_FUNCTIONS:
                exists = conn.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM pg_proc WHERE proname = :fn
                        )
                        """,
                    ),
                    {"fn": fn},
                ).scalar()
                key = f"fn:{fn}"
                summary[key] = bool(exists)
                if not exists:
                    summary["ok"] = False
                    logger.warning("DB preflight: missing function %s", fn)

            # Check current month partitions
            today = date.today()
            part_suffix = f"_{today.year:04d}_{today.month:02d}"
            missing_any = False
            for table in PARTITIONED_TABLES:
                partition_name = table + part_suffix
                exists = conn.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM pg_tables
                            WHERE schemaname = 'public' AND tablename = :name
                        )
                        """,
                    ),
                    {"name": partition_name},
                ).scalar()
                key = f"partition:{partition_name}"
                summary[key] = bool(exists)
                if not exists:
                    summary["ok"] = False
                    logger.warning("DB preflight: missing partition %s", partition_name)
                    missing_any = True

            # Attempt remediation if partitions missing
            if missing_any:
                try:
                    conn.execute(text("SELECT auto_create_partitions()"))
                    # Re-verify within the same transaction context is fine; results reflect post-DDL state
                    today2 = date.today()
                    suffix2 = f"_{today2.year:04d}_{today2.month:02d}"
                    for table in PARTITIONED_TABLES:
                        pname = table + suffix2
                        ok2 = conn.execute(
                            text(
                                "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename=:n)",
                            ),
                            {"n": pname},
                        ).scalar()
                        summary[f"partition:{pname}"] = bool(ok2)
                        if not ok2:
                            summary["ok"] = False
                except Exception as exc:
                    logger.warning("Partition remediation failed: %s", exc)

    except Exception as e:
        logger.error("DB preflight failed: %s", e)
        summary["ok"] = False
        summary["error"] = str(e)

    return summary
