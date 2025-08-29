"""
Lightweight database preflight checks for ML stores.

Verifies presence of required SQL functions and current-month partitions
for time-partitioned tables. Intended for optional use at service startup
or in operational scripts.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

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


def check_db_prereqs(connection_string: str) -> dict[str, bool | str]:
    """
    Run preflight checks and return a status summary.

    Checks:
    - Required SQL functions exist
    - Current-month partition exists for partitioned tables
    """
    engine = EngineManager.get_engine(connection_string)
    summary: dict[str, bool | str] = {"ok": True}

    try:
        with engine.connect() as conn:
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

    except Exception as e:
        logger.error("DB preflight failed: %s", e)
        summary["ok"] = False
        summary["error"] = str(e)

    return summary

