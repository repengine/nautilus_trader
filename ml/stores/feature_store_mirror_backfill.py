"""
Feature store mirror backfill utilities.

These helpers read rows from ``ml_feature_values`` and write them to the parquet
mirror used for API-less restoration. The logic is intentionally cold-path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from ml.config.feature_store_mirror import FeatureStoreMirrorBackfillConfig
from ml.config.feature_store_mirror import FeatureStoreMirrorConfig
from ml.core.db_engine import EngineManager
from ml.stores.feature_raw_writer import FeatureValuesParquetMirrorWriter


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureStoreMirrorBackfillResult:
    """Summary of a mirror backfill run."""

    rows_scanned: int
    rows_written: int
    batches: int


def backfill_feature_store_mirror(
    config: FeatureStoreMirrorBackfillConfig,
    *,
    mirror_config: FeatureStoreMirrorConfig | None = None,
) -> FeatureStoreMirrorBackfillResult:
    """
    Backfill the feature store parquet mirror from SQL.

    Args:
        config: Backfill configuration containing DSN and filters.
        mirror_config: Mirror writer configuration (defaults to env-based config).

    Returns:
        Backfill result summary.
    """
    mirror_cfg = mirror_config or FeatureStoreMirrorConfig.from_env()
    if not mirror_cfg.enabled:
        logger.info("Feature store mirror backfill disabled")
        return FeatureStoreMirrorBackfillResult(rows_scanned=0, rows_written=0, batches=0)

    writer = FeatureValuesParquetMirrorWriter(
        base_dir=mirror_cfg.base_dir,
        partition_field=mirror_cfg.partition_field,
        timestamp_field=mirror_cfg.timestamp_field,
        values_field=mirror_cfg.values_field,
    )

    engine = EngineManager.get_engine(
        config.db_connection,
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
    )

    sql, params = _build_query(config)
    rows_scanned = 0
    rows_written = 0
    batches = 0
    with engine.connect().execution_options(stream_results=True) as conn:
        result = conn.execute(text(sql), params)
        while True:
            batch = result.fetchmany(config.batch_size)
            if not batch:
                break
            rows_scanned += len(batch)
            payload: list[dict[str, Any]] = [dict(row._mapping) for row in batch]
            rows_written += writer.write_rows(payload)
            batches += 1

    logger.info(
        "Feature store mirror backfill complete",
        extra={
            "rows_scanned": rows_scanned,
            "rows_written": rows_written,
            "batches": batches,
            "base_dir": str(mirror_cfg.base_dir),
        },
    )
    return FeatureStoreMirrorBackfillResult(
        rows_scanned=rows_scanned,
        rows_written=rows_written,
        batches=batches,
    )


def _build_query(config: FeatureStoreMirrorBackfillConfig) -> tuple[str, dict[str, Any]]:
    columns = [
        "feature_set_id",
        "instrument_id",
        "ts_event",
        "ts_init",
        "values",
    ]
    where: list[str] = []
    params: dict[str, Any] = {}
    if config.start_ts is not None:
        where.append("ts_event >= :start_ts")
        params["start_ts"] = config.start_ts
    if config.end_ts is not None:
        where.append("ts_event < :end_ts")
        params["end_ts"] = config.end_ts
    if config.feature_set_id:
        where.append("feature_set_id = :feature_set_id")
        params["feature_set_id"] = config.feature_set_id
    if config.instrument_id:
        where.append("instrument_id = :instrument_id")
        params["instrument_id"] = config.instrument_id

    sql = f"SELECT {', '.join(columns)} FROM ml_feature_values"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ts_event"
    return sql, params


__all__ = [
    "FeatureStoreMirrorBackfillResult",
    "backfill_feature_store_mirror",
]
