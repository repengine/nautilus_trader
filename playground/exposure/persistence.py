"""Persistence helpers for cross-asset beta exposures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import polars as pl
import structlog
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.exc import SQLAlchemyError

from ml.common.db_connections import ConnectionRole
from ml.common.db_connections import collect_postgres_candidates
from ml.common.db_connections import select_first_working_connection
from ml.common.metrics_manager import MetricsManager
from ml.core.db_engine import EngineManager


LOGGER = structlog.get_logger(__name__)


@dataclass(slots=True)
class CrossAssetBetaPersistenceConfig:
    """Configuration for persisting EWMA betas into the feature store."""

    enabled: bool = False
    connection_role: ConnectionRole = ConnectionRole.PRIMARY
    connection_url: str | None = None
    chunk_size: int = 1000
    table_name: str = "ml_cross_asset_betas"
    upsert: bool = True


def persist_cross_asset_betas(
    exposures: pl.DataFrame,
    config: CrossAssetBetaPersistenceConfig,
    *,
    metrics: MetricsManager | None = None,
) -> int:
    """
    Persist exposure rows into ``ml_cross_asset_betas``.

    Parameters
    ----------
    exposures
        Long-form DataFrame with EWMA betas produced by
        :func:`playground.exposure.factor_exposure.compute_factor_exposures`.
    config
        Persistence configuration (connection resolution, chunk size, table name).
    metrics
        Optional metrics manager used for observability counters.

    Returns
    -------
    int
        Number of rows persisted (best effort). Returns ``0`` when disabled or on
        recoverable failures.
    """
    if not config.enabled:
        return 0

    if exposures.is_empty():
        return 0

    required = {
        "feature_set_id",
        "asset_id",
        "benchmark_id",
        "ts_event",
        "ts_init",
        "ewma_beta",
        "ewma_cov",
        "ewma_var_market",
        "n_observations",
        "alpha",
        "source",
    }
    if not required.issubset(exposures.columns):
        missing = required.difference(exposures.columns)
        raise ValueError(f"Missing required exposure columns: {sorted(missing)}")

    metrics_manager = metrics or MetricsManager.default()

    connection_url = _resolve_connection(config)
    if connection_url is None:
        metrics_manager.inc(
            "playground_cross_asset_beta_persist_total",
            "Total persistence attempts for cross-asset betas",
            labels={"status": "connection_unavailable"},
        )
        return 0

    try:
        engine = EngineManager.get_engine(connection_url)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to acquire engine for beta persistence", error=str(exc))
        metrics_manager.inc(
            "playground_cross_asset_beta_persist_total",
            "Total persistence attempts for cross-asset betas",
            labels={"status": "engine_error"},
        )
        return 0

    try:
        table = _reflect_table(engine, config.table_name)
    except NoSuchTableError:
        LOGGER.warning(
            "Beta persistence skipped because table is absent",
            table=config.table_name,
        )
        metrics_manager.inc(
            "playground_cross_asset_beta_persist_total",
            "Total persistence attempts for cross-asset betas",
            labels={"status": "missing_table"},
        )
        return 0

    rows = _normalise_records(exposures, required)
    if not rows:
        return 0

    persisted = _bulk_upsert(rows, engine, table, config)

    metrics_manager.inc(
        "playground_cross_asset_beta_persist_total",
        "Total persistence attempts for cross-asset betas",
        labels={"status": "success" if persisted else "no_rows"},
    )
    if persisted:
        metrics_manager.observe(
            "playground_cross_asset_beta_persist_rows",
            "Rows written to ml_cross_asset_betas",
            value=float(persisted),
            labels={"table": config.table_name},
        )
    return persisted


def _resolve_connection(config: CrossAssetBetaPersistenceConfig) -> str | None:
    if config.connection_url:
        return config.connection_url

    candidates = collect_postgres_candidates(config.connection_role)
    if not candidates.urls:
        LOGGER.debug("No connection candidates available for beta persistence")
        return None

    try:
        return select_first_working_connection(candidates.urls)
    except RuntimeError:
        LOGGER.debug("Unable to resolve working connection for beta persistence")
        return None


def _reflect_table(engine: Engine, table_name: str) -> Table:
    metadata = MetaData()
    return Table(table_name, metadata, autoload_with=engine)


def _normalise_records(
    frame: pl.DataFrame,
    columns: Iterable[str],
) -> list[dict[str, Any]]:
    selected = frame.select(sorted(columns))
    records: list[dict[str, Any]] = []
    for row in selected.iter_rows(named=True):
        records.append(
            {
                "feature_set_id": str(row["feature_set_id"]),
                "asset_id": str(row["asset_id"]),
                "benchmark_id": str(row["benchmark_id"]),
                "ts_event": int(row["ts_event"]),
                "ts_init": int(row["ts_init"]),
                "ewma_beta": float(row["ewma_beta"]),
                "ewma_cov": float(row["ewma_cov"]),
                "ewma_var_market": float(row["ewma_var_market"]),
                "n_observations": int(row["n_observations"]),
                "alpha": float(row["alpha"]),
                "source": str(row["source"]),
            },
        )
    return records


def _bulk_upsert(
    rows: list[dict[str, Any]],
    engine: Engine,
    table: Table,
    config: CrossAssetBetaPersistenceConfig,
) -> int:
    if not rows:
        return 0

    chunk_size = max(config.chunk_size, 1)
    persisted = 0

    dialect = engine.dialect.name
    use_upsert = config.upsert and dialect == "postgresql"

    with engine.begin() as conn:
        if use_upsert:
            stmt = pg_insert(table)
            stmt = stmt.on_conflict_do_update(
                index_elements=[
                    table.c.feature_set_id,
                    table.c.asset_id,
                    table.c.benchmark_id,
                    table.c.ts_event,
                ],
                set_={
                    "ts_init": stmt.excluded.ts_init,
                    "ewma_beta": stmt.excluded.ewma_beta,
                    "ewma_cov": stmt.excluded.ewma_cov,
                    "ewma_var_market": stmt.excluded.ewma_var_market,
                    "n_observations": stmt.excluded.n_observations,
                    "alpha": stmt.excluded.alpha,
                    "source": stmt.excluded.source,
                },
            )
            for chunk in _chunk(rows, chunk_size):
                conn.execute(stmt, chunk)
                persisted += len(chunk)
        else:
            stmt = table.insert()
            for chunk in _chunk(rows, chunk_size):
                try:
                    conn.execute(stmt, chunk)
                    persisted += len(chunk)
                except SQLAlchemyError:
                    LOGGER.warning("Failed to insert beta chunk", chunk_size=len(chunk))
                    raise

    return persisted


def _chunk(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


__all__ = ["CrossAssetBetaPersistenceConfig", "persist_cross_asset_betas"]
