"""
Async Observability DB persistor (optional; off hot-path).

Provides an optional asyncio-based adapter to persist observability DataFrames to a
SQL database using SQLAlchemy's async engine. This integrates with the
ObservabilityAsyncWorker when configured to use an async DB sink.

Notes
-----
- Requires an async SQLAlchemy driver (e.g., ``sqlite+aiosqlite://``, ``postgresql+asyncpg://``).
- Falls back gracefully if async engine creation fails at runtime (tests may skip).

Example
-------
>>> import pandas as pd
>>> from ml.observability.async_db_persistence import ObservabilityAsyncDBPersistor
>>> per = ObservabilityAsyncDBPersistor(connection_string="sqlite+aiosqlite:///./obs_async.db")
>>> # In async context: await per.persist_async({"metrics": pd.DataFrame({...})})

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class ObservabilityAsyncDBPersistor:
    """
    Persist observability DataFrames to SQL using SQLAlchemy async engine.

    Parameters
    ----------
    connection_string : str
        SQLAlchemy async database URL (e.g., ``sqlite+aiosqlite:///path.db`` or
        ``postgresql+asyncpg://user:pass@host/db``).

    """

    connection_string: str

    async def persist_async(self, tables: Mapping[str, pd.DataFrame | None]) -> dict[str, int]:
        """
        Persist non-empty DataFrames to their corresponding tables asynchronously.

        Supported keys: latency, metrics, correlation, health.
        Returns mapping of table name to row count inserted.

        This method uses ``AsyncConnection.run_sync`` to bridge pandas ``to_sql``
        (synchronous) with SQLAlchemy async drivers.

        """
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("SQLAlchemy async engine not available") from e

        written: dict[str, int] = {}
        engine = create_async_engine(self.connection_string)
        async with engine.begin() as conn:
            df_lat = tables.get("latency")
            if df_lat is not None and not df_lat.empty:
                await conn.run_sync(
                    lambda sync: df_lat.to_sql(
                        "obs_latency_watermarks",
                        sync,
                        if_exists="append",
                        index=False,
                        method="multi",
                    ),
                )
                written["latency"] = len(df_lat)
            df_met = tables.get("metrics")
            if df_met is not None and not df_met.empty:
                await conn.run_sync(
                    lambda sync: df_met.to_sql(
                        "obs_metrics",
                        sync,
                        if_exists="append",
                        index=False,
                        method="multi",
                    ),
                )
                written["metrics"] = len(df_met)
            df_cor = tables.get("correlation")
            if df_cor is not None and not df_cor.empty:
                await conn.run_sync(
                    lambda sync: df_cor.to_sql(
                        "obs_event_correlation",
                        sync,
                        if_exists="append",
                        index=False,
                        method="multi",
                    ),
                )
                written["correlation"] = len(df_cor)
            df_hea = tables.get("health")
            if df_hea is not None and not df_hea.empty:
                await conn.run_sync(
                    lambda sync: df_hea.to_sql(
                        "obs_health_scores",
                        sync,
                        if_exists="append",
                        index=False,
                        method="multi",
                    ),
                )
                written["health"] = len(df_hea)
        await engine.dispose()
        return written


__all__ = ["ObservabilityAsyncDBPersistor"]
