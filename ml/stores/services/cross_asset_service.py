"""
CrossAssetFeatureService for cross-asset relationship features.

Provides typed, testable service for persisting and retrieving cross-asset
relationship metrics (beta, spreads, correlations) following Protocol-First
Interface Design (Pattern 2).

Uses the existing `ml_feature_values` table with namespaced feature_set_id:
- Beta: `cross_asset:beta:{asset_id}:{benchmark_id}`
- Spread: `cross_asset:spread:{asset_1_id}:{asset_2_id}`
- Correlation: `cross_asset:correlation:{asset_1_id}:{asset_2_id}`

All writes use PostgreSQL upsert for idempotent operations.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert


if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.engine import Engine


logger = logging.getLogger(__name__)


__all__ = [
    "CrossAssetFeatureService",
]


class _CrossAssetDeps(Protocol):
    """Minimal facade contract for cross-asset operations."""

    engine: Engine
    feature_values_table: Table


@dataclass(slots=True)
class CrossAssetFeatureService:
    """
    Service for cross-asset relationship feature storage and retrieval.

    Provides methods for persisting beta, spread, and correlation values
    between asset pairs using the existing ml_feature_values table.

    All operations use namespaced feature_set_id to prevent collisions
    between different relationship types.

    Example:
        >>> store = FeatureStoreFacade(connection_string="postgresql://...")
        >>> service = store.cross_asset
        >>> service.write_beta(
        ...     asset_id="AAPL",
        ...     benchmark_id="SPY",
        ...     ts_event=1234567890000000000,
        ...     ts_init=1234567890000000000,
        ...     beta=1.25,
        ...     lookback_periods=60,
        ...     ewma_span=30,
        ... )

    """

    deps: _CrossAssetDeps

    # =========================================================================
    # Beta Operations
    # =========================================================================

    def write_beta(
        self,
        *,
        asset_id: str,
        benchmark_id: str,
        ts_event: int,
        ts_init: int,
        beta: float,
        lookback_periods: int,
        ewma_span: int,
    ) -> None:
        """
        Write beta value for an asset relative to a benchmark.

        Uses PostgreSQL upsert to handle conflicts on (feature_set_id, instrument_id, ts_event).

        Parameters
        ----------
        asset_id : str
            Primary asset identifier (e.g., "AAPL.DATABENTO").
        benchmark_id : str
            Benchmark identifier (e.g., "SPY.DATABENTO").
        ts_event : int
            Event timestamp in nanoseconds.
        ts_init : int
            Initialization timestamp in nanoseconds.
        beta : float
            Computed beta value.
        lookback_periods : int
            Number of periods used for beta calculation.
        ewma_span : int
            EWMA span parameter for beta calculation.

        """
        feature_set_id = f"cross_asset:beta:{asset_id}:{benchmark_id}"

        values_json = {
            "beta": beta,
            "lookback_periods": lookback_periods,
            "ewma_span": ewma_span,
            "benchmark_id": benchmark_id,
        }

        self._upsert_row(
            feature_set_id=feature_set_id,
            instrument_id=asset_id,
            ts_event=ts_event,
            ts_init=ts_init,
            values=values_json,
        )

    def get_beta_history(
        self,
        *,
        asset_id: str,
        benchmark_id: str,
        start_ts: int,
        end_ts: int,
    ) -> list[dict[str, Any]]:
        """
        Retrieve beta history for an asset/benchmark pair within a time range.

        Time range is half-open: [start_ts, end_ts).

        Parameters
        ----------
        asset_id : str
            Primary asset identifier.
        benchmark_id : str
            Benchmark identifier.
        start_ts : int
            Start timestamp in nanoseconds (inclusive).
        end_ts : int
            End timestamp in nanoseconds (exclusive).

        Returns
        -------
        list[dict[str, Any]]
            List of beta records with ts_event, beta, lookback_periods, ewma_span.
            Ordered by ts_event ascending.

        """
        feature_set_id = f"cross_asset:beta:{asset_id}:{benchmark_id}"
        return self._read_history(feature_set_id, start_ts, end_ts)

    # =========================================================================
    # Spread Operations
    # =========================================================================

    def write_spread(
        self,
        *,
        asset_1_id: str,
        asset_2_id: str,
        ts_event: int,
        ts_init: int,
        z_score: float,
        spread_value: float,
        lookback_periods: int,
    ) -> None:
        """
        Write spread values between two assets.

        Uses PostgreSQL upsert to handle conflicts on (feature_set_id, instrument_id, ts_event).

        Parameters
        ----------
        asset_1_id : str
            Primary asset identifier.
        asset_2_id : str
            Secondary asset identifier.
        ts_event : int
            Event timestamp in nanoseconds.
        ts_init : int
            Initialization timestamp in nanoseconds.
        z_score : float
            Z-score of the spread.
        spread_value : float
            Raw spread value.
        lookback_periods : int
            Number of periods used for calculation.

        """
        feature_set_id = f"cross_asset:spread:{asset_1_id}:{asset_2_id}"

        values_json = {
            "z_score": z_score,
            "spread_value": spread_value,
            "lookback_periods": lookback_periods,
            "asset_2_id": asset_2_id,
        }

        self._upsert_row(
            feature_set_id=feature_set_id,
            instrument_id=asset_1_id,
            ts_event=ts_event,
            ts_init=ts_init,
            values=values_json,
        )

    # =========================================================================
    # Correlation Operations
    # =========================================================================

    def write_correlation(
        self,
        *,
        asset_1_id: str,
        asset_2_id: str,
        ts_event: int,
        ts_init: int,
        correlation: float,
        lookback_periods: int,
    ) -> None:
        """
        Write correlation value between two assets.

        Uses PostgreSQL upsert to handle conflicts on (feature_set_id, instrument_id, ts_event).

        Parameters
        ----------
        asset_1_id : str
            Primary asset identifier.
        asset_2_id : str
            Secondary asset identifier.
        ts_event : int
            Event timestamp in nanoseconds.
        ts_init : int
            Initialization timestamp in nanoseconds.
        correlation : float
            Correlation coefficient.
        lookback_periods : int
            Number of periods used for calculation.

        """
        feature_set_id = f"cross_asset:correlation:{asset_1_id}:{asset_2_id}"

        values_json = {
            "correlation": correlation,
            "lookback_periods": lookback_periods,
            "asset_2_id": asset_2_id,
        }

        self._upsert_row(
            feature_set_id=feature_set_id,
            instrument_id=asset_1_id,
            ts_event=ts_event,
            ts_init=ts_init,
            values=values_json,
        )

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _upsert_row(
        self,
        *,
        feature_set_id: str,
        instrument_id: str,
        ts_event: int,
        ts_init: int,
        values: dict[str, Any],
    ) -> None:
        """
        Upsert a row to ml_feature_values with conflict handling.

        Conflict keys: (feature_set_id, instrument_id, ts_event).
        On conflict: updates values and ts_init.

        """
        table = self.deps.feature_values_table
        engine = self.deps.engine

        row = {
            "feature_set_id": feature_set_id,
            "instrument_id": instrument_id,
            "ts_event": ts_event,
            "ts_init": ts_init,
            "values": values,
        }

        stmt = insert(table).values(row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["feature_set_id", "instrument_id", "ts_event"],
            set_={
                "values": stmt.excluded["values"],
                "ts_init": stmt.excluded.ts_init,
            },
        )

        try:
            with engine.begin() as conn:
                conn.execute(stmt)
        except Exception:
            logger.exception(
                "Failed to upsert cross-asset feature",
                extra={
                    "feature_set_id": feature_set_id,
                    "instrument_id": instrument_id,
                    "ts_event": ts_event,
                },
            )
            raise

    def _read_history(
        self,
        feature_set_id: str,
        start_ts: int,
        end_ts: int,
    ) -> list[dict[str, Any]]:
        """
        Read history for a feature_set_id within time range [start_ts, end_ts).

        Returns list of dicts with ts_event and flattened values.

        """
        table = self.deps.feature_values_table
        engine = self.deps.engine

        stmt = (
            select(
                table.c.ts_event,
                table.c["values"],
            )
            .where(table.c.feature_set_id == feature_set_id)
            .where(table.c.ts_event >= start_ts)
            .where(table.c.ts_event < end_ts)
            .order_by(table.c.ts_event)
        )

        results: list[dict[str, Any]] = []
        try:
            with engine.connect() as conn:
                rows = conn.execute(stmt).fetchall()
                for row in rows:
                    record: dict[str, Any] = {"ts_event": row[0]}
                    # Merge values dict into record
                    values_data = row[1]
                    if isinstance(values_data, dict):
                        record.update(values_data)
                    results.append(record)
        except Exception:
            logger.exception(
                "Failed to read cross-asset history",
                extra={
                    "feature_set_id": feature_set_id,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                },
            )
            raise

        return results
