"""
FeatureStore service layer.

Extracts typed, testable services from the FeatureStore facade while preserving
the public API and behavior. Services are dependency-injected with small
protocols that the facade already satisfies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class _QueryDeps(Protocol):
    """Minimal facade contract for read operations."""

    engine: Any


@dataclass(slots=True)
class FeatureQueryService:
    """Read/query operations for feature values."""

    deps: _QueryDeps

    def read_range(
        self,
        *,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> pd.DataFrame:
        # Local import to avoid pandas at import time
        import pandas as pd
        from sqlalchemy import bindparam as _bind
        from sqlalchemy import column as _column
        from sqlalchemy import select as _select
        from sqlalchemy import table as _table

        params: dict[str, Any] = {"start_ns": int(start_ns), "end_ns": int(end_ns)}
        if instrument_id is not None:
            params["instrument_id"] = instrument_id

        engine = self.deps.engine
        # Keep compatibility with sqlite used in some tests
        table_name = "ml_feature_values" if engine.dialect.name == "sqlite" else "public.ml_feature_values"

        feature_table = _table(
            table_name,
            _column("feature_set_id"),
            _column("instrument_id"),
            _column("values"),
            _column("ts_event"),
            _column("ts_init"),
        )

        condition = (feature_table.c.ts_event >= _bind("start_ns")) & (
            feature_table.c.ts_event < _bind("end_ns")
        )
        if instrument_id is not None:
            condition = condition & (feature_table.c.instrument_id == _bind("instrument_id"))

        query = (
            _select(
                feature_table.c.feature_set_id,
                feature_table.c.instrument_id,
                feature_table.c["values"],
                feature_table.c.ts_event,
                feature_table.c.ts_init,
            )
            .where(condition)
            .order_by(feature_table.c.ts_event)
        )
        sql = query

        with engine.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)


class _ClearDeps(Protocol):
    """Minimal facade contract for clear/delete operations."""

    engine: Any
    feature_values_table: Any


@dataclass(slots=True)
class FeatureClearService:
    """Deletion/cleanup operations for feature values."""

    deps: _ClearDeps

    def clear(self, *, instrument_id: str | None = None, feature_version: str | None = None) -> None:
        with self.deps.engine.begin() as conn:
            delete_stmt = self.deps.feature_values_table.delete()

            if instrument_id is not None:
                delete_stmt = delete_stmt.where(self.deps.feature_values_table.c.instrument_id == instrument_id)

            if feature_version is not None:
                delete_stmt = delete_stmt.where(
                    self.deps.feature_values_table.c.feature_version == feature_version,
                )

            conn.execute(delete_stmt)


class _CrossAssetDeps(Protocol):
    """Minimal facade contract for cross-asset feature operations."""

    engine: Any
    feature_values_table: Any


@dataclass(slots=True)
class CrossAssetFeatureService:
    """
    Service for cross-asset relationship features (beta, spreads, correlations).

    Uses FeatureStore's existing infrastructure with namespaced feature_set_ids:
    - "cross_asset:beta:{asset_id}:{benchmark_id}"
    - "cross_asset:spread:{asset1_id}:{asset2_id}"
    - "cross_asset:correlation:{asset1_id}:{asset2_id}"

    This service enables storage and retrieval of cross-asset metrics using the
    existing ml_feature_values table, avoiding schema changes while maintaining
    type safety through namespace conventions.

    Example:
        >>> from ml.stores.feature_store import ComponentFeatureStore
        >>> store = ComponentFeatureStore(connection_string="postgresql://...")
        >>> store.cross_asset.write_beta(
        ...     asset_id="AAPL",
        ...     benchmark_id="SPY",
        ...     ts_event=1234567890000000000,
        ...     ts_init=1234567890000000000,
        ...     beta=1.25,
        ...     lookback_periods=60,
        ...     ewma_span=30,
        ... )
        >>> history = store.cross_asset.get_beta_history(
        ...     asset_id="AAPL",
        ...     benchmark_id="SPY",
        ...     start_ts=1234567890000000000,
        ...     end_ts=1234567891000000000,
        ... )
    """

    deps: _CrossAssetDeps

    def write_beta(
        self,
        asset_id: str,
        benchmark_id: str,
        ts_event: int,
        ts_init: int,
        beta: float,
        lookback_periods: int,
        ewma_span: int,
    ) -> None:
        """
        Write beta value using FeatureStore's existing table.

        Stores beta as a feature set with namespaced ID for type safety.
        Uses upsert semantics to handle duplicate timestamps gracefully.

        Args:
            asset_id: Primary asset instrument ID
            benchmark_id: Benchmark instrument ID (e.g., "SPY")
            ts_event: Event timestamp in nanoseconds
            ts_init: Initialization timestamp in nanoseconds
            beta: Beta coefficient value
            lookback_periods: Number of periods used in calculation
            ewma_span: EWMA span parameter used in calculation

        Example:
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
        from sqlalchemy.dialects.postgresql import insert

        feature_set_id = f"cross_asset:beta:{asset_id}:{benchmark_id}"

        # Store as JSON with metadata
        values_dict = {
            "beta": float(beta),
            "lookback_periods": int(lookback_periods),
            "ewma_span": int(ewma_span),
        }

        stmt = insert(self.deps.feature_values_table).values(
            feature_set_id=feature_set_id,
            instrument_id=asset_id,  # Primary asset
            values=values_dict,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        # Upsert on conflict
        stmt = stmt.on_conflict_do_update(
            index_elements=["feature_set_id", "instrument_id", "ts_event"],
            set_={"values": stmt.excluded["values"], "ts_init": stmt.excluded.ts_init},
        )

        with self.deps.engine.begin() as conn:
            conn.execute(stmt)

    def get_beta_history(
        self,
        asset_id: str,
        benchmark_id: str,
        start_ts: int,
        end_ts: int,
    ) -> list[dict[str, Any]]:
        """
        Retrieve historical beta values within time range.

        Returns list of dicts containing ts_event, beta, lookback_periods, and ewma_span.
        Time range is half-open: [start_ts, end_ts) - inclusive start, exclusive end.

        Args:
            asset_id: Primary asset instrument ID
            benchmark_id: Benchmark instrument ID
            start_ts: Start timestamp in nanoseconds (inclusive)
            end_ts: End timestamp in nanoseconds (exclusive)

        Returns:
            List of dicts with keys: ts_event, beta, lookback_periods, ewma_span.
            Returns empty list if no data found.
            Results ordered by ts_event ascending.

        Example:
            >>> history = service.get_beta_history(
            ...     asset_id="AAPL",
            ...     benchmark_id="SPY",
            ...     start_ts=1234567890000000000,
            ...     end_ts=1234567891000000000,
            ... )
            >>> assert len(history) > 0
            >>> assert history[0]["beta"] == 1.25
        """
        from sqlalchemy import select

        feature_set_id = f"cross_asset:beta:{asset_id}:{benchmark_id}"

        stmt = (
            select(
                self.deps.feature_values_table.c.ts_event,
                self.deps.feature_values_table.c["values"],
            )
            .where(self.deps.feature_values_table.c.feature_set_id == feature_set_id)
            .where(self.deps.feature_values_table.c.ts_event >= start_ts)
            .where(self.deps.feature_values_table.c.ts_event < end_ts)
            .order_by(self.deps.feature_values_table.c.ts_event)
        )

        with self.deps.engine.connect() as conn:
            results = conn.execute(stmt).fetchall()
            normalized: list[dict[str, Any]] = []
            for row in results:
                mapping = getattr(row, "_mapping", None)
                if mapping is not None:
                    values_payload = mapping["values"]
                    ts_event = mapping["ts_event"]
                else:  # pragma: no cover - compatibility
                    ts_event = row[0]
                    values_payload = row[1]
                normalized.append(
                    {
                        "ts_event": ts_event,
                        **values_payload,  # Unpack JSON {beta, lookback_periods, ewma_span}
                    },
                )
            return normalized

    def write_spread(
        self,
        asset_1_id: str,
        asset_2_id: str,
        ts_event: int,
        ts_init: int,
        z_score: float,
        spread_value: float,
        lookback_periods: int,
    ) -> None:
        """
        Write z-scored spread value.

        Stores spread as a feature set with namespaced ID.
        Uses upsert semantics to handle duplicate timestamps.

        Args:
            asset_1_id: First asset instrument ID (primary)
            asset_2_id: Second asset instrument ID
            ts_event: Event timestamp in nanoseconds
            ts_init: Initialization timestamp in nanoseconds
            z_score: Z-score of the spread
            spread_value: Raw spread value
            lookback_periods: Number of periods used in calculation

        Example:
            >>> service.write_spread(
            ...     asset_1_id="AAPL",
            ...     asset_2_id="MSFT",
            ...     ts_event=1234567890000000000,
            ...     ts_init=1234567890000000000,
            ...     z_score=2.5,
            ...     spread_value=10.5,
            ...     lookback_periods=60,
            ... )
        """
        from sqlalchemy.dialects.postgresql import insert

        feature_set_id = f"cross_asset:spread:{asset_1_id}:{asset_2_id}"

        values_dict = {
            "z_score": float(z_score),
            "spread_value": float(spread_value),
            "lookback_periods": int(lookback_periods),
        }

        stmt = insert(self.deps.feature_values_table).values(
            feature_set_id=feature_set_id,
            instrument_id=asset_1_id,  # Primary asset
            values=values_dict,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=["feature_set_id", "instrument_id", "ts_event"],
            set_={"values": stmt.excluded["values"], "ts_init": stmt.excluded.ts_init},
        )

        with self.deps.engine.begin() as conn:
            conn.execute(stmt)

    def write_correlation(
        self,
        asset_1_id: str,
        asset_2_id: str,
        ts_event: int,
        ts_init: int,
        correlation: float,
        lookback_periods: int,
    ) -> None:
        """
        Write correlation value.

        Stores correlation as a feature set with namespaced ID.
        Uses upsert semantics to handle duplicate timestamps.

        Args:
            asset_1_id: First asset instrument ID (primary)
            asset_2_id: Second asset instrument ID
            ts_event: Event timestamp in nanoseconds
            ts_init: Initialization timestamp in nanoseconds
            correlation: Correlation coefficient (-1 to 1)
            lookback_periods: Number of periods used in calculation

        Example:
            >>> service.write_correlation(
            ...     asset_1_id="GOOGL",
            ...     asset_2_id="AMZN",
            ...     ts_event=1234567890000000000,
            ...     ts_init=1234567890000000000,
            ...     correlation=0.85,
            ...     lookback_periods=30,
            ... )
        """
        from sqlalchemy.dialects.postgresql import insert

        feature_set_id = f"cross_asset:correlation:{asset_1_id}:{asset_2_id}"

        values_dict = {
            "correlation": float(correlation),
            "lookback_periods": int(lookback_periods),
        }

        stmt = insert(self.deps.feature_values_table).values(
            feature_set_id=feature_set_id,
            instrument_id=asset_1_id,  # Primary asset
            values=values_dict,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=["feature_set_id", "instrument_id", "ts_event"],
            set_={"values": stmt.excluded["values"], "ts_init": stmt.excluded.ts_init},
        )

        with self.deps.engine.begin() as conn:
            conn.execute(stmt)
