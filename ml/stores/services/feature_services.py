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
        from sqlalchemy import text as _text

        where_parts: list[str] = ["ts_event >= :start_ns", "ts_event < :end_ns"]
        params: dict[str, Any] = {"start_ns": int(start_ns), "end_ns": int(end_ns)}
        if instrument_id is not None:
            where_parts.append("instrument_id = :instrument_id")
            params["instrument_id"] = instrument_id

        engine = self.deps.engine
        # Keep compatibility with sqlite used in some tests
        table_name = "ml_feature_values" if engine.dialect.name == "sqlite" else "public.ml_feature_values"
        sql = _text(
            f"""
                SELECT feature_set_id,
                       instrument_id,
                       values,
                       ts_event,
                       ts_init
                FROM {table_name}
                WHERE {' AND '.join(where_parts)}
                ORDER BY ts_event
                """,
        )

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
