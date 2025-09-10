from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import pandas as pd
from sqlalchemy import BIGINT, Column, MetaData, String, Table, text
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager
from ml.stores.protocols import CoverageProviderProtocol, MarketDataWriterProtocol


DAY_NS: Final[int] = 86_400_000_000_000


@dataclass(slots=True)
class SqlCoverageProvider(CoverageProviderProtocol):
    """
    SQL coverage provider using a canonical market data table.

    Parameters
    ----------
    connection_string : str
        SQLAlchemy connection string (Postgres recommended).
    table_name : str
        Name of the market data table (default: "market_data").
    ts_field : str
        Timestamp column name (default: "ts_event").
    """

    connection_string: str
    table_name: str = "market_data"
    ts_field: str = "ts_event"
    _engine: Engine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._engine = EngineManager.get_engine(self.connection_string)

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        q = text(
            f"""
            SELECT (({self.ts_field} / :day_ns)) AS bucket
            FROM {self.table_name}
            WHERE instrument_id = :instrument_id
              AND {self.ts_field} >= :start_ns AND {self.ts_field} < :end_ns
            GROUP BY bucket
            """,
        )
        params = {
            "day_ns": DAY_NS,
            "instrument_id": instrument_id,
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
        }
        with self._engine.connect() as conn:
            rows = conn.execute(q, params).fetchall()
        return {int(r[0]) for r in rows}


@dataclass(slots=True)
class SqlMarketDataWriter(MarketDataWriterProtocol):
    """
    SQL writer for canonical market data table.

    Inserts rows with (instrument_id, ts_event, ts_init) and ignores conflicts
    on primary key (instrument_id, ts_event). Other columns are left NULL.
    """

    connection_string: str
    table_name: str = "market_data"
    _engine: Engine = field(init=False, repr=False)
    _meta: MetaData = field(init=False, repr=False)
    _table: Table = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._engine = EngineManager.get_engine(self.connection_string)
        self._meta = MetaData()
        try:
            self._table = Table(self.table_name, self._meta, autoload_with=self._engine)
        except Exception:
            # Fallback table for non-Postgres tests (SQLite); minimal schema
            self._table = Table(
                self.table_name,
                self._meta,
                Column("instrument_id", String(100), nullable=False),
                Column("ts_event", BIGINT, nullable=False),
                Column("ts_init", BIGINT, nullable=False),
            )
            self._meta.create_all(self._engine)

    def write(self, *, dataset_id: str, schema: str, instrument_id: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        records = [
            {
                "instrument_id": instrument_id,
                "ts_event": int(r.ts_event),
                "ts_init": int(r.ts_event),
            }
            for r in df.itertuples(index=False)
        ]
        dialect = self._engine.dialect.name
        with self._engine.begin() as conn:
            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                stmt = pg_insert(self._table).values(records)
                stmt = stmt.on_conflict_do_nothing(index_elements=["instrument_id", "ts_event"])
                conn.execute(stmt)
            else:
                # SQLite: use OR IGNORE semantic
                conn.execute(self._table.insert().prefix_with("OR IGNORE"), records)
        return int(len(records))

