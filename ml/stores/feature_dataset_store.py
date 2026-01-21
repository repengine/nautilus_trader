"""
SQL-backed store for feature-adjacent datasets (macro/events/micro/L2).

This store centralizes the upsert logic for canonical dataset tables used by
Full Dataset Readiness. DataFrames produced by ingestion flows are normalized
and written via ``INSERT .. ON CONFLICT DO UPDATE`` statements so callers can
use :meth:`ml.stores.data_store.DataStore.write_ingestion` without bespoke SQL.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from sqlalchemy import BIGINT
from sqlalchemy import VARCHAR
from sqlalchemy import Column
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from ml.common.db_utils import get_or_create_engine
from ml.common.timestamps import sanitize_timestamp_ns
from ml.ml_types import DataFrameLike


class FeatureDatasetStore:
    """
    Persist canonical macro/events/micro/L2 datasets to PostgreSQL tables.
    """

    def __init__(self, connection_string: str, *, schema: str = "ml") -> None:
        self._engine: Engine = get_or_create_engine(connection_string)
        self._schema = schema
        self._metadata = MetaData(schema=schema)
        self._macro_release_table = self._define_macro_release_table()
        self._macro_observation_table = self._define_macro_observation_table()
        self._events_table = self._define_events_table()
        self._micro_table = self._define_micro_table()
        self._l2_table = self._define_l2_table()

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#

    def write_macro_releases(self, frame: DataFrameLike) -> int:
        """
        Upsert macro release calendar rows.
        """
        records = self._frame_to_records(frame)
        processed: list[dict[str, Any]] = []
        ts_init_default = sanitize_timestamp_ns(time.time_ns(), context="feature_store:macro_release")
        for record in records:
            release_ts = self._coerce_int(record.get("release_ts"))
            observation_ts = self._coerce_int(record.get("observation_ts"))
            if release_ts is None or observation_ts is None:
                continue
            processed.append(
                {
                    "series_id": self._coerce_str(record.get("series_id")),
                    "observation_ts": observation_ts,
                    "release_ts": release_ts,
                    "release_end_ts": self._coerce_int(record.get("release_end_ts")),
                    "value": self._coerce_float(record.get("value")),
                    "ts_event": self._coerce_int(record.get("ts_event"), default=release_ts),
                    "ts_init": self._coerce_int(record.get("ts_init"), default=ts_init_default),
                    "source": self._coerce_str(record.get("source")),
                    "run_id": self._coerce_str(record.get("run_id")),
                },
            )
        return self._bulk_upsert(
            table=self._macro_release_table,
            records=processed,
            conflict_cols=("series_id", "observation_ts", "release_ts", "ts_event"),
        )

    def write_macro_observations(self, frame: DataFrameLike) -> int:
        """
        Upsert macro observation rows (long format).
        """
        records = self._frame_to_records(frame)
        processed: list[dict[str, Any]] = []
        ts_init_default = sanitize_timestamp_ns(time.time_ns(), context="feature_store:macro_obs")
        for record in records:
            observation_ts = self._coerce_int(record.get("observation_ts"))
            if observation_ts is None:
                continue
            processed.append(
                {
                    "series_id": self._coerce_str(record.get("series_id")),
                    "observation_ts": observation_ts,
                    "value": self._coerce_float(record.get("value")),
                    "ts_event": self._coerce_int(record.get("ts_event"), default=observation_ts),
                    "ts_init": self._coerce_int(record.get("ts_init"), default=ts_init_default),
                    "source": self._coerce_str(record.get("source")),
                    "run_id": self._coerce_str(record.get("run_id")),
                },
            )
        return self._bulk_upsert(
            table=self._macro_observation_table,
            records=processed,
            conflict_cols=("series_id", "observation_ts", "ts_event"),
        )

    def write_events_calendar(self, frame: DataFrameLike) -> int:
        """
        Upsert normalized events calendar rows.
        """
        records = self._frame_to_records(frame)
        processed: list[dict[str, Any]] = []
        ts_init_default = sanitize_timestamp_ns(time.time_ns(), context="feature_store:events")
        for record in records:
            event_ts = self._coerce_int(record.get("event_timestamp"))
            event_type = self._coerce_str(record.get("event_type"))
            name = self._coerce_str(record.get("name"))
            instrument_id = self._coerce_str(record.get("instrument_id"))
            if event_ts is None or not event_type or not name:
                continue
            processed.append(
                {
                    "event_timestamp": event_ts,
                    "event_type": event_type,
                    "name": name,
                    "instrument_id": instrument_id or "",
                    "importance": self._coerce_str(record.get("importance")),
                    "source": self._coerce_str(record.get("source")),
                    "metadata": self._coerce_json(record.get("metadata")),
                    "ts_event": self._coerce_int(record.get("ts_event"), default=event_ts),
                    "ts_init": self._coerce_int(record.get("ts_init"), default=ts_init_default),
                },
            )
        return self._bulk_upsert(
            table=self._events_table,
            records=processed,
            conflict_cols=("event_type", "event_timestamp", "instrument_id", "name", "ts_event"),
        )

    def write_micro_features(self, frame: DataFrameLike) -> int:
        """
        Upsert microstructure per-minute features.
        """
        return self._write_time_series_features(
            table=self._micro_table,
            frame=frame,
            conflict_cols=("instrument_id", "timestamp", "ts_event"),
        )

    def write_l2_features(self, frame: DataFrameLike) -> int:
        """
        Upsert L2 depth per-minute features.
        """
        return self._write_time_series_features(
            table=self._l2_table,
            frame=frame,
            conflict_cols=("instrument_id", "timestamp", "ts_event"),
        )

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#

    def _define_macro_release_table(self) -> Table:
        return Table(
            "macro_release_calendar",
            self._metadata,
            Column("series_id", VARCHAR(64), primary_key=True),
            Column("observation_ts", BIGINT, primary_key=True),
            Column("release_ts", BIGINT, primary_key=True),
            Column("release_end_ts", BIGINT),
            Column("value", DOUBLE_PRECISION),
            Column("ts_event", BIGINT, nullable=False),
            Column("ts_init", BIGINT, nullable=False),
            Column("source", VARCHAR(32)),
            Column("run_id", VARCHAR(64)),
            schema=self._schema,
        )

    def _define_macro_observation_table(self) -> Table:
        return Table(
            "macro_observations",
            self._metadata,
            Column("series_id", VARCHAR(64), primary_key=True),
            Column("observation_ts", BIGINT, primary_key=True),
            Column("value", DOUBLE_PRECISION),
            Column("ts_event", BIGINT, nullable=False),
            Column("ts_init", BIGINT, nullable=False),
            Column("source", VARCHAR(32)),
            Column("run_id", VARCHAR(64)),
            schema=self._schema,
        )

    def _define_events_table(self) -> Table:
        return Table(
            "events_calendar",
            self._metadata,
            Column("event_timestamp", BIGINT, primary_key=True),
            Column("event_type", VARCHAR(64), primary_key=True),
            Column("instrument_id", VARCHAR(64), primary_key=True),
            Column("name", VARCHAR(255), primary_key=True),
            Column("importance", VARCHAR(32)),
            Column("source", VARCHAR(64)),
            Column("metadata", JSONB),
            Column("ts_event", BIGINT, nullable=False),
            Column("ts_init", BIGINT, nullable=False),
            schema=self._schema,
        )

    def _define_micro_table(self) -> Table:
        return Table(
            "microstructure_minute",
            self._metadata,
            Column("instrument_id", VARCHAR(32), primary_key=True),
            Column("timestamp", BIGINT, primary_key=True),
            Column("ts_event", BIGINT, nullable=False),
            Column("ts_init", BIGINT, nullable=False),
            Column("midprice", DOUBLE_PRECISION),
            Column("spread_bps", DOUBLE_PRECISION),
            Column("quote_imbalance", DOUBLE_PRECISION),
            Column("trade_imbalance", DOUBLE_PRECISION),
            Column("realized_vol", DOUBLE_PRECISION),
            schema=self._schema,
        )

    def _define_l2_table(self) -> Table:
        columns: list[Column[Any]] = [
            Column("instrument_id", VARCHAR(32), primary_key=True),
            Column("timestamp", BIGINT, primary_key=True),
            Column("ts_event", BIGINT, nullable=False),
            Column("ts_init", BIGINT, nullable=False),
            Column("midprice", DOUBLE_PRECISION),
            Column("spread_bps", DOUBLE_PRECISION),
            Column("microprice_bps", DOUBLE_PRECISION),
        ]
        for prefix in ("depth_imbalance", "dwp_bps", "bid_slope", "ask_slope"):
            for level in (1, 3, 5, 10):
                columns.append(Column(f"{prefix}_top{level}", DOUBLE_PRECISION))

        column_defs: tuple[Column[Any], ...] = tuple(columns)
        return Table(
            "l2_minute",
            self._metadata,
            *column_defs,
            schema=self._schema,
        )

    def _write_time_series_features(
        self,
        *,
        table: Table,
        frame: DataFrameLike,
        conflict_cols: Sequence[str],
    ) -> int:
        records = self._frame_to_records(frame)
        processed: list[dict[str, Any]] = []
        ts_init_default = sanitize_timestamp_ns(time.time_ns(), context=f"feature_store:{table.name}")
        allowed_cols = {column.name for column in table.columns}
        for record in records:
            timestamp = self._coerce_int(record.get("timestamp"))
            instrument_id = self._coerce_str(record.get("instrument_id"))
            if timestamp is None or not instrument_id:
                continue
            normalized = {
                key: record.get(key)
                for key in allowed_cols
                if key in record
            }
            normalized["instrument_id"] = instrument_id
            normalized["timestamp"] = timestamp
            normalized["ts_event"] = self._coerce_int(record.get("ts_event"), default=timestamp)
            normalized["ts_init"] = self._coerce_int(record.get("ts_init"), default=ts_init_default)
            processed.append(normalized)
        return self._bulk_upsert(table=table, records=processed, conflict_cols=tuple(conflict_cols))

    def _bulk_upsert(
        self,
        *,
        table: Table,
        records: list[dict[str, Any]],
        conflict_cols: Sequence[str],
    ) -> int:
        if not records:
            return 0
        stmt = pg_insert(table).values(records)
        update_columns = {
            column.name: getattr(stmt.excluded, column.name)
            for column in table.columns
            if column.name not in conflict_cols
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[table.c[name] for name in conflict_cols],
            set_=update_columns,
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)
        return len(records)

    @staticmethod
    def _frame_to_records(data: DataFrameLike) -> list[dict[str, Any]]:
        to_dicts = getattr(data, "to_dicts", None)
        if callable(to_dicts):
            return [dict(row) for row in to_dicts()]  # polars
        to_dict = getattr(data, "to_dict", None)
        if callable(to_dict):
            try:
                return [dict(row) for row in to_dict("records")]  # pandas
            except TypeError:
                pass
        if isinstance(data, list):
            return [dict(row) for row in data]
        raise TypeError(f"Unsupported frame type {type(data)} for feature dataset ingestion")

    @staticmethod
    def _coerce_int(value: Any, *, default: int | None = None) -> int | None:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int,)):
            return int(value)
        if isinstance(value, float):
            return int(value)
        text = str(value).strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _coerce_str(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _coerce_json(value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except Exception:
            return value
