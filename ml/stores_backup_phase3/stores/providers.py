from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from sqlalchemy import BIGINT
from sqlalchemy import Column
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import bindparam
from sqlalchemy.engine import Engine
from sqlalchemy.sql import column as sa_column
from sqlalchemy.sql import func as sa_func
from sqlalchemy.sql import select as sa_select
from sqlalchemy.sql import table as sa_table
from sqlalchemy.sql.elements import ColumnClause

from ml.common.db_utils import get_or_create_engine
from ml.registry.dataclasses import DatasetType
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol
from ml.stores.raw_protocols import RawReaderProtocol
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


DAY_NS: Final[int] = 86_400_000_000_000
logger = logging.getLogger(__name__)


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(identifier: str, *, label: str) -> str:
    if not _IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Invalid SQL identifier for {label}: {identifier!r}")
    return identifier


def resolve_catalog_identifier(
    *,
    schema: str,
    instrument_id: str,
    identifier_template: str | None = None,
) -> str:
    """
    Resolve the catalog identifier used for parquet interval lookups.

    Bars typically store the bar_type (template) while tick datasets key on instrument_id.
    """
    normalized_schema = schema.lower()
    if "tbbo" in normalized_schema or "quote" in normalized_schema or "trade" in normalized_schema:
        return instrument_id
    if identifier_template:
        return identifier_template.format(instrument_id=instrument_id)
    return instrument_id


def _schema_to_dataclass(schema: str) -> type[Any]:
    s = schema.lower()
    if "bar" in s or "ohlcv" in s:
        return cast(type[Any], Bar)
    if "tbbo" in s or "quote" in s:
        return cast(type[Any], QuoteTick)
    if "trade" in s:
        return cast(type[Any], TradeTick)
    return cast(type[Any], Bar)


@dataclass(frozen=True, slots=True)
class SqlCoverageOverride:
    """
    Dataset-specific overrides for SQL coverage inspection.
    """

    table_name: str | None = None
    schema: str | None = None
    ts_field: str | None = None
    entity_field: str | None = None


@dataclass(slots=True)
class CatalogCoverageProvider(CoverageProviderProtocol):
    """
    Coverage provider backed by Nautilus ParquetDataCatalog.
    """

    catalog_path: str
    identifier_template: str | None = None

    def __post_init__(self) -> None:
        self._catalog = ParquetDataCatalog(self.catalog_path)

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
        entity_field: str = "instrument_id",
    ) -> set[int]:
        data_cls = _schema_to_dataclass(schema)
        identifier = resolve_catalog_identifier(
            schema=schema,
            instrument_id=instrument_id,
            identifier_template=self.identifier_template,
        )
        intervals = self._catalog.get_intervals(data_cls=data_cls, identifier=identifier)
        if not intervals:
            return set()
        buckets: set[int] = set()
        window_start = int(start_ns)
        window_end = int(end_ns)
        for s, e in intervals:
            s_clamped = max(int(s), window_start)
            e_clamped = min(int(e), window_end)
            if s_clamped >= e_clamped:
                continue
            start_bucket = s_clamped // DAY_NS
            end_bucket = (e_clamped - 1) // DAY_NS
            for b in range(int(start_bucket), int(end_bucket) + 1):
                buckets.add(b)
        return buckets


# =============================================================================
# SQL coverage + writer (from coverage_sql.py)
# =============================================================================


@dataclass(slots=True)
class SqlCoverageProvider(CoverageProviderProtocol):
    """
    SQL coverage provider using a canonical market data table.
    """

    connection_string: str
    table_name: str = "market_data"
    ts_field: str = "ts_event"
    dataset_overrides: Mapping[str, SqlCoverageOverride] | None = None
    _engine: Engine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.table_name, label="coverage.table")
        _validate_identifier(self.ts_field, label="coverage.ts_field")
        if self.dataset_overrides:
            for dataset_id, override in self.dataset_overrides.items():
                label_prefix = f"{dataset_id}.coverage"
                if override.table_name:
                    _validate_identifier(override.table_name, label=f"{label_prefix}.table")
                if override.ts_field:
                    _validate_identifier(override.ts_field, label=f"{label_prefix}.ts_field")
                if override.entity_field:
                    _validate_identifier(override.entity_field, label=f"{label_prefix}.entity_field")
                if override.schema:
                    _validate_identifier(override.schema, label=f"{label_prefix}.schema")
        self._engine = get_or_create_engine(self.connection_string)

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
        entity_field: str = "instrument_id",
    ) -> set[int]:
        override = self.dataset_overrides.get(dataset_id) if self.dataset_overrides else None
        table_name = override.table_name if override and override.table_name else self.table_name
        ts_field = override.ts_field if override and override.ts_field else self.ts_field
        entity_column = (
            entity_field
            if entity_field
            else (override.entity_field if override and override.entity_field else "instrument_id")
        )
        schema_name = override.schema if override and override.schema else None

        entity_col: ColumnClause[Any] = sa_column(entity_column)
        ts_col: ColumnClause[Any] = sa_column(ts_field)
        table = sa_table(
            table_name,
            entity_col,
            ts_col,
            schema=schema_name,
        )
        bucket_expr = sa_func.floor(ts_col / bindparam("day_ns")).label("bucket")
        stmt = (
            sa_select(bucket_expr)
            .select_from(table)
            .where(
                entity_col == bindparam("entity_id"),
                ts_col >= bindparam("start_ns"),
                ts_col < bindparam("end_ns"),
            )
            .group_by(bucket_expr)
        )
        params = {
            "day_ns": DAY_NS,
            "entity_id": instrument_id,
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
        }
        with self._engine.connect() as conn:
            rows = conn.execute(stmt, params).fetchall()
        return {int(r[0]) for r in rows}

    def latest_timestamp_ns(
        self,
        *,
        dataset_id: str | None,
        instrument_id: str,
        entity_field: str = "instrument_id",
    ) -> int | None:
        """
        Return the most recent ``ts_event`` recorded for ``instrument_id``.
        """
        override = None
        if dataset_id and self.dataset_overrides:
            override = self.dataset_overrides.get(dataset_id)
        table_name = override.table_name if override and override.table_name else self.table_name
        ts_field = override.ts_field if override and override.ts_field else self.ts_field
        schema_name = override.schema if override and override.schema else None
        entity_column = (
            entity_field
            if entity_field
            else (override.entity_field if override and override.entity_field else "instrument_id")
        )
        entity_col: ColumnClause[Any] = sa_column(entity_column)
        ts_col: ColumnClause[Any] = sa_column(ts_field)
        table = sa_table(
            table_name,
            entity_col,
            ts_col,
            schema=schema_name,
        )
        stmt = (
            sa_select(sa_func.max(ts_col))
            .select_from(table)
            .where(entity_col == bindparam("entity_id"))
        )
        with self._engine.connect() as conn:
            result = conn.execute(stmt, {"entity_id": instrument_id}).scalar()
        if result is None:
            return None
        return int(result)


@dataclass(frozen=True, slots=True)
class ParquetCoverageSpec:
    """
    Coverage metadata for datasets mirrored to partitioned Parquet files.
    """

    dataset_id: str
    base_path: Path
    partition_field: str = "instrument_id"
    timestamp_field: str = "ts_event"
    partition_template: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            msg = "dataset_id cannot be empty"
            raise ValueError(msg)
        if not self.base_path:
            msg = "base_path cannot be empty"
            raise ValueError(msg)
        if not self.partition_field:
            msg = "partition_field cannot be empty"
            raise ValueError(msg)
        if not self.timestamp_field:
            msg = "timestamp_field cannot be empty"
            raise ValueError(msg)

    def files_for_instrument(self, instrument_id: str) -> tuple[Path, ...]:
        """
        Resolve parquet files for an instrument respecting partition templates.
        """
        base_path = self.base_path
        if base_path.is_file():
            return (base_path,)
        template_value = self.partition_template
        if template_value is None:
            template_value = "{field}={value}"
        else:
            template_value = template_value.strip()
        partition_root = base_path
        if template_value:
            try:
                partition_path = template_value.format(
                    field=self.partition_field,
                    value=instrument_id.strip(),
                )
            except (KeyError, IndexError, ValueError) as exc:
                msg = f"Invalid partition_template for dataset {self.dataset_id}: {template_value!r}"
                raise ValueError(msg) from exc
            partition_root = base_path / partition_path
        if partition_root.is_file():
            return (partition_root,)
        if partition_root.is_dir():
            files = sorted(partition_root.glob("*.parquet"))
            return tuple(files)
        return tuple()


@dataclass(slots=True)
class PartitionedParquetCoverageProvider(CoverageProviderProtocol):
    """
    Coverage provider that inspects partitioned Parquet mirrors (e.g., earnings).
    """

    specs: Mapping[str, ParquetCoverageSpec]

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
        entity_field: str = "instrument_id",
    ) -> set[int]:
        spec = self.specs.get(dataset_id)
        if spec is None:
            return set()
        partition_field = spec.partition_field or entity_field
        instrument_value = instrument_id.strip()
        parquet_files = spec.files_for_instrument(instrument_value)
        if not parquet_files:
            logger.debug(
                "parquet_coverage.partition_missing",
                extra={
                    "dataset_id": dataset_id,
                    "schema": schema,
                    "instrument_id": instrument_value,
                    "base_path": str(spec.base_path),
                },
            )
            return set()
        buckets: set[int] = set()
        for parquet_file in parquet_files:
            buckets.update(
                self._buckets_from_file(
                    parquet_file,
                    timestamp_field=spec.timestamp_field,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    entity_field=partition_field,
                    instrument_id=instrument_value,
                ),
            )
        return buckets

    @staticmethod
    def _buckets_from_file(
        path: Path,
        *,
        timestamp_field: str,
        start_ns: int,
        end_ns: int,
        entity_field: str,
        instrument_id: str,
    ) -> set[int]:
        needs_entity = bool(entity_field and entity_field != timestamp_field)
        columns = [timestamp_field]
        if needs_entity:
            columns.append(entity_field)
        try:
            frame = pd.read_parquet(path, columns=columns)
            entity_available = entity_field in frame.columns if entity_field else False
        except Exception:
            if needs_entity:
                try:
                    frame = pd.read_parquet(path, columns=[timestamp_field])
                    entity_available = False
                except Exception:  # pragma: no cover - IO/backend issues
                    logger.warning("parquet_coverage.read_failed", exc_info=True, extra={"path": str(path)})
                    return set()
            else:  # pragma: no cover - IO/backend issues
                logger.warning("parquet_coverage.read_failed", exc_info=True, extra={"path": str(path)})
                return set()
        if timestamp_field not in frame:
            return set()
        filtered = frame
        if entity_field and entity_available and entity_field in frame.columns:
            instrument_value = instrument_id.strip()
            column = frame[entity_field].astype(str).str.strip()
            filtered = frame.loc[column == instrument_value]
        series = filtered[timestamp_field].dropna()
        if series.empty:
            return set()
        window = series[(series >= start_ns) & (series < end_ns)]
        if window.empty:
            return set()
        return {int(value // DAY_NS) for value in window.astype("int64").to_list()}


@dataclass(slots=True)
class UnionCoverageProvider(CoverageProviderProtocol):
    """
    Coverage provider that unions results from multiple backends.
    """

    providers: tuple[CoverageProviderProtocol, ...]

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
        entity_field: str = "instrument_id",
    ) -> set[int]:
        buckets: set[int] = set()
        for provider in self.providers:
            buckets.update(
                provider.read_bucket_coverage(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_id=instrument_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    entity_field=entity_field,
                ),
            )
        return buckets


@dataclass(slots=True)
class NullCoverageProvider(CoverageProviderProtocol):
    """
    No-op provider used when no catalog source is configured.
    """

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
        entity_field: str = "instrument_id",
    ) -> set[int]:
        return set()


@dataclass(slots=True)
class SqlMarketDataWriter(MarketDataWriterProtocol):
    """
    SQL writer for canonical market data table.

    Inserts (instrument_id, ts_event, ts_init) rows and ignores conflicts.

    """

    connection_string: str
    table_name: str = "market_data"
    default_source: str | None = "historical"
    _engine: Engine = field(init=False, repr=False)
    _meta: MetaData = field(init=False, repr=False)
    _table: Table = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.table_name, label="writer.table")
        self._engine = get_or_create_engine(self.connection_string)
        self._meta = MetaData()
        try:
            self._table = Table(self.table_name, self._meta, autoload_with=self._engine)
        except Exception:
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

        return self._write_standard(df, instrument_id)

    def _write_standard(self, df: pd.DataFrame, instrument_id: str) -> int:
        cols = set(map(str, df.columns))

        def _maybe(val: object) -> object | None:
            if val is None:
                return None
            try:
                import numpy as np

                if isinstance(val, np.generic):
                    scalar = val.item()
                    if isinstance(scalar, float) and math.isnan(scalar):
                        return None
                    return cast(object, scalar)
            except ModuleNotFoundError:  # pragma: no cover - numpy optional in tooling
                pass
            if isinstance(val, float):
                return None if math.isnan(val) else float(val)
            if hasattr(val, "item"):
                try:
                    scalar = val.item()
                except Exception:  # pragma: no cover - defensive fallback
                    scalar = val
                else:
                    if isinstance(scalar, float) and math.isnan(scalar):
                        return None
                    return cast(object, scalar)
            return val

        def _row(r: object) -> dict[str, object]:
            d: dict[str, object] = {
                "instrument_id": instrument_id,
                "ts_event": int(getattr(r, "ts_event")),
                "ts_init": int(getattr(r, "ts_event")),
            }
            if "open" in cols:
                d["open"] = _maybe(getattr(r, "open", None))
            if "high" in cols:
                d["high"] = _maybe(getattr(r, "high", None))
            if "low" in cols:
                d["low"] = _maybe(getattr(r, "low", None))
            if "close" in cols:
                d["close"] = _maybe(getattr(r, "close", None))
            if "volume" in cols:
                d["volume"] = _maybe(getattr(r, "volume", None))
            if "bid" in cols:
                d["bid"] = _maybe(getattr(r, "bid", None))
            if "ask" in cols:
                d["ask"] = _maybe(getattr(r, "ask", None))
            if "bid_size" in cols:
                d["bid_size"] = _maybe(getattr(r, "bid_size", None))
            if "ask_size" in cols:
                d["ask_size"] = _maybe(getattr(r, "ask_size", None))
            if "last" in cols:
                d["last"] = _maybe(getattr(r, "last", None))
            if "trade_count" in cols:
                d["trade_count"] = _maybe(getattr(r, "trade_count", None))
            if "vwap" in cols:
                d["vwap"] = _maybe(getattr(r, "vwap", None))
            if "quality_flags" in cols:
                d["quality_flags"] = _maybe(getattr(r, "quality_flags", None))
            if "source" in cols:
                d["source"] = _maybe(getattr(r, "source", None))
            elif self.default_source is not None:
                d["source"] = self.default_source
            if "source_dataset" in cols:
                d["source_dataset"] = _maybe(getattr(r, "source_dataset", None))
            return d

        records: list[dict[str, object]] = [_row(r) for r in df.itertuples(index=False)]
        dialect = self._engine.dialect.name
        with self._engine.begin() as conn:
            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                stmt = pg_insert(self._table).values(records)
                stmt = stmt.on_conflict_do_nothing(index_elements=["instrument_id", "ts_event"])
                conn.execute(stmt)
            else:
                conn.execute(self._table.insert().prefix_with("OR IGNORE"), records)
        return len(records)


@dataclass(slots=True)
class SqlMarketDataReader(RawReaderProtocol):
    """Read market data ranges from the canonical SQL store."""

    connection_string: str
    table_name: str = "market_data"
    _engine: Engine = field(init=False, repr=False)
    _available_columns: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.table_name, label="reader.table")
        self._engine = get_or_create_engine(self.connection_string)
        try:
            from sqlalchemy import inspect as _inspect

            inspector = _inspect(self._engine)
            columns = inspector.get_columns(self.table_name)
            self._available_columns = tuple(column["name"] for column in columns)
        except Exception:
            # Fallback to an empty tuple; read_range will surface a descriptive error.
            self._available_columns = ()

    def read_range(
        self,
        *,
        dataset_type: DatasetType,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> Any:
        if start_ns >= end_ns:
            raise ValueError(
                f"Invalid range for SQL market data read ({start_ns=} >= {end_ns=})",
            )

        available = set(self._available_columns)
        required_columns: tuple[str, ...] = ("instrument_id", "ts_event", "ts_init")
        missing_required = [column for column in required_columns if column not in available]
        if missing_required:
            raise RuntimeError(
                f"SqlMarketDataReader missing required columns {missing_required} on table {self.table_name}",
            )

        optional_columns: tuple[str, ...] = (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
            "last",
            "trade_count",
            "vwap",
            "source_dataset",
        )

        selected_columns = [*required_columns, *[col for col in optional_columns if col in available]]
        table = sa_table(
            self.table_name,
            *(sa_column(col) for col in selected_columns),
        )
        stmt = (
            sa_select(*(sa_column(col) for col in selected_columns))
            .select_from(table)
            .where(sa_column("instrument_id") == bindparam("instrument_id"))
            .where(sa_column("ts_event") >= bindparam("start_ns"))
            .where(sa_column("ts_event") < bindparam("end_ns"))
            .order_by(sa_column("ts_event"))
        )
        params: dict[str, int | str] = {
            "instrument_id": instrument_id,
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
        }

        with self._engine.connect() as conn:
            frame = pd.read_sql_query(stmt, conn, params=params)

        if frame.empty:
            return self._empty_frame()

        numeric_cols = [
            col
            for col in (
                "open",
                "high",
                "low",
                "close",
                "volume",
                "bid",
                "ask",
                "bid_size",
                "ask_size",
                "last",
                "trade_count",
                "vwap",
            )
            if col in frame.columns
        ]
        if numeric_cols:
            frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")

        frame = frame.rename(columns={"ts_event": "timestamp"})
        frame["instrument_id"] = frame["instrument_id"].astype(str)

        if "source_dataset" not in frame.columns:
            frame["source_dataset"] = None

        try:
            from ml._imports import HAS_POLARS
            from ml._imports import pl

            if not HAS_POLARS or pl is None:
                raise ImportError
        except ImportError:
            return frame

        if frame.empty:
            return pl.DataFrame(schema={col: frame.dtypes[col] for col in frame.columns})

        try:
            return pl.from_pandas(frame, include_index=False)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Failed to convert SQL market data frame to polars", exc_info=True)
            return frame

    def _empty_frame(self) -> Any:
        columns = [
            "instrument_id",
            "timestamp",
            "ts_init",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
            "last",
            "trade_count",
            "vwap",
            "source_dataset",
        ]
        empty = pd.DataFrame({col: [] for col in columns})
        try:
            from ml._imports import HAS_POLARS
            from ml._imports import pl

            if not HAS_POLARS or pl is None:
                raise ImportError
            return pl.DataFrame({col: [] for col in columns})
        except Exception:
            return empty


__all__ = [
    "DAY_NS",
    "CatalogCoverageProvider",
    "NullCoverageProvider",
    "ParquetCoverageSpec",
    "PartitionedParquetCoverageProvider",
    "SqlCoverageOverride",
    "SqlCoverageProvider",
    "SqlMarketDataReader",
    "SqlMarketDataWriter",
    "UnionCoverageProvider",
]
