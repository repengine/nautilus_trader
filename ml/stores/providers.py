from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import date
from datetime import datetime
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd
from sqlalchemy import BIGINT
from sqlalchemy import Column
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import bindparam
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import column as sa_column
from sqlalchemy.sql import func as sa_func
from sqlalchemy.sql import select as sa_select
from sqlalchemy.sql import table as sa_table

from ml._imports import HAS_PANDAS
from ml._imports import HAS_PYARROW
from ml._imports import check_ml_dependencies
from ml._imports import pd as pd_runtime
from ml._imports import pq as pq_runtime
from ml.common.db_utils import get_or_create_engine
from ml.config.market_data import MarketDataTableConfig
from ml.config.market_data import MarketDataTableProfile
from ml.data.coverage.types import GLOBAL_ENTITY_ID
from ml.registry.dataclasses import DatasetType
from ml.schema import default_identifier_template_for_dataset_type
from ml.schema import map_schema_to_dataset_type
from ml.schema import schema_to_dataclass
from ml.schema import schema_to_identifier_template
from ml.schema import validate_dataset_type_templates
from ml.schema import validate_identifier_template
from ml.schema import validate_schema_identifier_templates
from ml.stores.io_raw import RawReaderProtocol
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.funcs import urisafe_identifier


DAY_NS: Final[int] = 86_400_000_000_000
logger = logging.getLogger(__name__)


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(identifier: str, *, label: str) -> str:
    if not _IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Invalid SQL identifier for {label}: {identifier!r}")
    return identifier


def _relation_kind(inspector: Any, name: str) -> str | None:
    schema_candidates: list[str | None] = [None]
    default_schema = getattr(inspector, "default_schema_name", None)
    if default_schema:
        schema_candidates.append(default_schema)
    schema_candidates.append("public")

    get_views = getattr(inspector, "get_view_names", None)
    for schema in schema_candidates:
        try:
            if name in inspector.get_table_names(schema=schema):
                return "table"
        except Exception:
            continue
        if callable(get_views):
            try:
                if name in get_views(schema=schema):
                    return "view"
            except Exception:
                continue
    return None


def _is_missing_table_error(exc: Exception) -> bool:
    if isinstance(exc, NoSuchTableError):
        return True
    if isinstance(exc, OperationalError):
        message = str(exc).lower()
        return "no such table" in message or "does not exist" in message or "undefined table" in message
    return False


def _resolve_market_data_profile(
    engine: Engine,
    *,
    config: MarketDataTableConfig,
) -> MarketDataTableProfile:
    if config.profile is not MarketDataTableProfile.AUTO:
        return config.profile
    if engine.dialect.name != "postgresql":
        return MarketDataTableProfile.LEGACY
    inspector = inspect(engine)
    relation = _relation_kind(inspector, config.legacy_table)
    if relation == "table":
        return MarketDataTableProfile.LEGACY
    if relation == "view":
        return MarketDataTableProfile.CLASS_TABLES
    return MarketDataTableProfile.CLASS_TABLES


def _is_global_entity(instrument_id: str) -> bool:
    return instrument_id.strip().upper() == GLOBAL_ENTITY_ID


def _bucket_from_path(path: Path) -> int | None:
    year = None
    month = None
    day = None
    for part in path.parts:
        if part.startswith("year="):
            try:
                year = int(part.split("=", 1)[1])
            except ValueError:
                continue
        elif part.startswith("month="):
            try:
                month = int(part.split("=", 1)[1])
            except ValueError:
                continue
        elif part.startswith("day="):
            try:
                day = int(part.split("=", 1)[1].split(".", 1)[0])
            except ValueError:
                continue
    if year is None or month is None or day is None:
        return None
    try:
        dt = datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None
    return int(dt.timestamp() * 1_000_000_000) // DAY_NS


def _is_instrument_scoped_parquet(spec: ParquetCoverageSpec) -> bool:
    base_path = Path(spec.base_path)
    if base_path.is_file():
        return False
    template = spec.partition_template
    if template is not None and template.strip() == "":
        return False
    return True


def _coerce_parquet_stat_to_ns(value: object) -> int | None:
    if value is None:
        return None
    if hasattr(value, "as_py"):
        try:
            value = value.as_py()
        except Exception:
            return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return int(value.timestamp() * 1_000_000_000)
    if isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=UTC)
        return int(dt.timestamp() * 1_000_000_000)
    if pd_runtime is not None:
        try:
            ts = pd_runtime.to_datetime(value, utc=True, errors="coerce")
        except Exception:
            return None
        if pd_runtime.isna(ts):
            return None
        if hasattr(ts, "value"):
            return int(ts.value)
        try:
            return int(ts.to_datetime64().astype("datetime64[ns]").astype("int64"))
        except Exception:
            return None
    return None


def _buckets_from_parquet_stats(
    path: Path,
    *,
    timestamp_field: str,
    window_start: int,
    window_end: int,
) -> set[int] | None:
    if not HAS_PYARROW or pq_runtime is None:
        return None
    try:
        parquet_file = pq_runtime.ParquetFile(path)
    except Exception:
        logger.debug(
            "parquet_coverage.stats_open_failed",
            exc_info=True,
            extra={"path": str(path)},
        )
        return None
    schema = parquet_file.schema_arrow
    if schema is None:
        return None
    col_idx = schema.get_field_index(timestamp_field)
    if col_idx < 0:
        return None
    metadata = parquet_file.metadata
    if metadata is None:
        return None
    start_bucket = window_start // DAY_NS
    end_bucket = (window_end - 1) // DAY_NS
    buckets: set[int] = set()
    for row_group_idx in range(metadata.num_row_groups):
        column = metadata.row_group(row_group_idx).column(col_idx)
        stats = column.statistics
        if stats is None:
            return None
        min_value = _coerce_parquet_stat_to_ns(stats.min)
        max_value = _coerce_parquet_stat_to_ns(stats.max)
        if min_value is None or max_value is None:
            return None
        min_bucket = min_value // DAY_NS
        max_bucket = max_value // DAY_NS
        if min_bucket != max_bucket:
            return None
        if start_bucket <= min_bucket <= end_bucket:
            buckets.add(int(min_bucket))
    return buckets


def _buckets_from_parquet_dataset(
    path: Path,
    *,
    partition_field: str | None,
    instrument_id: str,
    timestamp_field: str,
    window_start: int,
    window_end: int,
) -> set[int] | None:
    if not HAS_PYARROW or pq_runtime is None:
        return None
    if not partition_field:
        return None
    instrument = instrument_id.strip()
    if not instrument:
        return None
    try:
        import pyarrow as _pa
        import pyarrow.compute as _pc
        import pyarrow.dataset as _ds
    except Exception:
        return None
    try:
        dataset = _ds.dataset(path, format="parquet")
    except Exception:
        logger.debug(
            "parquet_coverage.dataset_open_failed",
            exc_info=True,
            extra={"path": str(path)},
        )
        return None
    try:
        filter_expr = _ds.field(partition_field) == instrument
    except Exception:
        return None
    try:
        scanner = dataset.scanner(columns=[timestamp_field], filter=filter_expr)
    except Exception:
        return None

    start_bucket = window_start // DAY_NS
    end_bucket = (window_end - 1) // DAY_NS
    buckets: set[int] = set()
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        column = batch.column(0)
        try:
            if _pa.types.is_timestamp(column.type):
                if column.type.unit != "ns":
                    column = _pc.cast(column, _pa.timestamp("ns"))
                values = _pc.cast(column, _pa.int64())
            elif _pa.types.is_date(column.type):
                values = _pc.cast(column, _pa.timestamp("ns"))
                values = _pc.cast(values, _pa.int64())
            else:
                values = _pc.cast(column, _pa.int64(), safe=False)
        except Exception:
            return None
        try:
            raw_values = values.to_numpy(zero_copy_only=False)
        except Exception:
            return None
        for raw in raw_values:
            if raw is None:
                continue
            try:
                ts_value = int(raw)
            except Exception:
                continue
            if window_start <= ts_value < window_end:
                bucket = ts_value // DAY_NS
                if start_bucket <= bucket <= end_bucket:
                    buckets.add(int(bucket))
    return buckets


def _schema_to_dataclass(schema: str) -> type[Any]:
    return schema_to_dataclass(schema)


@dataclass(slots=True)
class CatalogCoverageProvider(CoverageProviderProtocol):
    """
    Coverage provider backed by Nautilus ParquetDataCatalog.
    """

    catalog_path: str
    identifier_template: str | None = None
    schema_identifier_templates: Mapping[str, str] | None = None
    dataset_type_identifier_templates: Mapping[DatasetType, str] | None = None
    use_uri_safe_identifiers: bool = True

    def __post_init__(self) -> None:
        self._catalog = ParquetDataCatalog(self.catalog_path)
        self._schema_templates: dict[str, str] = validate_schema_identifier_templates(
            self.schema_identifier_templates,
        )
        self._dataset_templates: dict[DatasetType, str] = validate_dataset_type_templates(
            self.dataset_type_identifier_templates,
        )
        self._identifier_template: str | None = (
            validate_identifier_template(self.identifier_template, label="identifier_template")
            if self.identifier_template is not None
            else None
        )

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        entity_field: str | None = None,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        data_cls = _schema_to_dataclass(schema)
        dataset_type = map_schema_to_dataset_type(schema)
        identifier = resolve_catalog_identifier(
            schema=schema,
            instrument_id=instrument_id,
            identifier_template=self._identifier_template,
            schema_templates=self._schema_templates,
            dataset_type=dataset_type,
            dataset_templates=self._dataset_templates,
            uri_safe=self.use_uri_safe_identifiers,
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
    dataset_overrides: dict[str, object] | None = None
    table_config: MarketDataTableConfig | None = None
    _engine: Engine = field(init=False, repr=False)
    _table_config: MarketDataTableConfig = field(init=False, repr=False)
    _table_profile: MarketDataTableProfile = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.table_name, label="coverage.table")
        _validate_identifier(self.ts_field, label="coverage.ts_field")
        self._engine = get_or_create_engine(self.connection_string)
        self._table_config = (
            self.table_config or MarketDataTableConfig.from_env(legacy_table=self.table_name)
        )
        self._table_profile = _resolve_market_data_profile(self._engine, config=self._table_config)

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        entity_field: str | None = None,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        table_name, schema_name, ts_field, entity = self._resolve_override(
            dataset_id=dataset_id,
            schema=schema,
            entity_field=entity_field,
        )
        table = sa_table(
            table_name,
            sa_column(entity),
            sa_column(ts_field),
            schema=schema_name,
        )
        bucket_expr = sa_func.floor(sa_column(ts_field) / bindparam("day_ns")).label("bucket")
        where_clauses = [
            sa_column(ts_field) >= bindparam("start_ns"),
            sa_column(ts_field) < bindparam("end_ns"),
        ]
        if not _is_global_entity(instrument_id):
            where_clauses.append(sa_column(entity) == bindparam("instrument_id"))
        stmt = sa_select(bucket_expr).select_from(table).where(*where_clauses).group_by(bucket_expr)
        params = {
            "day_ns": DAY_NS,
            "instrument_id": instrument_id,
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
        }
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(stmt, params).fetchall()
        except Exception as exc:
            if _is_missing_table_error(exc):
                logger.debug(
                    "coverage.table_missing",
                    exc_info=True,
                    extra={
                        "dataset_id": dataset_id,
                        "schema": schema,
                        "table_name": table_name,
                        "schema_name": schema_name,
                    },
                )
                return set()
            raise
        return {int(r[0]) for r in rows}

    def latest_timestamp_ns(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        schema: str | None = None,
    ) -> int | None:
        """
        Return the latest timestamp seen for an instrument, or None when missing.
        """
        table_name, schema_name, ts_field, entity = self._resolve_override(
            dataset_id=dataset_id,
            schema=schema or "",
            entity_field=None,
        )
        table = sa_table(
            table_name,
            sa_column(entity),
            sa_column(ts_field),
            schema=schema_name,
        )
        stmt: Any = sa_select(sa_func.max(sa_column(ts_field))).select_from(table)
        params: dict[str, object] = {}
        if not _is_global_entity(instrument_id):
            stmt = stmt.where(sa_column(entity) == bindparam("instrument_id"))
            params["instrument_id"] = instrument_id
        try:
            with self._engine.connect() as conn:
                result = conn.execute(stmt, params).scalar()
        except Exception as exc:
            if _is_missing_table_error(exc):
                logger.debug(
                    "coverage.latest_timestamp_missing",
                    exc_info=True,
                    extra={
                        "dataset_id": dataset_id,
                        "schema": schema,
                        "table_name": table_name,
                        "schema_name": schema_name,
                    },
                )
                return None
            raise
        return int(result) if result is not None else None

    def _resolve_override(
        self,
        *,
        dataset_id: str,
        schema: str,
        entity_field: str | None,
    ) -> tuple[str, str | None, str, str]:
        override = None
        if self.dataset_overrides is not None:
            candidate = self.dataset_overrides.get(dataset_id)
            if isinstance(candidate, SqlCoverageOverride):
                override = candidate
        table_name = (
            override.table_name
            if override and override.table_name
            else self._resolve_table_name(schema)
        )
        ts_field = override.ts_field if override and override.ts_field else self.ts_field
        entity = (
            override.entity_field
            if override and override.entity_field
            else entity_field
            if entity_field
            else "instrument_id"
        )
        schema_name = override.schema if override and override.schema else None
        _validate_identifier(table_name, label="coverage.table")
        _validate_identifier(ts_field, label="coverage.ts_field")
        _validate_identifier(entity, label="coverage.entity_field")
        if schema_name:
            _validate_identifier(schema_name, label="coverage.schema")
        return table_name, schema_name, ts_field, entity

    def _resolve_table_config(self) -> MarketDataTableConfig:
        return self._table_config

    def _resolve_table_name(self, schema: str) -> str:
        config = self._resolve_table_config()
        if self._table_profile is MarketDataTableProfile.LEGACY:
            return config.legacy_table
        return config.table_for_schema(schema)


@dataclass(slots=True)
class SqlMarketDataWriter(MarketDataWriterProtocol):
    """
    SQL writer for canonical market data table.

    Inserts (instrument_id, ts_event, ts_init) rows and ignores conflicts.

    """

    connection_string: str
    table_name: str = "market_data"
    default_source: str | None = "historical"
    table_config: MarketDataTableConfig | None = None
    _engine: Engine = field(init=False, repr=False)
    _meta: MetaData = field(init=False, repr=False)
    _tables: dict[str, Table] = field(init=False, repr=False)
    _table_config: MarketDataTableConfig = field(init=False, repr=False)
    _table_profile: MarketDataTableProfile = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.table_name, label="writer.table")
        self._engine = get_or_create_engine(self.connection_string)
        self._meta = MetaData()
        self._tables = {}
        self._table_config = (
            self.table_config or MarketDataTableConfig.from_env(legacy_table=self.table_name)
        )
        self._table_profile = _resolve_market_data_profile(self._engine, config=self._table_config)

    def write(self, *, dataset_id: str, schema: str, instrument_id: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        table = self._resolve_table(schema)
        table_columns = set(table.columns.keys())
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

        def _first_present_value(row: object, candidates: tuple[str, ...]) -> object | None:
            for key in candidates:
                if key not in cols:
                    continue
                value = _maybe(getattr(row, key, None))
                if value is not None:
                    return value
            return None

        bid_candidates = (
            "bid",
            "bid_px",
            "bid_price",
            "bid_px_0",
            "bid_px_00",
            "bid_px_1",
            "bid_px_01",
        )
        ask_candidates = (
            "ask",
            "ask_px",
            "ask_price",
            "ask_px_0",
            "ask_px_00",
            "ask_px_1",
            "ask_px_01",
        )
        bid_size_candidates = (
            "bid_size",
            "bid_sz",
            "bid_sz_0",
            "bid_sz_00",
            "bid_sz_1",
            "bid_sz_01",
        )
        ask_size_candidates = (
            "ask_size",
            "ask_sz",
            "ask_sz_0",
            "ask_sz_00",
            "ask_sz_1",
            "ask_sz_01",
        )

        def _row(r: object) -> dict[str, object]:
            d: dict[str, object] = {
                "instrument_id": instrument_id,
                "ts_event": int(getattr(r, "ts_event")),
                "ts_init": int(getattr(r, "ts_event")),
            }
            if "open" in cols and "open" in table_columns:
                d["open"] = _maybe(getattr(r, "open", None))
            if "high" in cols and "high" in table_columns:
                d["high"] = _maybe(getattr(r, "high", None))
            if "low" in cols and "low" in table_columns:
                d["low"] = _maybe(getattr(r, "low", None))
            if "close" in cols and "close" in table_columns:
                d["close"] = _maybe(getattr(r, "close", None))
            if "volume" in cols and "volume" in table_columns:
                d["volume"] = _maybe(getattr(r, "volume", None))
            if "bid" in table_columns:
                d["bid"] = _first_present_value(r, bid_candidates)
            if "ask" in table_columns:
                d["ask"] = _first_present_value(r, ask_candidates)
            if "bid_size" in table_columns:
                d["bid_size"] = _first_present_value(r, bid_size_candidates)
            if "ask_size" in table_columns:
                d["ask_size"] = _first_present_value(r, ask_size_candidates)
            if "last" in cols and "last" in table_columns:
                d["last"] = _maybe(getattr(r, "last", None))
            if "trade_count" in cols and "trade_count" in table_columns:
                d["trade_count"] = _maybe(getattr(r, "trade_count", None))
            if "vwap" in cols and "vwap" in table_columns:
                d["vwap"] = _maybe(getattr(r, "vwap", None))
            if "quality_flags" in cols and "quality_flags" in table_columns:
                d["quality_flags"] = _maybe(getattr(r, "quality_flags", None))
            if "source" in table_columns:
                if "source" in cols:
                    d["source"] = _maybe(getattr(r, "source", None))
                elif self.default_source is not None:
                    d["source"] = self.default_source
            if "source_dataset" in cols and "source_dataset" in table_columns:
                d["source_dataset"] = _maybe(getattr(r, "source_dataset", None))
            return d

        records: list[dict[str, object]] = [_row(r) for r in df.itertuples(index=False)]
        dialect = self._engine.dialect.name
        with self._engine.begin() as conn:
            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                stmt = pg_insert(table).values(records)
                stmt = stmt.on_conflict_do_nothing(index_elements=["instrument_id", "ts_event"])
                conn.execute(stmt)
            else:
                conn.execute(table.insert().prefix_with("OR IGNORE"), records)
        return len(records)

    def _resolve_table(self, schema: str) -> Table:
        table_name = self._resolve_table_name(schema)
        cached = self._tables.get(table_name)
        if cached is not None:
            return cached
        _validate_identifier(table_name, label="writer.table")
        try:
            table = Table(table_name, self._meta, autoload_with=self._engine)
        except Exception:
            table = Table(
                table_name,
                self._meta,
                Column("instrument_id", String(100), nullable=False),
                Column("ts_event", BIGINT, nullable=False),
                Column("ts_init", BIGINT, nullable=False),
            )
            self._meta.create_all(self._engine)
        self._tables[table_name] = table
        return table

    def _resolve_table_name(self, schema: str) -> str:
        if self._table_profile is MarketDataTableProfile.LEGACY:
            return self._table_config.legacy_table
        return self._table_config.table_for_schema(schema)


@dataclass(slots=True)
class SqlMarketDataReader(RawReaderProtocol):
    """Read market data ranges from the canonical SQL store."""

    connection_string: str
    table_name: str = "market_data"
    _engine: Engine = field(init=False, repr=False)
    table_config: MarketDataTableConfig | None = None
    _table_config: MarketDataTableConfig = field(init=False, repr=False)
    _table_profile: MarketDataTableProfile = field(init=False, repr=False)
    _columns_cache: dict[str, tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.table_name, label="reader.table")
        self._engine = get_or_create_engine(self.connection_string)
        self._table_config = (
            self.table_config or MarketDataTableConfig.from_env(legacy_table=self.table_name)
        )
        self._table_profile = _resolve_market_data_profile(self._engine, config=self._table_config)
        self._columns_cache = {}

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

        table_name = self._resolve_table_name(dataset_type)
        available = set(self._available_columns_for(table_name))
        required_columns: tuple[str, ...] = ("instrument_id", "ts_event", "ts_init")
        missing_required = [column for column in required_columns if column not in available]
        if missing_required:
            raise RuntimeError(
                f"SqlMarketDataReader missing required columns {missing_required} on table {table_name}",
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
        table = sa_table(table_name, *(sa_column(col) for col in selected_columns))
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

    def _resolve_table_name(self, dataset_type: DatasetType) -> str:
        if self._table_profile is MarketDataTableProfile.LEGACY:
            return self._table_config.legacy_table
        return self._table_config.table_for_dataset_type(dataset_type)

    def _available_columns_for(self, table_name: str) -> tuple[str, ...]:
        cached = self._columns_cache.get(table_name)
        if cached is not None:
            return cached
        try:
            inspector = inspect(self._engine)
            columns = inspector.get_columns(table_name)
            resolved = tuple(column["name"] for column in columns)
        except Exception:
            # Fallback to an empty tuple; read_range will surface a descriptive error.
            resolved = ()
        self._columns_cache[table_name] = resolved
        return resolved


# =============================================================================
# Null/Union/Partitioned Coverage Providers
# =============================================================================


@dataclass(slots=True)
class NullCoverageProvider(CoverageProviderProtocol):
    """
    Null coverage provider that always returns empty coverage.

    Useful as a fallback when no real provider is available.
    """

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        entity_field: str | None = None,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        """Return empty coverage set."""
        _ = dataset_id, schema, instrument_id, entity_field, start_ns, end_ns
        return set()


@dataclass(frozen=True, slots=True)
class ParquetCoverageSpec:
    """
    Specification for parquet-based coverage.

    Attributes
    ----------
    base_path : str | Path
        Base path to parquet files (directory or file path).
    partition_field : str
        Field used for partitioning (default: "date")
    """

    dataset_id: str
    base_path: str | Path
    partition_field: str = "date"
    timestamp_field: str = "ts_event"
    partition_template: str | None = None

    def files_for_instrument(self, instrument_id: str) -> list[str]:
        """
        Return parquet files for a given instrument.

        The spec supports a few common layouts:

        - File-backed datasets (``base_path`` is a parquet file): returns that file.
        - Partitioned datasets (default): ``{partition_field}={instrument_id}/*.parquet`` under ``base_path``.
        - Template layouts (``partition_template``): formats with ``field`` and ``value`` tokens.
          The resulting path may refer to a file or directory.
        """
        instrument = instrument_id.strip()
        if not instrument:
            return []
        base_path = Path(self.base_path)
        if not base_path.exists():
            return []
        if base_path.is_file():
            return [str(base_path)] if base_path.suffix == ".parquet" else []

        def _collect_parquet_files(candidate: Path) -> list[str]:
            if candidate.is_file():
                return [str(candidate)] if candidate.suffix == ".parquet" else []
            if candidate.is_dir():
                return sorted(str(path) for path in candidate.rglob("*.parquet") if path.is_file())
            return []

        if _is_global_entity(instrument):
            return _collect_parquet_files(base_path)

        template = self.partition_template
        if template is not None:
            if template.strip() == "":
                return _collect_parquet_files(base_path)
            try:
                rendered = template.format(field=self.partition_field, value=instrument)
            except Exception:
                logger.debug(
                    "parquet_coverage_spec.template_render_failed",
                    exc_info=True,
                    extra={
                        "dataset_id": self.dataset_id,
                        "partition_template": template,
                        "partition_field": self.partition_field,
                        "instrument_id": instrument,
                    },
                )
                return []
            return _collect_parquet_files(base_path / rendered)

        return _collect_parquet_files(base_path / f"{self.partition_field}={instrument}")


@dataclass(slots=True)
class PartitionedParquetCoverageProvider(CoverageProviderProtocol):
    """
    Coverage provider for partitioned parquet files.
    """

    specs: Mapping[str, ParquetCoverageSpec] | None = None
    spec: ParquetCoverageSpec | None = None

    def __post_init__(self) -> None:
        if self.specs is None and self.spec is not None:
            self.specs = {self.spec.dataset_id: self.spec}
        elif self.specs is None:
            self.specs = {}

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        entity_field: str | None = None,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        """Read coverage from partitioned parquet files."""
        _ = schema, entity_field
        if not self.specs:
            return set()
        spec = self.specs.get(dataset_id)
        if spec is None:
            return set()
        global_entity = _is_global_entity(instrument_id)
        files = spec.files_for_instrument(instrument_id)
        if not files:
            return set()
        window_start = int(start_ns)
        window_end = int(end_ns)
        start_bucket = window_start // DAY_NS
        end_bucket = (window_end - 1) // DAY_NS
        buckets: set[int] = set()
        for path_str in files:
            path = Path(path_str)
            bucket_idx = _bucket_from_path(path)
            if bucket_idx is not None:
                if start_bucket <= bucket_idx <= end_bucket:
                    buckets.add(bucket_idx)
                continue
            if global_entity or _is_instrument_scoped_parquet(spec):
                stats_buckets = _buckets_from_parquet_stats(
                    path,
                    timestamp_field=spec.timestamp_field,
                    window_start=window_start,
                    window_end=window_end,
                )
                if stats_buckets is not None:
                    buckets.update(stats_buckets)
                    continue
            if not global_entity:
                scan_buckets = _buckets_from_parquet_dataset(
                    path,
                    partition_field=spec.partition_field,
                    instrument_id=instrument_id,
                    timestamp_field=spec.timestamp_field,
                    window_start=window_start,
                    window_end=window_end,
                )
                if scan_buckets is not None:
                    buckets.update(scan_buckets)
                    continue
            if not HAS_PANDAS or pd_runtime is None:
                check_ml_dependencies(["pandas"])
            assert pd_runtime is not None
            pd_local = pd_runtime
            try:
                columns = [spec.timestamp_field]
                if spec.partition_field:
                    columns.append(spec.partition_field)
                try:
                    frame = pd_local.read_parquet(path, columns=columns)
                except Exception:
                    frame = pd_local.read_parquet(path, columns=[spec.timestamp_field])
            except Exception:
                logger.debug(
                    "parquet_coverage.read_failed",
                    exc_info=True,
                    extra={"path": str(path), "dataset_id": dataset_id},
                )
                continue
            if spec.timestamp_field not in frame.columns:
                continue
            if not global_entity and spec.partition_field in frame.columns:
                mask = (
                    frame[spec.partition_field]
                    .astype(str)
                    .str.strip()
                    .eq(instrument_id.strip())
                )
                frame = frame.loc[mask]
                if frame.empty:
                    continue
            ts_series = frame[spec.timestamp_field]
            if pd_local.api.types.is_datetime64_any_dtype(ts_series):
                numeric_ts = ts_series.view("int64")
            else:
                numeric_ts = pd_local.to_numeric(ts_series, errors="coerce")
            numeric_ts = numeric_ts.dropna()
            if numeric_ts.empty:
                continue
            numeric_ts = numeric_ts.astype("int64")
            numeric_ts = numeric_ts[(numeric_ts >= window_start) & (numeric_ts < window_end)]
            if numeric_ts.empty:
                continue
            bucket_values = (numeric_ts // DAY_NS).astype("int64").unique()
            for bucket in bucket_values:
                buckets.add(int(bucket))
        return buckets


@dataclass(frozen=True, slots=True)
class SqlCoverageOverride:
    """
    Override configuration for SQL coverage queries.

    Attributes
    ----------
    table_name : str
        Override table name
    date_column : str
        Override date column name
    """

    table_name: str | None = None
    schema: str | None = None
    ts_field: str | None = None
    entity_field: str | None = None
    date_column: str | None = None


@dataclass(slots=True)
class UnionCoverageProvider(CoverageProviderProtocol):
    """
    Coverage provider that unions results from multiple providers.
    """

    providers: list[CoverageProviderProtocol]

    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        entity_field: str | None = None,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        """Return union of all provider coverages."""
        result: set[int] = set()
        for provider in self.providers:
            result |= provider.read_bucket_coverage(
                dataset_id=dataset_id,
                schema=schema,
                instrument_id=instrument_id,
                entity_field=entity_field,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        return result


def resolve_catalog_identifier(
    *,
    schema: str,
    instrument_id: str,
    identifier_template: str | None = None,
    schema_templates: Mapping[str, str] | None = None,
    dataset_type: DatasetType | None = None,
    dataset_templates: Mapping[DatasetType, str] | None = None,
    uri_safe: bool = False,
) -> str:
    """
    Resolve catalog identifier from schema and instrument_id.

    Parameters
    ----------
    schema : str
        Data schema (e.g., "bar_1_minute")
    instrument_id : str
        Instrument identifier (e.g., "AAPL.NASDAQ")
    identifier_template : str | None
        Template for identifier resolution. Falls back to the registry default when None.
        Supports {schema} and {instrument_id} placeholders.
    schema_templates : Mapping[str, str] | None
        Optional per-schema identifier templates (case-insensitive keys).
    dataset_type : DatasetType | None
        Dataset type derived from schema. Used when a dataset-level template is provided.
    dataset_templates : Mapping[DatasetType, str] | None
        Optional per-dataset-type templates, used when a schema-specific template is
        not present.
    uri_safe : bool
        When True, normalize the resolved identifier using ``urisafe_identifier``.

    Returns
    -------
    str
        Resolved identifier string

    Raises
    ------
    ValueError
        If the schema is not registered in the schema registry.

    Examples
    --------
    >>> resolve_catalog_identifier(schema="bar_1_minute", instrument_id="AAPL.NASDAQ")
    'AAPL.NASDAQ-1-MINUTE-LAST-EXTERNAL'
    >>> resolve_catalog_identifier(
    ...     schema="bar_1_minute",
    ...     instrument_id="AAPL.NASDAQ",
    ...     identifier_template="{instrument_id}_{schema}"
    ... )
    'AAPL.NASDAQ_bar_1_minute'
    """
    normalized_schema = schema.strip().lower()
    resolved_dataset_type = dataset_type or map_schema_to_dataset_type(normalized_schema)
    template = None
    if schema_templates is not None:
        template = schema_templates.get(normalized_schema)
    if template is None and dataset_templates is not None:
        template = dataset_templates.get(resolved_dataset_type)
    if template is None and identifier_template is not None:
        template = validate_identifier_template(identifier_template, label="identifier_template")
    if template is None:
        try:
            template = schema_to_identifier_template(normalized_schema)
        except ValueError:
            template = default_identifier_template_for_dataset_type(resolved_dataset_type)
    resolved = template.format(schema=normalized_schema, instrument_id=instrument_id)
    if uri_safe:
        return urisafe_identifier(resolved)
    return resolved


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
    "resolve_catalog_identifier",
]
