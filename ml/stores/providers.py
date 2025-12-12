from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Final, cast

import pandas as pd
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

from ml.common.db_utils import get_or_create_engine
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
    _engine: Engine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.table_name, label="coverage.table")
        _validate_identifier(self.ts_field, label="coverage.ts_field")
        self._engine = get_or_create_engine(self.connection_string)

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
        table = sa_table(
            self.table_name,
            sa_column("instrument_id"),
            sa_column(self.ts_field),
        )
        bucket_expr = sa_func.floor(sa_column(self.ts_field) / bindparam("day_ns")).label("bucket")
        stmt = sa_select(bucket_expr).select_from(table).where(
            sa_column("instrument_id") == bindparam("instrument_id"),
            sa_column(self.ts_field) >= bindparam("start_ns"),
            sa_column(self.ts_field) < bindparam("end_ns"),
        ).group_by(bucket_expr)
        params = {
            "day_ns": DAY_NS,
            "instrument_id": instrument_id,
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
        }
        with self._engine.connect() as conn:
            rows = conn.execute(stmt, params).fetchall()
        return {int(r[0]) for r in rows}

    def latest_timestamp_ns(self, *, dataset_id: str, instrument_id: str) -> int | None:
        """
        Return the latest timestamp seen for an instrument, or None when missing.
        """
        _ = dataset_id
        table = sa_table(
            self.table_name,
            sa_column("instrument_id"),
            sa_column(self.ts_field),
        )
        stmt: Any = (
            sa_select(sa_func.max(sa_column(self.ts_field)))
            .select_from(table)
            .where(sa_column("instrument_id") == bindparam("instrument_id"))
        )
        with self._engine.connect() as conn:
            result = conn.execute(stmt, {"instrument_id": instrument_id}).scalar()
        return int(result) if result is not None else None


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
    base_path : str
        Base path to parquet files
    partition_field : str
        Field used for partitioning (default: "date")
    """

    dataset_id: str
    base_path: str
    partition_field: str = "date"
    timestamp_field: str = "ts_event"
    partition_template: str | None = None

    def files_for_instrument(self, instrument_id: str) -> list[str]:
        """
        Return parquet files for a given instrument.

        The default implementation returns an empty list so callers can safely
        handle missing mirrors without raising.
        """
        _ = instrument_id
        return []


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
        _ = schema, instrument_id, entity_field, start_ns, end_ns
        if not self.specs:
            return set()
        spec = self.specs.get(dataset_id)
        if spec is None:
            return set()
        # TODO: Implement actual parquet reading using spec.base_path/partition_field/timestamp_field
        _ = spec
        return set()


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
