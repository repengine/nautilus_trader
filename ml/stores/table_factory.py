"""
Table schema factory for ML stores.

Provides centralized table creation with standardized schemas, indexes, and
dialect-specific configuration. All ML stores should use these factories
instead of defining table schemas directly.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import BIGINT
from sqlalchemy import Column
from sqlalchemy import Index
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.engine import Engine


if TYPE_CHECKING:
    from sqlalchemy.sql.schema import SchemaEventTarget


__all__ = [
    "build_instrument_id_column",
    "build_nautilus_timestamp_columns",
    "build_standard_indexes",
    "create_ml_table",
    "get_schema_name",
]


class _UnboundIndex(Index):
    """
    Index subclass that exposes columns before table binding for tests.
    """

    def __init__(
        self,
        name: str,
        *expressions: str | Column[Any],
        **kw: Any,
    ) -> None:
        super().__init__(name, *expressions, **kw)

        from sqlalchemy.sql.base import ColumnCollection

        cols: list[Column[Any]] = []
        for expr in self.expressions:
            if isinstance(expr, str):
                cols.append(Column(expr, String))
            elif isinstance(expr, Column):
                cols.append(expr)

        collection: ColumnCollection[str, Column[Any]] = ColumnCollection()
        for col in cols:
            collection.add(col)

        self._unbound_columns = collection.as_readonly()
        self._is_bound = False

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        super()._set_parent(parent, **kw)
        self._is_bound = True

    @property
    def columns(self) -> Any:
        if getattr(self, "_is_bound", False):
            return self._columns
        return getattr(self, "_unbound_columns", [])


def get_schema_name(engine: Engine) -> str | None:
    """
    Get schema name based on database dialect.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine.

    Returns
    -------
    str | None
        Schema name ("public" for PostgreSQL, None for SQLite).

    Notes
    -----
    - PostgreSQL uses "public" schema by default
    - SQLite doesn't support schemas (returns None)
    - Other dialects default to None (can be extended)

    """
    dialect_name = getattr(getattr(engine, "dialect", None), "name", None)
    if dialect_name and dialect_name != "sqlite":
        return "public"
    return None


def build_nautilus_timestamp_columns() -> list[Column[Any]]:
    """
    Build standard Nautilus timestamp columns.

    Returns
    -------
    list[Column]
        Two BIGINT columns: ts_event (primary key) and ts_init.

    Notes
    -----
    - ts_event: Event timestamp in nanoseconds (primary key component)
    - ts_init: Initialization timestamp in nanoseconds
    - All ML tables must include these for joinability with bar/quote data

    """
    return [
        Column("ts_event", BIGINT, primary_key=True),
        Column("ts_init", BIGINT),
    ]


def build_instrument_id_column(primary_key: bool = True) -> Column[Any]:
    """
    Build standard instrument_id column.

    Parameters
    ----------
    primary_key : bool, optional
        Whether instrument_id is part of primary key. Defaults to True.

    Returns
    -------
    Column
        String(100) column for instrument identifier.

    Notes
    -----
    - Max length 100 accommodates all Nautilus instrument ID formats
    - Primary key for most ML tables (partitioned by instrument)

    """
    return Column("instrument_id", String(100), primary_key=primary_key)


def build_standard_indexes(
    table_name: str,
    include_instrument_ts: bool = True,
    additional_columns: list[str] | None = None,
) -> list[Index]:
    """
    Build standard indexes for ML tables.

    Parameters
    ----------
    table_name : str
        Name of the table.
    include_instrument_ts : bool, optional
        Include composite index on (instrument_id, ts_event). Defaults to True.
    additional_columns : list[str] | None, optional
        Additional columns to index.

    Returns
    -------
    list[Index]
        List of Index objects.

    Notes
    -----
    - Composite (instrument_id, ts_event) index optimizes time-series queries
    - Additional columns get individual indexes
    - Index names follow pattern: idx_{table}_{column(s)}

    """
    indexes: list[Index] = []

    if include_instrument_ts:
        indexes.append(
            _UnboundIndex(
                f"idx_{table_name}_instrument_ts",
                "instrument_id",
                "ts_event",
            ),
        )

    if additional_columns:
        for col in additional_columns:
            indexes.append(
                _UnboundIndex(f"idx_{table_name}_{col}", col),
            )

    return indexes


def create_ml_table(
    name: str,
    metadata: MetaData,
    engine: Engine,
    additional_columns: list[Column[Any]],
    indexes: list[Index] | None = None,
    include_standard_columns: bool = True,
) -> Table:
    """
    Factory function to create ML table with standard schema.

    Parameters
    ----------
    name : str
        Table name.
    metadata : MetaData
        SQLAlchemy metadata object.
    engine : Engine
        Database engine (for dialect detection).
    additional_columns : list[Column]
        Columns specific to this table.
    indexes : list[Index] | None, optional
        Additional indexes beyond standard ones.
    include_standard_columns : bool, optional
        Include instrument_id and timestamps. Defaults to True.

    Returns
    -------
    Table
        Configured SQLAlchemy Table object.

    Raises
    ------
    ValueError
        If name is empty or additional_columns is empty.

    Notes
    -----
    - Automatically includes instrument_id, ts_event, ts_init if include_standard_columns=True
    - Schema name auto-detected based on dialect
    - Standard indexes created automatically
    - All ML tables should use this factory for consistency

    Examples
    --------
    >>> from sqlalchemy import Column, JSON, Float
    >>>
    >>> table = create_ml_table(
    ...     name="ml_feature_values",
    ...     metadata=metadata,
    ...     engine=engine,
    ...     additional_columns=[
    ...         Column("feature_data", JSON),
    ...         Column("feature_version", String(50)),
    ...     ],
    ... )

    """
    if not name:
        raise ValueError("Table name cannot be empty")
    if not additional_columns:
        raise ValueError("Must provide at least one additional column")

    schema_name = get_schema_name(engine)
    columns: list[Column[Any]] = []

    # Add standard columns
    if include_standard_columns:
        columns.append(build_instrument_id_column(primary_key=True))
        columns.extend(build_nautilus_timestamp_columns())

    # Add table-specific columns
    columns.extend(additional_columns)

    # Create table
    table = Table(
        name,
        metadata,
        *columns,
        schema=schema_name,
    )

    # Add indexes
    if include_standard_columns:
        standard_indexes = build_standard_indexes(name)
        for idx in standard_indexes:
            idx._set_parent(table)

    if indexes:
        for idx in indexes:
            idx._set_parent(table)

    return table
