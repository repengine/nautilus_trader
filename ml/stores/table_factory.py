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
    Index subclass that populates columns before table binding.

    SQLAlchemy Index objects normally don't populate the .columns attribute
    until they're bound to a table. This subclass works around that limitation
    for test compatibility by creating Column objects from string expressions.

    Notes
    -----
    This is a workaround for testing. In production, indexes should be bound
    to tables via _set_parent() which properly resolves column references.

    """

    def __init__(
        self,
        name: str,
        *expressions: str | Column[Any],
        **kw: Any,
    ) -> None:
        """
        Create index with pre-populated columns.

        Parameters
        ----------
        name : str
            Index name.
        *expressions : str | Column
            Column names or Column objects.
        **kw : Any
            Additional Index kwargs.

        """
        super().__init__(name, *expressions, **kw)

        # Pre-populate columns from string expressions for test compatibility
        # This allows .columns to work before binding to a table
        from sqlalchemy.sql.base import ColumnCollection

        cols: list[Column[Any]] = []
        for expr in self.expressions:
            if isinstance(expr, str):
                # Create a Column object from the string name
                # Type is generic String since we don't know the actual type yet
                cols.append(Column(expr, String))
            elif isinstance(expr, Column):
                cols.append(expr)

        # Store columns for early access (before binding to table)
        col_collection: ColumnCollection[str, Column[Any]] = ColumnCollection()
        for col in cols:
            col_collection.add(col)

        # Store as readonly for early access before _set_parent()
        self._unbound_columns = col_collection.as_readonly()
        # Flag to track if we've been bound to a table
        self._is_bound: bool = False

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        """
        Bind index to a table.

        Parameters
        ----------
        parent : SchemaEventTarget
            The parent schema element to bind to.
        **kw : Any
            Additional keyword arguments.

        """
        # Call parent implementation to do the real binding
        super()._set_parent(parent, **kw)
        # Mark as bound so we use the real columns from now on
        self._is_bound = True

    @property
    def columns(self) -> Any:
        """
        Return columns collection.

        Returns
        -------
        ReadOnlyColumnCollection
            Immutable collection of columns in this index.

        """
        # If bound to a table, use the real columns from parent
        if self._is_bound:
            return self._columns
        # Otherwise, use our pre-populated columns for testing
        return self._unbound_columns


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
        List of Index objects with populated column references.

    Notes
    -----
    - Composite (instrument_id, ts_event) index optimizes time-series queries
    - Additional columns get individual indexes
    - Index names follow pattern: idx_{table}_{column(s)}
    - Uses _UnboundIndex to populate .columns before table binding

    Example
    -------
    >>> indexes = build_standard_indexes("ml_predictions")
    >>> indexes[0].name
    'idx_ml_predictions_instrument_ts'
    >>> [col.name for col in indexes[0].columns]
    ['instrument_id', 'ts_event']

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
        for col_name in additional_columns:
            indexes.append(
                _UnboundIndex(f"idx_{table_name}_{col_name}", col_name),
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
