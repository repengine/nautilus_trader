# Task: [Phase 1.2] Create Table Schema Factory

## Context
**Phase:** 1 - DRY Violations - Critical Path
**Task ID:** 1.2
**Depends On:** 1.1
**Estimated Effort:** 6 hours
**Impact Score:** 567 (6 store files affected)

## Scope
Create a centralized table schema factory in `ml/stores/table_factory.py` to eliminate duplicated `_setup_tables()` logic across FeatureStore, ModelStore, and StrategyStore. Extract common column patterns, schema detection, and table creation logic.

## Required Reading
- [x] REFACTORING_PLAN.md (Phase 1.2)
- [x] DRY Violations Report (SQL Table Setup Patterns section)
- [x] ml/docs/development/CODING_STANDARDS.md
- [x] ml/docs/architecture/universal_patterns_guide.md
- [x] ml/schema/features.sql (understand table requirements)

## Definition of Done
- [ ] New file created: `ml/stores/table_factory.py` with factory functions
- [ ] Helper functions implemented:
  - [ ] `get_schema_name(engine)` - Dialect-based schema detection
  - [ ] `build_nautilus_timestamp_columns()` - Standard ts_event/ts_init columns
  - [ ] `build_standard_indexes()` - Common index patterns
  - [ ] `create_ml_table()` - Table factory with standard schema
- [ ] Refactored `_setup_tables()` in:
  - [ ] ml/stores/feature_store.py
  - [ ] ml/stores/model_store.py
  - [ ] ml/stores/strategy_store.py
- [ ] All 3 stores use factory functions
- [ ] Comprehensive test suite created
- [ ] All tests pass (no behavioral changes)
- [ ] Ruff check passes
- [ ] MyPy strict passes
- [ ] Pattern validation passes
- [ ] Backward compatibility: table schemas unchanged

## Files to Modify

### Create New (2 files)
- [ ] `ml/stores/table_factory.py` - Table schema factory
- [ ] `ml/tests/unit/stores/test_table_factory.py` - Comprehensive tests

### Modify (3 stores)
- [ ] `ml/stores/feature_store.py` - Refactor `_setup_tables()`
- [ ] `ml/stores/model_store.py` - Refactor `_setup_tables()`
- [ ] `ml/stores/strategy_store.py` - Refactor `_setup_tables()`

### Update (1 file)
- [ ] `ml/stores/__init__.py` - Export factory functions

## Implementation Steps

### Step 1: Create ml/stores/table_factory.py

```python
"""
Table schema factory for ML stores.

Provides centralized table creation with standardized schemas, indexes, and
dialect-specific configuration. All ML stores should use these factories
instead of defining table schemas directly.

"""

from __future__ import annotations

from typing import Any

from sqlalchemy import BIGINT, Column, Index, MetaData, String, Table
from sqlalchemy.engine import Engine

__all__ = [
    "get_schema_name",
    "build_nautilus_timestamp_columns",
    "build_instrument_id_column",
    "build_standard_indexes",
    "create_ml_table",
]


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


def build_nautilus_timestamp_columns() -> list[Column]:
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


def build_instrument_id_column(primary_key: bool = True) -> Column:
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
    indexes = []

    if include_instrument_ts:
        indexes.append(
            Index(
                f"idx_{table_name}_instrument_ts",
                "instrument_id",
                "ts_event",
            ),
        )

    if additional_columns:
        for col in additional_columns:
            indexes.append(
                Index(f"idx_{table_name}_{col}", col),
            )

    return indexes


def create_ml_table(
    name: str,
    metadata: MetaData,
    engine: Engine,
    additional_columns: list[Column],
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
    columns = []

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
            idx._set_parent(table)  # Attach to table

    if indexes:
        for idx in indexes:
            idx._set_parent(table)

    return table
```

### Step 2: Update ml/stores/__init__.py

Add imports:
```python
from ml.stores.table_factory import (
    create_ml_table,
    get_schema_name,
    build_nautilus_timestamp_columns,
    build_standard_indexes,
)
```

### Step 3: Refactor ml/stores/feature_store.py

**OLD _setup_tables():**
```python
def _setup_tables(self) -> None:
    schema_name: str | None = None
    dialect_name = getattr(getattr(self.engine, "dialect", None), "name", None)
    if dialect_name and dialect_name != "sqlite":
        schema_name = "public"

    self.feature_values_table = Table(
        "ml_feature_values",
        self.metadata,
        Column("instrument_id", String(100), primary_key=True),
        Column("ts_event", BIGINT, primary_key=True),
        Column("ts_init", BIGINT),
        Column("feature_data", JSON),
        # ... etc
        schema=schema_name,
    )
```

**NEW _setup_tables():**
```python
from ml.stores.table_factory import create_ml_table

def _setup_tables(self) -> None:
    from sqlalchemy import JSON, String

    self.feature_values_table = create_ml_table(
        name="ml_feature_values",
        metadata=self.metadata,
        engine=self.engine,
        additional_columns=[
            Column("feature_data", JSON),
            Column("feature_version", String(50)),
            # ... other feature-specific columns
        ],
    )
```

### Step 4: Refactor ml/stores/model_store.py

Similar pattern - replace `_setup_tables()` with factory calls.

### Step 5: Refactor ml/stores/strategy_store.py

Similar pattern - replace `_setup_tables()` with factory calls.

### Step 6: Create comprehensive test suite

Create `ml/tests/unit/stores/test_table_factory.py`:

```python
"""Tests for table schema factory."""

import pytest
from sqlalchemy import Column, MetaData, String, create_engine

from ml.stores.table_factory import (
    get_schema_name,
    build_nautilus_timestamp_columns,
    build_instrument_id_column,
    build_standard_indexes,
    create_ml_table,
)


def test_get_schema_name_postgresql():
    """PostgreSQL returns 'public' schema."""
    engine = create_engine("postgresql://localhost/test")
    assert get_schema_name(engine) == "public"


def test_get_schema_name_sqlite():
    """SQLite returns None (no schemas)."""
    engine = create_engine("sqlite:///:memory:")
    assert get_schema_name(engine) is None


def test_build_nautilus_timestamp_columns():
    """Timestamp columns have correct types and primary key."""
    columns = build_nautilus_timestamp_columns()
    assert len(columns) == 2
    assert columns[0].name == "ts_event"
    assert columns[0].primary_key is True
    assert columns[1].name == "ts_init"


def test_build_instrument_id_column():
    """Instrument ID column has correct type and primary key."""
    col = build_instrument_id_column(primary_key=True)
    assert col.name == "instrument_id"
    assert col.primary_key is True
    assert col.type.length == 100


def test_create_ml_table_with_standard_columns():
    """Table created with standard columns."""
    metadata = MetaData()
    engine = create_engine("sqlite:///:memory:")

    table = create_ml_table(
        name="test_table",
        metadata=metadata,
        engine=engine,
        additional_columns=[
            Column("data", String(100)),
        ],
    )

    assert table.name == "test_table"
    assert "instrument_id" in table.columns
    assert "ts_event" in table.columns
    assert "ts_init" in table.columns
    assert "data" in table.columns


def test_create_ml_table_validates_name():
    """Empty table name raises ValueError."""
    metadata = MetaData()
    engine = create_engine("sqlite:///:memory:")

    with pytest.raises(ValueError, match="Table name cannot be empty"):
        create_ml_table(
            name="",
            metadata=metadata,
            engine=engine,
            additional_columns=[Column("data", String(100))],
        )
```

### Step 7: Run validation

```bash
# Unit tests
pytest ml/tests/unit/stores/test_table_factory.py -v

# Store tests (ensure no behavioral changes)
pytest ml/tests/unit/stores/test_feature_store.py -v
pytest ml/tests/unit/stores/test_model_store.py -v
pytest ml/tests/unit/stores/test_strategy_store.py -v

# Integration tests
pytest ml/tests/integration/stores/ -v

# Linting
ruff check ml/stores/table_factory.py
mypy ml/stores/table_factory.py --strict

# Pattern validation
make validate-nautilus-patterns
```

## Testing Requirements

- [ ] Unit tests for all factory functions
- [ ] Test dialect detection (PostgreSQL vs SQLite)
- [ ] Test standard column generation
- [ ] Test index generation
- [ ] Test table creation with various configurations
- [ ] Integration test: verify stores create identical schemas
- [ ] Regression test: compare table schemas before/after refactor

## Rollback Plan

```bash
git checkout ml/stores/table_factory.py
git checkout ml/stores/__init__.py
git checkout ml/stores/feature_store.py
git checkout ml/stores/model_store.py
git checkout ml/stores/strategy_store.py
git checkout ml/tests/unit/stores/test_table_factory.py
```

## Success Metrics
- Lines reduced: ~500 (duplicated schema code across 3 stores)
- DRY impact score: 567 → ~50 (91% reduction)
- Files affected: 5 (3 stores refactored + 2 new)
- Test coverage: New module at 100%
- Schema consistency: ✅ All tables follow same pattern
- Backward compatible: ✅ Table schemas unchanged

## Notes
- Table schemas must remain **byte-for-byte identical** after refactoring
- This is critical - any schema change breaks existing databases
- Test by comparing actual CREATE TABLE statements before/after
- All 3 stores have similar but not identical _setup_tables() - careful extraction
- Standard columns (instrument_id, ts_event, ts_init) are mandatory for ML tables
- Index patterns optimize time-series queries (common in ML workloads)
