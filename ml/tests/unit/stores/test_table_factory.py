"""Tests for table schema factory."""

from __future__ import annotations

import pytest
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import create_engine

from ml.stores.table_factory import build_instrument_id_column
from ml.stores.table_factory import build_nautilus_timestamp_columns
from ml.stores.table_factory import build_standard_indexes
from ml.stores.table_factory import create_ml_table
from ml.stores.table_factory import get_schema_name


class TestGetSchemaName:
    """Tests for get_schema_name function."""

    def test_get_schema_name_postgresql(self) -> None:
        """PostgreSQL returns 'public' schema."""
        engine = create_engine("postgresql://localhost/test", poolclass=None)
        assert get_schema_name(engine) == "public"

    def test_get_schema_name_sqlite(self) -> None:
        """SQLite returns None (no schemas)."""
        engine = create_engine("sqlite:///:memory:")
        assert get_schema_name(engine) is None

    def test_get_schema_name_with_none_dialect(self) -> None:
        """Handle engines with None dialect gracefully."""
        engine = create_engine("sqlite:///:memory:")
        # Simulate missing dialect
        engine.dialect = None  # type: ignore[assignment]
        assert get_schema_name(engine) is None


class TestBuildNautilusTimestampColumns:
    """Tests for build_nautilus_timestamp_columns function."""

    def test_build_nautilus_timestamp_columns(self) -> None:
        """Timestamp columns have correct types and primary key."""
        columns = build_nautilus_timestamp_columns()
        assert len(columns) == 2
        assert columns[0].name == "ts_event"
        assert columns[0].primary_key is True
        assert columns[1].name == "ts_init"
        assert columns[1].primary_key is False

    def test_timestamp_columns_are_bigint(self) -> None:
        """Timestamp columns use BIGINT type."""
        columns = build_nautilus_timestamp_columns()
        # Check type name
        for col in columns:
            assert str(col.type) == "BIGINT"


class TestBuildInstrumentIdColumn:
    """Tests for build_instrument_id_column function."""

    def test_build_instrument_id_column_primary_key(self) -> None:
        """Instrument ID column with primary_key=True."""
        col = build_instrument_id_column(primary_key=True)
        assert col.name == "instrument_id"
        assert col.primary_key is True
        assert col.type.length == 100

    def test_build_instrument_id_column_not_primary_key(self) -> None:
        """Instrument ID column with primary_key=False."""
        col = build_instrument_id_column(primary_key=False)
        assert col.name == "instrument_id"
        assert col.primary_key is False
        assert col.type.length == 100

    def test_instrument_id_column_default_primary_key(self) -> None:
        """Instrument ID column defaults to primary_key=True."""
        col = build_instrument_id_column()
        assert col.primary_key is True


class TestBuildStandardIndexes:
    """Tests for build_standard_indexes function."""

    def test_build_standard_indexes_with_instrument_ts(self) -> None:
        """Standard indexes include instrument_id + ts_event composite."""
        indexes = build_standard_indexes("test_table")
        assert len(indexes) == 1
        assert indexes[0].name == "idx_test_table_instrument_ts"
        # Check columns in index
        column_names = [col.name for col in indexes[0].columns]
        assert column_names == ["instrument_id", "ts_event"]

    def test_build_standard_indexes_without_instrument_ts(self) -> None:
        """Standard indexes can exclude instrument_id + ts_event."""
        indexes = build_standard_indexes("test_table", include_instrument_ts=False)
        assert len(indexes) == 0

    def test_build_standard_indexes_with_additional_columns(self) -> None:
        """Standard indexes include additional columns."""
        indexes = build_standard_indexes(
            "test_table",
            include_instrument_ts=True,
            additional_columns=["is_live", "model_id"],
        )
        assert len(indexes) == 3
        # Check names
        index_names = [idx.name for idx in indexes]
        assert "idx_test_table_instrument_ts" in index_names
        assert "idx_test_table_is_live" in index_names
        assert "idx_test_table_model_id" in index_names

    def test_build_standard_indexes_only_additional(self) -> None:
        """Can create indexes for additional columns only."""
        indexes = build_standard_indexes(
            "test_table",
            include_instrument_ts=False,
            additional_columns=["status"],
        )
        assert len(indexes) == 1
        assert indexes[0].name == "idx_test_table_status"


class TestCreateMLTable:
    """Tests for create_ml_table factory function."""

    def test_create_ml_table_with_standard_columns(self) -> None:
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

    def test_create_ml_table_without_standard_columns(self) -> None:
        """Table created without standard columns."""
        metadata = MetaData()
        engine = create_engine("sqlite:///:memory:")

        table = create_ml_table(
            name="test_table",
            metadata=metadata,
            engine=engine,
            additional_columns=[
                Column("id", Integer, primary_key=True),
                Column("data", String(100)),
            ],
            include_standard_columns=False,
        )

        assert table.name == "test_table"
        assert "instrument_id" not in table.columns
        assert "ts_event" not in table.columns
        assert "ts_init" not in table.columns
        assert "id" in table.columns
        assert "data" in table.columns

    def test_create_ml_table_validates_name(self) -> None:
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

    def test_create_ml_table_validates_columns(self) -> None:
        """Empty additional_columns raises ValueError."""
        metadata = MetaData()
        engine = create_engine("sqlite:///:memory:")

        with pytest.raises(ValueError, match="Must provide at least one additional column"):
            create_ml_table(
                name="test_table",
                metadata=metadata,
                engine=engine,
                additional_columns=[],
            )

    def test_create_ml_table_with_indexes(self) -> None:
        """Table created with additional indexes."""
        from sqlalchemy import Index

        metadata = MetaData()
        engine = create_engine("sqlite:///:memory:")

        custom_index = Index("idx_custom", "data")

        table = create_ml_table(
            name="test_table",
            metadata=metadata,
            engine=engine,
            additional_columns=[
                Column("data", String(100)),
            ],
            indexes=[custom_index],
        )

        # Check that custom index is attached to table
        index_names = [idx.name for idx in table.indexes]
        assert "idx_custom" in index_names
        # Standard indexes should also be present
        assert "idx_test_table_instrument_ts" in index_names

    def test_create_ml_table_postgresql_schema(self) -> None:
        """Table created with 'public' schema for PostgreSQL."""
        metadata = MetaData()
        engine = create_engine("postgresql://localhost/test", poolclass=None)

        table = create_ml_table(
            name="test_table",
            metadata=metadata,
            engine=engine,
            additional_columns=[
                Column("data", String(100)),
            ],
        )

        assert table.schema == "public"

    def test_create_ml_table_sqlite_no_schema(self) -> None:
        """Table created with None schema for SQLite."""
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

        assert table.schema is None

    def test_create_ml_table_column_order(self) -> None:
        """Columns appear in correct order: standard then additional."""
        metadata = MetaData()
        engine = create_engine("sqlite:///:memory:")

        table = create_ml_table(
            name="test_table",
            metadata=metadata,
            engine=engine,
            additional_columns=[
                Column("data1", String(100)),
                Column("data2", Integer),
            ],
        )

        column_names = list(table.columns.keys())
        # Standard columns first
        assert column_names[0] == "instrument_id"
        assert column_names[1] == "ts_event"
        assert column_names[2] == "ts_init"
        # Then additional columns
        assert column_names[3] == "data1"
        assert column_names[4] == "data2"


class TestFactoryIntegration:
    """Integration tests for factory usage with real stores."""

    def test_factory_produces_compatible_tables(self) -> None:
        """Factory produces tables compatible with store requirements."""
        from sqlalchemy import JSON
        from sqlalchemy import BOOLEAN
        from sqlalchemy import Float

        metadata = MetaData()
        engine = create_engine("sqlite:///:memory:")

        # Create a table similar to ml_model_predictions
        table = create_ml_table(
            name="ml_model_predictions",
            metadata=metadata,
            engine=engine,
            additional_columns=[
                Column("model_id", String(255), primary_key=True),
            ],
            include_standard_columns=False,  # Custom PK structure
        )

        # Add standard columns manually (like stores do)
        table.append_column(Column("instrument_id", String(100), primary_key=True))
        table.append_column(Column("ts_event", Integer, primary_key=True))
        table.append_column(Column("ts_init", Integer))
        table.append_column(Column("prediction", Float))
        table.append_column(Column("confidence", Float))
        table.append_column(Column("features_used", JSON))
        table.append_column(Column("is_live", BOOLEAN))

        # Verify table structure
        assert "model_id" in table.columns
        assert "instrument_id" in table.columns
        assert "ts_event" in table.columns
        assert table.name == "ml_model_predictions"

    def test_factory_schema_detection_matches_stores(self) -> None:
        """Factory schema detection matches what stores expect."""
        # PostgreSQL case
        pg_engine = create_engine("postgresql://localhost/test", poolclass=None)
        assert get_schema_name(pg_engine) == "public"

        # SQLite case
        sqlite_engine = create_engine("sqlite:///:memory:")
        assert get_schema_name(sqlite_engine) is None

    def test_standard_indexes_match_store_patterns(self) -> None:
        """Standard indexes match patterns used by stores."""
        # Stores typically create idx_{table}_instrument_ts
        indexes = build_standard_indexes("ml_model_predictions")
        assert len(indexes) == 1
        assert indexes[0].name == "idx_ml_model_predictions_instrument_ts"

        # Stores also create single-column indexes
        indexes_with_additional = build_standard_indexes(
            "ml_model_predictions",
            additional_columns=["is_live"],
        )
        index_names = [idx.name for idx in indexes_with_additional]
        assert "idx_ml_model_predictions_is_live" in index_names
