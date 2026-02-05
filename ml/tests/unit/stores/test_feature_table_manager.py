from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import MetaData
from sqlalchemy import create_engine
from sqlalchemy import select

from ml.stores.feature_table_manager import FeatureTableManager


def test_get_feature_values_table_raises_when_not_initialized() -> None:
    """
    Accessing the table before setup should raise a RuntimeError.
    """
    engine = create_engine("sqlite:///:memory:")
    manager = FeatureTableManager(engine, MetaData())

    with pytest.raises(RuntimeError):
        manager.get_feature_values_table()


def test_setup_tables_creates_fallback_table() -> None:
    """
    Fallback table creation should succeed when reflection fails.
    """
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    manager = FeatureTableManager(engine, metadata)

    table = manager.setup_tables()

    assert table.name == "ml_feature_values"
    assert manager.get_feature_values_table() is table


def test_clear_features_filters_by_instrument_id() -> None:
    """
    Clear should remove rows for a specific instrument.
    """
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    manager = FeatureTableManager(engine, metadata)
    table = manager.setup_tables()

    with engine.begin() as conn:
        conn.execute(
            table.insert().values(
                feature_set_id="fs_demo",
                instrument_id="EUR/USD.SIM",
                ts_event=1,
                ts_init=1,
                values={"f1": 1.0},
                is_live=False,
                source="computed",
                created_at=1,
            ),
        )

    manager.clear_features(instrument_id="EUR/USD.SIM")

    with engine.connect() as conn:
        rows: list[Any] = conn.execute(select(table.c.instrument_id)).fetchall()

    assert rows == []
