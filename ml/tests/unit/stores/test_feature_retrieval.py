from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import BIGINT
from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import create_engine

from ml.stores.feature_retrieval import FeatureRetrieval

BASE_TS_NS = 1_700_000_000_000_000_000
STEP_NS = 60_000_000_000


def _build_feature_values_table(engine: Any) -> Table:
    metadata = MetaData()
    table = Table(
        "ml_feature_values",
        metadata,
        Column("feature_set_id", String(255), primary_key=True),
        Column("instrument_id", String(100), primary_key=True),
        Column("ts_event", BIGINT, primary_key=True),
        Column("ts_init", BIGINT),
        Column("values", JSON, nullable=False),
    )
    metadata.create_all(engine)
    return table


def _insert_feature_row(
    table: Table,
    engine: Any,
    *,
    feature_set_id: str,
    instrument_id: str,
    ts_event: int,
    ts_init: int,
    values: dict[str, float] | str,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            table.insert().values(
                feature_set_id=feature_set_id,
                instrument_id=instrument_id,
                ts_event=ts_event,
                ts_init=ts_init,
                values=values,
            ),
        )


def test_get_training_data_filters_and_orders(
    isolated_prometheus_registry: object,
) -> None:
    """
    Training data retrieval should filter by feature set and preserve order.
    """
    engine = create_engine("sqlite:///:memory:")
    table = _build_feature_values_table(engine)

    _insert_feature_row(
        table,
        engine,
        feature_set_id="fs_demo",
        instrument_id="EUR/USD.SIM",
        ts_event=BASE_TS_NS,
        ts_init=BASE_TS_NS,
        values={"a": 1.0, "b": 2.0},
    )
    _insert_feature_row(
        table,
        engine,
        feature_set_id="fs_demo",
        instrument_id="EUR/USD.SIM",
        ts_event=BASE_TS_NS + STEP_NS,
        ts_init=BASE_TS_NS + STEP_NS,
        values={"a": 1.5, "b": 3.0},
    )
    _insert_feature_row(
        table,
        engine,
        feature_set_id="fs_other",
        instrument_id="EUR/USD.SIM",
        ts_event=BASE_TS_NS + STEP_NS // 2,
        ts_init=BASE_TS_NS + STEP_NS // 2,
        values={"a": 9.0, "b": 9.0},
    )

    retrieval = FeatureRetrieval(
        engine,
        table,
        feature_set_id="fs_demo",
        catalog_path=None,
    )

    features, timestamps, names = retrieval.get_training_data(
        "EUR/USD.SIM",
        BASE_TS_NS - STEP_NS,
        BASE_TS_NS + STEP_NS * 2,
        "fs_demo",
        feature_names=["b"],
    )

    assert names == ["b"]
    np.testing.assert_array_equal(
        timestamps,
        np.array([BASE_TS_NS, BASE_TS_NS + STEP_NS], dtype=np.int64),
    )
    np.testing.assert_allclose(features, np.array([[2.0], [3.0]], dtype=np.float64))


def test_get_training_data_returns_empty_on_missing_rows(
    isolated_prometheus_registry: object,
) -> None:
    """
    Missing rows should return empty arrays and names.
    """
    engine = create_engine("sqlite:///:memory:")
    table = _build_feature_values_table(engine)

    retrieval = FeatureRetrieval(
        engine,
        table,
        feature_set_id="fs_demo",
        catalog_path=None,
    )

    features, timestamps, names = retrieval.get_training_data(
        "EUR/USD.SIM",
        BASE_TS_NS - STEP_NS,
        BASE_TS_NS - 1,
        "fs_demo",
    )

    assert names == []
    assert features.size == 0
    assert timestamps.size == 0


def test_get_latest_at_or_before_filters_features(
    isolated_prometheus_registry: object,
) -> None:
    """
    Latest-at-or-before should respect feature filters.
    """
    engine = create_engine("sqlite:///:memory:")
    table = _build_feature_values_table(engine)

    _insert_feature_row(
        table,
        engine,
        feature_set_id="fs_demo",
        instrument_id="EUR/USD.SIM",
        ts_event=BASE_TS_NS,
        ts_init=BASE_TS_NS,
        values={"a": 1.0, "b": 2.0},
    )
    _insert_feature_row(
        table,
        engine,
        feature_set_id="fs_demo",
        instrument_id="EUR/USD.SIM",
        ts_event=BASE_TS_NS + STEP_NS,
        ts_init=BASE_TS_NS + STEP_NS,
        values={"a": 1.5, "b": 3.0},
    )

    retrieval = FeatureRetrieval(
        engine,
        table,
        feature_set_id="fs_demo",
        catalog_path=None,
    )

    latest = retrieval.get_latest_at_or_before(
        "EUR/USD.SIM",
        BASE_TS_NS + STEP_NS // 2,
        "fs_demo",
        feature_names=["b"],
    )

    assert latest == {"b": 2.0}


def test_read_range_filters_by_instrument(
    isolated_prometheus_registry: object,
) -> None:
    """
    read_range should return only rows for the requested instrument.
    """
    engine = create_engine("sqlite:///:memory:")
    table = _build_feature_values_table(engine)

    _insert_feature_row(
        table,
        engine,
        feature_set_id="fs_demo",
        instrument_id="EUR/USD.SIM",
        ts_event=BASE_TS_NS,
        ts_init=BASE_TS_NS,
        values={"a": 1.0},
    )
    _insert_feature_row(
        table,
        engine,
        feature_set_id="fs_demo",
        instrument_id="BTC/USD.SIM",
        ts_event=BASE_TS_NS,
        ts_init=BASE_TS_NS,
        values={"a": 9.0},
    )

    retrieval = FeatureRetrieval(
        engine,
        table,
        feature_set_id="fs_demo",
        catalog_path=None,
    )

    frame = retrieval.read_range(
        BASE_TS_NS - STEP_NS,
        BASE_TS_NS + STEP_NS,
        instrument_id="EUR/USD.SIM",
    )

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert frame.iloc[0]["instrument_id"] == "EUR/USD.SIM"


def test_features_exist_matches_timestamp(
    isolated_prometheus_registry: object,
) -> None:
    """
    Feature existence checks should track timestamps precisely.
    """
    engine = create_engine("sqlite:///:memory:")
    table = _build_feature_values_table(engine)

    _insert_feature_row(
        table,
        engine,
        feature_set_id="fs_demo",
        instrument_id="EUR/USD.SIM",
        ts_event=BASE_TS_NS,
        ts_init=BASE_TS_NS,
        values={"a": 1.0},
    )

    retrieval = FeatureRetrieval(
        engine,
        table,
        feature_set_id="fs_demo",
        catalog_path=None,
    )

    assert retrieval._features_exist("EUR/USD.SIM", BASE_TS_NS) is True
    assert retrieval._features_exist("EUR/USD.SIM", BASE_TS_NS + STEP_NS) is False
