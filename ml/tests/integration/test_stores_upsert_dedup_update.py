"""
DB tests for SQL upsert dedup/update behavior in ModelStore and StrategyStore.

These ensure that within-batch duplicates deduplicate to last occurrence and that cross-
batch upserts update existing rows as expected.

"""

from __future__ import annotations

import time
from typing import Any

import pytest

from ml.stores.base import ModelPrediction, StrategySignal
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.serial,
    pytest.mark.usefixtures("clean_postgres_db_module"),
]


def test_model_store_dedup_and_update(test_database: Any) -> None:
    store = ModelStore(connection_string=test_database.connection_string)

    model_id = "mdl_upsert"
    instrument_id = "EUR/USD"
    ts = int(time.time_ns())

    # Within-batch duplicate: last wins
    batch = [
        ModelPrediction(
            model_id=model_id,
            instrument_id=instrument_id,
            prediction=0.5,
            confidence=0.7,
            features_used={"f": 1.0},
            inference_time_ms=10.0,
            _ts_event=ts,
            _ts_init=ts,
        ),
        ModelPrediction(
            model_id=model_id,
            instrument_id=instrument_id,
            prediction=0.9,  # override
            confidence=0.2,  # override
            features_used={"f": 2.0},
            inference_time_ms=20.0,
            _ts_event=ts,
            _ts_init=ts,
        ),
    ]
    store.write_batch(batch)

    # Cross-batch update: change again
    store.write_prediction(
        model_id=model_id,
        instrument_id=instrument_id,
        prediction=0.65,
        confidence=0.85,
        features={"g": 3.14},
        inference_time_ms=7.5,
        ts_event=ts,
    )
    store.flush()

    # Read back via direct SQL to avoid pandas connection quirks
    from sqlalchemy import text

    with store.engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT prediction, confidence, inference_time_ms
                FROM public.ml_model_predictions
                WHERE model_id = :mid AND instrument_id = :iid AND ts_event = :ts
                """,
            ),
            {"mid": model_id, "iid": instrument_id, "ts": ts},
        ).fetchone()
        assert row is not None
        assert pytest.approx(float(row[0])) == 0.65
        assert pytest.approx(float(row[1])) == 0.85
        assert pytest.approx(float(row[2])) == 7.5


def test_strategy_store_dedup_and_update(test_database: Any) -> None:
    store = StrategyStore(connection_string=test_database.connection_string)

    strategy_id = "strat_upsert"
    instrument_id = "EUR/USD"
    ts = int(time.time_ns())

    # Within-batch duplicate: last wins
    batch = [
        StrategySignal(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            signal_type="BUY",
            strength=0.2,
            model_predictions={"m": 0.1},
            risk_metrics={"r": 0.1},
            execution_params={"x": 1},
            _ts_event=ts,
            _ts_init=ts,
        ),
        StrategySignal(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            signal_type="SELL",  # override
            strength=0.9,  # override
            model_predictions={"m": 0.9},
            risk_metrics={"r": 0.9},
            execution_params={"x": 2},
            _ts_event=ts,
            _ts_init=ts,
        ),
    ]
    store.write_batch(batch)

    # Cross-batch update: change again
    store.write_signal(
        strategy_id=strategy_id,
        instrument_id=instrument_id,
        signal_type="HOLD",
        strength=0.5,
        model_predictions={"m": 0.5},
        risk_metrics={"r": 0.5},
        execution_params={"x": 3},
        ts_event=ts,
    )
    store.flush()

    # Read back via direct SQL
    from sqlalchemy import text

    with store.engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT signal_type, strength
                FROM public.ml_strategy_signals
                WHERE strategy_id = :sid AND instrument_id = :iid AND ts_event = :ts
                """,
            ),
            {"sid": strategy_id, "iid": instrument_id, "ts": ts},
        ).fetchone()
        assert row is not None
        assert str(row[0]) == "HOLD"
        assert pytest.approx(float(row[1])) == 0.5
