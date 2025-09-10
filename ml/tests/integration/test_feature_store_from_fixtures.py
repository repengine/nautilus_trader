from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.data.fixtures import make_tbbo_fixture
from ml.stores.base import FeatureData
from ml.stores.feature_store import FeatureStore


def _to_feature_rows(df: pd.DataFrame, feature_set_id: str) -> list[FeatureData]:
    rows: list[FeatureData] = []
    for r in df.itertuples(index=False):
        # Minimal features from TBBO snapshot
        values = {"bid_px": float(r.bid_px), "ask_px": float(r.ask_px)}
        rows.append(
            FeatureData(
                feature_set_id=feature_set_id,
                instrument_id=str(r.instrument_id),
                values=values,
                _ts_event=int(r.ts_event),
                _ts_init=int(r.ts_event),
            ),
        )
    return rows


def test_feature_store_write_idempotent_and_ordered_sqlite(tmp_path: Path) -> None:
    # Provider-agnostic path: use file-backed SQLite to preserve schema across connections
    db_path = tmp_path / "fs.db"
    store = FeatureStore(connection_string=f"sqlite:///{db_path}")
    df, _man = make_tbbo_fixture(rows=8)
    feature_set_id = "fx_tbbo_demo"
    rows = _to_feature_rows(df, feature_set_id)

    # First write
    store.write_batch(rows)
    # Duplicate write should upsert (idempotent on unique key)
    store.write_batch(rows)

    # Read back using explicit query with quoted column names (SQLite compatibility)
    import pandas as pd
    from sqlalchemy import text

    with store.engine.connect() as conn:
        cnt = conn.execute(text("SELECT COUNT(*) FROM ml_feature_values")).scalar()
        assert int(cnt or 0) == len(df.index)
        rows = pd.read_sql_query(
            text(
                'SELECT feature_set_id, instrument_id, ts_event, ts_init, "values" FROM ml_feature_values ORDER BY ts_event',
            ),
            conn,
        )
    # Monotonic ordering by ts_event
    assert rows["ts_event"].is_monotonic_increasing
    # Feature set id preserved
    assert set(rows["feature_set_id"]) == {feature_set_id}
