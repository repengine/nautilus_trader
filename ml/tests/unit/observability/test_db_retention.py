from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pandas as pd

from ml.observability.db_persistence import ObservabilityDBPersistor


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


def test_apply_retention_deletes_old_rows(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    old = now - timedelta(days=10)
    recent = now - timedelta(days=1)

    lat = pd.DataFrame(
        [
            {
                "correlation_id": "c1",
                "instrument_id": "EURUSD.SIM",
                "pipeline_stage": "data_ingestion",
                "ts_stage_start": _ns(old),
                "ts_stage_end": _ns(old) + 1_000,
                "stage_latency_ns": 1_000,
                "cumulative_latency_ns": 1_000,
            },
            {
                "correlation_id": "c2",
                "instrument_id": "EURUSD.SIM",
                "pipeline_stage": "data_ingestion",
                "ts_stage_start": _ns(recent),
                "ts_stage_end": _ns(recent) + 1_000,
                "stage_latency_ns": 1_000,
                "cumulative_latency_ns": 2_000,
            },
        ],
    )

    db = tmp_path / "obs.db"
    per = ObservabilityDBPersistor(connection_string=f"sqlite:///{db}")
    # Seed with two rows
    per.persist(
        {
            "latency": lat,
            "metrics": pd.DataFrame(),
            "correlation": pd.DataFrame(),
            "health": pd.DataFrame(),
        },
    )

    # Apply retention keeping last 5 days
    out = per.apply_retention(retention_days=5)
    # One row deleted from latency table; others unaffected
    assert out.get("obs_latency_watermarks", 0) == 1

    # Verify remaining rows
    import sqlalchemy as sa

    eng = sa.create_engine(f"sqlite:///{db}")
    with eng.connect() as conn:
        remaining = pd.read_sql("select * from obs_latency_watermarks", conn)
        assert len(remaining) == 1
        assert remaining["ts_stage_end"].iloc[0] >= _ns(now - timedelta(days=5))
