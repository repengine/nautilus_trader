from __future__ import annotations

from pathlib import Path
from threading import Event

import pandas as pd

from ml.core.integration import MLIntegrationManager


def test_background_db_flusher(tmp_path: Path) -> None:
    mgr = object.__new__(MLIntegrationManager)  # type: ignore[misc]
    MLIntegrationManager.initialize_observability_pipeline(mgr)
    svc = mgr.observability_service  # type: ignore[attr-defined]
    assert svc is not None

    # Add one row so flush writes something
    svc.add_latency_stage(
        correlation_id="c1",
        instrument_id="EURUSD.SIM",
        pipeline_stage="data_ingestion",
        ts_stage_start=1,
        ts_stage_end=2,
    )

    db = tmp_path / "obs.db"
    # Start background DB flusher with very small interval for test
    MLIntegrationManager.start_observability_flush(
        mgr,
        base_path=tmp_path,  # unused for db sink
        interval_seconds=0.01,
        file_format="jsonl",
        sink="db",
        db_connection_string=f"sqlite:///{db}",
    )

    # Allow a brief period then stop
    import time

    time.sleep(0.05)
    MLIntegrationManager.stop_observability_flush(mgr)

    # Verify DB has the latency table populated
    import sqlalchemy as sa

    eng = sa.create_engine(f"sqlite:///{db}")
    with eng.connect() as conn:
        lat_df = pd.read_sql("select * from obs_latency_watermarks", conn)
        assert not lat_df.empty
