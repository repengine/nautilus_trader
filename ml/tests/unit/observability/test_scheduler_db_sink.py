from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from ml.core.integration import MLIntegrationManager
from ml.observability.service import ObservabilityService
from ml.tests.utils.stubs import build_integration_manager_stub
from nautilus_trader.model.identifiers import InstrumentId


def test_background_db_flusher(tmp_path: Path, default_instrument_id: InstrumentId) -> None:
    mgr: MLIntegrationManager = build_integration_manager_stub()
    MLIntegrationManager.initialize_observability_pipeline(mgr)

    svc = mgr.observability_service
    assert isinstance(svc, ObservabilityService)

    svc.add_latency_stage(
        correlation_id="c1",
        instrument_id=str(default_instrument_id),
        pipeline_stage="data_ingestion",
        ts_stage_start=1,
        ts_stage_end=2,
    )

    db = tmp_path / "obs.db"
    MLIntegrationManager.start_observability_flush(
        mgr,
        base_path=tmp_path,
        interval_seconds=0.01,
        file_format="jsonl",
        sink="db",
        db_connection_string=f"sqlite:///{db}",
    )

    import time

    time.sleep(0.05)
    MLIntegrationManager.stop_observability_flush(mgr)

    import sqlalchemy as sa

    engine = sa.create_engine(f"sqlite:///{db}")
    with engine.connect() as connection:
        latency_frame = pd.read_sql("select * from obs_latency_watermarks", connection)
        assert not latency_frame.empty
