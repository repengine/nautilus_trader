from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.asyncio
async def test_async_db_persistor_writes_and_validates(tmp_path: Path) -> None:
    """
    Persist small frames via async DB persistor using sqlite+aiosqlite backend.

    Skips if aiosqlite or SQLAlchemy async engine is unavailable.
    """
    aiosqlite = pytest.importorskip("aiosqlite")  # noqa: F401 - presence is enough
    try:
        from sqlalchemy.ext.asyncio import create_async_engine  # noqa: F401
    except Exception:  # pragma: no cover - environment specific
        pytest.skip("SQLAlchemy async engine not available")

    from ml.observability.async_db_persistence import ObservabilityAsyncDBPersistor
    from ml.observability.pipeline import (
        build_event_correlation,
        build_health_scores,
        build_latency_watermarks,
        build_metrics_collection,
    )

    # Build small frames
    lat = build_latency_watermarks(
        [
            {
                "correlation_id": "CID-1",
                "instrument_id": "EURUSD.SIM",
                "pipeline_stage": "data_ingestion",
                "ts_stage_start": 1000,
                "ts_stage_end": 2000,
            }
        ]
    )
    met = build_metrics_collection(
        [
            {
                "metric_name": "ml_model_inference_latency_seconds",
                "metric_type": "histogram",
                "value": 0.002,
                "timestamp": 1000,
                "labels": {"actor_id": "a1"},
            }
        ]
    )
    cor = build_event_correlation(
        [
            {
                "correlation_id": "CID-1",
                "event_id": "EID-2",
                "parent_event_id": None,
                "instrument_id": "EURUSD.SIM",
                "domain": "data",
                "lineage_depth": 0,
                "ts_event": 1000,
                "propagation_path": ["data"],
            }
        ]
    )
    hea = build_health_scores(
        [
            {
                "component_id": "data_store",
                "health_score": 0.9,
                "subsystem_scores": {"db": 1.0},
                "timestamp": 1000,
                "measurement_window_ms": 1000,
            }
        ]
    )

    db = tmp_path / "obs_async.db"
    per = ObservabilityAsyncDBPersistor(connection_string=f"sqlite+aiosqlite:///{db}")
    written = await per.persist_async({
        "latency": lat,
        "metrics": met,
        "correlation": cor,
        "health": hea,
    })

    assert set(written.keys()) == {"latency", "metrics", "correlation", "health"}

    import sqlalchemy as sa

    eng = sa.create_engine(f"sqlite:///{db}")
    with eng.connect() as conn:
        lat_df = pd.read_sql("select * from obs_latency_watermarks", conn)
        met_df = pd.read_sql("select * from obs_metrics", conn)
        cor_df = pd.read_sql("select * from obs_event_correlation", conn)
        hea_df = pd.read_sql("select * from obs_health_scores", conn)
        assert len(lat_df) == 1 and len(met_df) == 1 and len(cor_df) == 1 and len(hea_df) == 1

