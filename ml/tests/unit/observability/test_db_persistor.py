from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.observability.db_persistence import ObservabilityDBPersistor
from ml.observability.pipeline import build_event_correlation
from ml.observability.pipeline import build_health_scores
from ml.observability.pipeline import build_latency_watermarks
from ml.observability.pipeline import build_metrics_collection


def test_db_persistor_writes_and_validates(tmp_path: Path) -> None:
    # Optional dependency: skip if pandera not available (schemas rely on it)
    pytest.importorskip("pandera")
    from ml.tests.contracts.test_observability_pipeline_schemas import (
        EventCorrelationSchema,
        HealthScoreAggregationSchema,
        LatencyWatermarkSchema,
        MetricsCollectionSchema,
    )

    # Build small frames
    lat = build_latency_watermarks(
        [
            {
                "correlation_id": "00000000-0000-0000-0000-000000000001",
                "instrument_id": "EURUSD.SIM",
                "pipeline_stage": "data_ingestion",
                "ts_stage_start": 1000,
                "ts_stage_end": 2000,
            },
        ],
    )
    met = build_metrics_collection(
        [
            {
                "metric_name": "ml_model_inference_latency_seconds",
                "metric_type": "histogram",
                "value": 0.002,
                "timestamp": 1000,
                "labels": {"actor_id": "a1"},
            },
        ],
    )
    cor = build_event_correlation(
        [
            {
                "correlation_id": "00000000-0000-0000-0000-000000000001",
                "event_id": "00000000-0000-0000-0000-000000000002",
                "parent_event_id": None,
                "instrument_id": "EURUSD.SIM",
                "domain": "data",
                "lineage_depth": 0,
                "ts_event": 1000,
                "propagation_path": ["data"],
            },
        ],
    )
    hea = build_health_scores(
        [
            {
                "component_id": "data_store",
                "health_score": 0.9,
                "subsystem_scores": {"db": 1.0},
                "timestamp": 1000,
                "measurement_window_ms": 1000,
            },
        ],
    )

    db = tmp_path / "obs.db"
    per = ObservabilityDBPersistor(connection_string=f"sqlite:///{db}")
    written = per.persist(
        {
            "latency": lat,
            "metrics": met,
            "correlation": cor,
            "health": hea,
        },
    )
    assert set(written.keys()) == {"latency", "metrics", "correlation", "health"}

    import sqlalchemy as sa

    eng = sa.create_engine(f"sqlite:///{db}")
    with eng.connect() as conn:
        lat_df = pd.read_sql("select * from obs_latency_watermarks", conn)
        LatencyWatermarkSchema.validate(lat_df)
        met_df = pd.read_sql("select * from obs_metrics", conn)
        MetricsCollectionSchema.validate(met_df)
        cor_df = pd.read_sql("select * from obs_event_correlation", conn)
        EventCorrelationSchema.validate(cor_df)
        hea_df = pd.read_sql("select * from obs_health_scores", conn)
        HealthScoreAggregationSchema.validate(hea_df)
