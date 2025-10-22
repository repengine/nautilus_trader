from __future__ import annotations

import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy import inspect

from ml.observability.migrations import apply_observability_indices


_DEFAULT_URL = "postgresql://postgres:postgres@localhost:5434/nautilus_test"


def _pg_available(url: str) -> bool:
    try:
        eng = create_engine(url)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.usefixtures("clean_postgres_db_module"),
]


from typing import Any


@pytest.mark.skipif(
    not _pg_available(os.getenv("DATABASE_URL", _DEFAULT_URL)),
    reason="PostgreSQL not reachable",
)
def test_apply_observability_indices_creates_brin_and_composites(
    default_instrument_id: Any,
) -> None:
    url = os.getenv("DATABASE_URL", _DEFAULT_URL)
    eng = create_engine(url)

    # Ensure tables exist (empty frames are fine)
    with eng.begin() as conn:
        pd.DataFrame(
            [
                {
                    "correlation_id": "c1",
                    "instrument_id": str(default_instrument_id),
                    "pipeline_stage": "data_ingestion",
                    "ts_stage_start": 1,
                    "ts_stage_end": 2,
                    "stage_latency_ns": 1,
                    "cumulative_latency_ns": 1,
                },
            ],
        ).to_sql("obs_latency_watermarks", conn, if_exists="append", index=False)
        pd.DataFrame(
            [
                {
                    "metric_name": "m",
                    "metric_type": "counter",
                    "value": 1.0,
                    "timestamp": 1,
                    "labels": "{}",
                },
            ],
        ).to_sql(
            "obs_metrics",
            conn,
            if_exists="append",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "correlation_id": "c1",
                    "event_id": "e1",
                    "parent_event_id": None,
                    "instrument_id": str(default_instrument_id),
                    "domain": "data",
                    "lineage_depth": 0,
                    "ts_event": 1,
                    "propagation_path": "[]",
                },
            ],
        ).to_sql("obs_event_correlation", conn, if_exists="append", index=False)
        pd.DataFrame(
            [
                {
                    "component_id": "data_store",
                    "health_score": 1.0,
                    "subsystem_scores": "{}",
                    "timestamp": 1,
                    "measurement_window_ms": 1000,
                    "alert_threshold": 0.8,
                },
            ],
        ).to_sql("obs_health_scores", conn, if_exists="append", index=False)

    apply_observability_indices(eng)

    inspector = inspect(eng)

    # Check representative indexes exist (names are deterministic)
    idx_names = {
        n
        for n in (i.get("name") for i in inspector.get_indexes("obs_event_correlation"))
        if isinstance(n, str)
    }
    assert "obs_event_correlation_ts_event_brin" in idx_names or any(
        n.endswith("ts_event_brin") for n in idx_names
    )
    assert "obs_event_correlation_instrument_ts_idx" in idx_names

    idx_names_metrics = {
        n
        for n in (i.get("name") for i in inspector.get_indexes("obs_metrics"))
        if isinstance(n, str)
    }
    assert "obs_metrics_timestamp_brin" in idx_names_metrics or any(
        n.endswith("timestamp_brin") for n in idx_names_metrics
    )
    assert "obs_metrics_name_ts_idx" in idx_names_metrics
