from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandera.typing import Series

globals()["Series"] = Series  # Ensure Series available in pytest-xdist worker globals

from ml.observability.persistence import ObservabilityPersistor
from ml.observability.pipeline import build_event_correlation
from ml.observability.pipeline import build_health_scores
from ml.observability.pipeline import build_latency_watermarks
from ml.observability.pipeline import build_metrics_collection
from ml.tests.contracts.test_observability_pipeline_schemas import EventCorrelationSchema
from ml.tests.contracts.test_observability_pipeline_schemas import HealthScoreAggregationSchema
from ml.tests.contracts.test_observability_pipeline_schemas import LatencyWatermarkSchema
from ml.tests.contracts.test_observability_pipeline_schemas import MetricsCollectionSchema


def _read_jsonl(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, orient="records", lines=True)
    # Normalize integer timestamp-like columns for schema validation
    for col in (
        "timestamp",
        "ts_event",
        "ts_stage_start",
        "ts_stage_end",
        "measurement_window_ms",
    ):
        if col in df.columns:
            df[col] = df[col].astype("int64")
    return df


def test_persisted_jsonl_conforms_to_contracts(tmp_path: Path) -> None:
    # Build contract-compliant frames using canonical names
    lat = build_latency_watermarks(
        [
            {
                "correlation_id": "00000000-0000-0000-0000-000000000001",
                "instrument_id": "EURUSD.SIM",
                "pipeline_stage": "data_ingestion",
                "ts_stage_start": 1609459200000000000,
                "ts_stage_end": 1609459200001000000,
            },
        ],
    )
    met = build_metrics_collection(
        [
            {
                "metric_name": "ml_feature_computation_latency_seconds",
                "metric_type": "histogram",
                "value": 0.001,
                "timestamp": 1609459200001000000,
                "labels": {"instrument_id": "EURUSD.SIM", "domain": "features"},
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
                "domain": "features",
                "lineage_depth": 0,
                "ts_event": 1609459200000000000,
                "propagation_path": ["data", "features"],
            },
        ],
    )
    hea = build_health_scores(
        [
            {
                "component_id": "feature_store",
                "health_score": 0.95,
                "subsystem_scores": {"db": 1.0},
                "timestamp": 1609459200002000000,
                "measurement_window_ms": 1000,
            },
        ],
    )

    sink = ObservabilityPersistor(base_path=tmp_path, file_format="jsonl")
    out = sink.persist(
        {
            "latency": lat,
            "metrics": met,
            "correlation": cor,
            "health": hea,
        },
    )

    # All files written
    assert set(out.keys()) == {"latency", "metrics", "correlation", "health"}

    # Validate JSONL contents against Pandera schemas
    LatencyWatermarkSchema.validate(_read_jsonl(out["latency"]))
    MetricsCollectionSchema.validate(_read_jsonl(out["metrics"]))
    EventCorrelationSchema.validate(_read_jsonl(out["correlation"]))
    HealthScoreAggregationSchema.validate(_read_jsonl(out["health"]))
