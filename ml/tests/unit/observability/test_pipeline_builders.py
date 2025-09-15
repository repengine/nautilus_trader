from __future__ import annotations

import json
from typing import Any, cast as _cast

import pandas as pd

from ml.observability.pipeline import build_event_correlation
from ml.observability.pipeline import build_health_scores
from ml.observability.pipeline import build_latency_watermarks
from ml.observability.pipeline import build_metrics_collection


class TestLatencyWatermarks:
    def test_watermarks_monotonic_and_latency_match(self, default_instrument_id: Any) -> None:
        rows = [
            {
                "correlation_id": "c1",
                "instrument_id": str(default_instrument_id),
                "pipeline_stage": "data_ingestion",
                "ts_stage_start": 1000,
                "ts_stage_end": 1500,
            },
            {
                "correlation_id": "c1",
                "instrument_id": str(default_instrument_id),
                "pipeline_stage": "feature_computation",
                "ts_stage_start": 1500,
                "ts_stage_end": 1800,
            },
        ]
        df = build_latency_watermarks(rows)
        assert (df["ts_stage_end"] >= df["ts_stage_start"]).all()
        assert (
            df["stage_latency_ns"] == (df["ts_stage_end"] - df["ts_stage_start"]).clip(lower=0)
        ).all()
        s = df["cumulative_latency_ns"].diff().fillna(df["stage_latency_ns"])  # type: ignore[no-redef]
        assert (_cast(Any, (s >= 0)).all())


class TestMetricsCollection:
    def test_metrics_label_encoding_and_types(self) -> None:
        rows = [
            {
                "metric_name": "ml_predictions_total",
                "metric_type": "counter",
                "value": 3,
                "timestamp": 1000,
                "labels": {"model": "m1"},
            },
            {
                "metric_name": "ml_model_inference_latency_seconds",
                "metric_type": "histogram",
                "value": 0.2,
                "timestamp": 1001,
                "labels": {"model": "m1"},
            },
        ]
        df = build_metrics_collection(rows)
        assert df["labels"].apply(lambda s: isinstance(s, str) and json.loads(s)).all()
        assert pd.api.types.is_float_dtype(df["value"]) and pd.api.types.is_integer_dtype(df["timestamp"])


class TestEventCorrelation:
    def test_root_parent_null_and_depth_non_negative(self, default_instrument_id: Any) -> None:
        rows = [
            {
                "correlation_id": "c1",
                "event_id": "e1",
                "parent_event_id": None,
                "instrument_id": str(default_instrument_id),
                "domain": "data",
                "lineage_depth": 0,
                "ts_event": 1000,
                "propagation_path": ["data"],
            },
            {
                "correlation_id": "c1",
                "event_id": "e2",
                "parent_event_id": "e1",
                "instrument_id": str(default_instrument_id),
                "domain": "features",
                "lineage_depth": 1,
                "ts_event": 1500,
                "propagation_path": ["data", "features"],
            },
        ]
        rows = _cast(list[dict[str, Any]], rows)
        df = build_event_correlation(rows)
        assert (df["lineage_depth"] >= 0).all()
        # propagation_path is JSON string
        assert df["propagation_path"].apply(lambda s: isinstance(s, str) and json.loads(s)).all()


class TestHealthScores:
    def test_health_score_bounds_and_json_scores(self) -> None:
        rows = [
            {
                "component_id": "feature_store",
                "health_score": 1.2,  # will be clipped to 1.0
                "subsystem_scores": {"db": 0.9, "cache": 0.95},
                "timestamp": 1000,
                "measurement_window_ms": 5000,
            },
        ]
        df = build_health_scores(rows)
        assert float(df.iloc[0]["health_score"]) <= 1.0
        assert isinstance(df.iloc[0]["subsystem_scores"], str)
        assert json.loads(df.iloc[0]["subsystem_scores"]) == {"db": 0.9, "cache": 0.95}
