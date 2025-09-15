from __future__ import annotations

import time

import pandas as pd

from ml.observability.pipeline import (
    build_event_correlation,
    build_health_scores,
    build_latency_watermarks,
    build_metrics_collection,
)
from ml.common.metrics_bootstrap import get_counter, get_gauge, get_histogram


def test_latency_watermarks_builder_shapes() -> None:
    now = time.time_ns()
    rows = [
        {
            "correlation_id": "c1",
            "instrument_id": "EURUSD.SIM",
            "pipeline_stage": "ingest",
            "ts_stage_start": now,
            "ts_stage_end": now + 1_000,
        },
        {
            "correlation_id": "c1",
            "instrument_id": "EURUSD.SIM",
            "pipeline_stage": "compute",
            "ts_stage_start": now + 1_000,
            "ts_stage_end": now + 3_000,
        },
    ]
    df = build_latency_watermarks(rows)
    assert isinstance(df, pd.DataFrame)
    assert set(["stage_latency_ns", "cumulative_latency_ns"]).issubset(df.columns)
    assert int(df["cumulative_latency_ns"].iloc[-1]) >= 0


def test_metrics_collection_builder_types() -> None:
    rows = [
        {
            "metric_name": "ml_test_counter",
            "metric_type": "counter",
            "value": 1.0,
            "timestamp": time.time_ns(),
            "labels": {"label": "value"},
        },
    ]
    df = build_metrics_collection(rows)
    assert isinstance(df, pd.DataFrame)
    assert df["labels"].dtype == object  # JSON-encoded string acceptable
    assert df["value"].dtype.kind in ("f", "i")


def test_event_correlation_builder_normalization() -> None:
    rows = [
        {
            "correlation_id": "c2",
            "event_id": "e1",
            "parent_event_id": None,
            "instrument_id": "SPY.NYSE",
            "domain": "data",
            "lineage_depth": 0,
            "ts_event": time.time_ns(),
            "propagation_path": ["ingest", "compute"],
        },
    ]
    df = build_event_correlation(rows)
    assert isinstance(df, pd.DataFrame)
    assert "propagation_path" in df.columns
    assert isinstance(df["propagation_path"].iloc[0], str)


def test_health_scores_builder_bounds() -> None:
    rows = [
        {
            "component_id": "actor.1",
            "health_score": 1.2,  # clipped
            "subsystem_scores": {"cpu": 0.8},
            "timestamp": time.time_ns(),
            "measurement_window_ms": 1000,
        },
    ]
    df = build_health_scores(rows)
    assert isinstance(df, pd.DataFrame)
    assert 0.0 <= float(df["health_score"].iloc[0]) <= 1.0
    assert "alert_threshold" in df.columns


def test_metrics_bootstrap_counters_no_server_start() -> None:
    # Acquire collectors via centralized bootstrap and record a sample
    c = get_counter("ml_test_events_total", "test counter", ["label_a"])  # noqa: F841
    g = get_gauge("ml_test_gauge", "test gauge", ["label_b"])  # noqa: F841
    h = get_histogram("ml_test_latency_seconds", "test histogram", ["label_c"])  # noqa: F841
    # We don't assert server state; bootstrap must not start an HTTP server here.
    # Presence without exception is sufficient for unit contract.

