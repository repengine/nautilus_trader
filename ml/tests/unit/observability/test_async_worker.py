from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ml.observability.async_worker import ObservabilityAsyncWorker
from ml.observability.service import ObservabilityService


@pytest.mark.asyncio
async def test_async_worker_file_sink_flush(tmp_path: Path) -> None:
    """
    Async worker enqueues rows and flushes them to JSONL files.
    """
    svc = ObservabilityService()
    worker = ObservabilityAsyncWorker(
        service=svc,
        sink="file",
        base_path=tmp_path,
        flush_interval_seconds=0.1,
        queue_maxsize=64,
    )

    # Start the worker
    worker.start()

    # Enqueue one row per table
    assert worker.enqueue_latency(
        correlation_id="c1",
        instrument_id="EURUSD.SIM",
        pipeline_stage="data_ingestion",
        ts_stage_start=1,
        ts_stage_end=2,
    )
    assert worker.enqueue_metric(
        metric_name="nautilus_ml_test_metric",
        metric_type="counter",
        value=1.0,
        timestamp=2,
        labels={"k": "v"},
    )
    assert worker.enqueue_correlation(
        correlation_id="c1",
        event_id="e1",
        parent_event_id=None,
        instrument_id="EURUSD.SIM",
        domain="data",
        lineage_depth=0,
        ts_event=1,
        propagation_path=["data"],
    )
    assert worker.enqueue_health(
        component_id="data_store",
        health_score=0.9,
        subsystem_scores={"db": 1.0},
        timestamp=3,
        measurement_window_ms=100,
    )

    # Give time for flush
    await asyncio.sleep(0.3)
    await worker.stop(drain=True)

    # Verify files present for non-empty tables
    # File names are table.jsonl when rotate_daily is False
    assert (tmp_path / "latency.jsonl").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "correlation.jsonl").exists()
    assert (tmp_path / "health.jsonl").exists()


def test_async_worker_backpressure_drop(tmp_path: Path) -> None:
    """
    When the queue is full, worker drops new items and returns False.
    """
    svc = ObservabilityService()
    worker = ObservabilityAsyncWorker(
        service=svc,
        sink="file",
        base_path=tmp_path,
        flush_interval_seconds=5.0,  # irrelevant for this test
        queue_maxsize=1,
    )

    # Fill queue with one item; next enqueue should drop
    ok1 = worker.enqueue_health(
        component_id="x",
        health_score=1.0,
        subsystem_scores={"ok": 1.0},
        timestamp=1,
        measurement_window_ms=10,
    )
    ok2 = worker.enqueue_health(
        component_id="y",
        health_score=0.5,
        subsystem_scores={"ok": 0.5},
        timestamp=2,
        measurement_window_ms=10,
    )
    assert ok1 is True
    assert ok2 is False
