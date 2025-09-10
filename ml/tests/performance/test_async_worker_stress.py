from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ml.observability.async_worker import ObservabilityAsyncWorker
from ml.observability.service import ObservabilityService


@pytest.mark.asyncio
async def test_async_worker_stress_small_queue(tmp_path: Path) -> None:
    """
    Stress the async worker with a small queue and high enqueue rate.

    Ensures some backpressure (drops) occurs while still flushing to disk, and the
    worker stops cleanly within a short bounded time.

    """
    svc = ObservabilityService()
    worker = ObservabilityAsyncWorker(
        service=svc,
        sink="file",
        base_path=tmp_path,
        flush_interval_seconds=0.1,
        queue_maxsize=64,
        component_label="obs_async_worker",
    )

    worker.start()

    # Enqueue more items than capacity in a burst
    drops = 0
    ok = 0
    for i in range(2000):
        # Alternate item types to exercise all code paths
        if (i % 4) == 0:
            res = worker.enqueue_latency(
                correlation_id=f"c{i}",
                instrument_id="EURUSD.SIM",
                pipeline_stage="data_ingestion",
                ts_stage_start=i,
                ts_stage_end=i + 1,
            )
        elif (i % 4) == 1:
            res = worker.enqueue_metric(
                metric_name="nautilus_ml_test_metric",
                metric_type="counter",
                value=1.0,
                timestamp=i,
                labels={"k": "v"},
            )
        elif (i % 4) == 2:
            res = worker.enqueue_correlation(
                correlation_id=f"c{i}",
                event_id=f"e{i}",
                parent_event_id=None,
                instrument_id="EURUSD.SIM",
                domain="data",
                lineage_depth=0,
                ts_event=i,
                propagation_path=["data"],
            )
        else:
            res = worker.enqueue_health(
                component_id="data_store",
                health_score=0.9,
                subsystem_scores={"db": 1.0},
                timestamp=i,
                measurement_window_ms=100,
            )
        if res:
            ok += 1
        else:
            drops += 1

        # Small sleep every so often to yield so worker can drain
        if (i % 128) == 0:
            await asyncio.sleep(0.001)

    # Allow at least one periodic flush
    await asyncio.sleep(0.3)
    await worker.stop(drain=True, timeout=1.0)

    # Expect some drops due to small queue
    assert drops > 0
    assert ok > 0
    # Persistence occurred for non-empty tables
    assert (tmp_path / "latency.jsonl").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "correlation.jsonl").exists()
    assert (tmp_path / "health.jsonl").exists()
