"""Microbenchmark guardrails for streaming persistence throughput."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from ml.config.streaming_pipeline import StreamingPersistenceConfig
from ml.consumers.streaming_training_worker import ConsumerFactory
from ml.consumers.streaming_training_worker import StreamingTrainingPersistenceWorker
from ml.dashboard.config import DashboardConfig
from ml.dashboard.controllers import NoopServiceController
from ml.dashboard.service import DashboardService
from ml.tests.fixtures.streaming_events import build_streaming_test_payloads


class _BatchConsumer:
    """Deterministic consumer that replays pre-baked payload batches."""

    def __init__(
        self,
        service_handler: Any,
        batches: Iterable[list[tuple[str, dict[str, Any]]]],
    ) -> None:
        self._service_handler = service_handler
        self._batches = list(batches)

    def poll_once(self, *, count: int, block_ms: int, last_id: str = "$") -> int:  # noqa: ARG002
        if not self._batches:
            return 0
        batch = self._batches.pop(0)
        processed = 0
        for topic, payload in batch:
            self._service_handler(topic, payload)
            processed += 1
        return processed


def _build_consumer_factory(
    batches: Iterable[list[tuple[str, dict[str, Any]]]],
) -> ConsumerFactory:
    def _factory(service, message_bus_config):  # noqa: ANN001
        del message_bus_config
        return _BatchConsumer(service.handle, batches)

    return _factory


def test_streaming_persistence_worker_microbench(tmp_path: Path) -> None:
    """Ensure snapshot backlog processing stays within the latency budget."""
    batch_count = 12
    events_per_batch = 3
    batches: list[list[tuple[str, dict[str, Any]]]] = []

    for idx in range(batch_count):
        payloads = build_streaming_test_payloads(
            dataset_id=f"dataset-{idx}",
            plan_id=f"plan-{idx}",
            parquet_path=tmp_path / f"{idx}.parquet",
        )
        batches.append(
            [
                (f"events.ml.DATASET_PLANNED.{payloads.plan_event.dataset_id}", payloads.plan_message()),
                (
                    f"events.ml.MODEL_TRAINING_COMPLETED.{payloads.result_event.dataset_id}",
                    payloads.result_message(),
                ),
                (f"events.ml.WORKER_HEARTBEAT.{payloads.heartbeat_event.dataset_id}", payloads.heartbeat_message()),
            ],
        )

    config = StreamingPersistenceConfig(
        enabled=True,
        state_path=str(tmp_path / "state.json"),
        batch_size=64,
        block_ms=0,
        poll_interval_seconds=0.0,
    )
    worker = StreamingTrainingPersistenceWorker(
        config=config,
        consumer_factory=_build_consumer_factory(batches),
    )

    t_start = time.perf_counter()
    processed = 0
    while True:
        count = worker.poll_once()
        processed += count
        if count == 0:
            break
    duration = time.perf_counter() - t_start

    assert processed == batch_count * events_per_batch
    assert duration < 0.25
    state_snapshot = worker.service.snapshot()
    assert not worker.service.state_store.outstanding_plan_ids()
    assert len(state_snapshot["plans"]) == batch_count


def _isoformat(minutes: int) -> str:
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    return (base + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")


def _plan_entry(dataset: str, plan_id: str, created_minutes: int, status: str = "success") -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "dataset_id": dataset,
        "status": status,
        "created_at": _isoformat(created_minutes),
        "caps": {},
        "limits": {},
        "metadata_summary": {},
        "streaming_config": {},
        "correlation_id": f"{plan_id}-cid",
        "topic": f"events.ml.DATASET_PLANNED.{dataset}",
    }


def _result_entry(dataset: str, plan_id: str, completed_minutes: int, status: str = "success") -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "dataset_id": dataset,
        "status": status,
        "completed_at": _isoformat(completed_minutes),
        "model_id": "model",
        "metrics": {},
        "artifact_paths": {},
        "telemetry": {},
        "correlation_id": f"{plan_id}-result",
        "topic": f"events.ml.MODEL_TRAINING_COMPLETED.{dataset}",
    }


def _heartbeat_entry(
    dataset: str,
    plan_id: str,
    worker_id: str,
    progress: float,
    timestamp_minutes: int,
) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "dataset_id": dataset,
        "status": "partial",
        "worker_id": worker_id,
        "progress_pct": progress,
        "rss_mb": 128.0,
        "shards_processed": int(progress // 25) + 1,
        "timestamp": _isoformat(timestamp_minutes),
        "correlation_id": f"{worker_id}-{plan_id}",
        "topic": f"events.ml.WORKER_HEARTBEAT.{dataset}",
    }


@pytest.mark.performance
def test_streaming_scaling_regression(tmp_path: Path) -> None:
    """Validate multi-worker scenarios produce expected dashboard summaries."""
    snapshot: dict[str, dict[str, Any]] = {
        "plans": {},
        "results": {},
        "heartbeats": {},
    }

    def _register_plan(dataset: str, plan_id: str, created_minutes: int, status: str = "success") -> None:
        snapshot["plans"][plan_id] = _plan_entry(dataset, plan_id, created_minutes, status=status)

    def _register_result(dataset: str, plan_id: str, completed_minutes: int, status: str = "success") -> None:
        snapshot["results"][plan_id] = _result_entry(dataset, plan_id, completed_minutes, status=status)

    def _register_heartbeat(dataset: str, plan_id: str, worker_id: str, progress: float, minutes: int) -> None:
        key = f"{worker_id}::{plan_id}"
        snapshot["heartbeats"][key] = _heartbeat_entry(dataset, plan_id, worker_id, progress, minutes)

    # Baseline scenario: single worker, no backlog
    _register_plan("stream-baseline", "baseline-plan-1", created_minutes=0)
    _register_result("stream-baseline", "baseline-plan-1", completed_minutes=5)
    _register_heartbeat("stream-baseline", "baseline-plan-1", "worker-baseline-1", progress=100.0, minutes=5)

    # Dual worker scenario
    _register_plan("stream-dual", "dual-plan-1", created_minutes=1)
    _register_result("stream-dual", "dual-plan-1", completed_minutes=4)
    _register_heartbeat("stream-dual", "dual-plan-1", "worker-dual-1", progress=100.0, minutes=4)
    _register_heartbeat("stream-dual", "dual-plan-1", "worker-dual-2", progress=100.0, minutes=4)

    # Four worker scenario
    _register_plan("stream-quad", "quad-plan-1", created_minutes=2)
    _register_result("stream-quad", "quad-plan-1", completed_minutes=6)
    for idx in range(4):
        _register_heartbeat(
            "stream-quad",
            "quad-plan-1",
            f"worker-quad-{idx+1}",
            progress=75.0 + idx * 5.0,
            minutes=6 + idx,
        )

    # Stress scenario: backlog of 9 plans, two workers active
    stress_plan_ids = []
    for idx in range(9):
        plan_id = f"stress-plan-{idx+1}"
        stress_plan_ids.append(plan_id)
        _register_plan("stream-stress", plan_id, created_minutes=idx + 3)
    # Only complete the first plan to leave 8 outstanding? requirement says backlog 9, so skip results
    _register_heartbeat("stream-stress", "stress-plan-1", "worker-stress-1", progress=40.0, minutes=10)
    _register_heartbeat("stream-stress", "stress-plan-2", "worker-stress-2", progress=30.0, minutes=11)

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")

    config = DashboardConfig(streaming_state_path=state_path)
    service = DashboardService(config=config, controller=NoopServiceController())
    state = service.get_streaming_training_state()

    assert state["enabled"] is True
    summary = state["summary"]
    assert summary["dataset_count"] == 4
    assert summary["total_outstanding"] == 9
    assert summary["total_workers"] == 9
    assert summary["datasets_with_backlog"] == 1

    details = state["dataset_details"]
    assert details["stream-baseline"]["outstanding_count"] == 0
    assert details["stream-baseline"]["worker_count"] == 1

    assert details["stream-dual"]["outstanding_count"] == 0
    assert details["stream-dual"]["worker_count"] == 2

    assert details["stream-quad"]["outstanding_count"] == 0
    assert details["stream-quad"]["worker_count"] == 4

    stress_detail = details["stream-stress"]
    assert stress_detail["outstanding_count"] == 9
    assert stress_detail["worker_count"] == 2
    assert len(stress_detail["outstanding_plan_ids"]) == 9
    assert set(stress_detail["plan_ids"]) == set(stress_plan_ids)
