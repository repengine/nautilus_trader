from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml.dashboard.config import DashboardConfig
from ml.dashboard.controllers import NoopServiceController
from ml.dashboard.service import DashboardService


class _StubObservabilityService:
    def add_metric(
        self,
        *,
        metric_name: str,
        metric_type: str,
        value: float,
        timestamp: int,
        labels: dict[str, Any] | None = None,
    ) -> None:
        self.last_metric = (metric_name, metric_type, value, timestamp, labels)


@pytest.mark.integration
def test_dashboard_streaming_state_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_path = tmp_path / "streaming_state.json"
    snapshot = {
        "plans": {
            "plan-1": {
                "plan_id": "plan-1",
                "dataset_id": "dataset",
                "status": "success",
                "created_at": "2024-01-01T00:00:00Z",
                "caps": {},
                "limits": {},
                "metadata_summary": {},
                "streaming_config": {},
                "correlation_id": "cid-1",
                "topic": "events.ml.DATASET_PLANNED.dataset",
            },
        },
        "results": {
            "plan-1": {
                "plan_id": "plan-1",
                "dataset_id": "dataset",
                "status": "success",
                "completed_at": "2024-01-01T01:00:00Z",
                "model_id": "model",
                "metrics": {},
                "artifact_paths": {},
                "telemetry": {},
                "correlation_id": "cid-2",
                "topic": "events.ml.MODEL_TRAINING_COMPLETED.dataset",
            },
        },
        "heartbeats": {
            "worker::plan-1": {
                "plan_id": "plan-1",
                "dataset_id": "dataset",
                "status": "partial",
                "worker_id": "worker",
                "progress_pct": 50.0,
                "rss_mb": 128.0,
                "shards_processed": 1,
                "timestamp": "2024-01-01T00:30:00Z",
                "correlation_id": "cid-3",
                "topic": "events.ml.WORKER_HEARTBEAT.dataset",
            },
        },
        "stream_cursor": "9-0",
    }
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    monkeypatch.setattr(
        "ml.dashboard.service.ObservabilityService",
        _StubObservabilityService,
    )

    config = DashboardConfig(streaming_state_path=snapshot_path)
    service = DashboardService(config=config, controller=NoopServiceController())
    state = service.get_streaming_training_state()

    assert state["enabled"] is True
    assert state["path"] == str(snapshot_path)
    assert "plan-1" in state["plans"]
    assert state["datasets"]["dataset"] == ["plan-1"]
    summary = state["summary"]
    assert summary["dataset_count"] == 1
    assert summary["total_outstanding"] == 0
    details = state["dataset_details"]["dataset"]
    assert details["plan_ids"] == ["plan-1"]
    assert details["outstanding_count"] == 0
    assert details["worker_count"] == 1
    latest_plan = details["latest_plan"]
    assert latest_plan is not None
    assert latest_plan["plan_id"] == "plan-1"
    latest_result = details["latest_result"]
    assert latest_result is not None
    assert latest_result["plan_id"] == "plan-1"
    assert state["stream_cursor"] == "9-0"
