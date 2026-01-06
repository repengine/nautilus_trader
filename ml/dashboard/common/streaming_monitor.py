"""
Streaming training state monitor for the dashboard.

Tracks plan/result/heartbeat events and persists a lightweight snapshot so the
dashboard can render streaming training status without loading full artifacts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from ml.dashboard.config import DashboardConfig


logger = logging.getLogger(__name__)


@dataclass
class StreamingMonitorComponent:
    """
    Maintain streaming training state for dashboard monitoring.

    This component aggregates streaming plan/result/heartbeat events and stores
    a compact snapshot for UI consumption. It optionally persists the snapshot
    to disk when ``streaming_state_path`` is configured.
    """

    config: DashboardConfig
    _state: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize in-memory streaming state."""
        self._state = {
            "plans": {},
            "results": {},
            "heartbeats": {},
            "datasets": {},
            "outstanding_plan_ids": [],
            "stream_cursor": None,
        }

    def get_streaming_training_state(self) -> dict[str, Any]:
        """
        Load streaming training state from an optional snapshot file.

        Returns:
            Structured state containing plans, results, heartbeats, per-dataset
            summaries, and the last cursor.
        """
        snapshot: dict[str, Any] = dict(self._state)
        snapshot_path: Path | None = None
        path = self.config.streaming_state_path
        if path is not None:
            snapshot_path = Path(path)
            if snapshot_path.exists():
                try:
                    loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        snapshot.update(loaded)
                except Exception:
                    logger.debug("failed to read streaming state snapshot", exc_info=True)

        plans = cast(dict[str, Any], snapshot.get("plans") or {})
        results = cast(dict[str, Any], snapshot.get("results") or {})
        heartbeats = cast(dict[str, Any], snapshot.get("heartbeats") or {})
        stream_cursor = snapshot.get("stream_cursor")

        datasets_value = snapshot.get("datasets")
        datasets: dict[str, list[str]]
        if isinstance(datasets_value, dict) and datasets_value:
            datasets = {}
            for dataset_id, plan_ids in datasets_value.items():
                if not isinstance(plan_ids, list):
                    continue
                datasets[str(dataset_id)] = [str(plan_id) for plan_id in plan_ids]
        else:
            datasets = {}
            for plan_id, plan in plans.items():
                if not isinstance(plan, dict):
                    continue
                dataset_id = str(plan.get("dataset_id", "") or "") or "unknown"
                datasets.setdefault(dataset_id, []).append(str(plan_id))

        outstanding_plan_ids = [plan_id for plan_id in plans if plan_id not in results]

        dataset_details: dict[str, dict[str, Any]] = {}
        total_worker_ids: set[str] = set()
        datasets_with_backlog = 0
        for dataset_id, plan_ids in datasets.items():
            worker_ids: set[str] = set()
            for heartbeat in heartbeats.values():
                if not isinstance(heartbeat, dict):
                    continue
                if str(heartbeat.get("dataset_id", "") or "") != dataset_id:
                    continue
                worker_id = heartbeat.get("worker_id")
                if worker_id:
                    worker_ids.add(str(worker_id))
            total_worker_ids.update(worker_ids)

            latest_plan = None
            if plan_ids:
                candidate = plans.get(plan_ids[-1])
                if isinstance(candidate, dict):
                    latest_plan = candidate

            latest_result = next(
                (results[plan_id] for plan_id in reversed(plan_ids) if plan_id in results),
                None,
            )
            outstanding_plan_ids_for_dataset = [
                plan_id for plan_id in plan_ids if plan_id not in results
            ]
            outstanding_count = len(outstanding_plan_ids_for_dataset)
            if outstanding_count:
                datasets_with_backlog += 1
            dataset_details[dataset_id] = {
                "plan_ids": plan_ids,
                "outstanding_plan_ids": outstanding_plan_ids_for_dataset,
                "latest_plan": latest_plan,
                "latest_result": latest_result,
                "outstanding_count": outstanding_count,
                "worker_count": len(worker_ids),
            }

        summary = {
            "dataset_count": len(datasets),
            "total_outstanding": len(outstanding_plan_ids),
            "total_workers": len(total_worker_ids),
            "datasets_with_backlog": datasets_with_backlog,
        }

        return {
            "enabled": True,
            "path": str(snapshot_path) if snapshot_path is not None else None,
            "plans": plans,
            "results": results,
            "heartbeats": heartbeats,
            "datasets": datasets,
            "outstanding_plan_ids": outstanding_plan_ids,
            "dataset_details": dataset_details,
            "summary": summary,
            "stream_cursor": stream_cursor,
        }

    def process_streaming_event(self, topic: str, message: dict[str, Any]) -> None:
        """
        Update streaming state with an incoming event payload.

        Args:
            topic: Event topic containing the stage identifier.
            message: Event payload (dict).
        """
        plans = self._state.setdefault("plans", {})
        results = self._state.setdefault("results", {})
        heartbeats = self._state.setdefault("heartbeats", {})
        datasets = self._state.setdefault("datasets", {})

        topic_lower = topic.lower()
        plan_id = str(message.get("plan_id", "") or "")
        dataset_id = str(message.get("dataset_id", "") or "")

        if "dataset_planned" in topic_lower and plan_id:
            plans[plan_id] = message
            dataset_plans = datasets.setdefault(dataset_id or "unknown", [])
            if plan_id not in dataset_plans:
                dataset_plans.append(plan_id)
        elif "model_training_completed" in topic_lower and plan_id:
            payload = message.get("payload")
            if isinstance(payload, dict):
                flattened: dict[str, Any] = {**message, **payload}
                telemetry = payload.get("telemetry")
                if isinstance(telemetry, dict):
                    flattened["telemetry"] = telemetry
                    for key, value in telemetry.items():
                        flattened.setdefault(key, value)
                    caps_payload = telemetry.get("caps")
                    if isinstance(caps_payload, Mapping):
                        for key, value in caps_payload.items():
                            flattened.setdefault(str(key), value)
                metrics_obj = flattened.get("metrics")
                if isinstance(metrics_obj, Mapping):
                    calibration_summary: list[dict[str, Any]] = []
                    for prefix, kind in (
                        ("temperature_calibration", "Temperature"),
                        ("platt_calibration", "Platt"),
                        ("isotonic_calibration", "Isotonic"),
                    ):
                        entry: dict[str, Any] = {"kind": kind}
                        for key, value in metrics_obj.items():
                            if key.startswith(f"{prefix}_"):
                                entry[key[len(prefix) + 1 :]] = value
                        if len(entry) > 1:
                            calibration_summary.append(entry)
                    if calibration_summary:
                        flattened["calibration_summary"] = calibration_summary
                results[plan_id] = flattened
            else:
                results[plan_id] = message
        elif "worker_heartbeat" in topic_lower and plan_id:
            heartbeats[plan_id] = message

        if "cursor" in message:
            self._state["stream_cursor"] = message.get("cursor")

        path = self.config.streaming_state_path
        if path:
            try:
                Path(path).write_text(
                    json.dumps(self._state, default=str),
                    encoding="utf-8",
                )
            except Exception:
                logger.debug("failed to persist streaming state snapshot", exc_info=True)


__all__ = ["StreamingMonitorComponent"]
