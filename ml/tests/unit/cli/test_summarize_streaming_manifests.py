from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typing import Mapping

from ml.scripts.summarize_streaming_manifests import summarize_manifests


def _write_manifest(
    path: Path,
    *,
    plan_id: str,
    completed_at: str,
    metrics: Mapping[str, float],
) -> None:
    payload = {
        "cohort_run": {
            "plan_id": plan_id,
            "dataset_id": "full_tft_95",
            "completed_at": completed_at,
            "metrics": dict(metrics),
            "telemetry": {
                "selected_rows": {
                    "train": 10,
                    "validation": 5,
                },
                "resources": {
                    "max_gpu_memory_mb": 512.0,
                },
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_manifests_orders_by_completion(tmp_path: Path) -> None:
    first_path = tmp_path / "plan_a_manifest.json"
    second_path = tmp_path / "plan_b_manifest.json"
    _write_manifest(
        first_path,
        plan_id="plan_a",
        completed_at=datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
        metrics={
            "roc_auc": 0.5,
            "log_loss": 0.42,
            "temperature_calibration_log_loss": 0.40,
            "temperature_calibration_log_loss_delta": -0.02,
            "platt_calibration_log_loss": 0.41,
            "isotonic_calibration_log_loss": 0.415,
        },
    )
    _write_manifest(
        second_path,
        plan_id="plan_b",
        completed_at=datetime(2024, 2, 1, tzinfo=UTC).isoformat(),
        metrics={
            "roc_auc": 0.6,
            "log_loss": 0.35,
            "temperature_calibration_log_loss": 0.345,
            "platt_calibration_log_loss": 0.348,
            "isotonic_calibration_log_loss": 0.349,
        },
    )

    summaries = summarize_manifests(tmp_path, limit=None)

    assert [item.plan_id for item in summaries] == ["plan_b", "plan_a"]
    assert summaries[0].roc_auc == 0.6
    assert summaries[0].temperature_calibration_log_loss == 0.345
    assert summaries[0].temperature_calibration_log_loss_delta == 0.345 - 0.35
    assert summaries[0].platt_calibration_log_loss == 0.348
    assert summaries[0].isotonic_calibration_log_loss == 0.349
