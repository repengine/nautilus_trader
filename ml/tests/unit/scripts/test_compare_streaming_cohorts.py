from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from ml.scripts.compare_streaming_cohorts import CohortStats
from ml.scripts.compare_streaming_cohorts import _metric_value
from ml.scripts.compare_streaming_cohorts import compare_cohorts
from ml.scripts.compare_streaming_cohorts import collect_stats
from ml.scripts.compare_streaming_cohorts import main


def _write_manifest(
    path: Path,
    *,
    plan_id: str,
    roc_auc: float,
    logits_path: Path,
    fallback_join: bool,
) -> None:
    payload = {
        "cohort_run": {
            "plan_id": plan_id,
            "metrics": {
                "roc_auc": roc_auc,
            },
            "artifact_paths": {
                "logits": str(logits_path),
            },
            "telemetry": {
                "validation_returns": {
                    "fallback_join": fallback_join,
                    "mismatch_count": 0,
                    "missing_count": 0,
                },
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_logits(path: Path, *, z_val: np.ndarray, val_returns: np.ndarray) -> None:
    np.savez(path, z_val=z_val, val_returns=val_returns)


def test_compare_cohorts_orders_top_and_bottom(tmp_path: Path) -> None:
    logits_hi = tmp_path / "plan_hi_logits.npz"
    logits_lo = tmp_path / "plan_lo_logits.npz"
    _write_logits(logits_hi, z_val=np.array([0.0, 1.0]), val_returns=np.array([0.1, -0.1]))
    _write_logits(logits_lo, z_val=np.array([-0.5, -0.6]), val_returns=np.array([0.0, 0.0]))

    manifest_hi = tmp_path / "plan_hi_manifest.json"
    manifest_lo = tmp_path / "plan_lo_manifest.json"
    _write_manifest(manifest_hi, plan_id="plan_hi", roc_auc=0.7, logits_path=logits_hi, fallback_join=False)
    _write_manifest(manifest_lo, plan_id="plan_lo", roc_auc=0.5, logits_path=logits_lo, fallback_join=True)

    result = compare_cohorts(tmp_path, metric="roc_auc", count=1)
    assert result["top"][0]["plan_id"] == "plan_hi"
    assert result["bottom"][0]["plan_id"] == "plan_lo"
    assert result["top"][0]["z_val_std"] is not None
    assert result["bottom"][0]["fallback_join"] is True


def test_main_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    logits = tmp_path / "plan_logits.npz"
    _write_logits(logits, z_val=np.array([0.1]), val_returns=np.array([0.2]))
    manifest = tmp_path / "plan_manifest.json"
    _write_manifest(manifest, plan_id="plan", roc_auc=0.6, logits_path=logits, fallback_join=False)

    exit_code = main(
        [
            "--manifest-dir",
            str(tmp_path),
            "--metric",
            "roc_auc",
            "--count",
            "1",
            "--pretty",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    assert payload["metric"] == "roc_auc"
    assert payload["top"][0]["plan_id"] == "plan"


def test_collect_stats_resolves_container_mount(tmp_path: Path) -> None:
    logits = tmp_path / "plan_logits.npz"
    _write_logits(logits, z_val=np.array([0.2, 0.4]), val_returns=np.array([0.0, 0.1]))
    manifest = tmp_path / "plan_manifest.json"
    container_path = "/app/irrelevant/plan_logits.npz"
    _write_manifest(manifest, plan_id="plan", roc_auc=0.6, logits_path=Path(container_path), fallback_join=False)

    stats = collect_stats(manifest, metric="roc_auc")
    assert stats.logits_path == logits
    assert stats.z_val_std is not None


def test_compare_cohorts_error_when_metric_missing(tmp_path: Path) -> None:
    logits = tmp_path / "plan_logits.npz"
    _write_logits(logits, z_val=np.array([0.1]), val_returns=np.array([0.2]))
    manifest = tmp_path / "plan_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cohort_run": {
                    "plan_id": "plan",
                    "metrics": {},
                    "artifact_paths": {"logits": str(logits)},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        compare_cohorts(tmp_path, metric="roc_auc", count=1)


def test_metric_value_guard() -> None:
    stats = CohortStats(
        plan_id="plan",
        metric_value=None,
        metric_name="roc_auc",
        manifest_path=Path("manifest"),
        logits_path=None,
        z_val_mean=None,
        z_val_std=None,
        val_returns_mean=None,
        val_returns_std=None,
        fallback_join=None,
        mismatch_count=None,
        missing_count=None,
    )

    with pytest.raises(ValueError):
        _metric_value(stats)


def test_collect_stats_validates_cohort_mapping(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid_manifest.json"
    manifest.write_text(json.dumps({"cohort_run": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        collect_stats(manifest, metric="roc_auc")


def test_collect_stats_handles_non_mapping_metrics(tmp_path: Path) -> None:
    logits = tmp_path / "plan_logits.npz"
    _write_logits(logits, z_val=np.array([]), val_returns=np.array([]))
    manifest = tmp_path / "plan_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cohort_run": {
                    "plan_id": "plan",
                    "metrics": [],
                    "artifact_paths": [],
                }
            }
        ),
        encoding="utf-8",
    )

    stats = collect_stats(manifest, metric="roc_auc")
    assert stats.metric_value is None
    assert stats.logits_path is None


def test_collect_stats_prefers_host_relative_when_available(tmp_path: Path) -> None:
    host_dir = Path.cwd() / "tmp_streaming_host"
    host_dir.mkdir(exist_ok=True)
    logits = host_dir / "plan_logits.npz"
    _write_logits(logits, z_val=np.array([0.1, 0.2]), val_returns=np.array([0.0, 0.0]))
    manifest = tmp_path / "plan_manifest.json"
    container_path = f"/app/{host_dir.name}/plan_logits.npz"
    _write_manifest(manifest, plan_id="plan", roc_auc=0.5, logits_path=Path(container_path), fallback_join=False)

    try:
        stats = collect_stats(manifest, metric="roc_auc")
        assert stats.logits_path is not None
        assert stats.logits_path.resolve() == logits.resolve()
    finally:
        shutil.rmtree(host_dir, ignore_errors=True)
