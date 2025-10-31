from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.cli import promote_model_if_metrics_pass as promote


def _write_manifest(path: Path, metrics: dict[str, float]) -> None:
    payload = {
        "cohort_run": {
            "metrics": metrics,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_promotion_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        {
            "roc_auc": 0.62,
            "pr_auc_multiple": 1.7,
            "log_loss": 0.65,
            "economic_slippage_adjusted_sharpe": 0.18,
            "economic_hit_rate": 0.58,
            "economic_turnover": 0.42,
            "economic_max_drawdown": 0.12,
            "stability_ks_statistic": 0.01,
            "stability_calibration_drift": 0.015,
        },
    )

    exit_code = promote.main(
        [
            "--manifest",
            str(manifest),
            "--min-auc",
            "0.55",
            "--min-pr-auc-multiple",
            "1.5",
            "--max-log-loss",
            "0.8",
            "--min-slippage-adjusted-sharpe",
            "0.1",
            "--min-hit-rate",
            "0.5",
            "--max-turnover",
            "0.5",
            "--max-drawdown",
            "0.2",
            "--max-ks-statistic",
            "0.02",
            "--max-calibration-drift",
            "0.02",
        ],
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PROMOTE: pass gates" in captured.out


def test_manifest_missing_metric_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        {
            "roc_auc": 0.60,
            "pr_auc_multiple": 1.8,
            "log_loss": 0.7,
            "economic_slippage_adjusted_sharpe": 0.12,
            "stability_ks_statistic": 0.01,
            "stability_calibration_drift": 0.01,
        },
    )

    exit_code = promote.main(
        [
            "--manifest",
            str(manifest),
            "--min-auc",
            "0.5",
            "--min-hit-rate",
            "0.55",
        ],
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "missing metric 'economic_hit_rate'" in captured.err


def test_teacher_npz_promotion_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    probabilities = np.asarray([0.2, 0.8, 0.75, 0.1], dtype=np.float64)
    labels = np.asarray([0, 1, 1, 0], dtype=np.int_)
    artifact = tmp_path / "teacher_preds.npz"
    np.savez_compressed(artifact, q_val=probabilities, y_val_true=labels)

    exit_code = promote.main(
        [
            "--teacher-npz",
            str(artifact),
            "--min-auc",
            "0.5",
            "--min-pr-auc-multiple",
            "0.8",
        ],
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PROMOTE: pass gates" in captured.out
