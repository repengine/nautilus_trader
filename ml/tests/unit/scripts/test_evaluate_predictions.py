from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.cli.evaluate_predictions import main as eval_main


def test_evaluate_predictions_from_probs(tmp_path: Path) -> None:
    # Build a simple dataset with perfect separation
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    p = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    npz = tmp_path / "preds_q.npz"
    np.savez_compressed(npz, q_val=p, y_val_true=y)

    out_json = tmp_path / "metrics.json"
    rc = eval_main(["--preds", str(npz), "--out_json", str(out_json)])
    assert rc == 0 and out_json.exists()
    metrics = json.loads(out_json.read_text())
    assert 0.99 < metrics["roc_auc"] <= 1.0
    assert 0.99 < metrics["pr_auc"] <= 1.0
    assert metrics["logloss"] > 0.0


def test_evaluate_predictions_from_logits(tmp_path: Path) -> None:
    # Balanced case: logits symmetric around 0
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    z = np.array([-2.0, 2.0, -1.0, 1.0], dtype=np.float64)
    npz = tmp_path / "preds_z.npz"
    np.savez_compressed(npz, z_val=z, y_val_true=y)

    out_json = tmp_path / "metrics_z.json"
    rc = eval_main(["--preds", str(npz), "--out_json", str(out_json)])
    assert rc == 0 and out_json.exists()
    metrics = json.loads(out_json.read_text())
    assert 0.9 <= metrics["roc_auc"] <= 1.0
    assert metrics["logloss"] > 0.0
