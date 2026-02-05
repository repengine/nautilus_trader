from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.cli import hpo_tft
from ml.tests.utils.targets import build_default_target_semantics
from ml.training.datasets.target_generator import build_target_semantics_metadata


def _make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("time_index,y\n0,0\n1,1\n", encoding="utf-8")
    semantics = build_target_semantics_metadata(build_default_target_semantics())
    labels = semantics.get("labels", {})
    target_col = next(iter(labels)) if isinstance(labels, dict) and labels else "y"
    metadata = {
        "dataset_id": "dataset",
        "build_ts": "2025-01-01T00:00:00Z",
        "column_info": {"target_col": target_col},
        "target_semantics": semantics,
    }
    (tmp_path / "dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return dataset


def test_hpo_tft_grid_selects_best_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)

    def _fake_teacher(argv: list[str] | None = None) -> int:
        assert argv is not None
        assert "--dataset_metadata" in argv
        assert "--target_col" in argv
        out_dir = Path(argv[argv.index("--out_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        hidden_size = int(argv[argv.index("--hidden_size") + 1])
        metric = float(hidden_size) / 100.0
        (out_dir / "model_metrics.json").write_text(
            json.dumps({"prx": metric, "logloss": 1.0 / max(hidden_size, 1)}),
            encoding="utf-8",
        )
        np.savez(
            out_dir / "teacher_preds.npz",
            q_val=np.array([0.1, 0.9], dtype=np.float32),
            y_val_true=np.array([0, 1], dtype=np.float32),
        )
        return 0

    monkeypatch.setattr(hpo_tft, "teacher_main", _fake_teacher)

    out_dir = tmp_path / "hpo"
    argv = [
        "--dataset_csv",
        str(dataset),
        "--out_dir",
        str(out_dir),
        "--backend",
        "grid",
        "--hidden_sizes",
        "32,64",
        "--lstm_layers_list",
        "1",
        "--attention_heads",
        "2",
        "--dropouts",
        "0.1",
        "--learning_rates",
        "0.001",
        "--max_encoder_lengths",
        "30",
    ]

    rc = hpo_tft.main(argv)
    assert rc == 0
    summary = json.loads((out_dir / "hpo_summary.json").read_text(encoding="utf-8"))
    assert summary["metric"] == "prx"
    best = summary["best"]
    assert best["params"]["hidden_size"] == 64


def test_hpo_tft_falls_back_to_grid_when_optuna_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)

    def _fake_teacher(argv: list[str] | None = None) -> int:
        assert argv is not None
        assert "--dataset_metadata" in argv
        assert "--target_col" in argv
        out_dir = Path(argv[argv.index("--out_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model_metrics.json").write_text(json.dumps({"logloss": 0.5}), encoding="utf-8")
        return 0

    monkeypatch.setattr(hpo_tft, "teacher_main", _fake_teacher)
    monkeypatch.setattr(hpo_tft, "HAS_OPTUNA", False)

    out_dir = tmp_path / "hpo"
    argv = [
        "--dataset_csv",
        str(dataset),
        "--out_dir",
        str(out_dir),
        "--metric",
        "logloss",
    ]

    rc = hpo_tft.main(argv)
    assert rc == 0
    summary = json.loads((out_dir / "hpo_summary.json").read_text(encoding="utf-8"))
    assert summary["direction"] == "minimize"
