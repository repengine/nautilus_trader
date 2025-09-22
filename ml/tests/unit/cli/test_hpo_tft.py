from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.cli import hpo_tft


def _make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("time_index,y\n0,0\n1,1\n", encoding="utf-8")
    return dataset


def test_hpo_tft_grid_selects_best_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)

    def _fake_teacher(argv: list[str] | None = None) -> int:
        assert argv is not None
        out_dir = Path(argv[argv.index("--out_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        hidden_size = int(argv[argv.index("--hidden_size") + 1])
        metric = float(hidden_size) / 100.0
        (out_dir / "model_metrics.json").write_text(
            json.dumps({"prx": metric, "logloss": 1.0 / max(hidden_size, 1)}),
            encoding="utf-8",
        )
        np.savez(out_dir / "teacher_preds.npz", q_val=np.array([metric], dtype=np.float32), y_val_true=np.array([1], dtype=np.float32))
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
