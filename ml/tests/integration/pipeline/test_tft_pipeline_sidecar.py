from __future__ import annotations

import json
from pathlib import Path

import polars as pl

import ml.pipelines.tft_train_distill as pipeline_mod


def _stub_build_main(argv: list[str] | None = None) -> int:  # type: ignore[no-redef]
    # Create dataset and a sidecar feature_set.json
    out_idx = argv.index("--out_dir") + 1 if argv and "--out_dir" in argv else -1
    out_dir = Path(argv[out_idx]) if out_idx > 0 else Path("/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "time_index": [0, 1, 2],
            "instrument_id": ["SPY", "SPY", "SPY"],
            "f1": [0.1, 0.2, 0.3],
            "y": [0, 1, 0],
        },
    )
    df.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC"))).write_parquet(
        str(out_dir / "dataset.parquet"),
    )
    df.write_csv(str(out_dir / "dataset.csv"))
    sidecar = {
        "feature_set_id": "feature_set_123",
        "feature_registry_dir": str((out_dir / "features").resolve()),
        "feature_names": ["f1"],
        "flags": {"include_macro": False, "include_micro": False, "include_l2": False},
    }
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "feature_set.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f)
    return 0


def _stub_tft_main(argv: list[str] | None = None) -> int:  # type: ignore[no-redef]
    # Must include feature registry args populated from sidecar
    assert argv is not None and "--feature_set_id" in argv and "--feature_registry_dir" in argv
    # Write teacher preds
    out_idx = argv.index("--out_dir") + 1
    out_dir = Path(argv[out_idx])
    import numpy as np

    np.savez_compressed(
        out_dir / "teacher_preds.npz",
        q_train=np.array([0.5, 0.5, 0.5]),
        y_val_true=np.array([0, 1, 0]),
    )
    return 0


def _stub_distill_main(argv: list[str] | None = None) -> int:  # type: ignore[no-redef]
    # Must include feature registry args populated from sidecar
    assert argv is not None and "--feature_set_id" in argv and "--feature_registry_dir" in argv
    return 0


def test_pipeline_reads_sidecar_when_args_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("ml.scripts.build_tft_dataset.main", _stub_build_main)
    monkeypatch.setattr("ml.training.teacher.tft_cli.main", _stub_tft_main)
    monkeypatch.setattr("ml.training.distillation.cli.main", _stub_distill_main)

    args = [
        "--data_dir",
        str(tmp_path),
        "--symbols",
        "SPY",
        "--out_dir",
        str(tmp_path / "out"),
        "--horizon_minutes",
        "1",
        "--threshold",
        "0.001",
        "--lookback_periods",
        "1",
        "--train_teacher",
        "--teacher_model_id",
        "tft_v1",
        # Note: no feature_registry_dir/feature_set_id passed
        "--model_registry_dir",
        str(tmp_path / "models"),
        "--student_model_id",
        "student_v1",
    ]
    rc = pipeline_mod.main(args)
    assert rc == 0
