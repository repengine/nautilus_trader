from __future__ import annotations

from pathlib import Path

import polars as pl

import ml.pipelines.tft_train_distill as pipeline_mod


_CALLED = {"tft_args": None, "distill_args": None}


def _stub_build_main(argv: list[str] | None = None) -> int:  # type: ignore[no-redef]
    # argv contains flags including --out_dir; write minimal dataset
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
        str(out_dir / "dataset.parquet")
    )
    df.write_csv(str(out_dir / "dataset.csv"))
    return 0


def _stub_tft_main(argv: list[str] | None = None) -> int:  # type: ignore[no-redef]
    # Pretend to train; write teacher_preds.npz in out_dir
    out_idx = argv.index("--out_dir") + 1 if argv and "--out_dir" in argv else -1
    out_dir = Path(argv[out_idx]) if out_idx > 0 else Path("/tmp")
    _CALLED["tft_args"] = argv
    import numpy as np

    y = np.array([0, 1, 0], dtype=np.float32)
    z = np.array([0.0, 2.0, -1.0], dtype=np.float32)
    q = 1.0 / (1.0 + np.exp(-z))
    np.savez_compressed(out_dir / "teacher_preds.npz", q_train=q, y_val_true=y)
    return 0


def _stub_distill_main(argv: list[str] | None = None) -> int:  # type: ignore[no-redef]
    # Accept args and succeed
    _CALLED["distill_args"] = argv
    return 0


def test_pipeline_registers_features_and_passes_ids(monkeypatch, tmp_path: Path) -> None:
    # Patch sub-steps
    monkeypatch.setattr("ml.scripts.build_tft_dataset.main", _stub_build_main)
    monkeypatch.setattr("ml.training.teacher.tft_cli.main", _stub_tft_main)
    monkeypatch.setattr("ml.training.distillation.cli.main", _stub_distill_main)

    # Run pipeline with auto register
    args = [
        "--data_dir",
        str(tmp_path),
        "--symbols",
        "SPY",
        "--out_dir",
        str(tmp_path / "out"),
        "--include_macro",
        "--include_micro",
        "--include_l2",
        "--horizon_minutes",
        "1",
        "--threshold",
        "0.001",
        "--lookback_periods",
        "1",
        "--train_teacher",
        "--teacher_model_id",
        "tft_v1",
        "--feature_registry_dir",
        str(tmp_path / "features"),
        "--register_features",
        "--model_registry_dir",
        str(tmp_path / "models"),
        "--student_model_id",
        "student_v1",
    ]
    rc = pipeline_mod.main(args)
    assert rc == 0
    # Feature registry should contain a feature set
    from ml.registry.feature_registry import FeatureRegistry

    freg = FeatureRegistry(tmp_path / "features")
    assert len(freg.list_all()) >= 1
    # Ensure pipeline forwarded feature args into teacher and student steps
    assert _CALLED["tft_args"] is not None
    assert _CALLED["distill_args"] is not None
    tft_args = _CALLED["tft_args"] or []
    distill_args = _CALLED["distill_args"] or []
    assert "--feature_set_id" in tft_args
    assert "--feature_set_id" in distill_args
