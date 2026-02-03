from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

from pathlib import Path

import json
import numpy as np
import polars as pl
import pytest

import ml.pipelines.tft_train_distill as pipeline_mod


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

_CAPTURED: dict[str, list[str]] = {}


def _stub_build_main(argv: list[str] | None = None) -> int:
    if not argv:
        raise AssertionError("expected argv")
    out_idx = argv.index("--out_dir") + 1 if "--out_dir" in argv else -1
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
    metadata: dict[str, object] = {
        "dataset_id": "tft_dataset",
        "vintage_policy": "real_time",
        "vintage_cutoff": None,
        "build_ts": "2025-01-01T00:00:00",
        "ts_event_start": "2025-01-01T00:00:00",
        "ts_event_end": "2025-01-01T00:02:00",
        "overall_window": ["2025-01-01T00:00:00", "2025-01-01T00:02:00"],
        "train_window": None,
        "validation_window": None,
        "test_window": None,
        "macro_observation_counts": {},
    }
    (out_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "features_npz.npz",
        X_train=np.array([[0.1]], dtype=np.float32),
        X_val=np.array([[0.2]], dtype=np.float32),
        feature_names=np.array(["f1"], dtype=object),
    )
    return 0


def _stub_orchestrator_main(cli_args: list[str]) -> int:
    _CAPTURED["args"] = list(cli_args)
    if "--out_dir" in cli_args:
        Path(cli_args[cli_args.index("--out_dir") + 1]).mkdir(parents=True, exist_ok=True)
    return 0


def test_pipeline_registers_features_and_passes_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Patch orchestrator entrypoint
    monkeypatch.setattr("ml.pipelines.tft_train_distill.orchestrator_main", _stub_orchestrator_main)

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
        "--target_semantics",
        json.dumps(
            {
                "version": "v1",
                "horizons": [{"minutes": 1}],
                "binary": {"enabled": True, "threshold_bps": 10.0, "return_basis": "raw"},
            },
            ensure_ascii=True,
        ),
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
    captured = _CAPTURED.get("args", [])
    assert captured, "orchestrator_main should be invoked"
    assert "--dataset_register_features" in captured
    assert "--distill_student" in captured
    assert "--teacher_model_id" in captured
    assert "--student_model_id" in captured
