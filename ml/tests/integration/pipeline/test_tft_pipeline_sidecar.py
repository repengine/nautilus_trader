from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

import json
from pathlib import Path

import polars as pl
import pytest

import ml.pipelines.tft_train_distill as pipeline_mod


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

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
    sidecar = {
        "feature_set_id": "feature_set_123",
        "feature_registry_dir": str((out_dir / "features").resolve()),
        "feature_names": ["f1"],
        "flags": {
            "include_macro": False,
            "include_micro": False,
            "include_l2": False,
            "include_calendar": False,
            "include_events": False,
            "include_earnings": False,
            "include_macro_revisions": False,
        },
    }
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "feature_set.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f)
    return 0


def _stub_orchestrator_main(cli_args: list[str]) -> int:
    if "--out_dir" not in cli_args:
        raise AssertionError("Expected --out_dir in orchestrator args")
    out_dir = Path(cli_args[cli_args.index("--out_dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    # Simulate dataset build to produce sidecar metadata for downstream stages.
    _stub_build_main(["--out_dir", str(out_dir)])
    data = json.loads((out_dir / "feature_set.json").read_text(encoding="utf-8"))
    assert data["feature_set_id"] == "feature_set_123"
    return 0


def test_pipeline_reads_sidecar_when_args_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("ml.pipelines.tft_train_distill.orchestrator_main", _stub_orchestrator_main)

    args = [
        "--data_dir",
        str(tmp_path),
        "--symbols",
        "SPY",
        "--out_dir",
        str(tmp_path / "out"),
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
        # Note: no feature_registry_dir/feature_set_id passed
        "--model_registry_dir",
        str(tmp_path / "models"),
        "--student_model_id",
        "student_v1",
    ]
    rc = pipeline_mod.main(args)
    assert rc == 0
