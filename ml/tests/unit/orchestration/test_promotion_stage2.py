from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pytest

from ml.data.vintage import VintagePolicy
from ml.orchestration import promotions as promotions_module
from ml.orchestration.promotions import Stage2Config
from ml.orchestration.promotions import run_promotion_stage2
from ml.orchestration.stage2_engine import Stage2Result
from ml.registry.dataclasses import QualityGate

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

def _write_csv(path: Path, n: int = 100, symbol: str = "SPY") -> None:
    # Minimal dataset tail with required columns
    import pandas as pd

    ts0 = 1_700_000_000_000_000_000  # arbitrary ns
    rows = []
    for i in range(n):
        rows.append(
            {
                "time_index": i,
                "timestamp": ts0 + i * 60_000_000_000,  # minute steps
                "instrument_id": symbol,
            },
        )
    pd.DataFrame(rows).to_csv(path, index=False)

def _write_metadata_file(out_dir: Path, dataset_id: str = "stage2_ds") -> None:
    metadata = {
        "dataset_id": dataset_id,
        "vintage_policy": VintagePolicy.REAL_TIME.value,
        "vintage_cutoff": None,
        "build_ts": "2025-01-01T00:00:00",
        "ts_event_start": "2025-01-01T00:00:00",
        "ts_event_end": "2025-01-02T00:00:00",
        "overall_window": ["2025-01-01T00:00:00", "2025-01-02T00:00:00"],
        "train_window": None,
        "validation_window": None,
        "test_window": None,
        "macro_observation_counts": {},
    }
    (out_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

def test_stage2_returns_engine(tmp_path: Path) -> None:
    out_dir = tmp_path / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # teacher_preds.npz with q_val/y_val_true
    q = np.concatenate([np.zeros(50, dtype=np.float32), np.ones(50, dtype=np.float32)])
    y = np.concatenate([np.zeros(50, dtype=np.float32), np.ones(50, dtype=np.float32)])
    np.savez_compressed(out_dir / "teacher_preds.npz", q_val=q, y_val_true=y)

    # dataset.csv minimal tail
    csv_path = out_dir / "dataset.csv"
    _write_csv(csv_path, n=100, symbol="SPY")
    _write_metadata_file(out_dir)

    cfg = Stage2Config(
        out_dir=str(out_dir),
        dataset_csv=str(csv_path),
        data_dir=str(tmp_path),  # not used by this test
        horizon_minutes=1,
        engine_mode="returns",
        cost_bps=0.0,
        model_id_hint=None,
        gates=(),
        auto_promote=False,
        deploy_target=None,
        expected_dataset_id="stage2_ds",
        expected_vintage_policy=VintagePolicy.REAL_TIME,
    )
    result = run_promotion_stage2(cfg)
    assert result["status"] in {"passed", "failed", "skipped"}
    if result["status"] != "skipped":
        metrics = result["metrics"]
        assert "sharpe_ratio" in metrics and "max_drawdown" in metrics

def test_stage2_backtest_engine_fallback(tmp_path: Path) -> None:
    out_dir = tmp_path / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal artifacts
    np.savez_compressed(out_dir / "teacher_preds.npz", q_val=np.ones(10, dtype=np.float32), y_val_true=np.ones(10, dtype=np.float32))
    csv_path = out_dir / "dataset.csv"
    _write_csv(csv_path, n=10, symbol="SPY")
    _write_metadata_file(out_dir)

    cfg = Stage2Config(
        out_dir=str(out_dir),
        dataset_csv=str(csv_path),
        data_dir=str(tmp_path),
        horizon_minutes=1,
        engine_mode="backtest",
        gates=(),
        expected_dataset_id="stage2_ds",
        expected_vintage_policy=VintagePolicy.REAL_TIME,
    )
    result = run_promotion_stage2(cfg)
    # Engine mode may fall back; ensure it still returns a result
    assert result["status"] in {"passed", "failed", "skipped"}

def test_stage2_metadata_mismatch_raises(tmp_path: Path) -> None:
    out_dir = tmp_path / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "teacher_preds.npz",
        q_val=np.ones(4, dtype=np.float32),
        y_val_true=np.ones(4, dtype=np.float32),
    )
    csv_path = out_dir / "dataset.csv"
    _write_csv(csv_path, n=4, symbol="SPY")
    _write_metadata_file(out_dir, dataset_id="stage2_ds")

    cfg = Stage2Config(
        out_dir=str(out_dir),
        dataset_csv=str(csv_path),
        data_dir=str(tmp_path),
        horizon_minutes=1,
        expected_dataset_id="other_ds",
        expected_vintage_policy=VintagePolicy.REAL_TIME,
    )

    with pytest.raises(ValueError):
        run_promotion_stage2(cfg)

def test_stage2_gate_rejects_non_finite_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure promotion gates fail when metrics are NaN/inf.
    """
    out_dir = tmp_path / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "teacher_preds.npz",
        q_val=np.ones(4, dtype=np.float32),
        y_val_true=np.ones(4, dtype=np.float32),
    )
    csv_path = out_dir / "dataset.csv"
    _write_csv(csv_path, n=4, symbol="SPY")
    _write_metadata_file(out_dir, dataset_id="stage2_ds")

    class _NaNMetricsEngine:
        def run(self, cfg: Stage2Config) -> Stage2Result:
            del cfg
            return Stage2Result(
                status="passed",
                metrics={
                    "sharpe_ratio": float("nan"),
                    "max_drawdown": float("inf"),
                },
            )

    monkeypatch.setattr(promotions_module, "build_engine", lambda _: _NaNMetricsEngine())

    cfg = Stage2Config(
        out_dir=str(out_dir),
        dataset_csv=str(csv_path),
        data_dir=str(tmp_path),
        horizon_minutes=1,
        gates=(
            QualityGate(metric_name="sharpe_ratio", threshold=0.1, comparison="gte", required=True),
            QualityGate(metric_name="max_drawdown", threshold=0.5, comparison="lte", required=True),
        ),
        expected_dataset_id="stage2_ds",
        expected_vintage_policy=VintagePolicy.REAL_TIME,
    )

    result = run_promotion_stage2(cfg)
    assert result["status"] == "failed"
    failures = result["failures"]
    assert isinstance(failures, list)
    assert any("sharpe_ratio" in failure for failure in failures)
    assert any("max_drawdown" in failure for failure in failures)
